#!/usr/bin/env python3
'''
Apply FDM solution to solve Laplace boundary value problem with torch.
'''

import sys

import numpy
import torch
from .arrays import core_slices1
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity # For profiling the performance of the solver

from .fdm_generic import (edge_condition, edge_condition_fixed,
                          stencil, stencil_poisson,
                          stencil_poisson_harmonic, stencil_poisson_neumann,
                          neumann_coeff, mirror_masks, mirror_project,
                          padplane_noflux_geom, padplane_noflux)

# torch.set_default_dtype(torch.float32)
# torch.float64 = torch.float32

def set_core1(dst, src, core):
    dst[core] = src

def set_core2(dst, src, core):
    dst[core] = src

@torch.compile
def _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core,
                   core, periodic, spacing=1.0, source=None, epsilon=None,
                   masks=None, true_fixed=False):
    # Three-way relaxation step (branch is on None-ness, static per solve run,
    # so torch.compile specialises once with no recompilation blowup):
    #   - pad-plane no-flux masks present : plain Laplace on the WHOLE volume,
    #                              then the node-centered no-flux is imposed by
    #                              padplane_noflux -- the ghost node flanking each
    #                              gap node on the pad plane is set equal to the
    #                              drift-side node, so centered E_z = 0 AT the pad
    #                              node itself.  NO epsilon.
    #   - epsilon present        : harmonic-mean Poisson (dielectric path);
    #   - otherwise              : plain Poisson/Laplace (unchanged default).
    if masks is None and epsilon is not None:
        stencil_poisson_harmonic(iarr_pad, eps=epsilon, res=tmp_core)
    else:
        stencil_poisson(iarr_pad, source=source, spacing=spacing, res=tmp_core)
    iarr_pad[core] = bi_core + mutable_core * tmp_core
    if masks is not None:
        # Node-centered pad-plane no-flux: set the ghost (non-drift-side) node of
        # each gap node equal to its drift-side node so the centered z-derivative
        # at the pad node is exactly zero.  Applied after the relaxation update
        # and before edge_condition.  padplane_noflux is pure, so write its result
        # back into iarr_pad in place.
        iarr_pad[...] = padplane_noflux(iarr_pad, masks)
    # true_fixed selects the edge condition applied to the non-periodic
    # dimensions.  Like the masks/epsilon branches above it is a Python bool
    # held constant for a whole solve run, so torch.compile specialises on it
    # once and there is no recompilation blowup.
    #   - False (default) : edge_condition, whose non-periodic branch is a
    #                       Neumann mirror.  Correct for the drift solve.
    #   - True            : edge_condition_fixed, which leaves the non-periodic
    #                       halo slabs holding their construction values, i.e. a
    #                       genuinely held Dirichlet wall.  Wanted by the
    #                       weighting solve, whose reflecting transverse walls
    #                       otherwise stop W from decaying laterally.
    if true_fixed:
        edge_condition_fixed(iarr_pad, *periodic, info_msg=None)
    else:
        edge_condition(iarr_pad, *periodic, info_msg=None)


def solve(iarr, barr, periodic, prec, epoch, nepochs, info_msg=None, _dtype=torch.float64, phi0=None, ctx=None, potential=None, increment=None, params=None, epsilon=None, insulator=None, true_fixed=False):
    '''
    Solve boundary value problem

    Return (arr, err)

        - iarr gives array of initial values

        - barr gives bool array where True indicates value at that
          index is boundary (imutable).

        - periodic is list of Boolean.  If true, the corresponding
          dimension is periodic, else it is fixed.

        - true_fixed selects how the non-periodic ("fixed") dimensions
          are treated.  The default False keeps the historical
          edge_condition() behaviour, a Neumann mirror, which is correct
          for the drift solve (per,per,fix) because its paths are later
          replicated onto a larger domain.  True uses
          edge_condition_fixed(), which holds the boundary value written
          by domain construction; the weighting solve (fix,fix,fix)
          wants this so W decays laterally instead of plateauing.

        - epoch is number of iteration per precision check

        - nepochs limits the number of epochs

    Returned arrays "arr" is like iarr with updated solution including
    fixed boundary value elements.  "err" is difference between last
    and penultimate iteration.
    '''

    err = None
    import time 
    
    # Save original torch.tensor
    # original_tensor = torch.tensor
    torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
    start_time = time.time()

    _dtype = _dtype
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bi_core = torch.tensor(iarr*barr, requires_grad=False, dtype=_dtype).to(device)
    mutable_core = torch.tensor(numpy.invert(barr.astype(numpy.bool)), requires_grad=False, dtype=_dtype).to(device)
    tmp_core = torch.zeros(iarr.shape, requires_grad=False, dtype=_dtype).to(device)

    # barr_pad = torch.tensor(numpy.pad(barr, 1), requires_grad=False, dtype=_dtype).to(device)
    iarr_pad = torch.tensor(numpy.pad(iarr, 1), requires_grad=False, dtype=_dtype).to(device)
    core = core_slices1(iarr_pad)

    epsilon_pad = torch.tensor(numpy.pad(epsilon, 1), requires_grad=False, dtype=_dtype).to(device) if epsilon is not None else None

    # --- pad-plane no-flux (node-centered Neumann) boundary, NO epsilon ---
    # The `insulator` argument is now only an ENABLE signal (its content is
    # vestigial): when supplied, the whole volume relaxes with plain Laplace and
    # the no-flux condition is imposed on the pixel/pad plane as a NODE-centered
    # mirror.  The interface is auto-derived from the Dirichlet geometry
    # (padplane_noflux_geom(barr): the unique partially-Dirichlet z-plane is the
    # pad plane; its free cells are the gap nodes; the drift side is toward the
    # cathode).  Each step the ghost node flanking a gap node is set equal to its
    # drift-side node, so the centered z-derivative -- and hence E_z -- is exactly
    # zero AT the pad node the drift integrator samples, while the transverse E is
    # untouched and the region below the plane is not shielded.  The masks are
    # geometry-only, so they are built once here (padded to the halo, on device)
    # and reused every step.  NO permittivity enters; the dielectric (epsilon)
    # path is bypassed.  With insulator=None this block is skipped and the solve
    # stays byte-identical to plain Laplace.
    insulator_masks = None
    if insulator is not None:
        # Pass the insulator (FR4 slab) mask so a shield grid -- a second,
        # detached partially-Dirichlet plane -- is disambiguated from the pad
        # electrode; the no-flux interface stays on the pad/FR4 plane and the
        # grid is treated as an ordinary Dirichlet electrode.  Without a shield
        # grid this does not change the derived geometry.
        geom = padplane_noflux_geom(barr.astype(numpy.bool),
                                    insulator=insulator.astype(numpy.bool)
                                    if insulator is not None else None)
        # Pad the ghost mask with False (halo is never a ghost) and move it to the
        # device so it conforms to iarr_pad; roll_shift/axis are unchanged by the
        # padding (the ghost mask localises the roll to interior nodes).
        ghost_mask_pad = torch.tensor(numpy.pad(geom['ghost_mask'], 1),
                                      requires_grad=False).to(device)
        insulator_masks = {'ghost_mask': ghost_mask_pad,
                           'roll_shift': geom['roll_shift'],
                           'axis': geom['axis']}
        info_msg(f"pad-plane no-flux: z_pad={geom['z_pad']} "
                 f"drift_sign={geom['drift_sign']:+d} "
                 f"gap_nodes={int(geom['gap2d'].sum())}")
        if epsilon_pad is not None:
            info_msg('insulator enable supplied: ignoring epsilon (NO dielectric '
                     'path in the pad-plane no-flux boundary)')
            epsilon_pad = None
    # iarr_pad_source = iarr_pad.clone().detach().requires_grad_(False).to(device)
    source = None ## Variable to hold the source term for poisson equation, if phi0 is provided
    non_padded_phi0 = None ## Variable to hold the original phi0 before padding, for computing the source term without the influence of padding values.
    if phi0 is not None:
        non_padded_phi0 = phi0.clone().detach().requires_grad_(False).to(device)
        spacing = 1.0 ## Need to include this at the top of the function later.
        # print(f'device for phi0: {phi0.device}, dtype for phi0: {phi0.dtype}')
        iarr_pad = torch.zeros(iarr_pad.shape, requires_grad=False, dtype=_dtype).to(device)
        # cast phi0 to float64 and zero-pad it
        phi0_ = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=torch.float64).to(device)
        # numpy.pad fills the halo with zeros, but stencil(phi0) at the outermost interior
        # cells will then pull in those zeros as neighbour values, making the computed source
        # term artificially large at the domain edges.  edge_condition propagates the actual
        # interior values into the halo (same BCs as iarr_pad), fixing the stencil there.
        path_to_padded_phi0 = potential.split('/')[0] + '/padded_phi0'
        ctx.obj.put(path_to_padded_phi0, phi0_.cpu(), taxon='padded_phi0', **params)
        # print(f'padded phi0 saved to {path_to_padded_phi0}')
        # Gated the same way as the relaxation step: this call fills the phi0
        # halo used by stencil(phi0_) below, and it must agree with the BC the
        # solve itself applies.
        if true_fixed:
            edge_condition_fixed(phi0_, *periodic)
        else:
            edge_condition(phi0_, *periodic)
        # print(f'---shape of phi0 after edge condition: {phi0.shape}, max value: {torch.max(phi0)}, min value: {torch.min(phi0)}')
        s = stencil(phi0_).detach()
        # edge_condition(s, *periodic)
        # iarr_pad_source[core] = bi_core + mutable_core * s
        
        # Multiply by mutable_core to zero out the source at fixed boundary cells (barr=True).
        # At those cells ss[core]=bi_core, so any mismatch with phi0[core] would produce a
        # spurious non-zero source — but boundary cells are overwritten by bi_core anyway,
        # so their source term is physically meaningless and should be suppressed.
        # source = -(6/spacing**2)*(iarr_pad_source[core] - phi0[core]) * mutable_core
        source = -(6/spacing**2)*(s - non_padded_phi0)* mutable_core # * mutable_core to zero out the source at fixed boundary cells (barr=True) ::: This is the correct condition to apply when solving the Poisson equation.

        ##
        ## For debugging purposes, let's save the output of stencil and the source term without the changes first
        path_to_stencil = potential.split('/')[0] + '/stencil'
        path_to_source = potential.split('/')[0] + '/source'
        path_to_withBC = potential.split('/')[0] + '/phi0_withBC'
        ctx.obj.put(path_to_stencil, s.cpu(), taxon='stencil', **params)
        ctx.obj.put(path_to_source, source.cpu(), taxon='source', **params)
        ctx.obj.put(path_to_withBC, phi0_.cpu(), taxon='phi0_withBC', **params)
        # print(f'stencil saved to {path_to_stencil}, source saved to {path_to_source}, withBC saved to {path_to_withBC}')
        # print(f'shape of mutable core : {mutable_core.shape}, shape of stencil : {s.shape}')

        ## cast source to _dtype
        source = source.to(_dtype)
        ## save the source term casted to _dtype for debugging
        path_to_source = potential.split('/')[0] + '/source_FP32'
        ctx.obj.put(path_to_source, source.cpu(), taxon='source_FP32', **params)

        # # warmup outside of the profiler so one-time CUDA costs don't pollute results
        # _compiled_step(iarr_pad.clone(), tmp_core.clone(), bi_core.clone(), mutable_core.clone(), core, tuple(periodic), spacing=1.0, source=source.clone())

        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
        #     # with record_function("compiled_step"):
        #         _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, tuple(periodic), spacing=1.0, source=source)
        #         # torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
        # print(prof.key_averages().table(
        #     sort_by="cuda_time_total", 
        #     row_limit=20
        # ))
        # prof.export_chrome_trace(f"fdm_step_trace_epoch.json")
        # sys.exit()

    # Get indices of fixed boundary values and values themselves
    info_msg(f'bi_core shape = {bi_core.shape}, mutable_core shape = {mutable_core.shape}, tmp_core shape = {tmp_core.shape}, \n\tiarr_pad_shape = {iarr_pad.shape}')
    info_msg(f'bi_core device = {bi_core.device}, mutable_core device = {mutable_core.device}, tmp_core device = {tmp_core.device}, \n\tiarr_pad_device = {iarr_pad.device}')
    info_msg(f'bi_core dtype = {bi_core.dtype}, mutable_core dtype = {mutable_core.dtype}, tmp_core dtype = {tmp_core.dtype}, \n\tiarr_pad_dtype = {iarr_pad.dtype}')

    # print(f'potential path name = {potential}, increment path name = {increment}')
    # print('params = ', params)
    # sys.exit()
    prev = None
    for iepoch in range(nepochs):
        torch.cuda.synchronize()
        info_msg(f'====== epoch: {iepoch+1}/{nepochs} x {epoch} ===============')
        print(f'====== epoch: {iepoch+1}/{nepochs} x {epoch} ===============')
        epoch_start_time = time.time()
        # potential_path = f'{potential}_epoch{iepoch}'
        # increment_path = f'{increment}_epoch{iepoch}'
        _periodic = tuple(periodic)
        for istep in range(epoch):
            # # info_msg(f'step: {istep}/{epoch}')
            if istep%1000 ==0:
                prev = iarr_pad.clone().detach().requires_grad_(False)

            # _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic)
            _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic, spacing=1.0, source=source, epsilon=epsilon_pad, masks=insulator_masks, true_fixed=true_fixed)
            # stencil(iarr_pad, tmp_core)
            # iarr_pad[core] = bi_core + mutable_core * tmp_core
            # edge_condition(iarr_pad, *periodic, info_msg=None)
            
            # if epoch-istep == 1: # last in the iteration
            if (istep%1000 ==0) and (istep !=0):
                # err = iarr_pad[core] - prev[core]
                err = iarr_pad[core].clone().detach().to(torch.float64) - prev[core].to(torch.float64)
                maxerr = torch.max(torch.abs(err))
                info_msg(f'iteration : {istep}, maxerr = {maxerr}, prec = {prec}, dtype = {maxerr.dtype}')
                # # Removed this part for debugging ---- this is part of the original script
                # # Allowing the solver to run for all epochs to check the precision at the end of all epochs
                if prec and maxerr < prec:
                    info_msg(f'fdm reach max precision: {prec} > {maxerr}')
                    torch.cuda.synchronize()
                    epoch_end_time = time.time()
                    info_msg(f'epoch {iepoch} time: {epoch_end_time - epoch_start_time:.2f} seconds')
                    return (iarr_pad[core].cpu(), err.cpu())
                    # break
                
                if maxerr == 0.0:
                    info_msg(f'fdm reached maxerr = {maxerr} at iteration {istep}, epoch {iepoch}')
                    torch.cuda.synchronize()
                    epoch_end_time = time.time()
                    info_msg(f'epoch {iepoch} time: {epoch_end_time - epoch_start_time:.2f} seconds')
                    return (iarr_pad[core].cpu(), err.cpu())

        torch.cuda.synchronize()
        epoch_end_time = time.time()
        info_msg(f'epoch {iepoch} time: {epoch_end_time - epoch_start_time:.2f} seconds')
        print(f'maxerr = {maxerr}, prec = {prec}, maxerr dtype = {maxerr.dtype}')
        ##
        ## Save potential and error at the end of each epoch
        # ctx.obj.put(potential_path, iarr_pad[core].cpu(), taxon='potential', **params)
        # ctx.obj.put(increment_path, err.cpu(), taxon='increment', **params)
        # print(f'potential saved to {potential_path}, increment saved to {increment_path}')
        
    info_msg(f'iarr_pad_shape = {iarr_pad.shape}, periodic = {periodic}, prec = {prec}, epoch = {epoch}, nepochs = {nepochs}')
    info_msg(f'iarr_pad_dtype = {iarr_pad.dtype}, err dtype = {err.dtype}, maxerr = {maxerr}')
    info_msg(f'fdm reach max epoch {epoch} x {nepochs}, last prec {prec} < {maxerr}')
    torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
    end_time = time.time()
    info_msg(f'FDM solve time: {end_time - start_time:.2f} seconds, Nepochs = {nepochs}, steps per epoch = {epoch}')

    # if source is not None:
    #     spacing = 1.0 ## Need to include this at the top of the function later.
    #     # print(f'device for phi0: {phi0.device}, dtype for phi0: {phi0.dtype}')
    #     # iarr_pad = torch.zeros(iarr_pad.shape, requires_grad=False, dtype=_dtype).to(device)
    #     phi0 = phi0 + iarr_pad[core].cpu() ## Add the computed potential to the original phi0 to get the updated potential with BCs applied, which will be used to compute the source term for the next iteration if needed.
    #     phi0_ = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=_dtype).to(device)
    #     # numpy.pad fills the halo with zeros, but stencil(phi0) at the outermost interior
    #     # cells will then pull in those zeros as neighbour values, making the computed source
    #     # term artificially large at the domain edges.  edge_condition propagates the actual
    #     # interior values into the halo (same BCs as iarr_pad), fixing the stencil there.
    #     # path_to_padded_phi0 = potential.split('/')[0] + '/padded_phi0'
    #     # ctx.obj.put(path_to_padded_phi0, phi0.cpu(), taxon='padded_phi0', **params)
    #     # print(f'padded phi0 saved to {path_to_padded_phi0}')
    #     edge_condition(phi0_, *periodic)
    #     # print(f'---shape of phi0 after edge condition: {phi0.shape}, max value: {torch.max(phi0)}, min value: {torch.min(phi0)}')
    #     s = stencil(phi0_).detach()
    #     # edge_condition(s, *periodic)
    #     # iarr_pad_source[core] = bi_core + mutable_core * s
        
    #     # Multiply by mutable_core to zero out the source at fixed boundary cells (barr=True).
    #     # At those cells ss[core]=bi_core, so any mismatch with phi0[core] would produce a
    #     # spurious non-zero source — but boundary cells are overwritten by bi_core anyway,
    #     # so their source term is physically meaningless and should be suppressed.
    #     # source = -(6/spacing**2)*(iarr_pad_source[core] - phi0[core]) * mutable_core
    #     source = -(6/spacing**2)*(s - non_padded_phi0)#* mutable_core
    #     path_to_source = potential.split('/')[0] + '/nabla_phi'
    #     ctx.obj.put(path_to_source, source.cpu(), taxon='nabla_phi', **params)
    return (iarr_pad[core].cpu(), err.cpu())
