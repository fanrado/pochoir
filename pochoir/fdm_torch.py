#!/usr/bin/env python3
'''
Apply FDM solution to solve Laplace boundary value problem with torch.
'''

import numpy
import torch
from .arrays import core_slices1
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity # For profiling the performance of the solver

from .fdm_generic import edge_condition, stencil, stencil_poisson

# torch.set_default_dtype(torch.float32)
# torch.float64 = torch.float32

def set_core1(dst, src, core):
    dst[core] = src

def set_core2(dst, src, core):
    dst[core] = src
import sys

# def laplacian_3d_numerical(f, dx=1.0, dy=1.0, dz=1.0):
#     """
#     f: shape (D, H, W) — e.g. torch.Size([46, 46, 152])
#     """
#     gz, gy, gx = torch.gradient(f, spacing=(dz, dy, dx))
#     gzz = torch.gradient(gz, spacing=(dz, dy, dx))[0]
#     gyy = torch.gradient(gy, spacing=(dz, dy, dx))[1]
#     gxx = torch.gradient(gx, spacing=(dz, dy, dx))[2]
#     return gxx + gyy + gzz
def laplacian_3d_conv(f):
    """
    f: shape (D, H, W) — e.g. torch.Size([46, 46, 152])
    """
    # conv3d expects (B, C, D, H, W) — add both dims temporarily
    x = f.unsqueeze(0).unsqueeze(0)  # → (1, 1, 46, 46, 152)

    kernel = torch.zeros(1, 1, 3, 3, 3, dtype=f.dtype, device=f.device)
    kernel[0, 0, 1, 1, 0] = 1   # z-
    kernel[0, 0, 1, 1, 2] = 1   # z+
    kernel[0, 0, 1, 0, 1] = 1   # y-
    kernel[0, 0, 1, 2, 1] = 1   # y+
    kernel[0, 0, 0, 1, 1] = 1   # x-
    kernel[0, 0, 2, 1, 1] = 1   # x+
    kernel[0, 0, 1, 1, 1] = -6  # center

    out = F.conv3d(x, kernel, padding=1)  # → (1, 1, 46, 46, 152)
    return out.squeeze(0).squeeze(0)      # → (46, 46, 152)
def plot_laplacian_3d(lap: torch.Tensor, n_slices: int = 5):
    """
    lap: shape (D, H, W) — the Laplacian tensor
    n_slices: how many evenly-spaced slices to show per axis
    """
    data = lap.detach().cpu().numpy()
    D, H, W = data.shape

    # Symmetric colormap centered on 0, driven by the true max abs value
    vmax = np.abs(data).max()
    vmin = -vmax

    slice_indices = {
        "D (axis 0)": np.linspace(0, D - 1, n_slices, dtype=int),
        "H (axis 1)": np.linspace(0, H - 1, n_slices, dtype=int),
        "W (axis 2)": np.linspace(0, W - 1, n_slices, dtype=int),
    }

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Laplacian Tensor Visualization", fontsize=16, fontweight="bold")

    # --- Slices: 3 rows (one per axis) x n_slices cols ---
    total_rows = 4  # 3 slice rows + 1 histogram row
    for row, (axis_label, indices) in enumerate(slice_indices.items()):
        for col, idx in enumerate(indices):
            ax = fig.add_subplot(total_rows, n_slices, row * n_slices + col + 1)

            if row == 0:
                slice_2d = data[idx, :, :]   # fix D
                xlabel, ylabel = "W", "H"
            elif row == 1:
                slice_2d = data[:, idx, :]   # fix H
                xlabel, ylabel = "W", "D"
            else:
                slice_2d = data[:, :, idx]   # fix W
                xlabel, ylabel = "H", "D"

            im = ax.imshow(slice_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{axis_label}={idx}", fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

    plt.colorbar(im, ax=fig.axes[:-1], shrink=0.6, label="Laplacian value")

    # --- Histogram row ---
    ax_hist = fig.add_subplot(total_rows, 1, total_rows)
    flat = data.flatten()
    ax_hist.hist(flat, bins=200, color="steelblue", edgecolor="none", log=True)
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1, label="0")

    # Mark outlier thresholds (±3 std)
    std = flat.std()
    for s, color in zip([-3, 3], ["orange", "orange"]):
        ax_hist.axvline(s * std, color=color, linestyle=":", linewidth=1.2,
                        label=f"{s}σ = {s*std:.2e}")

    ax_hist.set_xlabel("Laplacian value")
    ax_hist.set_ylabel("Count (log scale)")
    ax_hist.set_title(
        f"Value distribution  |  min={flat.min():.2e}  max={flat.max():.2e}"
        f"  mean={flat.mean():.2e}  std={std:.2e}"
    )
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("laplacian_plot.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()
    print("Saved → laplacian_plot.png")

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_boundary_analysis(lap: torch.Tensor):
    """
    Focused analysis on the W-axis (z) boundary artifact.
    lap: shape (D, H, W)
    """
    data = lap.detach().cpu().numpy()
    D, H, W = data.shape

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Boundary Artifact Analysis — W axis (z)", fontsize=14, fontweight="bold")

    # --- Row 1: Mean, Max abs, Std per W-slice ---
    w_indices = np.arange(W)
    mean_per_w   = data.mean(axis=(0, 1))           # shape (W,)
    maxabs_per_w = np.abs(data).max(axis=(0, 1))    # shape (W,)
    std_per_w    = data.std(axis=(0, 1))            # shape (W,)

    ax = axes[0, 0]
    ax.plot(w_indices, mean_per_w, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Mean Laplacian per W-slice")
    ax.set_xlabel("W index")
    ax.set_ylabel("Mean value")

    ax = axes[0, 1]
    ax.plot(w_indices, maxabs_per_w, color="darkorange")
    ax.set_title("Max |Laplacian| per W-slice")
    ax.set_xlabel("W index")
    ax.set_ylabel("Max |value|")

    ax = axes[0, 2]
    ax.plot(w_indices, std_per_w, color="purple")
    ax.set_title("Std Laplacian per W-slice")
    ax.set_xlabel("W index")
    ax.set_ylabel("Std")

    # --- Row 2: Heatmaps of first, last, and second-to-last W-slices ---
    vmax = np.abs(data[:, :, -5:]).max()  # colormap driven by boundary region

    for col, (w_idx, label) in enumerate([
        (0,  "W=0 (first slice)"),
        (W-2, f"W={W-2} (second-to-last)"),
        (W-1, f"W={W-1} (last slice — boundary)"),
    ]):
        ax = axes[1, col]
        im = ax.imshow(data[:, :, w_idx], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("H")
        ax.set_ylabel("D")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig("boundary_analysis.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()
    print("Saved → boundary_analysis.png")

    # --- Print summary ---
    interior = data[:, :, 1:-1]  # strip boundary slices
    boundary_w0 = data[:, :, 0]
    boundary_w1 = data[:, :, -1]
    print(f"Interior  — mean: {interior.mean():.3e}  |  max|val|: {np.abs(interior).max():.3e}")
    print(f"Boundary W=0    — mean: {boundary_w0.mean():.3e}  |  max|val|: {np.abs(boundary_w0).max():.3e}")
    print(f"Boundary W={W-1} — mean: {boundary_w1.mean():.3e}  |  max|val|: {np.abs(boundary_w1).max():.3e}")

# @torch.compile
def _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, 
                   core, periodic, spacing=1.0, source=None):
    # if source is not None:
    #     # warmup outside of the profiler so one-time CUDA costs don't pollute results
    #     stencil_poisson(iarr_pad.clone(), source=source.clone(), spacing=spacing, res=tmp_core.clone())

    #     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
    #         # with record_function("compiled_step"):
    #             stencil_poisson(iarr_pad, source=source, spacing=spacing, res=tmp_core)
    #             # torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
    #     print(prof.key_averages().table(
    #         sort_by="cuda_time_total", 
    #         row_limit=20
    #     ))
    #     prof.export_chrome_trace(f"stencil_poisson_runtime_profile.json")
    #     sys.exit()
    stencil_poisson(iarr_pad, source=source, spacing=spacing, res=tmp_core)
    iarr_pad[core] = bi_core + mutable_core * tmp_core
    edge_condition(iarr_pad, *periodic, info_msg=None)


def solve(iarr, barr, periodic, prec, epoch, nepochs, info_msg=None, _dtype=torch.float64, phi0=None, ctx=None, potential=None, increment=None, params=None):
    '''
    Solve boundary value problem

    Return (arr, err)

        - iarr gives array of initial values

        - barr gives bool array where True indicates value at that
          index is boundary (imutable).

        - periodic is list of Boolean.  If true, the corresponding
          dimension is periodic, else it is fixed.

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

    barr_pad = torch.tensor(numpy.pad(barr, 1), requires_grad=False, dtype=_dtype).to(device)
    iarr_pad = torch.tensor(numpy.pad(iarr, 1), requires_grad=False, dtype=_dtype).to(device)
    core = core_slices1(iarr_pad)

    iarr_pad_source = iarr_pad.clone().detach().requires_grad_(False).to(device)
    source = None ## Variable to hold the source term for poisson equation, if phi0 is provided
    non_padded_phi0 = None ## Variable to hold the original phi0 before padding, for computing the source term without the influence of padding values.
    if phi0 is not None:
        non_padded_phi0 = phi0.clone().detach().requires_grad_(False).to(device)
        spacing = 1.0 ## Need to include this at the top of the function later.
        # print(f'device for phi0: {phi0.device}, dtype for phi0: {phi0.dtype}')
        iarr_pad = torch.zeros(iarr_pad.shape, requires_grad=False, dtype=_dtype).to(device)
        # phi0_ = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=_dtype).to(device)
        # cast phi0 to float64 and zero-pad it
        phi0_ = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=torch.float64).to(device)
        # numpy.pad fills the halo with zeros, but stencil(phi0) at the outermost interior
        # cells will then pull in those zeros as neighbour values, making the computed source
        # term artificially large at the domain edges.  edge_condition propagates the actual
        # interior values into the halo (same BCs as iarr_pad), fixing the stencil there.
        path_to_padded_phi0 = potential.split('/')[0] + '/padded_phi0'
        ctx.obj.put(path_to_padded_phi0, phi0_.cpu(), taxon='padded_phi0', **params)
        # print(f'padded phi0 saved to {path_to_padded_phi0}')
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
    # if phi0 is not None:
    #     ## create a copy of the original phi0 before zero-padding.
    #     non_padded_phi0 = phi0.clone().detach().requires_grad_(False).to(device)
    #     spacing = 1.0 ## Need to include this at the top of the function later.
    #     ##
    #     ## zero-pad phi0
    #     phi0 = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=_dtype).to(device)
    #     ##
    #     ## Apply the boundary conditions: periodic along x- and y-axes, fixed along z-axis.
    #     edge_condition(phi0, *periodic)
    #     ##
    #     ## Compute the average of the 6 neighboring cells for each interior cell.
    #     s = stencil(phi0).detach()
    #     ##
    #     ## Calculate the source term for the Poisson equation
    #     source = -(6/spacing**2)*(s - non_padded_phi0)
        
    #     print(f'phi0 shape {phi0.shape}')
    # core = core_slices1(iarr_pad)
    # info_msg(f'core slices = {core}, dtype : {type(core)}')
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
        info_msg(f'====== epoch: {iepoch}/{nepochs} x {epoch} ===============')
        print(f'====== epoch: {iepoch}/{nepochs} x {epoch} ===============')
        epoch_start_time = time.time()
        # potential_path = f'{potential}_epoch{iepoch}'
        # increment_path = f'{increment}_epoch{iepoch}'
        _periodic = tuple(periodic)
        for istep in range(epoch):
            # # info_msg(f'step: {istep}/{epoch}')
            if istep%1000 ==0:
                prev = iarr_pad.clone().detach().requires_grad_(False)

            # _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic)
            # _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic, phi0=phi0, spacing=1.0, iarr_pad_source=iarr_pad_source)
            _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic, spacing=1.0, source=source)
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

    if source is not None:
        spacing = 1.0 ## Need to include this at the top of the function later.
        # print(f'device for phi0: {phi0.device}, dtype for phi0: {phi0.dtype}')
        # iarr_pad = torch.zeros(iarr_pad.shape, requires_grad=False, dtype=_dtype).to(device)
        phi0 = phi0 + iarr_pad[core].cpu() ## Add the computed potential to the original phi0 to get the updated potential with BCs applied, which will be used to compute the source term for the next iteration if needed.
        phi0_ = torch.tensor(numpy.pad(phi0, 1), requires_grad=False, dtype=_dtype).to(device)
        # numpy.pad fills the halo with zeros, but stencil(phi0) at the outermost interior
        # cells will then pull in those zeros as neighbour values, making the computed source
        # term artificially large at the domain edges.  edge_condition propagates the actual
        # interior values into the halo (same BCs as iarr_pad), fixing the stencil there.
        # path_to_padded_phi0 = potential.split('/')[0] + '/padded_phi0'
        # ctx.obj.put(path_to_padded_phi0, phi0.cpu(), taxon='padded_phi0', **params)
        # print(f'padded phi0 saved to {path_to_padded_phi0}')
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
        source = -(6/spacing**2)*(s - non_padded_phi0)#* mutable_core
        path_to_source = potential.split('/')[0] + '/nabla_phi'
        ctx.obj.put(path_to_source, source.cpu(), taxon='nabla_phi', **params)
    return (iarr_pad[core].cpu(), err.cpu())