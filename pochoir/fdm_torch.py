#!/usr/bin/env python3
'''
Apply FDM solution to solve Laplace boundary value problem with torch.
'''

import numpy
import torch
from .arrays import core_slices1

from .fdm_generic import edge_condition, stencil

# torch.set_default_dtype(torch.float32)
# torch.float64 = torch.float32

def set_core1(dst, src, core):
    dst[core] = src

def set_core2(dst, src, core):
    dst[core] = src

@torch.compile
def _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, periodic):
    stencil(iarr_pad, tmp_core)
    iarr_pad[core] = bi_core + mutable_core * tmp_core
    edge_condition(iarr_pad, *periodic, info_msg=None)

import sys
def solve(iarr, barr, periodic, prec, epoch, nepochs, info_msg=None, ctx=None, potential=None, increment=None, params=None):
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

    _dtype = torch.float64
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bi_core = torch.tensor(iarr*barr, requires_grad=False, dtype=_dtype).to(device)
    mutable_core = torch.tensor(numpy.invert(barr.astype(numpy.bool)), requires_grad=False, dtype=_dtype).to(device)
    tmp_core = torch.zeros(iarr.shape, requires_grad=False, dtype=_dtype).to(device)

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # bi_core = torch.tensor(iarr*barr, requires_grad=False).to(device)
    # mutable_core = torch.tensor(numpy.invert(barr.astype(numpy.bool)), requires_grad=False).to(device)
    # tmp_core = torch.zeros(iarr.shape, requires_grad=False).to(device)

    barr_pad = torch.tensor(numpy.pad(barr, 1), requires_grad=False, dtype=_dtype).to(device)
    iarr_pad = torch.tensor(numpy.pad(iarr, 1), requires_grad=False, dtype=_dtype).to(device)
    core = core_slices1(iarr_pad)
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
        potential_path = f'{potential}_epoch{iepoch}'
        increment_path = f'{increment}_epoch{iepoch}'
        _periodic = tuple(periodic)
        for istep in range(epoch):
            # # info_msg(f'step: {istep}/{epoch}')
            if istep%100 ==0:
                prev = iarr_pad.clone().detach().requires_grad_(False)

            # # disable this part for debugging ---- this is part of the original script
            # if epoch-istep == 1: # last in the iteration
            #     prev = iarr_pad.clone().detach().requires_grad_(False)

            _compiled_step(iarr_pad, tmp_core, bi_core, mutable_core, core, _periodic)
            # stencil(iarr_pad, tmp_core)
            # iarr_pad[core] = bi_core + mutable_core * tmp_core
            # edge_condition(iarr_pad, *periodic, info_msg=None)
            # # err = iarr_pad[core] - prev[core]
            # # maxerr = torch.max(torch.abs(err))
            # # info_msg(f'maxerr = {maxerr}, prec = {prec}, dtype = {maxerr.dtype}')

            # # if prec and maxerr < prec:
            # #     break
            # ## disable this part for debugging ---- this is part of the original script
            # if epoch-istep == 1: # last in the iteration
            #     err = iarr_pad[core] - prev[core]
            #     maxerr = torch.max(torch.abs(err))
            #     info_msg(f'maxerr = {maxerr}, prec = {prec}, dtype = {maxerr.dtype}')
            #     # # Removed this part for debugging ---- this is part of the original script
            #     # # Allowing the solver to run for all epochs to check the precision at the end of all epochs
            #     # if prec and maxerr < prec:
            #     #     info_msg(f'fdm reach max precision: {prec} > {maxerr}')
            #     #     return (iarr_pad[core].cpu(), err.cpu())
            # if epoch-istep == 1: # last in the iteration
            if istep%100 ==0:
                err = iarr_pad[core] - prev[core]
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

    return (iarr_pad[core].cpu(), err.cpu())