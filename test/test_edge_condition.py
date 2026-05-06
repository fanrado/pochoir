#!/usr/bin/env python3
"""
Validate edge_condition's "fix" mode (Neumann mirror) on a 1D array.
"""

import numpy as np
import torch


def edge_condition(arr, *periodic, info_msg=None):
    np_ = len(periodic)
    na = len(arr.shape)
    if np_ != na:
        raise ValueError(f"dimension mismatch: {np_} != {na}")

    slices = [slice(0, s) for s in arr.shape]
    for dim, per in enumerate(periodic):
        n = arr.shape[dim]
        src1 = list(slices); src2 = list(slices)
        dst1 = list(slices); dst2 = list(slices)
        dst1[dim] = slice(0, 1);     src1[dim] = slice(n - 2, n - 1)
        dst2[dim] = slice(n - 1, n); src2[dim] = slice(1, 2)
        if per:
            arr[tuple(dst1)] = arr[tuple(src1)]
            arr[tuple(dst2)] = arr[tuple(src2)]
        else:
            arr[tuple(dst1)] = arr[tuple(src2)]
            arr[tuple(dst2)] = arr[tuple(src1)]


def main():
    Nx = 3

    rng = np.random.default_rng(0)
    iarr = rng.normal(size=(Nx,)).astype(np.float64)
    print(f'The interior values iarr = {iarr}')

    barr = np.zeros_like(iarr, dtype=bool)
    barr[0] = True
    barr[-1] = True
    print(f'The boundaries set using barr = {barr}')

    iarr[0] = 0.0
    iarr[-1] = 1.0
    print(f'Fixing the boundary values (Dirichlet boundary condition) : {iarr}')

    iarr_pad = torch.from_numpy(np.pad(iarr, 1))
    print(f'Padded array (one halo cell per side) : {iarr_pad}')

    # fix
    periodic = (True,)
    edge_condition(iarr_pad, *periodic)

    arr = iarr_pad.numpy()

    print(f"padded shape = {arr.shape}  (interior = {iarr.shape})")
    print(f'iarr after applying (per,) boundary condition : {arr}')

    # halo at index 0 should equal interior index 1; halo at -1 == interior -2
    print("x face (Neumann/mirror expected):")
    print(f"  halo[0]  == interior[1]  ? {np.isclose(arr[0],  arr[1])}   ({arr[0]:+.6f} vs {arr[1]:+.6f})")
    print(f"  halo[-1] == interior[-2] ? {np.isclose(arr[-1], arr[-2])}  ({arr[-1]:+.6f} vs {arr[-2]:+.6f})")
    print()

    # Contrast with periodic
    iarr_pad2 = torch.from_numpy(np.pad(iarr, 1))
    edge_condition(iarr_pad2, True)
    arr2 = iarr_pad2.numpy()
    print("Contrast: periodic wraps halo to the OPPOSITE interior face:")
    print(f"  halo[0]  == interior[-2] ? {np.isclose(arr2[0],  arr2[-2])}")
    print(f"  halo[-1] == interior[1]  ? {np.isclose(arr2[-1], arr2[1])}")


if __name__ == "__main__":
    main()
