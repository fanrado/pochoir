#!/usr/bin/env python3
'''
Solve initial value problem to get drift paths using pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .arrays import to_torch

from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# GPU-native trilinear interpolation (replaces torch_interpolations)
# ---------------------------------------------------------------------------

@torch.compile
def _interp_velocity(
    y: torch.Tensor,
    vfield: torch.Tensor,
    origin: torch.Tensor,
    inv_span: torch.Tensor,
) -> torch.Tensor:
    """
    Trilinear velocity interpolation via grid_sample (fused CUDA kernel).

    Parameters
    ----------
    y        : (N, 3) particle positions [dim0, dim1, dim2]
    vfield   : (1, 3, D, H, W) stacked velocity components on device
    origin   : (3,) physical origin per dimension
    inv_span : (3,) precomputed = 2 / ((shape - 1) * spacing)

    Returns
    -------
    (N, 3) velocities at each particle position
    """
    N = y.shape[0]
    # Normalise physical coords to [-1, 1] for each dimension
    norm = (y - origin) * inv_span - 1.0  # (N, 3)
    # grid_sample 5-D convention: grid[..., 0]=W-dim, [..1]=H-dim, [..2]=D-dim
    # our layout: dim0=D, dim1=H, dim2=W  →  flip to (dim2, dim1, dim0)
    grid = norm[:, [2, 1, 0]].view(1, 1, 1, N, 3)
    out = F.grid_sample(
        vfield, grid, mode='bilinear', padding_mode='border', align_corners=True
    )  # (1, 3, 1, 1, N)
    return out.view(3, N).T  # (N, 3)


# ---------------------------------------------------------------------------
# ODE callable
# ---------------------------------------------------------------------------

class Simple:
    '''
    ODE callable for drift path integration.

    Supports batched evaluation: torchdiffeq will call this with
    y of shape (N, 3) when N trajectories are integrated simultaneously.
    '''

    def __init__(self, domain, vfield, device):
        '''
        Parameters
        ----------
        domain : domain object with .shape, .spacing, .origin
        vfield : list of 3 array-like components, each (nz, ny, nx)
        device : torch device string or torch.device
        '''
        dev = torch.device(device)
        shape   = torch.tensor(domain.shape,   dtype=torch.float32, device=dev)
        spacing = torch.tensor(domain.spacing, dtype=torch.float32, device=dev)
        origin  = torch.tensor(domain.origin,  dtype=torch.float32, device=dev)

        self.origin   = origin                          # (3,)
        # Precompute 2 / ((shape-1) * spacing) to avoid repeated division in RHS
        self.inv_span = 2.0 / ((shape - 1) * spacing)  # (3,)
        self.calls    = 0

        # Stack velocity components into (1, 3, nz, ny, nx) for grid_sample
        components = []
        for v in vfield:
            t = torch.as_tensor(v, dtype=torch.float32).to(dev)
            components.append(t)
        self.vfield = torch.stack(components, dim=0).unsqueeze(0)  # (1, 3, D, H, W)

    def __call__(self, tick, y):
        '''
        Return velocity at position(s) y (time-independent field).

        Parameters
        ----------
        tick : scalar time (unused — field is static)
        y    : (3,) single particle  or  (N, 3) batched particles

        Returns
        -------
        same shape as y
        '''
        single = y.dim() == 1
        if single:
            y = y.unsqueeze(0)  # (1, 3)
        velo = _interp_velocity(y, self.vfield, self.origin, self.inv_span)
        self.calls += 1
        return velo.squeeze(0) if single else velo


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve(domain, start, velocity, times, **kwds):
    '''
    Return the path(s) of point(s) at times from start through velocity field.

    Parameters
    ----------
    domain   : domain object with .shape, .spacing, .origin
    start    : array-like (3,) for a single trajectory, or (N, 3) for N
               trajectories solved in a single batched ODE call
    velocity : list of 3 array-like components of the velocity field
    times    : 1-D array-like of time points

    Returns
    -------
    numpy array of shape (len(times), 3) for single, or
    (len(times), N, 3) for batched start points
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start    = torch.tensor(start,  dtype=torch.float32, device=device)
    velocity = [torch.as_tensor(v,  dtype=torch.float32).to(device) for v in velocity]
    times    = torch.tensor(times,  dtype=torch.float32, device=device)

    func = Simple(domain, velocity, device)
    print(f"starting path at {start}  (device={device})")
    res = odeint(func, start, times, rtol=0.01, atol=0.01)
    print(f"function called {func.calls} times")
    return res.cpu().numpy()
