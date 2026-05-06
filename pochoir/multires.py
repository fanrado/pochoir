#!/usr/bin/env python3
'''
Multi-resolution (coarse-to-fine) acceleration for Pochoir FDM.

See PLAN_acceleration_calculation.md for full design rationale.
'''

import numpy
from scipy.ndimage import map_coordinates

from .domain import Domain

# Preferred "nice" spacings used when auto-picking the coarsest stage (mm).
NICE_SPACINGS = [5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]


# ---------------------------------------------------------------------------
# Stage schedule helpers
# ---------------------------------------------------------------------------

def pick_stage_spacings(s_target, n_stages, s_coarsest=None):
    '''
    Return a list of n_stages spacings [s_0, ..., s_{N-1}] with
    s_0 >= s_coarsest > ... > s_{N-1} = s_target on a geometric ladder.

    If s_coarsest is None, pick the largest nice spacing that is both
    > s_target and <= 50 * s_target (giving roughly 50x coarsening).
    '''
    s_target = float(s_target)
    if n_stages == 1:
        return [s_target]

    if s_coarsest is None:
        raw = s_target * 50.0
        candidates = [s for s in NICE_SPACINGS if s_target < s <= raw]
        s_coarsest = float(max(candidates)) if candidates else raw
    else:
        s_coarsest = float(s_coarsest)

    if s_coarsest <= s_target:
        raise ValueError(f"s_coarsest ({s_coarsest}) must be > s_target ({s_target})")

    ratio = (s_coarsest / s_target) ** (1.0 / (n_stages - 1))
    spacings = [s_target * ratio ** (n_stages - 1 - k) for k in range(n_stages)]
    # Ensure exact endpoints (avoid floating point drift)
    spacings[0] = s_coarsest
    spacings[-1] = s_target
    return spacings


def _stage_prec(prec_final, s_k, s_target, alpha=1.0):
    '''
    Return the relaxed precision target for stage with spacing s_k.

    prec_k = prec_final * (s_k / s_target) ** alpha

    Returns None if prec_final is None (no convergence criterion).
    '''
    if prec_final is None:
        return None
    return float(prec_final) * (float(s_k) / float(s_target)) ** float(alpha)


def _default_nepochs(s_target, spacings, base=200):
    '''
    Return a list of per-stage epoch budgets that self-calibrate with spacing.

    nepochs[k] = max(base, round(base * (s_target / s_k) ** 2))

    Coarser stages (large s_k) get small budgets; finer stages get larger ones.
    '''
    s_target = float(s_target)
    result = []
    for s_k in spacings:
        result.append(max(int(base), int(round(base * (s_target / float(s_k)) ** 2))))
    return result


# ---------------------------------------------------------------------------
# Interpolation (prolongation): coarse -> fine grid
# ---------------------------------------------------------------------------

def _stage_shape(target_shape, s_target, s_k):
    '''
    Compute the shape of a stage domain given target shape and spacing ratio.
    Each dimension is rounded to the nearest odd integer >= 5.
    '''
    ratio = s_target / s_k
    raw = numpy.array(target_shape, dtype=float) * ratio
    # Round to nearest integer, then snap to odd, clamp to >= 5
    rounded = numpy.maximum(numpy.round(raw).astype(int), 5)
    # Make odd
    rounded = numpy.where(rounded % 2 == 0, rounded + 1, rounded)
    return tuple(rounded.tolist())


def lift_to_grid(coarse_arr, coarse_domain, fine_domain):
    '''
    Prolongate coarse_arr from coarse_domain onto fine_domain using
    separable multilinear interpolation (scipy.ndimage.map_coordinates, order=1).

    Works for any number of dimensions and arbitrary (non-integer) refinement
    ratios between stages.

    Returns a numpy float64 array of shape fine_domain.shape.
    '''
    coarse_arr = numpy.asarray(coarse_arr, dtype=numpy.float64)

    # Build the fine-grid sample coordinates in coarse-index space.
    # For each axis d:
    #   fine physical coord: x = fine_origin[d] + i * fine_spacing[d]
    #   coarse index:        u = (x - coarse_origin[d]) / coarse_spacing[d]
    fine_shape = tuple(int(s) for s in fine_domain.shape)
    ndim = len(fine_shape)

    axes_coords = []
    for d in range(ndim):
        fine_coords = (fine_domain.origin[d]
                       + numpy.arange(fine_shape[d]) * fine_domain.spacing[d])
        coarse_idx = (fine_coords - coarse_domain.origin[d]) / coarse_domain.spacing[d]
        axes_coords.append(coarse_idx)

    # Build meshgrid of coarse-index coordinates for all fine-grid points
    grids = numpy.meshgrid(*axes_coords, indexing='ij')
    coords = numpy.stack([g.ravel() for g in grids], axis=0)  # (ndim, N_fine)

    fine_flat = map_coordinates(coarse_arr, coords, order=1, mode='nearest')
    return fine_flat.reshape(fine_shape)


# ---------------------------------------------------------------------------
# Main multi-resolution solver
# ---------------------------------------------------------------------------

def solve_multires(store, gen_name, gen_cfg,
                   target_shape, target_spacing, origin,
                   n_stages, engine, prec, epoch, periodic,
                   nepochs_per_stage=None, coarsest_spacing=None,
                   epoch_base=200, prec_scale_alpha=1.0,
                   output_key='potential'):
    '''
    Multi-resolution FDM solve.

    Parameters
    ----------
    store       : pochoir.main.Main instance (provides get/put/get_domain)
    gen_name    : str, name of pochoir.gen.* generator function
    gen_cfg     : dict, generator configuration (already unit-ified)
    target_shape: sequence of int, shape of the finest (target) domain
    target_spacing: float or sequence of float, spacing of the finest domain
    origin      : sequence of float, shared origin across all stages
    n_stages    : int, number of resolution stages (>= 1)
    engine      : str, FDM backend ('numpy', 'torch', ...)
    prec        : float or None, final convergence tolerance
    epoch       : int, number of iterations per convergence check
    periodic    : list of bool, periodic boundary flags per axis
    nepochs_per_stage : list of int or None; if None, computed adaptively
    coarsest_spacing  : float or None
    epoch_base        : int, base for adaptive epoch budget
    prec_scale_alpha  : float, exponent for per-stage precision relaxation
    output_key        : str, store key for the final potential
    '''
    import pochoir.fdm as fdm_mod
    import pochoir.gen as gen_mod

    solve_fn = getattr(fdm_mod, f'solve_{engine}')
    gen_fn = getattr(gen_mod, gen_name)

    target_spacing = float(target_spacing)
    spacings = pick_stage_spacings(target_spacing, n_stages,
                                   s_coarsest=coarsest_spacing)

    if nepochs_per_stage is None:
        nepochs_per_stage = _default_nepochs(target_spacing, spacings,
                                              base=epoch_base)

    prev_sol = None
    prev_dom = None

    for k, s_k in enumerate(spacings):
        stage_shape = _stage_shape(target_shape, target_spacing, s_k)
        stage_dom = Domain(stage_shape, s_k, origin=origin)

        iarr, barr, epsilon = gen_fn(stage_dom, gen_cfg)
        iarr = numpy.asarray(iarr, dtype=numpy.float64)
        barr = numpy.asarray(barr, dtype=bool)

        # Build phi0
        if k == 0:
            phi0 = iarr.copy()
        else:
            lifted = lift_to_grid(prev_sol, prev_dom, stage_dom)
            # Re-imprint exact boundary values
            phi0 = numpy.where(barr, iarr, lifted)

        # Adaptive per-stage precision
        stage_prec = _stage_prec(prec, s_k, target_spacing,
                                  alpha=prec_scale_alpha)
        nep = nepochs_per_stage[k]

        print(f'[multires] stage {k}/{n_stages-1}  spacing={s_k}  '
              f'shape={stage_shape}  prec={stage_prec}  nepochs={nep}')

        # Call the FDM backend.  numpy solve uses iarr as seed; pass phi0 there.
        if engine in ('numpy', 'numba', 'cupy'):
            sol, err = solve_fn(phi0, barr, periodic,
                                stage_prec, epoch, nep)
        else:  # torch and cumba accept phi0 kwarg
            # fdm_torch.solve expects numpy arrays for iarr/barr/epsilon
            # and uses ctx.obj.put(..., **params) for debug artifacts when
            # phi0 is supplied.  Build a tiny shim so multires can drive it.
            class _Ctx:
                pass
            _ctx = _Ctx()
            _ctx.obj = store
            _params = dict(stage=k, spacing=s_k, shape=list(stage_shape))
            import torch
            phi0_t = torch.as_tensor(phi0, dtype=torch.float64)
            sol, err = solve_fn(iarr, barr, periodic,
                                stage_prec, epoch, nep,
                                info_msg=lambda *a, **k: None,
                                phi0=phi0_t,
                                ctx=_ctx,
                                potential=output_key,
                                params=_params,
                                epsilon=epsilon)
            if hasattr(sol, 'cpu'):
                sol = sol.cpu().numpy()

        prev_sol = numpy.asarray(sol, dtype=numpy.float64)
        prev_dom = stage_dom

        key = f'{output_key}_L{k}'
        store.put(key, prev_sol, taxon='potential',
                  stage=k, spacing=s_k, shape=list(stage_shape))
        print(f'[multires] stage {k} saved as "{key}"')

    # Also write final stage under canonical output_key
    store.put(output_key, prev_sol, taxon='potential',
              stage=n_stages - 1, spacing=target_spacing,
              shape=list(target_shape))
    print(f'[multires] final solution saved as "{output_key}"')
