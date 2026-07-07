#!/usr/bin/env python3
'''
Overlapping-Schwarz near/far-field solve for the drift potential.

The two-step coarse->fine near-field workflow (refine + near-bc + fdm +
coarsen + stitch-near) pins the near-field top plane to the coarse bulk
potential as a single Dirichlet condition.  That enforces only value (C0)
continuity at the interface: the near side is a fine Poisson solve and the
far side is an upsample of an independent coarse solve, so E_z = -dphi/dz is
generally discontinuous across the seam.

This module replaces the one-shot pin with an overlapping-Schwarz alternating
iteration on a 1-cell (2-plane) overlap.  Each sweep:

  - solves the fine near domain with its interface plane pinned (Dirichlet) to
    the current far solution, and the adjacent plane seeded from the far
    solution (free);
  - solves the full coarse (far) domain with an *interior* plane one coarse
    cell below the interface pinned (Dirichlet) to the near solution, and the
    interface plane seeded from the near solution (free).

Because the overlap interval [z_iface - coarse_dz, z_iface] is owned by the
far solve at its inner edge and by the near solve at the interface, the shared
solution converges to one that is continuous in value *and* gradient across
the seam.  The far solve on the full coarse domain re-solves the (discarded)
near region freely; only its far part is later stitched.

The FDM solver itself is injected as callables (`solve_near`, `solve_far`)
so this module stays engine-agnostic and unit-testable with
`pochoir.fdm.solve_numpy`.
'''

import numpy

from pochoir.arrays import rgi


def _clamped_points(pts, source_dom):
    '''
    Clamp query points (shape (N, ndim)) to the source domain's per-axis
    bounds so points outside the source take the nearest edge value
    (no extrapolation).  Returns the clamped points and the source
    linspaces (grid points per axis) for building the interpolator.
    '''
    cpoints = [numpy.asarray(ls, dtype=float) for ls in source_dom.linspaces]
    pts = numpy.array(pts, dtype=float, copy=True)
    for a in range(pts.shape[1]):
        pts[:, a] = numpy.clip(pts[:, a], cpoints[a][0], cpoints[a][-1])
    return pts, cpoints


def _plane_query(target_dom, axis, index):
    '''
    Build the (N, ndim) array of spatial query points for the target-domain
    plane at grid `index` along `axis`, plus the plane's output shape.

    Constructed from per-axis linspaces (not the full meshgrid) so it stays
    cheap for large domains.
    '''
    ls = target_dom.linspaces
    ndim = len(ls)
    coord_axis = ls[axis][index]
    other = [a for a in range(ndim) if a != axis]
    grids = numpy.meshgrid(*[ls[a] for a in other], indexing="ij")
    out_shape = tuple(int(target_dom.shape[a]) for a in other)
    cols = [None] * ndim
    cols[axis] = numpy.full(grids[0].size, coord_axis, dtype=float)
    for g, a in zip(grids, other):
        cols[a] = g.ravel()
    pts = numpy.stack(cols, axis=-1)
    return pts, out_shape


def resample_plane(source_arr, source_dom, target_dom, axis, index):
    '''
    Multi-linearly interpolate `source_arr` (a solution on `source_dom`) onto
    the `target_dom` grid plane at `index` along `axis`.

    Works for both up- and down-sampling because the interpolation is
    coordinate-based.  Query points are clamped to the source bounds.
    Returns an array of the plane's shape (target dims minus `axis`).
    '''
    src = numpy.asarray(source_arr, dtype=float)
    pts, out_shape = _plane_query(target_dom, axis, index)
    pts, cpoints = _clamped_points(pts, source_dom)
    interp = rgi(cpoints, src)
    return interp(pts).reshape(out_shape)


def resample_volume(source_arr, source_dom, target_dom):
    '''
    Multi-linearly interpolate `source_arr` onto the full `target_dom` grid.
    Query points clamped to source bounds (no extrapolation).
    '''
    src = numpy.asarray(source_arr, dtype=float)
    mesh = target_dom.meshgrid
    pts = numpy.stack([m.ravel() for m in mesh], axis=-1)
    pts, cpoints = _clamped_points(pts, source_dom)
    interp = rgi(cpoints, src)
    return interp(pts).reshape(tuple(int(s) for s in target_dom.shape))


def _plane_sel(ndim, axis, index):
    sel = [slice(None)] * ndim
    sel[axis] = index
    return tuple(sel)


def graft_plane(init, bmask, source_arr, source_dom, target_dom,
                axis, index, fix):
    '''
    Write the resampled source plane into `init` at grid `index` along `axis`
    (in place).  If `fix`, mark that plane immutable (Dirichlet) in `bmask`.

    `init` and `bmask` are modified in place; pass copies if the originals
    must be preserved.
    '''
    plane = resample_plane(source_arr, source_dom, target_dom, axis, index)
    sel = _plane_sel(init.ndim, axis, index)
    init[sel] = plane
    if fix:
        bmask[sel] = True


def seed_near(coarse_pot, coarse_dom, near_init, near_bmask, near_dom):
    '''
    Build a near-field FDM initial guess by upsampling the coarse solution
    onto the near grid, then merging the exact near boundary (electrode)
    values back at the boundary cells (mirrors the `refine` command).
    '''
    seed = resample_volume(coarse_pot, coarse_dom, near_dom)
    bmask = numpy.asarray(near_bmask).astype(bool)
    seed[bmask] = numpy.asarray(near_init, dtype=float)[bmask]
    return seed


def _interface_indices(coarse_dom, near_dom, axis, interface_z, atol=1e-6):
    '''
    Resolve the interface grid indices on both domains and validate alignment.

    Returns (ci, ci_in, nt, nt_in):
      ci     - coarse index at z = interface_z
      ci_in  - coarse index one coarse cell below (the far Dirichlet plane)
      nt     - near top index (the near Dirichlet plane, at z = interface_z)
      nt_in  - near index one near cell below (the near seed plane)
    '''
    cdz = float(coarse_dom.spacing[axis])
    corg = float(coarse_dom.origin[axis])
    ci = int(round((interface_z - corg) / cdz))
    ci_z = corg + ci * cdz
    if abs(ci_z - interface_z) > atol * max(1.0, abs(interface_z)):
        raise ValueError(
            f'interface z={interface_z} does not land on a coarse grid node '
            f'(nearest node z={ci_z}, spacing={cdz}, origin={corg})')
    ci_in = ci - 1
    if ci_in < 0 or ci >= int(coarse_dom.shape[axis]):
        raise ValueError(
            f'interface coarse index {ci} (inner {ci_in}) out of range for '
            f'coarse shape {int(coarse_dom.shape[axis])} on axis {axis}')

    nt = int(near_dom.shape[axis]) - 1
    ndz = float(near_dom.spacing[axis])
    norg = float(near_dom.origin[axis])
    nt_z = norg + nt * ndz
    if abs(nt_z - interface_z) > atol * max(1.0, abs(interface_z)):
        raise ValueError(
            f'near top plane z={nt_z} does not equal the interface z='
            f'{interface_z} (near spacing={ndz}, origin={norg}, shape='
            f'{int(near_dom.shape[axis])})')
    nt_in = nt - 1
    if nt_in < 0:
        raise ValueError('near domain has fewer than 2 planes along the '
                         f'stitch axis {axis}')
    return ci, ci_in, nt, nt_in


def schwarz_solve(coarse_pot, coarse_init, coarse_bmask, coarse_dom,
                  near_init, near_bmask, near_dom,
                  solve_near, solve_far,
                  axis=2, interface_z=None,
                  tol=1.0, max_iters=6, log=None, near_start=None):
    '''
    Alternating overlapping-Schwarz near/far solve.

    Each sweep solves the far domain (interior plane one coarse cell below the
    interface pinned to the near solution) then the near domain (interface
    plane pinned to the far solution).  Ending each sweep with the near solve
    makes the returned near exactly pinned to the returned far at the interface
    (exact C0 across the stitch seam).

    Parameters
    ----------
    coarse_pot : ndarray
        Coarse full-domain solved potential (the initial far field).
    coarse_init, coarse_bmask : ndarray
        Coarse full-domain initial values and boolean boundary mask
        (electrode Dirichlet cells) used for the far re-solves.
    coarse_dom : Domain
        Domain of the coarse arrays.
    near_init, near_bmask : ndarray
        Near-field initial values and boolean boundary mask (electrode cells
        only; the interface plane is grafted each sweep).
    near_dom : Domain
        Domain of the near arrays.
    near_start : ndarray or None
        An already-solved near potential to start from (e.g. the discrete
        refine->near-bc->fdm sweep-0 the driver computes, so its intermediate
        store files are preserved).  If None, sweep 0 is done internally by
        seeding from the coarse solve and pinning the interface to it.
    solve_near, solve_far : callable
        `solve(iarr, barr) -> ndarray`.  Each already bound to its engine,
        edges, precision, epoch and nepochs.  Returns the solved potential.
    axis : int
        Stitch axis (default 2, i.e. z).
    interface_z : float
        Spatial coordinate of the near/far interface (near top plane).
    tol : float
        Convergence tolerance on the max change of the near solution between
        successive sweeps.
    max_iters : int
        Maximum number of sweeps.
    log : callable or None
        Optional message sink, e.g. print or logging.info.

    Returns
    -------
    (near_pot, far_pot, n_iters, delta)
        Final near and far potentials, the number of sweeps performed, and
        the last measured near-solution delta.
    '''
    if interface_z is None:
        raise ValueError('interface_z is required')

    def _log(msg):
        if log:
            log(msg)

    coarse_init = numpy.asarray(coarse_init, dtype=float)
    coarse_bmask = numpy.asarray(coarse_bmask).astype(bool)
    near_init = numpy.asarray(near_init, dtype=float)
    near_bmask = numpy.asarray(near_bmask).astype(bool)

    ci, ci_in, nt, nt_in = _interface_indices(
        coarse_dom, near_dom, axis, interface_z)
    _log(f'schwarz: interface z={interface_z} coarse idx {ci} (inner {ci_in}), '
         f'near top idx {nt} (seed {nt_in})')

    far_pot = numpy.asarray(coarse_pot, dtype=float).copy()

    # Sweep 0: the near solve pinned to the coarse bulk.  When the driver has
    # already produced this (discrete refine->near-bc->fdm), pass it as
    # near_start so those intermediate store files are kept; otherwise do it
    # here by seeding from the coarse solve.
    if near_start is not None:
        near_new = numpy.asarray(near_start, dtype=float)
        _log('schwarz: starting from supplied near solution')
    else:
        ni = seed_near(coarse_pot, coarse_dom, near_init, near_bmask, near_dom)
        nb = near_bmask.copy()
        graft_plane(ni, nb, coarse_pot, coarse_dom, near_dom, axis, nt, fix=True)
        graft_plane(ni, nb, coarse_pot, coarse_dom, near_dom, axis, nt_in, fix=False)
        near_new = numpy.asarray(solve_near(ni, nb), dtype=float)

    n_iters = 0
    delta = None
    for it in range(max_iters):
        n_iters = it + 1

        # --- FAR solve: inner plane pinned to near, interface plane seeded ---
        # Warm-start the interior from the previous far, re-imposing electrode
        # Dirichlet values so the boundary is exact.
        fi = far_pot.copy()
        fi[coarse_bmask] = coarse_init[coarse_bmask]
        fb = coarse_bmask.copy()
        graft_plane(fi, fb, near_new, near_dom, coarse_dom, axis, ci_in, fix=True)
        graft_plane(fi, fb, near_new, near_dom, coarse_dom, axis, ci, fix=False)
        far_pot = numpy.asarray(solve_far(fi, fb), dtype=float)

        # --- NEAR solve: interface pinned to far, adjacent plane seeded ---
        near_prev = near_new
        ni = near_prev.copy()
        nb = near_bmask.copy()
        graft_plane(ni, nb, far_pot, coarse_dom, near_dom, axis, nt, fix=True)
        graft_plane(ni, nb, far_pot, coarse_dom, near_dom, axis, nt_in, fix=False)
        near_new = numpy.asarray(solve_near(ni, nb), dtype=float)

        delta = float(numpy.max(numpy.abs(near_new - near_prev)))
        _log(f'schwarz iter {it}: near delta = {delta}')

        if delta < tol:
            _log(f'schwarz converged after {n_iters} sweeps (delta {delta} < {tol})')
            break
    else:
        _log(f'schwarz hit max_iters={max_iters} (last delta {delta})')

    return near_new, far_pot, n_iters, delta
