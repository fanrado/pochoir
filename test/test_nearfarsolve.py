#!/usr/bin/env pytest
'''
Tests for the overlapping-Schwarz near/far solve (pochoir/nearfar.py and the
`pochoir near-far-solve` CLI command).

The scheme replaces the single-shot near-bc pin (value-only continuity at the
interface) with an alternating near<->far iteration on a 1-cell overlap, so
the stitched field is continuous in value *and* gradient across the seam.

Unit tests exercise the resample/graft helpers; integration tests drive the
full iteration with the real numpy FDM solver on tiny harmonic problems.
'''

import numpy
import pytest
from click.testing import CliRunner

from pochoir.domain import Domain
from pochoir.main import Main
from pochoir.__main__ import cli
from pochoir import nearfar
from pochoir.fdm import solve_numpy


def _linear3d(dom, ax=1.0, ay=2.0, az=3.0):
    gx, gy, gz = dom.meshgrid
    return ax * gx + ay * gy + az * gz


def _harmonic(dom, L=8.0):
    '''phi = cos(pi x / L) * cosh(pi z / L): harmonic, curved, non-polynomial.'''
    k = numpy.pi / L
    gx, gy, gz = dom.meshgrid
    return numpy.cos(k * gx) * numpy.cosh(k * gz)


# --------------------------------------------------------------------------
# resample_plane / resample_volume / graft_plane (unit)
# --------------------------------------------------------------------------

def test_resample_plane_downsample_linear_exact():
    '''A linear field resamples exactly (down-sampling: fine source -> coarse
    target plane).'''
    fine = Domain([9, 9, 9], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    src = _linear3d(fine)                                  # x + 2y + 3z
    # Coarse plane at z index 2 (coord z=2): phi = x + 2y + 6.
    got = nearfar.resample_plane(src, fine, coarse, axis=2, index=2)
    gx, gy, _ = coarse.meshgrid
    numpy.testing.assert_allclose(got, (gx + 2.0 * gy + 6.0)[:, :, 2], atol=1e-12)


def test_resample_plane_upsample_linear_exact():
    '''Up-sampling: coarse source -> fine target plane, linear field exact.'''
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    fine = Domain([5, 5, 9], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])
    src = _linear3d(coarse)
    # Fine top plane z index 8 (coord z=4): phi = x + 2y + 12.
    got = nearfar.resample_plane(src, coarse, fine, axis=2, index=8)
    gx, gy, _ = fine.meshgrid
    numpy.testing.assert_allclose(got, (gx + 2.0 * gy + 12.0)[:, :, 8], atol=1e-12)


def test_resample_volume_linear_exact():
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    fine = Domain([5, 5, 9], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])
    src = _linear3d(coarse)
    got = nearfar.resample_volume(src, coarse, fine)
    numpy.testing.assert_allclose(got, _linear3d(fine), atol=1e-12)


def test_graft_plane_sets_values_and_fixes_boundary():
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    near = Domain([5, 5, 5], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])   # z=0..2
    src = _linear3d(coarse)
    init = numpy.zeros((5, 5, 5))
    bmask = numpy.zeros((5, 5, 5), dtype=bool)

    # Fix the near top plane (z index 4, coord z=2): phi = x + 2y + 6.
    nearfar.graft_plane(init, bmask, src, coarse, near, axis=2, index=4, fix=True)
    gx, gy, _ = near.meshgrid
    numpy.testing.assert_allclose(init[:, :, 4], (gx + 2.0 * gy + 6.0)[:, :, 4],
                                  atol=1e-12)
    assert numpy.all(bmask[:, :, 4])
    # Other planes untouched.
    assert numpy.all(init[:, :, :4] == 0.0)
    assert not numpy.any(bmask[:, :, :4])

    # Seed a plane without fixing it.
    nearfar.graft_plane(init, bmask, src, coarse, near, axis=2, index=3, fix=False)
    assert not numpy.any(bmask[:, :, 3])
    assert numpy.any(init[:, :, 3] != 0.0)


def test_seed_near_merges_boundary():
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    near = Domain([5, 5, 9], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])   # z=0..4
    coarse_pot = _linear3d(coarse)
    near_init = numpy.zeros((5, 5, 9))
    near_bmask = numpy.zeros((5, 5, 9), dtype=bool)
    near_init[:, :, 0] = -42.0                    # electrode plane
    near_bmask[:, :, 0] = True

    seed = nearfar.seed_near(coarse_pot, coarse, near_init, near_bmask, near)
    # Interior seeded from the upsampled coarse (linear -> exact).
    numpy.testing.assert_allclose(seed[:, :, 1:], _linear3d(near)[:, :, 1:],
                                  atol=1e-12)
    # Electrode plane preserved exactly.
    assert numpy.all(seed[:, :, 0] == -42.0)


def test_interface_indices_validation():
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    near = Domain([5, 5, 5], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])   # z=0..2
    ci, ci_in, nt, nt_in = nearfar._interface_indices(coarse, near, 2, 2.0)
    assert (ci, ci_in, nt, nt_in) == (2, 1, 4, 3)

    # Interface off the coarse grid -> error.
    with pytest.raises(ValueError):
        nearfar._interface_indices(coarse, near, 2, 1.5)
    # Near top plane not at the interface -> error.
    near_bad = Domain([5, 5, 5], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])  # top z=2
    with pytest.raises(ValueError):
        nearfar._interface_indices(coarse, near_bad, 2, 3.0)


# --------------------------------------------------------------------------
# schwarz_solve integration (real numpy FDM solver)
# --------------------------------------------------------------------------

def _numpy_solver(prec=1e-7, epoch=4000, nepochs=2, edges=(False, False, False)):
    def _solve(iarr, barr):
        arr, _err = solve_numpy(numpy.asarray(iarr, dtype=float),
                                numpy.asarray(barr).astype(bool),
                                list(edges), prec, epoch, nepochs)
        return numpy.asarray(arr)
    return _solve


def _dirichlet_box(dom, analytic, faces="all"):
    '''Initial + boolean-mask arrays with Dirichlet faces set to `analytic`.

    faces="all": all 6 faces fixed.  faces="near": z=0 and the 4 transverse
    faces fixed (top left free, as for a near-field problem).
    '''
    init = numpy.zeros(tuple(int(s) for s in dom.shape))
    bmask = numpy.zeros(init.shape, dtype=bool)
    a = analytic
    sel = [
        (slice(None), slice(None), 0), (slice(None), slice(None), -1),
        (0, slice(None), slice(None)), (-1, slice(None), slice(None)),
        (slice(None), 0, slice(None)), (slice(None), -1, slice(None)),
    ]
    if faces == "near":
        sel = sel[:1] + sel[2:]           # drop the z-top face
    for s in sel:
        init[s] = a[s]
        bmask[s] = True
    return init, bmask


def _make_problem():
    coarse = Domain([5, 5, 5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])       # z=0..4
    near = Domain([5, 5, 5], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0])         # z=0..2
    ac = _harmonic(coarse)
    an = _harmonic(near)
    coarse_init, coarse_bmask = _dirichlet_box(coarse, ac, faces="all")
    near_init, near_bmask = _dirichlet_box(near, an, faces="near")
    solver = _numpy_solver()
    coarse_pot, _ = solve_numpy(coarse_init, coarse_bmask, [False, False, False],
                                1e-7, 4000, 2)
    coarse_pot = numpy.asarray(coarse_pot)
    return dict(coarse=coarse, near=near, coarse_init=coarse_init,
                coarse_bmask=coarse_bmask, near_init=near_init,
                near_bmask=near_bmask, coarse_pot=coarse_pot, solver=solver,
                interface_z=2.0)


def test_schwarz_converges_and_matches_on_overlap():
    '''After convergence the two solves share the overlap: the interface value
    is continuous, and the far inner plane holds exactly the near value there
    (the gradient carrier) -- i.e. the seam is C0 *and* the near-side slope is
    consistent with the far solve.'''
    p = _make_problem()
    near_pot, far_pot, n_iters, delta = nearfar.schwarz_solve(
        p["coarse_pot"], p["coarse_init"], p["coarse_bmask"], p["coarse"],
        p["near_init"], p["near_bmask"], p["near"],
        p["solver"], p["solver"],
        axis=2, interface_z=2.0, tol=1e-4, max_iters=12)

    assert delta < 1e-4                       # converged
    assert 1 < n_iters <= 12

    ci, ci_in, nt, nt_in = nearfar._interface_indices(p["coarse"], p["near"], 2, 2.0)
    # Interface value continuity: near top is pinned to the far solution.
    far_iface = nearfar.resample_plane(far_pot, p["coarse"], p["near"], 2, nt)
    numpy.testing.assert_allclose(near_pot[:, :, nt], far_iface, atol=1e-3)
    # Gradient carrier: the far inner plane (one coarse cell below the seam)
    # equals the near solution resampled there -- the two solves share it.
    near_on_coarse = nearfar.resample_plane(near_pot, p["near"], p["coarse"],
                                            2, ci_in)
    numpy.testing.assert_allclose(far_pot[:, :, ci_in], near_on_coarse, atol=1e-9)


def test_schwarz_improves_overlap_over_naive_pin():
    '''Schwarz makes near and far agree on the inner overlap plane far better
    than the naive single pin (near pinned to the raw coarse, far = coarse).'''
    p = _make_problem()
    ci, ci_in, nt, nt_in = nearfar._interface_indices(p["coarse"], p["near"], 2, 2.0)

    # Naive: one near solve pinned to the raw coarse; far stays coarse.
    seed = nearfar.seed_near(p["coarse_pot"], p["coarse"],
                             p["near_init"], p["near_bmask"], p["near"])
    ni, nb = seed.copy(), p["near_bmask"].copy()
    nearfar.graft_plane(ni, nb, p["coarse_pot"], p["coarse"], p["near"], 2, nt, True)
    nearfar.graft_plane(ni, nb, p["coarse_pot"], p["coarse"], p["near"], 2, nt_in, False)
    near_naive = p["solver"](ni, nb)
    coarse_inner = nearfar.resample_plane(p["coarse_pot"], p["coarse"], p["near"],
                                          2, nt_in)
    d_naive = float(numpy.max(numpy.abs(near_naive[:, :, nt_in] - coarse_inner)))

    # Schwarz.
    near_pot, far_pot, _, _ = nearfar.schwarz_solve(
        p["coarse_pot"], p["coarse_init"], p["coarse_bmask"], p["coarse"],
        p["near_init"], p["near_bmask"], p["near"],
        p["solver"], p["solver"], axis=2, interface_z=2.0,
        tol=1e-5, max_iters=12)
    far_inner = nearfar.resample_plane(far_pot, p["coarse"], p["near"], 2, nt_in)
    d_schwarz = float(numpy.max(numpy.abs(near_pot[:, :, nt_in] - far_inner)))

    # Schwarz makes the near and far solutions agree on the free overlap plane
    # markedly better than the naive single pin (limited only by the coarse
    # grid's own discretization error).
    assert d_schwarz < d_naive


def test_schwarz_respects_max_iters_guard():
    '''With an unreachable tolerance the loop stops at max_iters.'''
    p = _make_problem()
    _, _, n_iters, _ = nearfar.schwarz_solve(
        p["coarse_pot"], p["coarse_init"], p["coarse_bmask"], p["coarse"],
        p["near_init"], p["near_bmask"], p["near"],
        p["solver"], p["solver"], axis=2, interface_z=2.0,
        tol=-1.0, max_iters=3)
    assert n_iters == 3


# --------------------------------------------------------------------------
# CLI command end-to-end (engine=numpy)
# --------------------------------------------------------------------------

def test_cli_near_far_solve_end_to_end(tmp_path):
    p = _make_problem()
    storedir = str(tmp_path / "store")
    m = Main(storedir)
    m.put_domain("coarse_dom", p["coarse"])
    m.put_domain("near_dom", p["near"])
    m.put("coarse_pot", p["coarse_pot"], taxon="potential", domain="coarse_dom")
    m.put("coarse_init", p["coarse_init"], taxon="initial", domain="coarse_dom")
    m.put("coarse_bnd", p["coarse_bmask"], taxon="boundary", domain="coarse_dom")
    m.put("near_init", p["near_init"], taxon="initial", domain="near_dom")
    m.put("near_bnd", p["near_bmask"], taxon="boundary", domain="near_dom")

    result = CliRunner().invoke(cli, [
        "--store", storedir, "near-far-solve",
        "-C", "coarse_pot",
        "--coarse-initial", "coarse_init", "--coarse-boundary", "coarse_bnd",
        "--near-initial", "near_init", "--near-boundary", "near_bnd",
        "--interface", "2.0", "--axis", "2",
        "--edges", "fixed,fixed,fixed", "--engine", "numpy",
        "--epoch", "4000", "-n", "2",
        "--near-precision", "1e-7", "--far-precision", "1e-7",
        "--tol", "1e-4", "--max-iters", "12",
        "--near-out", "pot_near", "--far-out", "pot_far",
    ])
    assert result.exit_code == 0, result.output

    m = Main(storedir)
    near_pot, nmd = m.get("pot_near", True)
    far_pot, fmd = m.get("pot_far", True)
    assert numpy.asarray(near_pot).shape == (5, 5, 5)
    assert numpy.asarray(far_pot).shape == (5, 5, 5)
    assert nmd.get("taxon") == "potential"
    assert nmd.get("operation") == "near-far-solve"
    assert nmd.get("domain") == "near_dom"
    assert fmd.get("domain") == "coarse_dom"

    # Gradient-carrier overlap agreement holds through the CLI path too.
    ci, ci_in, nt, nt_in = nearfar._interface_indices(p["coarse"], p["near"], 2, 2.0)
    near_on_coarse = nearfar.resample_plane(numpy.asarray(near_pot), p["near"],
                                            p["coarse"], 2, ci_in)
    numpy.testing.assert_allclose(numpy.asarray(far_pot)[:, :, ci_in],
                                  near_on_coarse, atol=1e-9)
