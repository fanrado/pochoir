#!/usr/bin/env pytest
'''
Tests for the task13 hybrid driver: coarse -> near -> banded Schwarz sweep ->
stitch at 0.4/0.1 mm (commit d1cb650 / pochoir-uy3c, extended by the banded
near/far sweep, pochoir-4ipv).

The outer iteration is gone.  What must hold:
  * DEFAULT_SPACINGS has exactly coarse=0.4 / fine=0.1 (no 'full');
  * GRID_SPEC has three leaves and near/fine01 share the FINE spacing, which is
    what makes the stitch a plane-for-plane overwrite;
  * the derived shapes at the defaults are the ones the commit message pins:
    drift 11,11,151 / 44,44,201 / 44,44,601 and weighting
    55,55,151 / 220,220,201 / 220,220,601;
  * _key carries no iteration suffix, and _kk/_outer_iteration/_final_stage/
    _max_abs_delta are deleted;
  * the banded Schwarz sweep: band/sweep options reaching near-far-solve, its
    per-field store keys, max_sweeps=0 restoring the one-shot path, the
    unchanged output key and the band indices at the runner's settings;
  * _check_interface still raises on a mismatch with the near grid;
  * _manual_grids requires all three leaves and rejects malformed shapes;
  * the CLI surface: --precision added; --tol/--max-iters/--full-spacing/
    --near-coarse-shape gone; --fine-shape present; --fine-spacing default 0.1.

Solves are out of scope for a unit test -- these are pure-function and
option-surface checks.
'''

import inspect
import json
from pathlib import Path

import pytest

import pochoir.hybrid_iterate as hi
from pochoir.__main__ import hybrid_iterate as hi_cmd

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIFT_FINE = REPO_ROOT / "test" / "example_gen_pcb_drift_pixel_task13_fine.json"
WEIGHT_FINE = REPO_ROOT / "test" / "example_gen_pixel_with_grid_task13_fine.json"

INTERFACE_MM = 40.0


def _opts():
    return {p.name: p for p in hi_cmd.params}


def _shapes(field, cfg_path, spacings=None):
    prof = hi._profile(field)
    cfg = json.loads(cfg_path.read_text())
    grids = hi._derive_grids(prof, cfg, spacings or hi.DEFAULT_SPACINGS,
                             INTERFACE_MM)
    return {key.split('/')[-1]: (shape, spacing) for key, shape, spacing in grids}


# --------------------------------------------------------------------------
# Spacings and grid spec
# --------------------------------------------------------------------------

def test_default_spacings_are_two_only():
    assert hi.DEFAULT_SPACINGS == dict(coarse=0.4, fine=0.1)


def test_grid_spec_has_three_leaves_near_and_fine_share_spacing():
    leaves = [leaf for leaf, _, _ in hi.GRID_SPEC]
    assert leaves == ['coarse', 'near', 'fine01']
    spacing_of = {leaf: sp for leaf, sp, _ in hi.GRID_SPEC}
    depth_of = {leaf: d for leaf, _, d in hi.GRID_SPEC}
    assert spacing_of['near'] == spacing_of['fine01'] == 'fine'
    assert spacing_of['coarse'] == 'coarse'
    assert depth_of == dict(coarse='full', near='near', fine01='full')


def test_no_near_coarse_grid_any_more():
    assert 'near_coarse' not in [leaf for leaf, _, _ in hi.GRID_SPEC]


# --------------------------------------------------------------------------
# Derived shapes at the defaults (pinned in the commit message)
# --------------------------------------------------------------------------

# NOTE: the full-volume grid is keyed by FIELD (drift3d/weight3d) since
# e253dc8 / pochoir-efgb, not prefix+fine01.  See test_task13_full_leaf_naming.py.
@pytest.mark.parametrize("field,cfg,expect", [
    # Shapes are DERIVED from the live configs, so they move when the geometry
    # does: z = 0..160mm since the drift depth change, and 17x17 weighting
    # pixels (17 x 4.4mm pitch = 74.8mm) since a6abec7.  The interface is the
    # 40mm the runner passes.
    ("drift", DRIFT_FINE, {"coarse": ("11,11,401", "0.4*mm"),
                           "near": ("44,44,401", "0.1*mm"),
                           "drift3d": ("44,44,1601", "0.1*mm")}),
    ("weighting", WEIGHT_FINE, {"w_coarse": ("187,187,401", "0.4*mm"),
                                "w_near": ("748,748,401", "0.1*mm"),
                                "weight3d": ("748,748,1601", "0.1*mm")}),
])
def test_derived_shapes_at_default_spacings(field, cfg, expect):
    assert _shapes(field, cfg) == expect


def test_near_and_full_grids_share_the_fine_lattice():
    """Plane-for-plane stitch: same transverse shape and same spacing, the near
    grid just stops at the interface."""
    got = _shapes("drift", DRIFT_FINE)
    (nt_near, ny_near, nz_near) = got["near"][0].split(',')
    (nt_full, ny_full, nz_full) = got["drift3d"][0].split(',')
    assert (nt_near, ny_near) == (nt_full, ny_full)
    assert got["near"][1] == got["drift3d"][1]
    assert int(nz_near) < int(nz_full)
    # near depth is exactly the interface at the fine spacing
    assert int(nz_near) == int(round(INTERFACE_MM / 0.1)) + 1


def test_derived_shapes_fail_loudly_on_indivisible_spacing():
    cfg = json.loads(DRIFT_FINE.read_text())
    with pytest.raises(ValueError, match="whole number"):
        hi._derive_grids(hi._profile("drift"), cfg,
                         dict(coarse=0.4, fine=0.03), INTERFACE_MM)


# --------------------------------------------------------------------------
# Iteration machinery is gone
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dead", ["_kk", "_outer_iteration", "_final_stage",
                                  "_max_abs_delta"])
def test_iteration_helpers_deleted(dead):
    assert not hasattr(hi, dead)


def test_keys_carry_no_iteration_suffix():
    prof = hi._profile("drift")
    assert hi._key(prof, "potential", "near") == "potential/near"
    assert hi._key(hi._profile("weighting"), "potential", "near") == \
        "potential/w_near"


def test_driver_signature_has_precision_and_the_schwarz_tol():
    """`precision` is the per-solve fdm precision.  `tol` came BACK with the
    banded sweep, but as the inter-sweep Schwarz tolerance handed to
    near-far-solve -- not the outer convergence loop task13 removed, whose
    names must still be gone."""
    params = inspect.signature(hi.hybrid_iterate).parameters
    assert "precision" in params
    assert params["precision"].default == 2e-8
    assert params["tol"].default == hi.DEFAULT_TOL
    for dead in ("max_iters", "full_spacing"):
        assert dead not in params


def test_driver_source_has_no_convergence_loop():
    """Still no outer loop: max_sweeps caps the sweeps inside near-far-solve,
    it is not iterated here."""
    src = inspect.getsource(hi.hybrid_iterate)
    assert "while" not in src and "max_iters" not in src


# --------------------------------------------------------------------------
# Interface consistency check survives
# --------------------------------------------------------------------------

def test_check_interface_accepts_the_matching_plane():
    prof = hi._profile("drift")
    cfg = json.loads(DRIFT_FINE.read_text())
    grids = hi._derive_grids(prof, cfg, hi.DEFAULT_SPACINGS, INTERFACE_MM)
    hi._check_interface(prof, grids, '40*mm')     # must not raise


def test_check_interface_rejects_a_mismatch():
    prof = hi._profile("drift")
    cfg = json.loads(DRIFT_FINE.read_text())
    grids = hi._derive_grids(prof, cfg, hi.DEFAULT_SPACINGS, INTERFACE_MM)
    with pytest.raises(ValueError, match="disagrees with the split"):
        hi._check_interface(prof, grids, '30*mm')


# --------------------------------------------------------------------------
# --domain no
# --------------------------------------------------------------------------

def test_manual_grids_requires_every_leaf():
    prof = hi._profile("drift")
    with pytest.raises(ValueError, match="missing: near, fine01"):
        hi._manual_grids(prof, dict(coarse="11,11,151"), hi.DEFAULT_SPACINGS)


def test_manual_grids_rejects_malformed_shape():
    prof = hi._profile("drift")
    shapes = dict(coarse="11,11,151", near="44,44", fine01="44,44,601")
    with pytest.raises(ValueError, match='must be "nx,ny,nz"'):
        hi._manual_grids(prof, shapes, hi.DEFAULT_SPACINGS)


def test_manual_grids_uses_the_matching_spacing_per_leaf():
    prof = hi._profile("drift")
    shapes = dict(coarse="11,11,151", near="44,44,201", fine01="44,44,601")
    grids = hi._manual_grids(prof, shapes, dict(coarse=0.4, fine=0.1))
    assert grids == (("domain/coarse", "11,11,151", "0.4*mm"),
                     ("domain/near", "44,44,201", "0.1*mm"),
                     ("domain/drift3d", "44,44,601", "0.1*mm"))


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dead", ["tol", "max_iters", "full_spacing",
                                  "near_coarse_shape", "full_shape"])
def test_removed_cli_options(dead):
    assert dead not in _opts()


def test_precision_option_added():
    opt = _opts()["precision"]
    assert opt.default == 2e-8
    assert "--precision" in opt.opts


def test_fine_spacing_default_is_01_and_fine_shape_exists():
    opts = _opts()
    assert opts["fine_spacing"].default == 0.1
    assert opts["coarse_spacing"].default == 0.4
    assert "--fine-shape" in opts["fine_shape"].opts


def test_command_name_still_hybrid_iterate():
    assert hi_cmd.name == "hybrid-iterate"


def test_runner_script_passes_no_removed_flags():
    txt = (REPO_ROOT / "test" / "run-task13-hybrid.sh").read_text()
    # only look at the actual invocation flags, not the header prose
    flags = [ln for ln in txt.splitlines()
             if ln.strip().startswith("--")]
    body = "\n".join(flags)
    for dead in ("--tol", "--max-iters", "--full-spacing",
                 "--near-coarse-shape", "--full-shape"):
        assert dead not in body, f"{dead} was removed from the CLI"


# --------------------------------------------------------------------------
# The banded near/far Schwarz sweep (pochoir-4ipv)
#
# The one-shot pass is now sweep-0: it is followed by ONE banded near/far
# Schwarz sweep before the stitch.  Deeper per-argument coverage lives in
# test_hybrid_schwarz_step.py / test_hybrid_schwarz_wiring.py; what is pinned
# here is what this file has always pinned -- the scheme's shape, its store
# keys, and that the grids did NOT move because of it.
# --------------------------------------------------------------------------

import numpy

from pochoir import nearfar
from pochoir.domain import Domain
from pochoir.__main__ import near_far_solve


class _RecordingCtx:
    '''ctx.invoke recorder whose invocations "produce" their output keys, so
    the _want resume guard is satisfied.'''

    def __init__(self, tmp_path):
        self.calls = []

        class _Obj:
            instore_path = tmp_path

        self.obj = _Obj()

    def invoke(self, cmd, **kwds):
        self.calls.append((cmd, kwds))
        for key in (kwds.get('near_out'), kwds.get('far_out'),
                    kwds.get('output')):
            if key:
                path = self.obj.instore_path / (key + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(path, arr=numpy.zeros(2))


def _schwarz_kwds(tmp_path, field="drift", **kwds):
    ctx = _RecordingCtx(tmp_path)
    prof = hi._profile(field)
    kwds.setdefault("interface", "40*mm")
    interface = kwds.pop("interface")
    out = hi._schwarz(ctx, prof, hi._key(prof, "potential", "near"),
                      interface, **kwds)
    cmd, invoked = ctx.calls[0]
    assert cmd is near_far_solve
    return invoked, out


def test_schwarz_maps_band_and_sweeps_onto_near_far_solve(tmp_path):
    """--band-cells -> --overlap, --max-sweeps -> --max-iters."""
    kwds, _ = _schwarz_kwds(tmp_path, band_cells=2, max_sweeps=1)
    assert kwds["overlap"] == 2
    assert kwds["max_iters"] == 1


def test_schwarz_uses_one_precision_for_both_sides(tmp_path):
    """near-far-solve's own 2e-11/2e-7 split must not leak back in: this
    module has a SINGLE precision."""
    kwds, _ = _schwarz_kwds(tmp_path, precision=2e-8)
    assert kwds["near_precision"] == kwds["far_precision"] == 2e-8


@pytest.mark.parametrize("field,near_key,far_key", [
    ("drift", "potential/near_schwarz", "potential/coarse_schwarz"),
    ("weighting", "potential/w_near_schwarz", "potential/w_coarse_schwarz"),
])
def test_schwarz_output_keys_are_namespaced_per_field(tmp_path, field,
                                                      near_key, far_key):
    """Both fields share one store, so the sweep keys carry the w_ prefix for
    weighting exactly as the sweep-0 keys do."""
    kwds, out = _schwarz_kwds(tmp_path, field=field)
    assert out == (near_key, far_key)
    assert (kwds["near_out"], kwds["far_out"]) == (near_key, far_key)


def _drive(monkeypatch, tmp_path, **kwds):
    '''Run the driver with the heavy steps stubbed; return the stitch inputs
    and whether the sweep ran.'''
    seen = {}

    monkeypatch.setattr(hi, "_domains", lambda *a, **k: None)
    monkeypatch.setattr(hi, "_generate", lambda *a, **k: None)
    monkeypatch.setattr(hi, "_solve", lambda *a, **k: None)
    monkeypatch.setattr(hi, "_check_interface", lambda *a, **k: None)
    monkeypatch.setattr(
        hi, "_near_solve",
        lambda ctx, prof, precision, log: hi._key(prof, "potential", "near"))

    def fake_schwarz(ctx, prof, near_pot, interface, **kw):
        seen["swept"] = True
        return (hi._key(prof, "potential", "near_schwarz"),
                hi._key(prof, "potential", "coarse_schwarz"))

    def fake_stitch(ctx, prof, near_pot, coarse_pot, log):
        seen["stitch"] = (near_pot, coarse_pot)
        return hi._output_key(prof)

    monkeypatch.setattr(hi, "_schwarz", fake_schwarz)
    monkeypatch.setattr(hi, "_stitch", fake_stitch)

    shapes = dict(coarse="11,11,401", near="44,44,401", fine01="44,44,1601")
    final, _ = hi.hybrid_iterate(
        _RecordingCtx(tmp_path), "c.json", "f.json", derive_domain=False,
        shapes=shapes, log=lambda msg: None, **kwds)
    return final, seen


def test_driver_stitches_the_swept_fields_by_default(monkeypatch, tmp_path):
    final, seen = _drive(monkeypatch, tmp_path, interface="40*mm")
    assert seen["swept"] is True
    assert seen["stitch"] == ("potential/near_schwarz",
                              "potential/coarse_schwarz")
    assert final == "potential/drift3d"


def test_max_sweeps_zero_restores_the_one_shot_path(monkeypatch, tmp_path):
    """max_sweeps=0 skips _schwarz entirely and stitches the sweep-0 keys."""
    final, seen = _drive(monkeypatch, tmp_path, interface="40*mm",
                         max_sweeps=0)
    assert "swept" not in seen
    assert seen["stitch"] == ("potential/near", "potential/coarse")
    assert final == "potential/drift3d"


@pytest.mark.parametrize("field,expect", [
    ("drift", "potential/drift3d"),
    ("weighting", "potential/weight3d"),
])
def test_output_key_unchanged_by_the_sweep(monkeypatch, tmp_path, field,
                                           expect):
    """velo / induce-pixel are invoked with these names from the shell script,
    so the sweep must not rename the final field."""
    for max_sweeps in (1, 0):
        final, _ = _drive(monkeypatch, tmp_path, interface="40*mm",
                          field=field, max_sweeps=max_sweeps)
        assert final == expect == hi._output_key(hi._profile(field))


# --------------------------------------------------------------------------
# Band geometry at the runner's settings
# --------------------------------------------------------------------------

def _domains_at_default_spacings(field="drift", cfg=None):
    '''Real Domain objects for the coarse and near grids as derived.'''
    prof = hi._profile(field)
    cfg = json.loads((cfg or DRIFT_FINE).read_text())
    grids = hi._derive_grids(prof, cfg, hi.DEFAULT_SPACINGS, INTERFACE_MM)
    doms = {}
    for key, shape, spacing in grids:
        n = [int(v) for v in shape.split(',')]
        sp = float(spacing.split('*')[0])
        doms[key.split('/')[-1]] = Domain(n, [sp] * 3, [0.0, 0.0, 0.0])
    return doms


def test_band_indices_at_interface_40mm_coarse_04mm():
    """--band-cells 2 at --interface 40*mm and coarse 0.4mm: the far Dirichlet
    plane is coarse node 98 = z 39.2mm, two cells below node 100."""
    doms = _domains_at_default_spacings()
    ci, ci_in, nt, nt_in = nearfar._interface_indices(
        doms['coarse'], doms['near'], 2, INTERFACE_MM,
        overlap=hi.DEFAULT_BAND_CELLS)
    assert ci == 100
    assert ci_in == 98
    assert ci_in * hi.DEFAULT_SPACINGS['coarse'] == pytest.approx(39.2)
    # the near top plane is the interface itself
    assert nt == int(round(INTERFACE_MM / hi.DEFAULT_SPACINGS['fine']))
    assert nt_in == nt - 1


def test_inner_band_node_lies_inside_the_near_grid():
    """39.2mm < 40mm, so the band needs NO near-grid reshaping -- the same
    near shape serves overlap 1 and 2."""
    doms = _domains_at_default_spacings()
    for overlap in (1, hi.DEFAULT_BAND_CELLS):
        nearfar._interface_indices(doms['coarse'], doms['near'], 2,
                                   INTERFACE_MM, overlap=overlap)  # no raise


def test_absurd_band_is_rejected_not_silently_clamped():
    """An overlap deeper than the near grid must fail loudly."""
    doms = _domains_at_default_spacings()
    with pytest.raises(ValueError):
        nearfar._interface_indices(doms['coarse'], doms['near'], 2,
                                   INTERFACE_MM, overlap=0)


def test_grid_shapes_are_untouched_by_the_sweep():
    """The band changes no geometry: the derived shapes are exactly the ones
    pinned above, at both band widths."""
    before = _shapes("drift", DRIFT_FINE)
    assert before["near"][0] == "44,44,401"
    assert before["coarse"][0] == "11,11,401"
    assert before["drift3d"][0] == "44,44,1601"
