#!/usr/bin/env pytest
'''
Tests for Task 13 Phase 1 (commit 99fbd00): the fine + coarse geometry configs
that supersede the reverted fdb1d91.

  * example_gen_pcb_drift_pixel_task13_fine.json -- claimed to be a verbatim
    copy of example_gen_pcb_drift_pixel_task10b_insul.json with exactly three
    values changed (GridPotential/CathodePotential -1000 -> -2500, driftZDepth
    29.9 -> 59.9).  Drives the 0.05 mm near solve and the 0.1 mm final solve.
  * example_gen_pcb_drift_pixel_task13_coarse.json -- the same geometry
    transcribed onto the 0.4 mm coarse grid.

Two layers of testing:

  1. Config-level: the fine config really is task10b plus exactly those three
     values, the field stays -50 V/mm, and every coarse length is an exact
     multiple of 0.4 mm with the 4.4 mm pitch preserved.
  2. Derived-geometry: the real generator is run on all three grids the runner
     will use (coarse 11x11x151 @0.4, near 88x88x401 @0.05, final 44x44x601
     @0.1) and the numbers the commit message claims are checked against the
     actual arrays -- pad-plane index, 3-cell pad, free gap cells, pad span,
     eps=None, cathode on the last plane, insulator disjoint from the pad.

No solve is run.
'''

import json
from pathlib import Path

import numpy
import pytest

from pochoir.domain import Domain
import pochoir.gen_pcb_drift_pixel_with_grid as gen

TEST_DIR = Path(__file__).resolve().parent

FINE_CFG = TEST_DIR / "example_gen_pcb_drift_pixel_task13_fine.json"
COARSE_CFG = TEST_DIR / "example_gen_pcb_drift_pixel_task13_coarse.json"
TASK10B_CFG = TEST_DIR / "example_gen_pcb_drift_pixel_task10b_insul.json"

# The three grids the Task13 runner uses, per the DESIGN on pochoir-nm59.
GRIDS = {
    "coarse": dict(shape=[11, 11, 151], spacing=0.4, cfg=COARSE_CFG),
    "near": dict(shape=[88, 88, 401], spacing=0.05, cfg=FINE_CFG),
    "final01": dict(shape=[44, 44, 601], spacing=0.1, cfg=FINE_CFG),
}


@pytest.fixture(scope="module")
def fine():
    return json.loads(FINE_CFG.read_text())


@pytest.fixture(scope="module")
def coarse():
    return json.loads(COARSE_CFG.read_text())


@pytest.fixture(scope="module")
def task10b():
    return json.loads(TASK10B_CFG.read_text())


# ==========================================================================
# 1. Config level
# ==========================================================================

def test_both_configs_are_valid_json(fine, coarse):
    assert isinstance(fine, dict) and isinstance(coarse, dict)


def test_fine_differs_from_task10b_in_exactly_three_values(fine, task10b):
    assert set(fine) == set(task10b), "key set drifted from task10b"
    changed = {k for k in task10b if fine[k] != task10b[k]}
    assert changed == {"GridPotential", "CathodePotential", "driftZDepth"}


def test_fine_new_values_are_the_5cm_ones(fine, task10b):
    assert (task10b["GridPotential"], task10b["CathodePotential"],
            task10b["driftZDepth"]) == (-1000, -1000, 29.9)
    assert (fine["GridPotential"], fine["CathodePotential"],
            fine["driftZDepth"]) == (-2500, -2500, 59.9)


def _drift_gap(cfg, spacing):
    '''Domain top (cathode) minus pad top surface, in mm.'''
    pad_top = cfg["pixelPlaneLowEdgePosition"] + cfg["pixelPlaneWidth"]
    return (cfg["driftZDepth"] + spacing) - pad_top


def test_fine_keeps_task10b_field_of_minus_50V_per_mm(fine, task10b):
    # task10b: -1000 V over a 20 mm gap.  task13: -2500 V over 50 mm.
    assert _drift_gap(task10b, 0.1) == pytest.approx(20.0)
    assert _drift_gap(fine, 0.1) == pytest.approx(50.0)
    assert task10b["CathodePotential"] / 20.0 == pytest.approx(-50.0)
    assert fine["CathodePotential"] / 50.0 == pytest.approx(-50.0)


def test_fine_launch_plane_is_one_final_cell_below_the_cathode(fine):
    # driftZDepth 59.9 with the 0.1 mm final grid whose top plane is z=60 mm.
    assert fine["driftZDepth"] == pytest.approx(59.9)
    assert 60.0 - fine["driftZDepth"] == pytest.approx(0.1)
    # ...and an exact node on the 0.05 mm near grid too.
    assert (fine["driftZDepth"] / 0.05) == pytest.approx(round(fine["driftZDepth"] / 0.05))


def test_fine_is_the_task10b_validated_footprint(fine):
    # The revert of fdb1d91 was specifically because it used the task10a/task12
    # family (3.8/0.6, 0.4 mm chamfer) instead of task10b.
    assert fine["pixelSize"] == 3.5
    assert fine["pixelGap"] == 0.9
    assert fine["chamfer_r"] == 0.7
    assert fine["chamferMode"] == "dynamic"


def test_fine_transverse_lengths_land_on_both_fine_pitches(fine):
    for key in ("pixelPlaneLowEdgePosition", "pixelSize", "pixelGap",
                "chamfer_r", "FR4Thickness", "pixelPlaneWidth"):
        for spacing in (0.05, 0.1):
            cells = fine[key] / spacing
            assert cells == pytest.approx(round(cells)), \
                f"{key}={fine[key]} is not a whole number of {spacing} mm cells"


def test_pitch_is_44mm_in_both_configs(fine, coarse):
    for cfg in (fine, coarse):
        assert cfg["pixelSize"] + cfg["pixelGap"] == pytest.approx(4.4)


def test_pitch_matches_the_transverse_tile_of_every_grid(fine, coarse):
    for name, g in GRIDS.items():
        pitch = 4.4
        assert g["shape"][0] * g["spacing"] == pytest.approx(pitch), name
        assert g["shape"][0] == g["shape"][1], name


def test_coarse_gap_snaps_down_not_up(fine, coarse):
    # 0.9 mm is not a whole 0.4 mm cell.  Snapping UP to 1.2 would give a
    # 3.6+1.2 = 4.8 mm pitch and break the 11-cell coarse tile; snapping DOWN
    # to 0.8 keeps the pitch at exactly 4.4 mm.
    assert coarse["pixelGap"] == 0.8
    assert coarse["pixelGap"] < fine["pixelGap"]
    assert coarse["pixelSize"] + 1.2 == pytest.approx(4.8)   # the rejected option
    assert coarse["pixelSize"] + coarse["pixelGap"] == pytest.approx(4.4)


def test_coarse_lengths_are_all_whole_04mm_cells(coarse):
    for key in ("pixelPlaneLowEdgePosition", "pixelPlaneWidth", "padThickness",
                "FR4Thickness", "pixelSize", "pixelGap", "chamfer_r"):
        cells = coarse[key] / 0.4
        assert cells == pytest.approx(round(cells)), \
            f"{key}={coarse[key]} is not a whole number of 0.4 mm cells"


def test_coarse_plane_position_written_to_survive_int_truncation(coarse):
    # The generator does int(pixelPlaneLowEdgePosition/spacing).  9.9/0.4 =
    # 24.75 truncates to 24; 10.0 is needed to land on plane 25.
    assert coarse["pixelPlaneLowEdgePosition"] == 10.0
    assert int(9.9 / 0.4) == 24
    assert int(coarse["pixelPlaneLowEdgePosition"] / 0.4) == 25


def test_coarse_shares_the_fine_potentials_and_depth(fine, coarse):
    for key in ("GridPotential", "CathodePotential", "driftZDepth"):
        assert coarse[key] == fine[key], key


def test_coarse_shares_the_fine_solver_flags(fine, coarse):
    for key in ("GridHoleShape", "chamferMode", "Npixels", "padThicknessCells",
                "enableFR4", "enableInsulatorFR4"):
        assert coarse[key] == fine[key], key


def test_no_shield_grid_and_no_solid_fr4_in_either_config(fine, coarse):
    for cfg in (fine, coarse):
        assert cfg["GridHoleShape"] == "None"
        assert cfg["enableFR4"] is False
        assert cfg["enableInsulatorFR4"] is True


def test_pad_is_three_grounded_cells_in_both_configs(fine, coarse):
    for cfg in (fine, coarse):
        assert cfg["padThicknessCells"] >= 3


# ==========================================================================
# 2. Derived geometry, through the real generator
# ==========================================================================

@pytest.fixture(autouse=True)
def _no_plots(monkeypatch):
    monkeypatch.setattr(gen, "plot_barr_3d", lambda *a, **k: None, raising=False)


def _derived(name):
    g = GRIDS[name]
    cfg = json.loads(Path(g["cfg"]).read_text())
    sp = g["spacing"]
    out = gen.generator(Domain(list(g["shape"]), [sp] * 3), cfg)
    arr, barr, eps = out[0], out[1], out[2]
    insulator = out[3] if len(out) > 3 else None

    # Reproduce the generator's own derivation of the pad-plane geometry.
    pp_lower = int(cfg["pixelPlaneLowEdgePosition"] / sp)
    n_fr4 = max(1, round(cfg["FR4Thickness"] / sp))
    n_pad = max(1, round(cfg["padThickness"] / sp))
    z_top = pp_lower + (n_pad + n_fr4 - 1)
    foot = barr[:, :, z_top] != 0
    return dict(cfg=cfg, spacing=sp, shape=g["shape"], arr=arr, barr=barr,
                eps=eps, insulator=insulator, pp_lower=pp_lower, n_fr4=n_fr4,
                z_top=z_top, foot=foot)


@pytest.fixture(scope="module")
def gcoarse():
    return _derived("coarse")


@pytest.fixture(scope="module")
def gnear():
    return _derived("near")


@pytest.fixture(scope="module")
def gfinal():
    return _derived("final01")


@pytest.fixture(scope="module")
def all_grids(gcoarse, gnear, gfinal):
    return dict(coarse=gcoarse, near=gnear, final01=gfinal)


def test_insulator_plane_index_matches_commit_message(all_grids):
    # Commit claims "pad plane at z-index 25 / 198 / 99".  That is pp_lower,
    # i.e. the low edge of the pixel-plane layer / the no-flux FR4 slab.
    assert all_grids["coarse"]["pp_lower"] == 25
    assert all_grids["near"]["pp_lower"] == 198
    assert all_grids["final01"]["pp_lower"] == 99


def test_epsilon_is_none_on_every_grid(all_grids):
    # enableFR4 false -> the harmonic-mean epsilon path must stay off even
    # though LArPermittivity/FR4Permittivity are present in the config.
    for name, g in all_grids.items():
        assert g["eps"] is None, name


def test_cathode_potential_on_the_last_plane_of_every_grid(all_grids):
    for name, g in all_grids.items():
        last = g["arr"][:, :, -1]
        assert numpy.allclose(last, -2500.0), f"{name}: {last.min()}..{last.max()}"


def test_pad_is_three_contiguous_grounded_planes(all_grids):
    for name, g in all_grids.items():
        foot, z_top = g["foot"], g["z_top"]
        for z in (z_top, z_top - 1, z_top - 2):
            assert numpy.array_equal(g["barr"][:, :, z] != 0, foot), \
                f"{name}: plane {z} is not the pad footprint"
            assert numpy.allclose(g["arr"][:, :, z][foot], 0.0), \
                f"{name}: plane {z} pad nodes not grounded"
        # ...and exactly three: the plane below is not the pad.
        assert not numpy.array_equal(g["barr"][:, :, z_top - 3] != 0, foot), name


def test_free_gap_cells_on_the_pad_plane_match_commit_message(all_grids):
    # padplane_noflux_geom raises unless the pad plane is only PARTIALLY
    # Dirichlet, so free gap cells must exist on every grid.
    expected = {"coarse": 40, "near": 2964, "final01": 735}
    for name, g in all_grids.items():
        free = int((~g["foot"]).sum())
        assert free > 0, f"{name}: pad plane fully Dirichlet, no-flux BC cannot derive"
        assert free == expected[name], f"{name}: {free} free gap cells"


def test_pad_span_is_the_physical_pixel_not_a_cell_count(all_grids):
    # Commit claims 9 cells = 3.60 mm coarse, 70 = 3.50 mm near, 35 = 3.50 mm
    # at 0.1 mm -- i.e. the exact 3.5 mm pixel on the fine grids, not 36 cells.
    expected = {"coarse": (9, 3.60), "near": (70, 3.50), "final01": (35, 3.50)}
    for name, g in all_grids.items():
        cells, mm = expected[name]
        span = max(int(g["foot"][i].sum()) for i in range(g["shape"][0]))
        assert span == cells, f"{name}: pad span {span} cells"
        assert span * g["spacing"] == pytest.approx(mm), name
        assert max(int(g["foot"][:, j].sum())
                   for j in range(g["shape"][1])) == cells, f"{name}: asymmetric"


def test_fine_grids_reproduce_the_35mm_pixel_exactly(gnear, gfinal):
    for g in (gnear, gfinal):
        span = max(int(g["foot"][i].sum()) for i in range(g["shape"][0]))
        assert span * g["spacing"] == pytest.approx(g["cfg"]["pixelSize"])


def test_insulator_mask_is_the_fr4_slab_below_the_pad(all_grids):
    for name, g in all_grids.items():
        insul = g["insulator"]
        assert insul is not None, f"{name}: enableInsulatorFR4 produced no mask"
        zs = [z for z in range(g["shape"][2]) if insul[:, :, z].any()]
        assert zs == list(range(g["pp_lower"], g["pp_lower"] + g["n_fr4"])), \
            f"{name}: mask planes {zs}"


def test_insulator_mask_is_disjoint_from_the_thick_pad(all_grids):
    for name, g in all_grids.items():
        assert not (g["insulator"] & (g["barr"] != 0)).any(), name


def test_fine_grids_put_the_pad_top_surface_at_10mm(gnear, gfinal):
    for g in (gnear, gfinal):
        assert g["z_top"] * g["spacing"] == pytest.approx(10.0)


def test_coarse_pad_top_sits_one_coarse_cell_high(gcoarse):
    '''KNOWN, INTENTIONALLY PINNED DISCREPANCY.

    The coarse config writes pixelPlaneLowEdgePosition as 10.0 (not 9.9) to
    dodge int() truncation, and the derived pp_width then adds one more 0.4 mm
    cell on top -- so the coarse pad TOP surface lands at 10.4 mm while both
    fine grids put it at 10.0 mm.  The commit message calls the coarse config
    "the same geometry transcribed onto the 0.4 mm coarse grid" and does not
    mention this 0.4 mm offset.

    Pinned rather than asserted-correct: the coarse solve is only the far-field
    seed and the whole z<20 mm near region is re-solved at 0.05 mm, so a
    one-coarse-cell offset perturbs initial values (~20 V) that the iteration
    removes.  If the offset is ever meant to be closed, 9.6 mm gives
    int(9.6/0.4)=24 -> pad top at exactly 10.0 mm, and this test should be
    updated deliberately.
    '''
    assert gcoarse["z_top"] * gcoarse["spacing"] == pytest.approx(10.4)


def test_no_grid_electrode_anywhere(all_grids):
    # GridHoleShape "None" -> the only Dirichlet sets are the pad and cathode.
    for name, g in all_grids.items():
        vals = numpy.unique(g["arr"][g["barr"] != 0])
        assert set(numpy.round(vals, 6)) <= {0.0, -2500.0}, f"{name}: {vals}"
