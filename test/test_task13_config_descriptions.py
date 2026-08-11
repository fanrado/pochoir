#!/usr/bin/env pytest
'''
Tests for Phase B: the task13 configs carry a self-describing "_description"
key and NOTHING else changed (0da1c0c for the drift pair, 9db7244 for the
weighting pair).

The configs are plain dict lookups with no schema validation, so an extra key
is inert -- but "inert" is a claim worth testing, and so is every claim the
prose makes.  What is pinned:

  * every file still parses, and -- checked AT the _description commit, since
    the live configs are the user's run state and keep moving -- every
    pre-existing key was byte-identical to the pre-_description revision, with
    _description the only addition;
  * the extra key is inert: _derive_grids returns the same shapes;
  * the prose AGREES with the file it sits in -- the pad/gap split it quotes
    is the file's own, both splits really do divide their own spacing, and
    both files really are the same pitch;
  * the cathode formula is arithmetically right, and the value the prose says
    the file holds is the value the file actually holds.

THE CATHODE PHYSICS (corrected, pochoir-1u0v).  The bulk field is set by the
CATHODE against the PIXEL PLANE at 0 V, over driftZDepth -
pixelPlaneLowEdgePosition.  GridPotential does NOT enter: the shield grid has
holes and is transparent at the tile centre, so it is not an equipotential
plane terminating the field -- measured phi at the grid plane is about -71 V
against a GridPotential of -2500.  An earlier revision of these tests pinned
the opposite ("GridPotential - BulkField * grid-to-cathode"), which was wrong.

NOTE -- deliberately NOT pinned: CathodePotential itself, the target bulk
field, or any value derived from them.  The user maintains those numbers by
hand and they are expected to change.  These tests read the target field and
the quoted result back OUT of the prose and check self-consistency against the
file's own geometry, so re-documenting stays free.
'''

import json
import subprocess
from pathlib import Path

import pytest

import pochoir.hybrid_iterate as hi

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDIR = REPO_ROOT / "test"

DRIFT = {
    "coarse": TESTDIR / "example_gen_pcb_drift_pixel_task13_coarse.json",
    "fine": TESTDIR / "example_gen_pcb_drift_pixel_task13_fine.json",
}
WEIGHT = {
    "coarse": TESTDIR / "example_gen_pixel_with_grid_task13_coarse.json",
    "fine": TESTDIR / "example_gen_pixel_with_grid_task13_fine.json",
}
ALL_CONFIGS = {**{f"drift-{k}": v for k, v in DRIFT.items()},
               **{f"weighting-{k}": v for k, v in WEIGHT.items()}}

DRIFT_DESC_REV = "0da1c0c"          # commit that added _description to DRIFT
INTERFACE_MM = 40.0


def _load(path):
    return json.loads(path.read_text())


def _at_rev(rev, path):
    rel = path.relative_to(REPO_ROOT)
    res = subprocess.run(["git", "show", f"{rev}:{rel}"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"{rev}:{rel} not reachable")
    return json.loads(res.stdout)


# --------------------------------------------------------------------------
# Parsing and value preservation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ALL_CONFIGS))
def test_config_still_parses(name):
    cfg = _load(ALL_CONFIGS[name])
    assert isinstance(cfg, dict) and cfg


@pytest.mark.parametrize("name", sorted(ALL_CONFIGS))
def test_description_key_present(name):
    cfg = _load(ALL_CONFIGS[name])
    assert isinstance(cfg["_description"], str)
    assert len(cfg["_description"]) > 100


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_drift_config_values_are_byte_identical(which):
    '''NO VALUE CHANGES: _description is the only added key, nothing removed,
    nothing altered -- IN THE COMMIT THAT ADDED IT.

    Read at the commit rather than from the working tree: these configs are the
    user's live run state and keep moving (drift depth, cathode, Npixels).  The
    claim being tested is about what 0da1c0c did, and that stays checkable
    forever only if it is checked where it was made.'''
    path = DRIFT[which]
    before = _at_rev(f"{DRIFT_DESC_REV}^", path)
    after = _at_rev(DRIFT_DESC_REV, path)
    assert set(after) - set(before) == {"_description"}
    assert set(before) - set(after) == set()
    for key, value in before.items():
        assert after[key] == value, key


@pytest.mark.parametrize("field,cfg_path", [("drift", DRIFT["fine"]),
                                            ("weighting", WEIGHT["fine"])])
def test_extra_key_is_inert_for_grid_derivation(field, cfg_path):
    prof = hi._profile(field)
    grids = hi._derive_grids(prof, _load(cfg_path), hi.DEFAULT_SPACINGS,
                             INTERFACE_MM)
    stripped = {k: v for k, v in _load(cfg_path).items()
                if k != "_description"}
    assert grids == hi._derive_grids(prof, stripped, hi.DEFAULT_SPACINGS,
                                     INTERFACE_MM)


# --------------------------------------------------------------------------
# The prose must agree with the file it sits in
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", sorted(DRIFT))
def test_quoted_pad_gap_split_is_this_files_own(which):
    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    mine = f'pixelSize {cfg["pixelSize"]} / pixelGap {cfg["pixelGap"]}'
    assert mine in desc, (mine, which)


def test_both_drift_configs_are_the_same_pitch():
    '''The prose's central claim: 3.6/0.8 and 3.5/0.9 are one 4.4mm pitch.'''
    pitches = set()
    for path in DRIFT.values():
        cfg = _load(path)
        pitches.add(round(cfg["pixelSize"] + cfg["pixelGap"], 9))
    assert pitches == {4.4}
    for path in DRIFT.values():
        assert "4.4mm pitch" in _load(path)["_description"]


@pytest.mark.parametrize("which,spacing", [("coarse", 0.4), ("fine", 0.1)])
def test_each_split_divides_its_own_spacing_exactly(which, spacing):
    '''...which is the stated reason two files exist.'''
    cfg = _load(DRIFT[which])
    assert spacing == hi.DEFAULT_SPACINGS[
        "coarse" if which == "coarse" else "fine"]
    for key in ("pixelSize", "pixelGap"):
        quotient = cfg[key] / spacing
        assert abs(quotient - round(quotient)) < 1e-9, (which, key, quotient)


def test_the_fine_split_really_does_not_divide_the_coarse_spacing():
    '''3.5/0.4 = 8.75 -- the claim that makes the pair necessary.'''
    fine = _load(DRIFT["fine"])
    quotient = fine["pixelSize"] / hi.DEFAULT_SPACINGS["coarse"]
    assert abs(quotient - round(quotient)) > 1e-9
    assert "8.75" in fine["_description"]


# --------------------------------------------------------------------------
# The cathode formula
# --------------------------------------------------------------------------

def _drift_length(cfg):
    '''The bulk the cathode works across: pixel plane (0 V) to cathode.

    NOT grid-to-cathode.  The shield grid has holes and is transparent at the
    tile centre, so it is not an equipotential plane terminating the field --
    measured phi there is a few tens of volts, not GridPotential (pochoir-1u0v,
    measured on store_task13_hybrid_15cmDrift/potential/coarse.npz).'''
    return round(cfg["driftZDepth"] - cfg["pixelPlaneLowEdgePosition"], 9)


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_cathode_formula_geometry_matches_the_file(which):
    '''The quoted drift length is the file's own driftZDepth minus its own
    pixelPlaneLowEdgePosition, and the quoted grid-plane height is
    pixelPlaneLowEdgePosition + PcbWidth.'''
    cfg = _load(DRIFT[which])
    desc = cfg["_description"]

    length = _drift_length(cfg)
    assert f'{cfg["driftZDepth"]} - {cfg["pixelPlaneLowEdgePosition"]} ' \
           f'= {length}mm' in desc

    z_grid = round(cfg["pixelPlaneLowEdgePosition"] + cfg["PcbWidth"], 9)
    assert f"z={z_grid}mm" in desc


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_quoted_cathode_arithmetic_is_right(which):
    """The worked example must actually work out -- but the TARGET bulk field
    is read out of the prose, not hardcoded here.  The detector's wanted field
    is the user's to set (and to re-document); this only checks that whatever
    the prose claims is self-consistent with the file's own geometry."""
    import re

    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    length = _drift_length(cfg)

    worked = re.search(
        r"so ([\d.]+) V/mm gives (-[\d.]+) \* ([\d.]+) = (-[\d.]+) V", desc)
    assert worked, desc[-400:]
    target_field = float(worked.group(1))
    assert float(worked.group(2)) == pytest.approx(-target_field)
    assert float(worked.group(3)) == pytest.approx(length)
    assert -target_field * length == pytest.approx(float(worked.group(4)),
                                                   abs=0.05)


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_prose_reports_the_value_the_file_actually_holds(which):
    '''The prose says what the file holds.  That number is maintained BY HAND
    and is expected to change, so this follows the file rather than pinning
    either value.'''
    import re

    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    quoted = re.search(r"This file holds (-?[\d.]+)", desc)
    assert quoted, desc[-400:]
    assert float(quoted.group(1)) == pytest.approx(cfg["CathodePotential"])

    # ...and the implied nominal field, quoted to 2dp, follows from the same
    # value over the file's own drift length -- GridPotential does not enter.
    implied = re.search(r"i\.e\. ([\d.]+) V/mm nominal", desc)
    assert implied, desc[-400:]
    actual = abs(cfg["CathodePotential"]) / _drift_length(cfg)
    assert float(implied.group(1)) == pytest.approx(actual, abs=0.01)


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_prose_says_the_value_is_hand_maintained(which):
    desc = _load(DRIFT[which])["_description"]
    assert "BY HAND" in desc
    assert "CathodePotential = -BulkField * " \
           "(driftZDepth - pixelPlaneLowEdgePosition)" in desc


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_prose_says_the_grid_does_not_terminate_the_field(which):
    '''The correction of pochoir-1u0v, pinned so it cannot quietly revert: the
    grid is transparent, so GridPotential is NOT the potential at the grid
    plane and does NOT enter the cathode formula.  The measured value quoted in
    the prose is checked to be a small fraction of GridPotential rather than
    hardcoded -- it is a measurement, and the user may re-measure it.'''
    import re

    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    assert "GridPotential does NOT enter this calculation" in desc
    assert "transparent at the tile centre" in desc

    # Phase 1 (b26c884) retargeted the configs to the 59.9mm drift and rewrote
    # this sentence to attribute the measurement to the earlier 149.9mm run,
    # ending "nothing like GridPotential" instead of "not GridPotential".
    # Accept either wording: what is pinned is that a MEASURED value is quoted
    # and that it is far from GridPotential, not the phrasing around it.
    measured = re.search(
        r"was? about (-?[\d.]+) V, (?:not|nothing like) GridPotential", desc)
    assert measured, desc[-400:]
    assert abs(float(measured.group(1))) < 0.1 * abs(cfg["GridPotential"]), \
        "a grid plane sitting near GridPotential would mean it IS terminating"


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_each_file_names_its_own_role_and_sibling(which):
    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    assert ("COARSE" if which == "coarse" else "FINE") in desc
    assert DRIFT["fine" if which == "coarse" else "coarse"].name in desc


# --------------------------------------------------------------------------
# Phase B.2: the weighting pair (commit 9db7244)
# --------------------------------------------------------------------------

WEIGHT_DESC_REV = "9db7244"


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_config_values_are_byte_identical(which):
    '''As for the drift pair: the claim is about commit 9db7244, so it is
    checked at that commit, not against the live working tree.'''
    path = WEIGHT[which]
    before = _at_rev(f"{WEIGHT_DESC_REV}^", path)
    after = _at_rev(WEIGHT_DESC_REV, path)
    assert set(after) - set(before) == {"_description"}
    assert set(before) - set(after) == set()
    for key, value in before.items():
        assert after[key] == value, key


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_configs_carry_no_potentials(which):
    '''The prose's central claim: a unit probe has no volts, so neither
    GridPotential nor CathodePotential belongs here.'''
    cfg = _load(WEIGHT[which])
    assert "GridPotential" not in cfg
    assert "CathodePotential" not in cfg
    desc = cfg["_description"]
    assert "NO GridPotential and NO CathodePotential" in desc
    assert "dimensionless unit probe in [0,1]" in desc


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_probe_extent_claim_matches_the_config(which):
    '''Npixels x pitch = the quoted transverse extent.

    The COUNT is read out of the prose, not out of the file, and not
    hardcoded: Npixels is a knob the user turns for the run they want (17 since
    a6abec7, 25 at the time of writing) and the prose is re-documented by hand
    afterwards.  What must hold is that the sentence is arithmetically true at
    whatever count it quotes, on this file's own pitch.'''
    import re

    cfg = _load(WEIGHT[which])
    pitch = round(cfg["pixelSize"] + cfg["pixelGap"], 9)
    desc = cfg["_description"]

    quoted = re.search(r"Npixels (\d+) at the ([\d.]+)mm pitch = ([\d.]+)mm",
                       desc)
    assert quoted, desc[-400:]
    npix, quoted_pitch, extent = (int(quoted.group(1)),
                                  float(quoted.group(2)),
                                  float(quoted.group(3)))
    assert quoted_pitch == pytest.approx(pitch)
    assert npix * pitch == pytest.approx(extent)


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_edges_claim_matches_the_profile(which):
    '''fix,fix,fix comes from the profile, not the config -- the prose must
    not disagree with it.'''
    assert hi._profile("weighting")["edges"] == "fix,fix,fix"
    assert "fix,fix,fix" in _load(WEIGHT[which])["_description"]


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_quoted_pad_gap_split_is_this_files_own(which):
    cfg = _load(WEIGHT[which])
    mine = f'pixelSize {cfg["pixelSize"]} / pixelGap {cfg["pixelGap"]}'
    assert mine in cfg["_description"], (mine, which)


def test_both_weighting_configs_are_the_same_pitch():
    pitches = {round(_load(p)["pixelSize"] + _load(p)["pixelGap"], 9)
               for p in WEIGHT.values()}
    assert pitches == {4.4}


@pytest.mark.parametrize("which,spacing", [("coarse", 0.4), ("fine", 0.1)])
def test_weighting_split_divides_its_own_spacing(which, spacing):
    cfg = _load(WEIGHT[which])
    for key in ("pixelSize", "pixelGap"):
        quotient = cfg[key] / spacing
        assert abs(quotient - round(quotient)) < 1e-9, (which, key, quotient)


@pytest.mark.parametrize("which", sorted(WEIGHT))
def test_weighting_file_names_its_own_role_and_sibling(which):
    cfg = _load(WEIGHT[which])
    desc = cfg["_description"]
    assert "WEIGHTING" in desc
    assert ("COARSE" if which == "coarse" else "FINE") in desc
    assert WEIGHT["fine" if which == "coarse" else "coarse"].name in desc
