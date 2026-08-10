#!/usr/bin/env pytest
'''
Tests for Phase B: the task13 configs carry a self-describing "_description"
key and NOTHING else changed (0da1c0c for the drift pair, 9db7244 for the
weighting pair).

The configs are plain dict lookups with no schema validation, so an extra key
is inert -- but "inert" is a claim worth testing, and so is every claim the
prose makes.  What is pinned:

  * every file still parses, and every pre-existing key is byte-identical to
    the pre-_description revision, with _description the only addition;
  * the extra key is inert: _derive_grids returns the same shapes;
  * the prose AGREES with the file it sits in -- the pad/gap split it quotes
    is the file's own, both splits really do divide their own spacing, and
    both files really are the same pitch;
  * the cathode formula is arithmetically right, and the value the prose says
    the file "currently holds" is the value the file actually holds.

NOTE -- deliberately NOT pinned: CathodePotential itself, or any field value
derived from it.  The user maintains that number by hand and it is expected to
change (-7500 -> -9920).  These tests check self-consistency, so they follow
the value instead of fighting it.
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
    nothing altered.'''
    path = DRIFT[which]
    before = _at_rev(f"{DRIFT_DESC_REV}^", path)
    after = _load(path)
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

@pytest.mark.parametrize("which", sorted(DRIFT))
def test_cathode_formula_geometry_matches_the_file(which):
    '''z_grid = pixelPlaneLowEdgePosition + PcbWidth, and the quoted numbers
    are the file's own.'''
    cfg = _load(DRIFT[which])
    z_grid = cfg["pixelPlaneLowEdgePosition"] + cfg["PcbWidth"]
    assert z_grid == pytest.approx(11.6)
    desc = cfg["_description"]
    assert f'{cfg["pixelPlaneLowEdgePosition"]} + {cfg["PcbWidth"]}' in desc
    assert "11.6mm" in desc
    # z_cathode: driftZDepth rounded up to a whole coarse cell.
    depth = cfg["driftZDepth"]
    z_cathode = hi.DEFAULT_SPACINGS["coarse"] * -(-depth //
                                                  hi.DEFAULT_SPACINGS["coarse"])
    assert z_cathode == pytest.approx(160.0)
    assert f"{depth} -> 160.0mm" in desc


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_quoted_cathode_arithmetic_is_right(which):
    """The worked example must actually work out -- but the TARGET bulk field
    is read out of the prose, not hardcoded here.  The detector's wanted field
    is the user's to set (and to re-document); this only checks that whatever
    the prose claims is self-consistent with the file's own geometry and
    GridPotential."""
    import re

    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    bulk = 160.0 - (cfg["pixelPlaneLowEdgePosition"] + cfg["PcbWidth"])
    assert bulk == pytest.approx(148.4)
    assert f"{bulk}mm" in desc

    worked = re.search(r"At (-?[\d.]+) V/mm that gives .*?= (-?[\d.]+) V",
                       desc)
    assert worked, desc[-400:]
    target_field, quoted_result = (float(worked.group(1)),
                                   float(worked.group(2)))
    assert cfg["GridPotential"] - target_field * bulk == \
        pytest.approx(quoted_result)


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_prose_reports_the_value_the_file_actually_holds(which):
    '''The prose says what the file "currently holds".  That number is
    maintained BY HAND and is expected to change, so this follows the file
    rather than pinning either value.'''
    import re

    cfg = _load(DRIFT[which])
    desc = cfg["_description"]
    quoted = re.search(r"currently holds CathodePotential (-?[\d.]+)", desc)
    assert quoted, desc[-400:]
    assert float(quoted.group(1)) == pytest.approx(cfg["CathodePotential"])

    # ...and the implied field, quoted to 2dp, follows from the same value.
    implied = re.search(r"i\.e\. ([\d.]+) V/mm", desc)
    assert implied, desc[-400:]
    bulk = 160.0 - (cfg["pixelPlaneLowEdgePosition"] + cfg["PcbWidth"])
    actual = abs(cfg["CathodePotential"] - cfg["GridPotential"]) / bulk
    assert float(implied.group(1)) == pytest.approx(actual, abs=0.01)


@pytest.mark.parametrize("which", sorted(DRIFT))
def test_prose_says_the_value_is_hand_maintained(which):
    desc = _load(DRIFT[which])["_description"]
    assert "BY HAND" in desc
    assert "CathodePotential = GridPotential - BulkField" in desc


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
    path = WEIGHT[which]
    before = _at_rev(f"{WEIGHT_DESC_REV}^", path)
    after = _load(path)
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
    '''Npixels x pitch = the quoted transverse extent.'''
    cfg = _load(WEIGHT[which])
    npix = cfg["Npixels"]
    pitch = round(cfg["pixelSize"] + cfg["pixelGap"], 9)
    extent = round(npix * pitch, 9)
    desc = cfg["_description"]
    assert pitch == 4.4
    assert extent == 74.8
    assert f"Npixels {npix} at the {pitch}mm pitch = {extent}mm" in desc


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
