#!/usr/bin/env pytest
'''
Tests for commit 6126539 "Task1 runner+config: 2cm drift, 0.05mm, fixed-cell
chamfer (pochoir-a7ne)".

SHELL + CONFIG reproducibility artifacts (test/run-task1-fixedcell-2cm.sh +
test/example_gen_pcb_drift_pixel_padz10_fixedcell.json), no production code.  The
runner wraps run-near2cm-drift.sh to solve the drift field + paths for a 2cm-gap
fixed-cell-chamfer geometry at 0.05mm (NSHAPE 88,88,601, z=0..30mm, pad z=10mm,
-1000V -> -50 V/mm, pure Laplace / FR4 disabled), then reports the chamfer
footprint and an overshoot check.

Verifiable here: valid bash; the config physics/geometry; the generator yields a
pure-Laplace 3-tuple (eps None) with the pad at the expected z-cell and a
fixed-cell chamfer; the runner's env-var wrapping of run-near2cm-drift.sh.  The
heavy 0.05mm solve is experimenter compute (handed off; the store was produced
by the experimenter).

NOTE (reproducibility): the runner references two analysis helpers --
compute_chamfer_removed.py (step 1/3) and OVERSHOOT_STUDY/check_paths.py (step
3/3) -- that are present on disk but NOT git-tracked; a clean checkout of this
commit would fail those steps.  Filed as a repo-health bug; asserted here only
that the referenced files exist on disk (the tracked-status requirement lives in
the bug, not as a red test).
'''

import json
import subprocess
from pathlib import Path

import pytest

from pochoir.domain import Domain
import pochoir.gen_pcb_drift_pixel_with_grid as dgen

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "test" / "run-task1-fixedcell-2cm.sh"
CFG = REPO_ROOT / "test" / "example_gen_pcb_drift_pixel_padz10_fixedcell.json"


def _logical_lines(path):
    out = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if out and out[-1].rstrip().endswith("\\"):
            out[-1] = out[-1].rstrip()[:-1].rstrip() + " " + s
        else:
            out.append(raw)
    return out


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CFG.read_text())


# --------------------------------------------------------------------------
# valid bash + config physics
# --------------------------------------------------------------------------

def test_bash_syntax_ok():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_config_is_pure_laplace_fixed_cell(cfg):
    # FR4/grid disabled -> pure Laplace; fixed-cell chamfer.
    assert cfg["GridHoleShape"] == "None"
    assert cfg.get("enableFR4", False) is False
    assert cfg.get("enableInsulatorFR4", False) is False
    assert cfg["chamferMode"] == "fixed_cell"
    assert cfg["chamfer_r"] == 0.4


def test_config_geometry_and_field(cfg):
    assert cfg["pixelPlaneLowEdgePosition"] == 10
    assert cfg["pixelPlaneWidth"] == 0.1
    assert cfg["driftZDepth"] == 30
    assert cfg["pixelSize"] == 3.8
    assert cfg["pixelGap"] == 0.6
    # cathode -1000V over the 20mm gap (pad low edge 10 -> cathode 30) = 50 V/mm.
    gap = cfg["driftZDepth"] - cfg["pixelPlaneLowEdgePosition"]
    assert gap == 20
    assert abs(cfg["CathodePotential"]) / gap == pytest.approx(50.0)
    assert cfg["CathodePotential"] == -1000
    assert cfg["GridPotential"] == -1000


def test_generator_pure_laplace_pad_at_z200(cfg):
    # Reduced z-extent at 0.05mm to capture the pad plane (z-cell 200) cheaply.
    res = dgen.generator(Domain([88, 88, 250], [0.05] * 3), cfg)
    assert len(res) == 3, "pure-Laplace config must give a legacy 3-tuple"
    _arr, barr, eps = res
    assert eps is None
    import numpy
    zc = numpy.where((barr[:, 5, :] != 0).any(axis=0))[0]
    pad = zc[(zc >= 195) & (zc <= 210)]
    # pad low edge 10mm at 0.05mm -> z-cell 200; 0.1mm-thick pad -> 200..202.
    assert pad.min() == 200


# --------------------------------------------------------------------------
# runner wraps run-near2cm-drift.sh with the Task1 env
# --------------------------------------------------------------------------

def test_runner_wraps_near2cm_with_task1_env():
    line = [l for l in _logical_lines(RUNNER) if "run-near2cm-drift.sh" in l]
    assert line, "run-near2cm-drift.sh not invoked"
    txt = RUNNER.read_text()
    assert "NSHAPE=88,88,601" in txt          # 88x88 tile, 601 deep @0.05mm (30mm)
    assert "NSPACING='0.05*mm'" in txt
    assert "NLAUNCH=29.8" in txt              # launch just below the 30mm cathode
    assert "NTWIN=30" in txt
    assert "store_task1_fixedcell_2cm" in txt
    assert "example_gen_pcb_drift_pixel_padz10_fixedcell.json" in txt


def test_runner_is_drift_only():
    # Drift field + paths only; no weighting / induced current.  Check the
    # executable body, not the whole file: the runner's own header comment says
    # "no weighting, no induced current", so a bare substring search over the
    # raw text matches the prose that states the very property being tested.
    code = "\n".join(
        line for line in RUNNER.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ).lower()
    assert "induce" not in code
    assert "weight" not in code


# --------------------------------------------------------------------------
# reproducibility: referenced helpers exist (tracked-status is a filed bug)
# --------------------------------------------------------------------------

def test_referenced_helpers_exist_on_disk():
    # The runner invokes these; they must at least be present to run.
    assert (REPO_ROOT / "test" / "compute_chamfer_removed.py").exists()
    assert (REPO_ROOT / "test" / "OVERSHOOT_STUDY" / "check_paths.py").exists()
    assert (REPO_ROOT / "test" / "run-near2cm-drift.sh").exists()
