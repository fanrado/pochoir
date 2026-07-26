#!/usr/bin/env pytest
'''
Tests for commit 45ecc23 "m4lu: run drift near-far-solve in task9 (C1 interface,
no v_z seam step)".

SHELL only.  The task9 hybrid DRIFT field previously stitched the 0.05mm near
solve onto the raw coarse far with only the single-shot C0 near-bc Dirichlet pin,
so E_z (hence v_z) stepped ~0.8% across the z=20mm seam.  This commit:

  * adds Step 4b -- pochoir near-far-solve for the DRIFT field (overlapping
    Schwarz), the near re-solve insulator-aware (--insulator initial/near_insulator,
    pochoir-oz2l); the coarse far stays maskless.  -> C1 (gradient-continuous).
  * Step 5 stitch now uses the Schwarz-updated far field (potential/far) instead
    of the raw coarse bulk (potential/coarse).

Complements test_task9_stress_wide_gap_insul.py (the configs + rest of the
pipeline, still valid).  The AC (v_z continuous across z=20mm) needs the heavy
hybrid re-run and is handed off.
'''

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "test" / "run-task9-stress-wide-gap-insul.sh"


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


def _near_far_solve_lines():
    return [l for l in _logical_lines(RUNNER) if "pochoir near-far-solve" in l]


def _drift_nfs():
    # The DRIFT Schwarz solve is the periodic-transverse one (per,per,fix);
    # the weighting one is fix,fix,fix.
    hits = [l for l in _near_far_solve_lines() if "per,per,fix" in l]
    assert len(hits) == 1
    return hits[0]


# --------------------------------------------------------------------------

def test_runner_bash_syntax_ok():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_two_near_far_solves_now_present():
    # One for the drift field (new, this commit) and one for the weighting field.
    assert len(_near_far_solve_lines()) == 2


def test_drift_near_far_solve_is_insulator_aware():
    line = _drift_nfs()
    # Near re-solve carries the FR4 mask; coarse/far side is maskless (single
    # --insulator flag, applied to the near solver via oz2l).
    assert "--insulator initial/near_insulator" in line
    assert line.count("--insulator") == 1
    assert "--near-potential potential/near" in line
    assert "--near-out potential/near" in line
    assert "--far-out potential/far" in line


def test_drift_near_far_solve_uses_drift_edges_and_tol():
    line = _drift_nfs()
    assert "--edges per,per,fix" in line       # drift transverse-periodic
    # The drift tol was '1*V' when this test was written.  The runner was later
    # retuned to '0.001*V' (matching the weighting near-far-solve) because the
    # first Schwarz sweep changes the near solution by ~0.6 V, so '1*V' declared
    # convergence after ONE sweep and never reconciled the interface gradient
    # (near E_z 48.3 vs far 49.8 -> a v_z step).  Pin the current, intended value.
    assert "--tol '0.001*V'" in line
    assert "--coarse-potential potential/coarse" in line


def test_stitch_uses_schwarz_updated_far_field():
    stitch = [l for l in _logical_lines(RUNNER)
              if "pochoir stitch-near" in l and "drift3d" in l]
    assert len(stitch) == 1
    # C1 stitch: pull the far side from the Schwarz-updated potential/far,
    # NOT the raw coarse bulk.
    assert "--coarse potential/far" in stitch[0]
    assert "--coarse potential/coarse" not in stitch[0]
    assert "--near potential/near" in stitch[0]


def test_header_no_longer_omits_schwarz_for_drift():
    text = RUNNER.read_text()
    # The old header claimed the Schwarz step was OMITTED for the drift field;
    # that wording must be gone now that it is run.
    assert "OMITTED for the DRIFT" not in text
    assert "now RUN for the DRIFT field" in text
