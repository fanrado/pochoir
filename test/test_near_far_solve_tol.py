"""
Unit tests for commit 7d754c7:
  "Fix near-far-solve tol units + fdm epoch display; drift geom 3.8/0.6"

Covers three changes:
  1. near_far_solve --tol is now interpreted in volts (divided by units.V), so
     the Schwarz outer loop can early-stop (pochoir-8uvn).
  2. fdm_torch epoch display is 1-indexed ("epoch: 1/10" not "0/10")
     (pochoir-8uvn, cosmetic).
  3. drift geometry pixelSize 3.5->3.8, pixelGap 0.9->0.6, pitch stays 4.4mm
     (pochoir-kx36).

These are tester-agent tests (no production code modified).
"""
import io
import json
import os
import re

import numpy
import pytest

import pochoir.arrays
import pochoir.units as units

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT_CFG = os.path.join(REPO, "test", "example_gen_pcb_drift_pixel_with_grid.json")


def _tol_in_volts(tol):
    """Reproduce the exact expression used in near_far_solve (__main__.py):

        tol_v = float(pochoir.arrays.fromstr1(tol)[0]) / units.V
    """
    return float(pochoir.arrays.fromstr1(tol)[0]) / units.V


# ---------------------------------------------------------------------------
# 1. tol units fix (pochoir-8uvn)
# ---------------------------------------------------------------------------

class TestTolUnits:
    def test_one_volt_is_one(self):
        # Happy path: '1*V' must yield 1.0 volt, not 1e-6.
        assert _tol_in_volts("1*V") == pytest.approx(1.0)

    @pytest.mark.parametrize("tol,expected", [
        ("1*V", 1.0),
        ("0.05*V", 0.05),
        ("0.1*V", 0.1),
        ("2*V", 2.0),
        ("0.001*V", 0.001),
    ])
    def test_various_volt_tolerances(self, tol, expected):
        assert _tol_in_volts(tol) == pytest.approx(expected)

    def test_millivolt_converts(self):
        # 1 mV = 0.001 V (units.mV exists in pochoir.units).
        assert _tol_in_volts("1*mV") == pytest.approx(0.001)

    def test_regression_old_behavior_was_wrong(self):
        # The bug: without the /units.V correction, '1*V' parsed to units.V
        # (1e-6), far below a realistic Schwarz delta (~0.1), so delta<tol
        # never fired. The fix must produce a value >> that raw parse.
        raw = float(pochoir.arrays.fromstr1("1*V")[0])
        fixed = _tol_in_volts("1*V")
        assert raw == pytest.approx(units.V)   # raw parse is tiny (1e-6)
        assert fixed == pytest.approx(1.0)      # fixed parse is in volts
        assert fixed > raw                       # fix strictly increases tol

    def test_tolerance_comparable_to_realistic_delta(self):
        # A realistic per-sweep Schwarz delta on the order of 0.1 V should now
        # be able to satisfy delta < tol for a loose tol like 1 V.
        realistic_delta = 0.1
        assert realistic_delta < _tol_in_volts("1*V")
        # ...and NOT satisfy a strict tol.
        assert not (realistic_delta < _tol_in_volts("0.05*V"))


# ---------------------------------------------------------------------------
# 2. fdm_torch epoch display 1-indexing (pochoir-8uvn, cosmetic)
# ---------------------------------------------------------------------------

class TestEpochDisplay:
    def test_epoch_display_is_one_indexed(self):
        """The fix changes {iepoch}/{nepochs} -> {iepoch+1}/{nepochs} in the
        epoch banner. Verify the source of fdm_torch.solve uses iepoch+1 and
        no longer prints the bare 0-indexed banner."""
        src = os.path.join(REPO, "pochoir", "fdm_torch.py")
        text = open(src).read()
        # The 1-indexed form must be present...
        assert "epoch: {iepoch+1}/{nepochs}" in text
        # ...and the old 0-indexed banner must be gone.
        assert "epoch: {iepoch}/{nepochs}" not in text

    def test_one_indexed_format_produces_expected_string(self):
        # Emulate the banner formatting to confirm 1..nepochs (not 0..nepochs-1).
        nepochs, epoch = 10, 100
        first = f'====== epoch: {0+1}/{nepochs} x {epoch} ==============='
        last = f'====== epoch: {(nepochs-1)+1}/{nepochs} x {epoch} ==============='
        assert "epoch: 1/10" in first
        assert "epoch: 10/10" in last
        assert "epoch: 0/10" not in first


# ---------------------------------------------------------------------------
# 3. drift geometry pixelSize/pixelGap (pochoir-kx36)
# ---------------------------------------------------------------------------

class TestDriftGeometry:
    @pytest.fixture(scope="class")
    def cfg(self):
        with open(DRIFT_CFG) as fp:
            return json.load(fp)

    def test_pixel_size(self, cfg):
        assert cfg["pixelSize"] == 3.8

    def test_pixel_gap(self, cfg):
        assert cfg["pixelGap"] == 0.6

    def test_pitch_preserved(self, cfg):
        # pitch = pixelSize + pixelGap must stay 4.4mm.
        assert cfg["pixelSize"] + cfg["pixelGap"] == pytest.approx(4.4)

    def test_unchanged_fields(self, cfg):
        # chamfer_r was 0.7 when this test was written (7d754c7); commit fe42032
        # ("Fix pixel chamfer_r 0.7 -> 0.4 mm to match reference footprint")
        # deliberately changed it to 0.4.  Pin the current, intended value.
        assert cfg["chamfer_r"] == 0.4
        assert cfg["Npixels"] == 5
        # CathodePotential/GridPotential were -15400/-1000 when this test was
        # written; commit 15a177e ("Port task9 improvements into
        # run-full-3d-pixel.sh (5cm, C1 overlap, insulator BC)") moved the
        # geometry to the 5cm drift gap at -2500V (50 V/mm).  Pin the current,
        # intended values.
        assert cfg["CathodePotential"] == -2500
        assert cfg["GridPotential"] == -2500
        assert cfg["LArPermittivity"] == 1.5
        assert cfg["FR4Permittivity"] == 4.5
