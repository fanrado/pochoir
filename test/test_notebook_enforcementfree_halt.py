#!/usr/bin/env pytest
'''
Tests for commit 471d6ae "notebook: document enforcement-free task8 halt at FR4
(pochoir-0by4)".

NOTEBOOK-ONLY commit (test/PathPixBorder_neumannFullDetph_01mm.ipynb).  It
appends a geometry-reference cell + a markdown/code pair reporting that, after
pochoir-h3y1 removed the enforced terminal event and the in-FR4 v=0 zeroing,
task8 re-run on the SAME Neumann-BC drift potential still has all 1764 grid-node
paths halt inside the single FR4 cell band z in [9.85, 9.95] mm (z_min = 9.85mm,
none deeper) -- the no-flux field alone stops the electrons.

IMPORTANT (flagged separately as a repo-health bug): the production change that
this narrative depends on (pochoir-h3y1: removal of the terminal event + in-FR4
zeroing in pochoir/drift_numpy.py, plus the new pochoir/drift_numpy_enforced_
backup.py) is UNCOMMITTED / UNTRACKED in the working tree at the time this
notebook was committed.  A fresh checkout of 471d6ae still carries the committed
terminal-event code (09fb55c) and would contradict the notebook.  These tests do
NOT depend on that uncommitted code.

Verifiable here:
  * the notebook is valid nbformat v4 and carries the enforcement-free cells;
  * the laminate z-geometry the cells assert (FR4 insulator node 99, cell
    [9.85, 9.95] mm, top face 9.95 mm == pad-node-100 bottom) is reproducible
    directly from the committed drift generator, independent of the store and of
    the uncommitted drift code;
  * OPTIONALLY, when the experimenter store is present, the stored node paths
    are checked against the CURRENT drift contract.  NOTE: the original
    halt-in-FR4-band finding the notebook narrates (all endpoints in
    [9.85, 9.95] mm) was a naive-central-diff artifact; the store was
    RE-DRIFTED with the mask-aware efield fix (c8fb3eb, pochoir-9keo) and
    enforcement-free drift (pochoir-vsdz), after which every path terminates
    at the pad plane (pad node 100, cell [9.95, 10.05] mm) and none penetrate
    the FR4.  The store-backed test asserts that superseding contract, not
    the notebook's superseded numbers (pochoir-0z6u).  These store-backed
    checks SKIP when the store is absent (it is untracked and not in a fresh
    checkout).
'''

import json
from pathlib import Path

import nbformat
import numpy
import pytest

from pochoir.domain import Domain
import pochoir.gen_pcb_drift_pixel_with_grid as dgen

REPO_ROOT = Path(__file__).resolve().parent.parent
NB = REPO_ROOT / "test" / "PathPixBorder_neumannFullDetph_01mm.ipynb"
DRIFT_INSUL_CFG = REPO_ROOT / "test" / "example_gen_pcb_drift_pixel_with_grid_insul.json"
STORE = REPO_ROOT / "test" / "store_validate_neumann_fulldepth_01mm"

FR4_NODE = 99          # single FR4 insulator cell (centre 9.90mm, cell [9.85,9.95])
PAD_NODE = 100         # Cu pad node (centre 10.0mm, cell [9.95,10.05])
H = 0.1                # spacing (mm)


@pytest.fixture(autouse=True)
def _no_plots(monkeypatch):
    monkeypatch.setattr(dgen, "plot_barr_3d", lambda *a, **k: None, raising=False)


@pytest.fixture(scope="module")
def cells():
    nb = nbformat.read(str(NB), as_version=4)
    return [c.source for c in nb.cells]


@pytest.fixture(scope="module")
def drift_geometry():
    '''(insulator mask, boundary) from the committed drift generator on the same
    44x44x601 @0.1mm tile the fulldepth driver builds.'''
    cfg = json.loads(DRIFT_INSUL_CFG.read_text())
    arr, barr, eps, insul = dgen.generator(Domain([44, 44, 601], [0.1] * 3), cfg)
    return numpy.asarray(insul).astype(bool), numpy.asarray(barr)


# --------------------------------------------------------------------------
# Notebook validity + the enforcement-free cells
# --------------------------------------------------------------------------

def test_notebook_valid_nbformat():
    nb = nbformat.read(str(NB), as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) >= 15


def test_enforcement_free_cells_present(cells):
    joined = "\n".join(cells)
    # The narrative: enforcement removed in h3y1, backup kept, field-alone halt.
    assert "pochoir-h3y1" in joined
    assert "drift_numpy_enforced_backup.py" in joined
    assert "Enforcement-free" in joined or "enforcement-free" in joined
    # The reported band + z_min appear in the final analysis.
    assert "9.85" in joined and "9.95" in joined
    # Uses the stored node paths (paths_nodes) and reports the endpoint-z spread.
    assert "paths_nodes" in joined
    assert "zend" in joined


# --------------------------------------------------------------------------
# Laminate z-geometry -- reproduced from the committed drift generator
# (no store, no uncommitted drift code)
# --------------------------------------------------------------------------

def test_fr4_insulator_is_single_cell_node99(drift_geometry):
    insul, _ = drift_geometry
    zf = numpy.where(insul.any(axis=(0, 1)))[0]
    assert list(zf) == [FR4_NODE], list(zf)
    # cell [9.85, 9.95] mm, centre 9.90mm.
    assert (zf.max() - 0.5) * H == pytest.approx(9.85)
    assert (zf.max() + 0.5) * H == pytest.approx(9.95)


def test_pad_node_100_and_laminate_alignment(drift_geometry):
    insul, barr = drift_geometry
    zf = numpy.where(insul.any(axis=(0, 1)))[0]
    # Pad conductor: the first metal-bearing node above the FR4 cell.
    pad_nodes = [z for z in range(90, 110) if (barr[:, :, z] != 0).sum() > 100]
    assert PAD_NODE in pad_nodes
    assert min(pad_nodes) == PAD_NODE
    # FR4 top face == pad bottom face (laminated, aligned at 9.95mm).
    fr4_top = (zf.max() + 0.5) * H
    pad_bottom = (PAD_NODE - 0.5) * H
    assert fr4_top == pytest.approx(pad_bottom)
    assert fr4_top == pytest.approx(9.95)


def test_fr4_and_pad_are_adjacent_disjoint(drift_geometry):
    # FR4 node 99 and pad node 100 are adjacent and the insulator does not
    # overlap the pad node.
    insul, barr = drift_geometry
    assert not insul[:, :, PAD_NODE].any()
    assert PAD_NODE == FR4_NODE + 1


# --------------------------------------------------------------------------
# Endpoint-z distribution -- verified against the experimenter store when present
# --------------------------------------------------------------------------

def _load(store, key):
    d = numpy.load(store / f"{key}.npz")
    return d[d.files[0]]


@pytest.mark.skipif(not (STORE / "paths/drift3d_nodes.npz").exists(),
                    reason="experimenter store not present (untracked; absent in fresh checkout)")
def test_store_geometry_matches_generator():
    insul = _load(STORE, "initial/drift_full_insulator").astype(bool)
    barr = _load(STORE, "boundary/drift_full")
    zf = numpy.where(insul.any(axis=(0, 1)))[0]
    assert list(zf) == [FR4_NODE]
    pad_nodes = [z for z in range(90, 110) if (barr[:, :, z] > 0).sum() > 100]
    assert min(pad_nodes) == PAD_NODE


@pytest.mark.skipif(not (STORE / "paths/drift3d_nodes.npz").exists(),
                    reason="experimenter store not present (untracked; absent in fresh checkout)")
def test_all_node_paths_terminate_at_pad_plane():
    # Contract for the RE-DRIFTED store (mask-aware efield c8fb3eb + enforcement
    # -free drift, pochoir-vsdz): the halt-in-FR4-band the notebook narrates is
    # superseded; every path now terminates within the pad node cell
    # [9.95, 10.05] mm and none penetrate the FR4 (pochoir-0z6u).
    P = _load(STORE, "paths/drift3d_nodes")            # (Npaths, Nsteps, 3), mm
    assert P.ndim == 3 and P.shape[0] == 1764
    zend = P[:, -1, 2]
    zmin_all = float(P[:, :, 2].min())
    # Every endpoint lies within the pad node cell (bottom face 9.95mm, top
    # face 10.05mm); observed: 1757/1764 exactly at the 10.05mm top face, the
    # rest just above the 10.0mm pad node centre.
    assert zend.min() >= 9.95 - 1e-6
    assert zend.max() <= 10.05 + 1e-6
    # None stall in (or below) the old FR4 cell band [9.85, 9.95] mm.
    assert (zend <= 9.95 + 1e-6).sum() == 0
    # No path ever dips into the FR4: the global deepest z stays at/above the
    # pad cell bottom face.
    assert zmin_all >= 9.95 - 1e-6
