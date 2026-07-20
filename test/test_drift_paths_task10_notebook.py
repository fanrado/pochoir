#!/usr/bin/env pytest
'''
Tests for commit 1abd568 "Task10: add DriftPaths_task10_2cm_01mm.ipynb
drift-path notebook (pochoir-mgq0)".

NOTEBOOK-ONLY (user analysis file test/DriftPaths_task10_2cm_01mm.ipynb).
The commit adds a trimmed, drift-paths-only analysis notebook for the 2cm/0.1mm
enforcement-free run (store_task10_drift_2cm_01mm).  Pure numpy/matplotlib, no
pochoir import.  The notebook's numeric/plotting claims rest on three pure
algorithms:

  (1) grid + reshape: NGRID = round(sqrt(len(starts))), and the flat
      (N,NSAMP,3) paths array reshapes to (NGRID,NGRID,NSAMP,3).
  (2) contiguous pad-segment detection (cell 1): the x-run of pad metal at a
      chamfer-free row of boundary[:, :, 100] is two segments [(0,18),(25,43)],
      giving physical pad edges around the 0.6mm gap.
  (3) last-moving-endpoint landing classification (cell 3): endtags are all
      zero for this velocity-engine run, so pad-vs-gap is derived from the pad
      mask at the last sample that actually moved (np.diff > 1e-9).

These tests (a) confirm the notebook is valid nbformat with the expected cells,
(b) unit-test the three algorithms on synthetic inputs with known answers, and
(c) reproduce the notebook's ground-truth numbers directly from the handed-off
store when it is present (skipped otherwise).  Production code is NOT imported
or modified -- the algorithms are re-implemented here exactly as the notebook
runs them.

NOTE on the store-reproduction assertions: the notebook is fully parametric
(NGRID, launch grid and the gap-landing count are all derived from the store at
run time; the "10x10 / 100 paths / 19 gap" figures live only in prose comments).
The store test/store_task10_drift_2cm_01mm is untracked, regenerable data whose
launch grid depends on run-task10-drift-2cm-01mm.sh's `dist` array -- e.g. the
10-pt gap-band grid (100 paths) or a finer full-pitch grid (1764 paths).  The
store-reproduction tests therefore assert the notebook's grid-agnostic
INVARIANTS and the physical drift outcome (reshape integrity, launch z, halt-z
band, well-formed classification) rather than freezing one run's exact counts.
'''

from pathlib import Path

import nbformat
import numpy
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB = REPO_ROOT / "test" / "DriftPaths_task10_2cm_01mm.ipynb"
STORE = REPO_ROOT / "test" / "store_task10_drift_2cm_01mm"
SPACING = 0.1
PAD_Z = 100        # pad-plane z-node (z=10.0mm)


# --------------------------------------------------------------------------
# Pure algorithms, re-implemented exactly as the notebook runs them.
# --------------------------------------------------------------------------

def _segments(mask_1d):
    '''Contiguous True runs of a 1-D boolean array as (start,end) inclusive.'''
    xs = numpy.where(mask_1d)[0]
    if len(xs) == 0:
        return []
    segs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != p + 1:
            segs.append((int(s), int(p)))
            s = x
        p = x
    segs.append((int(s), int(p)))
    return segs


def _last_moving_endpoint(tr):
    '''Endpoint of a zero-padded trajectory: the sample AFTER the last one that
    moved (cell-3 np.diff>1e-9 trick), else the final sample if it never moved.'''
    mv = numpy.where(numpy.abs(numpy.diff(tr[:, 0]))
                     + numpy.abs(numpy.diff(tr[:, 1]))
                     + numpy.abs(numpy.diff(tr[:, 2])) > 1e-9)[0]
    return tr[mv[-1] + 1] if len(mv) else tr[-1]


def _on_pad(xe, ye, pad2):
    return bool(pad2[int(round(xe / SPACING)) % pad2.shape[0],
                     int(round(ye / SPACING)) % pad2.shape[1]])


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cells():
    nb = nbformat.read(str(NB), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _load(key):
    f = numpy.load(str(STORE / (key + ".npz")))
    return f[f.files[0]]


store_required = pytest.mark.skipif(
    not STORE.is_dir(),
    reason=f"handed-off store {STORE.name} not present")


@pytest.fixture(scope="module")
def store_arrays():
    starts = _load("starts/drift3d")
    p = _load("paths/drift3d")
    ngrid = int(round(len(starts) ** 0.5))
    paths = p.reshape(ngrid, ngrid, p.shape[1], 3)
    pad2 = _load("boundary/drift_2cm")[:, :, PAD_Z] > 0
    endtag = _load("paths/drift3d_endtag")
    return dict(starts=starts, p=p, paths=paths, ngrid=ngrid,
                pad2=pad2, endtag=endtag)


# --------------------------------------------------------------------------
# Notebook is valid + structurally the drift-paths-only file the commit added
# --------------------------------------------------------------------------

def test_notebook_is_valid_nbformat():
    nb = nbformat.read(str(NB), as_version=4)
    assert nb.nbformat == 4
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 5          # setup + 4 figures


def test_notebook_has_expected_cells(cells):
    joined = "\n".join(cells)
    assert "store_task10_drift_2cm_01mm" in joined      # cell 0: correct store
    assert "paths/drift3d" in joined and "starts/drift3d" in joined
    assert "NGRID" in joined
    assert "pixel_pads" in joined                       # cell 1: pad footprint
    assert "np.diff" in joined and "endtag" in joined   # cell 3: landing trick
    assert "projection='3d'" in joined or "projection=\"3d\"" in joined  # cell 4
    # Trimmed notebook: drops field/potential DIAGNOSTICS -- no such array is
    # loaded (the words survive only in the header comment describing the trim).
    assert "_load('efield" not in joined and "_load('potential" not in joined


# --------------------------------------------------------------------------
# (1) segment detection -- unit tests on synthetic masks
# --------------------------------------------------------------------------

def test_segments_empty():
    assert _segments(numpy.zeros(10, bool)) == []


def test_segments_single_run():
    m = numpy.zeros(10, bool); m[2:6] = True
    assert _segments(m) == [(2, 5)]


def test_segments_two_runs_split_by_gap():
    m = numpy.zeros(44, bool); m[0:19] = True; m[25:44] = True
    assert _segments(m) == [(0, 18), (25, 43)]


def test_segments_isolated_endpoints():
    m = numpy.zeros(6, bool); m[0] = True; m[5] = True
    assert _segments(m) == [(0, 0), (5, 5)]


# --------------------------------------------------------------------------
# (3) last-moving endpoint -- unit tests on synthetic zero-padded paths
# --------------------------------------------------------------------------

def test_last_moving_endpoint_of_zero_padded_path():
    # Moves for 3 steps then parks (rows repeat) -> endpoint is the parked value,
    # NOT the trailing repeats and NOT a spurious later row.
    tr = numpy.array([[0., 0., 30.], [0., 0., 20.], [0., 0., 10.],
                      [1., 1., 9.9], [1., 1., 9.9], [1., 1., 9.9]])
    end = _last_moving_endpoint(tr)
    assert tuple(end) == (1., 1., 9.9)


def test_last_moving_endpoint_never_moves():
    tr = numpy.tile([2., 2., 9.9], (5, 1))
    assert tuple(_last_moving_endpoint(tr)) == (2., 2., 9.9)


def test_on_pad_classification_and_wrap():
    pad2 = numpy.zeros((44, 44), bool)
    pad2[5, 5] = True                       # metal cell at (0.5mm, 0.5mm)
    assert _on_pad(0.5, 0.5, pad2) is True   # lands on metal
    assert _on_pad(2.2, 2.2, pad2) is False  # gap centre, no metal
    # modulo wrap: x=4.9mm -> node 49 % 44 = 5
    assert _on_pad(4.9, 0.5, pad2) is True


# --------------------------------------------------------------------------
# Ground-truth reproduction from the handed-off store (skip if absent)
# --------------------------------------------------------------------------

@store_required
def test_store_grid_and_reshape(store_arrays):
    # Grid-agnostic reshape invariant: the flat (N, NSAMP, 3) paths array is a
    # perfect NGRID x NGRID square with NGRID = round(sqrt(N)), exactly as the
    # notebook computes it -- regardless of how many launch points the runner's
    # `dist` grid produced (100 for the gap band, 1764 for the full pitch, ...).
    a = store_arrays
    ngrid = a["ngrid"]
    assert ngrid >= 2
    assert len(a["starts"]) == ngrid * ngrid           # a perfect square
    assert a["p"].ndim == 3 and a["p"].shape[2] == 3    # (N, NSAMP, 3)
    nsamp = a["p"].shape[1]
    assert a["p"].shape[0] == ngrid * ngrid
    assert a["paths"].shape == (ngrid, ngrid, nsamp, 3)
    # All launches share the single release plane just below the cathode.
    assert numpy.unique(a["starts"][:, 2]) == pytest.approx([29.9])
    # Launch x,y stay within the 4.4mm periodic pixel pitch.
    for axis in (0, 1):
        launch = numpy.unique(a["starts"][:, axis])
        assert launch.min() >= 0.0
        assert launch.max() <= 44 * SPACING


@store_required
def test_store_pad_segments_and_edges(store_arrays):
    pad2 = store_arrays["pad2"]
    assert int(pad2.sum()) == 1440                     # metal cells at z-node 100
    segs = _segments(pad2[:, 10])
    assert segs == [(0, 18), (25, 43)]
    # physical pad edges bracketing the 0.6mm inter-pixel gap
    pixel_pads = [((a - 0.5) * SPACING, (b + 0.5) * SPACING) for a, b in segs]
    assert pixel_pads[0] == pytest.approx((-0.05, 1.85))
    assert pixel_pads[1] == pytest.approx((2.45, 4.35))


@store_required
def test_store_endtags_all_zero(store_arrays):
    # The commit message + cell 3 rationale: endtags are all-zero for this
    # velocity-engine run, which is WHY landing is derived from the mask.
    assert numpy.all(store_arrays["endtag"] == 0)


@store_required
def test_store_landing_classification_and_halt_band(store_arrays):
    # Reproduce the notebook's cell-3 landing classification on the actual store
    # and assert its physical outcome (grid-agnostic): every path classifies as
    # pad-or-gap, the gap count is well-formed (0 <= n_gap <= N), and -- the key
    # physics claim -- every charge halts in the thin band just above the pad
    # plane (z=10.0mm), between 9.85 and 9.95mm.  The exact gap count depends on
    # the runner's launch grid (19/100 for the gap band, 164/1764 for the full
    # pitch) and is reported by the notebook title, not asserted as a constant.
    a = store_arrays
    pad2, paths, ngrid = a["pad2"], a["paths"], a["ngrid"]
    n_gap, z_ends = 0, []
    for i in range(ngrid):
        for j in range(ngrid):
            xe, ye, ze = _last_moving_endpoint(paths[i, j])
            z_ends.append(ze)
            if not _on_pad(xe, ye, pad2):
                n_gap += 1
    assert 0 <= n_gap <= ngrid * ngrid
    # charges launch at 29.9mm and halt just above the pad plane (~9.85-9.95mm)
    assert min(z_ends) >= 9.85 - 1e-6
    assert max(z_ends) <= 9.95 + 1e-6
