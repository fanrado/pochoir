#!/usr/bin/env pytest
"""Shape and offset checks for ``_shift_paths_pixel_grid`` (PLAN §3).

The ``pad_center`` tests cover commit d74492f (pochoir-lvp9): the base
collecting pixel is aligned to the pinned W=1 pad instead of the pitch
formula, so on-metal endpoints sample the exact W=1 cells.
"""

import numpy
import pytest

from pochoir.__main__ import _shift_paths_pixel_grid


def _make_paths(npaths, nsteps=3):
    """``npaths*npaths`` paths whose points are all the origin."""
    base = [[0.0, 0.0, float(k)] for k in range(nsteps)]
    return [list(base) for _ in range(npaths * npaths)]


@pytest.mark.parametrize("npixels", [3, 5, 9])
def test_output_count(npixels):
    npaths = 10
    paths = _make_paths(npaths)
    out = _shift_paths_pixel_grid(
        paths, npaths=npaths, npixels=npixels,
        pixel_pitch=4.4, pixel_gap=0.6, pixel_size=3.8,
    )
    npix = npixels // 2
    nedge = npaths // 2
    expected = (
        npix * npaths * (npix * npaths + nedge)
        + nedge * (npix * npaths + nedge)
    )
    assert len(out) == expected


def test_z_unchanged():
    paths = _make_paths(npaths=10, nsteps=4)
    out = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=4.4, pixel_gap=0.6, pixel_size=3.8,
    )
    for path in out:
        zs = [pt[2] for pt in path]
        assert zs == [0.0, 1.0, 2.0, 3.0]


def test_first_path_lands_at_central_pixel_centre():
    """With origin-only input, path[0] of the output should sit at the
    geometric centre of the first pixel column: (npix*pitch + gap/2 + size/2)."""
    npix = 2  # npixels=5 → npix=2
    pitch, gap, size = 4.4, 0.6, 3.8
    expected = npix * pitch + gap / 2 + size / 2
    paths = _make_paths(npaths=10)
    out = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=pitch, pixel_gap=gap, pixel_size=size,
    )
    assert out[0][0][0] == pytest.approx(expected)
    assert out[0][0][1] == pytest.approx(expected)


def test_row_offset_advances_by_pitch():
    """Within a single (ix_pix=0, lvl=0) block, successive iy_pix groups should
    differ in y by exactly pixel_pitch."""
    npaths = 10
    pitch = 4.4
    paths = _make_paths(npaths=npaths)
    out = _shift_paths_pixel_grid(
        paths, npaths=npaths, npixels=5,
        pixel_pitch=pitch, pixel_gap=0.6, pixel_size=3.8,
    )
    y0 = out[0][0][1]
    y1 = out[npaths][0][1]  # next iy_pix group, same lvl
    assert y1 - y0 == pytest.approx(pitch)


# --- pad_center alignment (commit d74492f / pochoir-lvp9) ---------------------


def test_pad_center_none_matches_pitch_formula():
    """pad_center=None must reproduce the legacy pitch-formula base center
    exactly (backward-compatible default)."""
    npix = 2  # npixels=5
    pitch, gap, size = 4.4, 0.6, 3.8
    expected = npix * pitch + gap / 2 + size / 2
    paths = _make_paths(npaths=10)
    out = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=pitch, pixel_gap=gap, pixel_size=size,
        pad_center=None,
    )
    assert out[0][0][0] == pytest.approx(expected)
    assert out[0][0][1] == pytest.approx(expected)


def test_pad_center_overrides_base_center():
    """When pad_center is given, the base pixel sits exactly there instead of
    at the pitch-formula center (the sub-cell fix: 10.9 not 11.0)."""
    paths = _make_paths(npaths=10)
    out = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=4.4, pixel_gap=0.6, pixel_size=3.8,
        pad_center=(10.9, 10.9),
    )
    assert out[0][0][0] == pytest.approx(10.9)
    assert out[0][0][1] == pytest.approx(10.9)
    # and it differs from the legacy 11.0 center
    assert out[0][0][0] != pytest.approx(11.0)


def test_pad_center_preserves_count_and_pitch_stepping():
    """pad_center only shifts the base origin; the tiling count and the
    per-pixel pitch stepping are unchanged from the legacy path."""
    npaths = 10
    pitch = 4.4
    paths = _make_paths(npaths=npaths)
    legacy = _shift_paths_pixel_grid(
        paths, npaths=npaths, npixels=5,
        pixel_pitch=pitch, pixel_gap=0.6, pixel_size=3.8,
    )
    aligned = _shift_paths_pixel_grid(
        paths, npaths=npaths, npixels=5,
        pixel_pitch=pitch, pixel_gap=0.6, pixel_size=3.8,
        pad_center=(10.9, 10.9),
    )
    assert len(aligned) == len(legacy)
    # next iy_pix group still advances by exactly one pitch
    assert aligned[npaths][0][1] - aligned[0][0][1] == pytest.approx(pitch)


def test_pad_center_shifts_whole_grid_rigidly():
    """A pad_center displaced by delta from the legacy center shifts every
    output point by the same delta (rigid translation, z untouched)."""
    npix = 2
    pitch, gap, size = 4.4, 0.6, 3.8
    legacy_center = npix * pitch + gap / 2 + size / 2
    delta = -0.1  # 11.0 -> 10.9
    paths = _make_paths(npaths=10, nsteps=4)
    legacy = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=pitch, pixel_gap=gap, pixel_size=size,
    )
    aligned = _shift_paths_pixel_grid(
        paths, npaths=10, npixels=5,
        pixel_pitch=pitch, pixel_gap=gap, pixel_size=size,
        pad_center=(legacy_center + delta, legacy_center + delta),
    )
    for pl, pa in zip(legacy, aligned):
        for ptl, pta in zip(pl, pa):
            assert pta[0] == pytest.approx(ptl[0] + delta)
            assert pta[1] == pytest.approx(ptl[1] + delta)
            assert pta[2] == pytest.approx(ptl[2])  # z unchanged


# --- pinned-W=1 pad-center derivation contract (induce_pixel inline code) -----
#
# induce_pixel derives pad_center from the solved weighting potential:
#     _on = numpy.isclose(wpot, 1.0).any(axis=2)
#     _ix, _iy = numpy.where(_on)
#     pad_center = (0.5*(xs[_ix.min()] + xs[_ix.max()]),
#                   0.5*(ys[_iy.min()] + ys[_iy.max()]))
# These tests pin that formula's contract against a synthetic wpot + domain
# axes.  (The code lives inline in the click command, so it is exercised here
# via the identical expression rather than a direct import.)


def _derive_pad_center(wpot, xs, ys):
    _on = numpy.isclose(wpot, 1.0).any(axis=2)
    _ix, _iy = numpy.where(_on)
    if not _ix.size:
        return None
    return (0.5 * (xs[_ix.min()] + xs[_ix.max()]),
            0.5 * (ys[_iy.min()] + ys[_iy.max()]))


def test_derive_pad_center_from_pinned_mask():
    """A W=1 block at index [100,118] centered in a 220-cell 0..21.9mm axis
    resolves to the block's geometric center (10.9mm), not the 11.0mm formula."""
    wpot = numpy.zeros((220, 220, 10))
    wpot[100:119, 100:119, 3:6] = 1.0
    xs = numpy.linspace(0.0, 21.9, 220)
    ys = numpy.linspace(0.0, 21.9, 220)
    cx, cy = _derive_pad_center(wpot, xs, ys)
    assert cx == pytest.approx(0.5 * (xs[100] + xs[118]))
    assert cy == pytest.approx(0.5 * (ys[100] + ys[118]))
    assert cx == pytest.approx(10.9, abs=1e-6)
    assert cy == pytest.approx(10.9, abs=1e-6)


def test_derive_pad_center_ignores_sub_unity_and_z_extent():
    """Only cells exactly at W=1 count; sub-unity values and the z-extent of
    the pinned block do not move the derived (x, y) center."""
    wpot = numpy.zeros((40, 40, 8))
    wpot[10:15, 20:25, 0:2] = 1.0      # pinned pad
    wpot[0:5, 0:5, :] = 0.97           # near-unity but not pinned -> ignored
    xs = numpy.linspace(0.0, 3.9, 40)
    ys = numpy.linspace(0.0, 3.9, 40)
    cx, cy = _derive_pad_center(wpot, xs, ys)
    assert cx == pytest.approx(0.5 * (xs[10] + xs[14]))
    assert cy == pytest.approx(0.5 * (ys[20] + ys[24]))


def test_derive_pad_center_none_when_no_pinned_cell():
    """No W=1 cell anywhere -> None, so induce_pixel falls back to the pitch
    formula (pad_center stays None)."""
    wpot = numpy.full((20, 20, 4), 0.5)
    xs = numpy.linspace(0.0, 1.9, 20)
    ys = numpy.linspace(0.0, 1.9, 20)
    assert _derive_pad_center(wpot, xs, ys) is None
