#!/usr/bin/env pytest
"""Shape and offset checks for ``_shift_paths_pixel_grid`` (PLAN §3)."""

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
