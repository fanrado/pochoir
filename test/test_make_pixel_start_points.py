#!/usr/bin/env pytest
"""Geometry correctness for ``make_pixel_start_points`` (PLAN §3).

Covers 1x1, 5x5 and 9x9 grids and checks: count, fixed z-depth, cell-centred
offset, uniform spacing, and absence of duplicates.
"""

import math

import pytest

from pochoir.__main__ import make_pixel_start_points


@pytest.mark.parametrize("n", [1, 5, 9])
def test_count_and_z_fixed(n):
    pts = make_pixel_start_points(z_depth=148.0, ngridpoints=n, pitch=4.4)
    assert len(pts) == n * n
    assert {p[2] for p in pts} == {148.0}


@pytest.mark.parametrize("n", [1, 5, 9])
def test_cell_centred_grid(n):
    pitch = 4.4
    pts = make_pixel_start_points(z_depth=0.0, ngridpoints=n, pitch=pitch)
    spacing = pitch / n
    half = spacing / 2.0

    xs = sorted({p[0] for p in pts})
    ys = sorted({p[1] for p in pts})
    assert len(xs) == n and len(ys) == n
    expected = [half + i * spacing for i in range(n)]
    for got, exp in zip(xs, expected):
        assert math.isclose(got, exp, rel_tol=0, abs_tol=1e-12)
    for got, exp in zip(ys, expected):
        assert math.isclose(got, exp, rel_tol=0, abs_tol=1e-12)


def test_no_duplicates():
    pts = make_pixel_start_points(z_depth=1.0, ngridpoints=9, pitch=4.4)
    assert len({tuple(p) for p in pts}) == len(pts)


def test_explicit_spacing_overrides_pitch():
    pts = make_pixel_start_points(z_depth=0.0, ngridpoints=4, pitch=4.4, spacing=1.0)
    xs = sorted({p[0] for p in pts})
    assert xs == [0.5, 1.5, 2.5, 3.5]
