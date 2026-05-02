#!/usr/bin/env pytest
"""Initial-condition arrays from gen_pcb_pixel_with_grid.generator.

PLAN_for_production.md §1, fifth bullet: the commented IDW seeding was
deleted (the referenced module was never committed), so the chosen init
is the zero-init produced by draw_pixel_plane alone. This test pins that
choice: ``arr`` returned by ``generator`` is all zeros except where the
central pixel of the pixel plane is set to 1.
"""

import os
import numpy
import pytest

from pochoir.domain import Domain
from pochoir import gen_pcb_pixel_with_grid as g


def _cfg(n_pix=3):
    # p_size=8, p_gap=2 in voxel units → trimCorner only carves the corners
    # of the central pixel, leaving a non-empty interior we can assert on.
    return {
        "HoleRadius": 0.5,
        "PcbWidth": 1.0,
        "GridHoleShape": "square",   # avoids the 'circular' branch's epsilon plotting
        "pixelSize": 8.0,
        "pixelGap": 2.0,
        "Npixels": n_pix,
        "pixelPlaneWidth": 1.0,
        "pixelPlaneLowEdgePosition": 2.0,
        "LArPermittivity": 1.5,
        "FR4Permittivity": 4.5,
    }


def test_generator_initial_arr_is_zero_except_central_pixel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.mkdir("store")

    n_pix = 3
    cfg = _cfg(n_pix=n_pix)
    # spacing 1 mm → pixel cell = 10 voxels, plane = 30x30, z = 20
    shape = (30, 30, 20)
    dom = Domain(shape=shape, spacing=1.0)

    arr, barr, epsilon = g.generator(dom, cfg)

    assert arr.shape == shape
    assert barr.shape == shape
    assert epsilon is None  # square branch does not build epsilon

    # arr is filled only inside the central pixel's footprint, on the
    # pixel-plane z-slab. Everywhere else must remain at the zero init.
    p_size = int(round(cfg["pixelSize"] / dom.spacing[0]))
    p_gap = int(round(cfg["pixelGap"] / dom.spacing[0]))
    pp_low = int(cfg["pixelPlaneLowEdgePosition"] / dom.spacing[0])
    pp_w = int(cfg["pixelPlaneWidth"] / dom.spacing[0])
    c = n_pix // 2
    x0 = p_gap // 2 + c * (p_size + p_gap)

    central = arr[x0:x0 + p_size, x0:x0 + p_size, pp_low:pp_low + pp_w + 1]
    # trimCorner carves the edges, but the interior of the central pixel is set to 1
    assert central.max() == 1
    assert central.sum() > 0

    # Outside the central pixel footprint everything stays at the zero init —
    # this is the contract the deleted IDW seeding would have changed.
    mask = numpy.ones_like(arr, dtype=bool)
    mask[x0:x0 + p_size, x0:x0 + p_size, pp_low:pp_low + pp_w + 1] = False
    assert numpy.all(arr[mask] == 0)


def test_no_idw_module_dependency():
    """The deleted IDW init referenced a module that was never committed.
    Importing the generator module must not pull torch or that module in."""
    import importlib
    import sys

    sys.modules.pop("pochoir.gen_pcb_pixel_with_grid", None)
    mod = importlib.import_module("pochoir.gen_pcb_pixel_with_grid")
    assert not hasattr(mod, "init_idw_pcb_pixel")
    assert not hasattr(mod, "torch")
