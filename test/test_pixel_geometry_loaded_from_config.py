#!/usr/bin/env pytest
"""Pixel geometry must come from JSON config, not __main__.py literals.

PLAN_for_production.md §1, third bullet.
"""

import json
import pytest

from pochoir.__main__ import _load_pixel_geometry


def _write_cfg(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def test_loads_from_single_config(tmp_path):
    path = _write_cfg(tmp_path, "ex.json", {
        "pixelSize": 3.8, "pixelGap": 0.6, "Npixels": 5,
    })
    geom = _load_pixel_geometry([path])
    assert geom == {
        "pixel_size": 3.8,
        "pixel_gap": 0.6,
        "pixel_pitch": 3.8 + 0.6,
        "npixels": 5,
    }


def test_pitch_derives_from_size_plus_gap(tmp_path):
    path = _write_cfg(tmp_path, "ex.json", {
        "pixelSize": 2.2, "pixelGap": 0.8, "Npixels": 3,
    })
    geom = _load_pixel_geometry([path])
    assert geom["pixel_pitch"] == pytest.approx(3.0)


def test_later_config_overrides_earlier(tmp_path):
    a = _write_cfg(tmp_path, "a.json", {
        "pixelSize": 3.8, "pixelGap": 0.6, "Npixels": 5,
    })
    b = _write_cfg(tmp_path, "b.json", {"Npixels": 9})
    geom = _load_pixel_geometry([a, b])
    assert geom["npixels"] == 9
    assert geom["pixel_size"] == 3.8


def test_missing_key_raises_clear_keyerror(tmp_path):
    path = _write_cfg(tmp_path, "bad.json", {
        "pixelSize": 3.8, "Npixels": 5,  # pixelGap missing
    })
    with pytest.raises(KeyError, match="pixelGap"):
        _load_pixel_geometry([path])


def test_no_config_raises_value_error():
    with pytest.raises(ValueError, match="--config"):
        _load_pixel_geometry([])


def test_real_pipeline_config_loads():
    """The JSON shipped with test-full-3d-pixel.sh must satisfy the contract."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    cfg = repo / "test" / "example_gen_pixel_with_grid.json"
    geom = _load_pixel_geometry([str(cfg)])
    assert geom["pixel_size"] > 0
    assert geom["pixel_gap"] > 0
    assert geom["npixels"] >= 1
    assert geom["pixel_pitch"] == pytest.approx(geom["pixel_size"] + geom["pixel_gap"])
