"""Tests for JSON-driven make_pixel_start_points parameter loading."""

import json
import pytest

from pochoir.__main__ import (
    _load_start_point_config,
    _START_POINT_DEFAULTS,
    make_pixel_start_points,
)


# ---------------------------------------------------------------------------
# _load_start_point_config tests
# ---------------------------------------------------------------------------


def test_all_keys_present(tmp_path):
    """All expected keys in config produce correct derived values."""
    cfg = {
        "driftZDepth": 35.0,
        "nGridPoints": 5,
        "pixelSize": 3.6,
        "pixelGap": 0.8,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    result = _load_start_point_config([str(p)])

    expected_pitch = 3.6 + 0.8  # 4.4
    expected_spacing = expected_pitch / 5  # 0.88

    assert result["z_depth"] == pytest.approx(35.0)
    assert result["ngridpoints"] == 5
    assert result["pitch"] == pytest.approx(expected_pitch)
    assert result["spacing"] == pytest.approx(expected_spacing)


def test_empty_config_paths_uses_defaults():
    """Empty config list falls back to _START_POINT_DEFAULTS; pitch and spacing are None."""
    result = _load_start_point_config([])

    assert result["z_depth"] == pytest.approx(_START_POINT_DEFAULTS["driftZDepth"])
    assert result["ngridpoints"] == _START_POINT_DEFAULTS["nGridPoints"]
    assert result["pitch"] is None
    assert result["spacing"] is None


def test_only_drift_keys_no_pixel_geometry(tmp_path):
    """Config with only driftZDepth/nGridPoints gives pitch=None, spacing=None."""
    cfg = {"driftZDepth": 50.0, "nGridPoints": 8}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    result = _load_start_point_config([str(p)])

    assert result["z_depth"] == pytest.approx(50.0)
    assert result["ngridpoints"] == 8
    assert result["pitch"] is None
    assert result["spacing"] is None


def test_explicit_grid_spacing_overrides_derived(tmp_path):
    """Explicit gridSpacing takes precedence over (pixelSize+pixelGap)/nGridPoints."""
    cfg = {
        "driftZDepth": 28.0,
        "nGridPoints": 10,
        "pixelSize": 3.6,
        "pixelGap": 0.8,
        "gridSpacing": 0.55,  # explicit override
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    result = _load_start_point_config([str(p)])

    assert result["spacing"] == pytest.approx(0.55)
    # pitch is still derived from pixelSize+pixelGap
    assert result["pitch"] == pytest.approx(4.4)


def test_two_config_files_later_wins(tmp_path):
    """Second config file overrides keys set by the first."""
    cfg1 = {"driftZDepth": 28.0, "nGridPoints": 10, "pixelSize": 3.6, "pixelGap": 0.8}
    cfg2 = {"driftZDepth": 42.0, "nGridPoints": 6}

    p1 = tmp_path / "cfg1.json"
    p2 = tmp_path / "cfg2.json"
    p1.write_text(json.dumps(cfg1))
    p2.write_text(json.dumps(cfg2))

    result = _load_start_point_config([str(p1), str(p2)])

    # cfg2 overrides driftZDepth and nGridPoints
    assert result["z_depth"] == pytest.approx(42.0)
    assert result["ngridpoints"] == 6
    # pixelSize/pixelGap came from cfg1 and survive the merge
    assert result["pitch"] == pytest.approx(3.6 + 0.8)
    assert result["spacing"] == pytest.approx((3.6 + 0.8) / 6)


# ---------------------------------------------------------------------------
# make_pixel_start_points integration tests
# ---------------------------------------------------------------------------


def test_make_pixel_start_points_count_and_z(tmp_path):
    """Result of _load_start_point_config fed to make_pixel_start_points produces
    ngridpoints**2 points all sharing the configured z_depth."""
    cfg = {
        "driftZDepth": 35.0,
        "nGridPoints": 4,
        "pixelSize": 3.6,
        "pixelGap": 0.8,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    params = _load_start_point_config([str(p)])
    points = make_pixel_start_points(**params)

    assert len(points) == 4 ** 2  # ngridpoints**2

    z_values = [pt[2] for pt in points]
    assert all(z == pytest.approx(35.0) for z in z_values)


def test_make_pixel_start_points_xy_positions(tmp_path):
    """x/y coordinates are cell-centred: first at spacing/2, last at pitch-spacing/2."""
    cfg = {
        "driftZDepth": 28.0,
        "nGridPoints": 4,
        "pixelSize": 3.6,
        "pixelGap": 0.8,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    params = _load_start_point_config([str(p)])
    points = make_pixel_start_points(**params)

    pitch = 4.4
    spacing = pitch / 4  # 1.1

    xs = sorted({pt[0] for pt in points})
    ys = sorted({pt[1] for pt in points})

    expected = [spacing / 2 + i * spacing for i in range(4)]
    assert xs == pytest.approx(expected)
    assert ys == pytest.approx(expected)
