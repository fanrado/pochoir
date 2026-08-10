#!/usr/bin/env pytest
'''
Tests for commit 348637b: test/run-task13-hybrid.sh passes the band options and
its header describes the Schwarz sweep.

The flags are the only executable change, so what must hold is:

  * BOTH hybrid-iterate calls (PART A drift, PART C weighting) pass
    --band-cells 2 --max-sweeps 1, and both still pass --interface 40*mm;
  * PART B keeps --interp-order linear (the seam is C0 but not known C1);
  * the header no longer sells the one-shot method, and states the band, the
    inner-node-only Dirichlet rule, the ~2x cost and --max-sweeps 0 as the way
    back;
  * the header's grid table matches what the driver actually derives from the
    configs today -- the numbers it replaced were stale, and a table nobody
    checks goes stale again.
'''

import json
import re
from pathlib import Path

import pytest

import pochoir.hybrid_iterate as hi

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "test" / "run-task13-hybrid.sh"
DRIFT_FINE = REPO_ROOT / "test" / "example_gen_pcb_drift_pixel_task13_fine.json"
WEIGHT_FINE = REPO_ROOT / "test" / "example_gen_pixel_with_grid_task13_fine.json"

TEXT = RUNNER.read_text()
INTERFACE_MM = 40.0


def _calls():
    '''The two `pochoir hybrid-iterate` invocations, backslash continuations
    joined.'''
    flat = TEXT.replace('\\\n', ' ')
    return [line for line in flat.splitlines() if 'hybrid-iterate' in line
            and line.strip().startswith('pochoir')]


def _header():
    '''The leading comment block.'''
    lines = []
    for line in TEXT.splitlines()[1:]:
        if line.startswith('#'):
            lines.append(line)
        elif lines and line.strip():
            break
    return '\n'.join(lines)


def _derived(field, cfg_path):
    prof = hi._profile(field)
    cfg = json.loads(cfg_path.read_text())
    grids = hi._derive_grids(prof, cfg, hi.DEFAULT_SPACINGS, INTERFACE_MM)
    return {key.split('/')[-1]: shape for key, shape, _ in grids}


# --------------------------------------------------------------------------
# The executable change: the flags
# --------------------------------------------------------------------------

def test_there_are_exactly_two_hybrid_iterate_calls():
    assert len(_calls()) == 2, _calls()


@pytest.mark.parametrize("flag", ["--band-cells 2", "--max-sweeps 1"])
def test_both_calls_pass_the_band_flags(flag):
    for call in _calls():
        assert re.search(flag.replace(' ', r'\s+'), call), (flag, call)


def test_both_calls_still_pass_the_same_interface_and_spacings():
    for call in _calls():
        assert '--interface "40*mm"' in call or "--interface 40*mm" in call
        assert '--coarse-spacing 0.4' in call
        assert '--fine-spacing 0.1' in call


def test_the_two_calls_are_drift_and_weighting():
    calls = _calls()
    assert sum('--field weighting' in c for c in calls) == 1
    # the drift call takes the default --field, so it names no field
    assert sum('--field' not in c for c in calls) == 1


def test_drift_chain_still_uses_linear_interpolation():
    '''C0 is not C1 -- cubic must stay off until the seam jump is measured.'''
    assert '--interp-order linear' in TEXT
    assert '--interp-order cubic' not in TEXT


def test_no_precision_or_sweep_tolerance_override_was_added():
    for call in _calls():
        assert '--precision' not in call
        assert '--schwarz-tol' not in call


# --------------------------------------------------------------------------
# The header
# --------------------------------------------------------------------------

def test_header_describes_the_sweep_not_the_one_shot_method():
    head = _header()
    assert 'BANDED SCHWARZ' in head
    assert 'coarse, near,\n# far, near' in head or 'coarse, near' in head
    for stale in ('ONE-SHOT hybrid solver',
                  'METHOD (single pass, no iteration',
                  'There is NO outer iteration any more'):
        assert stale not in head, stale


def test_header_states_the_band_and_the_inner_node_rule():
    head = _header()
    for needle in ('40.0 mm', '39.6 mm', '39.2 mm', 'INNER node', 'FREE'):
        assert needle in head, needle
    # ...and why the band needs no reshaping.
    assert 'no grid reshaping' in head


def test_header_states_the_cost_and_the_way_back():
    head = _header()
    assert '2x' in head
    assert '--max-sweeps 0' in head


def test_header_keeps_the_seam_caveat():
    head = _header()
    assert 'C0 by construction' in head
    assert 'NOT known to be C1' in head
    assert '--interp-order linear' in head


def test_header_separation_plane_agrees_with_the_flag():
    '''The old header said 20 mm while both calls pass --interface 40*mm.'''
    head = _header()
    assert 'z = 40 mm' in head
    assert 'separation at z = 20 mm' not in head


# --------------------------------------------------------------------------
# The grid table must match what the driver derives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field, cfg, leaves", [
    ('drift', DRIFT_FINE, ('coarse', 'near', 'drift3d')),
    ('weighting', WEIGHT_FINE, ('w_coarse', 'w_near', 'weight3d')),
])
def test_header_grid_table_matches_the_derived_shapes(field, cfg, leaves):
    head = _header()
    derived = _derived(field, cfg)
    for leaf in leaves:
        nx, ny, nz = derived[leaf].split(',')
        pretty = f'{nx} x {ny} x {nz}'
        assert re.search(re.escape(pretty).replace(r'\ ', r'\s*'), head), \
            (leaf, pretty)


def test_header_z_extent_matches_the_derived_full_depth():
    head = _header()
    nz = int(_derived('drift', DRIFT_FINE)['drift3d'].split(',')[2])
    depth_mm = (nz - 1) * hi.DEFAULT_SPACINGS['fine']
    assert f'z = 0..{depth_mm:.0f} mm' in head
    assert 'z = 0..60 mm' not in head        # the stale value


def test_header_records_the_17_pixel_weighting_probe():
    head = _header()
    cfg = json.loads(WEIGHT_FINE.read_text())
    npix = cfg.get('Npixels') or cfg.get('npixels')
    assert npix is not None
    assert str(npix) in head
