#!/usr/bin/env pytest
'''
Tests for test/run-task13-hybrid.sh: the band options and the runner's shape
(commits 348637b, d58ab1a).

d58ab1a reduced the runner to a SIZES block plus a `hybrid_field` helper, so
each field is one line and --band-cells/--max-sweeps are gone -- they were
exactly the CLI defaults, which come from the driver constants since 1aa122e.
That is a refactor, so the load-bearing test is EQUIVALENCE: the options the
old and new runners resolve to must be identical, per field.  The rest pins
the runner's shape and the header prose:

  * one `pochoir hybrid-iterate` invocation, inside the helper; both fields
    go through it, and the SIZES block is the only place a size is written;
  * the band settings still arrive as 2 / 1 / '1*V' via the CLI defaults;
  * PART B keeps --interp-order linear (the seam is C0 but not known C1);
  * the header does not CONTRADICT the code -- it is deliberately NOT asked
    to contain the method description any more.
'''

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import pochoir.hybrid_iterate as hi
from pochoir.__main__ import hybrid_iterate as hi_cmd

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDIR = REPO_ROOT / "test"
RUNNER = TESTDIR / "run-task13-hybrid.sh"
DRIFT_FINE = TESTDIR / "example_gen_pcb_drift_pixel_task13_fine.json"
WEIGHT_FINE = TESTDIR / "example_gen_pixel_with_grid_task13_fine.json"

TEXT = RUNNER.read_text()
INTERFACE_MM = 40.0

# The revision this runner was refactored from; its emitted command lines are
# the equivalence baseline.
BASELINE_REV = "d58ab1a^"

WANT_TARGETS = ("velocity/drift3d", "starts/drift3d", "paths/drift3d",
                "current/induced_current")


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
# Equivalence: run both revisions with a stubbed `pochoir` and compare the
# options each hybrid-iterate line resolves to.
# --------------------------------------------------------------------------

def _emitted_hybrid_lines(tmp_path, script_text):
    '''Run `script_text` with a recording `pochoir` stub and return its
    hybrid-iterate command lines (arguments only).'''
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "pochoir"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\n')
    stub.chmod(0o755)

    work = tmp_path / f"work{abs(hash(script_text)) % 10**8}"
    work.mkdir()
    (work / "helpers.sh").write_text((TESTDIR / "helpers.sh").read_text())
    runner = work / "run.sh"
    runner.write_text(script_text)
    runner.chmod(0o755)
    # The configs are read by the driver, which is stubbed out here, but the
    # runner cd's to its own directory, so give it the real ones.
    for cfg in TESTDIR.glob("example_gen_p*task13*.json"):
        (work / cfg.name).write_text(cfg.read_text())

    # Pre-create every `want` target so the drift chain short-circuits to
    # "have" instead of exiting on the stub's missing output.
    store = work / "store"
    for target in WANT_TARGETS:
        path = store / (target + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    capture = work / "capture.txt"
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["CAPTURE"] = str(capture)
    subprocess.run(["bash", str(runner), "store"], cwd=str(work), env=env,
                   capture_output=True, text=True)

    if not capture.exists():
        return []
    return [line[len("hybrid-iterate"):].strip()
            for line in capture.read_text().splitlines()
            if line.startswith("hybrid-iterate")]


def _resolved(argline):
    '''Resolve one hybrid-iterate argument line through the real click command
    (config paths exist relative to test/, so resolve from there).'''
    cwd = os.getcwd()
    os.chdir(TESTDIR)
    try:
        ctx = hi_cmd.make_context("hybrid-iterate", shlex.split(argline))
        return dict(ctx.params)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runner")
    old = subprocess.run(["git", "show", f"{BASELINE_REV}:test/run-task13-hybrid.sh"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True)
    if old.returncode != 0:
        pytest.skip(f"{BASELINE_REV} not reachable in this checkout")
    return (_emitted_hybrid_lines(tmp, old.stdout),
            _emitted_hybrid_lines(tmp, TEXT))


def test_both_revisions_emit_one_call_per_field(emitted):
    before, after = emitted
    assert len(before) == 2, before
    assert len(after) == 2, after


def test_refactor_resolves_to_identical_options(emitted):
    '''The whole point of d58ab1a: fewer flags, same resolved options.'''
    before, after = emitted
    for old_line, new_line in zip(before, after):
        old, new = _resolved(old_line), _resolved(new_line)
        assert old == new, (old_line, new_line,
                            {k: (old[k], new.get(k))
                             for k in old if old[k] != new.get(k)})


def test_resolved_band_settings_are_the_driver_defaults(emitted):
    '''--band-cells/--max-sweeps were dropped only because the CLI defaults
    already carry them.'''
    _, after = emitted
    for line in after:
        params = _resolved(line)
        assert params["band_cells"] == hi.DEFAULT_BAND_CELLS == 2
        assert params["max_sweeps"] == hi.DEFAULT_MAX_SWEEPS == 1
        assert params["schwarz_tol"] == hi.DEFAULT_TOL == "1*V"


def test_resolved_sizes_and_field_selection(emitted):
    _, after = emitted
    fields = sorted(_resolved(line)["field"] for line in after)
    assert fields == ["drift", "weighting"]
    for line in after:
        params = _resolved(line)
        assert params["interface"] == "40*mm"
        assert params["coarse_spacing"] == 0.4
        assert params["fine_spacing"] == 0.1
        assert params["precision"] == 2e-8      # untouched by the refactor


# --------------------------------------------------------------------------
# The runner's shape after d58ab1a
# --------------------------------------------------------------------------

def test_only_one_hybrid_iterate_invocation_in_the_script():
    '''The near/far structure lives in the driver; the script calls a helper.'''
    invocations = [ln for ln in TEXT.splitlines()
                   if ln.strip().startswith("pochoir hybrid-iterate")]
    assert len(invocations) == 1, invocations
    assert TEXT.count("hybrid_field ") >= 2


def test_sizes_block_holds_the_sizes():
    for name, value in (("INTERFACE", '"40\\*mm"'),
                        ("COARSE_SPACING", "0.4"),
                        ("FINE_SPACING", "0.1")):
        assert re.search(rf'^{name}={value}', TEXT, re.M), name
    assert "SIZES" in TEXT


def test_helper_uses_the_sizes_variables_not_literals():
    body = TEXT[TEXT.index("hybrid_field ()"):]
    body = body[:body.index("\n}")]
    for var in ("$INTERFACE", "$COARSE_SPACING", "$FINE_SPACING"):
        assert var in body, var
    assert "40*mm" not in body


def test_band_flags_are_gone_from_the_script():
    body = _command_lines()
    assert "--band-cells" not in body
    assert "--max-sweeps" not in body


def test_drift_chain_still_uses_linear_interpolation():
    '''C0 is not C1 -- cubic must stay off until the seam jump is measured.'''
    assert '--interp-order linear' in TEXT
    assert '--interp-order cubic' not in TEXT


def _command_lines():
    """Executable lines only -- comments and prose may still mention flags."""
    return "\n".join(ln for ln in TEXT.splitlines()
                     if ln.strip() and not ln.strip().startswith('#'))


def test_no_precision_or_sweep_tolerance_override_was_added():
    body = _command_lines()
    assert '--precision' not in body
    assert '--schwarz-tol' not in body


def test_echo_lines_are_not_stale():
    echoes = "\n".join(ln for ln in TEXT.splitlines()
                       if ln.strip().startswith("echo"))
    assert "one-shot" not in echoes
    assert "z=20mm" not in echoes
    assert not re.search(r'(?<!1)5cm drift', echoes)
    assert "15cm" in echoes


# --------------------------------------------------------------------------
# The header, where it still says anything
#
# NOT asserted: that the header CONTAINS the method / band / cost / seam prose
# or a grid table.  That description belongs to pochoir/hybrid_iterate.py's
# module docstring and `pochoir hybrid-iterate --help`; duplicating it in the
# runner is how it went stale (z=20mm, 151/201/601) in the first place, and
# d58ab1a cut it out.  What is checked is only that whatever the header DOES
# still say agrees with the code.
# --------------------------------------------------------------------------

def test_header_points_at_the_real_documentation():
    head = _header()
    assert 'hybrid-iterate --help' in head
    assert 'hybrid_iterate.py' in head


def test_header_does_not_restate_the_method():
    """The prose that went stale must not have crept back in."""
    head = _header()
    for stale in ('ONE-SHOT hybrid solver',
                  'METHOD (single pass, no iteration',
                  'There is NO outer iteration any more',
                  '39.2 mm', '39.6 mm'):
        assert stale not in head, stale


def test_header_quotes_no_grid_shapes():
    """A grid table here cannot track --domain yes; if one reappears, every
    number in it must be one the driver actually derives."""
    quoted = set(re.findall(r'(\d+)\s*x\s*(\d+)\s*x\s*(\d+)', _header()))
    if not quoted:
        return
    derived = set()
    for field, cfg in (('drift', DRIFT_FINE), ('weighting', WEIGHT_FINE)):
        for shape in _derived(field, cfg).values():
            derived.add(tuple(shape.split(',')))
    assert quoted <= derived, quoted - derived


def test_header_does_not_contradict_the_interface():
    """The old header claimed a 20mm split while the script passes 40mm."""
    head = _header()
    planes = set(re.findall(r'z\s*=\s*(\d+)\s*mm', head))
    assert planes <= {'40'}, planes
    assert 'z = 0..60 mm' not in head


def test_header_keeps_the_flag_rationale_that_lives_here():
    """--interp-order linear is a flag on THIS script's commands, so its
    reason stays with it."""
    head = _header()
    assert '--interp-order linear' in head
    assert 'kink' in head
