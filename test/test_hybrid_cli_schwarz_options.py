#!/usr/bin/env pytest
'''
Tests for commit 38abe13: the `hybrid-iterate` CLI exposes
--band-cells/--max-sweeps/--schwarz-tol.

Before this the sweep ran at hardcoded defaults with no way to tune or disable
it from the shell.  What must hold:

  * the three options exist with defaults 2 / 1 / '1*V' and the right types;
  * they are threaded to the driver as band_cells / max_sweeps / tol -- note
    the RENAME (--schwarz-tol -> tol), which is where a wiring bug would hide;
  * --precision stays a SINGLE option at 2e-8; no --near-precision/
    --far-precision was added;
  * --max-sweeps 0 reaches the driver as max_sweeps=0 (the one-shot switch);
  * the help text and docstring describe the sweep, not the old one-shot pass.

The driver itself is stubbed: this is an option-surface and wiring test.
'''

import pytest
from click.testing import CliRunner

import pochoir.hybrid_iterate as hi
from pochoir.__main__ import cli, hybrid_iterate as hi_cmd


def _opts():
    return {p.name: p for p in hi_cmd.params}


@pytest.fixture
def driver_kwds(monkeypatch, tmp_path):
    '''Invoke the CLI with the driver replaced by a recorder; return its
    keyword arguments.'''
    seen = {}

    def fake_driver(ctx, coarse_config, fine_config, **kwds):
        seen.update(kwds)
        seen['coarse_config'] = coarse_config
        seen['fine_config'] = fine_config
        return 'potential/drift3d', ()

    monkeypatch.setattr(hi, 'hybrid_iterate', fake_driver)

    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    def run(extra=()):
        seen.clear()
        res = CliRunner().invoke(cli, [
            "--store", str(tmp_path / "store"), "hybrid-iterate",
            "--coarse-config", str(cfg), "--fine-config", str(cfg),
        ] + list(extra))
        assert res.exit_code == 0, res.output
        return seen

    return run


# --------------------------------------------------------------------------
# Option surface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, default, typename", [
    ("band_cells", 2, "INTEGER"),
    ("max_sweeps", 1, "INTEGER"),
    ("schwarz_tol", "1*V", "TEXT"),
])
def test_new_options_exist_with_their_defaults(name, default, typename):
    opt = _opts()[name]
    assert opt.default == default
    assert opt.type.name.upper() == typename


def test_precision_stays_a_single_option():
    opts = _opts()
    assert opts["precision"].default == 2e-8
    assert "near_precision" not in opts
    assert "far_precision" not in opts


def test_tol_option_is_named_schwarz_tol_not_tol():
    '''--tol would collide with the old convergence-loop meaning that task13
    removed; the sweep tolerance is deliberately a distinct flag.'''
    opts = _opts()
    assert "schwarz_tol" in opts
    assert "tol" not in opts
    assert "--schwarz-tol" in opts["schwarz_tol"].opts


# --------------------------------------------------------------------------
# Wiring into the driver
# --------------------------------------------------------------------------

def test_defaults_reach_the_driver(driver_kwds):
    kwds = driver_kwds()
    assert kwds["band_cells"] == 2
    assert kwds["max_sweeps"] == 1
    assert kwds["tol"] == "1*V"
    assert kwds["precision"] == 2e-8


def test_explicit_values_reach_the_driver(driver_kwds):
    kwds = driver_kwds(["--band-cells", "3", "--max-sweeps", "4",
                        "--schwarz-tol", "0.05*V", "--precision", "1e-9"])
    assert kwds["band_cells"] == 3
    assert kwds["max_sweeps"] == 4
    assert kwds["tol"] == "0.05*V"          # renamed on the way through
    assert kwds["precision"] == 1e-9
    assert "schwarz_tol" not in kwds


def test_max_sweeps_zero_is_passed_through(driver_kwds):
    '''0 must survive as 0 -- a falsy value is easy to lose to an `or`.'''
    kwds = driver_kwds(["--max-sweeps", "0"])
    assert kwds["max_sweeps"] == 0


def test_other_driver_arguments_are_unchanged(driver_kwds):
    kwds = driver_kwds(["--interface", "40*mm", "--field", "weighting"])
    assert kwds["interface"] == "40*mm"
    assert kwds["field"] == "weighting"
    assert kwds["derive_domain"] is True


def test_non_integer_band_cells_is_rejected(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    res = CliRunner().invoke(cli, [
        "--store", str(tmp_path / "store"), "hybrid-iterate",
        "--coarse-config", str(cfg), "--fine-config", str(cfg),
        "--band-cells", "two",
    ])
    assert res.exit_code != 0
    assert "two" in res.output


# --------------------------------------------------------------------------
# Help text
# --------------------------------------------------------------------------

def test_help_describes_the_sweep_not_the_one_shot_pass():
    doc = hi_cmd.__doc__
    assert "A SINGLE pass, no iteration" not in doc
    assert "coarse -> near -> far -> near" in doc
    assert "--max-sweeps 0" in doc
    assert "C0 by construction" in doc
    assert "interp-order linear" in doc


@pytest.mark.parametrize("name, needle", [
    ("band_cells", "COARSE cells"),
    ("max_sweeps", "0 skips the sweep"),
    ("schwarz_tol", "Inter-sweep"),
])
def test_option_help_text(name, needle):
    assert needle in _opts()[name].help


# --------------------------------------------------------------------------
# Commit 1aa122e: the click defaults ARE the driver constants
# --------------------------------------------------------------------------

def test_click_defaults_are_the_driver_constants():
    '''Not just equal by value -- sourced from the constants, so changing a
    constant changes the CLI default instead of being silently overridden by a
    stale literal (the CLI always passes a value).'''
    opts = _opts()
    assert opts["band_cells"].default is hi.DEFAULT_BAND_CELLS
    assert opts["max_sweeps"].default is hi.DEFAULT_MAX_SWEEPS
    assert opts["schwarz_tol"].default is hi.DEFAULT_TOL


def test_option_source_has_no_hardcoded_default_literals():
    import inspect as _inspect
    import re

    src = _inspect.getsource(__import__('pochoir.__main__',
                                        fromlist=['__main__']))
    block = src[src.index('@click.option("--band-cells"'):
                src.index('@click.pass_context',
                          src.index('@click.option("--band-cells"'))]
    for flag, const in (("--band-cells", "DEFAULT_BAND_CELLS"),
                        ("--max-sweeps", "DEFAULT_MAX_SWEEPS"),
                        ("--schwarz-tol", "DEFAULT_TOL")):
        assert const in block, flag
    assert not re.search(r"default=\s*(2\b|1\b|'1\*V')", block)


def test_default_values_are_unchanged_by_the_refactor():
    '''--help must still print 2 / 1 / 1*V.'''
    opts = _opts()
    assert opts["band_cells"].default == 2
    assert opts["max_sweeps"].default == 1
    assert opts["schwarz_tol"].default == "1*V"
    assert opts["precision"].default == 2e-8


def test_no_import_cycle_importing_main_first():
    '''pochoir.__main__ imports pochoir.hybrid_iterate at module scope; make
    sure that import works from a cold interpreter in both orders.'''
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for first, second in (("pochoir.__main__", "pochoir.hybrid_iterate"),
                          ("pochoir.hybrid_iterate", "pochoir.__main__")):
        res = subprocess.run(
            [sys.executable, "-c",
             f"import {first}, {second}; print('ok')"],
            capture_output=True, text=True, cwd=str(root))
        assert res.returncode == 0, res.stderr
        assert "ok" in res.stdout
