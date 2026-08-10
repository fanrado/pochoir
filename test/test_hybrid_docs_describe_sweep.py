#!/usr/bin/env pytest
'''
Tests for commit 8876cc6: the hybrid_iterate docs describe the Schwarz sweep
rather than the old one-shot scheme.

Docs go stale silently, and this module's docstrings are the only place the
scheme is written down -- they were claiming "a SINGLE pass, no iteration",
"the far field is never re-solved" and that nearfar.schwarz_solve is not called
at all, all of which stopped being true at 5f0785d.  These tests pin the claims
that must not silently drift back:

  * the module docs describe the sweep and drop the "no iteration" claims;
  * the C0-by-construction statement is present AND still requires
    --interp-order linear (no C1 claim until the seam gradient jump is
    measured);
  * --max-sweeps 0 is documented as the one-shot fallback;
  * hybrid_iterate documents band_cells/max_sweeps/tol instead of "no tol, no
    max-iters";
  * _stitch gives both seam-continuity reasons.

The commit is comments and docstrings only, so it must also be a no-op on the
code itself.
'''

import ast
import inspect
import subprocess

import pochoir.hybrid_iterate as hi

MODULE_DOC = hi.__doc__
DRIVER_DOC = hi.hybrid_iterate.__doc__
STITCH_DOC = hi._stitch.__doc__


# --------------------------------------------------------------------------
# Module docs: the sweep is the scheme now
# --------------------------------------------------------------------------

def test_module_doc_describes_the_schwarz_sweep():
    assert 'Schwarz sweep' in MODULE_DOC
    assert 'coarse -> near -> far -> near' in MODULE_DOC
    # The sweep is run by the existing command/module, which is now called.
    assert 'near-far-solve' in MODULE_DOC
    assert 'schwarz_solve' in MODULE_DOC


def test_module_doc_drops_the_one_shot_claims():
    for stale in ('a SINGLE pass, no iteration',
                  'there is no sweep at all here',
                  'is never re-solved at 0.1mm',
                  'deliberately untouched by this work'):
        assert stale not in MODULE_DOC, stale


def test_module_doc_pins_the_band_geometry():
    '''3 coarse nodes at overlap 2, and why only the inner one is Dirichlet.'''
    assert '3-coarse-node band' in MODULE_DOC or '3-coarse-node' in MODULE_DOC
    assert '39.2' in MODULE_DOC and '40.0' in MODULE_DOC
    assert 'INNER band node' in MODULE_DOC
    assert 'FREE' in MODULE_DOC


# --------------------------------------------------------------------------
# The C0 / not-C1 claim
# --------------------------------------------------------------------------

def test_seam_is_documented_c0_but_not_c1():
    assert 'C0 BY CONSTRUCTION' in MODULE_DOC
    assert 'does NOT make it C1' in MODULE_DOC
    # ...so linear interpolation is still mandatory for drift.
    assert 'interp-order linear' in MODULE_DOC
    assert 'STILL REQUIRED' in MODULE_DOC


def test_max_sweeps_zero_documented_as_the_one_shot_fallback():
    assert 'max-sweeps 0' in MODULE_DOC
    assert 'one-shot' in MODULE_DOC


# --------------------------------------------------------------------------
# Driver and _stitch docstrings
# --------------------------------------------------------------------------

def test_driver_doc_documents_the_sweep_knobs():
    for knob in ('band_cells', 'max_sweeps', 'tol'):
        assert knob in DRIVER_DOC, knob
    assert 'no tol, no max-iters' not in DRIVER_DOC
    # precision is still distinguished from tol.
    assert 'precision' in DRIVER_DOC


def test_driver_doc_states_the_max_sweeps_zero_fallback():
    assert 'max_sweeps=0' in DRIVER_DOC


def test_stitch_doc_gives_both_seam_reasons():
    assert 'Schwarz sweep' in STITCH_DOC
    assert 'max_sweeps=0' in STITCH_DOC
    assert 'near_bc' in STITCH_DOC


# --------------------------------------------------------------------------
# Docs-only: the code must be untouched
# --------------------------------------------------------------------------

def _strip_docstrings(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return ast.dump(tree)


def test_commit_changed_no_code():
    old = subprocess.run(
        ['git', 'show', '8876cc6^:pochoir/hybrid_iterate.py'],
        capture_output=True, text=True, cwd=str(
            __import__('pathlib').Path(__file__).resolve().parent.parent))
    if old.returncode != 0:
        import pytest
        pytest.skip('commit 8876cc6 not reachable in this checkout')
    new = inspect.getsource(hi)
    assert _strip_docstrings(old.stdout) == _strip_docstrings(new)
