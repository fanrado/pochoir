#!/usr/bin/env pytest
"""Guard against debug ``print()`` calls leaking back into pipeline modules.

PLAN_for_production.md §1, second bullet: stdout must be empty at default log
level for the modules exercised by ``test/test-full-3d-pixel.sh``.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Modules listed in §1 of PLAN_for_production.md ("touched by the pipeline").
PIPELINE_MODULES = [
    REPO_ROOT / "pochoir" / "__main__.py",
    REPO_ROOT / "pochoir" / "gen_pcb_drift_pixel_with_grid.py",
    REPO_ROOT / "pochoir" / "gen_pcb_pixel_with_grid.py",
]

# Functions/CLI commands in __main__.py that the pipeline exercises and that
# previously contained debug prints. Limit the scan to these so we don't
# regress on prints inside unrelated CLI commands that aren't yet productionized.
TOUCHED_FUNCS_MAIN = {"velo", "starts", "drift"}


def _print_call_lines(tree, only_funcs=None):
    """Return [(func_name, lineno)] of every ``print(...)`` call.

    If ``only_funcs`` is given, restrict to calls inside those top-level
    function defs.
    """
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if only_funcs is not None and node.name not in only_funcs:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                    and child.func.id == "print":
                hits.append((node.name, child.lineno))
    return hits


def _all_print_calls(tree):
    return _print_call_lines(tree, only_funcs=None)


def test_no_prints_in_touched_main_functions():
    src = (REPO_ROOT / "pochoir" / "__main__.py").read_text()
    tree = ast.parse(src)
    hits = _print_call_lines(tree, only_funcs=TOUCHED_FUNCS_MAIN)
    assert not hits, (
        f"Stray print() calls in __main__.py productionized commands "
        f"{sorted(TOUCHED_FUNCS_MAIN)}: {hits}"
    )


def test_no_prints_in_gen_pcb_pixel_with_grid():
    src = (REPO_ROOT / "pochoir" / "gen_pcb_pixel_with_grid.py").read_text()
    tree = ast.parse(src)
    hits = _all_print_calls(tree)
    assert not hits, f"Stray print() calls in gen_pcb_pixel_with_grid.py: {hits}"


def test_no_prints_in_gen_pcb_drift_pixel_with_grid():
    src = (REPO_ROOT / "pochoir" / "gen_pcb_drift_pixel_with_grid.py").read_text()
    tree = ast.parse(src)
    hits = _all_print_calls(tree)
    assert not hits, (
        f"Stray print() calls in gen_pcb_drift_pixel_with_grid.py: {hits}"
    )
