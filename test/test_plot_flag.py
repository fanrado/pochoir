#!/usr/bin/env pytest
"""Verify the ``--plot/--no-plot`` flags on pipeline CLI commands.

PLAN_for_production.md §1: plotting blocks in ``starts``, ``drift`` and
``induce-pixel`` must be gated behind ``--plot`` (default off) so a default
pipeline run does not produce side-effect PNGs.
"""

import inspect

import pytest

from pochoir import __main__ as pmain


PLOT_COMMANDS = ("starts", "drift", "induce_pixel")


def _callback(cmd_name):
    fn = getattr(pmain, cmd_name)
    return getattr(fn, "callback", fn)


@pytest.mark.parametrize("cmd_name", PLOT_COMMANDS)
def test_plot_flag_exists(cmd_name):
    sig = inspect.signature(_callback(cmd_name))
    assert "plot" in sig.parameters, (
        f"{cmd_name} must take a `plot` argument from the --plot/--no-plot click option"
    )


@pytest.mark.parametrize("cmd_name", PLOT_COMMANDS)
def test_plot_flag_off_produces_no_files(cmd_name, tmp_path, monkeypatch):
    """With ``plot=False`` the function body must contain no top-level
    ``plt.savefig`` / ``plt.show`` calls — they must be inside an ``if plot:``
    branch so a default invocation cannot write PNGs into ``store/``."""
    src = inspect.getsource(_callback(cmd_name))

    import ast
    tree = ast.parse(inspect.cleandoc(src))
    func = tree.body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))

    def _calls(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr in ("savefig", "show"):
                    yield n

    plot_branches = []
    for n in ast.walk(func):
        if isinstance(n, ast.If):
            t = n.test
            if isinstance(t, ast.Name) and t.id == "plot":
                plot_branches.append(n)

    bad = []
    for call in _calls(func):
        inside_plot_branch = any(
            any(c is call for c in ast.walk(b))
            for b in plot_branches
        )
        if not inside_plot_branch:
            bad.append(call.lineno)

    assert not bad, (
        f"{cmd_name}: savefig/show calls outside `if plot:` at lines {bad}"
    )


@pytest.mark.parametrize("cmd_name", PLOT_COMMANDS)
def test_plot_flag_on_produces_expected_files(cmd_name):
    """With ``plot=True`` the body must contain at least one ``plt.savefig``
    inside the ``if plot:`` branch — i.e. enabling the flag is wired to a real
    plotting side-effect, not a no-op."""
    src = inspect.getsource(_callback(cmd_name))

    import ast
    tree = ast.parse(inspect.cleandoc(src))
    func = tree.body[0]

    found = False
    for n in ast.walk(func):
        if isinstance(n, ast.If):
            t = n.test
            if isinstance(t, ast.Name) and t.id == "plot":
                for inner in ast.walk(n):
                    if isinstance(inner, ast.Call):
                        f = inner.func
                        if isinstance(f, ast.Attribute) and f.attr == "savefig":
                            found = True
                            break
        if found:
            break

    assert found, (
        f"{cmd_name}: no `plt.savefig(...)` inside an `if plot:` block — the "
        f"--plot flag should produce at least one output file"
    )
