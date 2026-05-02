#!/usr/bin/env pytest
"""Guard against re-introducing the unconditional ``sys.exit()`` halt in
the ``induce-pixel`` CLI command (see PLAN_for_production.md §1)."""

import ast
import inspect

from pochoir import __main__ as pmain


def _induce_pixel_func():
    fn = pmain.induce_pixel
    return getattr(fn, "callback", fn)


def test_induce_pixel_has_no_sys_exit_call():
    src = inspect.getsource(_induce_pixel_func())
    tree = ast.parse(inspect.cleandoc(src))

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "exit":
            if isinstance(f.value, ast.Name) and f.value.id == "sys":
                bad.append(node.lineno)
        elif isinstance(f, ast.Name) and f.id == "exit":
            bad.append(node.lineno)

    assert not bad, f"sys.exit()/exit() calls remain in induce_pixel at lines {bad}"
