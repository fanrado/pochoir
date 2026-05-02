"""Guard against committing executed notebooks.

PLAN_for_production.md §2 requires the kept validation notebooks under
``test/for_pixel/`` to be checked in with cleared outputs (keeps the repo
small and review diffs noise-free). Notebooks under ``toy/`` are explicitly
excluded — those are personal scratch space the user keeps.
"""
import nbformat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KEPT_NOTEBOOKS = [
    "test/for_pixel/validate_FR.ipynb",
    "test/for_pixel/compare_with_without_grid.ipynb",
    "test/for_pixel/plot_weighting_potential.ipynb",
    "test/for_pixel/velocity_vs_err.ipynb",
]


def test_kept_notebooks_have_no_outputs():
    dirty = []
    for rel in KEPT_NOTEBOOKS:
        path = REPO_ROOT / rel
        nb = nbformat.read(path, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if cell.get("outputs"):
                dirty.append(f"{rel} cell {i}: has {len(cell['outputs'])} output(s)")
            if cell.get("execution_count") is not None:
                dirty.append(f"{rel} cell {i}: execution_count={cell['execution_count']}")
    assert not dirty, "Notebooks committed with outputs:\n" + "\n".join(dirty)
