"""Guard against TODO/FIXME regressions in pipeline modules.

The TODO sweep in PLAN_for_production.md §1 disposes of every TO DO / FIXME
in modules exercised by ``test/test-full-3d-pixel.sh``. This test pins that
state so a future stray marker in those files is caught immediately.
"""
import re
from pathlib import Path

PIPELINE_MODULES = [
    "pochoir/__main__.py",
    "pochoir/fdm_generic.py",
    "pochoir/fdm_torch.py",
    "pochoir/drift_numpy.py",
    "pochoir/gen_pcb_pixel_with_grid.py",
    "pochoir/gen_pcb_drift_pixel_with_grid.py",
]

# Word-boundaried so the marker words are matched as whole tokens; without the
# \b anchors "TO ?DO" also matches "to do" embedded in ordinary prose such as
# "...array to downsample" (a false positive at __main__.py's downsample help).
MARKER = re.compile(r"\bTO ?DO\b|\bFIXME\b", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_modules_have_no_todo_markers():
    hits = []
    for rel in PIPELINE_MODULES:
        path = REPO_ROOT / rel
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if MARKER.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, "Stray TODO/FIXME markers in pipeline modules:\n" + "\n".join(hits)
