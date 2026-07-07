#!/usr/bin/env pytest
'''
Tests for the combined drift+weighting near-field driver (commit 087b92d):
test/run-full-3d-pixel.sh

This driver orchestrates the full pixel calculation:
  PART A: drift field      -> potential/full
  PART B: weighting field  -> potential/weight3d
  PART C: velo -> starts -> drift -> induce-pixel -> current/induced_current
No production Python changed; the CLI commands it calls are unit-tested
elsewhere.  Running it performs multi-hour FDM solves, so it cannot be a
unit test.  We statically pin its integrity:

  - valid bash, no TODO/FIXME markers,
  - every `pochoir <subcommand>` it invokes is a registered CLI command,
  - the options it passes to refine/near-bc/stitch-near exist,
  - the core claim: PART A/B/C `want` store keys are pairwise DISJOINT
    (a collision would make `want` clobber/skip across parts),
  - PART C wiring consumes the upstream drift/weighting products,
  - the terminal outputs (potential/full, potential/weight3d,
    current/induced_current) exist.
'''

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from pochoir.__main__ import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "test" / "run-full-3d-pixel.sh"

MARKER = re.compile(r"TO ?DO|FIXME", re.IGNORECASE)
# All three section dividers, in order, for splitting the driver into parts.
PART_DIV_RE = re.compile(r"^##\s*PART [ABC]\b", re.MULTILINE)


def _text():
    return SCRIPT.read_text()


def _want_keys(text):
    '''
    Store keys produced by `want ...` lines.

    The driver's multi-key guard takes a quoted, space-separated list as its
    first argument -- e.g. `want "initial/coarse boundary/coarse" \\` -- while
    older single-key calls pass a bare token -- `want potential/drift3d \\`.
    Both forms are flattened into the full set of output keys.
    '''
    keys = []
    for m in re.finditer(r'^\s*want\s+(?:"([^"]+)"|(\S+))', text, re.MULTILINE):
        group = m.group(1) if m.group(1) is not None else m.group(2)
        keys.extend(group.split())
    return keys


def _invoked_subcommands(text):
    return {m.group(1) for m in re.finditer(r"\bpochoir\s+([a-z0-9-]+)", text)}


def test_script_exists():
    assert SCRIPT.is_file(), f"missing driver script {SCRIPT}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_script_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_script_has_no_todo_markers():
    hits = [f"{SCRIPT.name}:{i}: {line.strip()}"
            for i, line in enumerate(_text().splitlines(), 1)
            if MARKER.search(line)]
    assert not hits, "Stray TODO/FIXME markers:\n" + "\n".join(hits)


def test_invoked_subcommands_are_registered():
    invoked = _invoked_subcommands(_text())
    # PART A now uses the overlapping-Schwarz solve; PART B still uses the
    # refine/near-bc/stitch-near near-field commands.
    assert {"refine", "near-bc", "stitch-near", "near-far-solve"} <= invoked
    missing = sorted(invoked - set(cli.commands))
    assert not missing, f"script invokes unregistered commands: {missing}"


@pytest.mark.parametrize("name,needed", [
    ("refine", {"--coarse", "--initial", "--boundary", "--output"}),
    ("near-bc", {"--initial", "--boundary", "--coarse",
                 "--initial-out", "--boundary-out"}),
    ("stitch-near", {"--near", "--coarse", "--domain", "--output"}),
    ("near-far-solve", {"--coarse-potential", "--coarse-initial",
                        "--coarse-boundary", "--near-initial", "--near-boundary",
                        "--interface", "--near-out", "--far-out"}),
])
def test_command_options_used_by_script_exist(name, needed):
    cmd = cli.commands[name]
    opts = set()
    for p in cmd.params:
        opts.update(getattr(p, "opts", []))
    missing = sorted(needed - opts)
    assert not missing, f"{name} missing options used by script: {missing}"


def _split_parts(text):
    '''Split the driver into its PART A/B/C sections at the dividers.'''
    bounds = [m.start() for m in PART_DIV_RE.finditer(text)]
    assert len(bounds) == 3, f"expected 3 PART dividers, found {len(bounds)}"
    a = text[bounds[0]:bounds[1]]
    b = text[bounds[1]:bounds[2]]
    c = text[bounds[2]:]
    return a, b, c


def test_parts_use_disjoint_store_keys():
    '''
    The driver's core claim: the three parts (A drift, B weighting, C
    velocity/paths/current) write to pairwise non-overlapping `want` keys,
    so they run back-to-back without one clobbering another's intermediates.
    '''
    part_a, part_b, part_c = _split_parts(_text())
    keys_a = set(_want_keys(part_a))
    keys_b = set(_want_keys(part_b))
    keys_c = set(_want_keys(part_c))
    assert keys_a and keys_b and keys_c, \
        f"empty part keys: A={keys_a} B={keys_b} C={keys_c}"

    for (n1, k1), (n2, k2) in [
        (("A", keys_a), ("B", keys_b)),
        (("A", keys_a), ("C", keys_c)),
        (("B", keys_b), ("C", keys_c)),
    ]:
        overlap = sorted(k1 & k2)
        assert not overlap, f"PART {n1}/{n2} store keys collide: {overlap}"

    # Sanity: the documented naming convention holds.
    assert all(not k.split("/")[-1].startswith("weight") for k in keys_a), \
        "PART A unexpectedly uses weight_* keys"
    assert any("weight" in k for k in keys_b), \
        "PART B should use weight_* keys"


def _stitch_output(part_text):
    '''The `stitch-near --output <key>` produced in a part (its potential).'''
    m = re.search(r"stitch-near\b.*?--output\s+(\S+)", part_text, re.DOTALL)
    return m.group(1) if m else None


def test_part_c_wiring_consumes_upstream_products():
    '''
    PART C must consume the actual products of PART A (the stitched drift
    potential) and PART B (the stitched weighting potential), and chain
    velo->drift->induce through its own intermediate keys.

    The drift potential key is asserted by *internal consistency* (PART C's
    velo reads exactly what PART A's stitch-near wrote) rather than a
    hardcoded name, so a rename stays valid as long as both ends agree.
    '''
    part_a, part_b, part_c = _split_parts(_text())

    drift_pot = _stitch_output(part_a)
    weight_pot = _stitch_output(part_b)
    assert drift_pot, "PART A produces no stitch-near output"
    assert weight_pot, "PART B produces no stitch-near output"
    assert drift_pot != weight_pot, "drift and weighting potentials collide"

    # velo reads the stitched drift potential and emits the velocity field.
    assert re.search(rf"pochoir velo\b.*?--potential\s+{re.escape(drift_pot)}\b",
                     part_c, re.DOTALL), \
        f"PART C velo does not consume PART A's drift potential {drift_pot!r}"
    assert re.search(r"pochoir velo\b.*?--velocity\s+velocity/drift3d",
                     part_c, re.DOTALL)
    # drift consumes the velocity field and the start points.
    assert re.search(r"pochoir drift\b.*?--velocity\s+velocity/drift3d",
                     part_c, re.DOTALL)
    assert re.search(r"pochoir drift\b.*?--starts\s+starts/drift3d",
                     part_c, re.DOTALL)
    # induce-pixel reads the weighting field and the drift paths.
    assert re.search(rf"pochoir induce-pixel\b.*?--weighting\s+{re.escape(weight_pot)}\b",
                     part_c, re.DOTALL), \
        f"PART C induce-pixel does not consume PART B's weighting {weight_pot!r}"
    assert re.search(r"pochoir induce-pixel\b.*?--paths\s+paths/drift3d_tight",
                     part_c, re.DOTALL)

    keys_c = set(_want_keys(part_c))
    assert {"velocity/drift3d", "starts/drift3d",
            "paths/drift3d_tight", "current/induced_current"} <= keys_c


def test_velo_uses_full_fine_boundary():
    '''
    Fix for pochoir-h76: PART A must gen a full-fine boundary (boundary/fine
    on domain/fine), and PART C's velo must pass --boundary boundary/fine so
    the drift velocity gets the exact fine electrode treatment.
    '''
    part_a, _b, part_c = _split_parts(_text())

    # PART A produces a full-fine boundary mask via gen (no FDM solve).
    assert "boundary/fine" in set(_want_keys(part_a)), \
        "PART A does not gen boundary/fine for the stitched domain"
    assert re.search(r"pochoir gen\b.*?--domain\s+domain/fine\b.*?"
                     r"--boundary\s+boundary/fine\b",
                     part_a, re.DOTALL), "boundary/fine not gen'd on domain/fine"

    # PART C's velo consumes that full-fine boundary.
    assert re.search(r"pochoir velo\b.*?--boundary\s+boundary/fine\b",
                     part_c, re.DOTALL), \
        "PART C velo does not pass --boundary boundary/fine"


def test_terminal_outputs_present():
    '''The field products and final induced current the driver promises.'''
    text = _text()
    part_a, part_b, _c = _split_parts(text)
    keys = set(_want_keys(text))
    assert _stitch_output(part_a) in keys      # drift field (PART A)
    assert _stitch_output(part_b) in keys       # weighting field (PART B)
    assert "potential/weight3d" in keys          # induce-pixel input
    assert "current/induced_current" in keys     # final product (PART C)
