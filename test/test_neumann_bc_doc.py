#!/usr/bin/env pytest
'''
Tests for commit 9d82be8 "Doc: physics justification for the no-flux Neumann FR4
boundary".

DOCUMENTATION ONLY: a LaTeX writeup (test/neumann_bc_fr4_insulator.tex) arguing
why the homogeneous Neumann BC is the correct steady-state model of the charged
FR4 insulator.  There is no production code / feature to unit-test, and pdflatex
is not installed on this host (ships as .tex source).

What IS worth guarding: the writeup cross-references concrete code symbols/files.
These tests keep the doc honest -- if a referenced symbol is renamed or removed,
the doc reference should fail loudly rather than rot silently -- plus a minimal
LaTeX structural sanity check.
'''

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEX = REPO_ROOT / "test" / "neumann_bc_fr4_insulator.tex"


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


# --------------------------------------------------------------------------
# The document exists and is structurally sane LaTeX
# --------------------------------------------------------------------------

def test_tex_exists_and_nonempty(tex):
    assert TEX.is_file()
    assert len(tex) > 500


def test_document_environment_present(tex):
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex


def test_environments_balanced(tex):
    # Every \begin{...} has a matching \end{...} (count parity; sufficient for a
    # cheap rot guard without a full LaTeX parser).
    assert len(re.findall(r"\\begin\{", tex)) == len(re.findall(r"\\end\{", tex))


# --------------------------------------------------------------------------
# Code cross-references must point at symbols/files that actually exist
# --------------------------------------------------------------------------

def test_referenced_files_exist(tex):
    for rel in ("pochoir/fdm_generic.py", "pochoir/fdm_torch.py"):
        # The doc cites the path (with an escaped underscore in \texttt).
        assert rel.replace("_", r"\_") in tex
        assert (REPO_ROOT / rel).is_file()


def test_referenced_stencil_symbols_exist(tex):
    fg = (REPO_ROOT / "pochoir" / "fdm_generic.py").read_text()
    # The doc names the no-flux stencil helpers; they must exist in the module.
    assert r"neumann\_coeff" in tex
    assert r"stencil\_poisson\_neumann" in tex
    assert "def neumann_coeff" in fg
    assert "def stencil_poisson_neumann" in fg


def test_referenced_endtag_symbol_exists(tex):
    # pochoir-w3x9 (commit 2567924) removed all drift-side enforcement; the only
    # endtag is DRIFT_NONE and the doc must describe that contract, not the
    # retired DRIFT_SURFACE terminal event.
    dn = (REPO_ROOT / "pochoir" / "drift_numpy.py").read_text()
    assert r"DRIFT\_NONE" in tex
    assert "DRIFT_NONE" in dn
    assert "DRIFT_SURFACE" not in tex
    assert "DRIFT_SURFACE" not in dn


def test_doc_describes_no_flux_not_dielectric(tex):
    # Sanity that the writeup is about the no-flux Neumann BC (NO epsilon), the
    # actual shipped model, and explicitly contrasts the dielectric formulation.
    assert re.search(r"[Nn]eumann", tex)
    assert re.search(r"no.?flux", tex)
