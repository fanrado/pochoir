#!/usr/bin/env pytest
'''
Tests for the near/far Schwarz solve checkpoint keys (commit "fix(near-far-
solve): give the far solver its own checkpoint key").

_make_solver used to close over near_out for BOTH solvers, so the far solve
wrote its torch checkpoints into potential/<near_out> and
increment/<near_out>/inc, stomping the near checkpoints and creating
potential/<near_out>.npz before the Schwarz step had finished.  The near and
far solvers must now checkpoint under their own output keys.

The torch solver itself is stubbed out: we only care about which checkpoint
keys the CLI hands it, not about the FDM math (covered by
test_nearfarsolve.py).
'''

import numpy
import pytest
from click.testing import CliRunner

import pochoir.fdm
from pochoir.domain import Domain
from pochoir.main import Main
from pochoir.__main__ import cli

from test_nearfarsolve import _make_problem


def _store(tmp_path, p):
    storedir = str(tmp_path / "store")
    m = Main(storedir)
    m.put_domain("coarse_dom", p["coarse"])
    m.put_domain("near_dom", p["near"])
    m.put("coarse_pot", p["coarse_pot"], taxon="potential", domain="coarse_dom")
    m.put("coarse_init", p["coarse_init"], taxon="initial", domain="coarse_dom")
    m.put("coarse_bnd", p["coarse_bmask"], taxon="boundary", domain="coarse_dom")
    m.put("near_init", p["near_init"], taxon="initial", domain="near_dom")
    m.put("near_bnd", p["near_bmask"], taxon="boundary", domain="near_dom")
    return storedir


def _invoke(storedir, extra=()):
    return CliRunner().invoke(cli, [
        "--store", storedir, "near-far-solve",
        "-C", "coarse_pot",
        "--coarse-initial", "coarse_init", "--coarse-boundary", "coarse_bnd",
        "--near-initial", "near_init", "--near-boundary", "near_bnd",
        "--interface", "2.0", "--axis", "2",
        "--edges", "fixed,fixed,fixed", "--engine", "torch",
        "--epoch", "10", "-n", "1",
        "--near-precision", "1e-11", "--far-precision", "3e-7",
        "--tol", "1e-4", "--max-iters", "2",
        "--near-out", "pot_near", "--far-out", "pot_far",
    ] + list(extra))


@pytest.fixture
def spy_torch(monkeypatch):
    '''Replace solve_torch with a recorder that returns a trivially "solved"
    array (boundary values held, interior left as given).'''
    calls = []

    def fake_solve_torch(iarr, barr, edges, prec, epoch, nepochs, **kwds):
        calls.append(dict(shape=numpy.asarray(iarr).shape,
                          precision=prec, kwds=dict(kwds)))
        return numpy.asarray(iarr, dtype=float).copy(), 0.0

    monkeypatch.setattr(pochoir.fdm, "solve_torch", fake_solve_torch)
    return calls


def test_near_and_far_use_distinct_checkpoint_keys(tmp_path, spy_torch):
    '''Happy path: every torch solve checkpoints under its own output key.
    The near and far domains have the same shape here, so the solver role is
    identified by its convergence precision.'''
    p = _make_problem()
    storedir = _store(tmp_path, p)
    res = _invoke(storedir)
    assert res.exit_code == 0, res.output
    assert spy_torch, "the torch solver was never called"

    seen = set()
    for call in spy_torch:
        kwds = call["kwds"]
        key = "pot_near" if call["precision"] == 1e-11 else "pot_far"
        assert call["precision"] in (1e-11, 3e-7)
        assert kwds["potential"] == key
        assert kwds["increment"] == key + "/inc"
        seen.add(key)

    # Both solvers actually ran, so the distinction is exercised.
    assert seen == {"pot_near", "pot_far"}


def test_increment_key_derives_from_its_own_potential_key(tmp_path, spy_torch):
    '''The increment slot always hangs off the same key as the potential slot;
    a far solve must never write potential/pot_near or increment/pot_near/inc.'''
    p = _make_problem()
    storedir = _store(tmp_path, p)
    res = _invoke(storedir)
    assert res.exit_code == 0, res.output

    for call in spy_torch:
        kwds = call["kwds"]
        assert kwds["increment"] == kwds["potential"] + "/inc"
    assert not any(c["kwds"]["potential"] == "pot_near"
                   and c["precision"] == 3e-7 for c in spy_torch)


def test_precision_still_matches_the_solver_role(tmp_path, spy_torch):
    '''Edge case: adding the ckpt_key parameter must not shuffle the
    positional args -- near keeps near_precision, far keeps far_precision.'''
    p = _make_problem()
    storedir = _store(tmp_path, p)
    m = Main(storedir)
    mask = numpy.zeros(numpy.asarray(p["near_init"]).shape, dtype=bool)
    mask[1, 1, 1] = True
    m.put("near_ins", mask, taxon="boundary", domain="near_dom")

    # The insulator mask is bound to the near solver only, so it is an
    # independent marker of "this call is the near solve".
    res = _invoke(storedir, ["--insulator", "near_ins"])
    assert res.exit_code == 0, res.output
    assert spy_torch

    for call in spy_torch:
        is_near = "insulator" in call["kwds"]
        assert call["precision"] == pytest.approx(1e-11 if is_near else 3e-7)
        assert call["kwds"]["potential"] == ("pot_near" if is_near else "pot_far")


def test_insulator_mask_only_reaches_the_near_solver(tmp_path, spy_torch):
    '''Edge case: the insulator kwarg is bound per-solver alongside ckpt_key;
    only the near (pot_near) solves may receive it.'''
    p = _make_problem()
    storedir = _store(tmp_path, p)
    m = Main(storedir)
    mask = numpy.zeros(numpy.asarray(p["near_init"]).shape, dtype=bool)
    mask[1, 1, 1] = True
    m.put("near_ins", mask, taxon="boundary", domain="near_dom")

    res = _invoke(storedir, ["--insulator", "near_ins"])
    assert res.exit_code == 0, res.output

    for call in spy_torch:
        if call["kwds"]["potential"] == "pot_near":
            assert "insulator" in call["kwds"]
        else:
            assert "insulator" not in call["kwds"]
