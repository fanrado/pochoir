#!/usr/bin/env pytest
'''
Tests for `_schwarz()`, the banded near/far Schwarz sweep in
pochoir/hybrid_iterate.py (commits 48f9250, 5f0785d).

What must hold:
  * the new defaults (band=2 coarse cells, one sweep, 1*V tolerance);
  * _schwarz ctx.invokes near-far-solve with the sweep-0 near solution as the
    WARM START (--near-potential) and writes to the NEW near_schwarz /
    coarse_schwarz keys, so near / near_bc / near_refined survive;
  * both --near-precision and --far-precision take the driver's single
    precision, overriding near-far-solve's own 2e-11/2e-7 split;
  * band_cells -> overlap, max_sweeps -> max_iters, edges/engine from the
    profile and module constants;
  * the w_ prefix namespacing holds for the weighting profile;
  * the _want resume guard covers both outputs, and a step that produces
    nothing still fails loudly;
  * `interface` is required and positional (commit 5f0785d removed its
    misleading '20*mm' default).

No solving happens here: ctx.invoke is a recorder, as in the rest of the
hybrid-driver unit tests.
'''

import inspect

import numpy
import pytest

import pochoir.hybrid_iterate as hi
from pochoir.__main__ import near_far_solve


class FakeCtx:
    '''Records ctx.invoke calls; each one "produces" the keys it was told to
    write so the _want guard is satisfied.'''

    def __init__(self, tmp_path, produce=True):
        self.calls = []
        self.produce = produce

        class _Obj:
            instore_path = tmp_path

        self.obj = _Obj()

    def invoke(self, cmd, **kwds):
        self.calls.append((cmd, kwds))
        if not self.produce:
            return
        for key in (kwds.get('near_out'), kwds.get('far_out')):
            if key:
                path = self.obj.instore_path / (key + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(path, arr=numpy.zeros(2))

    def place(self, *keys):
        '''Pretend these store keys already exist (resume case).'''
        for key in keys:
            path = self.obj.instore_path / (key + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            numpy.savez(path, arr=numpy.zeros(2))


def _run(tmp_path, field='drift', **kwds):
    ctx = FakeCtx(tmp_path)
    prof = hi._profile(field)
    near_pot = hi._key(prof, 'potential', 'near')
    kwds.setdefault('interface', '20*mm')
    interface = kwds.pop('interface')
    out = hi._schwarz(ctx, prof, near_pot, interface, **kwds)
    return ctx, prof, near_pot, out


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

def test_schwarz_defaults():
    assert hi.DEFAULT_BAND_CELLS == 2
    assert hi.DEFAULT_MAX_SWEEPS == 1
    assert hi.DEFAULT_TOL == '1*V'


def test_signature_defaults_come_from_the_module_constants():
    sig = inspect.signature(hi._schwarz)
    assert sig.parameters['band_cells'].default == hi.DEFAULT_BAND_CELLS
    assert sig.parameters['max_sweeps'].default == hi.DEFAULT_MAX_SWEEPS
    assert sig.parameters['tol'].default == hi.DEFAULT_TOL


# --------------------------------------------------------------------------
# Happy path: one near-far-solve invocation with the right options
# --------------------------------------------------------------------------

def test_schwarz_invokes_near_far_solve_once(tmp_path):
    ctx, prof, near_pot, (near_out, far_out) = _run(tmp_path, precision=2e-8,
                                                    interface='20*mm')
    assert len(ctx.calls) == 1
    cmd, kwds = ctx.calls[0]
    assert cmd is near_far_solve

    assert kwds['near_potential'] == near_pot          # warm start
    assert kwds['near_out'] == near_out == 'potential/near_schwarz'
    assert kwds['far_out'] == far_out == 'potential/coarse_schwarz'
    assert kwds['coarse_potential'] == 'potential/coarse'
    assert kwds['coarse_initial'] == 'initial/coarse'
    assert kwds['coarse_boundary'] == 'boundary/coarse'
    assert kwds['near_initial'] == 'initial/near'
    assert kwds['near_boundary'] == 'boundary/near'
    assert kwds['insulator'] == 'initial/near_insulator'
    assert kwds['axis'] == 2
    assert kwds['interface'] == '20*mm'
    assert kwds['edges'] == prof['edges'] == 'per,per,fix'
    assert kwds['engine'] == hi.ENGINE
    assert kwds['epoch'] == hi.EPOCH
    assert kwds['nepochs'] == hi.NEPOCHS


def test_single_precision_overrides_the_near_far_split(tmp_path):
    '''Both sides take the driver's one precision, not 2e-11/2e-7.'''
    ctx, _, _, _ = _run(tmp_path, precision=3.5e-9)
    _, kwds = ctx.calls[0]
    assert kwds['near_precision'] == 3.5e-9
    assert kwds['far_precision'] == 3.5e-9


def test_band_and_sweeps_map_to_overlap_and_max_iters(tmp_path):
    ctx, _, _, _ = _run(tmp_path, band_cells=3, max_sweeps=4, tol='0.05*V')
    _, kwds = ctx.calls[0]
    assert kwds['overlap'] == 3
    assert kwds['max_iters'] == 4
    assert kwds['tol'] == '0.05*V'


def test_defaults_applied_when_not_passed(tmp_path):
    ctx, _, _, _ = _run(tmp_path)
    _, kwds = ctx.calls[0]
    assert kwds['overlap'] == hi.DEFAULT_BAND_CELLS
    assert kwds['max_iters'] == hi.DEFAULT_MAX_SWEEPS
    assert kwds['tol'] == hi.DEFAULT_TOL


# --------------------------------------------------------------------------
# Key hygiene: sweep-0 outputs must be preserved
# --------------------------------------------------------------------------

def test_outputs_do_not_collide_with_sweep0_keys(tmp_path):
    ctx, prof, _, (near_out, far_out) = _run(tmp_path)
    sweep0 = {hi._key(prof, 'potential', leaf)
              for leaf in ('near', 'coarse')} | {
              hi._key(prof, 'initial', 'near_refined'),
              hi._key(prof, 'initial', 'near_bc'),
              hi._key(prof, 'boundary', 'near_bc')}
    assert near_out not in sweep0
    assert far_out not in sweep0
    # ...and the sweep-0 near potential is only ever an INPUT here.
    _, kwds = ctx.calls[0]
    assert kwds['near_potential'] in sweep0


def test_weighting_profile_keeps_the_w_prefix(tmp_path):
    ctx, prof, near_pot, (near_out, far_out) = _run(tmp_path, field='weighting')
    assert near_pot == 'potential/w_near'
    assert near_out == 'potential/w_near_schwarz'
    assert far_out == 'potential/w_coarse_schwarz'
    _, kwds = ctx.calls[0]
    assert kwds['coarse_potential'] == 'potential/w_coarse'
    assert kwds['insulator'] == 'initial/w_near_insulator'
    assert kwds['edges'] == 'fix,fix,fix'


# --------------------------------------------------------------------------
# Resume guard (_want) edge cases
# --------------------------------------------------------------------------

def test_resume_skips_when_both_outputs_exist(tmp_path):
    ctx = FakeCtx(tmp_path)
    prof = hi._profile('drift')
    ctx.place('potential/near_schwarz', 'potential/coarse_schwarz')
    out = hi._schwarz(ctx, prof, hi._key(prof, 'potential', 'near'), '20*mm')
    assert ctx.calls == []
    assert out == ('potential/near_schwarz', 'potential/coarse_schwarz')


def test_resume_reruns_when_only_one_output_exists(tmp_path):
    '''A half-finished sweep must NOT be treated as done -- this is the whole
    point of the far solver having its own checkpoint key.'''
    ctx = FakeCtx(tmp_path)
    prof = hi._profile('drift')
    ctx.place('potential/near_schwarz')
    hi._schwarz(ctx, prof, hi._key(prof, 'potential', 'near'), '20*mm')
    assert len(ctx.calls) == 1


def test_missing_output_fails_loudly(tmp_path):
    ctx = FakeCtx(tmp_path, produce=False)
    prof = hi._profile('drift')
    with pytest.raises(RuntimeError, match='did not produce'):
        hi._schwarz(ctx, prof, hi._key(prof, 'potential', 'near'), '20*mm')


def test_interface_is_required_and_positional(tmp_path):
    """Commit 5f0785d removed the '20*mm' default: it disagreed with the
    runner's --interface 40*mm, so a forgotten argument silently banded the
    sweep at the wrong plane."""
    sig = inspect.signature(hi._schwarz)
    par = sig.parameters['interface']
    assert par.default is inspect.Parameter.empty
    assert list(sig.parameters).index('interface') == 3

    ctx = FakeCtx(tmp_path)
    prof = hi._profile('drift')
    with pytest.raises(TypeError):
        hi._schwarz(ctx, prof, hi._key(prof, 'potential', 'near'))


def test_interface_is_passed_through_verbatim(tmp_path):
    ctx, _, _, _ = _run(tmp_path, interface='40*mm')
    _, kwds = ctx.calls[0]
    assert kwds['interface'] == '40*mm'
