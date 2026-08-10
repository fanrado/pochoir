#!/usr/bin/env pytest
'''
Tests for commit 5f0785d: hybrid_iterate() runs the Schwarz sweep and stitches
its outputs.

What must hold:
  * hybrid_iterate gains band_cells/max_sweeps/tol, defaulting to the module
    constants, and forwards them (plus interface and precision) to _schwarz;
  * the stitch consumes the SWEEP outputs (potential/near_schwarz +
    potential/coarse_schwarz), not the sweep-0 keys;
  * max_sweeps=0 skips the sweep entirely and stitches the sweep-0
    near/coarse keys -- the one-shot scheme, exactly;
  * _stitch takes near and coarse keys as arguments and passes them to
    stitch-near, with the output key unchanged (drift3d / weight3d).

The heavy steps (_domains/_generate/_solve/_near_solve) are stubbed: this is a
wiring test, not a solve.
'''

import inspect

import numpy
import pytest

import pochoir.hybrid_iterate as hi


SHAPES = {'coarse': '11,11,151', 'near': '44,44,201', 'fine01': '44,44,601'}


class FakeCtx:
    def __init__(self, tmp_path):
        self.calls = []

        class _Obj:
            instore_path = tmp_path

        self.obj = _Obj()

    def invoke(self, cmd, **kwds):
        self.calls.append((getattr(cmd, 'name', getattr(cmd, '__name__', '?')),
                           kwds))
        for key in (kwds.get('output'), kwds.get('near_out'),
                    kwds.get('far_out')):
            if key:
                path = self.obj.instore_path / (key + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(path, arr=numpy.zeros(2))


@pytest.fixture
def driver(monkeypatch, tmp_path):
    '''Run hybrid_iterate with the heavy steps stubbed; return the recorder.'''
    rec = dict(schwarz=[], stitch=[])

    monkeypatch.setattr(hi, '_domains', lambda *a, **k: None)
    monkeypatch.setattr(hi, '_generate', lambda *a, **k: None)
    monkeypatch.setattr(hi, '_solve', lambda *a, **k: None)
    monkeypatch.setattr(hi, '_check_interface', lambda *a, **k: None)
    monkeypatch.setattr(
        hi, '_near_solve',
        lambda ctx, prof, precision, log: hi._key(prof, 'potential', 'near'))

    def fake_schwarz(ctx, prof, near_pot, interface, **kwds):
        rec['schwarz'].append(dict(near_pot=near_pot, interface=interface,
                                   **kwds))
        return (hi._key(prof, 'potential', 'near_schwarz'),
                hi._key(prof, 'potential', 'coarse_schwarz'))

    def fake_stitch(ctx, prof, near_pot, coarse_pot, log):
        rec['stitch'].append(dict(near=near_pot, coarse=coarse_pot))
        return hi._output_key(prof)

    monkeypatch.setattr(hi, '_schwarz', fake_schwarz)
    monkeypatch.setattr(hi, '_stitch', fake_stitch)

    def run(**kwds):
        ctx = FakeCtx(tmp_path)
        kwds.setdefault('log', lambda msg: None)
        final, grids = hi.hybrid_iterate(
            ctx, 'coarse.json', 'fine.json', derive_domain=False,
            shapes=SHAPES, **kwds)
        return final, grids, rec

    return run


# --------------------------------------------------------------------------
# Option surface
# --------------------------------------------------------------------------

def test_driver_signature_gains_the_schwarz_knobs():
    sig = inspect.signature(hi.hybrid_iterate)
    assert sig.parameters['band_cells'].default == hi.DEFAULT_BAND_CELLS
    assert sig.parameters['max_sweeps'].default == hi.DEFAULT_MAX_SWEEPS
    assert sig.parameters['tol'].default == hi.DEFAULT_TOL


# --------------------------------------------------------------------------
# Happy path: sweep runs and its outputs are stitched
# --------------------------------------------------------------------------

def test_sweep_runs_by_default_and_its_outputs_are_stitched(driver):
    final, _, rec = driver(interface='20*mm', precision=2e-8)
    assert len(rec['schwarz']) == 1
    assert len(rec['stitch']) == 1
    assert rec['stitch'][0] == dict(near='potential/near_schwarz',
                                    coarse='potential/coarse_schwarz')
    assert final == 'potential/drift3d'


def test_driver_forwards_every_schwarz_argument(driver):
    _, _, rec = driver(interface='40*mm', precision=1e-9, band_cells=3,
                       max_sweeps=2, tol='0.05*V')
    call = rec['schwarz'][-1]
    assert call['interface'] == '40*mm'          # NOT the old '20*mm' default
    assert call['precision'] == 1e-9
    assert call['band_cells'] == 3
    assert call['max_sweeps'] == 2
    assert call['tol'] == '0.05*V'
    assert call['near_pot'] == 'potential/near'  # the sweep-0 warm start


def test_defaults_reach_schwarz_when_not_given(driver):
    _, _, rec = driver(interface='20*mm')
    call = rec['schwarz'][-1]
    assert call['band_cells'] == hi.DEFAULT_BAND_CELLS
    assert call['max_sweeps'] == hi.DEFAULT_MAX_SWEEPS
    assert call['tol'] == hi.DEFAULT_TOL


def test_weighting_profile_stitches_its_own_sweep_keys(driver):
    final, _, rec = driver(interface='20*mm', field='weighting')
    assert rec['stitch'][-1] == dict(near='potential/w_near_schwarz',
                                     coarse='potential/w_coarse_schwarz')
    assert final == 'potential/weight3d'


# --------------------------------------------------------------------------
# max_sweeps == 0: exact one-shot fallback
# --------------------------------------------------------------------------

def test_max_sweeps_zero_skips_the_sweep_and_stitches_sweep0_keys(driver):
    final, _, rec = driver(interface='20*mm', max_sweeps=0)
    assert rec['schwarz'] == []                 # sweep never ran
    assert rec['stitch'][-1] == dict(near='potential/near',
                                     coarse='potential/coarse')
    assert final == 'potential/drift3d'


def test_max_sweeps_zero_logs_the_fallback(driver):
    msgs = []
    driver(interface='20*mm', max_sweeps=0, log=msgs.append)
    assert any('max_sweeps=0' in m for m in msgs)


# --------------------------------------------------------------------------
# _stitch now takes the far key as an argument
# --------------------------------------------------------------------------

def test_stitch_signature_takes_near_and_coarse():
    params = list(inspect.signature(hi._stitch).parameters)
    assert params == ['ctx', 'prof', 'near_pot', 'coarse_pot', 'log']


def test_stitch_passes_the_given_keys_to_stitch_near(tmp_path):
    ctx = FakeCtx(tmp_path)
    prof = hi._profile('drift')
    out = hi._stitch(ctx, prof, 'potential/near_schwarz',
                     'potential/coarse_schwarz', lambda msg: None)
    stitches = [kwds for name, kwds in ctx.calls if 'near' in kwds]
    assert stitches, ctx.calls
    kwds = stitches[-1]
    assert kwds['near'] == 'potential/near_schwarz'
    assert kwds['coarse'] == 'potential/coarse_schwarz'
    assert kwds['domain'] == 'domain/drift3d'
    assert kwds['axis'] == 2
    # The output key is unchanged by the Schwarz work.
    assert out == kwds['output'] == 'potential/drift3d'


def test_stitch_no_longer_hardcodes_potential_coarse():
    src = inspect.getsource(hi._stitch)
    assert "coarse=coarse_pot" in src
    assert "_key(prof, 'potential', 'coarse')" not in src
