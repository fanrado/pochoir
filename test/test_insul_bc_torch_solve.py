#!/usr/bin/env pytest
'''
Tests for the torch solver's optional `insulator` no-flux enable signal.

NOTE (e15x P2b, commit 00d3f1e): the original pre-P2b insulator contract
tested here -- _compiled_step(insulator=, coeff=) params and a FREEZE
reflecting-wall for an arbitrary insulator slab -- was removed by design.
_compiled_step now branches on `masks` (not insulator), and solve() uses
the insulator arg only as an ENABLE signal: when set, the pad-plane
interface is auto-derived from barr (padplane_noflux_geom) and node-centered
pad-plane no-flux is applied each step. The removed contract's tests were
retired (see test/test_insul_bc_torch_solve.py history / debug pochoir-ltzk);
the full pad-plane no-flux contract is covered in
test/test_padplane_noflux_solver.py.

What remains here: insulator=None keeps the solve byte-identical to plain
Laplace, and the solve.insulator signature default.

The solve() body calls torch.cuda.synchronize() unconditionally, so the
end-to-end tests require a CUDA device; the signature tests do not.
'''

import inspect

import numpy
import pytest

import pochoir.fdm_torch as ft

try:
    import torch
    _HAVE_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAVE_CUDA = False

needs_cuda = pytest.mark.skipif(not _HAVE_CUDA,
                                reason="fdm_torch.solve calls torch.cuda.synchronize()")

PER = [True, True, False]     # periodic transverse, fixed in z
NOOP = lambda *a, **k: None


def _plates(nx=8, ny=8, nz=12, v0=0.0, v1=10.0):
    '''Two fixed potential plates at z=0 and z=nz-1, free interior.'''
    iarr = numpy.zeros((nx, ny, nz))
    barr = numpy.zeros((nx, ny, nz), dtype=bool)
    iarr[:, :, 0] = v0
    iarr[:, :, -1] = v1
    barr[:, :, 0] = True
    barr[:, :, -1] = True
    return iarr, barr


# --------------------------------------------------------------------------
# Signatures: the new optional params exist and default to None (static branch)
# --------------------------------------------------------------------------

def test_solve_has_insulator_param_defaulting_none():
    sig = inspect.signature(ft.solve)
    assert "insulator" in sig.parameters
    assert sig.parameters["insulator"].default is None


# --------------------------------------------------------------------------
# Regression: insulator=None keeps the default plain-Laplace solve
# --------------------------------------------------------------------------

@needs_cuda
def test_none_path_solves_laplace_linear_ramp():
    # Laplace between two plates with periodic transverse BCs -> linear in z.
    iarr, barr = _plates(nz=12, v0=0.0, v1=10.0)
    sol, _err = ft.solve(iarr, barr, PER, prec=1e-10, epoch=1500, nepochs=8,
                         info_msg=NOOP)
    zprof = sol.numpy()[0, 0, :]
    assert numpy.allclose(zprof, numpy.linspace(0.0, 10.0, 12), atol=1e-3)


@needs_cuda
def test_explicit_none_is_byte_identical_to_default():
    # The critical regression: passing insulator=None explicitly must give a
    # bit-for-bit identical result to omitting it.
    iarr, barr = _plates()
    sol_default, _ = ft.solve(iarr, barr, PER, prec=1e-10, epoch=1500,
                              nepochs=4, info_msg=NOOP)
    sol_none, _ = ft.solve(iarr, barr, PER, prec=1e-10, epoch=1500,
                           nepochs=4, info_msg=NOOP, insulator=None)
    assert numpy.array_equal(sol_default.numpy(), sol_none.numpy())


# --------------------------------------------------------------------------
# The pre-P2b insulator path (FREEZE reflecting wall + _compiled_step
# insulator/coeff params) was removed by design in commit 00d3f1e (P2b);
# its tests were retired here.  The node-centered pad-plane no-flux contract
# that replaced it is covered in test/test_padplane_noflux_solver.py.
# --------------------------------------------------------------------------
