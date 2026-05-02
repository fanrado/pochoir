#!/usr/bin/env pytest
"""``stencil_poisson`` performs the Jacobi/Gauss-Seidel update for ∇²φ = -f.

PLAN_for_production.md §1, fourth bullet — kept (used by ``fdm_torch.solve``);
verify it matches a hand-computed reference and that the duplicate
``stencil`` definition has been collapsed to a single working implementation.
"""

import inspect

import numpy as np
import pytest

from pochoir.fdm_generic import stencil, stencil_poisson


def _hand_reference_1d(arr, src_core, spacing):
    """One Jacobi step in 1D for ∇²φ = -f.

    Interior cell update:  φ_new[i] = (φ[i-1] + φ[i+1]) / 2 - h² * f[i] / 2
    ``src_core`` has shape (N-2,) — interior cells only.
    """
    out = np.zeros(arr.shape[0] - 2, dtype=arr.dtype)
    for i in range(1, arr.shape[0] - 1):
        out[i - 1] = 0.5 * (arr[i - 1] + arr[i + 1]) - 0.5 * spacing ** 2 * src_core[i - 1]
    return out


def test_stencil_poisson_matches_hand_reference_1d():
    arr = np.array([0.0, 1.0, 2.0, 5.0, 11.0, 4.0, 0.0])
    # Core (interior) shape is (5,) — 7 - 2.
    src_core = np.array([0.5, -1.0, 2.0, 1.0, -0.5])
    h = 0.25

    got = stencil_poisson(arr.copy(), source=src_core, spacing=h)
    expect = _hand_reference_1d(arr, src_core, h)

    np.testing.assert_allclose(got, expect, rtol=0, atol=1e-12)


def test_stencil_poisson_no_source_matches_plain_stencil():
    """With source=None the Poisson update reduces to the plain neighbour mean."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((5, 6, 4))
    np.testing.assert_allclose(
        stencil_poisson(arr.copy()),
        stencil(arr.copy()),
        rtol=0, atol=1e-12,
    )


def test_stencil_poisson_uses_provided_res_buffer():
    arr = np.zeros((4, 4))
    arr[1:3, 1:3] = 1.0
    res = np.full((2, 2), 99.0)
    stencil_poisson(arr, res=res)
    # Pre-existing values must have been zeroed; result must equal the
    # neighbour-average update (source=None).
    expected = stencil(arr.copy())
    np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)


def test_stencil_poisson_preserves_input_array():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    src_core = np.array([1.0, 1.0, 1.0])  # interior shape = N - 2
    arr_before = arr.copy()
    stencil_poisson(arr, source=src_core, spacing=0.5)
    np.testing.assert_array_equal(arr, arr_before)


def test_only_one_stencil_definition_in_module():
    """Guard against the prior duplicate ``stencil`` (one missing ``return``)."""
    import pochoir.fdm_generic as m
    src = inspect.getsource(m)
    # Count top-level ``def stencil(`` occurrences.
    count = sum(
        1 for line in src.splitlines()
        if line.startswith("def stencil(")
    )
    assert count == 1, f"expected one top-level 'def stencil(' got {count}"
