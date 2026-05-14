#!/usr/bin/env pytest
"""Validate stencil_poisson_harmonic with uniform epsilon=1 matches stencil_poisson.

With epsilon uniform, harmonic mean of equal values equals the value itself,
and the denominator reduces to 2*N (same as the standard stencil), so the
two stencils must agree to numerical precision.
"""

import numpy as np
import pytest

from pochoir.fdm_generic import stencil_poisson, stencil_poisson_harmonic


def _random_phi(shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


@pytest.mark.parametrize("shape", [
    (7,),
    (5, 6),
    (5, 6, 8),
])
def test_harmonic_uniform_eps1_matches_stencil_poisson(shape):
    phi = _random_phi(shape)
    eps = np.ones(shape)
    got = stencil_poisson_harmonic(phi, eps)
    expected = stencil_poisson(phi)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_harmonic_uses_provided_res_buffer():
    phi = _random_phi((5, 6))
    eps = np.ones((5, 6))
    res = np.full((3, 4), 99.0)
    stencil_poisson_harmonic(phi, eps, res=res)
    expected = stencil_poisson(phi)
    np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)


def test_harmonic_preserves_input_arrays():
    phi = _random_phi((5, 6))
    eps = np.ones((5, 6))
    phi_before = phi.copy()
    eps_before = eps.copy()
    stencil_poisson_harmonic(phi, eps)
    np.testing.assert_array_equal(phi, phi_before)
    np.testing.assert_array_equal(eps, eps_before)
