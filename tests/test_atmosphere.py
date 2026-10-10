"""Tests for AtmosphericState and Guillot T-P profile."""

import numpy as np
import pytest

from plato_fno.physics.atmosphere import (
    N_PARAMS,
    AtmosphericState,
    guillot_temperature_profile,
)


def _make_state(seed: int = 0) -> AtmosphericState:
    rng = np.random.default_rng(seed)
    return AtmosphericState.sample_prior(rng)


def test_n_params_matches_prior_bounds():
    assert len(AtmosphericState.PRIOR_BOUNDS) == N_PARAMS


def test_to_vector_shape():
    state = _make_state()
    v = state.to_vector()
    assert v.shape == (N_PARAMS,)
    assert v.dtype == np.float32


def test_from_vector_roundtrip():
    state = _make_state(42)
    v = state.to_vector()
    reconstructed = AtmosphericState.from_vector(v)
    v2 = reconstructed.to_vector()
    np.testing.assert_allclose(v, v2, atol=1e-6)


def test_sample_prior_within_bounds():
    rng = np.random.default_rng(7)
    for _ in range(50):
        state = AtmosphericState.sample_prior(rng)
        assert state.in_prior(), "Sampled state is outside prior bounds"


def test_in_prior_out_of_bounds():
    state = _make_state()
    # Force T_int above its upper bound
    state.T_int = 9999.0
    assert not state.in_prior()


def test_guillot_profile_shape():
    pressures = np.logspace(2, -6, 60)
    T = guillot_temperature_profile(
        pressures, T_int=200.0, T_eq=1500.0, kappa_IR=0.01, gamma=0.4, gravity=1e3
    )
    assert T.shape == (60,)


def test_guillot_profile_positive():
    pressures = np.logspace(2, -6, 60)
    T = guillot_temperature_profile(
        pressures, T_int=200.0, T_eq=1500.0, kappa_IR=0.01, gamma=0.4, gravity=1e3
    )
    assert (T > 0).all(), "Temperatures should be positive"


def test_guillot_profile_physical_range():
    pressures = np.logspace(2, -6, 60)
    T = guillot_temperature_profile(
        pressures, T_int=200.0, T_eq=1500.0, kappa_IR=0.01, gamma=0.4, gravity=1e3
    )
    # Should be between ~50 K and ~5000 K for physical inputs
    assert T.min() > 50.0
    assert T.max() < 10000.0


def test_sample_prior_respects_transit_depth_constraint():
    from plato_fno.physics.atmosphere import MAX_TRANSIT_DEPTH, R_JUP_CM, R_SUN_CM

    rng = np.random.default_rng(1)
    for _ in range(500):
        s = AtmosphericState.sample_prior(rng)
        assert (s.R_planet * R_JUP_CM / (s.R_star * R_SUN_CM)) ** 2 < MAX_TRANSIT_DEPTH


def test_guillot_matches_petitradtrans():
    physics = pytest.importorskip("petitRADTRANS.physics")
    pressures = np.logspace(-6, 2, 60)
    for g in (300.0, 1e3, 3e4):
        ours = guillot_temperature_profile(
            pressures, T_int=200.0, T_eq=1500.0, kappa_IR=0.01, gamma=0.4, gravity=g
        )
        ref = physics.temperature_profile_function_guillot_global(
            pressures, 0.01, 0.4, g, 200.0, 1500.0
        )
        np.testing.assert_allclose(ours, ref, rtol=1e-10)


def test_guillot_depends_on_gravity():
    pressures = np.logspace(-6, 2, 60)
    kw = dict(T_int=200.0, T_eq=1500.0, kappa_IR=0.01, gamma=0.4)
    low_g = guillot_temperature_profile(pressures, gravity=300.0, **kw)
    high_g = guillot_temperature_profile(pressures, gravity=3e4, **kw)
    assert not np.allclose(low_g, high_g)
