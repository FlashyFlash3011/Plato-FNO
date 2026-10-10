"""Tests for MCMCRetrieval (no GPU needed, uses CPU)."""

import numpy as np
import pytest
import torch

from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.models.retrieval import MCMCRetrieval, convergence_diagnostics
from plato_fno.models.surrogate import AtmosphereSurrogate
from plato_fno.physics.atmosphere import N_PARAMS, PARAM_NAMES, AtmosphericState

N_WAVELENGTHS = 64
N_ENC         = 16


@pytest.fixture
def surrogate():
    torch.manual_seed(0)
    model = AtmosphereSurrogate(
        n_wavelengths=N_WAVELENGTHS,
        n_enc=N_ENC,
        hidden_channels=32,
        n_layers=2,
        modes=8,
    ).eval()
    return model


@pytest.fixture
def normalizer():
    rng = np.random.default_rng(0)
    params  = rng.uniform(-1, 1, (200, N_PARAMS)).astype(np.float32)
    spectra = rng.uniform(0.01, 0.02, (200, 1, N_WAVELENGTHS)).astype(np.float32)
    return SpectrumNormalizer().fit(params, spectra)


@pytest.fixture
def retrieval(surrogate, normalizer):
    return MCMCRetrieval(surrogate=surrogate, normalizer=normalizer, device="cpu")


@pytest.fixture
def observed():
    rng = np.random.default_rng(1)
    return rng.uniform(0.01, 0.02, N_WAVELENGTHS).astype(np.float32)


def test_log_prior_in_bounds(retrieval):
    rng = np.random.default_rng(0)
    state = AtmosphericState.sample_prior(rng)
    theta = state.to_vector()
    lp = retrieval.log_prior(theta)
    assert np.isfinite(lp)
    assert lp == 0.0  # uniform prior inside bounds


def test_log_prior_out_of_bounds(retrieval):
    theta = np.zeros(N_PARAMS, dtype=np.float32)
    theta[0] = -9999.0  # T_int way out of bounds
    lp = retrieval.log_prior(theta)
    assert lp == -np.inf


def test_log_likelihood_batch_shape(retrieval, observed):
    rng = np.random.default_rng(0)
    theta_batch = np.stack([
        AtmosphericState.sample_prior(rng).to_vector() for _ in range(8)
    ])
    log_liks = retrieval.log_likelihood_batch(theta_batch, observed, sigma=50e-6)
    assert log_liks.shape == (8,)
    assert np.isfinite(log_liks).all()


def test_log_posterior_vectorized_shape(retrieval, observed):
    rng = np.random.default_rng(0)
    theta_batch = np.stack([
        AtmosphericState.sample_prior(rng).to_vector() for _ in range(16)
    ])
    log_post = retrieval.log_posterior_vectorized(theta_batch, observed, sigma=50e-6)
    assert log_post.shape == (16,)


def test_log_posterior_oob_walkers(retrieval, observed):
    """Walkers outside prior should get -inf without crashing."""
    theta_batch = np.zeros((4, N_PARAMS), dtype=np.float32)
    theta_batch[:, 0] = -9999.0  # all out of bounds
    log_post = retrieval.log_posterior_vectorized(theta_batch, observed, sigma=50e-6)
    assert (log_post == -np.inf).all()


@pytest.mark.slow
def test_mcmc_runs_and_shape(retrieval, observed):
    flat_samples, sampler = retrieval.run(
        observed=observed,
        sigma=50e-6,
        n_walkers=2 * N_PARAMS,   # minimum valid
        n_steps=20,
        burn_in=5,
        thin=1,
        progress=False,
    )
    assert flat_samples.ndim == 2
    assert flat_samples.shape[1] == N_PARAMS
    assert flat_samples.shape[0] > 0


def test_log_prior_vectorised_matches_scalar(retrieval):
    rng = np.random.default_rng(0)
    batch = np.stack([AtmosphericState.sample_prior(rng).to_vector() for _ in range(5)])
    batch[2, 0] = -9999.0
    lp = retrieval.log_prior(batch)
    assert lp.shape == (5,)
    for i in range(5):
        assert lp[i] == retrieval.log_prior(batch[i])


def test_log_prior_rejects_planet_too_large_for_star(retrieval):
    theta = AtmosphericState.sample_prior(np.random.default_rng(0)).to_vector()
    theta[PARAM_NAMES.index("R_planet")] = 2.0  # R_jup
    theta[PARAM_NAMES.index("R_star")] = 0.5  # R_sun → depth ≈ 0.17
    assert retrieval.log_prior(theta) == -np.inf


def test_per_point_sigma_matches_scalar(retrieval, observed):
    theta = np.stack([AtmosphericState.sample_prior(np.random.default_rng(3)).to_vector()])
    a = retrieval.log_likelihood_batch(theta, observed, sigma=1e-4)
    b = retrieval.log_likelihood_batch(theta, observed, sigma=np.full(N_WAVELENGTHS, 1e-4))
    np.testing.assert_allclose(a, b)


def test_model_sigma_widens_likelihood(surrogate, normalizer, observed):
    theta = np.stack([AtmosphericState.sample_prior(np.random.default_rng(3)).to_vector()])
    plain = MCMCRetrieval(surrogate, normalizer)
    widened = MCMCRetrieval(surrogate, normalizer, model_sigma=np.full(N_WAVELENGTHS, 1e-3))
    # observed is far from this random model's prediction, so extra variance helps
    assert widened.log_likelihood_batch(theta, observed, 1e-5) > plain.log_likelihood_batch(
        theta, observed, 1e-5
    )


def test_obs_transform_and_grid_mismatch(surrogate, normalizer, observed):
    theta = np.stack([AtmosphericState.sample_prior(np.random.default_rng(3)).to_vector()])
    pairs = MCMCRetrieval(surrogate, normalizer, obs_transform=lambda p: p.reshape(len(p), -1, 2).mean(-1))
    ll = pairs.log_likelihood_batch(theta, observed[: N_WAVELENGTHS // 2], 1e-4)
    assert ll.shape == (1,)
    with pytest.raises(ValueError, match="obs_transform"):
        pairs.log_likelihood_batch(theta, observed, 1e-4)


def test_initial_walkers_from_prior_are_valid_and_spread(retrieval):
    p0 = retrieval.initial_walkers(32, np.random.default_rng(0))
    assert p0.shape == (32, N_PARAMS)
    assert np.isfinite(retrieval.log_prior(p0)).all()
    assert (p0.std(axis=0) > 0).all()


@pytest.mark.slow
def test_run_is_reproducible_with_seed(retrieval, observed):
    kwargs = dict(observed=observed, sigma=1e-3, n_walkers=2 * N_PARAMS, n_steps=15,
                  burn_in=5, thin=1, progress=False, seed=7)
    a, sampler = retrieval.run(**kwargs)
    b, _ = retrieval.run(**kwargs)
    np.testing.assert_array_equal(a, b)
    diag = convergence_diagnostics(sampler, burn_in=5)
    assert set(diag) >= {"tau", "max_tau", "n_independent", "acceptance", "converged"}
    assert diag["converged"] is False  # 10 kept steps can never be enough
