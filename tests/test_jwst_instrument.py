"""Tests for JWST binning and noise simulation."""

import numpy as np
import pytest

from plato_fno.physics.jwst_instrument import (
    JWST_INSTRUMENTS,
    bin_spectrum,
    binning_matrix,
    make_bin_transform,
    simulate_jwst_observation,
)

WAVELENGTHS = np.linspace(0.3, 5.0, 300)


def test_binning_matrix_rows_are_averages():
    for lam_min, lam_max, n_bins in JWST_INSTRUMENTS.values():
        W = binning_matrix(WAVELENGTHS, lam_min, lam_max, n_bins)
        assert W.shape == (n_bins, len(WAVELENGTHS))
        np.testing.assert_allclose(W.sum(axis=1), 1.0)


def test_bin_constant_spectrum_is_constant():
    binned, centers = bin_spectrum(np.full(300, 0.01), WAVELENGTHS, 0.6, 2.8, 40)
    np.testing.assert_allclose(binned, 0.01, rtol=1e-6)
    assert centers.shape == (40,) and np.all(np.diff(centers) > 0)


def test_bin_transform_matches_bin_spectrum():
    rng = np.random.default_rng(0)
    batch = rng.uniform(0.01, 0.02, (5, 300))
    out = make_bin_transform(WAVELENGTHS, "nirspec_prism")(batch)
    assert out.shape == (5, JWST_INSTRUMENTS["nirspec_prism"][2])
    expected, _ = bin_spectrum(batch[3], WAVELENGTHS, *JWST_INSTRUMENTS["nirspec_prism"])
    np.testing.assert_allclose(out[3], expected, rtol=1e-5)


def test_unknown_instrument_raises():
    with pytest.raises(ValueError, match="Unknown instrument"):
        simulate_jwst_observation(np.full(300, 0.01), WAVELENGTHS, instrument="hubble")


def test_simulated_noise_matches_reported_sigma():
    rng = np.random.default_rng(0)
    spectrum = np.full(300, 0.01)
    draws, sigma = [], None
    for _ in range(2000):
        obs, _, sigma = simulate_jwst_observation(spectrum, WAVELENGTHS, "niriss_soss", rng=rng)
        draws.append(obs)
    np.testing.assert_allclose(np.std(draws, axis=0), sigma, rtol=0.1)


def test_noise_does_not_scale_with_depth_but_with_magnitude():
    _, _, s_shallow = simulate_jwst_observation(np.full(300, 0.001), WAVELENGTHS, "nirspec_prism")
    _, _, s_deep = simulate_jwst_observation(np.full(300, 0.05), WAVELENGTHS, "nirspec_prism")
    np.testing.assert_allclose(s_shallow, s_deep)
    _, _, s_faint = simulate_jwst_observation(
        np.full(300, 0.01), WAVELENGTHS, "nirspec_prism", stellar_vmag=15.0
    )
    np.testing.assert_allclose(s_faint / s_deep, 10.0, rtol=1e-6)  # 5 mag → 10× noise
