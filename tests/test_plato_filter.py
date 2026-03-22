"""Tests for PLATO bandpass integration."""

import numpy as np
import pytest

from plato_fno.physics.plato_filter import (
    PLATO_BANDPASSES,
    add_plato_noise,
    integrate_spectrum_to_photometry,
    photometry_to_array,
)

# Use a dense grid so band edges are well-sampled by trapezoid integration
N_WAVELENGTHS = 5000
WAVELENGTHS   = np.linspace(0.3, 5.0, N_WAVELENGTHS)


def test_bandpass_keys():
    assert set(PLATO_BANDPASSES.keys()) == {"blue", "green", "red", "nir"}


def test_integrate_constant_spectrum():
    """Integrating a constant spectrum should return that constant for all bands.

    Uses a dense wavelength grid so band edges are well-sampled.
    """
    spectrum = np.ones(N_WAVELENGTHS, dtype=np.float32) * 0.015
    result   = integrate_spectrum_to_photometry(spectrum, WAVELENGTHS)
    for band, val in result.items():
        if not np.isnan(val):
            # 2% tolerance: trapezoid integration on discrete grid has endpoint-clipping
            # errors proportional to grid spacing / band width; worst case is narrow bands
            np.testing.assert_allclose(val, 0.015, rtol=2e-2, err_msg=f"Band {band} failed")


def test_integrate_returns_all_bands():
    spectrum = np.ones(N_WAVELENGTHS)
    result   = integrate_spectrum_to_photometry(spectrum, WAVELENGTHS)
    assert set(result.keys()) == {"blue", "green", "red", "nir"}


def test_integrate_custom_bandpasses():
    spectrum = np.ones(N_WAVELENGTHS) * 2.0
    custom   = {"full": (0.3, 5.0)}
    result   = integrate_spectrum_to_photometry(spectrum, WAVELENGTHS, bandpasses=custom)
    np.testing.assert_allclose(result["full"], 2.0, rtol=1e-3)


def test_add_plato_noise_changes_values():
    photometry = {"blue": 0.01, "green": 0.011, "red": 0.012, "nir": 0.013}
    snr        = {b: 100.0 for b in photometry}
    rng        = np.random.default_rng(0)
    noisy      = add_plato_noise(photometry, snr, rng)
    assert noisy != photometry


def test_add_plato_noise_reasonable_magnitude():
    photometry = {"blue": 0.01, "green": 0.011, "red": 0.012, "nir": 0.013}
    snr        = {b: 1000.0 for b in photometry}  # high SNR → small noise
    rng        = np.random.default_rng(0)
    noisy      = add_plato_noise(photometry, snr, rng)
    for band in photometry:
        rel_err = abs(noisy[band] - photometry[band]) / photometry[band]
        assert rel_err < 0.05, f"Noise too large for {band}: {rel_err:.3f}"


def test_photometry_to_array_order():
    photometry = {"blue": 1.0, "green": 2.0, "red": 3.0, "nir": 4.0}
    arr = photometry_to_array(photometry)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])
