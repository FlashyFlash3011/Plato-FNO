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


BANDS = {"ncam", "fcam_blue", "fcam_red"}


def test_bandpass_keys():
    assert set(PLATO_BANDPASSES.keys()) == BANDS


def test_integrate_constant_spectrum():
    """A constant spectrum averages to that constant in every band, on any grid."""
    for grid in (WAVELENGTHS, np.linspace(0.3, 5.0, 300)):
        spectrum = np.full(len(grid), 0.015)
        for band, val in integrate_spectrum_to_photometry(spectrum, grid).items():
            np.testing.assert_allclose(val, 0.015, rtol=1e-12, err_msg=f"Band {band} failed")


def test_integrate_returns_all_bands():
    spectrum = np.ones(N_WAVELENGTHS)
    result   = integrate_spectrum_to_photometry(spectrum, WAVELENGTHS)
    assert set(result.keys()) == BANDS


def test_integrate_custom_bandpasses():
    spectrum = np.ones(N_WAVELENGTHS) * 2.0
    custom   = {"full": (0.3, 5.0)}
    result   = integrate_spectrum_to_photometry(spectrum, WAVELENGTHS, bandpasses=custom)
    np.testing.assert_allclose(result["full"], 2.0, rtol=1e-3)


def test_add_plato_noise_changes_values():
    photometry = {"ncam": 0.01, "fcam_blue": 0.011, "fcam_red": 0.012}
    snr = {b: 100.0 for b in photometry}
    noisy = add_plato_noise(photometry, snr, np.random.default_rng(0))
    assert noisy != photometry


def test_photometry_to_array_order():
    photometry = {"fcam_red": 3.0, "ncam": 1.0, "fcam_blue": 2.0}
    np.testing.assert_array_equal(photometry_to_array(photometry), [1.0, 2.0, 3.0])
