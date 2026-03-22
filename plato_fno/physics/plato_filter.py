"""PLATO mission bandpass integration and noise simulation."""

from __future__ import annotations

import numpy as np

# PLATO photometric channel bandpasses [microns]
PLATO_BANDPASSES: dict[str, tuple[float, float]] = {
    "blue":  (0.50, 0.65),
    "green": (0.65, 0.75),
    "red":   (0.75, 0.90),
    "nir":   (0.90, 1.05),
}


def integrate_spectrum_to_photometry(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    bandpasses: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Integrate a transit depth spectrum over PLATO bandpasses.

    Parameters
    ----------
    spectrum : np.ndarray
        Transit depth (Rp/Rs)^2 as a function of wavelength, shape (n_wavelengths,).
    wavelengths : np.ndarray
        Wavelength grid in microns, shape (n_wavelengths,).
    bandpasses : dict, optional
        Map of band name → (lambda_min, lambda_max) in microns.
        Defaults to PLATO_BANDPASSES.

    Returns
    -------
    dict[str, float]
        Band-integrated transit depth per channel.
    """
    if bandpasses is None:
        bandpasses = PLATO_BANDPASSES

    result: dict[str, float] = {}
    for band, (lam_min, lam_max) in bandpasses.items():
        mask = (wavelengths >= lam_min) & (wavelengths <= lam_max)
        if mask.sum() < 2:
            result[band] = float("nan")
        else:
            result[band] = float(np.trapz(spectrum[mask], wavelengths[mask]) / (lam_max - lam_min))
    return result


def add_plato_noise(
    photometry: dict[str, float],
    snr_per_band: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Add Gaussian photon noise at the specified SNR per band.

    Parameters
    ----------
    photometry : dict[str, float]
        Noiseless band-integrated transit depths.
    snr_per_band : dict[str, float]
        Signal-to-noise ratio for each band. The noise sigma = value / SNR.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict[str, float]
        Noisy photometric measurements.
    """
    noisy: dict[str, float] = {}
    for band, value in photometry.items():
        snr = snr_per_band.get(band, 100.0)
        sigma = abs(value) / snr
        noisy[band] = value + rng.normal(0.0, sigma)
    return noisy


def photometry_to_array(photometry: dict[str, float]) -> np.ndarray:
    """Convert photometry dict to ordered float32 array (blue, green, red, nir)."""
    return np.array(
        [photometry[b] for b in ["blue", "green", "red", "nir"]],
        dtype=np.float32,
    )
