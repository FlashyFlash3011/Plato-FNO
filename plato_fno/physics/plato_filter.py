"""PLATO band integration and noise simulation.

PLATO observes in white light: its 24 normal cameras (N-CAMs) share one
broad band of roughly 500–1000 nm. The two fast cameras (F-CAMs) carry a blue
(505–700 nm) and a red (665–1000 nm) filter, which gives a coarse colour.
Band edges are from the PlatoSim paper (arXiv:2310.06985).

The bands are modelled as top-hats; the real responses (CCD quantum
efficiency, optics throughput) are not flat, so treat band-integrated depths
as approximate.
"""

from __future__ import annotations

import numpy as np

# PLATO passbands [microns]
PLATO_BANDPASSES: dict[str, tuple[float, float]] = {
    "ncam": (0.500, 1.000),
    "fcam_blue": (0.505, 0.700),
    "fcam_red": (0.665, 1.000),
}


def integrate_spectrum_to_photometry(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    bandpasses: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Average a transit depth spectrum over each (top-hat) band.

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
        Band-averaged transit depth per band (NaN if the band holds < 2 grid points).
    """
    if bandpasses is None:
        bandpasses = PLATO_BANDPASSES

    result: dict[str, float] = {}
    for band, (lam_min, lam_max) in bandpasses.items():
        mask = (wavelengths >= lam_min) & (wavelengths <= lam_max)
        if mask.sum() < 2:
            result[band] = float("nan")
            continue
        lam = wavelengths[mask]
        # Divide by the span actually integrated over, so a constant spectrum
        # averages to exactly that constant whatever the grid spacing.
        result[band] = float(np.trapezoid(spectrum[mask], lam) / (lam[-1] - lam[0]))
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


def photometry_to_array(
    photometry: dict[str, float],
    bands: list[str] | None = None,
) -> np.ndarray:
    """Convert a photometry dict to a float32 array in ``bands`` order.

    Defaults to the order of PLATO_BANDPASSES.
    """
    bands = list(PLATO_BANDPASSES) if bands is None else bands
    return np.array([photometry[b] for b in bands], dtype=np.float32)
