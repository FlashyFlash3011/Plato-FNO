"""JWST instrument models for transmission-spectroscopy stress tests.

Covers instruments within the FNO output range (0.3–5.0 µm):

  Instrument       λ range (µm)   Bins   Baseline SNR (V=10 star)
  ─────────────────────────────────────────────────────────────────
  niriss_soss       0.60 – 2.80    40     300
  nirspec_prism     0.60 – 5.00    30     200
  nirspec_g140m     0.97 – 1.87    50     400
  nirspec_g395m     2.87 – 5.00    40     150

Spectral bins are chosen so that each bin contains several native FNO grid
points (native grid: 300 points over 0.3–5.0 µm, δλ ≈ 0.016 µm/point).
This matches how real JWST retrievals operate — spectral data are binned
to a manageable resolution before fitting.

Noise model
-----------
Per-bin photon noise:
    σ(λ) = max(|depth(λ)| / SNR(λ), 1 ppm)

SNR scales with stellar V-band magnitude relative to a V=10 baseline:
    SNR(V) = SNR_V10 × 10^(-0.2 × (V - 10))
"""

from __future__ import annotations

import numpy as np

# Instrument: (lambda_min, lambda_max, n_bins)
JWST_INSTRUMENTS: dict[str, tuple[float, float, int]] = {
    "niriss_soss":   (0.60, 2.80, 40),
    "nirspec_prism": (0.60, 5.00, 30),
    "nirspec_g140m": (0.97, 1.87, 50),
    "nirspec_g395m": (2.87, 5.00, 40),
}

# Baseline per-bin SNR for a V=10 host star, one transit, 30-min integrations
_SNR_V10: dict[str, float] = {
    "niriss_soss":   300.0,
    "nirspec_prism": 200.0,
    "nirspec_g140m": 400.0,
    "nirspec_g395m": 150.0,
}

_ONE_PPM = 1e-6


def bin_spectrum(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    lam_min: float,
    lam_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average-bin a transit depth spectrum to an instrument spectral grid.

    Parameters
    ----------
    spectrum : np.ndarray, shape (L,) or (1, L)
        FNO output transit depth spectrum.
    wavelengths : np.ndarray, shape (L,)
        Wavelength grid in microns (FNO native: 300 points, 0.3–5.0 µm).
    lam_min, lam_max : float
        Instrument wavelength range in microns.
    n_bins : int
        Number of output spectral channels.

    Returns
    -------
    binned      : np.ndarray, shape (n_bins,)  float32
    bin_centers : np.ndarray, shape (n_bins,)  float32
    """
    if spectrum.ndim == 2:
        spectrum = spectrum[0]

    edges   = np.linspace(lam_min, lam_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]).astype(np.float32)
    binned  = np.empty(n_bins, dtype=np.float32)

    for i in range(n_bins):
        mask = (wavelengths >= edges[i]) & (wavelengths < edges[i + 1])
        if mask.sum() > 0:
            binned[i] = float(spectrum[mask].mean())
        else:
            # Nearest-neighbour fallback when bin is narrower than native grid
            nearest  = int(np.argmin(np.abs(wavelengths - centers[i])))
            binned[i] = float(spectrum[nearest])

    return binned, centers
