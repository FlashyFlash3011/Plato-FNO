"""JWST instrument models for transmission-spectroscopy stress tests.

Covers instruments within the FNO output range (0.3–5.0 µm):

  Instrument       λ range (µm)   Bins   Per-bin 1σ at V=10 (ppm)
  ────────────────────────────────────────────────────────────────
  niriss_soss       0.60 – 2.80    40     60
  nirspec_prism     0.60 – 5.00    30     50
  nirspec_g140m     0.97 – 1.87    50     80
  nirspec_g395m     2.87 – 5.00    40     90

Spectral bins are chosen so that each bin contains several native FNO grid
points (native grid: 300 points over 0.3–5.0 µm, δλ ≈ 0.016 µm/point).

Noise model
-----------
The uncertainty on a transit depth is set by the number of stellar photons
collected, not by the depth itself:

    σ(V) = σ_V10 × 10^(0.2 × (V - 10))

(photon-limited: flux ∝ 10^(-0.4 V), so σ ∝ 1/sqrt(flux)).

The σ_V10 values are illustrative, order-of-magnitude placeholders for one
transit at these coarse bin widths, not instrument-team numbers; use the JWST
ETC or PandExo for a real target. They also ignore saturation (NIRSpec PRISM
saturates on stars this bright) and correlated systematics.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Instrument: (lambda_min, lambda_max, n_bins)
JWST_INSTRUMENTS: dict[str, tuple[float, float, int]] = {
    "niriss_soss": (0.60, 2.80, 40),
    "nirspec_prism": (0.60, 5.00, 30),
    "nirspec_g140m": (0.97, 1.87, 50),
    "nirspec_g395m": (2.87, 5.00, 40),
}

# Illustrative per-bin 1σ transit-depth noise [ppm] for a V=10 host star,
# one transit (see module docstring)
NOISE_PPM_V10: dict[str, float] = {
    "niriss_soss": 60.0,
    "nirspec_prism": 50.0,
    "nirspec_g140m": 80.0,
    "nirspec_g395m": 90.0,
}


def _instrument(name: str) -> tuple[float, float, int]:
    if name not in JWST_INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{name}'. Choose from: {list(JWST_INSTRUMENTS)}")
    return JWST_INSTRUMENTS[name]


def bin_centers(lam_min: float, lam_max: float, n_bins: int) -> np.ndarray:
    """Centres [µm] of ``n_bins`` equal-width bins spanning [lam_min, lam_max]."""
    edges = np.linspace(lam_min, lam_max, n_bins + 1)
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)


def binning_matrix(
    wavelengths: np.ndarray,
    lam_min: float,
    lam_max: float,
    n_bins: int,
) -> np.ndarray:
    """Matrix W of shape (n_bins, L) such that ``W @ spectrum`` gives bin averages.

    Each row averages the native points inside the bin. A bin containing no
    native point (narrower than the grid spacing) takes the nearest point.
    """
    edges = np.linspace(lam_min, lam_max, n_bins + 1)
    centers = bin_centers(lam_min, lam_max, n_bins)
    W = np.zeros((n_bins, len(wavelengths)), dtype=np.float64)
    for i in range(n_bins):
        mask = (wavelengths >= edges[i]) & (wavelengths < edges[i + 1])
        if mask.any():
            W[i, mask] = 1.0 / mask.sum()
        else:
            W[i, np.argmin(np.abs(wavelengths - centers[i]))] = 1.0
    return W


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
    spectrum = np.asarray(spectrum).reshape(-1)
    W = binning_matrix(wavelengths, lam_min, lam_max, n_bins)
    return (W @ spectrum).astype(np.float32), bin_centers(lam_min, lam_max, n_bins)


def simulate_jwst_observation(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    instrument: str = "niriss_soss",
    stellar_vmag: float = 10.0,
    rng: np.random.Generator | None = None,
    noise_ppm_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a JWST transmission spectrum observation.

    Bins the FNO spectrum to the instrument grid and adds white Gaussian
    noise whose level depends on stellar brightness.

    Parameters
    ----------
    spectrum : np.ndarray, shape (L,) or (1, L)
        Noiseless transit depth spectrum.
    wavelengths : np.ndarray, shape (L,)
        FNO wavelength grid in microns.
    instrument : str
        One of JWST_INSTRUMENTS.
    stellar_vmag : float
        V-band apparent magnitude of the host star.
    rng : np.random.Generator, optional
        Defaults to a new generator.
    noise_ppm_override : float, optional
        Per-bin 1σ at V=10 [ppm], replacing the instrument default.

    Returns
    -------
    observed    : np.ndarray, shape (n_bins,)  — noisy binned transit depth
    bin_centers : np.ndarray, shape (n_bins,)  — bin wavelengths [µm]
    sigma       : np.ndarray, shape (n_bins,)  — per-bin 1σ noise [depth units]
    """
    lam_min, lam_max, n_bins = _instrument(instrument)
    binned, centers = bin_spectrum(spectrum, wavelengths, lam_min, lam_max, n_bins)

    noise_v10 = noise_ppm_override if noise_ppm_override is not None else NOISE_PPM_V10[instrument]
    sigma_value = noise_v10 * 1e-6 * 10.0 ** (0.2 * (stellar_vmag - 10.0))
    sigma = np.full(n_bins, sigma_value, dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()
    observed = (binned + rng.normal(0.0, 1.0, n_bins) * sigma).astype(np.float32)
    return observed, centers, sigma


def make_bin_transform(
    wavelengths: np.ndarray,
    instrument: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that bins a batch of spectra to the instrument grid.

    The returned function maps (B, L) → (B, n_bins) and is meant to be passed
    as ``obs_transform`` to MCMCRetrieval, so the likelihood is computed in
    the instrument's spectral space rather than on the 300-point FNO grid.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (L,)
        FNO native wavelength grid in microns.
    instrument : str
        One of JWST_INSTRUMENTS.
    """
    lam_min, lam_max, n_bins = _instrument(instrument)
    W_T = binning_matrix(wavelengths, lam_min, lam_max, n_bins).T  # (L, n_bins)

    def _transform(pred_batch: np.ndarray) -> np.ndarray:
        return np.asarray(pred_batch) @ W_T

    return _transform
