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


def simulate_jwst_observation(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    instrument: str = "niriss_soss",
    stellar_vmag: float = 10.0,
    rng: np.random.Generator | None = None,
    snr_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a JWST transmission spectrum observation.

    Bins the FNO spectrum to the instrument grid and adds per-channel
    photon noise scaled by stellar brightness.

    Parameters
    ----------
    spectrum : np.ndarray, shape (L,) or (1, L)
        Noiseless FNO-predicted transit depth spectrum.
    wavelengths : np.ndarray, shape (L,)
        FNO wavelength grid in microns.
    instrument : str
        One of JWST_INSTRUMENTS.
    stellar_vmag : float
        V-band apparent magnitude of the host star.
    rng : np.random.Generator, optional
        Defaults to a new generator.
    snr_override : float, optional
        Override the instrument baseline SNR (useful for controlled experiments).

    Returns
    -------
    observed    : np.ndarray, shape (n_bins,)  — noisy binned transit depth
    bin_centers : np.ndarray, shape (n_bins,)  — bin wavelengths [µm]
    sigma       : np.ndarray, shape (n_bins,)  — per-bin 1σ noise [same units as depth]
    """
    if instrument not in JWST_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Choose from: {list(JWST_INSTRUMENTS)}"
        )

    lam_min, lam_max, n_bins = JWST_INSTRUMENTS[instrument]
    binned, bin_centers = bin_spectrum(spectrum, wavelengths, lam_min, lam_max, n_bins)

    snr_base = snr_override if snr_override is not None else _SNR_V10[instrument]
    snr      = snr_base * 10.0 ** (-0.2 * (stellar_vmag - 10.0))

    # Per-bin sigma: signal / SNR with 1 ppm noise floor
    sigma = np.maximum(np.abs(binned) / snr, _ONE_PPM).astype(np.float32)

    if rng is None:
        rng = np.random.default_rng()
    noise    = rng.normal(0.0, 1.0, n_bins).astype(np.float32) * sigma
    observed = (binned + noise).astype(np.float32)

    return observed, bin_centers, sigma


def make_bin_transform(
    wavelengths: np.ndarray,
    instrument: str,
):
    """Return a callable that bins a batch of spectra to the instrument grid.

    The returned function maps (B, L) → (B, n_bins), suitable for use as
    ``obs_transform`` in MCMCRetrieval so the likelihood is computed in the
    instrument's spectral space rather than the full 300-point FNO grid.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (L,)
        FNO native wavelength grid in microns.
    instrument : str
        One of JWST_INSTRUMENTS.
    """
    lam_min, lam_max, n_bins = JWST_INSTRUMENTS[instrument]

    def _transform(pred_batch: np.ndarray) -> np.ndarray:
        B   = pred_batch.shape[0]
        out = np.empty((B, n_bins), dtype=np.float32)
        for i in range(B):
            out[i], _ = bin_spectrum(
                pred_batch[i], wavelengths, lam_min, lam_max, n_bins
            )
        return out

    return _transform
