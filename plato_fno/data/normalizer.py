"""Per-channel z-score normalisation for atmospheric parameters and spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SpectrumNormalizer:
    """Per-channel z-score normalisation for both parameters and spectra.

    Fits on training data only; applied consistently to val/cal/test splits.
    Training stores the stats inside the model checkpoint (see
    ``save_surrogate``), so a model is never paired with the wrong stats.

    Attributes
    ----------
    param_mean : np.ndarray, shape (N_PARAMS,)
    param_std  : np.ndarray, shape (N_PARAMS,)
    spec_mean  : np.ndarray, shape (1, n_wavelengths)
    spec_std   : np.ndarray, shape (1, n_wavelengths)
    """

    def __init__(self) -> None:
        self.param_mean: np.ndarray | None = None
        self.param_std:  np.ndarray | None = None
        self.spec_mean:  np.ndarray | None = None
        self.spec_std:   np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        params: np.ndarray,   # (N, N_PARAMS)
        spectra: np.ndarray,  # (N, 1, L)
    ) -> SpectrumNormalizer:
        """Compute and store normalisation statistics from training data.

        NaN values in spectra (failed pRT calls) are excluded from stats.
        """
        self.param_mean = params.mean(axis=0).astype(np.float32)
        self.param_std  = params.std(axis=0).clip(min=1e-8).astype(np.float32)

        # spectra: (N, 1, L) — compute mean/std over N, keep (1, L) shape
        valid_mask = np.isfinite(spectra).all(axis=(1, 2))  # (N,)
        valid_spectra = spectra[valid_mask]  # (N_valid, 1, L)
        self.spec_mean = valid_spectra.mean(axis=0).astype(np.float32)
        self.spec_std  = valid_spectra.std(axis=0).clip(min=1e-12).astype(np.float32)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / inverse transform
    # ------------------------------------------------------------------

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        """Z-score normalise parameters. Handles (N, N_PARAMS) or (N_PARAMS,)."""
        self._check_fitted()
        return ((params - self.param_mean) / self.param_std).astype(np.float32)

    def inverse_transform_params(self, params: np.ndarray) -> np.ndarray:
        """Undo z-score normalisation on parameters."""
        self._check_fitted()
        return (params * self.param_std + self.param_mean).astype(np.float32)

    def transform_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Z-score normalise spectra. Handles (N, 1, L) or (1, L)."""
        self._check_fitted()
        return ((spectra - self.spec_mean) / self.spec_std).astype(np.float32)

    def inverse_transform_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Undo z-score normalisation on spectra."""
        self._check_fitted()
        return (spectra * self.spec_std + self.spec_mean).astype(np.float32)

    def spec_std_ppm(self) -> np.ndarray:
        """Return spectral std in parts-per-million for reporting."""
        self._check_fitted()
        return (self.spec_std * 1e6).squeeze()  # (L,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return the fitted statistics (stored inside model checkpoints)."""
        self._check_fitted()
        return {
            "param_mean": self.param_mean,
            "param_std": self.param_std,
            "spec_mean": self.spec_mean,
            "spec_std": self.spec_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> SpectrumNormalizer:
        """Rebuild a fitted normaliser from :meth:`state_dict` output."""
        obj = cls()
        obj.param_mean = np.asarray(state["param_mean"], dtype=np.float32)
        obj.param_std = np.asarray(state["param_std"], dtype=np.float32)
        obj.spec_mean = np.asarray(state["spec_mean"], dtype=np.float32)
        obj.spec_std = np.asarray(state["spec_std"], dtype=np.float32)
        obj._fitted = True
        return obj

    def save(self, path: Path) -> None:
        """Write normalisation stats to .npz file."""
        np.savez(path, **self.state_dict())

    @classmethod
    def load(cls, path: Path) -> SpectrumNormalizer:
        """Reconstruct normaliser from saved .npz file."""
        with np.load(path) as data:
            return cls.from_state_dict(dict(data))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SpectrumNormalizer must be fitted before use. Call .fit() first.")
