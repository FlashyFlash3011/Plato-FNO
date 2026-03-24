"""Dataset and split utilities for the atmosphere→spectrum pipeline.

Wraps solaris.datapipes.dataset.HDF5Dataset and applies on-the-fly
normalisation so the HDF5 file always stores raw physical values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from solaris.datapipes.dataset import build_dataloader  # noqa: F401 (re-exported)

from plato_fno.data.normalizer import SpectrumNormalizer


class AtmosphereDataset(Dataset):
    """HDF5-backed dataset with on-the-fly normalisation.

    __getitem__ returns a dict with:
        "input"  : Tensor (N_PARAMS,)  — normalised atmospheric parameters
        "target" : Tensor (1, L)       — normalised transit depth spectrum

    Failed pRT samples (NaN spectra) are filtered out during __init__.

    Parameters
    ----------
    hdf5_path : Path
        Path to the HDF5 file produced by generate_dataset_hdf5.
    normalizer : SpectrumNormalizer
        Fitted normaliser. Applied at load time.
    indices : np.ndarray, optional
        Subset of sample indices to use. If None, all valid samples are used.
    transform : callable, optional
        Additional transform applied to the dict after normalisation.
    """

    def __init__(
        self,
        hdf5_path: Path,
        normalizer: SpectrumNormalizer,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.hdf5_path  = Path(hdf5_path)
        self.normalizer = normalizer
        self.transform  = transform

        # Find valid (non-NaN) samples
        with h5py.File(self.hdf5_path, "r") as f:
            targets = f["target"][:]  # (N, 1, L)
            n_total = targets.shape[0]

        valid_mask = np.isfinite(targets).all(axis=(1, 2))  # (N,)
        valid_indices = np.where(valid_mask)[0]

        if indices is not None:
            # Intersect requested indices with valid ones
            valid_set = set(valid_indices.tolist())
            self._indices = np.array([i for i in indices if i in valid_set])
        else:
            self._indices = valid_indices

        self._n = len(self._indices)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = int(self._indices[idx])

        with h5py.File(self.hdf5_path, "r") as f:
            params  = f["input"][sample_idx]   # (N_PARAMS,) float32
            spectrum = f["target"][sample_idx]  # (1, L) float32

        params_norm  = self.normalizer.transform_params(params)     # (N_PARAMS,)
        spectrum_norm = self.normalizer.transform_spectra(spectrum)  # (1, L)

        sample = {
            "input":  torch.from_numpy(params_norm),
            "target": torch.from_numpy(spectrum_norm),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def wavelengths(self) -> np.ndarray:
        """Return the wavelength grid [µm] from the HDF5 metadata."""
        with h5py.File(self.hdf5_path, "r") as f:
            return f["wavelengths"][:]


def make_splits(
    hdf5_path: Path,
    normalizer: SpectrumNormalizer,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[AtmosphereDataset, AtmosphereDataset, AtmosphereDataset]:
    """Split HDF5 dataset into train / val / calibration sets.

    The remaining fraction (1 - train_frac - val_frac) goes to calibration.
    The normalizer should be fitted on the training split *after* calling this.

    Parameters
    ----------
    hdf5_path : Path
        HDF5 file from generate_dataset_hdf5.
    normalizer : SpectrumNormalizer
        Must be already fitted before passing here.
    train_frac : float
        Fraction of valid samples for training.
    val_frac : float
        Fraction of valid samples for validation.
    seed : int
        RNG seed for shuffle.

    Returns
    -------
    train_ds, val_ds, cal_ds : AtmosphereDataset
        Three dataset views of the same HDF5 file.
    """
    # Discover all valid indices
    with h5py.File(hdf5_path, "r") as f:
        targets = f["target"][:]
    valid_mask = np.isfinite(targets).all(axis=(1, 2))
    all_indices = np.where(valid_mask)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)

    n = len(all_indices)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train:n_train + n_val]
    cal_idx   = all_indices[n_train + n_val:]

    train_ds = AtmosphereDataset(hdf5_path, normalizer, indices=train_idx)
    val_ds   = AtmosphereDataset(hdf5_path, normalizer, indices=val_idx)
    cal_ds   = AtmosphereDataset(hdf5_path, normalizer, indices=cal_idx)

    return train_ds, val_ds, cal_ds


def load_raw_arrays(
    hdf5_path: Path,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw (un-normalised) parameter and spectrum arrays for a set of indices.

    Used during normalizer fitting to compute stats on the training split only.

    Returns
    -------
    params  : np.ndarray, shape (len(indices), N_PARAMS)
    spectra : np.ndarray, shape (len(indices), 1, L)
    """
    with h5py.File(hdf5_path, "r") as f:
        sorted_idx = np.sort(indices)
        params  = f["input"][sorted_idx]   # (N, N_PARAMS)
        spectra = f["target"][sorted_idx]  # (N, 1, L)
    return params, spectra
