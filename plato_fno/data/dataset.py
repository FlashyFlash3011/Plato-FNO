"""Dataset and split utilities for the atmosphere→spectrum pipeline.

The HDF5 file always stores raw physical values; normalisation is applied
once when a split is loaded. The whole dataset is small (50k × 300 float32
≈ 60 MB), so each split is held in memory as tensors.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from plato_fno.data.normalizer import SpectrumNormalizer

# Spectra deeper than this are treated as broken pRT output (e.g. a runaway
# transit radius at very low gravity) and dropped, together with NaN spectra
# from failed pRT calls.
MAX_TRANSIT_DEPTH = 0.15

def valid_sample_mask(targets: np.ndarray) -> np.ndarray:
    """Return a (N,) bool mask of usable spectra in a (N, 1, L) target array."""
    finite = np.isfinite(targets).all(axis=(1, 2))
    deepest = np.nan_to_num(targets, nan=np.inf).max(axis=(1, 2))
    return finite & (deepest < MAX_TRANSIT_DEPTH)


def load_raw_arrays(
    hdf5_path: Path,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw (un-normalised) parameter and spectrum arrays for a set of indices.

    Returns
    -------
    params  : np.ndarray, shape (len(indices), N_PARAMS)
    spectra : np.ndarray, shape (len(indices), 1, L)
    """
    sorted_idx = np.sort(indices)
    with h5py.File(hdf5_path, "r") as f:
        params = f["input"][sorted_idx]
        spectra = f["target"][sorted_idx]
    return params, spectra


class AtmosphereDataset(Dataset):
    """In-memory dataset of normalised (parameters, spectrum) pairs.

    ``__getitem__`` returns a dict with:
        "input"  : Tensor (N_PARAMS,)  — normalised atmospheric parameters
        "target" : Tensor (1, L)       — normalised transit depth spectrum

    Parameters
    ----------
    hdf5_path : Path
        HDF5 file produced by ``generate_dataset_hdf5``.
    normalizer : SpectrumNormalizer
        Fitted normaliser.
    indices : np.ndarray
        Sample indices to load.
    """

    def __init__(
        self,
        hdf5_path: Path,
        normalizer: SpectrumNormalizer,
        indices: np.ndarray,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        params, spectra = load_raw_arrays(self.hdf5_path, indices)
        self.inputs = torch.from_numpy(normalizer.transform_params(params))
        self.targets = torch.from_numpy(normalizer.transform_spectra(spectra))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input": self.inputs[idx], "target": self.targets[idx]}

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
    """Split the valid samples into train / val / calibration datasets.

    The remaining fraction (1 - train_frac - val_frac) goes to calibration.
    The normaliser must already be fitted on the training split.
    """
    with h5py.File(hdf5_path, "r") as f:
        targets = f["target"][:]
    all_indices = np.where(valid_sample_mask(targets))[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)

    n = len(all_indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train : n_train + n_val]
    cal_idx = all_indices[n_train + n_val :]

    return (
        AtmosphereDataset(hdf5_path, normalizer, train_idx),
        AtmosphereDataset(hdf5_path, normalizer, val_idx),
        AtmosphereDataset(hdf5_path, normalizer, cal_idx),
    )
