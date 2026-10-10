"""Tests for the valid-sample filter, splits and in-memory dataset."""

import h5py
import numpy as np
import pytest

from plato_fno.data.dataset import (
    MAX_TRANSIT_DEPTH,
    load_raw_arrays,
    make_splits,
    valid_sample_mask,
)
from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.physics.atmosphere import N_PARAMS

N_SAMPLES = 200
N_WAVELENGTHS = 16


@pytest.fixture
def hdf5_path(tmp_path):
    rng = np.random.default_rng(0)
    targets = rng.uniform(0.01, 0.02, (N_SAMPLES, 1, N_WAVELENGTHS)).astype(np.float32)
    targets[0] = np.nan  # failed pRT call
    targets[1, 0, 3] = 0.5  # runaway spectrum
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f["input"] = rng.uniform(0, 1, (N_SAMPLES, N_PARAMS)).astype(np.float32)
        f["target"] = targets
        f["wavelengths"] = np.linspace(0.3, 5.0, N_WAVELENGTHS)
    return path


def test_valid_sample_mask_drops_nan_and_too_deep():
    targets = np.full((3, 1, 4), 0.01, dtype=np.float32)
    targets[0, 0, 1] = np.nan
    targets[1, 0, 2] = MAX_TRANSIT_DEPTH + 0.01
    np.testing.assert_array_equal(valid_sample_mask(targets), [False, False, True])


def test_make_splits_returns_normalised_tensors(hdf5_path):
    norm = SpectrumNormalizer().fit(*load_raw_arrays(hdf5_path, np.arange(2, N_SAMPLES)))
    train, val, cal = make_splits(hdf5_path, norm, 0.8, 0.1, seed=0)
    assert len(train) + len(val) + len(cal) == N_SAMPLES - 2
    sample = train[0]
    assert sample["input"].shape == (N_PARAMS,)
    assert sample["target"].shape == (1, N_WAVELENGTHS)
    assert train.wavelengths.shape == (N_WAVELENGTHS,)
