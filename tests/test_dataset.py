"""Tests for the valid-sample filter, splits and in-memory dataset."""

import h5py
import numpy as np
import pytest

from plato_fno.data.dataset import (
    MAX_TRANSIT_DEPTH,
    SPLIT_NAMES,
    load_raw_arrays,
    make_splits,
    split_indices,
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


def test_splits_are_disjoint_and_cover_valid_samples(hdf5_path):
    idx = split_indices(hdf5_path, 0.7, 0.1, 0.1, seed=0)
    assert set(idx) == set(SPLIT_NAMES)
    all_idx = np.concatenate(list(idx.values()))
    assert len(all_idx) == len(set(all_idx.tolist())) == N_SAMPLES - 2
    assert 0 not in all_idx and 1 not in all_idx
    assert len(idx["test"]) > 0


def test_splits_are_reproducible(hdf5_path):
    a = split_indices(hdf5_path, 0.7, 0.1, 0.1, seed=3)
    b = split_indices(hdf5_path, 0.7, 0.1, 0.1, seed=3)
    for name in SPLIT_NAMES:
        np.testing.assert_array_equal(a[name], b[name])


def test_split_fractions_must_leave_a_test_split(hdf5_path):
    with pytest.raises(ValueError):
        split_indices(hdf5_path, 0.8, 0.1, 0.1)


def test_make_splits_returns_normalised_tensors(hdf5_path):
    idx = split_indices(hdf5_path, 0.7, 0.1, 0.1, seed=0)
    norm = SpectrumNormalizer().fit(*load_raw_arrays(hdf5_path, idx["train"]))
    splits = make_splits(hdf5_path, norm, 0.7, 0.1, 0.1, seed=0)
    train = splits["train"]
    assert len(train) == len(idx["train"])
    sample = train[0]
    assert sample["input"].shape == (N_PARAMS,)
    assert sample["target"].shape == (1, N_WAVELENGTHS)
    # normalised on its own split → ~zero mean
    assert abs(train.targets.mean().item()) < 1e-4
    assert train.wavelengths.shape == (N_WAVELENGTHS,)
