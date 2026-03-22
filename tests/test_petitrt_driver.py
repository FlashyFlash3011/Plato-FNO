"""Tests for petitrt_driver (mock-based, no pRT install needed)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from plato_fno.physics.atmosphere import AtmosphericState, N_PARAMS

N_WAVELENGTHS = 30
N_SAMPLES     = 8


def _mock_spectrum(state_vec: np.ndarray) -> np.ndarray:
    """Deterministic mock spectrum: all ones scaled by T_eq."""
    state = AtmosphericState.from_vector(state_vec)
    return np.ones(N_WAVELENGTHS, dtype=np.float32) * (state.T_eq / 1000.0)


def test_atmospheric_state_sample_prior_vector_shape():
    rng = np.random.default_rng(0)
    state = AtmosphericState.sample_prior(rng)
    v = state.to_vector()
    assert v.shape == (N_PARAMS,)


def test_generate_dataset_hdf5_mock(tmp_path):
    """generate_dataset_hdf5 should create correct HDF5 structure with mocked pRT."""
    from plato_fno.physics import petitrt_driver

    output_path = tmp_path / "test_spectra.h5"

    # Patch _compute_one so no real pRT is needed
    with patch.object(petitrt_driver, "_compute_one", side_effect=_mock_spectrum):
        with patch.object(petitrt_driver.mp, "get_context") as mock_ctx:
            # Set up mock pool that applies _mock_spectrum via imap
            mock_pool = MagicMock()
            mock_pool.__enter__ = lambda s: s
            mock_pool.__exit__ = MagicMock(return_value=False)
            mock_pool.imap.return_value = iter([
                _mock_spectrum(np.zeros(N_PARAMS)) for _ in range(N_SAMPLES)
            ])
            mock_ctx.return_value.Pool.return_value = mock_pool

            petitrt_driver.generate_dataset_hdf5(
                output_path=output_path,
                n_samples=N_SAMPLES,
                wavelength_range=(0.3, 5.0),
                n_wavelengths=N_WAVELENGTHS,
                n_layers=10,
                n_workers=1,
                chunk_size=4,
                seed=0,
            )

    assert output_path.exists()

    with h5py.File(output_path, "r") as f:
        assert "input"  in f
        assert "target" in f
        assert "wavelengths" in f

        assert f["input"].shape  == (N_SAMPLES, N_PARAMS)
        assert f["target"].shape == (N_SAMPLES, 1, N_WAVELENGTHS)
        assert f["wavelengths"].shape == (N_WAVELENGTHS,)

        assert f.attrs["n_samples"] == N_SAMPLES
        assert f.attrs["seed"] == 0


def test_hdf5_input_values_are_physical(tmp_path):
    """Input vectors should be finite and within reasonable physical ranges."""
    from plato_fno.physics import petitrt_driver

    output_path = tmp_path / "test_phys.h5"

    with patch.object(petitrt_driver, "_compute_one", side_effect=_mock_spectrum):
        with patch.object(petitrt_driver.mp, "get_context") as mock_ctx:
            mock_pool = MagicMock()
            mock_pool.__enter__ = lambda s: s
            mock_pool.__exit__ = MagicMock(return_value=False)
            mock_pool.imap.return_value = iter([
                _mock_spectrum(np.zeros(N_PARAMS)) for _ in range(N_SAMPLES)
            ])
            mock_ctx.return_value.Pool.return_value = mock_pool

            petitrt_driver.generate_dataset_hdf5(
                output_path=output_path,
                n_samples=N_SAMPLES,
                wavelength_range=(0.3, 5.0),
                n_wavelengths=N_WAVELENGTHS,
                n_layers=10,
                n_workers=1,
                chunk_size=N_SAMPLES,
                seed=42,
            )

    with h5py.File(output_path, "r") as f:
        params = f["input"][:]
        assert np.isfinite(params).all()
        # T_int should be within [50, 800]
        assert params[:, 0].min() >= 50.0
        assert params[:, 0].max() <= 800.0
