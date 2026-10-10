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


def test_unwritten_rows_are_nan_after_crash(tmp_path):
    """If generation dies part-way, unwritten spectra must read back as NaN, not 0."""
    from plato_fno.physics import petitrt_driver

    output_path = tmp_path / "crashed.h5"

    def _crashing_results():
        for _ in range(3):
            yield _mock_spectrum(np.zeros(N_PARAMS))
        raise KeyboardInterrupt  # simulate the job being killed

    with patch.object(petitrt_driver.mp, "get_context") as mock_ctx:
        mock_pool = MagicMock()
        mock_pool.__enter__ = lambda s: s
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.imap.return_value = _crashing_results()
        mock_ctx.return_value.Pool.return_value = mock_pool

        with pytest.raises(KeyboardInterrupt):
            petitrt_driver.generate_dataset_hdf5(
                output_path=output_path,
                n_samples=N_SAMPLES,
                wavelength_range=(0.3, 5.0),
                n_wavelengths=N_WAVELENGTHS,
                n_workers=1,
                chunk_size=N_SAMPLES,
            )

    with h5py.File(output_path, "r") as f:
        targets = f["target"][:]
    assert np.isfinite(targets[:3]).all()
    assert np.isnan(targets[3:]).all()


def test_compute_spectrum_inputs_to_prt():
    """compute_spectrum passes gravity-aware temperatures to pRT."""
    from plato_fno.physics import petitrt_driver

    if not petitrt_driver._PRT_AVAILABLE:
        pytest.skip("petitRADTRANS not installed")

    state = AtmosphericState.sample_prior(np.random.default_rng(0))
    pressures = np.logspace(-6, 2, 20)
    wavelengths = np.linspace(0.3, 5.0, N_WAVELENGTHS)

    prt = MagicMock()
    wl_cm = np.linspace(0.29, 5.1, 500) * 1e-4
    prt.calculate_transit_radii.return_value = (wl_cm, np.full(500, 7e9), {})
    out = petitrt_driver.compute_spectrum(state, prt, pressures, wavelengths)

    kwargs = prt.calculate_transit_radii.call_args.kwargs
    assert kwargs["reference_gravity"] == pytest.approx(10**state.log_g)
    expected_T = petitrt_driver.guillot_temperature_profile(
        pressures, state.T_int, state.T_eq, 10**state.kappa_IR, 10**state.gamma,
        gravity=10**state.log_g,
    )
    np.testing.assert_allclose(kwargs["temperatures"], expected_T)
    assert out.shape == (N_WAVELENGTHS,)
    np.testing.assert_allclose(out, (7e9 / (state.R_star * 6.957e10)) ** 2, rtol=1e-6)
