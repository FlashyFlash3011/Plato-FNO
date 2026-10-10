"""Tests for SpectrumNormalizer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.physics.atmosphere import N_PARAMS

N_SAMPLES = 200
N_WAVELENGTHS = 50


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    params = rng.uniform(-1, 1, (N_SAMPLES, N_PARAMS)).astype(np.float32)
    spectra = rng.uniform(0.01, 0.02, (N_SAMPLES, 1, N_WAVELENGTHS)).astype(np.float32)
    return params, spectra


def test_fit_returns_self():
    params, spectra = _make_data()
    norm = SpectrumNormalizer()
    result = norm.fit(params, spectra)
    assert result is norm


def test_fit_zero_mean_params():
    params, spectra = _make_data()
    norm = SpectrumNormalizer().fit(params, spectra)
    transformed = norm.transform_params(params)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-4)


def test_fit_unit_std_params():
    params, spectra = _make_data()
    norm = SpectrumNormalizer().fit(params, spectra)
    transformed = norm.transform_params(params)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-3)


def test_invertibility_params():
    params, spectra = _make_data()
    norm = SpectrumNormalizer().fit(params, spectra)
    recovered = norm.inverse_transform_params(norm.transform_params(params))
    np.testing.assert_allclose(recovered, params, atol=1e-5)


def test_invertibility_spectra():
    params, spectra = _make_data()
    norm = SpectrumNormalizer().fit(params, spectra)
    recovered = norm.inverse_transform_spectra(norm.transform_spectra(spectra))
    np.testing.assert_allclose(recovered, spectra, atol=1e-5)


def test_save_load_roundtrip():
    params, spectra = _make_data()
    norm = SpectrumNormalizer().fit(params, spectra)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)

    try:
        norm.save(path)
        norm2 = SpectrumNormalizer.load(path)

        np.testing.assert_allclose(norm.param_mean, norm2.param_mean)
        np.testing.assert_allclose(norm.spec_std, norm2.spec_std)
    finally:
        path.unlink(missing_ok=True)


def test_unfitted_raises():
    norm = SpectrumNormalizer()
    params = np.ones((5, N_PARAMS), dtype=np.float32)
    with pytest.raises(RuntimeError, match="fitted"):
        norm.transform_params(params)


def test_nan_spectra_excluded_from_fit():
    params, spectra = _make_data()
    # Inject NaN in some spectra
    spectra[::10] = np.nan
    norm = SpectrumNormalizer().fit(params, spectra)
    # Stats should still be finite
    assert np.isfinite(norm.spec_mean).all()
    assert np.isfinite(norm.spec_std).all()
