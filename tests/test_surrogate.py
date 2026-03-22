"""Tests for AtmosphereSurrogate."""

import torch
import pytest

from plato_fno.models.surrogate import AtmosphereSurrogate
from plato_fno.physics.atmosphere import N_PARAMS

N_WAVELENGTHS = 64  # small for fast tests
N_ENC         = 16
HIDDEN        = 32
N_LAYERS      = 2
MODES         = 8
BATCH         = 4


@pytest.fixture
def surrogate():
    return AtmosphereSurrogate(
        n_wavelengths=N_WAVELENGTHS,
        n_enc=N_ENC,
        hidden_channels=HIDDEN,
        n_layers=N_LAYERS,
        modes=MODES,
    ).eval()


def test_forward_output_shape(surrogate):
    theta = torch.randn(BATCH, N_PARAMS)
    out = surrogate(theta)
    assert out.shape == (BATCH, 1, N_WAVELENGTHS)


def test_encode_params_shape(surrogate):
    theta = torch.randn(BATCH, N_PARAMS)
    enc = surrogate.encode_params(theta)
    # n_enc + 1 (wavelength positional embedding)
    assert enc.shape == (BATCH, N_ENC + 1, N_WAVELENGTHS)


def test_gradient_flow(surrogate):
    surrogate.train()
    theta = torch.randn(BATCH, N_PARAMS, requires_grad=False)
    out = surrogate(theta)
    loss = out.mean()
    loss.backward()
    for name, param in surrogate.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_deterministic_in_eval(surrogate):
    theta = torch.randn(BATCH, N_PARAMS)
    with torch.no_grad():
        out1 = surrogate(theta)
        out2 = surrogate(theta)
    torch.testing.assert_close(out1, out2)


def test_num_parameters_positive(surrogate):
    assert surrogate.num_parameters() > 0


def test_modes_constraint_enforced():
    with pytest.raises(AssertionError):
        AtmosphereSurrogate(n_wavelengths=10, modes=10)  # modes >= n_wavelengths // 2


def test_single_sample(surrogate):
    theta = torch.randn(1, N_PARAMS)
    out = surrogate(theta)
    assert out.shape == (1, 1, N_WAVELENGTHS)
