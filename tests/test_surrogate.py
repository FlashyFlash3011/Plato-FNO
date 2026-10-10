"""Tests for AtmosphereSurrogate."""

import pytest
import torch
from solaris.nn.spectral import SpectralConv1d

from plato_fno.models.surrogate import AtmosphereSurrogate, apply_overtone_spectral_init
from plato_fno.physics.atmosphere import N_PARAMS

N_WAVELENGTHS = 64  # small for fast tests
N_ENC = 16
HIDDEN = 32
N_LAYERS = 2
MODES = 8
BATCH = 4


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


def test_spectral_shape_depends_on_params(surrogate):
    """The FNO must see theta: different inputs need different spectral *shapes*.

    Regression test: with InstanceNorm inside the FNO blocks, the broadcast
    parameter encoding was cancelled exactly and every input produced the same
    shape (only the mean_predictor offset varied).
    """
    torch.manual_seed(0)
    theta = torch.randn(2, N_PARAMS)
    with torch.no_grad():
        out = surrogate(theta)[:, 0, :]
    shape = out - out.mean(dim=-1, keepdim=True)
    diff = (shape[0] - shape[1]).abs().max() / shape.abs().max()
    assert diff > 1e-3, f"spectral shape ignores theta (relative diff {diff:.2e})"


def test_num_parameters_positive(surrogate):
    assert surrogate.num_parameters() > 0


def test_modes_constraint_enforced():
    with pytest.raises(AssertionError):
        AtmosphereSurrogate(n_wavelengths=10, modes=10)  # modes >= n_wavelengths // 2


def test_single_sample(surrogate):
    theta = torch.randn(1, N_PARAMS)
    out = surrogate(theta)
    assert out.shape == (1, 1, N_WAVELENGTHS)


def test_architecture_kwargs_roundtrip(surrogate):
    """architecture_kwargs should contain exactly the constructor args."""
    kwargs = surrogate.architecture_kwargs()
    assert kwargs["n_wavelengths"] == N_WAVELENGTHS
    assert kwargs["n_enc"] == N_ENC
    assert kwargs["hidden_channels"] == HIDDEN
    assert kwargs["n_layers"] == N_LAYERS
    assert kwargs["modes"] == MODES


def test_overtone_init_lowers_high_mode_energy():
    """After overtone init, low-frequency modes should have more energy than high."""
    torch.manual_seed(0)
    model = AtmosphereSurrogate(
        n_wavelengths=N_WAVELENGTHS,
        n_enc=N_ENC,
        hidden_channels=HIDDEN,
        n_layers=N_LAYERS,
        modes=MODES,
    )
    apply_overtone_spectral_init(model, exponent=0.5)

    for module in model.modules():
        if isinstance(module, SpectralConv1d):
            w = module.weights.data.abs()  # (in_ch, out_ch, modes)
            # Mean magnitude of mode 0 should exceed mode -1 (highest)
            low_mode_energy = w[:, :, 0].mean().item()
            high_mode_energy = w[:, :, -1].mean().item()
            assert low_mode_energy > high_mode_energy, (
                f"Expected low-mode energy ({low_mode_energy:.4f}) > "
                f"high-mode energy ({high_mode_energy:.4f}) after overtone init"
            )


def test_forward_bf16_autocast(surrogate):
    """Model should produce finite output under BF16 autocast on CPU."""
    theta = torch.randn(BATCH, N_PARAMS)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = surrogate(theta)
    assert out.isfinite().all(), "Non-finite values under BF16 autocast"
    assert out.shape == (BATCH, 1, N_WAVELENGTHS)


def test_overtone_init_preserves_weight_scale():
    """Overtone init should not drastically change the overall weight scale."""
    torch.manual_seed(0)
    model = AtmosphereSurrogate(
        n_wavelengths=N_WAVELENGTHS,
        n_enc=N_ENC,
        hidden_channels=HIDDEN,
        n_layers=N_LAYERS,
        modes=MODES,
    )
    # Record RMS before
    before_rms = (
        torch.cat(
            [
                m.weights.data.abs().flatten()
                for m in model.modules()
                if isinstance(m, SpectralConv1d)
            ]
        )
        .mean()
        .item()
    )

    apply_overtone_spectral_init(model, exponent=0.5)

    after_rms = (
        torch.cat(
            [
                m.weights.data.abs().flatten()
                for m in model.modules()
                if isinstance(m, SpectralConv1d)
            ]
        )
        .mean()
        .item()
    )

    # RMS should be within 50% — the init redistributes but doesn't explode
    assert abs(after_rms - before_rms) / (before_rms + 1e-9) < 0.5


def test_save_load_surrogate_roundtrip(tmp_path, surrogate):
    import numpy as np

    from plato_fno.data.normalizer import SpectrumNormalizer
    from plato_fno.models.surrogate import load_surrogate, save_surrogate

    rng = np.random.default_rng(0)
    norm = SpectrumNormalizer().fit(
        rng.normal(size=(50, N_PARAMS)), rng.normal(size=(50, 1, N_WAVELENGTHS))
    )
    path = tmp_path / "model.pt"
    save_surrogate(path, surrogate, norm, epoch=3)

    model, norm2, ckpt = load_surrogate(path)
    assert ckpt["epoch"] == 3
    assert model.architecture_kwargs() == surrogate.architecture_kwargs()
    np.testing.assert_allclose(norm2.spec_std, norm.spec_std)
    theta = torch.randn(BATCH, N_PARAMS)
    with torch.no_grad():
        torch.testing.assert_close(model(theta), surrogate(theta))


def test_load_surrogate_rejects_bare_state_dict(tmp_path, surrogate):
    from plato_fno.models.surrogate import load_surrogate

    path = tmp_path / "old.pt"
    torch.save(surrogate.state_dict(), path)
    with pytest.raises(ValueError, match="older version"):
        load_surrogate(path)
