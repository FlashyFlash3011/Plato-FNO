"""AtmosphereSurrogate: MLP parameter encoder + Solaris FNO1d.

Architecture
------------
1. param_encoder MLP maps theta (N_PARAMS,) → (n_enc,)
   using 2 linear layers with GELU activations.
2. The encoded vector is broadcast-expanded to (n_enc, L), forming
   the channel dimension over wavelength.
3. A wavelength positional embedding (1, L) is concatenated as an extra
   channel, giving (n_enc + 1, L) total input to the FNO.
4. FNO1d processes (B, n_enc+1, L) → (B, 1, L): transit depth spectrum.

The MLP encoder learns non-linear parameter combinations before the
Fourier layers act, which is important because the mapping from scalar
atmospheric parameters to spectral features is highly non-linear.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from solaris.models.fno import FNO
from solaris.nn.spectral import SpectralConv1d

from plato_fno.physics.atmosphere import N_PARAMS


def apply_overtone_spectral_init(model: nn.Module, exponent: float = 0.5) -> None:
    """Initialise FNO spectral convolution weights with a power-law mode decay.

    Ported from the parameter-golf overtone spectral embedding init. Each
    Fourier mode k is scaled by (k+1)^{-exponent} relative to the default
    uniform initialisation. This matches the statistics of physical signals
    where low-frequency components dominate (atmospheric spectra included).

    Specifically, for each SpectralConv1d weight tensor of shape
    (in_ch, out_ch, modes), we multiply the k-th mode slice by:
        gain_k = (1 / (k+1))^exponent

    This does NOT change the total weight energy (it is renormalised so
    the mean L2 norm matches the default init), only redistributes it
    toward lower modes.

    Parameters
    ----------
    model : nn.Module
        Any module tree containing SpectralConv1d layers.
    exponent : float
        Power-law exponent. 0.5 matches the parameter-golf value.
        Higher values concentrate more energy in the lowest modes.
    """
    for module in model.modules():
        if not isinstance(module, SpectralConv1d):
            continue

        w = module.weights.data  # (in_ch, out_ch, modes), cfloat
        modes = w.shape[-1]

        # Build per-mode gain vector: shape (modes,)
        gains = torch.tensor(
            [(1.0 / (k + 1)) ** exponent for k in range(modes)],
            dtype=torch.float32,
            device=w.device,
        )

        # Renormalise so RMS gain = 1 (preserve overall weight scale)
        gains = gains / gains.pow(2).mean().sqrt()

        # Apply to the mode dimension (last dim of complex weight)
        module.weights.data = w * gains.to(torch.cfloat)


class AtmosphereSurrogate(nn.Module):
    """FNO1d surrogate for the petitRADTRANS radiative transfer operator.

    Parameters
    ----------
    n_wavelengths : int
        Spatial dimension L (wavelength grid size). Default 300.
    n_enc : int
        Encoded parameter channel dimension fed to FNO. Default 32.
    hidden_channels : int
        FNO hidden channel width. Default 128.
    n_layers : int
        Number of FNO Fourier blocks. Default 6.
    modes : int
        Number of Fourier modes retained. Must be < n_wavelengths // 2.
        Default 24.
    """

    def __init__(
        self,
        n_wavelengths: int = 300,
        n_enc: int = 32,
        hidden_channels: int = 128,
        n_layers: int = 6,
        modes: int = 24,
    ) -> None:
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.n_enc = n_enc
        # Store arch kwargs explicitly so architecture_kwargs() is never fragile
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._modes = modes

        assert modes < n_wavelengths // 2, (
            f"modes={modes} must be < n_wavelengths//2={n_wavelengths // 2}"
        )

        # 2-layer MLP encoder: R^N_PARAMS → R^n_enc
        self.param_encoder = nn.Sequential(
            nn.Linear(N_PARAMS, 128),
            nn.GELU(),
            nn.Linear(128, n_enc),
        )

        # FNO1d: (B, n_enc + 1, L) → (B, 1, L)
        # +1 for the wavelength positional embedding channel
        self.fno = FNO(
            in_channels=n_enc + 1,
            out_channels=1,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            dim=1,
        )

        # Per-sample mean predictor: bypasses InstanceNorm inside FNO blocks.
        # InstanceNorm1d strips the mean over the wavelength axis so the FNO
        # cannot learn per-sample transit depth scale. This linear layer maps
        # the encoded params directly to a scalar offset added to the FNO
        # output, restoring the ability to predict different overall levels.
        self.mean_predictor = nn.Linear(n_enc, 1)

        # Wavelength positional embedding in [0, 1]: (1, 1, L)
        # Registered as buffer so it follows .to(device) automatically
        self.register_buffer(
            "_wl_embed",
            torch.linspace(0.0, 1.0, n_wavelengths).reshape(1, 1, n_wavelengths),
        )

    def encode_params(self, theta: torch.Tensor) -> torch.Tensor:
        """Map normalised parameter vectors to FNO input tensor.

        Parameters
        ----------
        theta : Tensor, shape (B, N_PARAMS)

        Returns
        -------
        Tensor, shape (B, n_enc + 1, L)
            Broadcast-encoded parameters concatenated with wavelength embedding.
        """
        enc = self.param_encoder(theta)               # (B, n_enc)
        enc = enc.unsqueeze(-1).expand(-1, -1, self.n_wavelengths)  # (B, n_enc, L)

        # Tile wavelength embedding to batch size
        wl = self._wl_embed.expand(theta.shape[0], -1, -1)  # (B, 1, L)

        return torch.cat([enc, wl], dim=1)              # (B, n_enc+1, L)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict normalised transit depth spectrum.

        Parameters
        ----------
        theta : Tensor, shape (B, N_PARAMS)
            Normalised atmospheric parameter vectors.

        Returns
        -------
        Tensor, shape (B, 1, L)
            Predicted normalised transit depth spectrum.
        """
        x = self.encode_params(theta)                          # (B, n_enc+1, L)
        fno_out = self.fno(x)                                  # (B, 1, L)
        enc = self.param_encoder(theta)                        # (B, n_enc)
        offset = self.mean_predictor(enc).unsqueeze(-1)        # (B, 1, 1)
        return fno_out + offset                                # (B, 1, L)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_kwargs(self) -> dict:
        """Return constructor kwargs for checkpoint reconstruction."""
        return {
            "n_wavelengths":   self.n_wavelengths,
            "n_enc":           self.n_enc,
            "hidden_channels": self._hidden_channels,
            "n_layers":        self._n_layers,
            "modes":           self._modes,
        }
