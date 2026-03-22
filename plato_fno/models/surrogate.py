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

import torch
import torch.nn as nn

from solaris.models.fno import FNO

from plato_fno.physics.atmosphere import N_PARAMS


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
        x = self.encode_params(theta)  # (B, n_enc+1, L)
        return self.fno(x)             # (B, 1, L)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_kwargs(self) -> dict:
        """Return constructor kwargs for checkpoint reconstruction."""
        return {
            "n_wavelengths":   self.n_wavelengths,
            "n_enc":           self.n_enc,
            "hidden_channels": self.fno.blocks[0].conv.out_channels if hasattr(self.fno, "blocks") else 128,
            "n_layers":        len(self.fno.blocks) if hasattr(self.fno, "blocks") else 6,
            "modes":           self.fno.blocks[0].spectral.modes[0] if hasattr(self.fno, "blocks") else 24,
        }
