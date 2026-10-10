"""Phase 3: Wrap trained surrogate in ConformalNeuralOperator and calibrate.

Calibration uses the held-out calibration split (10% of data).
After this step, predict() returns (lower, upper, point) with
guaranteed ≥(1-alpha) coverage over the wavelength axis.

Usage
-----
    python scripts/calibrate.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from solaris.models.conformal import ConformalNeuralOperator
from solaris.utils.checkpoint import load_checkpoint
from solaris.utils.logging import get_logger

from plato_fno.data.dataset import make_splits
from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.models.surrogate import AtmosphereSurrogate
from solaris.datapipes.dataset import build_dataloader


class _SurrogateAdapter(torch.nn.Module):
    """Thin adapter so ConformalNeuralOperator can wrap AtmosphereSurrogate.

    ConformalNeuralOperator calls model(x) where x must be the same tensor
    passed to calibrate(cal_inputs, ...). Here x is (B, N_PARAMS).
    """

    def __init__(self, surrogate: AtmosphereSurrogate) -> None:
        super().__init__()
        self.surrogate = surrogate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.surrogate(x)  # (B, N_PARAMS) → (B, 1, L)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger(__name__)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg.checkpoint_dir)
    hdf5_path = Path(cfg.data.output_path)

    # ------------------------------------------------------------------
    # Load trained surrogate
    # ------------------------------------------------------------------
    model = AtmosphereSurrogate(
        n_wavelengths=cfg.model.n_wavelengths,
        n_enc=cfg.model.n_enc,
        hidden_channels=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        modes=cfg.model.modes,
    )
    ckpt = torch.load(ckpt_dir / "best_surrogate.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt["model_state_dict"])
    model.to(device).eval()
    log.info("Loaded best_surrogate.pt")

    # ------------------------------------------------------------------
    # Load normalizer
    # ------------------------------------------------------------------
    normalizer = SpectrumNormalizer.load(ckpt_dir / "norm_stats.npz")

    # ------------------------------------------------------------------
    # Calibration split
    # ------------------------------------------------------------------
    cal_ds = make_splits(
        hdf5_path,
        normalizer,
        train_frac=cfg.train.train_frac,
        val_frac=cfg.train.val_frac,
        cal_frac=cfg.train.cal_frac,
        seed=cfg.seed,
    )["cal"]
    log.info(f"Calibration set size: {len(cal_ds)}")

    cal_loader = build_dataloader(cal_ds, batch_size=256, shuffle=False)

    # Collect all calibration inputs and targets as tensors
    all_inputs  = []
    all_targets = []
    for batch in cal_loader:
        all_inputs.append(batch["input"])
        all_targets.append(batch["target"])

    cal_inputs  = torch.cat(all_inputs,  dim=0).to(device)   # (N_cal, N_PARAMS)
    cal_targets = torch.cat(all_targets, dim=0).to(device)   # (N_cal, 1, L)

    # ------------------------------------------------------------------
    # Conformal calibration
    # ------------------------------------------------------------------
    adapter   = _SurrogateAdapter(model)
    predictor = ConformalNeuralOperator(adapter)

    alpha = cfg.train.conformal_alpha
    q_hat = predictor.calibrate(cal_inputs, cal_targets, alpha=alpha, batch_size=256)

    log.info(f"Conformal calibration done | alpha={alpha} | q_hat (normalised)={q_hat:.4f}")

    # Convert q_hat to physical units (ppm) for reporting
    q_hat_ppm = float(q_hat) * float(normalizer.spec_std.mean()) * 1e6
    log.info(f"q_hat ≈ {q_hat_ppm:.1f} ppm (physical transit depth units)")

    # Coverage check on calibration set
    report = predictor.coverage_report(cal_inputs, cal_targets, batch_size=256)
    log.info(
        f"Calibration coverage: {report['coverage']:.3f} "
        f"(target ≥ {1-alpha:.2f}) | "
        f"mean interval width: {report['mean_interval_width']:.4f}"
    )

    if report["coverage"] < (1 - alpha):
        log.warning("Coverage is below target — check calibration set size or model quality")

    # ------------------------------------------------------------------
    # Save conformal predictor state
    # ------------------------------------------------------------------
    torch.save(
        {
            "q_hat":          predictor._q_hat,
            "is_calibrated":  predictor.is_calibrated,
            "alpha":          alpha,
            "coverage_report": report,
        },
        ckpt_dir / "conformal_state.pt",
    )
    log.success(f"Conformal state saved to {ckpt_dir / 'conformal_state.pt'}")


if __name__ == "__main__":
    main()
