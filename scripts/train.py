"""Phase 2: Train the AtmosphereSurrogate FNO on generated spectra.

Follows the training loop pattern from Solaris/projects/hurricane_flood/train.py.

Usage
-----
    python scripts/train.py
    python scripts/train.py train.epochs=100 train.batch_size=128
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig

from solaris.metrics.metrics import relative_l2_error, rmse
from solaris.utils.checkpoint import save_checkpoint
from solaris.utils.logging import get_logger
from solaris.utils.seed import set_seed
from solaris.utils.training import AutoCheckpoint, EarlyStopping, GradientClipper, WarmupCosineScheduler
from solaris.datapipes.dataset import build_dataloader

from plato_fno.data.dataset import make_splits, load_raw_arrays
from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.models.surrogate import AtmosphereSurrogate

import h5py
import numpy as np


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE + 0.1 * relative L2 error (scale-invariant term)."""
    mse = F.mse_loss(pred, target)
    rel_l2 = relative_l2_error(pred, target)
    return mse + 0.1 * rel_l2


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    log = get_logger(__name__)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = Path(cfg.data.output_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found at {hdf5_path}. "
            "Run scripts/generate_data.py first."
        )

    # ------------------------------------------------------------------
    # Discover training indices and fit normalizer on training split only
    # ------------------------------------------------------------------
    log.info("Computing normalisation statistics from training split ...")

    with h5py.File(hdf5_path, "r") as f:
        targets = f["target"][:]
    valid_mask = np.isfinite(targets).all(axis=(1, 2))
    all_valid = np.where(valid_mask)[0]

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_valid)

    n = len(all_valid)
    n_train = int(n * cfg.train.train_frac)
    n_val   = int(n * cfg.train.val_frac)
    train_idx = all_valid[:n_train]

    train_params, train_spectra = load_raw_arrays(hdf5_path, train_idx)
    normalizer = SpectrumNormalizer().fit(train_params, train_spectra)
    normalizer.save(ckpt_dir / "norm_stats.npz")
    log.info(f"Normalizer saved to {ckpt_dir / 'norm_stats.npz'}")

    # ------------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------------
    train_ds, val_ds, cal_ds = make_splits(
        hdf5_path,
        normalizer,
        train_frac=cfg.train.train_frac,
        val_frac=cfg.train.val_frac,
        seed=cfg.seed,
    )
    log.info(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, cal: {len(cal_ds)}")

    train_loader = build_dataloader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = build_dataloader(val_ds,   batch_size=cfg.train.batch_size * 2, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AtmosphereSurrogate(
        n_wavelengths=cfg.model.n_wavelengths,
        n_enc=cfg.model.n_enc,
        hidden_channels=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        modes=cfg.model.modes,
    ).to(device)

    log.info(f"AtmosphereSurrogate: {model.num_parameters():,} parameters")

    # ------------------------------------------------------------------
    # Optimiser and schedulers (Solaris utilities)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
    )
    early_stop = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="min")
    clipper    = GradientClipper(max_norm=cfg.train.grad_clip)
    auto_ckpt  = AutoCheckpoint(ckpt_dir / "best_surrogate.pt", mode="min")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info("Starting training ...")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input"].to(device)    # (B, N_PARAMS)
            y = batch["target"].to(device)   # (B, 1, L)

            optimizer.zero_grad()
            pred = model(x)                  # (B, 1, L)
            loss = loss_fn(pred, y)
            loss.backward()
            clipper(model)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
                val_rmse += rmse(pred, y).item()

        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{cfg.train.epochs} | "
                f"train_loss={train_loss:.4e} | "
                f"val_loss={val_loss:.4e} | "
                f"val_rmse={val_rmse:.4e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Checkpoint + early stopping
        save_checkpoint(
            ckpt_dir / "last_surrogate.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            extra=model.architecture_kwargs(),
        )
        auto_ckpt(val_loss, model, optimizer, scheduler, epoch)

        if early_stop(val_loss):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.success(f"Training complete. Best model at {ckpt_dir / 'best_surrogate.pt'}")


if __name__ == "__main__":
    main()
