"""Phase 2: Train the AtmosphereSurrogate FNO on generated spectra.

Writes two checkpoints to ``checkpoint_dir``, both loadable with
``plato_fno.models.load_surrogate`` (they carry the architecture and the
normaliser stats):

- ``best_surrogate.pt``: lowest validation loss
- ``last_surrogate.pt``: latest epoch, plus optimizer state

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
from omegaconf import DictConfig, OmegaConf
from solaris.datapipes.dataset import build_dataloader
from solaris.metrics.metrics import relative_l2_error, rmse
from solaris.utils.seed import set_seed
from solaris.utils.training import EarlyStopping, GradientClipper, WarmupCosineScheduler
from torch.amp import GradScaler, autocast

from plato_fno.data.dataset import load_raw_arrays, make_splits, split_indices
from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.models.surrogate import (
    AtmosphereSurrogate,
    apply_overtone_spectral_init,
    save_surrogate,
)
from plato_fno.utils import resolve_device, setup_run_logging


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE + 0.1 * relative L2 error (scale-invariant term)."""
    mse = F.mse_loss(pred, target)
    rel_l2 = relative_l2_error(pred, target)
    return mse + 0.1 * rel_l2


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    log = setup_run_logging("train")
    device = resolve_device(cfg.device)
    log.info(f"Device: {device}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = Path(cfg.data.output_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found at {hdf5_path}. Run scripts/generate_data.py first."
        )

    # ------------------------------------------------------------------
    # Fit the normaliser on the training split only
    # ------------------------------------------------------------------
    split_kwargs = dict(
        train_frac=cfg.train.train_frac,
        val_frac=cfg.train.val_frac,
        cal_frac=cfg.train.cal_frac,
        seed=cfg.seed,
    )
    train_idx = split_indices(hdf5_path, **split_kwargs)["train"]
    normalizer = SpectrumNormalizer().fit(*load_raw_arrays(hdf5_path, train_idx))

    splits = make_splits(hdf5_path, normalizer, **split_kwargs)
    log.info("Splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # Data lives in memory, so worker processes would only add overhead.
    train_loader = build_dataloader(
        splits["train"], batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    val_loader = build_dataloader(
        splits["val"], batch_size=cfg.train.batch_size * 2, shuffle=False, num_workers=0
    )

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

    # Overtone spectral init: reinitialise FNO Fourier weights with power-law
    # mode decay. Ported from parameter-golf. Skipped if exponent == 0.
    exponent = cfg.train.spectral_init_exponent
    if exponent > 0.0:
        apply_overtone_spectral_init(model, exponent=exponent)
        log.info(f"Applied overtone spectral init (exponent={exponent})")

    # ------------------------------------------------------------------
    # Optimiser and schedulers (Solaris utilities)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.beta1, cfg.train.beta2),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
    )
    early_stop = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="min")
    clipper = GradientClipper(max_norm=cfg.train.grad_clip)

    # Mixed precision. BF16 needs no loss scaling; FP16 needs a GradScaler.
    amp_dtype_str = cfg.train.amp_dtype
    if amp_dtype_str not in ("none", "bf16", "fp16"):
        raise ValueError(f"train.amp_dtype must be none, bf16 or fp16, got {amp_dtype_str!r}")
    use_amp = amp_dtype_str != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    scaler = GradScaler(device=device.type) if (use_amp and amp_dtype == torch.float16) else None
    log.info(f"Mixed precision: {amp_dtype_str if use_amp else 'disabled'}")

    run_info = {"config": OmegaConf.to_container(cfg, resolve=True)}
    best_val = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info("Starting training ...")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input"].to(device)  # (B, N_PARAMS)
            y = batch["target"].to(device)  # (B, 1, L)

            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(x)  # (B, 1, L)
                loss = loss_fn(pred, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clipper(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clipper(model)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation (always in fp32 for accurate metrics)
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
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            save_surrogate(
                ckpt_dir / "best_surrogate.pt",
                model,
                normalizer,
                epoch=epoch,
                val_loss=val_loss,
                **run_info,
            )
        save_surrogate(
            ckpt_dir / "last_surrogate.pt",
            model,
            normalizer,
            epoch=epoch,
            val_loss=val_loss,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            **run_info,
        )

        if early_stop.step(val_loss):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.success(
        f"Training complete. Best val_loss={best_val:.4e} at {ckpt_dir / 'best_surrogate.pt'}"
    )


if __name__ == "__main__":
    main()
