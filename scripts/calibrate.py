"""Phase 3: Wrap trained surrogate in ConformalNeuralOperator and calibrate.

Calibration uses the held-out ``cal`` split; the coverage check then runs on
the separate ``test`` split (checking on the calibration data itself would
pass by construction).

Conformal guarantee: for a new sample, the whole true spectrum lies inside
``prediction ± q_hat`` (in normalised units) with probability ≥ 1 - alpha.
In physical units that band is ``q_hat * spec_std(λ)`` wide, so it differs per
wavelength; ``band_ppm`` in the saved state holds it.

The band is a worst-case bound over the whole spectrum, not a 1σ error, so the
retrieval does not use it as noise. It uses ``model_sigma`` instead: the
surrogate's RMSE per wavelength on the calibration set.

Usage
-----
    python scripts/calibrate.py
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from solaris.models.conformal import ConformalNeuralOperator

from plato_fno.data.dataset import make_splits
from plato_fno.models.surrogate import load_surrogate
from plato_fno.utils import resolve_device, setup_run_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_run_logging("calibrate")
    device = resolve_device(cfg.device)

    ckpt_dir = Path(cfg.checkpoint_dir)
    model, normalizer, _ = load_surrogate(ckpt_dir / "best_surrogate.pt", device)
    log.info("Loaded best_surrogate.pt")

    splits = make_splits(
        Path(cfg.data.output_path),
        normalizer,
        train_frac=cfg.train.train_frac,
        val_frac=cfg.train.val_frac,
        cal_frac=cfg.train.cal_frac,
        seed=cfg.seed,
    )
    cal_inputs, cal_targets = splits["cal"].inputs, splits["cal"].targets
    test_inputs, test_targets = splits["test"].inputs, splits["test"].targets
    log.info(f"Calibration set: {len(cal_inputs)} | test set: {len(test_inputs)}")

    predictor = ConformalNeuralOperator(model)
    alpha = cfg.train.conformal_alpha
    q_hat = predictor.calibrate(cal_inputs, cal_targets, alpha=alpha, batch_size=256)
    log.info(f"Conformal calibration done | alpha={alpha} | q_hat (normalised)={q_hat:.4f}")

    band_ppm = q_hat * normalizer.spec_std_ppm()  # (L,)
    log.info(
        f"Band half-width: median {np.median(band_ppm):.1f} ppm "
        f"(range {band_ppm.min():.1f}–{band_ppm.max():.1f} ppm across wavelength)"
    )

    # Surrogate RMSE per wavelength on the calibration set, in transit-depth
    # units: the retrieval adds this to the observational noise.
    with torch.no_grad():
        cal_pred = model(cal_inputs.to(device)).cpu().numpy()
    resid = normalizer.inverse_transform_spectra(cal_pred) - normalizer.inverse_transform_spectra(
        cal_targets.numpy()
    )
    model_sigma = np.sqrt((resid**2).mean(axis=(0, 1)))  # (L,)
    log.info(f"Surrogate RMSE: median {np.median(model_sigma) * 1e6:.1f} ppm per wavelength")

    report = predictor.coverage_report(test_inputs, test_targets, batch_size=256)
    log.info(
        f"Test-set coverage: {report['coverage']:.3f} (target ≥ {1 - alpha:.2f}) | "
        f"mean interval width (normalised): {report['mean_interval_width']:.4f}"
    )
    if report["coverage"] < (1 - alpha):
        log.warning("Coverage is below target — check calibration set size or model quality")

    out_path = ckpt_dir / "conformal_state.pt"
    torch.save(
        {
            "q_hat": q_hat,
            "alpha": alpha,
            "band_ppm": band_ppm,
            "model_sigma": model_sigma,
            "test_coverage_report": report,
        },
        out_path,
    )
    log.success(f"Conformal state saved to {out_path}")


if __name__ == "__main__":
    main()
