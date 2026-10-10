"""Phase 5: Benchmark FNO surrogate accuracy and MCMC speedup vs petitRADTRANS.

Reports:
  1. Spectral accuracy on held-out test set (relative L2, RMSE in ppm, R²)
  2. Retrieval fidelity on 20 synthetic targets (posterior coverage, bias)
  3. Inference speedup vs petitRADTRANS

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py evaluate.n_retrieval_targets=5  # quick smoke test
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from solaris.metrics.metrics import relative_l2_error, rmse, r2_score
from plato_fno.utils import resolve_device, setup_run_logging
from solaris.datapipes.dataset import build_dataloader

from plato_fno.data.dataset import make_splits
from plato_fno.metrics.retrieval_metrics import (
    benchmark_inference_time,
    parameter_bias,
    posterior_coverage,
    speedup_ratio,
)
from plato_fno.models.retrieval import MCMCRetrieval, convergence_diagnostics
from plato_fno.models.surrogate import load_surrogate
from plato_fno.physics.atmosphere import AtmosphericState, PARAM_NAMES


def _time_prt(cfg: DictConfig, log) -> tuple[float, str]:
    """Time real petitRADTRANS calls, or fall back to the configured estimate.

    Returns (seconds per call, "measured" or "assumed").
    """
    if not cfg.evaluate.time_prt:
        return float(cfg.evaluate.prt_seconds_per_call), "assumed"
    try:
        from plato_fno.physics.petitrt_driver import build_radtrans, compute_spectrum

        prt, pressures, wavelengths = build_radtrans(
            list(cfg.data.species),
            tuple(cfg.data.wavelength_range),
            cfg.data.n_wavelengths,
            cfg.data.n_layers,
        )
        rng = np.random.default_rng(cfg.seed)
        states = [
            AtmosphericState.sample_prior(rng) for _ in range(cfg.evaluate.n_prt_timing_calls)
        ]
        compute_spectrum(states[0], prt, pressures, wavelengths)  # warm-up
        t0 = time.perf_counter()
        for state in states:
            compute_spectrum(state, prt, pressures, wavelengths)
        return (time.perf_counter() - t0) / len(states), "measured"
    except Exception as exc:  # pRT or its opacity data missing
        log.warning(
            f"Could not time petitRADTRANS ({exc}); "
            f"assuming {cfg.evaluate.prt_seconds_per_call} s per call"
        )
        return float(cfg.evaluate.prt_seconds_per_call), "assumed"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_run_logging("evaluate")
    device = resolve_device(cfg.device)

    ckpt_dir    = Path(cfg.checkpoint_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path   = Path(cfg.data.output_path)

    n_retrieval_targets = cfg.evaluate.n_retrieval_targets

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, normalizer, _ = load_surrogate(ckpt_dir / "best_surrogate.pt", device)

    log.info("Model and normalizer loaded")

    # ------------------------------------------------------------------
    # 1. Spectral accuracy on test split
    # ------------------------------------------------------------------
    log.info("=== 1. Spectral accuracy (test split) ===")

    test_ds = make_splits(
        hdf5_path,
        normalizer,
        train_frac=cfg.train.train_frac,
        val_frac=cfg.train.val_frac,
        cal_frac=cfg.train.cal_frac,
        seed=cfg.seed,
    )["test"]
    test_loader = build_dataloader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    preds   = torch.cat(all_preds,   dim=0)  # (N, 1, L)
    targets = torch.cat(all_targets, dim=0)  # (N, 1, L)

    rel_l2 = relative_l2_error(preds, targets).item()
    rmse_norm = rmse(preds, targets).item()
    r2    = r2_score(preds, targets).item()

    # RMSE in physical units (the normalised scale differs per wavelength)
    preds_phys = normalizer.inverse_transform_spectra(preds.numpy())
    targets_phys = normalizer.inverse_transform_spectra(targets.numpy())
    rmse_per_wl_ppm = np.sqrt(((preds_phys - targets_phys) ** 2).mean(axis=(0, 1))) * 1e6
    rmse_ppm = float(np.sqrt((rmse_per_wl_ppm**2).mean()))

    spectral_metrics = {
        "relative_l2":  rel_l2,
        "rmse_normalised": rmse_norm,
        "rmse_ppm":     rmse_ppm,
        "rmse_ppm_worst_wavelength": float(rmse_per_wl_ppm.max()),
        "r2_score":     r2,
        "n_test_samples": len(test_ds),
    }

    log.info(f"  Relative L2:   {rel_l2:.4f}")
    log.info(
        f"  RMSE:          {rmse_ppm:.2f} ppm "
        f"(worst wavelength {rmse_per_wl_ppm.max():.2f} ppm)"
    )
    log.info(f"  R² score:      {r2:.4f}")

    # ------------------------------------------------------------------
    # 2. Inference speedup
    # ------------------------------------------------------------------
    log.info("=== 2. Inference speedup ===")

    t_fno_single = benchmark_inference_time(model, batch_size=1, device=device)
    t_fno_batch = benchmark_inference_time(
        model, batch_size=cfg.retrieval.n_walkers, device=device
    )
    t_prt, prt_timing = _time_prt(cfg, log)

    n_walkers = cfg.retrieval.n_walkers
    speedup = speedup_ratio(t_prt, t_fno_single, t_fno_batch, n_walkers=n_walkers)
    speedup["t_fno_single_ms"] = t_fno_single * 1e3
    speedup["t_fno_batch_ms"] = t_fno_batch * 1e3
    speedup["t_prt_s"] = t_prt
    speedup["prt_timing"] = prt_timing

    log.info(f"  FNO single call:  {t_fno_single * 1e3:.2f} ms")
    log.info(f"  FNO batch={n_walkers}:     {t_fno_batch * 1e3:.2f} ms")
    log.info(f"  pRT per call:     {t_prt * 1e3:.0f} ms ({prt_timing})")
    log.info(f"  Single speedup:   {speedup['single_call']:.0f}×")
    log.info(
        f"  Batched speedup:  {speedup['batched_step']:.0f}× "
        f"(per MCMC step, {n_walkers} walkers)"
    )

    # ------------------------------------------------------------------
    # 3. Retrieval fidelity on synthetic targets
    # ------------------------------------------------------------------
    log.info(f"=== 3. Retrieval fidelity ({n_retrieval_targets} synthetic targets) ===")

    # Targets are petitRADTRANS spectra from the test split (never seen in
    # training or calibration) plus Gaussian noise, so coverage reflects the
    # surrogate's error as well as the sampler's.
    model_sigma = None
    state_path = ckpt_dir / "conformal_state.pt"
    if cfg.retrieval.include_model_error and state_path.exists():
        model_sigma = np.asarray(torch.load(state_path, weights_only=False)["model_sigma"])
    elif cfg.retrieval.include_model_error:
        log.warning(f"{state_path} missing; retrieving without the surrogate error term")
    retrieval = MCMCRetrieval(model, normalizer, device=device, model_sigma=model_sigma)

    coverage_per_param: dict[str, list[bool]] = {name: [] for name in PARAM_NAMES}
    bias_per_param: dict[str, list[float]] = {name: [] for name in PARAM_NAMES}
    n_converged = 0

    rng = np.random.default_rng(cfg.seed + 100)
    picks = rng.choice(len(test_ds), size=min(n_retrieval_targets, len(test_ds)), replace=False)
    true_thetas = normalizer.inverse_transform_params(test_ds.inputs[picks].numpy())
    true_spectra = normalizer.inverse_transform_spectra(test_ds.targets[picks].numpy())[:, 0, :]

    for i in tqdm(range(len(picks)), desc="Retrieval fidelity"):
        true_theta = true_thetas[i]
        observed = true_spectra[i] + rng.normal(0, cfg.retrieval.sigma, true_spectra[i].shape)

        flat_samples, sampler = retrieval.run(
            observed=observed,
            sigma=cfg.retrieval.sigma,
            n_walkers=cfg.retrieval.n_walkers,
            n_steps=cfg.evaluate.mcmc_n_steps,
            burn_in=cfg.evaluate.mcmc_burn_in,
            thin=cfg.retrieval.thin,
            progress=False,
            seed=cfg.seed + i,
        )
        n_converged += convergence_diagnostics(sampler, cfg.evaluate.mcmc_burn_in)["converged"]

        cov = posterior_coverage(flat_samples, true_theta, credible_level=0.68)
        bia = parameter_bias(flat_samples, true_theta)
        for name in PARAM_NAMES:
            coverage_per_param[name].append(cov[name])
            bias_per_param[name].append(bia[name])

    mean_coverage = {name: float(np.mean(v)) for name, v in coverage_per_param.items()}
    mean_bias     = {name: float(np.mean(v)) for name, v in bias_per_param.items()}
    overall_coverage = float(np.mean(list(mean_coverage.values())))

    log.info(f"  Overall 68% CI coverage: {overall_coverage:.3f}  (ideal ≈ 0.68)")
    log.info(f"  Converged chains: {n_converged}/{len(picks)}")
    if n_converged < len(picks):
        log.warning(
            "Some chains did not converge; raise evaluate.mcmc_n_steps before trusting coverage"
        )
    log.info("  Per-parameter coverage:")
    for name in PARAM_NAMES:
        log.info(f"    {name:15s}: coverage={mean_coverage[name]:.2f}  bias={mean_bias[name]:+.4f}")

    # ------------------------------------------------------------------
    # Save all metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "spectral_accuracy": spectral_metrics,
        "speedup":           speedup,
        "retrieval_fidelity": {
            "n_targets":       len(picks),
            "n_converged":     n_converged,
            "overall_coverage_68": overall_coverage,
            "per_param_coverage":  mean_coverage,
            "per_param_mean_bias": mean_bias,
        },
    }

    metrics_path = results_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.success(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
