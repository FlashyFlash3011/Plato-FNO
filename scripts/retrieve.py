"""Phase 4: Run Bayesian MCMC atmospheric retrieval on a target spectrum.

Usage
-----
    # Retrieve from a stored HDF5 spectrum (index 0 by default)
    python scripts/retrieve.py

    # Specify target spectrum file and sample index
    python scripts/retrieve.py retrieval.target_spectrum=data/raw/target.h5 retrieval.target_index=5

    # Use a synthetic test spectrum (no target file needed)
    python scripts/retrieve.py retrieval.synthetic_test=true
"""

from __future__ import annotations

from pathlib import Path

import corner
import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from plato_fno.utils import resolve_device, setup_run_logging

from plato_fno.models.retrieval import MCMCRetrieval
from plato_fno.models.surrogate import load_surrogate
from plato_fno.physics.atmosphere import AtmosphericState, PARAM_NAMES


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_run_logging("retrieve")
    device = resolve_device(cfg.device)

    ckpt_dir  = Path(cfg.checkpoint_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load surrogate
    # ------------------------------------------------------------------
    model, normalizer, _ = load_surrogate(ckpt_dir / "best_surrogate.pt", device)

    log.info("Loaded surrogate and normalizer")

    # ------------------------------------------------------------------
    # Load or generate target spectrum
    # ------------------------------------------------------------------
    synthetic_test = OmegaConf.select(cfg, "retrieval.synthetic_test", default=False)

    if synthetic_test:
        # Generate a synthetic target using a known ground-truth atmosphere
        log.info("Generating synthetic target spectrum with known ground truth ...")
        rng = np.random.default_rng(0)
        true_state = AtmosphericState.sample_prior(rng)
        true_theta = true_state.to_vector()

        # Run surrogate to get the "observed" spectrum (noiseless)
        theta_norm = normalizer.transform_params(true_theta[np.newaxis, :])
        theta_t = torch.tensor(theta_norm, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(theta_t)  # (1, 1, L)
        pred_phys = normalizer.inverse_transform_spectra(pred_norm.cpu().numpy())[0, 0, :]

        # Add synthetic noise at 50 ppm
        rng2 = np.random.default_rng(1)
        observed = pred_phys + rng2.normal(0, cfg.retrieval.sigma, pred_phys.shape)

        log.info(f"True theta: {dict(zip(PARAM_NAMES, true_theta.round(3).tolist()))}")
        np.savez(results_dir / "synthetic_true_theta.npz", true_theta=true_theta)
    else:
        target_path = OmegaConf.select(cfg, "retrieval.target_spectrum", default=None)
        if target_path is None:
            raise ValueError(
                "Provide retrieval.target_spectrum=<path> or retrieval.synthetic_test=true"
            )
        target_idx = OmegaConf.select(cfg, "retrieval.target_index", default=0)

        with h5py.File(target_path, "r") as f:
            observed = f["target"][target_idx, 0, :]  # (L,)

        true_theta = None
        log.info(f"Loaded target spectrum from {target_path}[{target_idx}]")

    # ------------------------------------------------------------------
    # MCMC retrieval
    # ------------------------------------------------------------------
    retrieval = MCMCRetrieval(
        surrogate=model,
        normalizer=normalizer,
        device=device,
    )

    log.info(
        f"Starting MCMC: {cfg.retrieval.n_walkers} walkers, "
        f"{cfg.retrieval.n_steps} steps, burn_in={cfg.retrieval.burn_in}"
    )

    flat_samples, sampler = retrieval.run(
        observed=observed,
        sigma=cfg.retrieval.sigma,
        n_walkers=cfg.retrieval.n_walkers,
        n_steps=cfg.retrieval.n_steps,
        burn_in=cfg.retrieval.burn_in,
        thin=cfg.retrieval.thin,
        progress=cfg.retrieval.progress,
    )

    log.info(f"MCMC done | posterior samples: {flat_samples.shape[0]}")
    log.info(f"Mean acceptance fraction: {sampler.acceptance_fraction.mean():.3f}")

    # ------------------------------------------------------------------
    # Save posterior samples
    # ------------------------------------------------------------------
    samples_path = results_dir / "posterior_samples.npz"
    np.savez(samples_path, flat_samples=flat_samples, param_names=PARAM_NAMES)
    log.info(f"Posterior samples saved to {samples_path}")

    # ------------------------------------------------------------------
    # Corner plot
    # ------------------------------------------------------------------
    fig = corner.corner(
        flat_samples,
        labels=PARAM_NAMES,
        truths=true_theta if true_theta is not None else None,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 8},
    )
    fig.suptitle("PLATO-FNO Atmospheric Retrieval Posterior", y=1.01)
    corner_path = results_dir / "posterior_corner.png"
    fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.success(f"Corner plot saved to {corner_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    medians  = np.median(flat_samples, axis=0)
    lo       = np.quantile(flat_samples, 0.16, axis=0)
    hi       = np.quantile(flat_samples, 0.84, axis=0)

    log.info("\nPosterior summary (median ± 1σ):")
    for i, name in enumerate(PARAM_NAMES):
        err_lo = medians[i] - lo[i]
        err_hi = hi[i] - medians[i]
        truth_str = f"  [truth: {true_theta[i]:.3f}]" if true_theta is not None else ""
        log.info(f"  {name:15s}: {medians[i]:.3f} -{err_lo:.3f} +{err_hi:.3f}{truth_str}")


if __name__ == "__main__":
    main()
