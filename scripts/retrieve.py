"""Phase 4: Run Bayesian MCMC atmospheric retrieval on a target spectrum.

Target file (HDF5):
    target       (N, 1, M) or (N, M)  transit depth (Rp/Rs)^2
    sigma        optional, (N, M) or (M,)  per-point 1σ errors
                 (otherwise retrieval.sigma is used for every point)
    wavelengths  optional, (M,) µm; checked against the expected grid

M is the model's native grid (300 points, 0.3–5.0 µm), or an instrument grid
when retrieval.instrument is set.

Usage
-----
    python scripts/retrieve.py retrieval.synthetic_test=true
    python scripts/retrieve.py retrieval.target_spectrum=target.h5 retrieval.target_index=5
    python scripts/retrieve.py retrieval.synthetic_test=true retrieval.instrument=nirspec_prism
"""

from __future__ import annotations

import json
from pathlib import Path

import corner
import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from plato_fno.models.retrieval import MCMCRetrieval, convergence_diagnostics
from plato_fno.models.surrogate import load_surrogate
from plato_fno.physics.atmosphere import PARAM_NAMES, AtmosphericState
from plato_fno.physics.jwst_instrument import (
    JWST_INSTRUMENTS,
    bin_centers,
    make_bin_transform,
    simulate_jwst_observation,
)
from plato_fno.utils import resolve_device, setup_run_logging


def observation_setup(cfg: DictConfig, wavelengths: np.ndarray, model_sigma: np.ndarray | None):
    """Return (obs_wavelengths, obs_transform, model_sigma on the observation grid)."""
    instrument = cfg.retrieval.instrument
    if instrument is None:
        return wavelengths, None, model_sigma

    lam_min, lam_max, n_bins = JWST_INSTRUMENTS[instrument]
    transform = make_bin_transform(wavelengths, instrument)
    centers = bin_centers(lam_min, lam_max, n_bins)
    if model_sigma is not None:
        # Conservative: averaging spectra inside a bin does not average away
        # surrogate errors (they are correlated across neighbouring points).
        model_sigma = np.sqrt(transform(model_sigma[np.newaxis, :] ** 2)[0])
    return centers, transform, model_sigma


def load_target(path: str, index: int, n_points: int, obs_wavelengths: np.ndarray, default_sigma):
    with h5py.File(path, "r") as f:
        target = np.asarray(f["target"][index], dtype=np.float64).reshape(-1)
        if "sigma" in f:
            sig = f["sigma"]
            sigma = np.asarray(sig[index] if sig.ndim == 2 else sig[:], dtype=np.float64)
        else:
            sigma = np.full(n_points, float(default_sigma))
        file_wavelengths = f["wavelengths"][:] if "wavelengths" in f else None

    if target.shape[0] != n_points:
        raise ValueError(
            f"Target has {target.shape[0]} points but the observation grid has {n_points}. "
            "Resample it, or set retrieval.instrument to match how it was binned."
        )
    if file_wavelengths is not None and not np.allclose(file_wavelengths, obs_wavelengths, atol=1e-4):
        raise ValueError("Target wavelengths do not match the expected observation grid.")
    return target, sigma


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_run_logging("retrieve")
    device = resolve_device(cfg.device)
    rc = cfg.retrieval

    ckpt_dir = Path(cfg.checkpoint_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer, ckpt = load_surrogate(ckpt_dir / "best_surrogate.pt", device)
    wavelengths = np.asarray(ckpt["wavelengths"])
    log.info("Loaded surrogate and normalizer")

    model_sigma = None
    if rc.include_model_error:
        state_path = ckpt_dir / "conformal_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"{state_path} not found. Run scripts/calibrate.py first, "
                "or set retrieval.include_model_error=false."
            )
        model_sigma = np.asarray(torch.load(state_path, weights_only=False)["model_sigma"])

    obs_wavelengths, obs_transform, model_sigma = observation_setup(cfg, wavelengths, model_sigma)
    n_points = len(obs_wavelengths)

    retrieval = MCMCRetrieval(
        surrogate=model,
        normalizer=normalizer,
        device=device,
        obs_transform=obs_transform,
        model_sigma=model_sigma,
    )

    # ------------------------------------------------------------------
    # Target spectrum
    # ------------------------------------------------------------------
    if rc.synthetic_test:
        # Known ground truth. The "observation" is made by the surrogate itself,
        # so this checks the sampler, not the surrogate's accuracy.
        rng = np.random.default_rng(rc.seed)
        true_theta = AtmosphericState.sample_prior(rng).to_vector()
        native = MCMCRetrieval(model, normalizer, device).predict_observed(true_theta[None])[0]
        if rc.instrument is None:
            sigma = np.full(n_points, float(rc.sigma))
            observed = native + rng.normal(0.0, sigma)
        else:
            observed, _, sigma = simulate_jwst_observation(
                native, wavelengths, instrument=rc.instrument, rng=rng
            )
        truth = {n: round(float(v), 3) for n, v in zip(PARAM_NAMES, true_theta)}
        log.info(f"Synthetic target, truth: {truth}")
    else:
        target_path, target_idx = rc.target_spectrum, rc.target_index
        if target_path is None:
            raise ValueError(
                "Provide retrieval.target_spectrum=<path> or retrieval.synthetic_test=true"
            )
        observed, sigma = load_target(target_path, target_idx, n_points, obs_wavelengths, rc.sigma)
        true_theta = None
        log.info(f"Loaded target spectrum from {target_path}[{target_idx}]")

    # ------------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------------
    log.info(f"Starting MCMC: {rc.n_walkers} walkers, {rc.n_steps} steps, burn_in={rc.burn_in}")
    flat_samples, sampler = retrieval.run(
        observed=observed,
        sigma=sigma,
        n_walkers=rc.n_walkers,
        n_steps=rc.n_steps,
        burn_in=rc.burn_in,
        thin=rc.thin,
        progress=rc.progress,
        seed=rc.seed,
    )
    diag = convergence_diagnostics(sampler, burn_in=rc.burn_in)
    log.info(
        f"MCMC done | {flat_samples.shape[0]} samples | acceptance {diag['acceptance']:.3f} | "
        f"max τ {diag['max_tau']:.0f} steps | ~{diag['n_independent']:.0f} independent samples"
    )
    if not diag["converged"]:
        log.warning(
            "Chain is probably too short to trust: aim for n_steps - burn_in ≥ 50 τ and "
            "burn_in ≥ 2 τ. Increase retrieval.n_steps / retrieval.burn_in."
        )

    np.savez(
        results_dir / "posterior_samples.npz",
        flat_samples=flat_samples,
        param_names=PARAM_NAMES,
        **({"true_theta": true_theta} if true_theta is not None else {}),
    )
    with open(results_dir / "mcmc_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    fig = corner.corner(
        flat_samples,
        labels=PARAM_NAMES,
        truths=true_theta,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 8},
    )
    fig.suptitle("PLATO-FNO Atmospheric Retrieval Posterior", y=1.01)
    corner_path = results_dir / "posterior_corner.png"
    fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.success(f"Results saved to {results_dir}")

    medians = np.median(flat_samples, axis=0)
    lo = np.quantile(flat_samples, 0.16, axis=0)
    hi = np.quantile(flat_samples, 0.84, axis=0)
    log.info("Posterior summary (median, 16th–84th percentile):")
    for i, name in enumerate(PARAM_NAMES):
        truth_str = f"  [truth: {true_theta[i]:.3f}]" if true_theta is not None else ""
        log.info(
            f"  {name:15s}: {medians[i]:.3f} -{medians[i] - lo[i]:.3f} "
            f"+{hi[i] - medians[i]:.3f}{truth_str}"
        )


if __name__ == "__main__":
    main()
