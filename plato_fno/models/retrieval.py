"""Bayesian atmospheric retrieval using FNO surrogate inside emcee.

The key performance gain comes from emcee's vectorize=True mode:
all walkers in a step are batched into a single GPU forward call (~5ms),
versus sequential petitRADTRANS calls (~0.5s each).

Speedup estimate:
    Sequential pRT: 64 walkers × 5000 steps × 0.5s = 44 hours
    FNO batched:    5000 steps × ~5ms/batch       = ~25 seconds
"""

from __future__ import annotations

from pathlib import Path

import emcee
import numpy as np
import torch

from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.physics.atmosphere import AtmosphericState, N_PARAMS


class MCMCRetrieval:
    """Bayesian atmospheric retrieval using FNO surrogate + emcee.

    Parameters
    ----------
    surrogate : nn.Module
        Trained AtmosphereSurrogate in eval mode.
    normalizer : SpectrumNormalizer
        Fitted normaliser — used to normalise input params and
        de-normalise predicted spectra for likelihood computation.
    conformal : ConformalNeuralOperator, optional
        If provided, q_hat is used as an adaptive noise floor per step.
    device : str or torch.device
        Device for FNO inference. Should be "cuda" for ROCm.
    """

    def __init__(
        self,
        surrogate: torch.nn.Module,
        normalizer: SpectrumNormalizer,
        conformal=None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.surrogate  = surrogate
        self.normalizer = normalizer
        self.conformal  = conformal
        self.device     = torch.device(device)

        self.surrogate.to(self.device)
        self.surrogate.eval()

        bounds = np.array(list(AtmosphericState.PRIOR_BOUNDS.values()), dtype=np.float64)
        self._lo, self._hi = bounds[:, 0], bounds[:, 1]

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    def log_prior(self, theta: np.ndarray) -> np.ndarray | float:
        """Uniform log-prior within ``AtmosphericState.PRIOR_BOUNDS``.

        Accepts (N_PARAMS,) or (B, N_PARAMS); returns 0 inside the prior and
        -inf outside (scalar or (B,) array accordingly).
        """
        theta = np.asarray(theta)
        inside = np.all((theta >= self._lo) & (theta <= self._hi), axis=-1)
        inside &= AtmosphericState.satisfies_constraints(theta)
        out = np.where(inside, 0.0, -np.inf)
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Likelihood (vectorised for emcee)
    # ------------------------------------------------------------------

    def log_likelihood_batch(
        self,
        theta_batch: np.ndarray,    # (B, N_PARAMS) un-normalised
        observed: np.ndarray,       # (n_wavelengths,) observed transit depth
        sigma: float,
    ) -> np.ndarray:
        """Batched Gaussian log-likelihood for a set of parameter vectors.

        Parameters
        ----------
        theta_batch : np.ndarray, shape (B, N_PARAMS)
            Un-normalised parameter vectors (physical units).
        observed : np.ndarray, shape (n_wavelengths,)
            Observed transit depth spectrum.
        sigma : float
            Per-wavelength observational noise standard deviation.

        Returns
        -------
        np.ndarray, shape (B,)
            Log-likelihood values.
        """
        # Normalise parameters
        theta_norm = self.normalizer.transform_params(theta_batch)  # (B, N_PARAMS)
        theta_t = torch.tensor(theta_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_norm = self.surrogate(theta_t)  # (B, 1, L)

        pred_norm_np = pred_norm.cpu().numpy()   # (B, 1, L)
        pred_phys = self.normalizer.inverse_transform_spectra(pred_norm_np)  # (B, 1, L)
        pred_phys = pred_phys[:, 0, :]           # (B, L)

        # Determine effective sigma — optionally inflate by conformal q_hat
        if self.conformal is not None and self.conformal.is_calibrated:
            q_hat_phys = float(self.conformal._q_hat) * self.normalizer.spec_std.mean()
            eff_sigma = max(sigma, float(q_hat_phys))
        else:
            eff_sigma = sigma

        # Gaussian log-likelihood: -0.5 * sum((obs - pred)^2 / sigma^2)
        n_wl = observed.shape[0]
        resid = pred_phys - observed[np.newaxis, :]  # (B, L)
        log_liks = -0.5 * np.sum(resid**2, axis=1) / (eff_sigma**2)
        log_liks -= 0.5 * n_wl * np.log(2.0 * np.pi * eff_sigma**2)

        return log_liks  # (B,)

    def log_posterior_vectorized(
        self,
        theta_batch: np.ndarray,
        observed: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Vectorised log-posterior for emcee's ``vectorize=True`` mode. Returns (B,)."""
        log_priors = self.log_prior(theta_batch)
        valid = np.isfinite(log_priors)
        out = np.full(len(theta_batch), -np.inf)
        if valid.any():
            out[valid] = self.log_likelihood_batch(theta_batch[valid], observed, sigma)
        return out

    # ------------------------------------------------------------------
    # MCMC run
    # ------------------------------------------------------------------

    def run(
        self,
        observed: np.ndarray,
        sigma: float = 50e-6,
        n_walkers: int = 64,
        n_steps: int = 5000,
        burn_in: int = 1000,
        thin: int = 5,
        initial_theta: np.ndarray | None = None,
        progress: bool = True,
    ) -> tuple[np.ndarray, emcee.EnsembleSampler]:
        """Run emcee ensemble sampler.

        Parameters
        ----------
        observed : np.ndarray, shape (n_wavelengths,)
            Observed transit depth spectrum.
        sigma : float
            Per-wavelength observational noise std dev.
        n_walkers : int
            Number of emcee walkers. Must be even and > 2 * N_PARAMS.
        n_steps : int
            Total MCMC steps (including burn-in).
        burn_in : int
            Steps to discard from the start.
        thin : int
            Thinning factor for posterior samples.
        initial_theta : np.ndarray, shape (N_PARAMS,), optional
            Starting point for walkers. Defaults to prior centre.
        progress : bool
            Show tqdm progress bar.

        Returns
        -------
        flat_samples : np.ndarray, shape (n_effective, N_PARAMS)
            Posterior samples after burn-in and thinning.
        sampler : emcee.EnsembleSampler
            Full sampler object for diagnostics.
        """
        assert n_walkers >= 2 * N_PARAMS, (
            f"n_walkers={n_walkers} must be >= 2*N_PARAMS={2*N_PARAMS}"
        )

        # Initialise walkers around the prior centre with small perturbations
        if initial_theta is None:
            bounds = list(AtmosphericState.PRIOR_BOUNDS.values())
            initial_theta = np.array([(lo + hi) / 2.0 for lo, hi in bounds])

        rng = np.random.default_rng(42)
        bounds_arr = np.array(list(AtmosphericState.PRIOR_BOUNDS.values()))
        width = (bounds_arr[:, 1] - bounds_arr[:, 0]) * 0.01
        p0 = initial_theta[np.newaxis, :] + rng.normal(0, 1, (n_walkers, N_PARAMS)) * width[np.newaxis, :]

        def log_prob(theta_batch: np.ndarray) -> np.ndarray:
            return self.log_posterior_vectorized(theta_batch, observed, sigma)

        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=N_PARAMS,
            log_prob_fn=log_prob,
            vectorize=True,
        )

        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        return flat_samples, sampler
