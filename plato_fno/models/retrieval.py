"""Bayesian atmospheric retrieval using the FNO surrogate inside emcee.

The performance gain comes from emcee's ``vectorize=True`` mode: all walkers
in a step are batched into a single forward call of the surrogate, instead of
one petitRADTRANS call per walker. Run ``scripts/evaluate.py`` for measured
timings.

Likelihood
----------
Gaussian, independent per spectral point::

    log L = -0.5 * Σ [ (d - m)^2 / s^2 + log(2π s^2) ],   s^2 = σ_obs^2 + σ_model^2

``σ_obs`` is the observational error (scalar or per point). ``σ_model`` is
optional: the surrogate's own per-wavelength RMSE, measured on held-out data
by ``scripts/calibrate.py``. Adding it in quadrature stops the posterior from
being over-confident where the surrogate is less accurate than the data.
"""

from __future__ import annotations

from collections.abc import Callable

import emcee
import numpy as np
import torch

from plato_fno.data.normalizer import SpectrumNormalizer
from plato_fno.physics.atmosphere import N_PARAMS, AtmosphericState

ObsTransform = Callable[[np.ndarray], np.ndarray]


class MCMCRetrieval:
    """Bayesian atmospheric retrieval using FNO surrogate + emcee.

    Parameters
    ----------
    surrogate : nn.Module
        Trained AtmosphereSurrogate.
    normalizer : SpectrumNormalizer
        The normaliser stored with the surrogate's checkpoint.
    device : str or torch.device
        Device for surrogate inference.
    obs_transform : callable, optional
        Maps predicted spectra (B, L) on the native grid to the observation
        grid (B, M), e.g. ``make_bin_transform`` for a JWST instrument.
        Defaults to identity (observation on the native grid).
    model_sigma : np.ndarray, optional
        Surrogate error per observed point, shape (M,), in transit-depth units,
        added in quadrature to the observational error.
    """

    def __init__(
        self,
        surrogate: torch.nn.Module,
        normalizer: SpectrumNormalizer,
        device: str | torch.device = "cpu",
        obs_transform: ObsTransform | None = None,
        model_sigma: np.ndarray | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.obs_transform = obs_transform
        self.model_sigma = None if model_sigma is None else np.asarray(model_sigma, np.float64)

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

    def predict_observed(self, theta_batch: np.ndarray) -> np.ndarray:
        """Surrogate spectra for (B, N_PARAMS) physical parameters, on the observation grid."""
        theta_norm = self.normalizer.transform_params(theta_batch)
        theta_t = torch.as_tensor(theta_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_norm = self.surrogate(theta_t).cpu().numpy()  # (B, 1, L)
        pred = self.normalizer.inverse_transform_spectra(pred_norm)[:, 0, :]  # (B, L)
        if self.obs_transform is not None:
            pred = self.obs_transform(pred)
        return pred.astype(np.float64)

    def log_likelihood_batch(
        self,
        theta_batch: np.ndarray,
        observed: np.ndarray,
        sigma: float | np.ndarray,
    ) -> np.ndarray:
        """Batched Gaussian log-likelihood.

        Parameters
        ----------
        theta_batch : np.ndarray, shape (B, N_PARAMS)
            Physical (un-normalised) parameter vectors.
        observed : np.ndarray, shape (M,)
            Observed transit depths on the observation grid.
        sigma : float or np.ndarray of shape (M,)
            Observational 1σ errors.

        Returns
        -------
        np.ndarray, shape (B,)
        """
        observed = np.asarray(observed, dtype=np.float64)
        var = np.broadcast_to(np.asarray(sigma, dtype=np.float64) ** 2, observed.shape)
        if self.model_sigma is not None:
            var = var + self.model_sigma**2

        pred = self.predict_observed(theta_batch)  # (B, M)
        if pred.shape[1] != observed.shape[0]:
            raise ValueError(
                f"Prediction has {pred.shape[1]} points but the observation has "
                f"{observed.shape[0]}; pass an obs_transform that maps between them."
            )
        resid = pred - observed[np.newaxis, :]
        return -0.5 * (np.sum(resid**2 / var, axis=1) + np.sum(np.log(2.0 * np.pi * var)))

    def log_posterior_vectorized(
        self,
        theta_batch: np.ndarray,
        observed: np.ndarray,
        sigma: float | np.ndarray,
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

    def initial_walkers(
        self,
        n_walkers: int,
        rng: np.random.Generator,
        initial_theta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Starting positions, shape (n_walkers, N_PARAMS).

        Without ``initial_theta`` the walkers are drawn from the prior, which
        avoids starting every chain in the same spot. With it, they form a
        small ball (1% of each prior width) around that point.
        """
        if initial_theta is None:
            return np.stack(
                [AtmosphericState.sample_prior(rng).to_vector() for _ in range(n_walkers)]
            ).astype(np.float64)
        width = (self._hi - self._lo) * 0.01
        p0 = initial_theta[np.newaxis, :] + rng.normal(0, 1, (n_walkers, N_PARAMS)) * width
        return np.clip(p0, self._lo, self._hi)

    def run(
        self,
        observed: np.ndarray,
        sigma: float | np.ndarray = 50e-6,
        n_walkers: int = 64,
        n_steps: int = 5000,
        burn_in: int = 1000,
        thin: int = 5,
        initial_theta: np.ndarray | None = None,
        progress: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, emcee.EnsembleSampler]:
        """Run the emcee ensemble sampler.

        Parameters
        ----------
        observed : np.ndarray, shape (M,)
            Observed transit depth spectrum (on the observation grid).
        sigma : float or np.ndarray of shape (M,)
            Observational 1σ errors.
        n_walkers : int
            Number of walkers; must be at least 2 * N_PARAMS.
        n_steps : int
            Total MCMC steps (including burn-in).
        burn_in : int
            Steps to discard from the start.
        thin : int
            Thinning factor for posterior samples.
        initial_theta : np.ndarray, shape (N_PARAMS,), optional
            Start walkers in a small ball here instead of across the prior.
        progress : bool
            Show tqdm progress bar.
        seed : int, optional
            Seed for walker initialisation and emcee's proposals.

        Returns
        -------
        flat_samples : np.ndarray, shape (n_effective, N_PARAMS)
        sampler : emcee.EnsembleSampler
            Use :func:`convergence_diagnostics` on it before trusting the result.
        """
        if n_walkers < 2 * N_PARAMS:
            raise ValueError(f"n_walkers={n_walkers} must be >= 2*N_PARAMS={2 * N_PARAMS}")
        if not 0 <= burn_in < n_steps:
            raise ValueError(f"burn_in={burn_in} must be in [0, n_steps={n_steps})")

        rng = np.random.default_rng(seed)
        p0 = self.initial_walkers(n_walkers, rng, initial_theta)

        def log_prob(theta_batch: np.ndarray) -> np.ndarray:
            return self.log_posterior_vectorized(theta_batch, observed, sigma)

        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=N_PARAMS,
            log_prob_fn=log_prob,
            vectorize=True,
        )
        sampler.random_state = np.random.RandomState(rng.integers(2**32)).get_state()
        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        return flat_samples, sampler


def convergence_diagnostics(
    sampler: emcee.EnsembleSampler,
    burn_in: int,
    min_tau_multiple: float = 50.0,
) -> dict:
    """Summarise whether a chain is long enough to trust.

    Uses the integrated autocorrelation time τ. A common rule of thumb
    (emcee docs) is that the post-burn-in chain should be ≳ 50 τ long and the
    burn-in a few τ.

    Returns
    -------
    dict with keys ``tau`` (per parameter), ``max_tau``, ``n_kept_steps``,
    ``n_independent`` (≈ kept steps × walkers / max τ), ``acceptance`` and
    ``converged``.
    """
    n_kept = sampler.iteration - burn_in
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = sampler.get_autocorr_time(discard=burn_in, quiet=True)
    tau = np.where(np.isfinite(tau), tau, np.inf)  # a parameter that never moved
    max_tau = float(np.max(tau))
    acceptance = float(np.mean(sampler.acceptance_fraction))
    return {
        "tau": tau.tolist(),
        "max_tau": max_tau,
        "n_kept_steps": int(n_kept),
        "n_independent": float(n_kept * sampler.nwalkers / max_tau),
        "acceptance": acceptance,
        "converged": bool(n_kept >= min_tau_multiple * max_tau and burn_in >= 2 * max_tau),
    }
