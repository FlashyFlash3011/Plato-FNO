"""Science-specific evaluation metrics for atmospheric retrieval."""

from __future__ import annotations

import time

import numpy as np
import torch

from plato_fno.physics.atmosphere import N_PARAMS, PARAM_NAMES


def posterior_coverage(
    flat_samples: np.ndarray,   # (N_samples, N_PARAMS)
    true_theta: np.ndarray,     # (N_PARAMS,)
    credible_level: float = 0.68,
) -> dict[str, bool]:
    """Check whether each true parameter falls within its credible interval.

    Parameters
    ----------
    flat_samples : np.ndarray, shape (N_samples, N_PARAMS)
        Posterior samples from MCMC.
    true_theta : np.ndarray, shape (N_PARAMS,)
        Ground-truth parameter vector.
    credible_level : float
        Credible interval level. Default 0.68 (1-sigma equivalent).

    Returns
    -------
    dict[str, bool]
        Maps parameter name → True if true value is within CI.
    """
    alpha = (1.0 - credible_level) / 2.0
    lo = np.quantile(flat_samples, alpha, axis=0)    # (N_PARAMS,)
    hi = np.quantile(flat_samples, 1.0 - alpha, axis=0)

    return {
        name: bool(lo[i] <= true_theta[i] <= hi[i])
        for i, name in enumerate(PARAM_NAMES)
    }


def parameter_bias(
    flat_samples: np.ndarray,   # (N_samples, N_PARAMS)
    true_theta: np.ndarray,     # (N_PARAMS,)
) -> dict[str, float]:
    """Compute signed bias (posterior median - truth) for each parameter.

    Parameters
    ----------
    flat_samples : np.ndarray, shape (N_samples, N_PARAMS)
    true_theta : np.ndarray, shape (N_PARAMS,)

    Returns
    -------
    dict[str, float]
        Maps parameter name → bias (median - truth).
    """
    medians = np.median(flat_samples, axis=0)  # (N_PARAMS,)
    return {
        name: float(medians[i] - true_theta[i])
        for i, name in enumerate(PARAM_NAMES)
    }


def posterior_width(
    flat_samples: np.ndarray,
    credible_level: float = 0.68,
) -> dict[str, float]:
    """Return the width of the credible interval for each parameter.

    Parameters
    ----------
    flat_samples : np.ndarray, shape (N_samples, N_PARAMS)
    credible_level : float

    Returns
    -------
    dict[str, float]
        Maps parameter name → CI width.
    """
    alpha = (1.0 - credible_level) / 2.0
    lo = np.quantile(flat_samples, alpha, axis=0)
    hi = np.quantile(flat_samples, 1.0 - alpha, axis=0)
    return {name: float(hi[i] - lo[i]) for i, name in enumerate(PARAM_NAMES)}


def speedup_ratio(
    t_prt_single: float,
    t_fno_single: float,
    n_walkers: int = 64,
) -> dict[str, float]:
    """Compute speedup metrics.

    Parameters
    ----------
    t_prt_single : float
        Time for one petitRADTRANS forward call [seconds].
    t_fno_single : float
        Time for one FNO forward call (batch=1) [seconds].
    n_walkers : int
        Number of MCMC walkers used for the batched estimate.

    Returns
    -------
    dict with keys:
        single_call : speedup for a single evaluation
        batched_step : speedup for one MCMC step (n_walkers evaluations)
    """
    return {
        "single_call":  t_prt_single / max(t_fno_single, 1e-9),
        "batched_step": (n_walkers * t_prt_single) / max(t_fno_single, 1e-9),
    }


def benchmark_inference_time(
    surrogate: torch.nn.Module,
    n_warmup: int = 20,
    n_repeat: int = 100,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> float:
    """Benchmark FNO forward pass time in seconds.

    Parameters
    ----------
    surrogate : AtmosphereSurrogate
    n_warmup : int
        Warm-up iterations (not timed).
    n_repeat : int
        Number of timed iterations.
    batch_size : int
        Batch size for timing (use 64 for batched MCMC step estimate).
    device : str
        Torch device.

    Returns
    -------
    float
        Mean inference time per call [seconds].
    """
    device = torch.device(device)
    surrogate = surrogate.to(device)
    surrogate.eval()
    dummy = torch.randn(batch_size, N_PARAMS, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            surrogate(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            surrogate(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

    return (t1 - t0) / n_repeat
