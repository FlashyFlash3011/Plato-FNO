"""Tests for retrieval metrics."""

import numpy as np

from plato_fno.metrics.retrieval_metrics import (
    parameter_bias,
    posterior_coverage,
    posterior_width,
    speedup_ratio,
)
from plato_fno.physics.atmosphere import N_PARAMS, PARAM_NAMES


def test_speedup_uses_batched_fno_time_for_mcmc_step():
    s = speedup_ratio(t_prt_single=0.5, t_fno_single=0.001, t_fno_batch=0.004, n_walkers=64)
    assert s["single_call"] == 500.0
    assert s["batched_step"] == 64 * 0.5 / 0.004


def test_posterior_coverage_and_bias():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, (20000, N_PARAMS))
    truth = np.zeros(N_PARAMS)
    truth[0] = 5.0  # far outside the 68% interval
    cov = posterior_coverage(samples, truth, credible_level=0.68)
    assert not cov[PARAM_NAMES[0]]
    assert all(cov[name] for name in PARAM_NAMES[1:])
    bias = parameter_bias(samples, truth)
    assert abs(bias[PARAM_NAMES[0]] + 5.0) < 0.05


def test_posterior_width_of_standard_normal():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, (50000, N_PARAMS))
    widths = posterior_width(samples, credible_level=0.68)
    for w in widths.values():
        assert abs(w - 2 * 0.9945) < 0.05
