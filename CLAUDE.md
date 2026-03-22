# CLAUDE.md — PLATO-FNO

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e "/home/ethan/projects/Solaris[datapipes,viz]"
pip install -e ".[dev]"
```

petitRADTRANS opacity data (one-time, several GB):
```bash
export PRTDATA=/home/ethan/projects/plato-fno/data/prt_input_data
python -c "from petitRADTRANS import Radtrans; Radtrans(['H2O'])"
```

## Commands

```bash
# Data generation (50k samples, ~4-8h on 4 CPU cores)
python scripts/generate_data.py data.n_samples=50000

# Training (~2-4h on RX 9070 XT)
python scripts/train.py

# Conformal calibration
python scripts/calibrate.py

# MCMC retrieval
python scripts/retrieve.py retrieval.target_spectrum=data/raw/target.h5

# Full benchmark
python scripts/evaluate.py

# Tests (fast, no GPU/pRT needed)
pytest tests/ -v -m "not slow"
pytest tests/ -v  # full suite

# Lint
ruff check plato_fno/ tests/ scripts/
ruff format plato_fno/ tests/ scripts/
```

## Architecture

### Input/Output
- Input theta: 15-dimensional parameter vector (see `plato_fno/physics/atmosphere.py`)
- Output: transit depth spectrum `(1, 300)` over 0.3–5.0 µm

### Parameter vector (N_PARAMS = 15)
```
[T_int, T_eq, kappa_IR, gamma,           # Guillot T-P profile
 log_X_H2O, log_X_CO2, log_X_CH4,        # gas abundances (log10)
 log_X_CO, log_X_NH3,
 log_g, R_planet, R_star,                # system parameters
 log_P_cloud, f_sed]                     # cloud parameters
```

### Model pipeline
```
theta (B, 15) → MLP encoder → (B, 32) → broadcast → (B, 32, L)
             → FNO1d(in=32, out=1, hidden=128, layers=6, modes=24)
             → (B, 1, L)  [normalised transit depth]
```

### Normalisation
`SpectrumNormalizer` saves per-channel z-score stats to `checkpoints/norm_stats.npz`.
Always load this before retrieval — the FNO operates on normalised quantities.

### Conformal UQ
`ConformalNeuralOperator` wraps the trained surrogate. After `calibrate(alpha=0.1)`,
`predict(theta)` returns `(lower, upper, point)` with guaranteed ≥90% coverage.
q_hat is stored in normalised units; multiply by `normalizer.spec_std` for ppm.

### MCMC
`emcee` with `vectorize=True` batches 64 walkers into one GPU forward call.
Speedup vs sequential pRT: ~6000× per MCMC step.

## Solaris components used
- `solaris.models.fno.FNO` — core operator (dim=1)
- `solaris.models.conformal.ConformalNeuralOperator` — UQ wrapper
- `solaris.utils.training` — WarmupCosineScheduler, EarlyStopping, AutoCheckpoint, GradientClipper
- `solaris.utils.checkpoint` — save_checkpoint / load_checkpoint
- `solaris.datapipes.dataset` — HDF5Dataset, build_dataloader
- `solaris.metrics.metrics` — relative_l2_error, rmse, r2_score

## Hardware
AMD Radeon RX 9070 XT (gfx1200 / RDNA4), ROCm 6.2.
PyTorch device: `cuda` (ROCm uses the CUDA API alias).
