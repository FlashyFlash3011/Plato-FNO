# PLATO-FNO: Exoplanet Atmospheric Retrieval

FNO-based surrogate for radiative transfer in exoplanet atmospheric retrieval, targeting the ESA PLATO mission. Replaces petitRADTRANS inside a Bayesian MCMC loop for 100–6000× speedup with calibrated uncertainty via conformal prediction.

Built on [Solaris](https://github.com/FlashyFlash3011/Solaris) (AMD ROCm neural operator framework).

## Pipeline

```
petitRADTRANS (slow, ~0.5s/call)          emcee MCMC
   ↓ generate 50k training pairs              ↓
   HDF5 dataset                          FNO forward (~5ms/batch of 64)
   ↓                                         ↓
   AtmosphereSurrogate (MLP + FNO1d)    posterior samples
   ↓                                         ↓
   ConformalNeuralOperator               corner plot + arviz diagnostics
   (≥90% coverage guarantee)
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e /path/to/Solaris"[datapipes,viz]"
pip install -e ".[dev]"
```

## Usage

```bash
# 1. Generate training data (~4-8h, 50k samples)
python scripts/generate_data.py data.n_samples=50000

# 2. Train the FNO surrogate (~2-4h on RX 9070 XT)
python scripts/train.py

# 3. Calibrate conformal uncertainty
python scripts/calibrate.py

# 4. Run MCMC retrieval on a target spectrum
python scripts/retrieve.py retrieval.target_spectrum=data/raw/target.h5

# 5. Benchmark vs petitRADTRANS
python scripts/evaluate.py
```

## Architecture

`AtmosphereSurrogate` encodes 15 atmospheric parameters through a 2-layer MLP, broadcasts the latent to the wavelength dimension, then processes with a 1D FNO (6 layers, 128 channels, 24 Fourier modes). Output: transit depth spectrum over 300 wavelength points (0.3–5.0 µm).

## Hardware

Optimised for AMD Radeon RX 9070 XT via ROCm 6.2. All training and inference runs on ROCm PyTorch.
