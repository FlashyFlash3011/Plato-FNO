# PLATO-FNO: Fast Exoplanet Atmosphere Retrieval

Atmospheric retrieval is slow. You guess an atmosphere, run a radiative transfer code to get its
spectrum, compare that with the data, and then do it all again a few hundred thousand times. This
project trains a neural network (a 1D Fourier Neural Operator) to stand in for the radiative transfer
code, so each guess takes milliseconds instead of most of a second.

It's built on [Solaris](https://github.com/FlashyFlash3011/Solaris), my PyTorch library of neural
operators.

## Where it fits

This project doesn't find planets. It comes in after a planet has been found and confirmed, once
someone has measured its transmission spectrum (with JWST, say). Its job is the last step: working
out what the atmosphere is made of, and how sure we can be.

Why bother making that step fast? Missions like PLATO are about to turn up far more good targets than
anyone can fit by hand, one slow retrieval at a time. With a fast surrogate you can run lots of
retrievals, try different assumptions, and get a first look at a new planet in minutes.

## How it works

There are five steps, and each one is a script in `scripts/`.

**1. Generate training data** (`generate_data.py`). We pick thousands of random atmospheres and
run [petitRADTRANS](https://petitradtrans.readthedocs.io) on each one. Every atmosphere is
described by 14 numbers:

- temperature profile (Guillot 2010): `T_int`, `T_eq`, `log_kappa_IR`, `log_gamma`
- gas abundances: `H2O`, `CO2`, `CH4`, `CO`, `NH3` (everything else is H₂/He)
- planet and star: `log_g`, `R_planet`, `R_star`
- clouds: `log_P_cloud`, `f_sed`

Each one becomes a transit depth spectrum with 300 points from 0.3 to 5.0 µm, and the pairs are
saved to HDF5.

Planets too large for their star (transit depth above 15%) are left out of both the prior and the
training data, so the network is never asked about planets it hasn't seen.

**2. Train the surrogate** (`train.py`). A small MLP turns the 14 numbers into a set of features.
Those get spread along the wavelength axis, and the FNO shapes them into a spectrum. The FNO works
on the whole curve at once: it keeps the 24 smoothest Fourier components and learns how to weight
each one, which suits absorption bands well. One extra layer predicts the overall depth (mostly set
by planet-to-star size) directly. The data is split 70/10/10/10 into train, validation,
calibration and test sets; the test set is only ever used by `evaluate.py`. Each checkpoint
stores the architecture, the normalisation statistics and the wavelength grid, so the later steps
need nothing else.

**3. Calibrate the error** (`calibrate.py`). The network is never perfect, so this step measures it
on calibration data it hasn't seen. Conformal prediction picks an error band that contains the
whole true spectrum at least 90% of the time (checked on the test set). It also saves the
network's typical error at each wavelength, which the retrieval adds to the observational noise so
the posterior isn't over-confident where the network is less accurate.

**4. Retrieve** (`retrieve.py`). `emcee` runs 64 walkers through the 14-parameter space, starting
from random draws of the prior. On each step the network predicts spectra for all 64 guesses in
one GPU batch, and each guess is scored against the data with a Gaussian likelihood (per-point
error bars if you have them). After the burn-in, wherever the walkers spent their time is your
posterior. You get the samples, a corner plot, a median ± 1σ summary, and convergence diagnostics
(autocorrelation time); the script warns when the chain is too short to trust.

**5. Evaluate** (`evaluate.py`). This measures spectrum accuracy on the test set, times the
network, and runs 20 retrievals on test-set planets: real petitRADTRANS spectra plus noise, where
we know the right answer. That shows whether the 68% intervals really contain the truth about 68%
of the time, including the network's own error.

`plato_fno/physics/` also has instrument models for building test observations: JWST binning and
noise (NIRISS SOSS, NIRSpec PRISM/G140M/G395M; `retrieval.instrument` retrieves on that grid), and
band-averaged depths in PLATO's real passbands (the ~500–1000 nm white-light band of the normal
cameras, and the blue and red filters of the two fast cameras).

## Setup

Python 3.10–3.12 is required.

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch for your hardware first (CUDA, ROCm or CPU): https://pytorch.org/get-started/locally/
pip install -e "/path/to/Solaris[datapipes,viz]"   # Solaris >= 0.2, not on PyPI
pip install -e ".[dev]"

# Only needed to generate data: petitRADTRANS v3 plus its opacity files
pip install -e ".[prt]"
```

petitRADTRANS asks to download missing opacity files the first time they are needed (several GB),
so run data generation interactively once.

Run the tests with `pytest`. They mock petitRADTRANS, so you don't need it installed for them.

## Usage

The configs live in `configs/` and use Hydra, so you can override anything from the command line.

```bash
python scripts/generate_data.py data.n_samples=50000 data.n_workers=8
python scripts/train.py
python scripts/calibrate.py
python scripts/retrieve.py retrieval.synthetic_test=true          # fake planet, known answer
python scripts/retrieve.py retrieval.target_spectrum=target.h5    # your own spectrum
python scripts/evaluate.py                                        # evaluate.time_prt=true to time pRT
```

A target spectrum is an HDF5 file with a `target` dataset of shape `(N, M)` or `(N, 1, M)`,
holding `(Rp/Rs)²`. M is the 300-point 0.3–5.0 µm grid, or the bins of a JWST instrument when you
set `retrieval.instrument` (e.g. `nirspec_prism`). An optional `sigma` dataset gives per-point
1σ errors; otherwise `retrieval.sigma` (50 ppm) is used for every point. An optional `wavelengths`
dataset is checked against the expected grid.

## Known gaps

- Real data must be on the native grid or binned exactly like one of the four JWST instrument
  presets; arbitrary instrument grids aren't supported yet.
- The JWST noise levels are illustrative placeholders, not instrument-team numbers; use the JWST
  ETC or PandExo for a real target. Noise is white (no correlated systematics).
- `evaluate.py` only times petitRADTRANS with `evaluate.time_prt=true` (needs its opacity data);
  otherwise the speed-up assumes 0.5 s per call and says so.
- `R_planet` and `R_star` are both free, but the spectrum mostly depends on their ratio, so their
  posteriors are strongly correlated. Real retrievals usually fix the stellar radius.
- PLATO photometry is modelled (top-hat bands) but not yet used in the retrieval.
- The atmosphere model is deliberately simple: abundances are the same at every altitude, clouds
  are a single opaque layer, there's no haze and no chemistry.

## License

See [LICENSE](LICENSE).
