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

- temperature profile (Guillot 2010): `T_int`, `T_eq`, `kappa_IR`, `gamma`
- gas abundances: `H2O`, `CO2`, `CH4`, `CO`, `NH3` (everything else is H₂/He)
- planet and star: `log_g`, `R_planet`, `R_star`
- clouds: `log_P_cloud`, `f_sed`

Each one becomes a transit depth spectrum with 300 points from 0.3 to 5.0 µm, and the pairs are
saved to HDF5.

**2. Train the surrogate** (`train.py`). A small MLP turns the 14 numbers into a set of features.
Those get spread along the wavelength axis, and the FNO shapes them into a spectrum. The FNO works
on the whole curve at once: it keeps the 24 smoothest Fourier components and learns how to weight
each one, which suits absorption bands well. One extra layer predicts the overall depth (mostly set
by planet-to-star size), because the FNO's normalisation would otherwise strip that out. The data is
split 80/10/10 into train, validation and calibration sets.

**3. Calibrate the error** (`calibrate.py`). The network is never perfect, so this step checks it
on calibration data it hasn't seen. Conformal prediction then picks an error band (`q_hat`, reported
in ppm) that contains the true spectrum at least 90% of the time.

**4. Retrieve** (`retrieve.py`). `emcee` runs 64 walkers through the 14-parameter space. On each
step the network predicts spectra for all 64 guesses in one GPU batch, and each guess is scored
against the data with a Gaussian likelihood. After the burn-in, wherever the walkers spent their
time is your posterior. You get the samples, a corner plot and a median ± 1σ summary.

**5. Evaluate** (`evaluate.py`). This measures spectrum accuracy on held-out data, times the
network, and runs 20 retrievals on fake planets where we know the right answer, to see whether the
68% intervals really contain the truth about 68% of the time.

`plato_fno/physics/` also has instrument models for building realistic test observations: JWST
binning and noise (NIRISS SOSS, NIRSpec PRISM/G140M/G395M), plus broadband colour photometry in
the style of PLATO.

## Setup

Python 3.10–3.12 is required

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch for your hardware first (CUDA, ROCm or CPU): https://pytorch.org/get-started/locally/
pip install -e "/path/to/Solaris[datapipes,viz]"
pip install -e ".[dev]"

# Only needed if you want to generate data: petitRADTRANS v3 plus its opacity files
pip install petitRADTRANS
```

Run the tests with `pytest`. They mock petitRADTRANS, so you don't need it installed for them.

## Usage

The configs live in `configs/` and use Hydra, so you can override anything from the command line.

```bash
python scripts/generate_data.py data.n_samples=50000 data.n_workers=8
python scripts/train.py
python scripts/calibrate.py
python scripts/retrieve.py retrieval.synthetic_test=true          # fake planet, known answer
python scripts/retrieve.py retrieval.target_spectrum=target.h5    # your own spectrum
python scripts/evaluate.py
```

A target spectrum should be an HDF5 file with a `target` dataset of shape `(N, 1, 300)`, holding
`(Rp/Rs)²` on the 0.3–5.0 µm grid. Set the noise with `retrieval.sigma` (the default is 50 ppm).

## Known gaps

- Real data has to be resampled onto the 300-point grid first, and it gets a single noise value.
  Per-bin JWST error bars aren't wired in yet, even though `make_bin_transform()` exists for that.
- `retrieve.py` doesn't use the calibrated error band yet.
- `evaluate.py` doesn't time petitRADTRANS. It assumes 0.5 s per call, so treat the speed-up number
  as a rough estimate.
- The evaluation reuses the calibration split, and its test retrievals use spectra made by the
  network itself.
- The atmosphere model is deliberately simple: abundances are the same at every altitude, clouds
  are a single opaque layer, there's no haze and no chemistry.

## License

See [LICENSE](LICENSE).
