"""petitRADTRANS v3 wrapper for generating atmosphere→spectrum training data.

petitRADTRANS is process-safe but not thread-safe: each worker process
must instantiate its own Radtrans object. This module uses multiprocessing
to parallelise over CPU cores.

pRT v3 API notes (vs v2):
- Pressures and wavelength_boundaries are now constructor args
- calc_transm() → calculate_transit_radii() which returns (radii, wlen, dict)
- continuum_opacities → gas_continuum_contributors
- wlen_bords_micron → wavelength_boundaries (in microns)
- mode → line_opacity_mode
- setup_opa_structure() removed — pressures in constructor
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from plato_fno.physics.atmosphere import AtmosphericState, guillot_temperature_profile

# petitRADTRANS is optional at import time so tests can run without it
try:
    from petitRADTRANS.radtrans import Radtrans  # type: ignore[import]
    _PRT_AVAILABLE = True
except ImportError:
    _PRT_AVAILABLE = False
    Radtrans = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Radtrans factory — one instance per worker process
# ---------------------------------------------------------------------------

_worker_prt: "Radtrans | None" = None
_worker_pressures: np.ndarray | None = None
_worker_wavelengths: np.ndarray | None = None
_worker_sleep: float = 0.0  # seconds to sleep after each spectrum (throttle)


def _init_worker(
    species: list[str],
    wavelength_range: tuple[float, float],
    n_wavelengths: int,
    n_layers: int,
    sleep_seconds: float = 0.0,
) -> None:
    """Initialise Radtrans in each worker process (called once per worker)."""
    global _worker_prt, _worker_pressures, _worker_wavelengths, _worker_sleep
    _worker_sleep = sleep_seconds

    if not _PRT_AVAILABLE:
        raise RuntimeError("petitRADTRANS is not installed. Run: pip install petitRADTRANS")

    # Log-spaced pressure grid from 1e-6 to 1e2 bar (top→surface)
    # pRT v3 requires ASCENDING order
    _worker_pressures = np.logspace(-6, 2, n_layers)  # bar, low→high

    # pRT v3: pass pressures and wavelength_boundaries to constructor
    # Add slight margin to wavelength range so pRT's native grid covers ours
    wl_lo = wavelength_range[0] * 0.98
    wl_hi = wavelength_range[1] * 1.02

    _worker_prt = Radtrans(
        pressures=_worker_pressures,
        wavelength_boundaries=np.array([wl_lo, wl_hi]),
        line_species=species,
        rayleigh_species=["H2", "He"],
        gas_continuum_contributors=["H2-H2", "H2-He"],
        line_opacity_mode="c-k",
    )

    # Our uniform output wavelength grid (for interpolation)
    _worker_wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)


def _compute_one(state_vec: np.ndarray) -> np.ndarray | None:
    """Compute one transit spectrum from a parameter vector.

    Called inside worker processes. Returns shape (n_wavelengths,) or None on error.
    """
    import time
    global _worker_prt, _worker_pressures, _worker_wavelengths, _worker_sleep

    try:
        state = AtmosphericState.from_vector(state_vec)
        spectrum = compute_spectrum(state, _worker_prt, _worker_pressures, _worker_wavelengths)
        if _worker_sleep > 0:
            time.sleep(_worker_sleep)
        return spectrum
    except Exception as exc:
        logger.warning(f"pRT failed for sample: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core spectrum computation (pRT v3 API)
# ---------------------------------------------------------------------------

def compute_spectrum(
    state: AtmosphericState,
    prt: "Radtrans",
    pressures: np.ndarray,
    wavelengths_micron: np.ndarray,
) -> np.ndarray:
    """Run one petitRADTRANS v3 forward pass.

    Parameters
    ----------
    state : AtmosphericState
        Atmospheric parameters (physical units, not normalised).
    prt : Radtrans
        Pre-initialised petitRADTRANS Radtrans object (v3).
    pressures : np.ndarray
        Pressure grid in bar, shape (n_layers,), high→low.
    wavelengths_micron : np.ndarray
        Target wavelength grid in microns for output interpolation.

    Returns
    -------
    np.ndarray
        Transit depth (Rp/Rs)^2 at each wavelength point, shape (n_wavelengths,).
    """
    n_layers = len(pressures)

    # Build T-P profile using Guillot parameterisation
    temperature = guillot_temperature_profile(
        pressures,
        T_int=state.T_int,
        T_eq=state.T_eq,
        kappa_IR=10.0**state.kappa_IR,
        gamma=10.0**state.gamma,
    )

    # Mass fractions — broadcast to (n_layers,) arrays
    log_x_map = {
        "H2O": state.log_X_H2O,
        "CO2": state.log_X_CO2,
        "CH4": state.log_X_CH4,
        "CO":  state.log_X_CO,
        "NH3": state.log_X_NH3,
    }
    total_trace = sum(10.0**lx for lx in log_x_map.values())
    h2_he_frac = max(0.0, 1.0 - total_trace)
    h2_frac = 0.75 * h2_he_frac
    he_frac = 0.25 * h2_he_frac

    mass_fractions: dict[str, np.ndarray] = {}
    for species, log_x in log_x_map.items():
        mass_fractions[species] = np.full(n_layers, 10.0**log_x)
    mass_fractions["H2"] = np.full(n_layers, h2_frac)
    mass_fractions["He"] = np.full(n_layers, he_frac)

    # Mean molecular weight (approximate, mostly H2/He)
    mean_molar_masses = np.full(n_layers, 2.33)

    gravity = 10.0**state.log_g          # cm/s^2
    R_planet_cm = state.R_planet * 7.1492e9   # R_jup → cm
    R_star_cm   = state.R_star   * 6.957e10   # R_sun → cm

    # pRT v3 returns (wlen_cm, transit_radii_cm, extras_dict)
    # reference_pressure: highest pressure point = pressures[-1] in ascending grid
    wlen_cm, transit_radii, _ = prt.calculate_transit_radii(
        temperatures=temperature,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=gravity,
        reference_pressure=pressures[-1],  # bar (surface, highest pressure)
        planet_radius=R_planet_cm,
        variable_gravity=True,
        opaque_cloud_top_pressure=10.0**state.log_P_cloud,  # bar
        frequencies_to_wavelengths=True,   # wlen_cm in ascending cm
    )

    # Convert wavelength from cm to µm; compute transit depth (Rp_eff/Rs)^2
    wlen_um = wlen_cm * 1e4
    transit_depth = (transit_radii / R_star_cm) ** 2

    return np.interp(wavelengths_micron, wlen_um, transit_depth).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset_hdf5(
    output_path: Path,
    n_samples: int,
    wavelength_range: tuple[float, float],
    n_wavelengths: int = 300,
    n_layers: int = 60,
    species: list[str] | None = None,
    seed: int = 42,
    n_workers: int = 4,
    chunk_size: int = 1000,
    throttle_sleep: float = 0.0,
) -> None:
    """Generate n_samples (atmosphere, spectrum) pairs and write to HDF5.

    HDF5 layout
    -----------
    /input        (n_samples, N_PARAMS)         float32  — raw parameter vectors
    /target       (n_samples, 1, n_wavelengths) float32  — transit depth spectra
    /wavelengths  (n_wavelengths,)              float64  — wavelength grid [µm]
    attrs: n_samples, seed, species, wavelength_range

    Parameters
    ----------
    output_path : Path
        Destination HDF5 file.
    n_samples : int
        Number of (theta, spectrum) pairs to generate.
    wavelength_range : tuple[float, float]
        (lambda_min, lambda_max) in microns.
    n_wavelengths : int
        Number of wavelength points in output grid.
    n_layers : int
        Number of pressure layers for petitRADTRANS.
    species : list[str], optional
        Line species for pRT. Defaults to H2O, CO2, CH4, CO, NH3.
    seed : int
        Global RNG seed for reproducibility.
    n_workers : int
        Number of parallel worker processes.
    chunk_size : int
        HDF5 chunk size along the sample axis.
    """
    if species is None:
        species = ["H2O", "CO2", "CH4", "CO", "NH3"]

    from plato_fno.physics.atmosphere import N_PARAMS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)

    # Sample all parameter vectors upfront for reproducibility
    rng = np.random.default_rng(seed)
    states = [AtmosphericState.sample_prior(rng) for _ in range(n_samples)]
    param_vecs = np.stack([s.to_vector() for s in states], axis=0)  # (N, N_PARAMS)

    logger.info(f"Generating {n_samples} spectra with {n_workers} workers → {output_path}")

    # Create HDF5 file with pre-allocated datasets
    with h5py.File(output_path, "w") as f:
        f.attrs["n_samples"] = n_samples
        f.attrs["seed"] = seed
        f.attrs["species"] = species
        f.attrs["wavelength_range"] = list(wavelength_range)

        chunk = min(chunk_size, n_samples)
        ds_input  = f.create_dataset("input",  shape=(n_samples, N_PARAMS),            dtype="float32", chunks=(chunk, N_PARAMS))
        ds_target = f.create_dataset("target", shape=(n_samples, 1, n_wavelengths),     dtype="float32", chunks=(chunk, 1, n_wavelengths))
        _         = f.create_dataset("wavelengths", data=wavelengths)

        # Write parameter vectors immediately (always available even if spectrum fails)
        ds_input[:] = param_vecs

        ctx = mp.get_context("spawn")
        initargs = (species, wavelength_range, n_wavelengths, n_layers, throttle_sleep)

        written = 0
        failed = 0

        with ctx.Pool(n_workers, initializer=_init_worker, initargs=initargs) as pool:
            results = pool.imap(_compute_one, param_vecs, chunksize=50)

            for i, spectrum in enumerate(tqdm(results, total=n_samples, desc="Generating spectra")):
                if spectrum is None:
                    ds_target[i, 0, :] = np.full(n_wavelengths, float("nan"), dtype=np.float32)
                    failed += 1
                else:
                    ds_target[i, 0, :] = spectrum
                    written += 1

                if (i + 1) % chunk == 0:
                    f.flush()

    logger.info(f"Done: {written} OK, {failed} failed → {output_path}")
