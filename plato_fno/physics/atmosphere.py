"""Atmospheric state representation and prior sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np

# Species included in the retrieval
SPECIES: list[str] = ["H2O", "CO2", "CH4", "CO", "NH3"]

# Total number of free parameters
N_PARAMS: int = 15

# Ordered parameter names (matches to_vector / from_vector)
PARAM_NAMES: list[str] = [
    "T_int", "T_eq", "kappa_IR", "gamma",
    "log_X_H2O", "log_X_CO2", "log_X_CH4", "log_X_CO", "log_X_NH3",
    "log_g", "R_planet", "R_star",
    "log_P_cloud", "f_sed",
]

assert len(PARAM_NAMES) == N_PARAMS - 1, "Update PARAM_NAMES if N_PARAMS changes"


@dataclass
class AtmosphericState:
    """Physical parameter vector for one atmosphere sample.

    All log-abundances are log10 mass fractions. Temperatures in Kelvin.
    Pressures in bar. This is the canonical representation used throughout
    the pipeline; petitrt_driver converts to pRT API format.
    """

    # Guillot (2010) T-P profile parameters
    T_int: float        # internal temperature [K]
    T_eq: float         # equilibrium temperature [K]
    kappa_IR: float     # IR opacity [cm2/g]  (log10 stored)
    gamma: float        # ratio kappa_V / kappa_IR  (log10 stored)

    # Gas log10 mass fractions
    log_X_H2O: float
    log_X_CO2: float
    log_X_CH4: float
    log_X_CO: float
    log_X_NH3: float

    # System parameters
    log_g: float        # log10 surface gravity [cm/s^2]
    R_planet: float     # planet radius [R_jup]
    R_star: float       # star radius [R_sun]

    # Cloud parameters
    log_P_cloud: float  # cloud-top pressure log10 [bar]
    f_sed: float        # sedimentation factor

    # Prior bounds: (min, max) uniform prior for each parameter
    PRIOR_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "T_int":       (50.0,   800.0),
        "T_eq":        (300.0,  3000.0),
        "kappa_IR":    (-4.0,   1.0),    # log10
        "gamma":       (-2.0,   2.0),    # log10
        "log_X_H2O":   (-6.0,  -1.0),
        "log_X_CO2":   (-6.0,  -1.0),
        "log_X_CH4":   (-6.0,  -1.0),
        "log_X_CO":    (-6.0,  -1.0),
        "log_X_NH3":   (-6.0,  -1.0),
        "log_g":       (2.5,    4.5),
        "R_planet":    (0.5,    2.0),
        "R_star":      (0.5,    2.0),
        "log_P_cloud": (-4.0,   2.0),
        "f_sed":       (0.5,    10.0),
    }

    def to_vector(self) -> np.ndarray:
        """Return flat float32 array of shape (N_PARAMS,)."""
        return np.array(
            [getattr(self, f.name) for f in fields(self) if f.name != "PRIOR_BOUNDS"],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "AtmosphericState":
        """Construct from flat parameter vector of shape (N_PARAMS,)."""
        field_names = [f.name for f in fields(cls) if f.name != "PRIOR_BOUNDS"]
        return cls(**dict(zip(field_names, v.tolist())))

    @classmethod
    def sample_prior(cls, rng: np.random.Generator) -> "AtmosphericState":
        """Draw one sample uniformly from prior bounds."""
        values = {
            name: rng.uniform(lo, hi)
            for name, (lo, hi) in cls.PRIOR_BOUNDS.items()
        }
        return cls(**values)

    def in_prior(self) -> bool:
        """Return True if all parameters are within prior bounds."""
        for name, (lo, hi) in self.PRIOR_BOUNDS.items():
            val = getattr(self, name)
            if not (lo <= val <= hi):
                return False
        return True


def guillot_temperature_profile(
    pressures: np.ndarray,
    T_int: float,
    T_eq: float,
    kappa_IR: float,
    gamma: float,
) -> np.ndarray:
    """Analytical Guillot (2010) T-P profile.

    Parameters
    ----------
    pressures : np.ndarray
        Pressure array in bar, shape (n_layers,). Should be sorted high→low
        (surface→top of atmosphere), as petitRADTRANS expects.
    T_int : float
        Internal temperature [K].
    T_eq : float
        Equilibrium temperature [K].
    kappa_IR : float
        IR opacity [cm2/g] (linear, not log10).
    gamma : float
        Ratio of optical to IR opacity (linear, not log10).

    Returns
    -------
    np.ndarray
        Temperature profile [K], shape (n_layers,).
    """
    # Convert pressures to cgs (bar → dyne/cm^2), then to tau
    # tau = kappa_IR * P / g  — use reference gravity g=1000 cm/s^2 for profile shape
    # The actual gravity is accounted for in the transmission spectrum calculation
    g_ref = 1000.0  # cm/s^2 reference (profile shape only)
    pressures_cgs = pressures * 1e6  # bar → dyne/cm^2

    tau = kappa_IR * pressures_cgs / g_ref

    # Guillot Eq. 29
    T4 = (
        0.75 * T_int**4 * (2.0 / 3.0 + tau)
        + 0.75
        * T_eq**4
        * (2.0 / 3.0 + 1.0 / (gamma * np.sqrt(3.0)) + (gamma / np.sqrt(3.0) - 1.0 / (gamma * np.sqrt(3.0))) * np.exp(-gamma * tau * np.sqrt(3.0)))
    )
    return T4**0.25
