"""Atmospheric state representation and prior sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np

# Species included in the retrieval
SPECIES: list[str] = ["H2O", "CO2", "CH4", "CO", "NH3"]

# Total number of free parameters
N_PARAMS: int = 14

# Ordered parameter names (matches to_vector / from_vector)
PARAM_NAMES: list[str] = [
    "T_int", "T_eq", "log_kappa_IR", "log_gamma",
    "log_X_H2O", "log_X_CO2", "log_X_CH4", "log_X_CO", "log_X_NH3",
    "log_g", "R_planet", "R_star",
    "log_P_cloud", "f_sed",
]

assert len(PARAM_NAMES) == N_PARAMS, "Update PARAM_NAMES if N_PARAMS changes"

R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10

# Upper limit on transit depth. Spectra deeper than this are dropped from the
# training data (see plato_fno.data.dataset), so the prior excludes planets
# whose bare disk is already this large relative to the star: the surrogate
# has never seen them.
MAX_TRANSIT_DEPTH = 0.15


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
    log_kappa_IR: float  # log10 IR opacity [cm2/g]
    log_gamma: float     # log10 ratio kappa_V / kappa_IR

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
        "log_kappa_IR": (-4.0,  1.0),
        "log_gamma":   (-2.0,   2.0),
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
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> AtmosphericState:
        """Construct from flat parameter vector of shape (N_PARAMS,)."""
        field_names = [f.name for f in fields(cls)]
        return cls(**dict(zip(field_names, v.tolist())))

    @classmethod
    def sample_prior(cls, rng: np.random.Generator) -> AtmosphericState:
        """Draw one sample uniformly from the prior (bounds + constraints)."""
        while True:
            values = {name: rng.uniform(lo, hi) for name, (lo, hi) in cls.PRIOR_BOUNDS.items()}
            state = cls(**values)
            if cls.satisfies_constraints(state.to_vector()):
                return state

    @staticmethod
    def satisfies_constraints(theta: np.ndarray) -> np.ndarray | bool:
        """Joint prior constraints beyond the per-parameter bounds.

        ``theta`` is (N_PARAMS,) or (B, N_PARAMS) in physical units. Currently
        one constraint: geometric transit depth (R_planet / R_star)^2 below
        MAX_TRANSIT_DEPTH.
        """
        theta = np.asarray(theta)
        r_planet = theta[..., _R_PLANET] * R_JUP_CM
        r_star = theta[..., _R_STAR] * R_SUN_CM
        return r_planet**2 < MAX_TRANSIT_DEPTH * r_star**2  # no division: R_star may be 0

    def in_prior(self) -> bool:
        """Return True if the state has non-zero prior probability."""
        for name, (lo, hi) in self.PRIOR_BOUNDS.items():
            val = getattr(self, name)
            if not (lo <= val <= hi):
                return False
        return bool(self.satisfies_constraints(self.to_vector()))


def guillot_temperature_profile(
    pressures: np.ndarray,
    T_int: float,
    T_eq: float,
    kappa_IR: float,
    gamma: float,
    gravity: float,
) -> np.ndarray:
    """Analytical Guillot (2010) T-P profile, global average (his Eq. 29).

    Same formula as petitRADTRANS' ``temperature_profile_function_guillot_global``.

    Parameters
    ----------
    pressures : np.ndarray
        Pressure array in bar, shape (n_layers,), any order.
    T_int : float
        Internal temperature [K].
    T_eq : float
        Equilibrium temperature [K].
    kappa_IR : float
        IR opacity [cm2/g] (linear, not log10).
    gamma : float
        Ratio of optical to IR opacity (linear, not log10).
    gravity : float
        Surface gravity [cm/s^2]; sets the column mass, tau = kappa_IR * P / g.

    Returns
    -------
    np.ndarray
        Temperature profile [K], shape (n_layers,).
    """
    tau = kappa_IR * pressures * 1e6 / gravity  # bar → dyne/cm^2
    sqrt3 = np.sqrt(3.0)
    # Global average: f * T_irr^4 with f = 1/4 and T_irr^4 = 4 T_eq^4 gives T_eq^4
    irradiation = (
        2.0 / 3.0
        + 1.0 / (gamma * sqrt3)
        + (gamma / sqrt3 - 1.0 / (gamma * sqrt3)) * np.exp(-gamma * tau * sqrt3)
    )
    T4 = 0.75 * T_int**4 * (2.0 / 3.0 + tau) + 0.75 * T_eq**4 * irradiation
    return T4**0.25


# The vector layout, the dataclass fields and PRIOR_BOUNDS must all agree:
# MCMCRetrieval.log_prior indexes bounds by position.
assert [f.name for f in fields(AtmosphericState)] == PARAM_NAMES == list(
    AtmosphericState.PRIOR_BOUNDS
), "PARAM_NAMES, AtmosphericState fields and PRIOR_BOUNDS are out of order"

_R_PLANET = PARAM_NAMES.index("R_planet")
_R_STAR = PARAM_NAMES.index("R_star")
