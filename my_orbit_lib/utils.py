import sympy as sp
import numpy as np

from my_orbit_lib.constants import *
from my_orbit_lib.extensions import *

def get_specific_energy(x, v):
    return np.norm(v) ** 2 / 2 - strip_units(STANDARD_GRAVITATIONAL_PARAMETER_EARTH) / np.norm(x)

def get_sma(x, v):
    return -strip_units(STANDARD_GRAVITATIONAL_PARAMETER_EARTH) / (2 * get_specific_energy(strip_units(x), strip_units(v)))

def get_eccentricity_vector(x, v):
    """
    Returns the vector pointing from apoapsis to periapsis 
    with magnitude equal to the orbit's eccentricity
    """
    x = strip_units(x)
    v = strip_units(v)
    h = np.cross(x, v)
    return np.cross(v, h) / strip_units(STANDARD_GRAVITATIONAL_PARAMETER_EARTH) - (x / np.norm(x))
