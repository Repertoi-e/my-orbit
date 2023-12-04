import numpy as np
import sympy as sp

from sympy.physics.units.quantities import Quantity

sp.Vector = lambda arr: sp.Matrix([arr])
sp.norm = lambda v: sp.sqrt(v.dot(v))

np.Vector = lambda x: np.array(x)
np.norm = lambda v: np.sqrt(v.dot(v))

def strip_units(x):
    """
    Converts floats with units to just floats, 
    sumpy vectors and matrices with units to np.ndarrays of floats.
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, complex):
        raise ValueError("Can't do complex")

    if isinstance(x, np.ndarray):
        return x.astype(np.float64)

    def to_float(expr):
        d = {y: 1 for y in expr.args if y.has(Quantity)}
        if not d:
            return 1.0 # Degenerate case, e.g. 1 * km is represented just as km for some reason 
        return float(expr.subs(d))

    if isinstance(x, list):
        return strip_units(sp.Matrix(x))

    if hasattr(x, 'is_number') and x.is_number:
        return float(x)

    if hasattr(x, 'tolist'):
        return np.array([[to_float(z) for z in y] for y in x.tolist()]).squeeze()
    else:
        return to_float(x) # Just sp.Quantity