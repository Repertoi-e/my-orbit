from sympy.physics.units import meter, kilometer, second, hour, minute, kilogram, convert_to

MASS_OF_EARTH = 5.972e24 * kilogram
EARTH_RADIUS = 6371 * kilometer
G = 6.674e-11 * meter**3 * kilogram**-1 * second**-2

STANDARD_GRAVITATIONAL_PARAMETER_EARTH = MASS_OF_EARTH * convert_to(G, kilometer) 