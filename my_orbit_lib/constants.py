from sympy.physics.units import meter, kilometer, second, hour, minute, kilogram, convert_to

G = 6.674e-11 * meter**3 * kilogram**-1 * second**-2

MASS_OF_EARTH = 5.972e24 * kilogram
EARTH_RADIUS = 6371 * kilometer

MASS_OF_MARS = 6.39e23 * kilogram
MARS_RADIUS = 3389.5 * kilometer

# Hack
EARTH_RADIUS = MARS_RADIUS
MASS_OF_EARTH = MASS_OF_MARS

STANDARD_GRAVITATIONAL_PARAMETER_EARTH = MASS_OF_EARTH * convert_to(G, kilometer) 
STANDARD_GRAVITATIONAL_PARAMETER_MARS = MASS_OF_MARS * convert_to(G, kilometer) 

STANDARD_GRAVITATIONAL_PARAMETER = STANDARD_GRAVITATIONAL_PARAMETER_MARS
