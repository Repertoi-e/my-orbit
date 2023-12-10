import sympy as sp
import numpy as np

from sympy.physics.units import meter, kilometer, second, hour, minute, kilogram, convert_to, quantities

from my_orbit_lib.extensions import *
from my_orbit_lib.constants import STANDARD_GRAVITATIONAL_PARAMETER

from numba.core import types

from my_orbit_lib.epoch import Epoch

from numba import njit

force_function_type = types.FunctionType(types.float32[:](types.float32, types.float32[:], types.float32[:]))

@njit
def norm(x):
    return np.sqrt(np.sum(x**2))

@njit
def NO_FORCES(t, x, v):
    return np.zeros_like(x), 0

class Body:
    def __init__(self, pos, vel, epoch, name='(Unknown)', mass=1*kilogram, engine:force_function_type = NO_FORCES, external_forces: force_function_type = NO_FORCES):
        """
        pos, vel, epoch, name, mass  Current body state.
        
        engine  A function which should return the thrust in newtons and the mass flow (negative for loss);
                gets called like that: engine(t, x, v) where t is the number
                of seconds since the epoch of the body and 'x' and 'v' 
                are the position and velocity of the simulated body at 't'.

        external_forces  A function which should return the external forcies in newtons and mass flow (negative for loss);
                for example, atmospheric drag modeling, gets called like the function 'engine' above.
        """
        self.pos = pos
        self.vel = vel
        self.epoch = epoch
        self.name = name
        self.mass = mass
        self.engine = engine
        self.external_forces = external_forces
        self.landed = False
    
    def update(self, t):
        'Update body from a calculated trajectory, this is as if the body has traveled along'
        if len(t.points) == 0: return
        if self.epoch.jde != t.epoch_begin.jde:
            print(f"Couldn't update. Body epoch ({self.epoch}) is different to the beginning of the trajectory ({t.epoch_begin}).")
            return
        
        self.pos, self.vel = t.points[-1]
        self.mass = t.end_mass
        self.epoch = t.epoch_end
        self.landed = t.landed

MU_NP = strip_units(STANDARD_GRAVITATIONAL_PARAMETER)

@njit
def solve_rk4_fast(pos_0: types.float32[:], vel_0: types.float32[:], t_0: types.float32, t_1: types.float32, hmin: types.float32, hmax: types.float32, forces: force_function_type = NO_FORCES, external_forces: force_function_type = NO_FORCES, mass: types.float32 = 1, tol: types.float32 =1e-3):
    """
    Same as solve_rk4, but with JIT and only works on numpy
    """
    pos_0 = pos_0.copy()
    vel_0 = vel_0.copy()
    if t_1 <= t_0: return pos_0, vel_0, t_0, 0, mass
    
    tol_vel = tol
    tol_pos = tol
    
    forces_work = 0
    
    def accel(t, x, v):
        a1, m1 = forces(t, x, v)
        a2, m2 = external_forces(t, x, v)

        ag = -MU_NP * x / norm(x)**3
        
        # Note: mass flow is not updated yet, because the time step might get discarded
        a = ag + (a1 + a2) / 1000 / mass      # / 1000 because we use 'kilometer/kilogram/second' units 
        return a, m1 + m2
    
    h = hmax
    
    while t_0 < t_1:
        a, mdot = accel(t_0, pos_0, vel_0)
        k1m = h * mdot
        k1v = h * a
        k1x = h * vel_0
        
        a, mdot = accel(t_0 + h/2, pos_0 + k1x/2, vel_0 + k1v/2)
        k2m = h * mdot
        k2v = h * a
        k2x = h * (vel_0 + k1v/2)
        
        a, mdot = accel(t_0 + h/2, pos_0 + k2x/2, vel_0 + k2v/2)
        k3m = h * mdot
        k3v = h * a
        k3x = h * (vel_0 + k2v/2)

        a, mdot = accel(t_0 + h, pos_0 + k3x, vel_0 + k3v)
        k4m = h * mdot
        k4v = h * a
        k4x = h * (vel_0 + k3v)
        
        mass_1 = mass + (k1m + 2 * k2m + 2 * k3m + k4m) / 6
        #if mass_1 != mass:
        #    print(mass, "new mass", mass_1, "h", h, "mdot", mdot)
        vel_1 = vel_0 + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        pos_1 = pos_0 + (k1x + 2 * k2x + 2 * k3x + k4x) / 6

        if hmin is not None:
            # Compute another estimate of the solution with step sizes h/2
            vel_1_smaller_h = vel_0 + (k1v/2 + k2v + k3v + k4v/2) / 6
            pos_1_smaller_h = pos_0 + (k1x/2 + k2x + k3x + k4x/2) / 6
        
            err_vel = norm(vel_1 - vel_1_smaller_h)
            err_pos = norm(pos_1 - pos_1_smaller_h)
            
            # If the error is within the tolerance, accept the solution and increase the step size
            if (err_vel < tol_vel and err_pos < tol_pos) or h == hmin:
                if forces is not None:
                    forces_work += forces(t_0, pos_0, vel_0)[0].dot(pos_1 - pos_0)
                pos_0 = pos_1
                vel_0 = vel_1
                mass = mass_1
                t_0 += h
                h = min(hmax, h*2)
            else:
                # If the error is too large, reject the solution and decrease the step size
                h = max(hmin, h/2)
        else:
            # Don't do adaptive time step if hmin is passed as None
            pos_0 = pos_1
            vel_0 = vel_1
            t_0 += h
    return pos_0, vel_0, t_0, forces_work, mass

def solve_rk4(pos_0, vel_0, t_0, t_1, hmin, hmax, forces=None, mass=1*kilogram, tol=1e-3, nm=sp):
    """
    pos_0, vel_0, t_0  Describe the initial conditions
    t_1                Up until when to integrate
    hmin               Min time step (pass None to disable adaptive time step)
    hmax               Max time step (h starts at this); we choose an adaptive step based on the tolerance
    tol                If numerical error is bigger than this, then we increase the timestep, if it's 10 times 
                       smaller, we decrease the time step

    forces  Should be None or a function that returns a 3D vector with units newtons
            to be accounted for as acceleration in addition to gravity; we call forces(t, x, v) 
            where 't' is the number of seconds since the epoch of the body (usually when 
            the trajectory integration began), and 'x' and 'v' are the position and velocity at 't'.
    
    mass  Should be set if 'forces' are present
    """
    pos_0 = pos_0.copy()
    vel_0 = vel_0.copy()
    if t_1 <= t_0: return pos_0, vel_0, t_0, 0
    
    mu = STANDARD_GRAVITATIONAL_PARAMETER_EARTH
    if nm == np: 
        pos_0 = strip_units(pos_0)
        vel_0 = strip_units(vel_0)
        t_0 = strip_units(t_0)
        t_1 = strip_units(t_1)
        mass = strip_units(mass)
        mu = strip_units(mu)
        tol_vel = tol
        tol_pos = tol
    else:
        tol_vel = tol * kilometer/second
        tol_pos = tol * kilometer
    
    forces_work = 0
    
    if forces is None:
        accel = lambda t, x, v: (-mu * x / nm.norm(x)**3)
    else:
        if nm == np: 
            accel = lambda t, x, v: (-mu * x / nm.norm(x)**3) + forces(t, x, v) / 1000 / mass
        else:
            accel = lambda t, x, v: (-mu * x / nm.norm(x)**3) + convert_to(forces(t, x, v), [kilometer, kilogram, second]) / (strip_units(convert_to(mass, kilogram)) * kilogram)

    h = hmax
    
    while t_0 < t_1:
        k1v = h * accel(t_0, pos_0, vel_0)
        k1x = h * vel_0
        
        k2v = h * accel(t_0 + h/2, pos_0 + k1x/2, vel_0 + k1v/2)
        k2x = h * (vel_0 + k1v/2)
        
        k3v = h * accel(t_0 + h/2, pos_0 + k2x/2, vel_0 + k2v/2)
        k3x = h * (vel_0 + k2v/2)

        k4v = h * accel(t_0 + h, pos_0 + k3x, vel_0 + k3v)
        k4x = h * (vel_0 + k3v)
        
        vel_1 = vel_0 + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        pos_1 = pos_0 + (k1x + 2 * k2x + 2 * k3x + k4x) / 6

        if hmin is not None:
            # Compute another estimate of the solution with step sizes h/2
            vel_1_smaller_h = vel_0 + (k1v/2 + k2v + k3v + k4v/2) / 6
            pos_1_smaller_h = pos_0 + (k1x/2 + k2x + k3x + k4x/2) / 6
        
            err_vel = nm.norm(vel_1 - vel_1_smaller_h)
            err_pos = nm.norm(pos_1 - pos_1_smaller_h)
            
            # If the error is within the tolerance, accept the solution and increase the step size
            if (err_vel < tol_vel and err_pos < tol_pos) or h == hmin:
                if forces is not None:
                    forces_work += forces(t_0, pos_0, vel_0).dot(pos_1 - pos_0)
                pos_0 = pos_1
                vel_0 = vel_1
                t_0 += h
                h = min(hmax, h*2)
            else:
                # If the error is too large, reject the solution and decrease the step size
                h = max(hmin, h/2)
        else:
            # Don't to adaptive time step if hmin is passed as None
            pos_0 = pos_1
            vel_0 = vel_1
            t_0 += h
    return pos_0, vel_0, t_0, forces_work