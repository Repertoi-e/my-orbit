import sympy as sp
import numpy as np

from sympy.physics.units import meter, kilometer, second, hour, minute, kilogram, convert_to, quantities

from my_orbit_lib.extensions import *
from my_orbit_lib.constants import STANDARD_GRAVITATIONAL_PARAMETER_EARTH

from my_orbit_lib.epoch import Epoch

class Body:
    def __init__(self, pos, vel, epoch, name='(Unknown)', mass=1*kilogram, engine=None):
        """
        pos, vel, epoch, name, mass  Current body state.
        
        engine  A function which should return the thrust in newtons;
                gets called like that: engine(t, x, v) where t is the number
                of seconds since the epoch of the body and 'x' and 'v' 
                are the position and velocity of the simulated body at 't'.
        """
        self.pos = pos
        self.vel = vel
        self.epoch = epoch
        self.name = name
        self.mass = mass
        self.engine = engine
    
    def update(self, t):
        'Update body from a calculated trajectory, this is as if the body has traveled along'
        if len(t.points) == 0: return
        if self.epoch.jde != t.epoch_begin.jde:
            print(f"Couldn't update. Body epoch ({self.epoch}) is different to the beginning of the trajectory ({t.epoch_begin}).")
            return
        
        self.pos, self.vel = t.points[-1]
        self.epoch = t.epoch_end
    
def solve_keplers_problem_rk4(pos_0, vel_0, t_0, t_1, hmin, hmax, forces=None, mass=1*kilogram, tol=1e-3, nm=sp):
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

