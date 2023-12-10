import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sympy.physics.units import meter, kilometer, second, hour, minute, kilogram, convert_to

from my_orbit_lib.extensions import strip_units

from my_orbit_lib.body import solve_rk4_fast, NO_FORCES
from my_orbit_lib.epoch import Epoch

from my_orbit_lib.constants import EARTH_RADIUS

from numba import njit

class Trajectory:
    def __init__(self, points, epoch_begin, epoch_end, end_mass, forces_work=[], is_maneuver=False, landed=False, label=None):
        """
        points       Should be an np.array of shape (n, 2, 3), i.e. pos and vel vectors at 'n' points
        is_maneuver  Should be true if the engines were on during the trajectory (this field
                     is just used as metadata for pretty plots. 
        """
        self.points = points
        self.epoch_begin = epoch_begin 
        self.epoch_end = epoch_end
        self.end_mass = end_mass
        self.forces_work = forces_work
        self.is_maneuver = is_maneuver
        self.landed = landed
        self.label = label
        self.color = 'cyan' if is_maneuver else 'darkorange'

def get_trajectory(body, target_epoch=None, duration=None, until_condition_becomes_false=None, step_size=1, hmin=1e-2, hmax=1e2, tol=1e-3):
    """
    Gets the trajectory between two points of time.
    Each trajectory contains an array of points, whose granularity are specified by the argument step_size.

    step_size  Figures out the number of steps by dividing the duration by this;
               the result is how many points you'll get in the result.
               step_size must be sufficiently small when specifying 'until_condition_becomes_false'
               because in that case we check for the condition at each step (and not at each step size).

    There's a number of ways to specify the two points of time:

    1) Set duration parameter to the number of seconds for which to integrate.
    2) Set target_epoch which gets used to figure out the time delta
       from the current body epoch and integrate from there.
    3) Set until_condition_becomes_false to a lambda that 
       receives the arguments: n, t, x, v   at each step and 
       returns whether to continue integrating, where

       n  is the step number
       t  is the time since beginning of integration
       x  is the current body's position
       v  is the current body's velocity 
    """
    if until_condition_becomes_false is not None:
        if target_epoch is not None:
            print("Warning: until_condition_becomes_false is set, ignoring target_epoch")
        if duration is not None:
            print("Warning: until_condition_becomes_false is set, ignoring duration")
        duration = -1
    else:
        if target_epoch is not None:
            if body.epoch.jde > target_epoch.jde:
                raise ValueError('Body epoch is after target epoch')
            if duration is not None:
                print("Warning: target_epoch is set, ignoring duration")
            duration = (target_epoch.jde - body.epoch.jde) * 86400
        else:
            if duration is None:
                raise ValueError('Please specify a duration in one of the following arguments: duration, '
                                 'target_epoch or until_condition_becomes_false.')
            duration = strip_units(convert_to(duration, second))

    pos, vel = strip_units(body.pos), strip_units(body.vel)
    duration = strip_units(convert_to(duration, second))
    step_size = strip_units(convert_to(step_size, second))
    mass = strip_units(convert_to(body.mass, kilogram))
    
    if duration != -1:
        steps = duration/step_size
    else:
        steps = -1
        
    points = [np.array([pos.copy(), vel.copy()])] # Initial point
    forces_work = [0]
    n = 0
    t = 0
    
    duration_condition = lambda: steps != -1 and n < steps
    other_condition = lambda: until_condition_becomes_false is not None and until_condition_becomes_false(n, t, pos, vel)
    
    landed = False

    while duration_condition() or other_condition():
        pos, vel, t, fw, mass = solve_rk4_fast(pos_0=pos,
                                    vel_0=vel,
                                    t_0=t,
                                    t_1=t + step_size,
                                    hmin=hmin,
                                    hmax=hmax,
                                    forces=body.engine,
                                    external_forces=body.external_forces,
                                    mass=mass,
                                    tol=tol)
        forces_work.append(forces_work[-1] + fw)
        n += 1

        if np.norm(np.array(pos)) < strip_units(EARTH_RADIUS):
            landed = True
            vel = np.array([0, 0, 0])

        points.append(np.array([pos.copy(), vel.copy()]))

        if landed:
            break

    epoch_begin = body.epoch
    epoch_end = Epoch(body.epoch.jde + t/86400)
    is_maneuver = body.engine != NO_FORCES
    return Trajectory(np.array(points), epoch_begin=epoch_begin, epoch_end=epoch_end, end_mass=mass, forces_work=forces_work, is_maneuver=is_maneuver, landed=landed)

def plot_trajectory(trajectories=[], body=None, vel_arrow_scale=400, x_view_angle=-75, y_view_angle=25, scale=0.7):
    """
    Plots the trajectories in a nice visualization.
    
    If body is not None we plot it's current position and velocity vector 
    and use it's name and epoch for the title.
    """
    
    fig = plt.figure()
    if body is not None:
        plt.title(f'Trajectory Plot: {body.name}')
    else:
        plt.title('Trajectory Plot')

    ax = fig.add_subplot(1, 1, 1, projection=Axes3D.name)
    ax.view_init(y_view_angle, x_view_angle)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    
    # Coordinate system arrows
    RADIUS = strip_units(EARTH_RADIUS)
    r = RADIUS * 0.8
    ax.quiver([0], [0], [0], [r], [0], [0], color='r', linewidth = 0.5)
    ax.quiver([0], [0], [0], [0], [r], [0], color='g', linewidth = 0.5)
    ax.quiver([0], [0], [0], [0], [0], [r], color='b', linewidth = 0.5)

    # Plot Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    r = RADIUS
    ax.plot_wireframe(x * r, y * r, z * r, color=(0.1, 0.2, 0.5, 0.2), linewidth=0.5)

    # Plot first body position 
    if len(trajectories):
        t = trajectories[0]
        sat_pos, _ = t.points[0]

        label = f'Body on {t.epoch_begin.todatetime().strftime("%d/%m/%Y, %H:%M:%S UTC")}'
        sat = ax.scatter([sat_pos[0]], [sat_pos[1]], [sat_pos[2]], label=label,color="green")

    # Plot numerical points
    last_point = None
    last_point_t = None
    
    for trajectory in trajectories:
        if len(trajectory.points) == 0: continue
        
        t_duration = (trajectory.epoch_end.jde - trajectory.epoch_begin.jde) * 86400
        label = 'Engine Burn {:.2f}s'.format(t_duration) if trajectory.is_maneuver else 'Trajectory'
        style = '-.' if trajectory.is_maneuver else '--'
        color = 'cyan' if trajectory.is_maneuver else 'darkorange'
        
        # If we have at least 3 points to form a path...
        if len(trajectory.points) > 2:
            ax.plot(trajectory.points[:, 0, 0], trajectory.points[:, 0, 1], trajectory.points[:, 0, 2], style, label=label, color=color)
            if trajectory.is_maneuver:
                for p in [trajectory.points[0], trajectory.points[-1]]: 
                    ax.scatter(p[0, 0], p[0, 1], p[0, 2], color=color)
        else:
            # else scatter points
            for p in trajectory.points:
                ax.scatter(p[0, 0], p[0, 1], p[0, 2], color=color)
            ax.scatter(-1000000, 0, 0, label=label, color=color) # Legend hack to make sure only one entry appears
        
        if last_point_t is None or (last_point_t.jde < trajectory.epoch_end.jde):
            last_point = trajectory.points[-1]
            last_point_t = trajectory.epoch_end

    # Plot last body position 
    if last_point is not None:
        label = f'Body on {last_point_t.todatetime().strftime("%d/%m/%Y, %H:%M:%S UTC")}'

        sat_pos, sat_vel = last_point.copy()
        sat_vel *= vel_arrow_scale
        
        sat = ax.scatter([sat_pos[0]], [sat_pos[1]], [sat_pos[2]], label=label)
        ax.quiver([sat_pos[0]], [sat_pos[1]], [sat_pos[2]], [sat_vel[0]], [sat_vel[1]], [sat_vel[2]], color=sat.get_fc())
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.3, 1), borderaxespad=0)

    # Rest of plot
    r = RADIUS * scale
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    
    ax.set_axis_off()
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()