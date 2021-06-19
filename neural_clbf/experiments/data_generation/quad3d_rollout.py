from copy import copy
from typing import Callable

import numpy as np
import torch
import tqdm

from neural_clbf.systems import Quad3D


@torch.no_grad()
def quad3d_rollout(
    controller_fn: Callable[[torch.Tensor], torch.Tensor],
    controller_name: str,
    controller_period: float,
    dynamics_model: Quad3D,
    save: bool = False,
):
    """
    Simulate a rollout of the Quad3D and saves it to a CSV

    args:
        controller_fn: the function from x to control input
        controller_name: the name of the controller (e.g. neural_clbf or robust_mpc)
        controller_period: how often to call that controller
        dynamics_model: the Quad3D model used to simulate
        save: True to save to csv
    """
    simulation_dt = dynamics_model.dt

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 1
    n_sims = 1
    T = int(t_sim // simulation_dt)
    start_x = torch.zeros(n_sims, dynamics_model.n_dims) + 1.0
    start_x[:, Quad3D.PZ] = -1.0
    x_sim = torch.zeros(T, n_sims, dynamics_model.n_dims)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x

    u_sim = torch.zeros(T, n_sims, dynamics_model.n_controls)

    # And create a place to store the reference path
    params = copy(dynamics_model.nominal_params)
    params["mass"] = round(torch.Tensor(n_sims, 1).uniform_(1.0, 1.5).item(), 4)

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="Rollout", leave=True)
    for tstep in prog_bar_range:
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = controller_fn(x_current)
            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Update the dynamics
        for i in range(n_sims):
            xdot = dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                params,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

    # Save the data
    data_to_save = np.hstack([x_sim[:, 0, :]])

    import matplotlib.pyplot as plt

    plt.plot(x_sim[:, 0, Quad3D.PZ])
    plt.show()

    # Figure out the filename
    filename = "sim_traces/"
    controller_period_str = f"{controller_period}".replace(".", "-")
    mass_str = f"{params['mass']}".replace(".", "-")
    filename += f"quad3d_{controller_name}_dt={controller_period_str}"
    filename += f"_m=random{mass_str}"
    filename += ".csv"

    # Construct the header
    header = "px,py,pz,vx,vy,vz,phi,theta,psi"

    if save:
        np.savetxt(filename, data_to_save, delimiter=",", header=header, comments="")

    # Return goal error and safety
    last_quarter = int(x_sim.shape[0] * 0.75)
    goal_error = x_sim[last_quarter:, 0, Quad3D.PZ].abs().min().item()
    failure = torch.any(dynamics_model.unsafe_mask(x_sim[:, 0, :]))

    return goal_error, failure
