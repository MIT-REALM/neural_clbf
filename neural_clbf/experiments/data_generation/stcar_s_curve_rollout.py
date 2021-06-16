from copy import copy
from typing import Callable

import numpy as np
import torch
import tqdm

from neural_clbf.systems import STCar


@torch.no_grad()
def save_stcar_s_curve_rollout(
    controller_fn: Callable[[torch.Tensor], torch.Tensor],
    controller_name: str,
    controller_period: float,
    dynamics_model: STCar,
    randomize_path: bool = False,
):
    """
    Simulate a rollout of the STCar and saves it to a CSV

    args:
        controller_fn: the function from x to control input
        controller_name: the name of the controller (e.g. neural_clbf or robust_mpc)
        controller_period: how often to call that controller
        dynamics_model: the STCar model used to simulate
        randomize_path: if True, generate a random path. If false, just use a sine to
                        vary omega_ref
    """
    simulation_dt = dynamics_model.dt

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0
    n_sims = 1
    T = int(t_sim // simulation_dt)
    start_x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x_sim = torch.zeros(T, n_sims, dynamics_model.n_dims)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x

    u_sim = torch.zeros(T, n_sims, dynamics_model.n_controls)

    # And create a place to store the reference path
    params = copy(dynamics_model.nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(T)
    y_ref = np.zeros(T)
    psi_ref = np.zeros(T)
    psi_ref[0] = 1.0
    omega_ref = np.zeros(T)

    # If we're randomizing the reference path, set the period at which to randomize
    if randomize_path:
        path_update_freq = int(T / 10.0)

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="S-Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the path parameters at this point
        if randomize_path:
            if tstep == 1 or tstep % path_update_freq == 0:
                omega_ref[tstep] = np.random.uniform(low=-1.5, high=1.5)
            else:
                omega_ref[tstep] = omega_ref[tstep - 1]
        else:
            omega_ref[tstep] = 1.5 * np.sin(tstep * simulation_dt)

        psi_ref[tstep] = simulation_dt * omega_ref[tstep] + psi_ref[tstep - 1]
        pt = copy(params)
        pt["omega_ref"] = omega_ref[tstep]
        pt["psi_ref"] = psi_ref[tstep]
        x_ref[tstep] = x_ref[tstep - 1] + simulation_dt * pt["v_ref"] * np.cos(
            psi_ref[tstep]
        )
        y_ref[tstep] = y_ref[tstep - 1] + simulation_dt * pt["v_ref"] * np.sin(
            psi_ref[tstep]
        )

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
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

    # Save the data
    data_to_save = np.hstack(
        [
            x_sim[:, 0, :],
            np.reshape(omega_ref, (-1, 1)),
            np.reshape(psi_ref, (-1, 1)),
            params["v_ref"] * np.ones((omega_ref.shape[0], 1)),
            params["a_ref"] * np.ones((omega_ref.shape[0], 1)),
            np.reshape(x_ref, (-1, 1)),
            np.reshape(y_ref, (-1, 1)),
        ]
    )

    # Figure out the filename
    controller_period_str = f"{controller_period}".replace(".", "-")
    filename = "sim_traces/"
    filename += f"stcar_{controller_name}_dt={controller_period_str}"
    if randomize_path:
        filename += f"_omega_ref=1-5-randomized-{np.random.randint(10000)}"
    else:
        filename += "_omega_ref=1-5-sine"

    filename += ".csv"

    # Construct the header
    header = "sxe,sye,delta,ve,psi_e,psi_e_dot,beta,omega_ref,psi_ref,v_ref,a_ref"
    header += ",x_ref,y_ref"

    np.savetxt(filename, data_to_save, delimiter=",", header=header, comments="")
