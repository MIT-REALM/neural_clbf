from copy import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from neural_clbf.systems import Quad3D
from neural_clbf.setup.robust_mpc import robust_mpc_path  # type: ignore

import matlab  # type: ignore
import matlab.engine  # type: ignore

if __name__ == "__main__":
    # Import the plotting callbacks, which seem to be needed to load from the checkpoint
    from neural_clbf.experiments.train_kinematic_car import (  # noqa
        rollout_plotting_cb,  # noqa
        clbf_plotting_cb,  # noqa
    )


def doMain():

    controller_period = 0.1
    simulation_dt = 0.001

    # Define the dynamics model
    nominal_params = {
        "m": 1.0,
    }
    quad3d = Quad3D(nominal_params, dt=simulation_dt, controller_dt=controller_period)

    single_rollout_s_path(quad3d)
    plt.show()


@torch.no_grad()
def single_rollout_s_path(
    dynamics_model: "Quad3D",
) -> Tuple[str, plt.figure]:
    simulation_dt = dynamics_model.dt
    controller_period = dynamics_model.controller_dt

    eng = matlab.engine.connect_matlab()
    eng.cd(robust_mpc_path)

    # Make sure the controller if for a Quad3D
    if not isinstance(dynamics_model, Quad3D):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 3.0
    n_sims = 1
    T = int(t_sim // simulation_dt)
    start_x = torch.tensor([[1.0, 1.0, 1, 1, 1, -1, 1, 1, 1]], device="cpu")
    x_sim = torch.zeros(T, n_sims, dynamics_model.n_dims).type_as(start_x)
    x_sim[0, :, :] = start_x
    u_sim = torch.zeros(T, n_sims, dynamics_model.n_controls).type_as(start_x)

    # And create a place to store the reference path
    params = copy(dynamics_model.nominal_params)
    params["m"] = 1.0

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="Rollout", leave=True)
    for tstep in prog_bar_range:
        # for the nominal controller
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            A, B = dynamics_model.linearized_dt_dynamics_matrices()

            x_current_np = x_current.cpu().numpy().T
            A = matlab.double(A.tolist())
            B = matlab.double(B.tolist())
            x_current_np = matlab.double(x_current_np.tolist())
            u_matlab = eng.mpc_quad3d(A, B, x_current_np)
            u_sim[tstep, :, :] = torch.from_numpy(np.array(u_matlab, dtype=np.float32))

        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                params,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        t_final = tstep

    # Plot!
    plt.plot(x_sim[:, 0, 2])


if __name__ == "__main__":
    doMain()
