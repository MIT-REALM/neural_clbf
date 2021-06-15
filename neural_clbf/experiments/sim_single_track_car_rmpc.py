from copy import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from neural_clbf.systems import STCar
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

    controller_period = 0.15
    simulation_dt = 0.001

    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    stcar = STCar(nominal_params, dt=simulation_dt, controller_dt=controller_period)

    single_rollout_s_path(stcar)
    plt.show()


@torch.no_grad()
def single_rollout_s_path(
    dynamics_model: "STCar",
) -> Tuple[str, plt.figure]:
    simulation_dt = dynamics_model.dt
    controller_period = dynamics_model.controller_dt

    eng = matlab.engine.connect_matlab()
    eng.cd(robust_mpc_path)

    # Make sure the controller if for a STCar
    if not isinstance(dynamics_model, STCar):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0 / 1
    n_sims = 1
    T = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, -np.pi / 6, 0.0, 0.0]], device="cpu"
    )
    x_sim = torch.zeros(T, n_sims, dynamics_model.n_dims).type_as(start_x)
    u_sim = torch.zeros(T, n_sims, dynamics_model.n_controls).type_as(start_x)

    # And create a place to store the reference path
    params = copy(dynamics_model.nominal_params)
    params["omega_ref"] = 0.3
    params["mu_scale"] = 1.0
    x_ref = np.zeros(T)
    y_ref = np.zeros(T)
    psi_ref = np.zeros(T)
    psi_ref[0] = 1.0
    omega_ref = np.zeros(T)

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="S-Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the path parameters at this point
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
            u_matlab = eng.mpc_stcar(A, B, x_current_np)
            u_sim[tstep, :, :] = torch.from_numpy(np.array(u_matlab, dtype=np.float32))

        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        t_final = tstep

    # Plot!
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    # fig.set_size_inches(6, 12)

    # Get reference path
    t = np.linspace(0, t_sim, T)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T
    psi_ref = np.tile(psi_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, dynamics_model.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, dynamics_model.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(
        x_world[:t_final, 0],
        y_world[:t_final, 0],
        linestyle="dashed",
        label="RMPC",
        color="green",
    )
    ax1.plot(
        x_ref[:t_final, 0],
        y_ref[:t_final, 0],
        linestyle="dotted",
        label="Ref",
        color="black",
    )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_ylim([np.min(y_ref) - 3, np.max(y_ref) + 3])
    ax1.set_xlim([np.min(x_ref) - 3, np.max(x_ref) + 3])
    # ax1.set_aspect("equal")
    ax1.set_title("Single-Track Car Trajectory", fontsize=14)

    # Correct for reference steering angle when calculating tracking error
    car_params = dynamics_model.car_params
    g = 9.81  # [m/s^2]
    C_Sf = -car_params.tire.p_ky1 / car_params.tire.p_dy1
    C_Sr = -car_params.tire.p_ky1 / car_params.tire.p_dy1
    lf = car_params.a
    lr = car_params.b
    x0 = 0.0 * x_sim
    x0[:, 0, :] += dynamics_model.goal_point.type_as(x0)
    x0[:, 0, STCar.PSI_E_DOT] = torch.tensor(omega_ref).type_as(x0)
    x0[:, 0, STCar.DELTA] = torch.tensor(
        (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
        / (lf * C_Sf * g * lr)
        * omega_ref
        / params["v_ref"]
    ).type_as(x0)
    x0[:, 0, STCar.DELTA] /= lf * C_Sf * g * lr

    x_sim -= x0

    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(
        t[:t_final],
        x_sim[:t_final, 0, : STCar.SYE + 1].norm(dim=-1).squeeze().cpu().numpy(),
        linestyle=":",
        label="RMPC",
        color="green",
    )

    ax2.legend()
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("Tracking error")
    ax2.set_title("Single-Track Car Tracking Error", fontsize=14)

    fig.tight_layout()

    # Return the figure along with its name
    return "S-Curve Tracking", fig


if __name__ == "__main__":
    doMain()
