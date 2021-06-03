from copy import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import STCar

if __name__ == "__main__":
    # Import the plotting callbacks, which seem to be needed to load from the checkpoint
    from neural_clbf.experiments.train_kinematic_car import (  # noqa
        rollout_plotting_cb,  # noqa
        clbf_plotting_cb,  # noqa
    )


def doMain():
    checkpoint_file = "saved_models/stcar/774ba0b.ckpt"

    controller_period = 0.01
    simulation_dt = 0.001

    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    stcar = STCar(nominal_params, dt=simulation_dt, controller_dt=controller_period)

    # Initialize the DataModule
    initial_conditions = [
        (-0.1, 0.1),  # sxe
        (-0.1, 0.1),  # sye
        (-0.1, 0.1),  # delta
        (-0.1, 0.1),  # ve
        (-0.1, 0.1),  # psi_e
        (-0.1, 0.1),  # psi_dot
        (-0.1, 0.1),  # beta
    ]

    # Define the scenarios
    scenarios = []
    omega_ref_vals = [-1.5, 1.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    data_module = EpisodicDataModule(
        stcar,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=10,
        fixed_samples=100,
        max_points=5000000,
        val_split=0.1,
        batch_size=64,
    )

    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint_file,
        map_location=torch.device("cpu"),
        dynamics_model=stcar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clbf_lambda=0.1,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=1e1,
        num_controller_init_epochs=5,
        epochs_per_episode=10,
    )

    single_rollout_s_path(clbf_controller)
    plt.show()


@torch.no_grad()
def single_rollout_s_path(
    clbf_controller: "NeuralCLBFController",
) -> Tuple[str, plt.figure]:
    simulation_dt = clbf_controller.dynamics_model.dt
    controller_period = clbf_controller.controller_period

    # Make sure the controller if for a STCar
    if not isinstance(clbf_controller.dynamics_model, STCar):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0 / 1
    n_sims = 1
    T = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, -np.pi / 6, 0.0, 0.0]], device=clbf_controller.device
    )
    x_sim = torch.zeros(T, n_sims, clbf_controller.dynamics_model.n_dims).type_as(
        start_x
    )
    V_sim = torch.zeros(T, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)
    Vdot_sim = torch.zeros(T, n_sims, clbf_controller.n_scenarios + 1, 1).type_as(
        start_x
    )

    u_sim = torch.zeros(T, n_sims, clbf_controller.dynamics_model.n_controls).type_as(
        start_x
    )

    # Also create somewhere to save the simulations from the learned controller...
    x_nn = torch.clone(x_sim)
    V_nn = torch.clone(V_sim)
    Vdot_nn = torch.clone(Vdot_sim)
    u_nn = torch.clone(u_sim)

    # And the nominal controller
    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)
    Vdot_nominal = torch.clone(Vdot_sim)
    u_nominal = torch.clone(u_sim)

    # And create a place to store the reference path
    params = copy(clbf_controller.dynamics_model.nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(T)
    y_ref = np.zeros(T)
    psi_ref = np.zeros(T)
    psi_ref[0] = 1.0
    omega_ref = np.zeros(T)

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="S-Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the path parameters at this point
        omega_ref[tstep] = 1.5 * np.sign(np.sin(tstep * simulation_dt))
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
            u = clbf_controller(x_current)
            u_sim[tstep, :, :] = u

            # Also predict the difference in V from linearization
            # ... in each scenario
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current)
            for i in range(clbf_controller.n_scenarios):
                Vdot = Lf_V[:, i, :].unsqueeze(1) + torch.bmm(
                    Lg_V[:, i, :].unsqueeze(1), u.unsqueeze(-1)
                )
                Vdot = Vdot.reshape(-1, 1)
                Vdot_sim[tstep, :, i, 0] = Vdot
            # and with the true parameters
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current, [pt])
            Vdot = Lf_V[:, 0, :].unsqueeze(1) + torch.bmm(
                Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(-1)
            )
            Vdot = Vdot.reshape(-1, 1)
            Vdot_sim[tstep, :, -1, 0] = Vdot
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

            # Copy the estimate of Vdot
            Vdot = Vdot_sim[tstep - 1, :, :, :]
            Vdot_sim[tstep, :, :, :] = Vdot

        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_sim[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        # and repeat for the NN controller
        # Get the current state
        x_current = x_nn[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_controller.u(x_current)
            u_nn[tstep, :, :] = u

            # Also predict the difference in V from linearization
            # ... in each scenario
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current)
            for i in range(clbf_controller.n_scenarios):
                Vdot = Lf_V[:, i, :].unsqueeze(1) + torch.bmm(
                    Lg_V[:, i, :].unsqueeze(1), u.unsqueeze(-1)
                )
                Vdot = Vdot.reshape(-1, 1)
                Vdot_nn[tstep, :, i, 0] = Vdot
            # and with the true parameters
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current, [pt])
            Vdot = Lf_V[:, 0, :].unsqueeze(1) + torch.bmm(
                Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(-1)
            )
            Vdot = Vdot.reshape(-1, 1)
            Vdot_nn[tstep, :, -1, 0] = Vdot
        else:
            u = u_nn[tstep - 1, :, :]
            u_nn[tstep, :, :] = u

            # Copy the estimate of Vdot
            Vdot = Vdot_nn[tstep - 1, :, :, :]
            Vdot_nn[tstep, :, :, :] = Vdot

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_nn[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_nn[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_nn[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        # and repeat for the nominal controller
        # Get the current state
        x_current = x_nominal[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_controller.dynamics_model.u_nominal(x_current, pt)
            u_nominal[tstep, :, :] = u

            # Also predict the difference in V from linearization
            # ... in each scenario
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current)
            for i in range(clbf_controller.n_scenarios):
                Vdot = Lf_V[:, i, :].unsqueeze(1) + torch.bmm(
                    Lg_V[:, i, :].unsqueeze(1), u.unsqueeze(-1)
                )
                Vdot = Vdot.reshape(-1, 1)
                Vdot_nominal[tstep, :, i, 0] = Vdot
            # and with the true parameters
            Lf_V, Lg_V = clbf_controller.V_lie_derivatives(x_current, [pt])
            Vdot = Lf_V[:, 0, :].unsqueeze(1) + torch.bmm(
                Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(-1)
            )
            Vdot = Vdot.reshape(-1, 1)
            Vdot_nominal[tstep, :, -1, 0] = Vdot
        else:
            u = u_nominal[tstep - 1, :, :]
            u_nominal[tstep, :, :] = u

            # Copy the estimate of Vdot
            Vdot = Vdot_nominal[tstep - 1, :, :, :]
            Vdot_nominal[tstep, :, :, :] = Vdot

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_nominal[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_nominal[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_nominal[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        t_final = tstep

    # Plot!
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    fig.set_size_inches(12, 12)

    # Get reference path
    t = np.linspace(0, t_sim, T)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T
    psi_ref = np.tile(psi_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nn = x_nn[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_nn = x_nn[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world_nn = x_ref + x_err_nn * np.cos(psi_ref) - y_err_nn * np.sin(psi_ref)
    y_world_nn = y_ref + x_err_nn * np.sin(psi_ref) + y_err_nn * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world_nom = x_ref + x_err_nom * np.cos(psi_ref) - y_err_nom * np.sin(psi_ref)
    y_world_nom = y_ref + x_err_nom * np.sin(psi_ref) + y_err_nom * np.cos(psi_ref)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(
        x_world[:t_final, 0],
        y_world[:t_final, 0],
        linestyle="solid",
        label="CLBF",
        color="red",
    )
    ax1.plot(
        x_world_nn[:t_final, 0],
        y_world_nn[:t_final, 0],
        linestyle="dashdot",
        label="NN",
        color="blue",
    )
    ax1.plot(
        x_world_nom[:t_final, 0],
        y_world_nom[:t_final, 0],
        linestyle="dashed",
        label="Nominal",
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
    ax1.set_aspect("equal")

    # Correct for reference steering angle when calculating tracking error
    car_params = clbf_controller.dynamics_model.car_params
    g = 9.81  # [m/s^2]
    C_Sf = -car_params.tire.p_ky1 / car_params.tire.p_dy1
    C_Sr = -car_params.tire.p_ky1 / car_params.tire.p_dy1
    lf = car_params.a
    lr = car_params.b
    x0 = 0.0 * x_sim
    x0[:, 0, :] += clbf_controller.dynamics_model.goal_point.type_as(x0)
    x0[:, 0, STCar.PSI_E_DOT] = torch.tensor(omega_ref).type_as(x0)
    x0[:, 0, STCar.DELTA] = torch.tensor(
        (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
        / (lf * C_Sf * g * lr)
        * omega_ref
        / params["v_ref"]
    ).type_as(x0)
    x0[:, 0, STCar.DELTA] /= lf * C_Sf * g * lr

    x_sim -= x0
    x_nn -= x0
    x_nominal -= x0

    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(
        t[:t_final],
        x_sim[:t_final, 0, :].norm(dim=-1).squeeze().cpu().numpy(),
        linestyle="-",
        label="CLBF",
        color="red",
    )
    ax2.plot(
        t[:t_final],
        x_nn[:t_final, 0, :].norm(dim=-1).squeeze().cpu().numpy(),
        linestyle=":",
        label="NN",
        color="blue",
    )
    ax2.plot(
        t[:t_final],
        x_nominal[:t_final, 0, :].norm(dim=-1).squeeze().cpu().numpy(),
        linestyle=":",
        label="Nominal",
        color="green",
    )

    ax2.legend()
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("Tracking error")

    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.plot(
        t[:t_final],
        V_sim[:t_final, :, :].squeeze().cpu().numpy(),
        label="CLBF",
        linestyle="solid",
        color="red",
    )
    ax3.plot(
        t[:t_final],
        V_nn[:t_final, :, :].squeeze().cpu().numpy(),
        label="NN",
        linestyle="solid",
        color="blue",
    )
    ax3.plot(
        t[:t_final],
        V_nominal[:t_final, :, :].squeeze().cpu().numpy(),
        label="Nominal",
        linestyle="solid",
        color="green",
    )
    # Plot markers indicating where the simulations were unsafe
    zeros = np.zeros((T,))
    ax3.plot(
        t[:t_final],
        zeros[:t_final],
        linestyle="dotted",
        color="black",
    )
    ax3.legend()
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$V$")

    ax4 = fig.add_subplot(gs[2, 1:])
    Vdot_actual_sim = torch.diff(V_sim, dim=0) / simulation_dt
    Vdot_actual_nn = torch.diff(V_nn, dim=0) / simulation_dt
    Vdot_actual_nominal = torch.diff(V_nominal, dim=0) / simulation_dt
    ax4.plot(
        t[1:t_final],
        Vdot_sim[1:t_final, :, -1, :].squeeze().cpu().numpy(),
        linestyle="solid",
        color="red",
    )
    ax4.fill_between(
        t[1:t_final],
        Vdot_sim[1:t_final, :, 0, :].squeeze().cpu().numpy(),
        Vdot_sim[1:t_final, :, 1, :].squeeze().cpu().numpy(),
        alpha=0.1,
        color="red",
    )
    ax4.plot(
        t[1:t_final],
        Vdot_actual_sim[1:t_final, :, :].squeeze().cpu().numpy(),
        linestyle="dotted",
        color="red",
    )
    ax4.plot(
        t[1:t_final],
        Vdot_nn[1:t_final, :, -1, :].squeeze().cpu().numpy(),
        linestyle="solid",
        color="blue",
    )
    ax4.fill_between(
        t[1:t_final],
        Vdot_nn[1:t_final, :, 0, :].squeeze().cpu().numpy(),
        Vdot_nn[1:t_final, :, 1, :].squeeze().cpu().numpy(),
        alpha=0.1,
        color="blue",
    )
    ax4.plot(
        t[1:t_final],
        Vdot_actual_nn[1:t_final, :, :].squeeze().cpu().numpy(),
        linestyle="dotted",
        color="blue",
    )
    ax4.plot(
        t[1:t_final],
        Vdot_nominal[1:t_final, :, -1, :].squeeze().cpu().numpy(),
        label="Linearized (true parameters)",
        linestyle="solid",
        color="green",
    )
    ax4.fill_between(
        t[1:t_final],
        Vdot_nominal[1:t_final, :, 0, :].squeeze().cpu().numpy(),
        Vdot_nominal[1:t_final, :, 1, :].squeeze().cpu().numpy(),
        label="Linearized (scenarios)",
        alpha=0.1,
        color="green",
    )
    ax4.plot(
        t[1:t_final],
        Vdot_actual_nominal[1:t_final, :, :].squeeze().cpu().numpy(),
        label="Simulated",
        linestyle="dotted",
        color="green",
    )
    # Plot markers indicating where the simulations were unsafe
    zeros = np.zeros((T,))
    ax4.plot(
        t[:t_final],
        zeros[:t_final],
        linestyle="dotted",
        color="black",
    )
    ax4.legend()
    ax4.set_xlabel("$t$")
    ax4.set_ylabel("$dV/dt$")

    fig.tight_layout()

    # Return the figure along with its name
    return "S-Curve Tracking", fig


if __name__ == "__main__":
    doMain()
