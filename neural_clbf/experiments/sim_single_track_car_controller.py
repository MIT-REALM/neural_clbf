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
    checkpoint = "logs/stcar_basic/v5.ckpt"

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

    # Define the scenarios (we need 2^3 = 6)
    scenarios = []
    omega_ref_vals = [-0.5, 0.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    data_module = EpisodicDataModule(
        stcar,
        initial_conditions,
        trajectories_per_episode=10,
        trajectory_length=1000,
        fixed_samples=10000,
        max_points=5000000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
    )

    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint,
        dynamics_model=stcar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=2,
        clbf_hidden_size=256,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=256,
        clbf_lambda=0.1,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=1e8,
        num_controller_init_epochs=5,
        epochs_per_episode=10,
    )

    # plot_v_vs_tracking_error(clbf_controller)
    # plt.show()

    # single_rollout_straight_path(clbf_controller)
    # plt.show()
    # single_rollout_circle_path(clbf_controller)
    # plt.show()
    single_rollout_s_path(clbf_controller)
    plt.show()


@torch.no_grad()
def plot_v_vs_tracking_error(
    clbf_controller: "NeuralCLBFController",
) -> Tuple[str, plt.figure]:
    # Get the CLBF value at a bunch of points
    x_state = clbf_controller.dynamics_model.sample_state_space(1000)
    x_safe = clbf_controller.dynamics_model.sample_safe(1000)
    x_unsafe = clbf_controller.dynamics_model.sample_unsafe(1000)
    x_goal = clbf_controller.dynamics_model.sample_goal(1000)
    x = torch.cat((x_state, x_safe, x_unsafe, x_goal), dim=0)

    tracking_error = x.norm(dim=-1)
    V = clbf_controller.V(x)

    # Create helpful masks
    correctly_labelled = torch.logical_or(
        torch.logical_and(
            clbf_controller.dynamics_model.safe_mask(x), (V <= 1.0).squeeze()
        ),
        torch.logical_and(
            clbf_controller.dynamics_model.unsafe_mask(x), (V >= 1.0).squeeze()
        ),
    )
    incorrectly_labelled = torch.logical_or(
        torch.logical_and(
            clbf_controller.dynamics_model.safe_mask(x), (V >= 1.0).squeeze()
        ),
        torch.logical_and(
            clbf_controller.dynamics_model.unsafe_mask(x), (V <= 1.0).squeeze()
        ),
    )

    # Plot them
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)

    ax1 = axs
    ax1.scatter(tracking_error, V, color="b")
    ax1.scatter(tracking_error[correctly_labelled], V[correctly_labelled], color="g")
    ax1.scatter(
        tracking_error[incorrectly_labelled], V[incorrectly_labelled], color="r"
    )
    ax1.legend(["?", "Correct", "Incorrect"])

    return "V Scatter", fig


@torch.no_grad()
def single_rollout_straight_path(
    clbf_controller: "NeuralCLBFController",
) -> Tuple[str, plt.figure]:
    # Test a bunch of hyperparams if you want
    penalties = [100, 2e6]

    simulation_dt = clbf_controller.dynamics_model.dt
    controller_period = clbf_controller.controller_period

    # Make sure the controller if for a STCar
    if not isinstance(clbf_controller.dynamics_model, STCar):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0
    n_sims = len(penalties)
    num_timesteps = int(t_sim // simulation_dt)
    start_x = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, -np.pi / 6, 0.0, 0.0]], device=clbf_controller.device
    )
    x_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_dims
    ).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    u_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_controls
    ).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, num_timesteps, desc="Straight Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            for j in range(n_sims):
                clbf_controller.clbf_relaxation_penalty = penalties[j]
                u = clbf_controller(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                clbf_controller.dynamics_model.nominal_params,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_sim[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        # and repeat for the nominal controller
        # Get the current state
        x_current = x_nominal[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            for j in range(n_sims):
                u = clbf_controller.dynamics_model.u_nominal(
                    x_current[j, :].unsqueeze(0)
                )
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                clbf_controller.dynamics_model.nominal_params,
            )
            x_nominal[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_nominal[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        t_final = tstep

    # Plot!
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 6)

    # Get reference path
    t = np.linspace(0, t_sim, num_timesteps)
    psi_ref = clbf_controller.dynamics_model.nominal_params["psi_ref"]
    x_ref = t * clbf_controller.dynamics_model.nominal_params["v_ref"] * np.cos(psi_ref)
    y_ref = t * clbf_controller.dynamics_model.nominal_params["v_ref"] * np.sin(psi_ref)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world_nom = x_ref + x_err_nom * np.cos(psi_ref) - y_err_nom * np.sin(psi_ref)
    y_world_nom = y_ref + x_err_nom * np.sin(psi_ref) + y_err_nom * np.cos(psi_ref)

    ax1 = axs[0]
    ax1.plot([], [], linestyle="-", label="CLBF Tracking")
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
        linestyle="-",
    )
    ax1.plot(
        x_world_nom[:t_final, 0],
        y_world_nom[:t_final, 0],
        linestyle="-.",
        label="Nominal Tracking",
    )
    ax1.plot(
        x_ref[:t_final, 0],
        y_ref[:t_final, 0],
        linestyle=":",
        label="Reference",
    )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_ylim([np.min(y_ref) - 3, np.max(y_ref) + 3])
    ax1.set_xlim([np.min(x_ref) - 3, np.max(x_ref) + 3])
    ax1.set_aspect("equal")

    ax3 = axs[1]
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            label=f"Tracking Error, r={penalties[i]}",
        )
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            linestyle=":",
            label="Tracking Error (nominal)",
        )
        break
    # ax3.plot(
    #     t[:t_final],
    #     V_sim[:t_final, :, :].squeeze().numpy(),
    #     label="V",
    # )
    # # Plot markers indicating where the simulations were unsafe
    # zeros = np.zeros((num_timesteps,))
    # ax3.plot(
    #     t[:t_final],
    #     zeros[:t_final],
    # )

    ax3.legend()
    ax3.set_xlabel("$t$")

    fig.tight_layout()

    # Return the figure along with its name
    return "Straight Line Tracking", fig


@torch.no_grad()
def single_rollout_circle_path(
    clbf_controller: "NeuralCLBFController",
) -> Tuple[str, plt.figure]:
    # Test a bunch of hyperparams if you want
    penalties = [100, 2e6]

    simulation_dt = clbf_controller.dynamics_model.dt
    controller_period = clbf_controller.controller_period

    # Make sure the controller if for a STCar
    if not isinstance(clbf_controller.dynamics_model, STCar):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 10.0
    n_sims = len(penalties)
    num_timesteps = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, -np.pi / 6, 0.0, 0.0]], device=clbf_controller.device
    )
    x_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_dims
    ).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    # And create a place to store the reference path
    params = copy(clbf_controller.dynamics_model.nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    u_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_controls
    ).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, num_timesteps, desc="Circle Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the path parameters at this point
        psi_ref[tstep] = simulation_dt * params["omega_ref"] + psi_ref[tstep - 1]
        pt = copy(params)
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
            for j in range(n_sims):
                clbf_controller.clbf_relaxation_penalty = penalties[j]
                u = clbf_controller(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_sim[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        # and repeat for the nominal controller
        # Get the current state
        x_current = x_nominal[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            for j in range(n_sims):
                u = clbf_controller.dynamics_model.u_nominal(
                    x_current[j, :].unsqueeze(0)
                )
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_nominal[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_nominal[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        t_final = tstep

    # Plot!
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 12)

    # Get reference path
    t = np.linspace(0, t_sim, num_timesteps)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T
    psi_ref = np.tile(psi_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world_nom = x_ref + x_err_nom * np.cos(psi_ref) - y_err_nom * np.sin(psi_ref)
    y_world_nom = y_ref + x_err_nom * np.sin(psi_ref) + y_err_nom * np.cos(psi_ref)

    ax1 = axs[0]
    ax1.plot([], [], linestyle="-", label="CLBF Tracking")
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
        linestyle="-",
    )
    ax1.plot(
        x_world_nom[:t_final, 0],
        y_world_nom[:t_final, 0],
        linestyle="-.",
        label="Nominal Tracking",
    )
    ax1.plot(
        x_ref[:t_final, 0],
        y_ref[:t_final, 0],
        linestyle=":",
        label="Reference",
    )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_ylim([np.min(y_ref) - 3, np.max(y_ref) + 3])
    ax1.set_xlim([np.min(x_ref) - 3, np.max(x_ref) + 3])
    ax1.set_aspect("equal")

    ax3 = axs[1]
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            label=f"Tracking Error, r={penalties[i]}",
        )
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            linestyle=":",
            label="Tracking Error (nominal)",
        )
        break
    # ax3.plot(
    #     t[:t_final],
    #     V_sim[:t_final, :, :].squeeze().numpy(),
    #     label="V",
    # )
    # # Plot markers indicating where the simulations were unsafe
    # zeros = np.zeros((num_timesteps,))
    # ax3.plot(
    #     t[:t_final],
    #     zeros[:t_final],
    # )

    ax3.legend()
    ax3.set_xlabel("$t$")

    fig.tight_layout()

    # Return the figure along with its name
    return "Circle Tracking", fig


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
    t_sim = 5.0
    n_sims = 1
    num_timesteps = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, -np.pi / 6, 0.0, 0.0]], device=clbf_controller.device
    )
    x_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_dims
    ).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    # And create a place to store the reference path
    params = copy(clbf_controller.dynamics_model.nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    u_sim = torch.zeros(
        num_timesteps, n_sims, clbf_controller.dynamics_model.n_controls
    ).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, num_timesteps, desc="S-Curve", leave=True)
    for tstep in prog_bar_range:
        # Get the path parameters at this point
        omega_ref_t = 1.5 * np.sign(np.sin(tstep * simulation_dt))
        psi_ref[tstep] = simulation_dt * omega_ref_t + psi_ref[tstep - 1]
        pt = copy(params)
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
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_sim[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        # and repeat for the nominal controller
        # Get the current state
        x_current = x_nominal[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_controller.dynamics_model.u_nominal(x_current, pt)
            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = clbf_controller.dynamics_model.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                pt,
            )
            x_nominal[tstep, i, :] = x_current[i, :] + simulation_dt * xdot.squeeze()

        # Get the CLBF values
        V_nominal[tstep, :, 0] = clbf_controller.V(x_current).squeeze()

        t_final = tstep

    # Plot!
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 12)

    # Get reference path
    t = np.linspace(0, t_sim, num_timesteps)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T
    psi_ref = np.tile(psi_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, clbf_controller.dynamics_model.SYE].cpu().numpy()
    x_world_nom = x_ref + x_err_nom * np.cos(psi_ref) - y_err_nom * np.sin(psi_ref)
    y_world_nom = y_ref + x_err_nom * np.sin(psi_ref) + y_err_nom * np.cos(psi_ref)

    ax1 = axs[0]
    ax1.plot([], [], linestyle="-", label="CLBF Tracking")
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
        linestyle="-",
    )
    ax1.plot(
        x_world_nom[:t_final, 0],
        y_world_nom[:t_final, 0],
        linestyle="-.",
        label="Nominal Tracking",
    )
    ax1.plot(
        x_ref[:t_final, 0],
        y_ref[:t_final, 0],
        linestyle=":",
        label="Reference",
    )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_ylim([np.min(y_ref) - 3, np.max(y_ref) + 3])
    ax1.set_xlim([np.min(x_ref) - 3, np.max(x_ref) + 3])
    ax1.set_aspect("equal")

    ax2 = axs[1]
    ax2.plot([], [], linestyle="-", label="CLBF-QP")
    ax2.plot([], [], linestyle=":", label="Nominal")
    for i in range(n_sims):
        ax2.plot(
            t[:t_final],
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            linestyle="-",
        )
    for i in range(n_sims):
        ax2.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().cpu().numpy(),
            linestyle=":",
        )
        break

    ax2.legend()
    ax2.set_xlabel("$t$")

    ax3 = axs[2]
    ax3.plot(
        t[:t_final],
        V_sim[:t_final, :, :].squeeze().cpu().numpy(),
        label="V",
    )
    # Plot markers indicating where the simulations were unsafe
    zeros = np.zeros((num_timesteps,))
    ax3.plot(
        t[:t_final],
        zeros[:t_final],
    )

    fig.tight_layout()

    # Return the figure along with its name
    return "S-Curve Tracking", fig


if __name__ == "__main__":
    doMain()
