from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import KSCar
from neural_clbf.experiments.train_kinematic_car import (  # noqa
    rollout_plotting_cb,  # noqa
    clbf_plotting_cb,  # noqa
)


checkpoint = "logs/kinematic_car/qp_in_loop/penalty_sched_checkpoint/v8.ckpt"

controller_period = 0.01
simulation_dt = 0.001

# Define the dynamics model
nominal_params = {
    "psi_ref": 0.5,
    "v_ref": 10.0,
    "a_ref": 0.0,
    "omega_ref": 0.0,
}
kscar = KSCar(nominal_params, dt=simulation_dt, controller_dt=controller_period)

# Initialize the DataModule
initial_conditions = [
    (-2.0, 2.0),  # sxe
    (-2.0, 2.0),  # sye
    (-1.0, 1.0),  # delta
    (-2.0, 2.0),  # ve
    (-1.0, 1.0),  # psi_e
]
data_module = EpisodicDataModule(
    kscar,
    initial_conditions,
    trajectories_per_episode=100,
    trajectory_length=500,
    fixed_samples=100000,
    max_points=500000,
    val_split=0.1,
    batch_size=64,
    safe_unsafe_goal_quotas=(0.2, 0.2, 0.2),
)

# Define the scenarios (we need 2^3 = 6)
scenarios = []
omega_ref_vals = [-0.3, 0.3]
for omega_ref in omega_ref_vals:
    s = copy(nominal_params)
    s["omega_ref"] = omega_ref

    scenarios.append(s)


def single_rollout_straight_path():
    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint,
        dynamics_model=kscar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=3,
        clbf_hidden_size=64,
        u_nn_hidden_layers=3,
        u_nn_hidden_size=64,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=10.0,
        penalty_scheduling_rate=25.0,
        epochs_per_episode=5,
    )

    # Test a bunch of hyperparams if you want
    clbf_lambdas = [1.0]

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0
    n_sims = len(clbf_lambdas)
    num_timesteps = int(t_sim // simulation_dt)
    start_x = torch.tensor([[0.0, 1.0, 0.0, 1.0, -np.pi / 6]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    for tstep in tqdm.trange(1, num_timesteps):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            for j in range(n_sims):
                clbf_controller.clbf_lambda = clbf_lambdas[j]
                u = clbf_controller(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                nominal_params,
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
                u = kscar.u_nominal(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_sim[tstep, i, :].unsqueeze(0),
                nominal_params,
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
    psi_ref = nominal_params["psi_ref"]
    x_ref = t * nominal_params["v_ref"] * np.cos(psi_ref)
    y_ref = t * nominal_params["v_ref"] * np.sin(psi_ref)
    x_ref = np.tile(x_ref, (n_sims, 1)).T
    y_ref = np.tile(y_ref, (n_sims, 1)).T

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, kscar.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, kscar.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, kscar.SYE].cpu().numpy()
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
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
            label=f"Tracking Error, lambda={clbf_lambdas[i]}",
        )
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
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

    plt.show()


def single_rollout_circle_path():
    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint,
        dynamics_model=kscar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=3,
        clbf_hidden_size=64,
        u_nn_hidden_layers=3,
        u_nn_hidden_size=64,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=10.0,
        penalty_scheduling_rate=25.0,
        epochs_per_episode=5,
    )

    # Test a bunch of hyperparams if you want
    clbf_lambdas = [0.01]

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 10.0
    n_sims = len(clbf_lambdas)
    num_timesteps = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor([[0.0, 1.0, 0.0, 1.0, -np.pi / 6]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    # And create a place to store the reference path
    params = copy(nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    for tstep in tqdm.trange(1, num_timesteps):
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
                clbf_controller.clbf_lambda = clbf_lambdas[j]
                u = clbf_controller(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
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
                u = kscar.u_nominal(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
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
    x_err_path = x_sim[:, :, kscar.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, kscar.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, kscar.SYE].cpu().numpy()
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
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
            label=f"Tracking Error, lambda={clbf_lambdas[i]}",
        )
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
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

    plt.show()


def single_rollout_s_path():
    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint,
        dynamics_model=kscar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=3,
        clbf_hidden_size=64,
        u_nn_hidden_layers=3,
        u_nn_hidden_size=64,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=2000.0,
        penalty_scheduling_rate=25.0,
        epochs_per_episode=5,
    )

    # Test a bunch of hyperparams if you want
    penalties = [1.0, 10.0, 100.0, 1000.0, 5000.0]

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 5.0
    n_sims = len(penalties)
    num_timesteps = int(t_sim // simulation_dt)
    start_x = 0.0 * torch.tensor([[0.0, 1.0, 0.0, 1.0, -np.pi / 6]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x
        V_sim[0, i, 0] = clbf_controller.V(start_x)

    x_nominal = torch.clone(x_sim)
    V_nominal = torch.clone(V_sim)

    # And create a place to store the reference path
    params = copy(nominal_params)
    params["omega_ref"] = 0.3
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / simulation_dt)
    for tstep in tqdm.trange(1, num_timesteps):
        # Get the path parameters at this point
        omega_ref_t = 0.3 * np.sign(np.sin(tstep * simulation_dt))
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
            for j in range(n_sims):
                clbf_controller.clbf_relaxation_penalty = penalties[j]
                u = clbf_controller(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
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
                u = kscar.u_nominal(x_current[j, :].unsqueeze(0))
                u_sim[tstep, j, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
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
    x_err_path = x_sim[:, :, kscar.SXE].cpu().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)

    x_err_nom = x_nominal[:, :, kscar.SXE].cpu().numpy()
    y_err_nom = x_nominal[:, :, kscar.SYE].cpu().numpy()
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
            x_sim[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
            label=f"Tracking Error, r={penalties[i]}",
        )
    for i in range(n_sims):
        ax3.plot(
            t[:t_final],
            x_nominal[:t_final, i, :].norm(dim=-1).squeeze().numpy(),
            linestyle=":",
            label="Tracking Error (nominal)",
        )
        break
    ax3.plot(
        t[:t_final],
        V_sim[:t_final, :, :].squeeze().numpy(),
        label="V",
    )
    # # Plot markers indicating where the simulations were unsafe
    # zeros = np.zeros((num_timesteps,))
    # ax3.plot(
    #     t[:t_final],
    #     zeros[:t_final],
    # )

    ax3.legend()
    ax3.set_xlabel("$t$")

    plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        # single_rollout_straight_path()
        # single_rollout_circle_path()
        single_rollout_s_path()
