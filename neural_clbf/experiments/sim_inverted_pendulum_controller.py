from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import InvertedPendulum

if __name__ == "__main__":
    # Import the plotting callbacks, which seem to be needed to load from the checkpoint
    from neural_clbf.experiments.train_inverted_pendulum import (  # noqa
        rollout_plotting_cb,  # noqa
        clbf_plotting_cb,  # noqa
    )

# Set up indices for convenience
THETA = InvertedPendulum.THETA
THETA_DOT = InvertedPendulum.THETA_DOT
U = InvertedPendulum.U


def doMain():
    checkpoint = "logs/inverted_pendulum/v1.ckpt"

    controller_period = 0.01
    simulation_dt = 0.001
    clbf_lambda = 0.5

    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    inverted_pendulum = InvertedPendulum(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-np.pi / 4, np.pi / 4),  # theta
        (-1.0, 1.0),  # theta_dot
    ]
    # Define the scenarios (we need 2^3 = 6)
    scenarios = [
        nominal_params,
        {"m": 1.25, "L": 1.0, "b": 0.01},
        {"m": 1.0, "L": 1.25, "b": 0.01},
        {"m": 1.25, "L": 1.25, "b": 0.01},
    ]

    data_module = EpisodicDataModule(
        inverted_pendulum,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=500,
        fixed_samples=100000,
        max_points=500000,
        val_split=0.1,
        batch_size=64,
    )

    clbf_net = NeuralCLBFController.load_from_checkpoint(
        checkpoint,
        dynamics_model=inverted_pendulum,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=2,
        clbf_hidden_size=256,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=256,
        clbf_lambda=clbf_lambda,
        controller_period=controller_period,
        lookahead=controller_period,
        clbf_relaxation_penalty=1e5,
        # clbf_relaxation_penalty=2e6,
        epochs_per_episode=10,
    )

    single_rollout_stabilization(clbf_net)
    plt.show()


@torch.no_grad()
def single_rollout_stabilization(
    clbf_net: "NeuralCLBFController",
) -> Tuple[str, plt.figure]:
    simulation_dt = clbf_net.dynamics_model.dt
    controller_period = clbf_net.controller_period

    # Make sure the controller if for an inverted pendulum
    if not isinstance(clbf_net.dynamics_model, InvertedPendulum):
        raise ValueError()

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 10.0
    n_sims_per_start = 1
    T = int(t_sim // simulation_dt)
    start_x = torch.tensor(
        [
            [0.2, 1.0],
            [-0.2, 1.0],
            [0.2, -1.0],
            [-0.2, -1.0],
        ],
        device=clbf_net.device,
    )
    n_sims = n_sims_per_start * start_x.shape[0]
    x_sim = torch.zeros(T, n_sims, clbf_net.dynamics_model.n_dims).type_as(start_x)
    V_sim = torch.zeros(T, n_sims, 1).type_as(start_x)
    u_sim = torch.zeros(T, n_sims, clbf_net.dynamics_model.n_controls).type_as(start_x)
    for i in range(start_x.shape[0]):
        for j in range(n_sims_per_start):
            x_sim[0, i * n_sims_per_start + j, :] = start_x[i]
            V_sim[0, i * n_sims_per_start + j, 0] = clbf_net.V(start_x[i].unsqueeze(0))

    # Also create somewhere to save the simulations from the learned controller...
    x_nn = torch.clone(x_sim)
    V_nn = torch.clone(V_sim)
    u_nn = torch.zeros(T, n_sims, clbf_net.dynamics_model.n_controls).type_as(start_x)

    # And the nominal controller
    x_nom = torch.clone(x_sim)
    V_nom = torch.clone(V_sim)
    u_nom = torch.zeros(T, n_sims, clbf_net.dynamics_model.n_controls).type_as(start_x)

    controller_update_freq = int(controller_period / simulation_dt)
    prog_bar_range = tqdm.trange(1, T, desc="Stabilization", leave=True)
    for tstep in prog_bar_range:
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]

        # Simulate out the CLBF QP controller
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_net(x_current)
        else:
            u = u_sim[tstep - 1, :, :]
        u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        xdot = clbf_net.dynamics_model.closed_loop_dynamics(
            x_current,
            u_sim[tstep, :, :],
            clbf_net.dynamics_model.nominal_params,
        )
        x_sim[tstep, :, :] = x_current + simulation_dt * xdot

        # Get the CLBF values
        V_sim[tstep, :, 0] = clbf_net.V(x_current).squeeze()

        # and repeat for the neural net controller
        # Get the current state
        x_current = x_nn[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_net.u(x_current)
        else:
            u = u_nn[tstep - 1, :, :]
        u_nn[tstep, :, :] = u

        # Simulate forward using the dynamics
        xdot = clbf_net.dynamics_model.closed_loop_dynamics(
            x_current,
            u_nn[tstep, :, :],
            clbf_net.dynamics_model.nominal_params,
        )
        x_nn[tstep, :, :] = x_current + simulation_dt * xdot

        # Get the CLBF values
        V_nn[tstep, :, 0] = clbf_net.V(x_current).squeeze()

        # and repeat for the nominal controller
        # Get the current state
        x_current = x_nom[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = clbf_net.dynamics_model.u_nominal(x_current)
        else:
            u = u_nom[tstep - 1, :, :]
        u_nom[tstep, :, :] = u

        # Simulate forward using the dynamics
        xdot = clbf_net.dynamics_model.closed_loop_dynamics(
            x_current,
            u_nom[tstep, :, :],
            clbf_net.dynamics_model.nominal_params,
        )
        x_nom[tstep, :, :] = x_current + simulation_dt * xdot

        # Get the CLBF values
        V_nom[tstep, :, 0] = clbf_net.V(x_current).squeeze()

        t_final = tstep

    # Plot!
    fig, axs = plt.subplots(4, 3)
    fig.set_size_inches(12, 10)

    # Plot trajectories in the phase plane for the three controllers
    ax_clbf_phase = axs[0, 0]
    ax_clbf_phase.set_title("CLBF")
    ax_clbf_phase.plot(
        x_sim[:t_final, :, THETA],
        x_sim[:t_final, :, THETA_DOT],
    )
    ax_clbf_phase.set_xlabel("$\\theta$")
    ax_clbf_phase.set_ylabel("$\\dot{\\theta}$")
    ax_clbf_phase.set_xlim([-np.pi / 2, np.pi / 2])
    ax_clbf_phase.set_ylim([-np.pi / 2, np.pi / 2])
    ax_clbf_phase.plot([-np.pi / 2, np.pi / 2], [0, 0], color="k", linewidth=0.5)
    ax_clbf_phase.plot([0, 0], [-np.pi / 2, np.pi / 2], color="k", linewidth=0.5)
    goal_region = plt.Circle((0, 0), 0.3, color="k", fill=False, linewidth=0.5)
    ax_clbf_phase.add_patch(goal_region)
    ax_clbf_phase.set_aspect("equal")

    ax_nn_phase = axs[0, 1]
    ax_nn_phase.set_title("NN")
    ax_nn_phase.plot(
        x_nn[:t_final, :, THETA],
        x_nn[:t_final, :, THETA_DOT],
    )
    ax_nn_phase.set_xlabel("$\\theta$")
    ax_nn_phase.set_ylabel("$\\dot{\\theta}$")
    ax_nn_phase.set_xlim([-np.pi / 2, np.pi / 2])
    ax_nn_phase.set_ylim([-np.pi / 2, np.pi / 2])
    ax_nn_phase.plot([-np.pi / 2, np.pi / 2], [0, 0], color="k", linewidth=0.5)
    ax_nn_phase.plot([0, 0], [-np.pi / 2, np.pi / 2], color="k", linewidth=0.5)
    goal_region = plt.Circle((0, 0), 0.3, color="k", fill=False, linewidth=0.5)
    ax_nn_phase.add_patch(goal_region)
    ax_nn_phase.set_aspect("equal")

    ax_nom_phase = axs[0, 2]
    ax_nom_phase.set_title("Nominal")
    ax_nom_phase.plot(
        x_nom[:t_final, :, THETA],
        x_nom[:t_final, :, THETA_DOT],
    )
    ax_nom_phase.set_xlabel("$\\theta$")
    ax_nom_phase.set_ylabel("$\\dot{\\theta}$")
    ax_nom_phase.set_xlim([-np.pi / 2, np.pi / 2])
    ax_nom_phase.set_ylim([-np.pi / 2, np.pi / 2])
    ax_nom_phase.plot([-np.pi / 2, np.pi / 2], [0, 0], color="k", linewidth=0.5)
    ax_nom_phase.plot([0, 0], [-np.pi / 2, np.pi / 2], color="k", linewidth=0.5)
    goal_region = plt.Circle((0, 0), 0.3, color="k", fill=False, linewidth=0.5)
    ax_nom_phase.add_patch(goal_region)
    ax_nom_phase.set_aspect("equal")

    # Also plot in the time domain
    t = np.linspace(0, T * simulation_dt, t_final)
    ax_clbf_time = axs[1, 0]
    ax_clbf_time.plot([], [], linestyle="-", label="$\\theta$")
    ax_clbf_time.plot([], [], linestyle=":", label="$\\dot{\\theta}$")
    ax_clbf_time.plot(t, x_sim[:t_final, :, THETA], linestyle="-")
    ax_clbf_time.plot(t, x_sim[:t_final, :, THETA_DOT], linestyle=":")
    ax_clbf_time.set_xlabel("$t$")
    ax_clbf_time.set_ylim([-np.pi / 2, np.pi / 2])
    ax_clbf_time.legend()

    ax_nn_time = axs[1, 1]
    ax_nn_time.plot([], [], linestyle="-", label="$\\theta$")
    ax_nn_time.plot([], [], linestyle=":", label="$\\dot{\\theta}$")
    ax_nn_time.plot(t, x_nn[:t_final, :, THETA], linestyle="-")
    ax_nn_time.plot(t, x_nn[:t_final, :, THETA_DOT], linestyle=":")
    ax_nn_time.set_xlabel("$t$")
    ax_nn_time.set_ylim([-np.pi / 2, np.pi / 2])
    ax_nn_time.legend()

    ax_nom_time = axs[1, 2]
    ax_nom_time.plot([], [], linestyle="-", label="$\\theta$")
    ax_nom_time.plot([], [], linestyle=":", label="$\\dot{\\theta}$")
    ax_nom_time.plot(t, x_nom[:t_final, :, THETA], linestyle="-")
    ax_nom_time.plot(t, x_nom[:t_final, :, THETA_DOT], linestyle=":")
    ax_nom_time.set_xlabel("$t$")
    ax_nom_time.set_ylim([-np.pi / 2, np.pi / 2])
    ax_nom_time.legend()

    # Also plot control inputs
    ax_clbf_u_time = axs[2, 0]
    ax_clbf_u_time.plot(t, u_sim[:t_final, :, U])
    ax_clbf_u_time.set_ylabel("$u$")
    ax_clbf_u_time.set_xlabel("$t$")
    ax_clbf_u_time.set_ylim([-11, 11])

    ax_nn_u_time = axs[2, 1]
    ax_nn_u_time.plot(t, u_nn[:t_final, :, U])
    ax_nn_u_time.set_ylabel("$u$")
    ax_nn_u_time.set_xlabel("$t$")
    ax_nn_u_time.set_ylim([-11, 11])

    ax_nom_u_time = axs[2, 2]
    ax_nom_u_time.plot(t, u_nom[:t_final, :, U])
    ax_nom_u_time.set_ylabel("$u$")
    ax_nom_u_time.set_xlabel("$t$")
    ax_nom_u_time.set_ylim([-11, 11])

    # Also plot CLBF values
    ax_clbf_V_time = axs[3, 0]
    ax_clbf_V_time.plot(t, V_sim[:t_final, :, 0])
    ax_clbf_V_time.set_ylabel("$V$")
    ax_clbf_V_time.set_xlabel("$t$")
    ax_clbf_V_time.set_ylim([-1, 5])
    ax_clbf_V_time.plot([0, t[-1]], [0, 0], color="k", linewidth=0.5)
    ax_clbf_V_time.plot([0, t[-1]], [clbf_net.safe_level] * 2, color="k", linewidth=0.5)

    ax_nn_V_time = axs[3, 1]
    ax_nn_V_time.plot(t, V_nn[:t_final, :, 0])
    ax_nn_V_time.set_ylabel("$V$")
    ax_nn_V_time.set_xlabel("$t$")
    ax_nn_V_time.set_ylim([-1, 5])
    ax_nn_V_time.plot([0, t[-1]], [0, 0], color="k", linewidth=0.5)
    ax_nn_V_time.plot([0, t[-1]], [clbf_net.safe_level] * 2, color="k", linewidth=0.5)

    ax_nom_V_time = axs[3, 2]
    ax_nom_V_time.plot(t, V_nom[:t_final, :, 0])
    ax_nom_V_time.set_ylabel("$V$")
    ax_nom_V_time.set_xlabel("$t$")
    ax_nom_V_time.set_ylim([-1, 5])
    ax_nom_V_time.plot([0, t[-1]], [0, 0], color="k", linewidth=0.5)
    ax_nom_V_time.plot([0, t[-1]], [clbf_net.safe_level] * 2, color="k", linewidth=0.5)

    fig.tight_layout()

    # Return the figure along with its name
    return "Stabilization", fig


if __name__ == "__main__":
    doMain()
