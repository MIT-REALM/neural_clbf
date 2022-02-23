"""Implement a nonlinear MPC scheme using Casadi"""
import inspect
from math import pi
import os
import sys
import time
from typing import Dict, Any, List

import casadi
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

# Add the parent directory to the path to load the trainer module
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))  # type: ignore
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from dynamics import (  # noqa
    f_turtlebot,
    f_quad6d,
)
from simulation import simulate, generate_random_reference  # noqa


def turtlebot_mpc_casadi_torch(
    x_current: torch.Tensor,
    x_ref: torch.Tensor,
    u_ref: torch.Tensor,
    controller_dt: float,
    control_bounds: List[float],
) -> torch.Tensor:
    """Wrapper for turtlebot_mpc_casadi with torch tensors.

    args:
        x_current: (N_batch, n_state_dims) tensor of current state
        x_ref: (N_batch, planning_horizon, n_state_dims) tensor of reference states
        u_ref: (N_batch, planning_horizon, n_control_dims) tensor of reference controls
        controller_dt: planning timestep
    returns:
        (N_batch, n_control_dims) tensor of control inputs
    """
    N_batch = x_current.shape[0]
    n_control_dims = u_ref.shape[-1]
    control_inputs = torch.zeros((N_batch, n_control_dims)).type_as(x_current)

    for batch_idx in range(N_batch):
        control_inputs[batch_idx] = torch.tensor(
            turtlebot_mpc_casadi(
                x_current[batch_idx].cpu().detach().numpy(),
                x_ref[batch_idx].cpu().detach().numpy(),
                u_ref[batch_idx].cpu().detach().numpy(),
                controller_dt,
                control_bounds,
            )
        )

    return control_inputs


def turtlebot_mpc_casadi(
    x_current: np.ndarray,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    controller_dt: float,
    control_bounds: List[float],
) -> np.ndarray:
    """
    Find a control input by solving a multiple-step direct transcription nonlinear MPC
    problem with turtlebot/dubins car dynamics.

    args:
        x_current: (n_state_dims,) array of current state
        x_ref: (planning_horizon, n_state_dims) array of reference state trajectory
        u_ref: (planning_horizon, n_control_dims) array of reference control trajectory
        controller_dt: planning timestep
    returns:
        (n_control_dims,) array of control inputs
    """
    # Define constants for turtlebot problem
    n_state_dims = 3
    n_control_dims = 2

    # Get length of plan from reference length
    planning_horizon = x_ref.shape[0]

    # Create opt problem and decision variables
    opti = casadi.Opti()
    x = opti.variable(planning_horizon + 1, n_state_dims)  # state (x, z, theta)
    u = opti.variable(planning_horizon, n_control_dims)  # control (v, omega)

    # Simple objective from LQR
    error_penalty = 10
    x_tracking_error = x[1:, :] - x_ref
    opti.minimize(error_penalty * casadi.sumsqr(x_tracking_error))

    # Set initial conditions
    opti.subject_to(x[0, 0] == x_current[0])
    opti.subject_to(x[0, 1] == x_current[1])
    opti.subject_to(x[0, 2] == x_current[2])

    # Set control bounds
    for control_idx, bound in enumerate(control_bounds):
        for t in range(planning_horizon):
            opti.subject_to(u[t, control_idx] <= bound)
            opti.subject_to(u[t, control_idx] >= -bound)

    # Impose dynamics constraints via direct transcription
    for t in range(planning_horizon):
        # Extract states and controls
        px_next = x[t + 1, 0]
        py_next = x[t + 1, 1]
        theta_next = x[t + 1, 2]
        px_now = x[t, 0]
        py_now = x[t, 1]
        theta_now = x[t, 2]
        v = u[t, 0]
        omega = u[t, 1]

        # These dynamics are smooth enough that we probably can get away with a simple
        # forward Euler integration.

        # x_dot = v * cos(theta)
        opti.subject_to(px_next == px_now + v * casadi.cos(theta_now) * controller_dt)
        # y_dot = v * sin(theta)
        opti.subject_to(py_next == py_now + v * casadi.sin(theta_now) * controller_dt)
        # theta_dot = omega
        opti.subject_to(theta_next == theta_now + omega * controller_dt)

    # Set an initial guess based on the reference trajectory
    x_initial = np.vstack((x_current.reshape(1, n_state_dims), x_ref))
    opti.set_initial(x, x_initial)
    opti.set_initial(u, u_ref)

    # Optimizer setting
    p_opts: Dict[str, Any] = {"expand": True}
    s_opts: Dict[str, Any] = {"max_iter": 1000}
    quiet = True
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"

    # Solve!
    opti.solver("ipopt", p_opts, s_opts)
    sol1 = opti.solve()

    # Return the first control input
    return sol1.value(u[0, :])


def simulate_and_plot_turtle():
    # Define the dynamics
    n_state_dims = 3
    n_control_dims = 2
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]
    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    # Define the timestep and planning horizon for MPC
    controller_dt = 0.1
    controller_horizon_s = 1

    # Measure MPC control frequency
    mpc_seconds = 0.0
    mpc_calls = 0

    # Make a bunch of plot
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    axs = [ax for row in axs for ax in row]

    for ax in axs:
        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 7.0 + controller_horizon_s  # length of trajectory
        dt = 1e-2  # timestep
        x_init, x_ref, u_ref = generate_random_reference(
            N_batch,
            T,
            dt,
            n_state_dims,
            n_control_dims,
            state_space,
            control_bounds,
            error_bounds,
            f_turtlebot,
        )
        # Convert to numpy
        x_init = x_init.cpu().numpy().squeeze()
        x_ref = x_ref.cpu().numpy().squeeze()
        u_ref = u_ref.cpu().numpy().squeeze()
        t = np.arange(0, T, dt)
        N_steps = t.shape[0]

        # Make sure we don't overrun the end of the reference while planning
        N_steps -= int(controller_horizon_s / dt)

        # Create some places to store the simulation results
        x_sim = np.zeros((N_steps, n_state_dims))
        x_sim[0, :] = x_init
        u_sim = np.zeros((N_steps, n_control_dims))
        u_current = np.zeros((n_control_dims,))

        # Simulate using the MPC controller function
        sim_range = tqdm(range(N_steps - 1))
        sim_range.set_description("Simulating")  # type: ignore
        controller_update_freq = int(controller_dt / dt)
        for tstep in sim_range:
            # Get the current states
            x_current = x_sim[tstep].reshape(n_state_dims)

            # Downsample reference for use with MPC
            x_ref_horizon = x_ref[tstep : tstep + int(controller_horizon_s // dt)]
            u_ref_horizon = u_ref[tstep : tstep + int(controller_horizon_s // dt)]
            full_samples = t[tstep : tstep + int(controller_horizon_s // dt)]
            mpc_samples = np.arange(full_samples[0], full_samples[-1], controller_dt)
            x_ref_mpc = interp1d(full_samples, x_ref_horizon, axis=0)(mpc_samples)
            u_ref_mpc = interp1d(full_samples, u_ref_horizon, axis=0)(mpc_samples)

            # Run MPC
            if tstep % controller_update_freq == 0:
                start_time = time.perf_counter()
                u_current = turtlebot_mpc_casadi(
                    x_current,
                    x_ref_mpc,
                    u_ref_mpc,
                    controller_dt,
                    control_bounds,
                )
                end_time = time.perf_counter()
                mpc_seconds += end_time - start_time
                mpc_calls += 1
            u_sim[tstep + 1, :] = u_current

            # Get the derivatives and update the state
            x_dot = (
                f_turtlebot(
                    torch.tensor(x_current).unsqueeze(0),
                    torch.tensor(u_current).unsqueeze(0),
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            x_sim[tstep + 1, :] = x_current + dt * x_dot

        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref[:N_steps, 0],
            x_ref[:N_steps, 1],
            linestyle=":",
        )
        ax.plot(
            x_ref[0, 0],
            x_ref[0, 1],
            marker="o",
            color="k",
        )
        ax.set_prop_cycle(None)  # Re-use colors for the reference
        ax.plot([], [], linestyle="-", color="k", label="Actual")
        ax.plot(
            x_sim[:, 0],
            x_sim[:, 1],
            linestyle="-",
        )
        ax.plot(
            x_sim[0, 0],
            x_sim[0, 1],
            marker="o",
            color="k",
        )
        ax.legend()

    print(f"MPC control period is {mpc_seconds / mpc_calls}")
    print(f"({mpc_seconds} s over {mpc_calls} calls)")
    plt.show()


if __name__ == "__main__":
    simulate_and_plot_turtle()
