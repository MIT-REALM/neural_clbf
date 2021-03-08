"""Functions for plotting experimental results"""
from typing import List, Tuple, Optional, TYPE_CHECKING
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# We only need these imports if type checking
if TYPE_CHECKING:
    from neural_clbf.controllers import NeuralrCLBFController
    from neural_clbf.systems.utils import ScenarioList


# Beautify plots
sns.set_theme(context="talk", style="white")
sim_color = sns.color_palette("pastel")[1]


@torch.no_grad()
def plot_CLBF(
    clbf_net: 'NeuralrCLBFController',
    domain: Optional[List[Tuple[float, float]]] = None,
    n_grid: int = 50,
    x_axis_index: int = 0,
    y_axis_index: int = 1,
    x_axis_label: str = "$x$",
    y_axis_label: str = "$y$",
    default_state: Optional[torch.Tensor] = None,
):
    """Plot the value of the CLBF, V, and dV/dt.

    Instead of actually plotting dV/dt, we plot relu(dV/dt + clbf_net.clbf_lambda * V)
    to highlight the violation region

    args:
        clbf_net: the CLBF network
        n_grid: the number of points in each direction at which to compute V
        x_axis_index: the index of the state variable to plot on the x axis
        y_axis_index: the index of the state variable to plot on the y axis
        x_axis_label: the label for the x axis
        y_axis_label: the label for the y axis
        default_state: 1 x clbf_net.dynamics_model.n_dims tensor of default state
                       values. The values at x_axis_index and y_axis_index will be
                       overwritten by the grid values.
    returns:
        a matplotlib.pyplot.figure containing the plots of V and dV/dt
    """
    # Deal with optional parameters
    # Default to plotting over [-1, 1] in all directions
    if domain is None:
        domain = [(-1.0, 1.0), (-1.0, 1.0)]
    # Set up the plotting grid
    x_vals = torch.linspace(domain[0][0], domain[0][1], n_grid)
    y_vals = torch.linspace(domain[1][0], domain[1][1], n_grid)
    grid_x, grid_y = torch.meshgrid(x_vals, y_vals)

    # Set up tensors to store the results
    V_grid = torch.zeros(n_grid, n_grid)
    V_dot_grid = torch.zeros(n_grid, n_grid)

    # If the default state is not provided, use zeros
    if (
        default_state is None
        or default_state.nelement() != clbf_net.dynamics_model.n_dims
    ):
        default_state = torch.zeros(1, clbf_net.dynamics_model.n_dims)

    # Make a copy of the default state, which we'll modify on every loop
    x = torch.tensor(default_state).reshape(1, clbf_net.dynamics_model.n_dims)
    for i in range(n_grid):
        for j in range(n_grid):
            # Adjust x to be at the current grid point
            x[0, x_axis_index] = x_vals[i]
            x[0, y_axis_index] = y_vals[j]

            # Get the value of the CLBF
            V_grid[j, i] = clbf_net.V(x)

            # Get the derivative from the Lie derivatives and controller
            Lf_V, Lg_V = clbf_net.V_lie_derivatives(x)
            # Try to get the control input; if it fails, default to zero
            try:
                u = clbf_net(x)
            except (Exception):
                u = torch.zeros(1, clbf_net.dynamics_model.n_controls)
            # Accumulate violation across all scenarios
            for i in range(clbf_net.n_scenarios):
                Vdot = Lf_V[:, i, :] + torch.bmm(Lg_V[:, i, :], u)
                V_dot_grid[j, i] += F.relu(Vdot + clbf_net.clbf_lambda * V_grid[j, i])

    # Make the plots
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8, 8)

    # First for V
    contours = axs[0].contourf(x_vals, y_vals, V_grid, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0], orientation="horizontal")
    contours = axs[0].contour(
        x_vals, y_vals, V_grid, colors=["blue"], levels=[clbf_net.clbf_safety_level]
    )
    axs[0].set_xlabel(x_axis_label)
    axs[0].set_ylabel(y_axis_label)
    axs[0].set_title("$V$")
    axs[0].plot([], [], c="blue", label="V(x) = c")
    axs[0].legend()

    # Then for dV/dt
    contours = axs[1].contourf(x_vals, y_vals, V_dot_grid, cmap="Greys", levels=20)

    def fmt(x, pos):
        a, b = "{:.2e}".format(x).split("e")
        b = int(b)
        return r"${} \times 10^{{{}}}$".format(a, b)

    cbar = plt.colorbar(
        contours, ax=axs[1], orientation="horizontal", format=ticker.FuncFormatter(fmt)
    )
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    axs[1].set_xlabel(x_axis_label)
    axs[1].set_ylabel(y_axis_label)
    axs[1].set_title("$[dV/dt + \\lambda V]_+$")

    fig.tight_layout()

    return fig


@torch.no_grad()
def rollout_CLBF(
    clbf_net: 'NeuralrCLBFController',
    scenarios: 'ScenarioList',
    start_x: Optional[torch.Tensor] = None,
    plot_x_indices: Optional[List[int]] = None,
    plot_x_labels: Optional[List[str]] = None,
    plot_u_indices: Optional[List[int]] = None,
    plot_u_labels: Optional[List[str]] = None,
    n_sims_per_start: int = 5,
    t_sim: float = 10.0,
    delta_t: float = 0.001,
):
    """Simulate the performance of the controller over time

    args:
        clbf_net: the CLBF network
        start_x: n x clbf_net.dynamics_model.n_dims tensor of starting states
        scenarios: a list of parameter scenarios to sample from.
        plot_x_indices: a list of the indices of the state variables to plot
        plot_x_labels: a list of the labels for each state variable trace
        plot_indices: a list of the indices of the control inputs to plot
        plot_labels: a list of the labels for each control trace
        n_sims_per_start: the number of simulations to run (with random parameters),
                          per row in start_x
        t_sim: the amount of time to simulate for
        delta_t: the simulation timestep
    returns:
        a matplotlib.pyplot.figure containing the plots of state and control input
        over time.
    """
    # Deal with optional parameters
    # Default to starting from all state variables = 1.0
    if start_x is None:
        start_x = torch.ones(1, clbf_net.dynamics_model.n_dims)
    # Default to just plotting the first state variable
    if plot_x_indices is None:
        plot_x_indices = [0]
    if plot_x_labels is None:
        plot_x_labels = ["x0"]
    assert len(plot_x_labels) == len(plot_x_indices)
    # Default to just plotting the first control variable
    if plot_u_indices is None:
        plot_u_indices = [0]
    if plot_u_labels is None:
        plot_u_labels = ["x0"]
    assert len(plot_u_labels) == len(plot_u_indices)

    # Compute the number of simulations to run
    n_sims = n_sims_per_start * start_x.shape[0]

    # Determine the parameter range to sample from
    parameter_ranges = {}
    for param_name in scenarios[0].keys():
        param_max = max([s[param_name] for s in scenarios])
        param_min = min([s[param_name] for s in scenarios])
        parameter_ranges[param_name] = (param_min, param_max)

    # Generate a tensor of start states and corresponding scenarios
    x_sim_start = torch.zeros(n_sims, clbf_net.dynamics_model.n_dims)
    random_scenarios = []
    for i in range(0, n_sims, n_sims_per_start):
        x_sim_start[i : i + n_sims_per_start, :] = start_x

        # Generate a random scenario from the given scenarios
        random_scenario = {}
        for param_name in scenarios[0].keys():
            param_min = parameter_ranges[param_name][0]
            param_max = parameter_ranges[param_name][1]
            random_scenario[param_name] = random.uniform(param_min, param_max)
        random_scenarios.append(random_scenario)

    # Simulate!
    # (but first make somewhere to save the results)
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, n_sims, clbf_net.dynamics_model.n_dims)
    u_sim = torch.zeros(num_timesteps, n_sims, clbf_net.dynamics_model.n_controls)
    t_final = 0
    controller_failed = False
    try:
        for tstep in range(1, num_timesteps):
            # Get the current state
            x_current = x_sim[tstep - 1, :, :]
            # Get the control input at the current state
            u = clbf_net(x_current)
            u_sim[tstep, :, :] = u

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = clbf_net.dynamics_model.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0), u, random_scenarios[i]
                )
                x_sim[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final = tstep
    except (Exception):
        raise
        controller_failed = True

    fig, axs = plt.subplots(2, 1)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    for i_trace in range(len(plot_x_indices)):
        ax1.plot(
            t[:t_final],
            x_sim[:t_final, :, plot_x_indices[i_trace]],
            label=plot_x_labels[i_trace],
        )

    # If the controller fails, mark that on the plot
    if controller_failed:
        ax1.title("Controller failure!")

    ax1.set_xlabel("$t$")
    ax1.legend()

    ax2 = axs[1, 0]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[:t_final],
            u_sim[:t_final, :, plot_u_indices[i_trace]],
            label=plot_u_labels[i_trace],
        )

    ax2.set_xlabel("$t$")
    ax2.legend()
