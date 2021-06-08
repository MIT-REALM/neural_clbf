"""Functions for plotting experimental results"""
from typing import Callable, List, Tuple, Optional, TYPE_CHECKING
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker
import seaborn as sns
import tqdm

from neural_clbf.systems.utils import ScenarioList

# We only need these imports if type checking, to avoid circular imports
if TYPE_CHECKING:
    from neural_clbf.controllers import Controller


# Beautify plots
sns.set_theme(context="talk", style="white")
sim_color = sns.color_palette("pastel")[1]


@torch.no_grad()
def plot_CLBF(
    clbf_net: "Controller",
    domain: Optional[List[Tuple[float, float]]] = None,
    n_grid: int = 50,
    x_axis_index: int = 0,
    y_axis_index: int = 1,
    x_axis_label: str = "$x$",
    y_axis_label: str = "$y$",
    default_state: Optional[torch.Tensor] = None,
) -> Tuple[str, plt.figure]:
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
    x_vals = torch.linspace(domain[0][0], domain[0][1], n_grid, device=clbf_net.device)
    y_vals = torch.linspace(domain[1][0], domain[1][1], n_grid, device=clbf_net.device)

    # Set up tensors to store the results
    V_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    relax_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    lin_descent_loss_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    sim_descent_loss_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    safe_loss_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    unsafe_loss_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    lower_bound_loss_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    # V_dot_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    unsafe_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    safe_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    goal_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)

    # If the default state is not provided, use zeros
    if (
        default_state is None
        or default_state.nelement() != clbf_net.dynamics_model.n_dims
    ):
        default_state = torch.zeros(1, clbf_net.dynamics_model.n_dims)

    default_state = default_state.type_as(x_vals)

    # Make a copy of the default state, which we'll modify on every loop
    x = default_state.clone().detach().reshape(1, clbf_net.dynamics_model.n_dims)
    prog_bar_range = tqdm.trange(n_grid, desc="Plotting CLBF", leave=True)
    print("Plotting CLBF on grid...")
    for i in prog_bar_range:
        for j in range(n_grid):
            # Adjust x to be at the current grid point
            x[0, x_axis_index] = x_vals[i]
            x[0, y_axis_index] = y_vals[j]

            # Get the value of the CLBF
            V_grid[j, i] = clbf_net.V(x)

            # And get the losses for this point
            goal_mask = clbf_net.dynamics_model.goal_mask(x)
            safe_mask = clbf_net.dynamics_model.safe_mask(x)
            unsafe_mask = clbf_net.dynamics_model.unsafe_mask(x)
            dist_to_goal = clbf_net.dynamics_model.distance_to_goal(x)
            descent_losses = clbf_net.descent_loss(  # type: ignore
                x, goal_mask, safe_mask, unsafe_mask, dist_to_goal
            )
            boundary_losses = clbf_net.boundary_loss(  # type: ignore
                x, goal_mask, safe_mask, unsafe_mask, dist_to_goal
            )
            lin_descent_loss_grid[j, i] = descent_losses[0][1]
            sim_descent_loss_grid[j, i] = descent_losses[1][1]
            # safe_loss_grid[j, i] = boundary_losses[1][1]
            # unsafe_loss_grid[j, i] = boundary_losses[2][1]
            lower_bound_loss_grid[j, i] = boundary_losses[1][1]

            # Get the QP relaxation
            _, r, _ = clbf_net.solve_CLBF_QP(x)  # type: ignore
            relax_grid[j, i] = r.max()

            # Get the goal, safe, or unsafe classification
            if clbf_net.dynamics_model.goal_mask(x).all():
                goal_grid[j, i] = 1
            elif clbf_net.dynamics_model.safe_mask(x).all():
                safe_grid[j, i] = 1
            elif clbf_net.dynamics_model.unsafe_mask(x).all():
                unsafe_grid[j, i] = 1

            # # Try to get the control input; if it fails, default to zero
            # try:
            #     u = clbf_net(x)
            # except (Exception):
            #     u = torch.zeros(1, clbf_net.dynamics_model.n_controls).type_as(x_vals)
            # V_dot_grid[j, i] = clbf_net.V_decrease_violation(x)

    # Make the plots
    fig, axes = plt.subplots(4, 2)
    # fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(18, 23)

    # First plot V
    axs = axes[0, 0]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), V_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("CLBF")
    # Plot safe levels
    safe_level = clbf_net.safe_level
    if isinstance(safe_level, torch.Tensor):
        safe_level = safe_level.item()
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        unsafe_grid.cpu(),
        colors=["green"],
        levels=[0.5],  # type: ignore
    )
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        safe_grid.cpu(),
        colors=["green"],
        levels=[0.5],  # type: ignore
    )
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        goal_grid.cpu(),
        colors=["green"],
        levels=[0.5],  # type: ignore
    )
    # And unsafe levels
    unsafe_level = clbf_net.unsafe_level
    if isinstance(unsafe_level, torch.Tensor):
        unsafe_level = unsafe_level.item()
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        V_grid.cpu(),
        colors=["blue"],
        levels=[unsafe_level],  # type: ignore
    )
    # And goal levels
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        V_grid.cpu(),
        colors=["white"],
        levels=[0.0],
    )
    axs.set_xlabel(x_axis_label)
    axs.set_ylabel(y_axis_label)
    axs.set_title("$V$")
    axs.plot([], [], c="green", label="Unsafe set")  # type: ignore
    axs.plot(
        [],
        [],
        c="blue",
        label=f"Unsafe V={unsafe_level}",  # type: ignore
    )
    axs.plot([], [], c="white", label="Goal V=0")
    axs.legend()

    # Also plot the QP relaxation
    axs = axes[0, 1]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), relax_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Max r")

    # Then plot the losses
    axs = axes[1, 0]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), lin_descent_loss_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Lin Descent Loss")

    axs = axes[1, 1]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), sim_descent_loss_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Sim Descent Loss")

    axs = axes[2, 0]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), safe_loss_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Safe loss")

    axs = axes[2, 1]
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), unsafe_loss_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Unsafe loss")

    axs = axes[3, 0]
    contours = axs.contourf(
        x_vals.cpu(),
        y_vals.cpu(),
        lower_bound_loss_grid.cpu(),
        cmap="magma",
        levels=20,
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    plt.title("Lower bound loss")

    # # Then for dV/dt
    # contours = axs[1].contourf(
    #     x_vals.cpu(), y_vals.cpu(), V_dot_grid.cpu(), cmap="Greys", levels=20
    # )

    # def fmt(x, pos):
    #     a, b = "{:.2e}".format(x).split("e")
    #     b = int(b)
    #     return r"${} \times 10^{{{}}}$".format(a, b)

    # cbar = plt.colorbar(
    #     contours,
    #     ax=axs[1],
    #     orientation="horizontal",
    #     format=ticker.FuncFormatter(fmt)
    # )
    # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    # axs[1].set_xlabel(x_axis_label)
    # axs[1].set_ylabel(y_axis_label)
    # axs[1].set_title("$[dV/dt + \\lambda V]_+$")

    fig.tight_layout()

    # Return the figure along with its name
    return "CLBF Plot", fig


@torch.no_grad()
def rollout_CLBF(
    clbf_net: "Controller",
    scenarios: Optional[ScenarioList] = None,
    start_x: Optional[torch.Tensor] = None,
    plot_x_indices: Optional[List[int]] = None,
    plot_x_labels: Optional[List[str]] = None,
    plot_u_indices: Optional[List[int]] = None,
    plot_u_labels: Optional[List[str]] = None,
    n_sims_per_start: int = 5,
    t_sim: float = 5.0,
    controller_period: float = 0.01,
    goal_check_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    out_of_bounds_check_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[str, plt.figure]:
    """Simulate the performance of the controller over time

    args:
        clbf_net: the CLBF network
        start_x: n x clbf_net.dynamics_model.n_dims tensor of starting states
        scenarios: a list of parameter scenarios to sample from. If None, defaults to
                   the clbf_net's scenarios
        plot_x_indices: a list of the indices of the state variables to plot
        plot_x_labels: a list of the labels for each state variable trace
        plot_indices: a list of the indices of the control inputs to plot
        plot_labels: a list of the labels for each control trace
        n_sims_per_start: the number of simulations to run (with random parameters),
                          per row in start_x
        t_sim: the amount of time to simulate for
        controller_period: the period determining how often the controller is run
        goal_check_fn: a function that takes a tensor and returns a tensor of booleans
                       indicating whether a state is in the goal
        out_of_bounds_check_fn: a function that takes a tensor and returns a tensor of
                                booleans indicating whether a state is out of bounds
    returns:
        a matplotlib.pyplot.figure containing the plots of state and control input
        over time.
    """
    # Deal with optional parameters
    # Default to clbf_net scenarios
    if scenarios is None:
        scenarios = clbf_net.scenarios
    # Default to starting from all state variables = 1.0
    if start_x is None:
        start_x = torch.ones(1, clbf_net.dynamics_model.n_dims, device=clbf_net.device)
    else:
        start_x = start_x.to(device=clbf_net.device)
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

    # Generate a tensor of start states
    x_sim_start = torch.zeros(n_sims, clbf_net.dynamics_model.n_dims).type_as(start_x)
    for i in range(0, start_x.shape[0]):
        for j in range(0, n_sims_per_start):
            x_sim_start[i * n_sims_per_start + j, :] = start_x[i, :]

    # Generate a random scenario for each rollout from the given scenarios
    random_scenarios = []
    for i in range(n_sims):
        random_scenario = {}
        for param_name in scenarios[0].keys():
            param_min = parameter_ranges[param_name][0]
            param_max = parameter_ranges[param_name][1]
            random_scenario[param_name] = random.uniform(param_min, param_max)
        random_scenarios.append(random_scenario)

    # Simulate!
    # (but first make somewhere to save the results)
    delta_t = clbf_net.dynamics_model.dt
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, n_sims, clbf_net.dynamics_model.n_dims).type_as(
        start_x
    )
    unsafe_mask_sim = torch.zeros_like(x_sim[:, :, 0], dtype=torch.bool)
    x_sim[0, :, :] = x_sim_start
    u_sim = torch.zeros(
        num_timesteps, n_sims, clbf_net.dynamics_model.n_controls
    ).type_as(start_x)
    V_sim = torch.zeros(num_timesteps, n_sims, 1).type_as(start_x)
    V_sim[0, :, 0] = clbf_net.V(x_sim[0, :, :]).squeeze()
    t_final = 0
    controller_failed = False
    goal_reached = False
    out_of_bounds = False
    controller_update_freq = int(controller_period / delta_t)
    try:
        print("Simulating CLBF rollout...")
        prog_bar_range = tqdm.trange(1, num_timesteps, desc="CLBF Rollout", leave=True)
        for tstep in prog_bar_range:
            # Get the current state
            x_current = x_sim[tstep - 1, :, :]
            # Get the control input at the current state if it's time
            if tstep == 1 or tstep % controller_update_freq == 0:
                u = clbf_net(x_current)
                u_sim[tstep, :, :] = u
            else:
                u = u_sim[tstep - 1, :, :]
                u_sim[tstep, :, :] = u

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = clbf_net.dynamics_model.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u[i, :].unsqueeze(0),
                    random_scenarios[i],
                )
                x_sim[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            # Compute the CLBF value
            V_sim[tstep, :, 0] = clbf_net.V(x_sim[tstep, :, :]).squeeze()

            t_final = tstep
            # If we've reached the goal, then stop the rollout
            if goal_check_fn is not None:
                if goal_check_fn(x_sim[tstep, :, :]).all():
                    goal_reached = True
                    break
            # Or if we've gone out of bounds, stop the rollout
            if out_of_bounds_check_fn is not None:
                if out_of_bounds_check_fn(x_sim[tstep, :, :]).any():
                    out_of_bounds = True
                    print(f"out of bounds at\n{x_sim[tstep, :, :]}")
                    break
            # Also check if we're in the unsafe region
            unsafe_mask_sim[tstep, :] = clbf_net.dynamics_model.unsafe_mask(
                x_sim[tstep, :, :]
            )

    except (Exception, RuntimeError, OverflowError):
        controller_failed = True

    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 12)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0]
    for i_trace in range(len(plot_x_indices)):
        ax1.plot(
            t[:t_final],
            x_sim[:t_final, :, plot_x_indices[i_trace]].cpu(),
            label=plot_x_labels[i_trace],
        )

    # Plot markers indicating where the simulations were unsafe
    zeros = np.zeros((num_timesteps,))
    ax1.plot(
        t[unsafe_mask_sim.any(dim=-1).cpu().numpy()],
        zeros[unsafe_mask_sim.any(dim=-1).cpu().numpy()],
        label="Unsafe",
    )

    # If the controller fails, mark that on the plot
    if goal_reached:
        ax1.set_title(f"Goal reached! s[0] = {random_scenarios[0]}")
    if out_of_bounds:
        ax1.set_title(f"Out of bounds! s[0] = {random_scenarios[0]}")
    if controller_failed:
        ax1.set_title(f"Controller failure! s[0] = {random_scenarios[0]}")

    ax1.set_xlabel("$t$")
    # ax1.legend(loc="lower center", ncol=n_sims * len(plot_x_labels) + 1)

    ax2 = axs[1]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[1:t_final],
            u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
            label=plot_u_labels[i_trace],
        )

    ax2.set_xlabel("$t$")
    # ax2.legend(loc="lower center", ncol=n_sims * len(plot_x_labels))

    ax3 = axs[2]
    ax3.plot(
        t[:t_final],
        V_sim[:t_final, :, 0].cpu(),
    )

    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$V$")

    fig.tight_layout()

    # Return the figure along with its name
    return "Rollout Plot", fig
