"""Define utilities for simulating the behavior of controlled dynamical systems"""
from typing import Callable, Tuple, List, Optional

import torch
from tqdm import tqdm
import numpy as np


# Define types for control and dynamics functions
# The dynamics take tensors of state and control and return a tensor of derivatives
DynamicsCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# The controllers take tensors of current state, reference state, and reference control
# and return a tensor of controls
ControllerCallable = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# The metrics takes a tensor of current state error and return a
# tensor of metric values
MetricCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricDerivCallable = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def simulate(
    x_init: torch.Tensor,
    x_ref: torch.Tensor,
    u_ref: torch.Tensor,
    sim_dt: float,
    controller_dt: float,
    dynamics: DynamicsCallable,
    controller: ControllerCallable,
    metric: Optional[MetricCallable] = None,
    metric_derivative: Optional[MetricDerivCallable] = None,
    control_bounds: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate the evolution of the system described by dx/dt = dynamics(x, controller(x))

    args:
        x_init: (N_batch, n_state_dims) tensor of initial states
        x_ref: (N_batch, N_steps, n_state_dims) tensor of reference states.
        u_ref: (N_batch, N_steps, n_control_dims) tensor of reference controls
        sim_dt: the timestep to use for Euler integration
        controller_dt: how often to evaluate the controller
        dynamics: the function giving the state derivatives for a state/control pair
        controller: the function giving the control for a state/reference triple
        metric: the function giving the metric for a state/reference double
        metric_derivative: the function giving the time derivative of the metric
        control_bounds: if provided, clamp the control inputs to the symmetric range
                        with bounds given by elements in this list.
    returns: a tuple of
        x_sim: (N, n_state_dims) tensor of simulated states
        u_sim: (N, n_control_dims) tensor of simulated control actions
        M_sim: (N, 1) tensor of metric values
    """
    # Sanity checks
    N_batch = x_ref.shape[0]
    N_steps = x_ref.shape[1]
    assert u_ref.shape[0] == N_batch, "References must have same number of batches"
    assert u_ref.shape[1] == N_steps, "References must have same number of steps"
    n_state_dims = x_ref.shape[-1]
    n_control_dims = u_ref.shape[-1]
    assert x_init.nelement() == n_state_dims * N_batch
    x_init = x_init.reshape(N_batch, n_state_dims)

    # Set up tensors to store state and control trajectories
    x_sim = torch.zeros(N_batch, N_steps, n_state_dims).type_as(x_ref)
    x_sim[:, 0, :] = x_init
    u_sim = torch.zeros(N_batch, N_steps, n_control_dims).type_as(x_sim)
    u_current = np.zeros((N_batch, n_control_dims))
    M_sim = torch.zeros(N_batch, N_steps, 1).type_as(x_sim)
    dMdt_sim = torch.zeros(N_batch, N_steps, 1).type_as(x_sim)

    # Simulate
    sim_range = tqdm(range(N_steps - 1))
    sim_range.set_description("Simulating")  # type: ignore
    controller_update_freq = int(controller_dt / sim_dt)
    for tstep in sim_range:
        # Get the current states
        x_current = x_sim[:, tstep].reshape(N_batch, n_state_dims)
        x_ref_current = x_ref[:, tstep].reshape(N_batch, n_state_dims)
        u_ref_current = u_ref[:, tstep].reshape(N_batch, n_control_dims)

        # Compute the metric
        if metric is not None:
            M_sim[:, tstep, 0] = metric(x_current, x_ref_current)
        if metric_derivative is not None:
            dMdt_sim[:, tstep, 0] = metric_derivative(
                x_current, x_ref_current, u_ref_current
            ).detach()

        # Get the control and save it
        if tstep % controller_update_freq == 0:
            u_current = controller(x_current, x_ref_current, u_ref_current)

            # if control_bounds is not None:
            #     # Clamp to control bounds
            #     for dim_idx in range(u_current.shape[-1]):
            #         u_current[:, dim_idx] = torch.clamp(
            #             u_current[:, dim_idx],
            #             min=-control_bounds[dim_idx],
            #             max=control_bounds[dim_idx],
            #         )

        u_sim[:, tstep + 1, :] = u_current

        # Get the derivatives and update the state
        x_dot = dynamics(x_current, u_current).detach()
        x_current = x_current.detach()
        x_sim[:, tstep + 1, :] = x_current + sim_dt * x_dot

    # Return the state and control trajectories
    return x_sim, u_sim, M_sim, dMdt_sim


@torch.no_grad()
def generate_random_reference(
    N_batch: int,
    T: float,
    dt: float,
    n_state_dims: int,
    n_control_dims: int,
    state_space: List[Tuple[float, float]],
    control_bounds: List[float],
    error_bounds: List[float],
    dynamics: DynamicsCallable,
    x_ref_init_0: Optional[torch.Tensor] = None,
    u_ref_init_0: Optional[torch.Tensor] = None,
    x_init_0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a random dynamically-consistent reference trajectory

    args:
        N_batch: the number of batches to simulate
        T: the length of time to simulate
        dt: the timestep
        n_state_dims: the number of state dimensions
        n_control_dims: the number of control dimensions
        state_space: a list of tuples of upper and lower bounds for each state dimension
        control_bounds: a list of tuples of magnitude bounds for each control input
        error_bounds: a list of tuples of magnitude bounds for each state error
        dynamics: the function giving the state derivatives for a state/control pair
        x_ref_init_0: start of the reference trajectory (if not provided, will be
            randomly selected)
        u_ref_init_0: start of the reference trajectory (if not provided, will be
            randomly selected)
        x_init_0: initial state error (if not provided, will be randomly selected)
    returns: a tuple of
        x_init: (N_batch, n_state_dims) tensor of initial states
        x_ref: (N_batch, T // dt, n_state_dims) tensor of reference states
        u_ref: (N_batch, T // dt, n_control_dims) tensor of reference controls
    """
    # If no start points provided, generate a bunch of random starting points
    if x_ref_init_0 is None:
        x_ref_init = torch.Tensor(N_batch, n_state_dims).uniform_(0.0, 1.0)
        for state_dim, state_limits in enumerate(state_space):
            x_max, x_min = state_limits
            x_ref_init[:, state_dim] = (
                x_ref_init[:, state_dim] * (x_max - x_min) + x_min
            )
    else:
        x_ref_init = torch.zeros((N_batch, n_state_dims)) + x_ref_init_0[0, :]

    # Generate random control inputs by combining sinusoids of different frequencies
    N_frequencies = 10
    t = torch.arange(0, T, dt)
    N_steps = t.shape[0]
    u_ref_weights = torch.Tensor(N_batch, N_frequencies, n_control_dims).uniform_(
        0.0, 1.0
    )
    u_ref = torch.zeros(N_batch, N_steps, n_control_dims)

    for batch_idx in range(N_batch):
        for control_idx in range(n_control_dims):
            # Normalize the contributions from each frequency
            weight_sum = u_ref_weights[batch_idx, :, control_idx].sum()
            u_ref_weights[batch_idx, :, control_idx] /= weight_sum

            for i in range(N_frequencies):
                weight = u_ref_weights[batch_idx, i, control_idx]
                u_ref[batch_idx, :, control_idx] += weight * torch.cos(
                    i * np.pi * t / T
                )

    # This process yields controls in [-1, 1], so renormalize to the control limits
    u_ref = 0.5 * (u_ref + 1.0)
    for control_dim, u_limit in enumerate(control_bounds):
        u_ref[:, :, control_dim] = u_ref[:, :, control_dim] * 2 * u_limit - u_limit

    # Hack for turtlebot, add some to the velocity to go forward
    if u_ref.shape[2] == 2:
        u_ref[:, :, 0] = 0.1 * u_ref[:, :, 0] + 0.1 * control_bounds[0]
        u_ref[:, :, 1] = 0.8 * u_ref[:, :, 1]
    else:
        u_ref = 0.1 * u_ref

    # If an initial control is provided, include that in the reference trajectory
    if u_ref_init_0 is not None:
        u_ref[:, 0, :] = 0 * u_ref[:, 0, :] + u_ref_init_0[0, :]

    # Generate the reference states by simulating under the reference control
    x_ref = torch.zeros(N_batch, N_steps, n_state_dims)
    x_ref[:, 0, :] = x_ref_init
    for tstep, _ in enumerate(t[:-1]):
        # Get the current states
        x_ref_current = x_ref[:, tstep].reshape(N_batch, n_state_dims)
        u_ref_current = u_ref[:, tstep].reshape(N_batch, n_control_dims)

        # Get the derivatives and update the state
        x_dot = dynamics(x_ref_current, u_ref_current)
        x_ref[:, tstep + 1, :] = x_ref_current + dt * x_dot

    # Generate some random initial states if no initial state is provided
    if x_init_0 is None:
        x_errors = torch.Tensor(N_batch, n_state_dims).uniform_(0.0, 1.0)
        for state_dim, error_limit in enumerate(error_bounds):
            error_limit *= 1.0
            x_errors[:, state_dim] = (
                x_errors[:, state_dim] * 2 * error_limit - error_limit
            )
        x_init = x_ref_init + x_errors
    else:
        x_init = torch.zeros((N_batch, n_state_dims)) + x_init_0[0, :]

    return x_init, x_ref, u_ref
