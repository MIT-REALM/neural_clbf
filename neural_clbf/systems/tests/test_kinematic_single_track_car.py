"""Test the 2D quadrotor dynamics"""
from copy import copy

import matplotlib.pyplot as plt

import numpy as np
import torch

from neural_clbf.systems import KSCar


def test_kscar_init():
    """Test initialization of kinematic car"""
    # Test instantiation with valid parameters
    valid_params = {
        "psi_ref_c": 0.5403,
        "psi_ref_s": 0.8415,
        "v_ref": 1.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    kscar = KSCar(valid_params)
    assert kscar is not None
    assert kscar.n_dims == 5
    assert kscar.n_controls == 2


def plot_kscar_straight_path():
    """Test the dynamics of the kinematic car tracking a straight path"""
    # Create the system
    params = {
        "psi_ref_c": 0.5403,
        "psi_ref_s": 0.8415,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dt = 0.01
    kscar = KSCar(params, dt)

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 10.0
    n_sims = 1
    controller_period = dt
    num_timesteps = int(t_sim // dt)
    start_x = torch.tensor([[0.0, 3.0, 0.0, -10.0, 1.0]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / dt)
    for tstep in range(1, num_timesteps):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = kscar.u_nominal(x_current)
            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u[i, :].unsqueeze(0),
            )
            x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()

        t_final = tstep

    t = np.linspace(0, t_sim, num_timesteps)
    x_ref = t * params["v_ref"] * params["psi_ref_c"]
    y_ref = t * params["v_ref"] * params["psi_ref_s"]
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_sim[:t_final, :, kscar.SXE].cpu().squeeze() + x_ref[:t_final],
        x_sim[:t_final, :, kscar.SYE].cpu().squeeze() + y_ref[:t_final],
        linestyle="-",
        label="Tracking",
    )
    ax1.plot(
        x_ref[:t_final],
        y_ref[:t_final],
        linestyle=":",
        label="Reference",
    )

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()

    ax2 = axs[1]
    plot_u_indices = [kscar.VDELTA, kscar.ALONG]
    plot_u_labels = ["$v_\\delta$", "$a_{long}$"]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[1:t_final],
            u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
            label=plot_u_labels[i_trace],
        )

    plt.show()


def plot_kscar_circle_path():
    """Test the dynamics of the kinematic car tracking a circle path"""
    # Create the system
    params = {
        "psi_ref_c": 0.5403,
        "psi_ref_s": 0.8415,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.5,
    }
    dt = 0.01
    kscar = KSCar(params, dt)

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 20.0
    n_sims = 1
    controller_period = dt
    num_timesteps = int(t_sim // dt)
    start_x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / dt)

    # And create a place to store the reference path
    psi_ref_0 = 1.0
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)

    # Simulate!
    for tstep in range(1, num_timesteps):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = kscar.u_nominal(x_current)
            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Get the path parameters at this point
        psi_ref_t = tstep * dt * params["omega_ref"] + psi_ref_0
        pt = copy(params)
        pt["psi_ref_c"] = np.cos(psi_ref_t)
        pt["psi_ref_s"] = np.sin(psi_ref_t)
        x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * pt["psi_ref_c"]
        y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * pt["psi_ref_s"]

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u[i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()

        t_final = tstep

    t = np.linspace(0, t_sim, num_timesteps)
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_sim[:t_final, :, kscar.SXE].cpu().squeeze() + x_ref[:t_final],
        x_sim[:t_final, :, kscar.SYE].cpu().squeeze() + y_ref[:t_final],
        linestyle="-",
        label="Tracking",
    )
    ax1.plot(
        x_ref[:t_final],
        y_ref[:t_final],
        linestyle=":",
        label="Reference",
    )

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()

    ax2 = axs[1]
    plot_u_indices = [kscar.VDELTA, kscar.ALONG]
    plot_u_labels = ["$v_\\delta$", "$a_{long}$"]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[1:t_final],
            u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
            label=plot_u_labels[i_trace],
        )

    plt.show()


def plot_kscar_s_path():
    """Test the dynamics of the kinematic car tracking a S path"""
    # Create the system
    params = {
        "psi_ref_c": 0.5403,
        "psi_ref_s": 0.8415,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dt = 0.01
    kscar = KSCar(params, dt)

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 20.0
    n_sims = 1
    controller_period = dt
    num_timesteps = int(t_sim // dt)
    start_x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    x_sim = torch.zeros(num_timesteps, n_sims, kscar.n_dims).type_as(start_x)
    for i in range(n_sims):
        x_sim[0, i, :] = start_x

    u_sim = torch.zeros(num_timesteps, n_sims, kscar.n_controls).type_as(start_x)
    controller_update_freq = int(controller_period / dt)

    # And create a place to store the reference path
    x_ref = np.zeros(num_timesteps)
    y_ref = np.zeros(num_timesteps)

    # Simulate!
    pt = copy(params)
    psi_ref_t = 1.0
    for tstep in range(1, num_timesteps):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = kscar.u_nominal(x_current)
            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Get the path parameters at this point
        omega_ref_t = 1.5 * np.sin(tstep * dt) + params["omega_ref"]
        psi_ref_t = dt * omega_ref_t + psi_ref_t
        pt = copy(pt)
        pt["psi_ref_c"] = np.cos(psi_ref_t)
        pt["psi_ref_s"] = np.sin(psi_ref_t)
        pt["omega_ref"] = omega_ref_t
        x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * pt["psi_ref_c"]
        y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * pt["psi_ref_s"]

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u[i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()

        t_final = tstep

    t = np.linspace(0, t_sim, num_timesteps)
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_sim[:t_final, :, kscar.SXE].cpu().squeeze() + x_ref[:t_final],
        x_sim[:t_final, :, kscar.SYE].cpu().squeeze() + y_ref[:t_final],
        linestyle="-",
        label="Tracking",
    )
    ax1.plot(
        x_ref[:t_final],
        y_ref[:t_final],
        linestyle=":",
        label="Reference",
    )

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()

    ax2 = axs[1]
    plot_u_indices = [kscar.VDELTA, kscar.ALONG]
    plot_u_labels = ["$v_\\delta$", "$a_{long}$"]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[1:t_final],
            u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
            label=plot_u_labels[i_trace],
        )

    plt.show()


if __name__ == "__main__":
    plot_kscar_straight_path()
    plot_kscar_circle_path()
    plot_kscar_s_path()
