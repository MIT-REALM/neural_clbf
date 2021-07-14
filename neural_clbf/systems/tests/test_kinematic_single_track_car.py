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
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    kscar = KSCar(valid_params, controller_dt=0.01, dt=0.001)
    assert kscar is not None
    assert kscar.n_dims == 5
    assert kscar.n_controls == 2

    print(kscar.K)


def plot_kscar_straight_path():
    """Test the dynamics of the kinematic car tracking a straight path"""
    # Create the system
    params = {
        "psi_ref": 0.5,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dt = 0.001
    kscar = KSCar(params, dt)
    upper_u_lim, lower_u_lim = kscar.control_limits

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 1.0
    n_sims = 1
    controller_period = 0.01
    num_timesteps = int(t_sim // dt)
    start_x = torch.tensor([[0.0, 1.0, 0.0, 1.0, -np.pi / 6]])
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
            for dim_idx in range(kscar.n_controls):
                u[:, dim_idx] = torch.clamp(
                    u[:, dim_idx], min=lower_u_lim[dim_idx], max=upper_u_lim[dim_idx]
                )

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
            # # check that the dynamics are rotation invariant
            # for j in range(10):
            #     psi_ref_new = params["psi_ref"] + j * np.pi / 10
            #     test_params = copy(params)
            #     test_params["psi_ref"] = psi_ref_new
            #     xdot_test = kscar.closed_loop_dynamics(
            #         x_current[i, :].unsqueeze(0), u[i, :].unsqueeze(0), test_params
            #     )
            #     assert torch.allclose(xdot, xdot_test)
            x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()

        t_final = tstep

    # Get reference path
    t = np.linspace(0, t_sim, num_timesteps)
    psi_ref = params["psi_ref"]
    x_ref = t * params["v_ref"] * np.cos(psi_ref)
    y_ref = t * params["v_ref"] * np.sin(psi_ref)

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, kscar.SXE].cpu().squeeze().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().squeeze().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
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
    ax1.set_ylim([-t_sim * params["v_ref"], t_sim * params["v_ref"]])
    ax1.set_xlim([-t_sim * params["v_ref"], t_sim * params["v_ref"]])
    ax1.set_aspect("equal")

    # psi_err_path = x_sim[:, :, kscar.PSI_E].cpu().squeeze().numpy()
    # delta_path = x_sim[:, :, kscar.DELTA].cpu().squeeze().numpy()
    # v_err_path = x_sim[:, :, kscar.VE].cpu().squeeze().numpy()
    # ax1.plot(t[:t_final], y_err_path[:t_final])
    # ax1.plot(t[:t_final], x_err_path[:t_final])
    # ax1.plot(t[:t_final], psi_err_path[:t_final])
    # ax1.plot(t[:t_final], delta_path[:t_final])
    # ax1.plot(t[:t_final], v_err_path[:t_final])
    # ax1.legend(["y", "x", "psi", "delta", "ve"])

    ax2 = axs[1]
    plot_u_indices = [kscar.VDELTA, kscar.ALONG]
    plot_u_labels = ["$v_\\delta$", "$a_{long}$"]
    for i_trace in range(len(plot_u_indices)):
        ax2.plot(
            t[1:t_final],
            u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
            label=plot_u_labels[i_trace],
        )
    ax2.legend()

    ax3 = axs[2]
    ax3.plot(
        t[:t_final],
        x_sim[:t_final, :, :].norm(dim=-1).squeeze().numpy(),
        label="Tracking Error",
    )
    ax3.legend()
    ax3.set_xlabel("$t$")

    plt.show()


def plot_kscar_circle_path():
    """Test the dynamics of the kinematic car tracking a circle path"""
    # Create the system
    params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.3,
    }
    dt = 0.01
    kscar = KSCar(params, dt)
    upper_u_lim, lower_u_lim = kscar.control_limits

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
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    # Simulate!
    for tstep in range(1, num_timesteps):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = kscar.u_nominal(x_current)
            for dim_idx in range(kscar.n_controls):
                u[:, dim_idx] = torch.clamp(
                    u[:, dim_idx], min=lower_u_lim[dim_idx], max=upper_u_lim[dim_idx]
                )

            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

        # Get the path parameters at this point
        psi_ref[tstep] = dt * params["omega_ref"] + psi_ref[tstep - 1]
        pt = copy(params)
        pt["psi_ref"] = psi_ref[tstep]
        x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * np.cos(psi_ref[tstep])
        y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * np.sin(psi_ref[tstep])

        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = kscar.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u[i, :].unsqueeze(0),
                pt,
            )
            x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()

        t_final = tstep

    # Get reference path
    t = np.linspace(0, t_sim, num_timesteps)

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, kscar.SXE].cpu().squeeze().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().squeeze().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
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
    ax2.legend()

    ax3 = axs[2]
    ax3.plot(
        t[:t_final],
        x_sim[:t_final, :, :].norm(dim=-1).squeeze().numpy(),
        label="Tracking Error",
    )
    ax3.legend()
    ax3.set_xlabel("$t$")

    plt.show()


def plot_kscar_s_path():
    """Test the dynamics of the kinematic car tracking a S path"""
    # Create the system
    params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dt = 0.01
    kscar = KSCar(params, dt)
    upper_u_lim, lower_u_lim = kscar.control_limits

    # Simulate!
    # (but first make somewhere to save the results)
    t_sim = 50.0
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
    psi_ref = np.zeros(num_timesteps)
    psi_ref[0] = 1.0

    # Simulate!
    pt = copy(params)
    for tstep in range(1, num_timesteps):
        # Get the path parameters at this point
        omega_ref_t = 1.5 * np.sin(tstep * dt) + params["omega_ref"]
        psi_ref[tstep] = dt * omega_ref_t + psi_ref[tstep - 1]
        pt = copy(pt)
        pt["psi_ref"] = psi_ref[tstep]
        x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * np.cos(psi_ref[tstep])
        y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * np.sin(psi_ref[tstep])
        pt["omega_ref"] = omega_ref_t

        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state if it's time
        if tstep == 1 or tstep % controller_update_freq == 0:
            u = kscar.u_nominal(x_current, pt)
            for dim_idx in range(kscar.n_controls):
                u[:, dim_idx] = torch.clamp(
                    u[:, dim_idx], min=lower_u_lim[dim_idx], max=upper_u_lim[dim_idx]
                )

            u_sim[tstep, :, :] = u
        else:
            u = u_sim[tstep - 1, :, :]
            u_sim[tstep, :, :] = u

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

    # Convert trajectory from path-centric to world coordinates
    x_err_path = x_sim[:, :, kscar.SXE].cpu().squeeze().numpy()
    y_err_path = x_sim[:, :, kscar.SYE].cpu().squeeze().numpy()
    x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
    y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 12)
    ax1 = axs[0]
    ax1.plot(
        x_world[:t_final],
        y_world[:t_final],
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
    ax2.legend()

    ax3 = axs[2]
    ax3.plot(
        t[:t_final],
        x_sim[:t_final, :, :].norm(dim=-1).squeeze().numpy(),
        label="Tracking Error",
    )
    ax3.legend()
    ax3.set_xlabel("$t$")

    plt.show()


if __name__ == "__main__":
    # plot_kscar_straight_path()
    # plot_kscar_circle_path()
    plot_kscar_s_path()
