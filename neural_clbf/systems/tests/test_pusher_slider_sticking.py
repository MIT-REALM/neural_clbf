"""Test the 2D quadrotor dynamics"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch

from neural_clbf.systems import StickingPusherSlider
from neural_clbf.systems import AutoRally


def test_StickingPusherSlider_init():
    """Test initialization of StickingPusherSlider model"""
    # Test instantiation with valid parameters
    valid_params = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
    }
    ps0 = StickingPusherSlider(valid_params)
    assert ps0 is not None


# def plot_autorally_straight_path():
#     """Test the dynamics of the kinematic car tracking a straight path"""
#     # Create the system
#     params = {
#         "psi_ref": 0.5,
#         "v_ref": 10.0,
#         "omega_ref": 0.0,
#     }
#     dt = 0.001
#     arcar = AutoRally(params, dt)
#     upper_u_lim, lower_u_lim = arcar.control_limits
#
#     # Simulate!
#     # (but first make somewhere to save the results)
#     t_sim = 10.0
#     n_sims = 1
#     controller_period = 0.01
#     num_timesteps = int(t_sim // dt)
#     start_x = arcar.goal_point.clone()
#     start_x[:, AutoRally.SYE] = 1.0
#     start_x[:, AutoRally.SXE] = -1.0
#     x_sim = torch.zeros(num_timesteps, n_sims, arcar.n_dims).type_as(start_x)
#     for i in range(n_sims):
#         x_sim[0, i, :] = start_x
#
#     u_sim = torch.zeros(num_timesteps, n_sims, arcar.n_controls).type_as(start_x)
#     controller_update_freq = int(controller_period / dt)
#     for tstep in range(1, num_timesteps):
#         # Get the current state
#         x_current = x_sim[tstep - 1, :, :]
#         # Get the control input at the current state if it's time
#         if tstep == 1 or tstep % controller_update_freq == 0:
#             u = arcar.u_nominal(x_current)
#             for dim_idx in range(arcar.n_controls):
#                 u[:, dim_idx] = torch.clamp(
#                     u[:, dim_idx],
#                     min=lower_u_lim[dim_idx].item(),
#                     max=upper_u_lim[dim_idx].item(),
#                 )
#
#             u_sim[tstep, :, :] = u
#         else:
#             u = u_sim[tstep - 1, :, :]
#             u_sim[tstep, :, :] = u
#
#         # Simulate forward using the dynamics
#         for i in range(n_sims):
#             xdot = arcar.closed_loop_dynamics(
#                 x_current[i, :].unsqueeze(0),
#                 u[i, :].unsqueeze(0),
#             )
#             x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()
#
#         t_final = tstep
#
#     # Get reference path
#     t = np.linspace(0, t_sim, num_timesteps)
#     psi_ref = params["psi_ref"]
#     x_ref = t * params["v_ref"] * np.cos(psi_ref)
#     y_ref = t * params["v_ref"] * np.sin(psi_ref)
#
#     # Convert trajectory from path-centric to world coordinates
#     x_err_path = x_sim[:, :, arcar.SXE].cpu().squeeze().numpy()
#     y_err_path = x_sim[:, :, arcar.SYE].cpu().squeeze().numpy()
#     x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
#     y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
#     fig, axs = plt.subplots(3, 1)
#     fig.set_size_inches(10, 12)
#     ax1 = axs[0]
#     ax1.plot(
#         x_world[:t_final],
#         y_world[:t_final],
#         linestyle="-",
#         label="Tracking",
#     )
#     ax1.plot(
#         x_ref[:t_final],
#         y_ref[:t_final],
#         linestyle=":",
#         label="Reference",
#     )
#     ax1.set_xlabel("$x$")
#     ax1.set_ylabel("$y$")
#     ax1.legend()
#     ax1.set_ylim([-t_sim * params["v_ref"], t_sim * params["v_ref"]])
#     ax1.set_xlim([-t_sim * params["v_ref"], t_sim * params["v_ref"]])
#     ax1.set_aspect("equal")
#
#     # psi_err_path = x_sim[:, :, arcar.PSI_E].cpu().squeeze().numpy()
#     # delta_path = x_sim[:, :, arcar.DELTA].cpu().squeeze().numpy()
#     # v_err_path = x_sim[:, :, arcar.VE].cpu().squeeze().numpy()
#     # ax1.plot(t[:t_final], y_err_path[:t_final])
#     # ax1.plot(t[:t_final], x_err_path[:t_final])
#     # ax1.plot(t[:t_final], psi_err_path[:t_final])
#     # ax1.plot(t[:t_final], delta_path[:t_final])
#     # ax1.plot(t[:t_final], v_err_path[:t_final])
#     # ax1.legend(["y", "x", "psi", "delta", "ve"])
#
#     ax2 = axs[1]
#     plot_u_indices = [arcar.VDELTA, arcar.OMEGA_R_DOT]
#     plot_u_labels = ["$v_\\delta$", r"$\dot{\omega}_r$"]
#     for i_trace in range(len(plot_u_indices)):
#         ax2.plot(
#             t[1:t_final],
#             u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
#             label=plot_u_labels[i_trace],
#         )
#     ax2.legend()
#
#     ax3 = axs[2]
#     ax3.plot(
#         t[:t_final],
#         # x_sim[:t_final, :, :AutoRally.PSI_E + 1].norm(dim=-1).squeeze().numpy(),
#         x_sim[:t_final, :, :4].squeeze(),
#         # label="Tracking Error x-y-psi",
#         label=[
#             "SXE",
#             "SYE",
#             "PSI_E",
#             "DELTA",
#         ],
#     )
#     ax3.plot(
#         t[:t_final],
#         # x_sim[:t_final, :, :AutoRally.PSI_E + 1].norm(dim=-1).squeeze().numpy(),
#         x_sim[:t_final, :, 7:].squeeze(),
#         # label="Tracking Error x-y-psi",
#         label=[
#             "VY",
#             "PSI_E_DOT",
#         ],
#     )
#     ax3.legend()
#     ax3.set_xlabel("$t$")
#
#     plt.show()
#
#
# def plot_autorally_circle_path():
#     """Test the dynamics of the kinematic car tracking a circle path"""
#     # Create the system
#     params = {
#         "psi_ref": 1.0,
#         "v_ref": 10.0,
#         "omega_ref": 0.5,
#     }
#     dt = 0.01
#     arcar = AutoRally(params, dt)
#     upper_u_lim, lower_u_lim = arcar.control_limits
#
#     # Simulate!
#     # (but first make somewhere to save the results)
#     t_sim = 20.0
#     n_sims = 1
#     controller_period = dt
#     num_timesteps = int(t_sim // dt)
#     start_x = arcar.goal_point.clone()
#     start_x[:, AutoRally.SYE] = 1.0
#     start_x[:, AutoRally.SXE] = -1.0
#     x_sim = torch.zeros(num_timesteps, n_sims, arcar.n_dims).type_as(start_x)
#     for i in range(n_sims):
#         x_sim[0, i, :] = start_x
#
#     u_sim = torch.zeros(num_timesteps, n_sims, arcar.n_controls).type_as(start_x)
#     controller_update_freq = int(controller_period / dt)
#
#     # And create a place to store the reference path
#     x_ref = np.zeros(num_timesteps)
#     y_ref = np.zeros(num_timesteps)
#     psi_ref = np.zeros(num_timesteps)
#     psi_ref[0] = 1.0
#
#     # Simulate!
#     for tstep in range(1, num_timesteps):
#         # Get the current state
#         x_current = x_sim[tstep - 1, :, :]
#         # Get the control input at the current state if it's time
#         if tstep == 1 or tstep % controller_update_freq == 0:
#             u = arcar.u_nominal(x_current)
#             for dim_idx in range(arcar.n_controls):
#                 u[:, dim_idx] = torch.clamp(
#                     u[:, dim_idx],
#                     min=lower_u_lim[dim_idx].item(),
#                     max=upper_u_lim[dim_idx].item(),
#                 )
#
#             u_sim[tstep, :, :] = u
#         else:
#             u = u_sim[tstep - 1, :, :]
#             u_sim[tstep, :, :] = u
#
#         # Get the path parameters at this point
#         psi_ref[tstep] = dt * params["omega_ref"] + psi_ref[tstep - 1]
#         pt = copy(params)
#         pt["psi_ref"] = psi_ref[tstep]
#         x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * np.cos(psi_ref[tstep])
#         y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * np.sin(psi_ref[tstep])
#
#         # Simulate forward using the dynamics
#         for i in range(n_sims):
#             xdot = arcar.closed_loop_dynamics(
#                 x_current[i, :].unsqueeze(0),
#                 u[i, :].unsqueeze(0),
#                 pt,
#             )
#             x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()
#
#         t_final = tstep
#
#     # Get reference path
#     t = np.linspace(0, t_sim, num_timesteps)
#
#     # Convert trajectory from path-centric to world coordinates
#     x_err_path = x_sim[:, :, arcar.SXE].cpu().squeeze().numpy()
#     y_err_path = x_sim[:, :, arcar.SYE].cpu().squeeze().numpy()
#     x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
#     y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
#     fig, axs = plt.subplots(3, 1)
#     fig.set_size_inches(10, 12)
#     ax1 = axs[0]
#     ax1.plot(
#         x_world[:t_final],
#         y_world[:t_final],
#         linestyle="-",
#         label="Tracking",
#     )
#     ax1.plot(
#         x_ref[:t_final],
#         y_ref[:t_final],
#         linestyle=":",
#         label="Reference",
#     )
#
#     ax1.set_xlabel("$x$")
#     ax1.set_ylabel("$y$")
#     ax1.legend()
#     ax1.set_aspect("equal")
#
#     ax2 = axs[1]
#     plot_u_indices = [arcar.VDELTA, arcar.OMEGA_R_DOT]
#     plot_u_labels = ["$v_\\delta$", r"$\dot{\omega}_r$"]
#     for i_trace in range(len(plot_u_indices)):
#         ax2.plot(
#             t[1:t_final],
#             u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
#             label=plot_u_labels[i_trace],
#         )
#     ax2.legend()
#
#     ax3 = axs[2]
#     ax3.plot(
#         t[:t_final],
#         x_sim[:t_final, :, : AutoRally.PSI_E + 1].norm(dim=-1).squeeze().numpy(),
#         label="Tracking Error",
#     )
#     ax3.legend()
#     ax3.set_xlabel("$t$")
#
#     plt.show()
#
#
# def plot_autorally_s_path(v_ref: float = 10.0):
#     """Test the dynamics of the kinematic car tracking a S path"""
#     # Create the system
#     params = {
#         "psi_ref": 1.0,
#         "v_ref": v_ref,
#         "omega_ref": 0.0,
#     }
#     dt = 0.01
#     arcar = AutoRally(params, dt)
#     upper_u_lim, lower_u_lim = arcar.control_limits
#
#     # Simulate!
#     # (but first make somewhere to save the results)
#     t_sim = 10.0
#     n_sims = 1
#     controller_period = dt
#     num_timesteps = int(t_sim // dt)
#     start_x = arcar.goal_point.clone()
#     start_x[:, AutoRally.SYE] = 1.0
#     start_x[:, AutoRally.SXE] = -1.0
#     x_sim = torch.zeros(num_timesteps, n_sims, arcar.n_dims).type_as(start_x)
#     for i in range(n_sims):
#         x_sim[0, i, :] = start_x
#
#     u_sim = torch.zeros(num_timesteps, n_sims, arcar.n_controls).type_as(start_x)
#     controller_update_freq = int(controller_period / dt)
#
#     # And create a place to store the reference path
#     x_ref = np.zeros(num_timesteps)
#     y_ref = np.zeros(num_timesteps)
#     psi_ref = np.zeros(num_timesteps)
#     psi_ref[0] = 1.0
#
#     # Simulate!
#     pt = copy(params)
#     for tstep in tqdm.trange(1, num_timesteps):
#         # Get the path parameters at this point
#         omega_ref_t = 1.0 * np.sin(tstep * dt) + params["omega_ref"]
#         psi_ref[tstep] = dt * omega_ref_t + psi_ref[tstep - 1]
#         pt = copy(pt)
#         pt["psi_ref"] = psi_ref[tstep]
#         x_ref[tstep] = x_ref[tstep - 1] + dt * pt["v_ref"] * np.cos(psi_ref[tstep])
#         y_ref[tstep] = y_ref[tstep - 1] + dt * pt["v_ref"] * np.sin(psi_ref[tstep])
#         pt["omega_ref"] = omega_ref_t
#
#         # Get the current state
#         x_current = x_sim[tstep - 1, :, :]
#         # Get the control input at the current state if it's time
#         if tstep == 1 or tstep % controller_update_freq == 0:
#             u = arcar.u_nominal(x_current, pt)
#             for dim_idx in range(arcar.n_controls):
#                 u[:, dim_idx] = torch.clamp(
#                     u[:, dim_idx],
#                     min=lower_u_lim[dim_idx].item(),
#                     max=upper_u_lim[dim_idx].item(),
#                 )
#
#             u_sim[tstep, :, :] = u
#         else:
#             u = u_sim[tstep - 1, :, :]
#             u_sim[tstep, :, :] = u
#
#         # Simulate forward using the dynamics
#         for i in range(n_sims):
#             xdot = arcar.closed_loop_dynamics(
#                 x_current[i, :].unsqueeze(0),
#                 u[i, :].unsqueeze(0),
#                 pt,
#             )
#             x_sim[tstep, i, :] = x_current[i, :] + dt * xdot.squeeze()
#
#         t_final = tstep
#
#     t = np.linspace(0, t_sim, num_timesteps)
#
#     # Convert trajectory from path-centric to world coordinates
#     x_err_path = x_sim[:, :, arcar.SXE].cpu().squeeze().numpy()
#     y_err_path = x_sim[:, :, arcar.SYE].cpu().squeeze().numpy()
#     x_world = x_ref + x_err_path * np.cos(psi_ref) - y_err_path * np.sin(psi_ref)
#     y_world = y_ref + x_err_path * np.sin(psi_ref) + y_err_path * np.cos(psi_ref)
#     fig, axs = plt.subplots(3, 1)
#     fig.set_size_inches(10, 12)
#     ax1 = axs[0]
#     ax1.plot(
#         x_world[:t_final],
#         y_world[:t_final],
#         linestyle="-",
#         label="Tracking",
#     )
#     ax1.plot(
#         x_ref[:t_final],
#         y_ref[:t_final],
#         linestyle=":",
#         label="Reference",
#     )
#
#     ax1.set_xlabel("$x$")
#     ax1.set_ylabel("$y$")
#     ax1.legend()
#     # ax1.set_aspect("equal")
#
#     ax2 = axs[1]
#     plot_u_indices = [arcar.VDELTA, arcar.OMEGA_R_DOT]
#     plot_u_labels = ["$v_\\delta$", r"$\dot{\omega}_r$"]
#     for i_trace in range(len(plot_u_indices)):
#         ax2.plot(
#             t[1:t_final],
#             u_sim[1:t_final, :, plot_u_indices[i_trace]].cpu(),
#             label=plot_u_labels[i_trace],
#         )
#     ax2.legend()
#
#     ax3 = axs[2]
#     ax3.plot(
#         t[:t_final],
#         x_sim[:t_final, :, : AutoRally.PSI_E + 1].norm(dim=-1).squeeze().numpy(),
#         label="Tracking Error",
#     )
#     ax3.legend()
#     ax3.set_xlabel("$t$")
#
#     plt.show()
#
#     # Return the maximum tracking error
#     tracking_error = x_sim[:, :, : AutoRally.PSI_E + 1]
#     return tracking_error[:t_final, :, :].norm(dim=-1).squeeze().max()

if __name__ == "__main__":
    # plot_autorally_straight_path()
    # plot_autorally_circle_path()
    # plot_autorally_s_path()
    test_StickingPusherSlider_init()
