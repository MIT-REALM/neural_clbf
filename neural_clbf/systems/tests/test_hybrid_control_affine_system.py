"""
test_hybrid_control_affine_system.py
Description:
    Tests the new template system for dynamical systems like the
    pusher-slider that have this hybrid form of control affine system.
"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch

from neural_clbf.systems import HybridControlAffineSystem

import jax.numpy as jnp
from torch.autograd.functional import jacobian

def test_jacobian():
    """Test initialization of StickingPusherSlider model"""
    # Constants
    x0 = torch.zeros((1, 3))
    x0[:, 0] = 1.0
    x0[:, 1] = 2.0
    x0[:, 2] = 3.0
    # x0 = torch.tensor([[1.0, 2.0, 3.0]])
    u0 = torch.tensor([[4.0, 5.0]])

    n_batch = 1
    x = torch.zeros((n_batch, 3, 1))
    u = torch.zeros((n_batch, 2, 1))

    #x[0, :, :] = x0.squeeze()
    #u[0, :, :] = u0

    def simple_f(x: torch.Tensor, u: torch.Tensor):
        """ A simple linear dynamics that depends on BOTH x and u """
        batch_size = x.shape[0]
        fx = torch.zeros((batch_size, 3, 1))
        fx = fx.type_as(x)

        fx[:, 0, 0] = x[:, 0]
        fx[:, 1, 0] = x[:, 1] + u[:, 1]
        fx[:, 2, 0] = x[:, 2] + u[:, 1]

        return fx

    def simple_g(x:torch.Tensor, u:torch.Tensor):
        """ A simple linear dynamics that depends on BOTH x and u """
        batch_size = x.shape[0]
        gx = torch.zeros((batch_size, 3, 2))
        gx = gx.type_as(x)

        gx[:, 0, 0] = x[:, 0] + u[:, 1]
        gx[:, 1, 0] = x[:, 1] + u[:, 0]
        gx[:, 0, 1] = x[:, 2] + u[:, 0]

        return gx

    # Jacobians with respect to two functions of dynamics
    u = torch.zeros((n_batch,2,1))
    for batch_index in range(n_batch):
        u[batch_index,:,:] = u0.T
    dynamics = lambda x_in: (simple_f(x_in, u0) + torch.bmm(simple_g(x_in, u), u)).squeeze()
    # A1 = jacobian(dynamics, x0).squeeze().cpu().numpy()
    A2 = jacobian(dynamics, x0).squeeze().cpu().numpy()


    print(A2.view())

    return

def test_mat_mul1():
    """
    test_mat_mul1
    Description:
        Attempting to see how easy it is to use tensor multiplication to convert a 4D tensor
        into a 3d tensor when the tensor contains a lot of data (hopefully this matches the format of the tensor).
    """
    # Constants
    batch_size = 1
    n_modes = 4
    f = torch.zeros((batch_size, 3, 1, n_modes))

    # Create f
    for mode_index in range(4):
        f[:, 0, 0, mode_index] = float(mode_index)
        f[:, 1, 0, mode_index] = float(mode_index)
        f[:, 2, 0, mode_index] = float(mode_index)

    print(f)

    # If I want to select the elements only corresponding to mode target_mode,
    # then we can get it via simple matrix multiplication.
    target_mode = 3
    c = torch.zeros(n_modes,1)
    c[target_mode-1] = 1

    prod1 = torch.matmul(f,c)

    print(prod1)
    print(prod1.squeeze())
    print(prod1.shape)

    f_finalized = torch.zeros((batch_size,3,1))
    f_finalized[0,:,0] = prod1.squeeze()
    print(f_finalized)

    assert len(f_finalized.shape) == 3

def test_mat_mul2():
    """
    test_mat_mul1
    Description:
        Attempting to see how easy it is to use tensor multiplication to convert a

    """
    # Constants
    batch_size = 1
    n_modes = 4
    f = torch.zeros((batch_size, 3, n_modes))

    # Create f
    for mode_index in range(4):
        f[:, 0, mode_index] = float(mode_index)
        f[:, 1, mode_index] = float(mode_index)
        f[:, 2, mode_index] = float(mode_index)

    print(f)

    c = torch.zeros(n_modes,1)
    c[2] = 1

    prod1 = torch.matmul(f,c)

    print(prod1)
    print(prod1.squeeze())
    print(prod1.shape)

    assert len(prod1.shape) == 3

def test_mat_mul3():
    """
    test_mat_mul3
    Description:
        Attempting to see how easy it is to insert a tensor slice into the massive
        tensor that we need for the mode selection multiplication.

    """
    print("test_mat_mul3")
    print("=============")
    print(" ")

    # Constants
    batch_size = 1
    n_modes = 4
    f = torch.zeros((batch_size, 3, n_modes))

    f_x_star = torch.zeros((batch_size, 3, 1))
    f_x_star[:, 0, 0] = 15.0
    f_x_star[:, 1, 0] = 35.0
    f_x_star[:, 2, 0] = 36.0

    # Create f
    for mode_index in range(4):
        f[:, 0, mode_index] = float(mode_index)
        f[:, 1, mode_index] = float(mode_index)
        f[:, 2, mode_index] = float(mode_index)

    f[:, :, 2] = f_x_star.flatten()

    print(f)

    c = torch.zeros(n_modes, 1)
    c[2] = 1

    prod1 = torch.matmul(f, c)

    print(prod1)
    print(prod1.squeeze())
    print(prod1.shape)

    assert len(prod1.shape) == 3

def test_squeeze1():
    """
    test_squeeze1
    Description:
        Attempting to squeeze a four dimensional vector (n_x x n_y x n_z x 1) into a
        three dimensional vector. Hopefully nothing bad happens.

    """
    print("test_squeeze1")
    print("=============")
    print(" ")

    # Constants
    batch_size = 1
    n_modes = 4
    n_x = 3
    n_y = 2
    g = torch.zeros((batch_size, n_x, n_y, n_modes))

    # Create f
    for x_index in range(n_x):
        for y_index in range(n_y):
            for mode_index in range(n_modes):
                g[:, x_index, y_index, mode_index] = float(x_index) + 0.1*float(y_index) + 0.01*float(mode_index)

    print(g)

    c = torch.zeros(n_modes, 1)
    c[2] = 1

    prod1 = torch.matmul(g, c)
    print(prod1)
    print(f"prod1.shape = {prod1.shape}")

    assert len(prod1.shape) == 4

    print("now compressing prod1 into a smaller vector")
    g_prime = torch.zeros((batch_size, n_x, n_y))
    g_prime[:,:,:] = prod1.squeeze()
    print(g_prime)

    assert len(g_prime.shape) == 3

# def test_init():
#     """
#     test_init
#     Description:
#         Tests whether or not the default constructor for a HybridControlAffineSystem
#         works correctly.
#     """
#
#     # constants
#     sys0 = HybridControlAffineSystem()
#
#     assert sys0.n_modes == 1

# def test_jacobian():
#     """Test initialization of StickingPusherSlider model"""
#     # Test instantiation with valid parameters
#     valid_params = {
#         "s_x_ref": 1.0,
#         "s_y_ref": 1.0,
#     }
#     ps0 = StickingPusherSlider(valid_params)
#
#
#
#     assert ps0 is not None


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
    # Test Jacobian
    test_jacobian()

    # Test Some of the Tensor Manipulations that we need to use in this code
    test_mat_mul1()
    test_mat_mul2()
    test_mat_mul3()
    test_squeeze1()

    # Test HybridControlAffineSystem object
    # test_init() # Cannot create an abstract class unfortunately.