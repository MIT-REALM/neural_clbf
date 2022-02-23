"""Define the dynamics and jacobians in python."""
from typing import Tuple

import torch
import numpy as np

"""
Throughout this file, we will use the naming convention

    f_system_name(x, u) - compute the state derivative given x and u.

        args:
            x: batch_size x n_dims tensor of state
            u: batch_size x n_controls tensor of control input
        returns
            batch_size x n_dims tensor of state derivatives

    AB_system_name(x, u) - compute the Jacobian of the dynamics wrt x and u
                           about the point (x, u).

        args:
            x: batch_size x n_dims tensor of reference state
            u: batch_size x n_controls tensor of reference control input
        returns:
            A: batch_size x n_dims x n_dims tensor of Jacobian wrt state
            B: batch_size x n_dims x n_controls tensor of Jacobian wrt control
"""

# ----------------------------------------------------------------------------
# Damped integrator
#
# state: px, vx
# controls: u
#
# dynamics:
#
#   d/dt px = vx
#   d/dt vx = -0.01 * vx + u
#
# Models a rigid body moving on a line with damping. Very simple dynamics and
# low-dimensional.
# ----------------------------------------------------------------------------


def f_damped_integrator(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Get state variables
    vx = x[:, 1]

    xdot = torch.zeros_like(x)
    xdot[:, 0] = vx
    xdot[:, 1] = -0.1 * vx + u[:, 0]
    return xdot


def AB_damped_integrator(
    x: torch.Tensor, u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # This is a linear system, so the jacobian is constant
    A = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, -0.1],
        ]
    )
    B = torch.tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    batch_size = x.shape[0]
    return A.expand(batch_size, -1, -1), B.expand(batch_size, -1, -1)


# ----------------------------------------------------------------------------
# Turtlebot
#
# state: px, py, theta
# controls: v, omega
#
# dynamics:
#
#   d/dt px = v cos(theta)
#   d/dt py = v sin(theta)
#   d/dt theta = omega
#
# Models a 2-wheel differential drive robot (turtlebot!). Tricky non-holonomic
# dynamics (can't go sideways! uncontrollable linearization!)
# ----------------------------------------------------------------------------


def f_turtlebot(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Convenience: define indices for state and controls
    PXi, PYi, THETAi, Vi, OMEGAi = (0, 1, 2, 0, 1)

    # Get state variables and controls
    theta = x[:, THETAi]
    v = u[:, Vi]
    omega = u[:, OMEGAi]

    # Construct the derivatives tensor
    xdot = torch.zeros_like(x)
    xdot[:, PXi] = v * torch.cos(theta)
    xdot[:, PYi] = v * torch.sin(theta)
    xdot[:, THETAi] = omega

    return xdot


def AB_turtlebot(x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # A is defined as df/dx (the Jacobian of f with respect to x)
    # Convenience: define indices for state and controls
    PXi, PYi, THETAi, Vi, OMEGAi = (0, 1, 2, 0, 1)

    # Get state variables and controls
    theta = x[:, THETAi]
    v = u[:, Vi]

    # A is the jacobian of xdot with respect to x
    batch_size = x.shape[0]
    num_state_dims = x.shape[1]
    A = torch.zeros((batch_size, num_state_dims, num_state_dims)).type_as(x)
    A[:, PXi, PXi] = 0.0  # d/dx f_x = 0
    A[:, PXi, PYi] = 0.0  # d/dy f_x = 0
    A[:, PXi, THETAi] = -v * torch.sin(theta)  # d/dtheta f_x

    A[:, PYi, PXi] = 0.0  # d/dx f_y = 0
    A[:, PYi, PYi] = 0.0  # d/dy f_y = 0
    A[:, PYi, THETAi] = v * torch.cos(theta)  # d/dtheta f_y

    A[:, THETAi, PXi] = 0.0  # d/dx f_theta = 0
    A[:, THETAi, PYi] = 0.0  # d/dy f_theta = 0
    A[:, THETAi, THETAi] = 0.0  # d/dtheta f_theta = 0

    # B is the jacobian of xdot with respect to u
    num_controls = u.shape[1]
    B = torch.zeros((batch_size, num_state_dims, num_controls)).type_as(x)
    B[:, PXi, Vi] = torch.cos(theta)  # d/dv f_x = 0
    B[:, PXi, OMEGAi] = 0.0  # d/dw f_x = 0

    B[:, PYi, Vi] = torch.sin(theta)  # d/dv f_y = 0
    B[:, PYi, OMEGAi] = 0.0  # d/dw f_y = 0

    B[:, THETAi, Vi] = 0.0  # d/dv f_theta = 0
    B[:, THETAi, OMEGAi] = 1.0  # d/dw f_theta = 0

    return A, B


# ----------------------------------------------------------------------------
# 9D Quadrotor
#
# state: px, py, pz, vx, vy, vz, phi, theta, psi
# controls: f, phi_dot, theta_dot, psi_dot
#
# ----------------------------------------------------------------------------


def f_quad9d(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Convenience: define indices for state and controls
    PXi, PYi, PZi, VXi, VYi, VZi, PHIi, THETAi, PSIi = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    Fi, PHI_DOTi, THETA_DOTi, PSI_DOTi = (0, 1, 2, 3)

    # Get state variables and controls
    vx = x[:, VXi]
    vy = x[:, VYi]
    vz = x[:, VYi]
    phi = x[:, PHIi]
    theta = x[:, THETAi]
    f = u[:, Fi]
    phi_dot = x[:, PHI_DOTi]
    theta_dot = x[:, THETA_DOTi]
    psi_dot = x[:, PSI_DOTi]

    # Convenience: sine and cosine of some angles
    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)
    s_phi = torch.sin(phi)
    c_phi = torch.cos(phi)

    # Construct the derivatives tensor
    mass = 0.05
    xdot = torch.zeros_like(x)
    xdot[:, PXi] = vx
    xdot[:, PYi] = vy
    xdot[:, PZi] = vz

    xdot[:, VXi] = -f * s_theta / mass
    xdot[:, VYi] = f * c_theta * s_phi / mass
    xdot[:, VZi] = 9.81 - f * c_theta * c_phi / mass

    xdot[:, PHIi] = phi_dot
    xdot[:, THETAi] = theta_dot
    xdot[:, PSIi] = psi_dot

    return xdot


def AB_quad9d(x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convenience: define indices for state and controls
    PXi, PYi, PZi, VXi, VYi, VZi, PHIi, THETAi, PSIi = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    Fi, PHI_DOTi, THETA_DOTi, PSI_DOTi = (0, 1, 2, 3)

    # Get state variables and controls
    phi = x[:, PHIi]
    theta = x[:, THETAi]
    f = u[:, Fi]

    # Convenience: sine and cosine of some angles
    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)
    s_phi = torch.sin(phi)
    c_phi = torch.cos(phi)

    # A is the jacobian of xdot with respect to x
    mass = 0.05
    batch_size = x.shape[0]
    num_state_dims = x.shape[1]
    A = torch.zeros((batch_size, num_state_dims, num_state_dims)).type_as(x)
    A[:, PXi, VXi] = 1.0
    A[:, PYi, VYi] = 1.0
    A[:, PZi, VZi] = 1.0

    A[:, VXi, THETAi] = -f * c_theta / mass

    A[:, VYi, THETAi] = -f * s_theta * s_phi / mass
    A[:, VYi, PHIi] = f * c_theta * c_phi / mass

    A[:, VZi, THETAi] = f * s_theta * c_phi / mass
    A[:, VZi, PHIi] = f * c_theta * s_phi / mass

    # B is the jacobian of xdot with respect to u
    num_controls = u.shape[1]
    B = torch.zeros((batch_size, num_state_dims, num_controls)).type_as(x)
    B[:, VXi, Fi] = -s_theta / mass
    B[:, VYi, Fi] = c_theta * s_phi / mass
    B[:, VZi, Fi] = -c_theta * c_phi / mass

    B[:, PHIi, PHI_DOTi] = 1.0
    B[:, THETAi, THETA_DOTi] = 1.0
    B[:, PSIi, PSI_DOTi] = 1.0

    return A, B


# ----------------------------------------------------------------------------
# 6D Quadrotor (ignore orientation)
#
# state: px, py, pz, vx, vy, vz
# controls: ux, uy, uz (force in each direction)
#
# ----------------------------------------------------------------------------


def f_quad6d(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # TODO convert to nonlinear 6D
    # Convenience: define indices for state and controls
    PXi, PYi, PZi, VXi, VYi, VZi = (0, 1, 2, 3, 4, 5)
    UXi, UYi, UZi = (0, 1, 2)

    # Get state variables and controls
    vx = x[:, VXi]
    vy = x[:, VYi]
    vz = x[:, VYi]
    ux = u[:, UXi]
    uy = u[:, UYi]
    uz = u[:, UZi]

    # Construct the derivatives tensor
    mass = 0.05
    xdot = torch.zeros_like(x)
    xdot[:, PXi] = vx
    xdot[:, PYi] = vy
    xdot[:, PZi] = vz

    xdot[:, VXi] = ux / mass
    xdot[:, VYi] = uy / mass
    xdot[:, VZi] = uz / mass - 9.81  # TODO how to deal with gravity?

    return xdot


def AB_quad6d(x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convenience: define indices for state and controls
    PXi, PYi, PZi, VXi, VYi, VZi = (0, 1, 2, 3, 4, 5)
    UXi, UYi, UZi = (0, 1, 2)

    # A is the jacobian of xdot with respect to x
    mass = 0.05
    batch_size = x.shape[0]
    num_state_dims = x.shape[1]
    A = torch.zeros((batch_size, num_state_dims, num_state_dims)).type_as(x)
    A[:, PXi, VXi] = 1.0
    A[:, PYi, VYi] = 1.0
    A[:, PZi, VZi] = 1.0

    # B is the jacobian of xdot with respect to u
    num_controls = u.shape[1]
    B = torch.zeros((batch_size, num_state_dims, num_controls)).type_as(x)
    B[:, VXi, UXi] = 1.0 / mass
    B[:, VYi, UYi] = 1.0 / mass
    B[:, VZi, UZi] = 1.0 / mass

    return A, B


"""Utils Section."""


def wrap_numpy(func):
    """
    A numpy wrapper for julia for testing that the julia and python versions are
    equivalent.

    func can be f(x,u) or AB(x,u)
    """

    def wrapped_func(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_torch = torch.Tensor(x)
        u_torch = torch.Tensor(u)
        output = func(x_torch, u_torch)
        if isinstance(output, torch.Tensor):
            return output.numpy()
        elif isinstance(output, tuple):
            return tuple(o.numpy() for o in output)

    return wrapped_func
