"""Test the 2D quadrotor dynamics"""
import pytest
import torch

from neural_clbf.systems import Quad2D
from neural_clbf.systems.utils import grav


def test_quad2d_init():
    """Test initialization of Quad2D"""
    # Test instantiation with valid parameters
    valid_params = {
        "m": 1.0,
        "I": 0.001,
        "r": 0.25,
    }
    quad2d = Quad2D(valid_params)
    assert quad2d is not None
    assert quad2d.n_dims == 6
    assert quad2d.n_controls == 2

    # Test instantiation without all needed parameters
    incomplete_params_list = [
        {},
        {"m": 1.0},
        {"I": 0.001},
        {"r": 0.25},
        {"m": 1.0, "r": 0.25},
        {"m": 1.0, "I": 0.001},
        {"I": 0.001, "r": 0.25},
    ]
    for incomplete_params in incomplete_params_list:
        with pytest.raises(ValueError):
            quad2d = Quad2D(incomplete_params)

    # Test instantiation with unphysical parameters
    non_physical_params_list = [
        {"m": -1.0, "I": 0.001, "r": 0.25},
        {"m": 1.0, "I": -0.001, "r": 0.25},
        {"m": 1.0, "I": 0.001, "r": -0.25},
        {"m": 0.0, "I": 0.0, "r": 0.0},
    ]
    for non_physical_params in non_physical_params_list:
        with pytest.raises(ValueError):
            quad2d = Quad2D(non_physical_params)


def test_quad2d_dynamics():
    """Test the dynamics of the 2D quadrotor"""
    # Create the quadrotor system
    params = {"m": 1.0, "I": 0.001, "r": 0.25}
    quad2d = Quad2D(params)

    # The dynamics should have a fixed point at the origin with equilibrium controls
    x_origin = torch.zeros((1, quad2d.n_dims))
    u_eq = torch.zeros((1, quad2d.n_controls)) + grav * params["m"] / 2.0
    xdot = quad2d.closed_loop_dynamics(x_origin, u_eq)
    assert torch.allclose(xdot, torch.zeros((1, quad2d.n_dims)))

    # At the origin, if controls are both increased, then we should experience positive
    # z acceleration
    u = u_eq + 1.0
    xdot = quad2d.closed_loop_dynamics(x_origin, u)
    assert xdot[0, Quad2D.VZ] > 0.0
    # all other derivatives should be zero
    # (check by zeroing vz derivative and checking all of xdot)
    xdot[0, Quad2D.VZ] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, quad2d.n_dims)))

    # At the origin, if controls are both decreased, then we should experience negative
    # z acceleration
    u = u_eq - 1.0
    xdot = quad2d.closed_loop_dynamics(x_origin, u)
    assert xdot[0, Quad2D.VZ] < 0.0
    # all other derivatives should be zero
    # (check by zeroing vz derivative and checking all of xdot)
    xdot[0, Quad2D.VZ] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, quad2d.n_dims)))

    # At the origin, if controls are changed in opposite directions, then we should
    # experience theta acceleration
    u = u_eq - 1.0
    u[:, 0] += 2.0
    xdot = quad2d.closed_loop_dynamics(x_origin, u)
    assert xdot[0, Quad2D.THETA_DOT] > 0.0
    # all other derivatives should be zero
    # (check by zeroing vz derivative and checking all of xdot)
    xdot[0, Quad2D.THETA_DOT] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, quad2d.n_dims)))


def test_quad2d_u_nominal():
    """Test the nominal controller for the 2D quadrotor"""
    # Create the quadrotor system
    params = {"m": 1.0, "I": 0.001, "r": 0.25}
    quad2d = Quad2D(params)

    # The nominal controller is a linear one, so we'll test about the origin
    x_origin = torch.zeros((1, quad2d.n_dims))
    u_eq = torch.zeros((1, quad2d.n_controls)) + grav * params["m"] / 2.0

    # At the origin, the nominal control should be equal to the equilibrium control
    u_nominal = quad2d.u_nominal(x_origin)
    assert torch.allclose(u_nominal, u_eq)

    # Increase z slightly, and u_nominal should decrease
    x = x_origin.clone()
    x[:, quad2d.PZ] += 0.1
    u_nominal = quad2d.u_nominal(x)
    assert (u_nominal < u_eq).all()

    # Decrease z slightly, and u_nominal should increase
    x = x_origin.clone()
    x[:, quad2d.PZ] -= 0.1
    u_nominal = quad2d.u_nominal(x)
    assert (u_nominal > u_eq).all()

    # Increasing x should cause the right motor to thrust more (tilting the quad to
    # move left to the origin)
    x = x_origin.clone()
    x[:, quad2d.PX] += 0.1
    u_nominal = quad2d.u_nominal(x)
    assert (u_nominal[:, quad2d.U_RIGHT] > u_nominal[:, quad2d.U_LEFT]).all()

    # Decreasing x should cause the left motor to thrust more (tilting the quad to
    # move right to the origin)
    x = x_origin.clone()
    x[:, quad2d.PX] -= 0.1
    u_nominal = quad2d.u_nominal(x)
    assert (u_nominal[:, quad2d.U_RIGHT] < u_nominal[:, quad2d.U_LEFT]).all()
