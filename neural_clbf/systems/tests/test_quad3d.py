"""Test the 2D quadrotor dynamics"""
import pytest
import torch

from neural_clbf.systems import Quad3D
from neural_clbf.systems.utils import grav


def test_quad3d_init():
    """Test initialization of Quad3D"""
    # Test instantiation with valid parameters
    valid_params = {
        "m": 1.0,
    }
    quad3d = Quad3D(valid_params)
    assert quad3d is not None
    assert quad3d.n_dims == 9
    assert quad3d.n_controls == 4

    # Test that we can get nominal controls for a bunch of random points
    N = 100
    x = quad3d.sample_state_space(N)
    u = quad3d.u_nominal(x)
    assert u.shape[0] == N
    assert u.shape[1] == quad3d.n_controls
    assert u.ndim == 2

    # Test instantiation without all needed parameters
    incomplete_params_list = [
        {},
    ]
    for incomplete_params in incomplete_params_list:
        with pytest.raises(ValueError):
            quad3d = Quad3D(incomplete_params)

    # Test instantiation with unphysical parameters
    non_physical_params_list = [
        {"m": -1.0},
        {"m": 0.0},
    ]
    for non_physical_params in non_physical_params_list:
        with pytest.raises(ValueError):
            quad3d = Quad3D(non_physical_params)


def test_quad3d_dynamics():
    """Test the dynamics of the 2D quadrotor"""
    # Create the quadrotor system
    params = {"m": 1.0}
    quad3d = Quad3D(params)

    # The dynamics should have a fixed point at the origin with equilibrium controls
    x_origin = torch.zeros((1, quad3d.n_dims))
    u_eq = torch.zeros((1, quad3d.n_controls))
    u_eq[0, 0] = grav * params["m"]
    xdot = quad3d.closed_loop_dynamics(x_origin, u_eq)
    assert torch.allclose(xdot, torch.zeros((1, quad3d.n_dims)))

    # At the origin, if thrust is increased, then we should experience negative
    # z acceleration (upwards)
    u = u_eq.clone().detach()
    u[0, 0] += 1.0
    xdot = quad3d.closed_loop_dynamics(x_origin, u)
    assert xdot[0, Quad3D.VZ] < 0.0
    # all other derivatives should be zero
    # (check by zeroing vz derivative and checking all of xdot)
    xdot[0, Quad3D.VZ] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, quad3d.n_dims)))

    # At the origin, if thrust is decreased, then we should experience positive
    # z acceleration (downwards)
    u = u_eq.clone().detach()
    u[0, 0] -= 1.0
    xdot = quad3d.closed_loop_dynamics(x_origin, u)
    assert xdot[0, Quad3D.VZ] > 0.0
    # all other derivatives should be zero
    # (check by zeroing vz derivative and checking all of xdot)
    xdot[0, Quad3D.VZ] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, quad3d.n_dims)))


def test_quad3d_u_nominal():
    """Test the nominal controller for the 2D quadrotor"""
    # Create the quadrotor system
    params = {"m": 1.0}
    quad3d = Quad3D(params)

    # The nominal controller is a linear one, so we'll test about the origin
    x_origin = torch.zeros((1, quad3d.n_dims))
    u_eq = torch.zeros((1, quad3d.n_controls))
    u_eq[0, 0] = grav * params["m"]

    # At the origin, the nominal control should be equal to the equilibrium control
    u_nominal = quad3d.u_nominal(x_origin)
    assert torch.allclose(u_nominal, u_eq)

    # Increase z slightly, and u_nominal thrust should increase (since we need to go up)
    x = x_origin.clone()
    x[:, quad3d.PZ] += 0.1
    u_nominal = quad3d.u_nominal(x)
    assert (u_nominal[0, 0] > u_eq[0, 0]).all()

    # Decrease z slightly, and u_nominal should decrease (we need to go down)
    x = x_origin.clone()
    x[:, quad3d.PZ] -= 0.1
    u_nominal = quad3d.u_nominal(x)
    assert (u_nominal[0, 0] < u_eq[0, 0]).all()


def test_quad3d_obstacles_safe_unsafe_mask():
    """Test the safe and unsafe mask for the 2D quadrotor with obstacles"""
    valid_params = {
        "m": 1.0,
    }
    quad3d = Quad3D(valid_params)
    # These points should all be safe
    safe_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # origin
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],  # near origin
        ]
    )
    assert torch.all(quad3d.safe_mask(safe_x))

    # These points should all be unsafe
    unsafe_x = torch.tensor(
        [
            [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # too low
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Too far
        ]
    )
    assert torch.all(quad3d.unsafe_mask(unsafe_x))
