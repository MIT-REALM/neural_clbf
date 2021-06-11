"""Test the 2D quadrotor dynamics"""
import torch

from neural_clbf.systems import NeuralLander
from neural_clbf.systems.utils import grav


def test_neurallander_init():
    """Test initialization of NeuralLander"""
    # Test instantiation with valid parameters
    valid_params = {}
    nl = NeuralLander(valid_params)
    assert nl is not None
    assert nl.n_dims == 6
    assert nl.n_controls == 3

    # Test that we can get nominal controls for a bunch of random points
    N = 100
    x = nl.sample_state_space(N)
    u = nl.u_nominal(x)
    assert u.shape[0] == N
    assert u.shape[1] == nl.n_controls
    assert u.ndim == 2


def test_neurallander_u_nominal():
    """Test the nominal controller for the 2D quadrotor"""
    # Create the quadrotor system
    params = {}
    nl = NeuralLander(params)

    # The nominal controller is a linear one, so we'll test about the origin
    x_origin = torch.zeros((1, nl.n_dims))
    u_eq = torch.zeros((1, nl.n_controls))
    u_eq[0, 2] = grav * NeuralLander.mass

    # At the origin, the nominal control should be equal to the equilibrium control
    u_nominal = nl.u_nominal(x_origin)
    assert torch.allclose(u_nominal, u_eq)

    # Increase z slightly, and u_nominal thrust should decrease (since we need to go up)
    x = x_origin.clone()
    x[:, nl.PZ] += 0.1
    u_nominal = nl.u_nominal(x)
    assert (u_nominal[0, NeuralLander.AZ] < u_eq[0, NeuralLander.AZ]).all()

    # Decrease z slightly, and u_nominal should increase (we need to go down)
    x = x_origin.clone()
    x[:, nl.PZ] -= 0.1
    u_nominal = nl.u_nominal(x)
    assert (u_nominal[0, NeuralLander.AZ] > u_eq[0, NeuralLander.AZ]).all()


def test_neurallander_obstacles_safe_unsafe_mask():
    """Test the safe and unsafe mask for the 2D quadrotor with obstacles"""
    valid_params = {
        "m": 1.0,
    }
    nl = NeuralLander(valid_params)
    # These points should all be safe
    safe_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # origin
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # near origin
        ]
    )
    assert torch.all(nl.safe_mask(safe_x))

    # These points should all be unsafe
    unsafe_x = torch.tensor(
        [
            [0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # too low
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # Too far
        ]
    )
    assert torch.all(nl.unsafe_mask(unsafe_x))
