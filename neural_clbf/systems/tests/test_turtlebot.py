"""Test the TurtleBot3 dynamics"""
import pytest
import torch
import numpy as np

from neural_clbf.systems.turtlebot import TurtleBot


def test_turtlebot_init():
    """Test initialization of TurtleBot3"""
    # Test instantiation with valid parameters
    valid_params = {
        "R": 0.1,
        "L": 0.5,
    }

    turtlebot = TurtleBot(valid_params)
    assert turtlebot is not None
    assert turtlebot.n_dims == 3
    assert turtlebot.n_controls == 2

    # Check control limits
    upper_lim, lower_lim = turtlebot.control_limits
    expected_upper = torch.ones(2)
    expected_upper[0] = 1000.0
    expected_upper[1] = 4.0 * np.pi
    expected_lower = -1 * expected_upper
    assert torch.allclose(upper_lim, expected_upper, atol=0.1)
    assert torch.allclose(lower_lim, expected_lower, atol=0.1)

    # Test instantiation without all necessary parameters
    incomplete_params_list = [
        {},
        {"R": 0.3},
        {"L": 0.8},
        {"r": 0.1},
    ]

    for incomplete_params in incomplete_params_list:
        with pytest.raises(ValueError):
            turtlebot = TurtleBot(incomplete_params)

    # Test instantiation with unphysical parameters
    non_physical_params_list = [
        {"R": -0.2, "L": 0.5},
        {"R": 0.2, "L": -0.5},
        {"R": 0.0, "L": 0.0},
    ]

    for non_physical_params in non_physical_params_list:
        with pytest.raises(ValueError):
            turtlebot = TurtleBot(non_physical_params)


def test_turtlebot_dynamics():
    """Test the dynamics of the TurtleBot3"""
    # Create the turtlebot system
    params = {"R": 0.07, "L": 0.3}
    turtlebot = TurtleBot(params)

    # The dynamics should be fixed at the orgin with zero controls
    x_origin = torch.zeros((1, turtlebot.n_dims))
    u_eq = torch.zeros((1, turtlebot.n_controls))
    xdot = turtlebot.closed_loop_dynamics(x_origin, u_eq)
    assert torch.allclose(xdot, torch.zeros((1, turtlebot.n_dims)))

    # If linear velocity is increased and angular velocity constant
    # then we should experience positive position change in x and y, and no
    # change in theta
    u = u_eq.clone()
    u[0, TurtleBot.V] += 1.0
    xdot = turtlebot.closed_loop_dynamics(x_origin, u)
    assert xdot[0, TurtleBot.X] > 0.0
    assert xdot[0, TurtleBot.Y] == 0.0
    # all other columns should be zero
    xdot[0, TurtleBot.X] = 0.0
    xdot[0, TurtleBot.Y] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, turtlebot.n_dims)))

    # If controls in the linear velocity are decreased, then we should
    # experience negative x and y position change
    u = u_eq.clone()
    u[0, TurtleBot.V] -= 1.0
    xdot = turtlebot.closed_loop_dynamics(x_origin, u)
    assert xdot[0, TurtleBot.X] < 0.0
    assert xdot[0, TurtleBot.Y] == 0.0
    # all other columns should be zero
    xdot[0, TurtleBot.X] = 0.0
    xdot[0, TurtleBot.Y] = 0.0
    assert torch.allclose(xdot, torch.zeros((1, turtlebot.n_dims)))

    # If angular velocity is positive and linear velocity is zero,
    # then we should experience positive theta orientation
    u = u_eq.clone()
    u[0, TurtleBot.THETA_DOT] += 1.0
    xdot = turtlebot.closed_loop_dynamics(x_origin, u)
    # x and y position should be zero
    assert xdot[0, TurtleBot.X] == 0.0
    assert xdot[0, TurtleBot.Y] == 0.0
    # theta should be negative
    assert xdot[0, TurtleBot.THETA] > 0.0

    # If both angular velocity and linear velocity are increased, we should
    # experience positive change in x and y position and
    # decrease in orientation angle
    u = u_eq.clone()
    u[0, TurtleBot.V] += 1.0
    u[0, TurtleBot.THETA_DOT] += 1.0

    xdot = turtlebot.closed_loop_dynamics(x_origin, u)
    assert xdot[0, TurtleBot.X] > 0.0
    assert xdot[0, TurtleBot.Y] == 0.0
    assert xdot[0, TurtleBot.THETA] > 0.0
