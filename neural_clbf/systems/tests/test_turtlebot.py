"""Test the TurtleBot3 dynamics"""
import pytest
import torch

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
    expected_upper = 1000 * torch.ones(2)
    expected_lower = -1000 * (torch.ones(2))
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
    assert turtlebot.n_dims == 3
    # TODO @bethlow
