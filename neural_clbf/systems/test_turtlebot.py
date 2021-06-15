"""Test the TurtleBot3 dynamics"""
import pytest
import torch

from neural_clbf.systems import TurtleBot

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
    
    # Test instantiation without all necessary parameters
    
