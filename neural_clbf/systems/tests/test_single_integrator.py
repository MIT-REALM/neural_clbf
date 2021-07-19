"""Test the 2D quadrotor dynamics"""
import numpy as np

from neural_clbf.systems import SingleIntegrator2D
from neural_clbf.systems.planar_lidar_system import Scene


def test_single_integrator2d_init():
    """Test initialization of SingleIntegrator2D"""
    # Test instantiation with valid parameters and scene
    valid_params = {}

    scene = Scene([])
    room_size = 10.0
    num_obstacles = 5
    box_size_range = (0.2, 0.8)
    position_range = (-4.0, 4.0)
    rotation_range = (-np.pi, np.pi)
    scene.add_walls(room_size)
    scene.add_random_boxes(
        num_obstacles,
        box_size_range,
        position_range,
        position_range,
        rotation_range,
    )
    vehicle = SingleIntegrator2D(valid_params, scene)
    assert vehicle is not None
    assert vehicle.n_dims == 2
    assert vehicle.n_controls == 2

    # Test that we can get nominal controls for a bunch of random points
    N = 100
    x = vehicle.sample_state_space(N)
    u = vehicle.u_nominal(x)
    assert u.shape[0] == N
    assert u.shape[1] == vehicle.n_controls
    assert u.ndim == 2
