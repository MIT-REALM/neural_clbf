import pytest
import numpy as np
from shapely.geometry import box
import torch

from neural_clbf.systems.planar_lidar_system import Scene


def test_scene_initializaton():
    """Test instantiation of a scene object with and without obstacles"""

    # First test instantiation without any obstacles provided
    scene = Scene([])
    assert (
        scene.obstacles == []
    ), "Environment should not contain any obstacles if initialized with an empty list"

    # Next test instantiation with some obstacles provided
    obstacles = [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)]
    scene = Scene(obstacles)
    assert len(scene.obstacles) == len(
        obstacles
    ), "Environment.obstacles should contain all provided obstacles"
    assert np.all(
        [obs in obstacles for obs in scene.obstacles]
    ), "Environment.obstacles should contain all provided obstacles"


def test_scene_add_obstacle():
    """Test adding an obstacle to a scene"""
    # Create an empty scene
    scene = Scene([])
    # Add an obstacle to it
    new_box = box(0.0, 0.0, 1.0, 1.0)
    scene.add_obstacle(new_box)
    # Make sure it was successfully added
    assert len(scene.obstacles) == 1
    assert scene.obstacles[0] == new_box

    # Make sure adding is idempotent (adding the box again doesn't do anything)
    scene.add_obstacle(new_box)
    assert len(scene.obstacles) == 1


def test_scene_remove_obstacle():
    """Test removing an obstacle from a scene"""
    # Create a scene with one obstacle
    box1 = box(0.0, 0.0, 1.0, 1.0)
    scene = Scene([box1])
    # Remove the box
    scene.remove_obstacle(box1)
    assert len(scene.obstacles) == 0

    # Check that we get a ValueError if we try to remove something
    # that's not already in the scene
    with pytest.raises(ValueError):
        scene.remove_obstacle(box1)


def test_scene_lidar_measurement():
    """Test whether the simulated LIDAR measurement is working"""
    # Create a test scene
    obstacles = [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)]
    scene = Scene(obstacles)

    # Get a lidar measurement from x=y=3, theta = 0 (all rays should saturate since this
    # looks away from all the obstacles)
    q = torch.tensor([3.0, 3.0, 0.0])
    num_rays = 10
    field_of_view = (-np.pi / 2, np.pi / 2)
    max_distance = 10
    measurement = scene.lidar_measurement(q, num_rays, field_of_view, max_distance)
    assert measurement.shape == (1, num_rays)
    assert np.allclose(measurement, max_distance)

    # Get some lidar measurement right up next to the blocks
    # (these should actually measure something)
    q = torch.tensor(
        [
            [0.7, 1.5, 0.0],
            [0.75, 1.25, -np.pi / 4],
        ]
    )
    num_rays = 50
    field_of_view = (-np.pi / 4, np.pi / 4)
    max_distance = 10
    measurement = scene.lidar_measurement(q, num_rays, field_of_view, max_distance)
    assert measurement.shape == (2, num_rays)

    measurement1 = measurement[0, :]
    measurement2 = measurement[1, :]

    # Test the expected measurements based on some pen-and-paper geometry
    assert np.isclose(measurement1.max(), 0.3 / np.sin(np.pi / 4))
    assert np.isclose(measurement1.min(), 0.3)

    assert np.isclose(measurement2.max(), np.sqrt(0.25 ** 2 + 0.25 ** 2))
    assert np.isclose(measurement2.min(), 0.25)
