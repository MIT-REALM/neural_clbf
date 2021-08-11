import pytest
import numpy as np
from shapely.geometry import box
import torch
import matplotlib.pyplot as plt

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

    # Make sure we can get the minimum distance to an obstacle
    q = torch.tensor([[0.5, 1.2, 0.0]])
    min_distances = scene.min_distance_to_obstacle(q)
    assert torch.allclose(min_distances, torch.zeros_like(min_distances) + 0.2)


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
    # looks away from all the obstacles), with all velocities zero
    q = torch.tensor([3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    num_rays = 10
    field_of_view = (-np.pi / 2, np.pi / 2)
    max_distance = 10
    measurement = scene.lidar_measurement(q, num_rays, field_of_view, max_distance)
    # Check that the measurements are the proper shape
    assert measurement.ndim == 3
    assert measurement.shape[0] == 1  # one point queried
    assert measurement.shape[1] == 2  # x, y are measured
    assert measurement.shape[2] == num_rays  # number of measurements

    # Get some lidar measurement right up next to the blocks
    # (these should actually measure something), with some velocities
    q = torch.tensor(
        [
            [0.7, 1.5, 0.0],
            [0.75, 1.25, -np.pi / 4],
        ]
    )
    num_rays = 3
    field_of_view = (-np.pi / 4, np.pi / 4)
    max_distance = 10
    measurement = scene.lidar_measurement(q, num_rays, field_of_view, max_distance)
    assert measurement.ndim == 3
    assert measurement.shape == (2, 2, num_rays)

    # Check each measurement individually
    measurement1 = measurement[0]
    measurement2 = measurement[1]

    # Measurement for the first point
    expect_x = 0.3
    expect_y_lower = -0.3
    expect_y_mid = 0.0
    expect_y_upper = 0.3
    expect_meas_upper = torch.tensor([expect_x, expect_y_upper])
    expect_meas_mid = torch.tensor([expect_x, expect_y_mid])
    expect_meas_lower = torch.tensor([expect_x, expect_y_lower])
    assert np.allclose(measurement1[:, 0], expect_meas_lower)
    assert np.allclose(measurement1[:, 1], expect_meas_mid)
    assert np.allclose(measurement1[:, 2], expect_meas_upper)

    # Measurement for the second point
    # Check the first ray first (pointing down)
    expect_x = 0.25 * np.cos(field_of_view[0])
    expect_y = 0.25 * np.sin(field_of_view[0])
    expect_meas = torch.tensor([expect_x, expect_y])
    assert np.allclose(measurement2[:, 0], expect_meas)
    # Then the second ray second (pointing straight)
    expect_x = 0.25 * np.sqrt(2)
    expect_y = 0.0
    expect_meas = torch.tensor([expect_x, expect_y])
    assert np.allclose(measurement2[:, 1], expect_meas)
    # And the third ray third (pointing up)
    expect_x = 0.25 * np.cos(field_of_view[1])
    expect_y = 0.25 * np.sin(field_of_view[1])
    expect_meas = torch.tensor([expect_x, expect_y])
    assert np.allclose(measurement2[:, 2], expect_meas)


def plot_scene_lidar_measurement():
    """Test whether the simulated LIDAR measurement is working"""
    # Create a test scene
    obstacles = [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)]
    scene = Scene(obstacles)

    # Get some lidar measurement right up next to the blocks
    # (these should actually measure something)
    q = torch.tensor(
        [
            [0.7, 1.5, -1.0],
        ]
    )
    num_rays = 10
    field_of_view = (-np.pi / 4, np.pi / 4)
    max_distance = 1.0
    measurement = scene.lidar_measurement(q, num_rays, field_of_view, max_distance)

    fig, ax = plt.subplots()
    scene.plot(ax)
    ax.set_aspect("equal")

    ax.plot(q[:, 0], q[:, 1], "ko")

    lidar_pts = measurement[0, :, :]
    rotation_mat = torch.tensor(
        [
            [torch.cos(q[0, 2]), -torch.sin(q[0, 2])],
            [torch.sin(q[0, 2]), torch.cos(q[0, 2])],
        ]
    )
    lidar_pts = rotation_mat @ lidar_pts
    lidar_pts[0, :] += q[0, 0]
    lidar_pts[1, :] += q[0, 1]
    ax.plot(lidar_pts[0, :], lidar_pts[1, :], "k-o")

    plt.show()


if __name__ == "__main__":
    plot_scene_lidar_measurement()
