"""Test the data generation for the 2D quadrotor with obstacles"""
import torch
import numpy as np

from neural_clbf.experiments.quad2d_obstacles.data_generation import (
    Quad2DObstaclesDataModule,
)


def test_quad2d_obstacles_safe_unsafe_mask():
    """Test the safe and unsafe mask for the 2D quadrotor with obstacles"""
    dm = Quad2DObstaclesDataModule()
    # These points should all be safe
    safe_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # origin
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # near origin
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],  # near origin
            [-1.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # left of obstacle 1
            [0.5, 1.6, 0.0, 0.0, 0.0, 0.0],  # above obstacle 2
        ]
    )
    assert torch.all(dm.safe_mask_fn(safe_x))

    # These points should all be unsafe
    unsafe_x = torch.tensor(
        [
            [0.0, -0.4, 0.0, 0.0, 0.0, 0.0],  # too low
            [-0.7, 0.5, 0.0, 0.0, 0.0, 0.0],  # inside obstacle 1
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],  # inside obstacle 2
        ]
    )
    assert torch.all(dm.unsafe_mask_fn(unsafe_x))


def test_quad2d_obstacles_goal_mask():
    """Test the goal mask for the 2D quadrotor with obstacles"""
    dm = Quad2DObstaclesDataModule()
    # These points should all be in the goal
    in_goal = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # origin
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # near origin
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],  # near origin in x
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],  # near origin in z
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],  # near origin in theta
            [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],  # slow enough in x
            [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # slow enough in z
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2],  # slow enough in theta dot
        ]
    )
    assert torch.all(dm.goal_mask_fn(in_goal))

    # These points should all be unsafe
    out_of_goal_mask = torch.tensor(
        [
            [0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # too far in xz
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],  # too far in xz
            [0.0, 0.0, 0.0, 0.5, 0.1, 0.0],  # too fast in xz
            [0.0, 0.0, 0.0, 0.1, 0.5, 0.0],  # too fast in xz
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],  # too fast in theta dot
        ]
    )
    assert torch.all(torch.logical_not(dm.goal_mask_fn(out_of_goal_mask)))


def test_quad2d_obstacles_datamodule():
    """Test the custom DataModule for the 2D quadrotor with obstacles"""
    domains = [
        [
            (-4.0, 4.0),  # x
            (-4.0, 4.0),  # z
            (-np.pi, np.pi),  # theta
            (-10.0, 10.0),  # vx
            (-10.0, 10.0),  # vz
            (-np.pi, np.pi),  # roll
        ],
        [
            (-0.5, 0.5),
            (-0.5, 0.5),
            (-0.4 * np.pi, 0.4 * np.pi),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-np.pi, np.pi),
        ],
    ]
    N_samples = 1000
    split = 0.1
    dm = Quad2DObstaclesDataModule(N_samples=N_samples, domains=domains, split=split)
    assert dm is not None

    # After preparing data, there should be 2 * N_samples points
    dm.prepare_data()
    total_samples = len(domains) * N_samples
    assert len(dm.dataset) == total_samples

    # After setup, the data should be structured and split into DataLoaders
    dm.setup()
    # Make sure we have the right amount of data
    num_validation_pts = int(split * total_samples)
    num_train_pts = total_samples - num_validation_pts
    assert len(dm.train_data) == num_train_pts
    assert len(dm.validation_data) == num_validation_pts
    # Each of those things should have the appropriate number of items
    for data in dm.train_data:
        assert len(data) == 4
    for data in dm.validation_data:
        assert len(data) == 4


def test_quad2d_obstacles_datamodule_dataloaders():
    """Test the custom DataModule's DataLoaders for the 2D quadrotor with obstacles"""
    domains = [
        [
            (-4.0, 4.0),  # x
            (-4.0, 4.0),  # z
            (-np.pi, np.pi),  # theta
            (-10.0, 10.0),  # vx
            (-10.0, 10.0),  # vz
            (-np.pi, np.pi),  # roll
        ],
        [
            (-0.5, 0.5),
            (-0.5, 0.5),
            (-0.4 * np.pi, 0.4 * np.pi),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-np.pi, np.pi),
        ],
    ]
    N_samples = 1000
    split = 0.1
    batch_size = 10
    dm = Quad2DObstaclesDataModule(
        N_samples=N_samples, domains=domains, split=split, batch_size=batch_size
    )
    dm.prepare_data()
    dm.setup()

    # Make sure the data loaders are batched appropriately
    total_samples = len(domains) * N_samples
    train_dl = dm.train_dataloader()
    assert len(train_dl) == (total_samples - int(total_samples * split)) // batch_size
    val_dl = dm.val_dataloader()
    assert len(val_dl) == int(total_samples * split) // batch_size
