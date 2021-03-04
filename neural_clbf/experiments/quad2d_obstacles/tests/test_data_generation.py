"""Test the data generation for the 2D quadrotor with obstacles"""
import torch
import numpy as np

from neural_clbf.experiments.quad2d_obstacles.data_generation import (
    safe_mask_fn,
    unsafe_mask_fn,
    Quad2DObstaclesDataModule,
)


def test_quad2d_obstacles_safe_mask():
    """Test the safe mask for the 2D quadrotor with obstacles"""
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
    assert torch.all(safe_mask_fn(safe_x))

    # These points should all be unsafe
    unsafe_x = torch.tensor(
        [
            [0.0, -0.4, 0.0, 0.0, 0.0, 0.0],  # too low
            [-0.7, 0.5, 0.0, 0.0, 0.0, 0.0],  # inside obstacle 1
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],  # inside obstacle 2
        ]
    )
    assert torch.all(unsafe_mask_fn(unsafe_x))


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
    # Each DataLoader should have three things in it (x, safe_mask, and unsafe_mask)
    assert len(dm.train_data) == 3
    assert len(dm.validation_data) == 3
    # Each of those things should have the appropriate split length
    num_validation_pts = int(split * total_samples)
    num_train_pts = total_samples - num_validation_pts
    for data in dm.train_data:
        assert len(data) == num_train_pts
    for data in dm.validation_data:
        assert len(data) == num_validation_pts
