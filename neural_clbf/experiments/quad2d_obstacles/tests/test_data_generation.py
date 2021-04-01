"""Test the data generation for the 2D quadrotor with obstacles"""
import numpy as np

from neural_clbf.experiments.quad2d_obstacles.data_generation import (
    Quad2DObstaclesDataModule,
)


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
