"""Test the data generation for the f16 gcas"""
import random
from typing import Dict

import torch

from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.systems.tests.mock_system import MockSystem


params: Dict[str, float] = {}
model = MockSystem(params)


def test_episodic_datamodule():
    """Test the EpisodicDataModule"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(
        model,
        initial_domain,
        trajectories_per_episode=100,
        trajectory_length=50,
        fixed_samples=1000,
        val_split=0.1,
        batch_size=10,
    )
    assert dm is not None

    # After preparing data, there should be a bunch of sample points
    dm.prepare_data()
    expected_num_datapoints = dm.trajectories_per_episode * dm.trajectory_length
    expected_num_datapoints += dm.fixed_samples
    val_pts = int(expected_num_datapoints * dm.val_split)
    train_pts = expected_num_datapoints - val_pts
    assert dm.x_training.shape[0] == train_pts
    assert dm.x_training.shape[1] == model.n_dims
    assert dm.x_validation.shape[0] == val_pts
    assert dm.x_validation.shape[1] == model.n_dims

    # These points should also be located in DataLoaders
    assert len(dm.training_data) == train_pts
    assert len(dm.validation_data) == val_pts
    # Each of those things should have the appropriate number of items
    # (point, goal, safe, unsafe)
    for data in dm.training_data:
        assert len(data) == 4
    for data in dm.validation_data:
        assert len(data) == 4

    # Also make sure the data loaders are batched appropriately
    train_dl = dm.train_dataloader()
    assert len(train_dl) == round(train_pts / dm.batch_size)
    val_dl = dm.val_dataloader()
    assert len(val_dl) == round(val_pts / dm.batch_size)


def test_episodic_datamodule_quotas():
    """Test the EpisodicDataModule with sampling quotas"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(
        model,
        initial_domain,
        trajectories_per_episode=100,
        trajectory_length=50,
        fixed_samples=1000,
        val_split=0.1,
        batch_size=10,
        quotas={"safe": 0.1, "unsafe": 0.1, "goal": 0.1},
    )
    assert dm is not None

    # After preparing data, there should be a bunch of sample points
    dm.prepare_data()
    expected_num_datapoints = dm.trajectories_per_episode * dm.trajectory_length
    expected_num_datapoints += dm.fixed_samples
    val_pts = int(expected_num_datapoints * dm.val_split)
    train_pts = expected_num_datapoints - val_pts
    assert dm.x_training.shape[0] == train_pts
    assert dm.x_training.shape[1] == model.n_dims
    assert dm.x_validation.shape[0] == val_pts
    assert dm.x_validation.shape[1] == model.n_dims

    # These points should also be located in DataLoaders
    assert len(dm.training_data) == train_pts
    assert len(dm.validation_data) == val_pts
    # Each of those things should have the appropriate number of items
    # (point, goal, safe, unsafe)
    for data in dm.training_data:
        assert len(data) == 4
    for data in dm.validation_data:
        assert len(data) == 4

    # Also make sure the data loaders are batched appropriately
    train_dl = dm.train_dataloader()
    assert len(train_dl) == round(train_pts / dm.batch_size)
    val_dl = dm.val_dataloader()
    assert len(val_dl) == round(val_pts / dm.batch_size)
