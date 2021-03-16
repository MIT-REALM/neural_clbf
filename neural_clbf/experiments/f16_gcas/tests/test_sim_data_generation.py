"""Test the data generation for the f16 gcas"""
import torch
import numpy as np

from neural_clbf.experiments.f16_gcas.sim_data_generation import (
    F16GcasSimDataModule,
)
from neural_clbf.systems.f16 import F16


nominal_params = {"lag_error": 0.0}
f16_model = F16(nominal_params)


def test_f16_safe_unsafe_mask():
    """Test the safe and unsafe mask for the F16"""
    dm = F16GcasSimDataModule(f16_model)
    # This point should be safe
    safe_x = torch.tensor(
        [
            [
                540.0,
                0.035,
                0.0,
                -np.pi / 8,
                -0.15 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1000.0,
                9.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    assert torch.all(dm.safe_mask_fn(safe_x))

    # Thise point should be unsafe
    unsafe_x = torch.tensor(
        [
            [
                540.0,
                0.035,
                0.0,
                -np.pi / 8,
                -0.15 * np.pi,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                100.0,
                9.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    assert torch.all(dm.unsafe_mask_fn(unsafe_x))


def test_f16_datamodule():
    """Test the custom DataModule for the f16"""
    initial_domains = [
        [
            (400, 600),  # vt
            (-0.1, 0.1),  # alpha
            (-0.1, 0.1),  # beta
            (-0.1, 0.1),  # phi
            (-0.1, 0.1),  # theta
            (-0.1, 0.1),  # psi
            (-0.5, 0.5),  # P
            (-0.5, 0.5),  # Q
            (-0.5, 0.5),  # R
            (-100, 100),  # pos_n
            (-100, 100),  # pos_e
            (100, 1500),  # alt
            (0, 10),  # pow
            (-0.0, 0.0),  # nz_int
            (-0.0, 0.0),  # ps_int
            (-0.0, 0.0),  # nyr_int
        ],
    ]
    N_samples = 1000
    sim_steps = 10
    split = 0.1
    dm = F16GcasSimDataModule(
        f16_model,
        N_samples=N_samples,
        sim_steps=sim_steps,
        initial_domains=initial_domains,
        val_split=split,
    )
    assert dm is not None

    # After preparing data, there should be 2 * N_sample points
    dm.prepare_data()
    total_samples = len(initial_domains) * N_samples
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


def test_f16_datamodule_dataloaders():
    """Test the custom DataModule's DataLoaders for the f16 gcas"""
    initial_domains = [
        [
            (400, 600),  # vt
            (-0.1, 0.1),  # alpha
            (-0.1, 0.1),  # beta
            (-0.1, 0.1),  # phi
            (-0.1, 0.1),  # theta
            (-0.1, 0.1),  # psi
            (-0.5, 0.5),  # P
            (-0.5, 0.5),  # Q
            (-0.5, 0.5),  # R
            (-100, 100),  # pos_n
            (-100, 100),  # pos_e
            (100, 1500),  # alt
            (0, 10),  # pow
            (-0.0, 0.0),  # nz_int
            (-0.0, 0.0),  # ps_int
            (-0.0, 0.0),  # nyr_int
        ],
    ]
    N_samples = 1000
    sim_steps = 10
    split = 0.1
    batch_size = 10
    dm = F16GcasSimDataModule(
        f16_model,
        N_samples=N_samples,
        sim_steps=sim_steps,
        initial_domains=initial_domains,
        val_split=split,
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup()

    # Make sure the data loaders are batched appropriately
    total_samples = len(initial_domains) * N_samples
    train_dl = dm.train_dataloader()
    assert len(train_dl) == (total_samples - int(total_samples * split)) // batch_size
    val_dl = dm.val_dataloader()
    assert len(val_dl) == int(total_samples * split) // batch_size
