"""Test the data generation for the 2D quadrotor with obstacles"""
import torch
import numpy as np

from neural_clbf.experiments.f16_gcas.data_generation import (
    F16GcasDataModule,
)


def test_f16_safe_unsafe_mask():
    """Test the safe and unsafe mask for the F16"""
    dm = F16GcasDataModule()
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


# def test_f16_goal_mask():
#     """Test the goal mask for the F16"""
#     dm = F16GcasDataModule()
#     # These points should all be in the goal
#     in_goal = torch.tensor(
#         [
#             # TODO
#         ]
#     )
#     assert torch.all(dm.goal_mask_fn(in_goal))

#     # These points should all be unsafe
#     out_of_goal_mask = torch.tensor(
#         [
#             # TODO
#         ]
#     )
#     assert torch.all(torch.logical_not(dm.goal_mask_fn(out_of_goal_mask)))


def test_f16_datamodule():
    """Test the custom DataModule for the 2D quadrotor with obstacles"""
    domains = [
        [
            (400, 600),  # vt
            (-np.pi, np.pi),  # alpha
            (-np.pi, np.pi),  # beta
            (-np.pi, np.pi),  # phi
            (-np.pi, np.pi),  # theta
            (-np.pi, np.pi),  # psi
            (-2 * np.pi, 2 * np.pi),  # P
            (-2 * np.pi, 2 * np.pi),  # Q
            (-2 * np.pi, 2 * np.pi),  # R
            (-1000, 1000),  # pos_n
            (-1000, 1000),  # pos_e
            (-500, 1500),  # alt
            (0, 10),  # pow
            (-20, 20),  # nz_int
            (-20, 20),  # ps_int
            (-20, 20),  # nyr_int
        ],
    ]
    N_samples = 1000
    split = 0.1
    dm = F16GcasDataModule(N_samples=N_samples, domains=domains, split=split)
    assert dm is not None

    # And the center point and range should be set correctly
    x_center = torch.tensor(
        [
            500.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            500.0,
            5.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    x_range = torch.tensor(
        [
            600 - 400,
            np.pi + np.pi,
            np.pi + np.pi,
            np.pi + np.pi,
            np.pi + np.pi,
            np.pi + np.pi,
            2 * np.pi + 2 * np.pi,
            2 * np.pi + 2 * np.pi,
            2 * np.pi + 2 * np.pi,
            1000 + 1000,
            1000 + 1000,
            1500 + 500,
            10 - 0,
            20 + 20,
            20 + 20,
            20 + 20,
        ]
    )
    assert torch.allclose(x_center, dm.x_center)
    assert torch.allclose(x_range, dm.x_range)

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


def test_f16_datamodule_dataloaders():
    """Test the custom DataModule's DataLoaders for the 2D quadrotor with obstacles"""
    domains = [
        [
            (400, 600),  # vt
            (-np.pi, np.pi),  # alpha
            (-np.pi, np.pi),  # beta
            (-np.pi, np.pi),  # phi
            (-np.pi, np.pi),  # theta
            (-np.pi, np.pi),  # psi
            (-2 * np.pi, 2 * np.pi),  # P
            (-2 * np.pi, 2 * np.pi),  # Q
            (-2 * np.pi, 2 * np.pi),  # R
            (-1000, 1000),  # pos_n
            (-1000, 1000),  # pos_e
            (-500, 1500),  # alt
            (0, 10),  # pow
            (-20, 20),  # nz_int
            (-20, 20),  # ps_int
            (-20, 20),  # nyr_int
        ],
    ]
    N_samples = 1000
    split = 0.1
    batch_size = 10
    dm = F16GcasDataModule(
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
