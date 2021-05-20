from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments.common.plotting import (
    plot_CLBF,
    rollout_CLBF,
)
from neural_clbf.systems import InvertedPendulum


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01
clbf_lambda = 0.5

start_x = torch.tensor(
    [
        [0.5, 0.5],
        # [-0.2, 1.0],
        # [0.2, -1.0],
        # [-0.2, -1.0],
    ]
)
simulation_dt = 0.001


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=start_x,
        # plot_x_indices=[InvertedPendulum.THETA, InvertedPendulum.THETA_DOT],
        plot_x_indices=[InvertedPendulum.THETA],
        plot_x_labels=["$\\theta$"],
        plot_u_indices=[InvertedPendulum.U],
        plot_u_labels=["$u$"],
        t_sim=6.0,
        n_sims_per_start=5,
        controller_period=controller_period,
        goal_check_fn=clbf_net.dynamics_model.goal_mask,
        out_of_bounds_check_fn=clbf_net.dynamics_model.out_of_bounds_mask,
    )


def clbf_plotting_cb(clbf_net):
    return plot_CLBF(
        clbf_net,
        domain=[(-2.0, 2.0), (-2.0, 2.0)],  # plot for theta, theta_dot
        n_grid=15,
        x_axis_index=InvertedPendulum.THETA,
        y_axis_index=InvertedPendulum.THETA_DOT,
        x_axis_label="$\\theta$",
        y_axis_label="$\\dot{\\theta}$",
    )


def main(args):
    # Define the dynamics model
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    dynamics_model = InvertedPendulum(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-np.pi / 2, np.pi / 2),  # theta
        (-1.0, 1.0),  # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=2000,
        trajectory_length=10,
        # fixed_samples=10000,
        fixed_samples=0,
        max_points=100000,
        val_split=0.1,
        batch_size=batch_size,
        quotas={"safe": 0.2, "goal": 0.2, "unsafe": 0.2},
    )

    # Define the scenarios
    scenarios = [
        nominal_params,
        {"m": 1.25, "L": 1.0, "b": 0.01},
        {"m": 1.0, "L": 1.25, "b": 0.01},
        {"m": 1.25, "L": 1.25, "b": 0.01},
    ]

    # Define the plotting callbacks
    plotting_callbacks = [
        # This plotting function plots V and dV/dt violation on a grid
        clbf_plotting_cb,
        # This plotting function simulates rollouts of the controller
        rollout_plotting_cb,
    ]

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=2,
        clbf_hidden_size=256,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=256,
        clbf_lambda=clbf_lambda,
        controller_period=controller_period,
        clbf_relaxation_penalty=1e5,
        # clbf_relaxation_penalty=2e6,
        epochs_per_episode=10,
    )
    # Add the DataModule hooks
    clbf_controller.prepare_data = data_module.prepare_data
    clbf_controller.setup = data_module.setup
    clbf_controller.train_dataloader = data_module.train_dataloader
    clbf_controller.val_dataloader = data_module.val_dataloader
    clbf_controller.test_dataloader = data_module.test_dataloader

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/basic_experiments",
        name="sim_only",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
