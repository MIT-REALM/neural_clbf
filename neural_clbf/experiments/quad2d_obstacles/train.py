from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralrCLBFController
from neural_clbf.experiments.quad2d_obstacles.data_generation import (
    Quad2DObstaclesDataModule,
)
from neural_clbf.experiments.common.plotting import (
    plot_CLBF,
    rollout_CLBF,
)
from neural_clbf.systems import Quad2D


torch.multiprocessing.set_sharing_strategy("file_system")


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=torch.tensor([[-1.5, 0.1, 0.0, 0.0, 0.0, 0.0]]),
        plot_x_indices=[Quad2D.PX, Quad2D.PZ],
        plot_x_labels=["$x$", "$z$"],
        plot_u_indices=[Quad2D.U_RIGHT, Quad2D.U_LEFT],
        plot_u_labels=["$u_r$", "$u_l$"],
        t_sim=1.0,
        n_sims_per_start=1,
    )


def clbf_plotting_cb(clbf_net):
    return plot_CLBF(
        clbf_net,
        domain=[(-2.0, 1.0), (-0.5, 1.5)],  # plot for x, z in [-2, 1], [-0.5, 1.5]
        n_grid=25,
        x_axis_index=Quad2D.PX,
        y_axis_index=Quad2D.PZ,
        x_axis_label="$x$",
        y_axis_label="$z$",
    )


def main(args):
    # Initialize the DataModule
    data_module = Quad2DObstaclesDataModule(
        N_samples=1000000, split=0.1, batch_size=256
    )

    # ## Setup trainer parameters ##
    # Define the dynamics model
    nominal_params = {"m": 1.0, "I": 0.001, "r": 0.25}
    dynamics_model = Quad2D(nominal_params)

    # Define the scenarios
    scenarios = [
        nominal_params,
        {"m": 1.0, "I": 0.00105, "r": 0.25},
        {"m": 1.05, "I": 0.001, "r": 0.25},
        {"m": 1.05, "I": 0.00105, "r": 0.25},
    ]

    # Define the plotting callbacks
    plotting_callbacks = [
        # This plotting function plots V and dV/dt violation on a grid
        clbf_plotting_cb,
        # This plotting function simulates rollouts of the controller
        rollout_plotting_cb,
    ]

    # Initialize the controller
    rclbf_controller = NeuralrCLBFController(
        dynamics_model,
        scenarios,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=5,
        clbf_hidden_size=48,
        learning_rate=1e-3,
    )
    # Add the DataModule hooks
    rclbf_controller.prepare_data = data_module.prepare_data
    rclbf_controller.setup = data_module.setup
    rclbf_controller.train_dataloader = data_module.train_dataloader
    rclbf_controller.val_dataloader = data_module.val_dataloader
    rclbf_controller.test_dataloader = data_module.test_dataloader

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger("logs/quad2d_obstacles/")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = tb_logger

    # Train
    trainer.fit(rclbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
