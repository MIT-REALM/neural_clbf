from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import CLBFDDPGController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments.common.plotting import (
    plot_CLBF,
    rollout_CLBF,
)
from neural_clbf.systems import Quad2D


torch.multiprocessing.set_sharing_strategy("file_system")

start_x = torch.tensor([[-0.75, 0.75, 0.0, 0.0, 0.0, 0.0]])
controller_period = 0.01
simulation_dt = 0.001


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=start_x,
        plot_x_indices=[Quad2D.PX, Quad2D.PZ],
        plot_x_labels=["$x$", "$z$"],
        plot_u_indices=[Quad2D.U_RIGHT, Quad2D.U_LEFT],
        plot_u_labels=["$u_r$", "$u_l$"],
        t_sim=6.0,
        n_sims_per_start=1,
        controller_period=controller_period,
        goal_check_fn=clbf_net.dynamics_model.goal_mask,
        out_of_bounds_check_fn=clbf_net.dynamics_model.out_of_bounds_mask,
    )


def clbf_plotting_cb(clbf_net):
    return plot_CLBF(
        clbf_net,
        domain=[(-1.0, 1.0), (-0.5, 1.0)],  # plot for x, z in [-2, 1], [-0.5, 1.5]
        n_grid=15,
        x_axis_index=Quad2D.PX,
        y_axis_index=Quad2D.PZ,
        x_axis_label="$x$",
        y_axis_label="$z$",
    )


def main(args):
    # Define the dynamics model
    nominal_params = {"m": 1.0, "I": 0.001, "r": 0.25}
    dynamics_model = Quad2D(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.75, 0.75),  # x
        (0.0, 1.0),  # z
        (-np.pi / 4, np.pi / 4),  # theta
        (-1.0, 1.0),  # vx
        (-1.0, 1.0),  # vz
        (-1.0, 1.0),  # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=1,
        fixed_samples=1000000,
        max_points=10000000,
        val_split=0.1,
        batch_size=1024,
    )

    # Define the scenarios
    scenarios = [
        nominal_params,
    ]

    # Define the plotting callbacks
    plotting_callbacks = [
        # This plotting function plots V and dV/dt violation on a grid
        clbf_plotting_cb,
        # This plotting function simulates rollouts of the controller
        rollout_plotting_cb,
    ]

    # Initialize the controller
    clbf_controller = CLBFDDPGController(
        dynamics_model,
        scenarios,
        data_module,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=4,
        clbf_hidden_size=64,
        u_nn_hidden_layers=4,
        u_nn_hidden_size=64,
        f_nn_hidden_layers=4,
        f_nn_hidden_size=64,
        discrete_timestep=controller_period,
        primal_learning_rate=1e-3,
        polyak=0.995,
        epochs_per_episode=100,
    )
    # Add the DataModule hooks
    clbf_controller.prepare_data = data_module.prepare_data
    clbf_controller.setup = data_module.setup
    clbf_controller.train_dataloader = data_module.train_dataloader
    clbf_controller.val_dataloader = data_module.val_dataloader
    clbf_controller.test_dataloader = data_module.test_dataloader

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/quad2d_obstacles/",
        name="DDPG",
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
