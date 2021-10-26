from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.systems import Quad2D
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

start_x = torch.tensor([[-0.75, 0.75, 0.0, 0.0, 0.0, 0.0]])
controller_period = 0.01
simulation_dt = 0.001


def main(args):
    # Define the dynamics model
    nominal_params = {"m": 1.0, "I": 0.01, "r": 0.25}
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
        fixed_samples=5000,
        max_points=1000000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = [
        nominal_params,
        {"m": 1.05, "I": 0.01, "r": 0.25},
        {"m": 1.0, "I": 0.0105, "r": 0.25},
        {"m": 1.05, "I": 0.0105, "r": 0.25},
    ]

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.0, 1.0), (-0.5, 1.0)],
        n_grid=15,
        x_axis_index=Quad2D.PX,
        y_axis_index=Quad2D.PZ,
        x_axis_label="$x$",
        y_axis_label="$z$",
    )
    rollout_ts_experiment = RolloutTimeSeriesExperiment(
        "Rollout (time series)",
        start_x,
        plot_x_indices=[Quad2D.PX, Quad2D.PZ],
        plot_x_labels=["$x$", "$z$"],
        plot_u_indices=[Quad2D.U_RIGHT, Quad2D.U_LEFT],
        plot_u_labels=["$u_r$", "$u_l$"],
        t_sim=6.0,
        n_sims_per_start=1,
    )
    rollout_ss_experiment = RolloutStateSpaceExperiment(
        "Rollout (time series)",
        start_x,
        plot_x_index=Quad2D.PX,
        plot_x_label="$x$",
        plot_y_index=Quad2D.PZ,
        plot_y_label="$z$",
        t_sim=6.0,
        n_sims_per_start=1,
    )
    experiment_suite = ExperimentSuite(
        [V_contour_experiment, rollout_ts_experiment, rollout_ss_experiment]
    )

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=3,
        clbf_hidden_size=32,
        controller_period=controller_period,
        clf_relaxation_penalty=50.0,
        epochs_per_episode=1000,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/quad2d_obstacles/",
        name=f"commit_{current_git_hash()}",
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
