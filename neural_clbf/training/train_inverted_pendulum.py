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
from neural_clbf.systems import InvertedPendulum
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [0.5, 0.5],
        # [-0.2, 1.0],
        # [0.2, -1.0],
        # [-0.2, -1.0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    scenarios = [
        nominal_params,
        {"m": 1.25, "L": 1.0, "b": 0.01},
        {"m": 1.0, "L": 1.25, "b": 0.01},
        {"m": 1.25, "L": 1.25, "b": 0.01},
    ]

    # Define the dynamics model
    dynamics_model = InvertedPendulum(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-np.pi / 2, np.pi / 2),  # theta
        (-1.0, 1.0),  # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=20,
        x_axis_index=InvertedPendulum.THETA,
        y_axis_index=InvertedPendulum.THETA_DOT,
        x_axis_label="$\\theta$",
        y_axis_label="$\\dot{\\theta}$",
    )
    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[InvertedPendulum.THETA, InvertedPendulum.THETA_DOT],
        plot_x_labels=["$\\theta$", "$\\dot{\\theta}$"],
        plot_u_indices=[],
        plot_u_labels=[],
        scenarios=scenarios,
        n_sims_per_start=2,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e3,
        num_init_epochs=5,
        epochs_per_episode=100,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/inverted_pendulum",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=21,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
