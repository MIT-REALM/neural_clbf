from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import LinearSatellite
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    nominal_params = {
        "a": 6871,
        "ux_target": 0.0,
        "uy_target": 0.0,
        "uz_target": 0.0,
    }
    scenarios = [
        nominal_params,
    ]
    for ux in [-0.01, 0.01]:
        for uy in [-0.01, 0.01]:
            for uz in [-0.01, 0.01]:
                scenarios.append(
                    {
                        "a": 6871,
                        "ux_target": ux,
                        "uy_target": uy,
                        "uz_target": uz,
                    }
                )

    # Define the dynamics model
    dynamics_model = LinearSatellite(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-1.0, 1.0),  # z
        (-1.0, 1.0),  # xdot
        (-1.0, 1.0),  # ydot
        (-1.0, 1.0),  # zdot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=50000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.5, 1.5), (-1.5, 1.5)],
        n_grid=50,
        x_axis_index=LinearSatellite.X,
        y_axis_index=LinearSatellite.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_state_space_experiment = RolloutStateSpaceExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=LinearSatellite.X,
        plot_x_label="$x$",
        plot_y_index=LinearSatellite.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=1,
        t_sim=10.0,
    )
    experiment_suite = ExperimentSuite(
        [
            V_contour_experiment,
            rollout_state_space_experiment,
        ]
    )

    # Initialize the controller
    clbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=4,
        cbf_hidden_size=128,
        cbf_lambda=1.0,
        controller_period=controller_period,
        cbf_relaxation_penalty=1e2,
        scale_parameter=10.0,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/linear_satellite_cbf",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=25
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
