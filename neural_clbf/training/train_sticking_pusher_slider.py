from argparse import ArgumentParser
from copy import copy

import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    CarSCurveExperiment,
    PusherObstacleAvoidanceExperiment,
    CarSCurveExperiment2,
)
from neural_clbf.systems import StickingPusherSlider
from neural_clbf.training.utils import current_git_hash


# Setup
torch.multiprocessing.set_sharing_strategy("file_system")

# Simulation Setup
start_x = torch.tensor(
    [
        [-1.0, -1.0, 0.0],
        # [0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0, 0.0],
    ]
)
controller_period = 0.01
simulation_dt = 0.001


def main(args):
    # Define the dynamics model
    nominal_params = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
    }
    dynamics_model = StickingPusherSlider(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.1, -0.9),  # s_x
        (-1.1, -0.9),  # s_y
        (-0.1, 0.1),  # s_theta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=1,  # disable collecting data from trajectories
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = []
    s_x_ref_vals = [1.0, 1.5]
    s_y_ref_vals = [1.0, 1.5]
    for ref_index in range(len(s_x_ref_vals)):
        s = copy(nominal_params)
        s["s_x_ref"] = s_x_ref_vals[ref_index]
        s["s_y_ref"] = s_y_ref_vals[ref_index]

        scenarios.append(s)

    # Define the experiments
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=100,
        x_axis_index=StickingPusherSlider.S_X,
        y_axis_index=StickingPusherSlider.S_Y,
        x_axis_label="$x - x_{ref}$",
        y_axis_label="$y - y_{ref}$",
    )
    s_curve_experiment = PusherObstacleAvoidanceExperiment(
        "Reaching around pole",
        t_sim=7.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, s_curve_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        # primal_learning_rate=1e-3,
        # penalty_scheduling_rate=0,
        num_init_epochs=15,
        epochs_per_episode=200,  # disable new data-gathering
        barrier=False,  # disable fitting level sets to a safe/unsafe boundary
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/sticking-ps/", name=f"commit_{current_git_hash()}"
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=150,
        num_processes=0,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
