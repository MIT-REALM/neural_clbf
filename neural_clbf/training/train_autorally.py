from argparse import ArgumentParser
from copy import copy
import subprocess

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
)
from neural_clbf.systems import AutoRally


torch.multiprocessing.set_sharing_strategy("file_system")

controller_period = 0.01
simulation_dt = 0.01


def main(args):
    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 7.0,
        "omega_ref": 0.0,
    }
    dynamics_model = AutoRally(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = []
    upper_limit, lower_limit = dynamics_model.state_limits
    for ul, ll in zip(upper_limit, lower_limit):
        initial_conditions.append((0.2 * ll.item(), 0.2 * ul.item()))
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=500,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=512,
        # quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = []
    omega_ref_vals = [-1.5, 1.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.0, 1.0), (-1.0, 1.0)],
        n_grid=20,
        x_axis_index=AutoRally.SXE,
        y_axis_index=AutoRally.SYE,
        x_axis_label="$x - x_{ref}$",
        y_axis_label="$y - y_{ref}$",
        plot_unsafe_region=True,
        default_state=dynamics_model.goal_point,
    )
    s_curve_experiment = CarSCurveExperiment(
        "S-Curve Tracking",
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, s_curve_experiment])
    # experiment_suite = ExperimentSuite([])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=0.01,
        safe_level=0.5,
        controller_period=controller_period,
        clf_relaxation_penalty=1e3,
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=0,
        epochs_per_episode=1000,  # disable new data-gathering
        barrier=True,
        add_nominal=True,
        normalize_V_nominal=True,
    )

    # Initialize the logger and trainer
    current_git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/autorally/", name=f"commit_{current_git_hash}"
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=201
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
