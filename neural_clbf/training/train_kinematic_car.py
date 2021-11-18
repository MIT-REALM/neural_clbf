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
)
from neural_clbf.systems import KSCar
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

start_x = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, -np.pi / 6, 0.0, 0.0],
        # [0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0, 0.0],
    ]
)
controller_period = 0.01
simulation_dt = 0.001


def main(args):
    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dynamics_model = KSCar(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.1, 0.1),  # sxe
        (-0.1, 0.1),  # sye
        (-0.1, 0.1),  # delta
        (-0.1, 0.1),  # ve
        (-0.1, 0.1),  # psi_e
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
    omega_ref_vals = [-1.5, 1.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    # Define the experiments
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=50,
        x_axis_index=KSCar.SXE,
        y_axis_index=KSCar.SYE,
        x_axis_label="$x - x_{ref}$",
        y_axis_label="$y - y_{ref}$",
    )
    s_curve_experiment = CarSCurveExperiment(
        "S-Curve Tracking",
        t_sim=5.0,
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
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=11,
        epochs_per_episode=200,  # disable new data-gathering
        barrier=False,  # disable fitting level sets to a safe/unsafe boundary
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/kscar/", name=f"commit_{current_git_hash()}"
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=26,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
