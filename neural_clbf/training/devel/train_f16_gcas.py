from warnings import warn
from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)

from neural_clbf.training.utils import current_git_hash


imported_F16 = False
try:
    from neural_clbf.systems import F16

    imported_F16 = True
except ImportError:
    warn("Could not import F16 module")

torch.multiprocessing.set_sharing_strategy("file_system")


init = [
    540.0,  # vt
    0.035,  # alpha
    0.0,  # beta
    -np.pi / 8,  # phi
    -0.15 * np.pi,  # theta
    0.0,  # psi
    0.0,  # P
    0.0,  # Q
    0.0,  # R
    0.0,  # PN
    0.0,  # PE
    1000.0,  # H
    9.0,  # pow
    0.0,  # integrator state 1
    0.0,  # integrator state 2
    0.0,  # integrator state 3
]
start_x = torch.tensor([init])

controller_period = 0.01


def main(args):
    # Define the dynamics model
    nominal_params = {"lag_error": 0.0}
    dynamics_model = F16(nominal_params, dt=controller_period)

    # Initialize the DataModule
    initial_conditions = [
        (500.0, 700.0),  # vt
        (-0.1, 0.1),  # alpha
        (-0.1, 0.1),  # beta
        (-np.pi / 4, np.pi / 4),  # phi
        (-np.pi / 4, 0.0),  # theta
        (-np.pi / 4, np.pi / 4),  # psi
        (-5.0, 5.0),  # P
        (-5.0, 5.0),  # Q
        (-5.0, 5.0),  # R
        (-100.0, 100.0),  # PN
        (-100.0, 100.0),  # PE
        (500.0, 1000.0),  # H
        (1.0, 9.0),  # pow
        (0.0, 0.0),  # integrator state 1
        (0.0, 0.0),  # integrator state 2
        (0.0, 0.0),  # integrator state 3
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=100000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    # Define the scenarios
    scenarios = [
        nominal_params,
    ]

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(400.0, 600.0), (0.0, 1500)],
        default_state=torch.tensor(
            [[500.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 5, 0, 0, 0]]
        ),
        n_grid=20,
        x_axis_index=F16.VT,
        y_axis_index=F16.H,
        x_axis_label="$v$",
        y_axis_label="$h$",
    )
    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[F16.H],
        plot_x_labels=["$h$"],
        plot_u_indices=[F16.U_NZ, F16.U_SR],
        plot_u_labels=["$N_z$", "$SR$"],
        t_sim=10.0,
        n_sims_per_start=1,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

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
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/f16_gcas/", name=f"commit_{current_git_hash()}"
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True
    )

    # Train
    trainer.fit(clbf_controller)


if __name__ == "__main__" and imported_F16:
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
