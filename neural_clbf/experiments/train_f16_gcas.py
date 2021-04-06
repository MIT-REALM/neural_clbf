from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralSIDCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments.common.plotting import (
    plot_CLBF,
    rollout_CLBF,
)
from neural_clbf.systems import F16


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


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=start_x,
        plot_x_indices=[F16.H],
        plot_x_labels=["$h$"],
        plot_u_indices=[F16.U_NZ, F16.U_SR],
        plot_u_labels=["$N_z$", "$SR$"],
        t_sim=10.0,
        controller_period=controller_period,
        n_sims_per_start=1,
    )


def clbf_plotting_cb(clbf_net):
    return plot_CLBF(
        clbf_net,
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
    # initial_conditions = [
    #     (400.0, 700.0),  # vt
    #     (0.035, 0.035),  # alpha
    #     (0.0, 0.0),  # beta
    #     (-np.pi / 8, -np.pi / 8),  # phi
    #     (-0.15 * np.pi, -0.15 * np.pi),  # theta
    #     (0.0, 0.0),  # psi
    #     (0.0, 0.0),  # P
    #     (0.0, 0.0),  # Q
    #     (0.0, 0.0),  # R
    #     (0.0, 0.0),  # PN
    #     (0.0, 0.0),  # PE
    #     (1000.0, 1000.0),  # H
    #     (9.0, 9.0),  # pow
    #     (0.0, 0.0),  # integrator state 1
    #     (0.0, 0.0),  # integrator state 2
    #     (0.0, 0.0),  # integrator state 3
    # ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=5000,
        val_split=0.1,
        batch_size=256,
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
    clbf_controller = NeuralSIDCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=3,
        clbf_hidden_size=32,
        u_nn_hidden_layers=3,
        u_nn_hidden_size=32,
        f_nn_hidden_layers=3,
        f_nn_hidden_size=32,
        dynamics_timestep=controller_period,
        primal_learning_rate=1e-3,
        dual_learning_rate=1e-3,
        epochs_per_episode=5,
    )
    # Add the DataModule hooks
    clbf_controller.prepare_data = data_module.prepare_data
    clbf_controller.setup = data_module.setup
    clbf_controller.train_dataloader = data_module.train_dataloader
    clbf_controller.val_dataloader = data_module.val_dataloader
    clbf_controller.test_dataloader = data_module.test_dataloader

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/f16_gcas/",
        name="episodic",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True
    )

    # Train
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
