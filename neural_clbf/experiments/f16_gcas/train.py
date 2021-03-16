from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralrCLBFController
from neural_clbf.experiments.f16_gcas.data_generation import (
    F16GcasDataModule,
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
    0.0,  # integrator state 1
    0.0,  # integrator state 1
]
start_x = torch.tensor([init])


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=start_x,
        plot_x_indices=[F16.H],
        plot_x_labels=["$h$"],
        plot_u_indices=[F16.U_NZ, F16.U_SR],
        plot_u_labels=["$N_z$", "$SR$"],
        t_sim=10.0,
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
    # Initialize the DataModule
    data_module = F16GcasDataModule(N_samples=500000, split=0.1, batch_size=256)

    # ## Setup trainer parameters ##
    # Define the dynamics model
    nominal_params = {"lag_error": 0.0}
    dynamics_model = F16(nominal_params)

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
    rclbf_controller = NeuralrCLBFController(
        dynamics_model,
        scenarios,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=1,
        clbf_hidden_size=8,
        qp_clbf_relaxation_penalty=1e6,
        learning_rate=1e-3,
        x_center=data_module.x_center,
        x_range=data_module.x_range,
    )
    # Add the DataModule hooks
    rclbf_controller.prepare_data = data_module.prepare_data
    rclbf_controller.setup = data_module.setup
    rclbf_controller.train_dataloader = data_module.train_dataloader
    rclbf_controller.val_dataloader = data_module.val_dataloader
    rclbf_controller.test_dataloader = data_module.test_dataloader

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger("logs/f16_gcas/")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = tb_logger

    # Train
    trainer.fit(rclbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
