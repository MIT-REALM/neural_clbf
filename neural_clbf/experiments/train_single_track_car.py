from argparse import ArgumentParser
from copy import copy

import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments.common.plotting import (
    plot_CLBF,
    rollout_CLBF,
)
from neural_clbf.experiments.sim_single_track_car_controller import (
    # single_rollout_straight_path,
    # single_rollout_circle_path,
    single_rollout_s_path,
)
from neural_clbf.systems import STCar


torch.multiprocessing.set_sharing_strategy("file_system")

start_x = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, -np.pi / 6, 0.0, 0.0],
        # [0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0, 0.0],
    ]
)
controller_period = 0.01
simulation_dt = 0.001


def rollout_plotting_cb(clbf_net):
    return rollout_CLBF(
        clbf_net,
        start_x=start_x,
        plot_x_indices=[STCar.SXE, STCar.SYE],
        plot_x_labels=["$x - x_{ref}$", "$y - y_{ref}$"],
        plot_u_indices=[STCar.VDELTA, STCar.ALONG],
        plot_u_labels=["$v_\\delta$", "$a_{long}$"],
        t_sim=6.0,
        n_sims_per_start=1,
        controller_period=controller_period,
        goal_check_fn=clbf_net.dynamics_model.goal_mask,
        # out_of_bounds_check_fn=clbf_net.dynamics_model.out_of_bounds_mask,
    )


def clbf_plotting_cb(clbf_net):
    return plot_CLBF(
        clbf_net,
        domain=[(-1.0, 1.0), (-1.0, 1.0)],
        n_grid=50,
        x_axis_index=STCar.SXE,
        y_axis_index=STCar.SYE,
        x_axis_label="$x - x_{ref}$",
        y_axis_label="$y - y_{ref}$",
    )


def main(args):
    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    dynamics_model = STCar(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.1, 0.1),  # sxe
        (-0.1, 0.1),  # sye
        (-0.1, 0.1),  # delta
        (-0.1, 0.1),  # ve
        (-0.1, 0.1),  # psi_e
        (-0.1, 0.1),  # psi_dot
        (-0.1, 0.1),  # beta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=10,
        trajectory_length=1000,
        fixed_samples=10000,
        max_points=5000000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = []
    omega_ref_vals = [-0.5, 0.5]
    # omega_ref_vals = [0.0]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    # Define the plotting callbacks
    plotting_callbacks = [
        # This plotting function plots V and dV/dt violation on a grid
        clbf_plotting_cb,
        # # This plotting function simulates rollouts of the controller
        # rollout_plotting_cb,
        # # Plot some rollouts
        # single_rollout_straight_path,
        # single_rollout_circle_path,
        single_rollout_s_path,
    ]

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        plotting_callbacks=plotting_callbacks,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clbf_lambda=0.1,
        safety_level=0.1,
        goal_level=0.0,
        controller_period=controller_period,
        clbf_relaxation_penalty=1e6,
        penalty_scheduling_rate=50,
        num_init_epochs=5,
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
        "logs/stcar_basic/",
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
