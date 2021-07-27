from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)
from neural_clbf.systems import Crazyflie


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

#TODO @dylan look at implementing unit tests. see other files for examples
start_x = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    # mass of crazyflie is 28 grams = 0.028 kg
    nominal_params = {"m": 0.028}
    scenarios = [
        nominal_params,
        # can list some different mass values here to test different scenarios
    ]

    # Define the dynamics model
    dynamics_model = Crazyflie(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    
    # Initialize the DataModule
    initial_conditions = [
        (-2.0, 2.0),  # x
        (-2.0, 2.0),  # y
        (-4.0, 0), # z
        (-8.0, 8.0),  # vx
        (-8.0, 8.0),  # vy
        (-8.0, 8.0),  # vz
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=500,
        fixed_samples=20000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )
    
    # Define the experiment suite
    # TODO @dylan look at clf_contour_experiment.py for more info; change y to z; maybe add slices for y as input argument
    V_contour_experiment_xy = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=50,
        x_axis_index=Crazyflie.X,
        y_axis_index=Crazyflie.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    
    V_contour_experiment_xz = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=50,
        x_axis_index=Crazyflie.X
        y_axis_index=Crazyflie.Z,
        x_axis_label="$x",
        y_axis_label="$z$",
    )
    
    V_contour_experiment_yz = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=50,
        x_axis_index=Crazyflie.Y
        y_axis_index=Crazyflie.Z,
        x_axis_label="$y",
        y_axis_label="$z$",
    )
    
    # Three plots in XZ with force and each angle plotted for inputs
    rollout_experiment_xz_force_phi = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.X Crazyflie.Z]
        plot_x_labels=["$x$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.PHI]
        plot_u_labels=["$F$", "$Phi$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
                       
    rollout_experiment_xz_force_theta = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.X Crazyflie.Z]
        plot_x_labels=["$x$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.THETA]
        plot_u_labels=["$F$", "$Theta$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
                       
    rollout_experiment_xz_force_psi = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.X Crazyflie.Z]
        plot_x_labels=["$x$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.PSI]
        plot_u_labels=["$F$", "$Psi$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
                       
    # Three plots in YZ with force and each angle plotted for inputs
    rollout_experiment_yz_force_phi = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.Y Crazyflie.Z]
        plot_x_labels=["$y$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.PHI]
        plot_u_labels=["$F$", "$Phi$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
                       
    rollout_experiment_yz_force_theta = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.Y Crazyflie.Z]
        plot_x_labels=["$y$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.THETA]
        plot_u_labels=["$F$", "$Theta$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
                       
    rollout_experiment_yz_force_psi = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[Crazyflie.Y Crazyflie.Z]
        plot_x_labels=["$y$, $z$],
        plot_u_indices=[Crazyflie.F, Crazyflie.PSI]
        plot_u_labels=["$F$", "$Psi$"]
        t_sim=6.0,
        n_sims_per_start=2,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e5,
        num_init_epochs=5,
        epochs_per_episode=100,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/crazyflie",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, 
        logger=tb_logger, 
        reload_dataloaders_every_epoch=True
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
