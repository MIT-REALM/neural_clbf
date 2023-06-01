from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.systems.cartpole import Cartpole
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)

from neural_clbf.experiments import (
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
    ExperimentSuite,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

def maybe_init_wandb(args):
    if args.track:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=args.wandb_run_name,
        )
        return run

def main(args):

    nominal_params = {}
    simulation_dt = args.dt
    controller_period = args.controller_dt
    scenarios = [
        nominal_params,
    ]

    dynamics_model = Cartpole(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0), # cart_pos
        (-4.0, 4.0), # cart_vel
        (-np.pi / 2, np.pi / 2),  # pole_angle
        (-1.0, 1.0),  # pole_angvel
    ]

    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=args.n_fixed_samples,
        max_points=100000,
        val_split=0.1,
        batch_size=args.batch_size,
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # TODO: Define the experiment suite
    # V_contour_experiment, rollout_experiment
    V_contour_thetas_experiment = CLFContourExperiment(
        "V_Contour_thetas",
        domain=[(-1.5, 1.5), (-1.5, 1.5)],
        n_grid=25,
        x_axis_index=Cartpole.POLE_ANGLE,
        y_axis_index=Cartpole.POLE_ANGVEL,
        x_axis_label="$\\theta$",
        y_axis_label="$\\dot{\\theta}$",
    )
    start_x = torch.Tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [-0.2, 1.0, -0.2, 1.0],
            [0.2, -1.0, 0.2, -1.0],
            [-0.2, -1.0, -0.2, -1.0],
        ]
    )
    rollout_thetas_experiment = RolloutStateSpaceExperiment(
        "Rollout_thetas",
        start_x,
        Cartpole.CART_POS,
        "$\\theta$",
        Cartpole.CART_VEL,
        "$\\dot{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([
        V_contour_thetas_experiment,
        rollout_thetas_experiment,
    ])

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
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        disable_gurobi=True,
        barrier=args.barrier,
    )

    maybe_init_wandb(args)
    
    # Initialize the logger and trainer
    logger = pl_loggers.TensorBoardLogger(
        "logs/cartpole",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=args.n_epochs,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)

def add_clbf_argparse_args(parser: ArgumentParser):
    parser.add_argument('--dt', type=float, default=0.01, help='Simulation timestep')
    parser.add_argument('--controller_dt', type=float, default=0.05, help='Controller timestep')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-fixed-samples', type=int, default=10000, help='Number of fixed samples')
    parser.add_argument('--barrier', action='store_true', help='Use barrier function')
    parser.add_argument('-t', '--track', action='store_true', default=False)
    parser.add_argument('--n-epochs', type=int, default=51)
    return parser

def add_wandb_argparse_args(parser: ArgumentParser):
    parser.add_argument('--wandb-entity', type=str, default='dtch1997')
    parser.add_argument('--wandb-group', type=str, default="default")
    parser.add_argument('--wandb-project', type=str, default='NeuralCLBF')
    parser.add_argument('--wandb-run-name', type=str, default='cartpole')
    return parser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_clbf_argparse_args(parser)
    parser = add_wandb_argparse_args(parser)
    args = parser.parse_args()
    main(args)
