""""""
import torch
import numpy as np
from neural_clbf.controllers import Controller
from neural_clbf.experiments import multiagent_experiment
from neural_clbf.systems import Multiagent

from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)

class MultiagentController(Controller):
    controller_period = 0.05
    start_x = torch.tensor(
        [
            [0.5, 0.5],
            [-0.2, 1.0],
            [0.2, -1.0],
            [-0.2, -1.0],
        ]
    )
    simulation_dt = 0.01

    def __init__(
        self,
        dynamics_model: Multiagent,
        experiment_suite: multiagent_experiment,
        controller_period: float = 0.01,
    ):
        super(MultiagentController, self).__init__()
        self.controller_period = controller_period
        self.dynamics_model = dynamics_model
        self.experiment_suite = experiment_suite

   
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    scenarios = [
        nominal_params,
    ]
    # simulation_dt = 0.01
    # Define the dynamics model
    dynamics_model = Multiagent(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    #Initialize DataModule
    initial_conditions = [
        (-np.pi / 2, np.pi / 2),  # theta
        (-1.0, 1.0),  # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    def u(self, x):
        """Get the control input for a given state"""
        return self.dynamics_model.u_nominal(x)