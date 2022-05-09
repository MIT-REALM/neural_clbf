""""""
import torch
import numpy as np
from neural_clbf.controllers import Controller
from neural_clbf.experiments.multiagent_experiment import RolloutMultiagentStateSpaceExperiment
from neural_clbf.systems.multiagent_system import Multiagent



class MultiagentController(Controller):
    controller_period = 0.05


    def __init__(
        self,
        # experiment_suite: multiagent_experiment,
        controller_period: float = 0.01,
    ):
        nominal_params = {"m": 1.0, "R": 1.0, "L": 1.0}
        simulation_dt = 0.01
        
        self.controller_period = controller_period
        dynamics_model = Multiagent(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
        )
        # self.experiment_suite = 
        
        super(MultiagentController, self).__init__(dynamics_model, RolloutMultiagentStateSpaceExperiment)


    def u(self, x):
        """Get the control input for a given state"""
        return self.dynamics_model.u_nominal(x)