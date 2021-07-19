"""Define an base class for a systems that yields observations"""
from abc import abstractmethod, abstractproperty
from typing import Optional

import torch

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
)


class ObservableSystem(ControlAffineSystem):
    """
    Represents a generic dynamical system that yields observations.
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = True,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize a system.

        args:
            nominal_params: a dictionary giving the parameter values for the system
            dt: the timestep to use for simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_linearized_controller: if True, linearize the system model to derive a
                                       LQR controller. If false, the system is must
                                       set self.P itself to be a tensor n_dims x n_dims
                                       positive definite matrix.
            scenarios: an optional list of scenarios for robust control
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super(ObservableSystem, self).__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            use_linearized_controller=use_linearized_controller,
            scenarios=scenarios,
        )

    @abstractproperty
    def n_obs(self) -> int:
        """Number of observations per dimension"""
        pass

    @abstractproperty
    def obs_dim(self) -> int:
        """Number of dimensions observed"""
        pass

    @abstractmethod
    def get_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Get the vector of measurements at this point

        args:
            x: an N x self.n_dims tensor of state

        returns:
            an N x self.obs_dim x self.n_obs tensor containing the observed data
        """
        pass
