"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Tuple, Optional

import torch

from neural_clbf.systems.utils import Scenario


class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamcial system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + g(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(self, nominal_params: Scenario):
        """
        Initialize a system.

        args:
            nominal_params: a dictionary giving the parameter values for the system
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__()

        # Validate parameters, raise error if they're not valid
        if not self.validate_params(nominal_params):
            raise ValueError(f"Parameters not valid: {nominal_params}")

        self.nominal_params = nominal_params

    @abstractmethod
    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "I", "r"]
        returns:
            True if parameters are valid, False otherwise
        """
        pass

    @abstractproperty
    def n_dims(self) -> int:
        pass

    @abstractproperty
    def n_controls(self) -> int:
        pass

    def control_affine_dynamics(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        # If no params required, use nominal params
        if params is None:
            params = self.nominal_params

        return self._f(x, params), self._g(x, params)

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x, params=params)
        # Compute state derivatives using control-affine form
        xdot = f + torch.bmm(g, u.unsqueeze(-1))
        return xdot.view(x.shape)

    @abstractmethod
    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        pass
