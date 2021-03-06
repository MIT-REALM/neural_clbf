"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Tuple

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

    def __init__(self, params: Scenario):
        """
        Initialize a system.

        args:
            params: a dictionary giving the parameter values for the system
        """
        super().__init__()

        self.params = params

    @abstractproperty
    def n_dims(self) -> int:
        pass

    @abstractproperty
    def n_controls(self) -> int:
        pass

    def control_affine_dynamics(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        return self._f(x), self._g(x)

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x)
        # Compute state derivatives using control-affine form
        xdot = f + torch.bmm(g, u.unsqueeze(-1))
        return xdot.view(x.shape)

    @abstractmethod
    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        pass
