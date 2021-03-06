"""Define a mock ControlAffineSystem for testing use"""
import torch

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario


class MockSystem(ControlAffineSystem):
    """
    Represents a mock system.
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2

    def __init__(self, params: Scenario):
        """
        Initialize the mock system.

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires no keys
        """
        super().__init__(params)

    @property
    def n_dims(self) -> int:
        return MockSystem.N_DIMS

    @property
    def n_controls(self) -> int:
        return MockSystem.N_CONTROLS

    def _f(self, x: torch.Tensor):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))

        # Mock dynamics
        f[:, 0, 0] = 1.0
        f[:, 1, 0] = 2.0

        return f

    def _g(self, x: torch.Tensor):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))

        # Mock dynamics
        g[:, 0, 0] = 1.0
        g[:, 0, 1] = 2.0
        g[:, 1, 0] = 3.0
        g[:, 1, 1] = 4.0

        return g
