"""Define a mock ControlAffineSystem for testing use"""
from typing import Tuple, List

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

    def __init__(self, nominal_params: Scenario):
        """
        Initialize the mock system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                    Requires no keys
        """
        super().__init__(nominal_params)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "I", "r"]
        returns:
            True if parameters are valid, False otherwise
        """
        # Nothing to validate for the mock system
        return True

    @property
    def n_dims(self) -> int:
        return MockSystem.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [1]

    @property
    def n_controls(self) -> int:
        return MockSystem.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        lower_limit = -1.0 * torch.ones(self.n_dims)
        upper_limit = 10.0 * torch.ones(self.n_dims)

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1.0, 1.0])
        lower_limit = torch.tensor([-1.0, -1.0])

        return (upper_limit, lower_limit)

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal (e.g. LQR or proportional) control for the nominal
        parameters. MockSystem just returns a zero input

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        batch_size = x.shape[0]
        u_nominal = torch.zeros((batch_size, self.n_controls)).type_as(x)

        return u_nominal

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
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

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
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

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the GCAS

        args:
            x: a tensor of points in the state space
        """
        safe_mask = x[:, 0] >= 0

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = x[:, 0] <= 0

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = x[:, 0].abs() <= 0.1

        return goal_mask
