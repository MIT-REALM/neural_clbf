"""Define a dymamical system for an Segway"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class Segway(ControlAffineSystem):
    """
    Represents a Segway.

    The system has state

        x = [p, theta, v, theta_dot]

    representing the position, angle, and velocity of the Segway, and it
    has control inputs

        u = [u]

    representing the force applied at the base

    The system is parameterized by
        m: mass
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    POS = 0
    THETA = 1
    V = 2
    THETA_DOT = 3
    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "m" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Segway.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [Segway.THETA]

    @property
    def n_controls(self) -> int:
        return Segway.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Segway.POS] = 3.0
        upper_limit[Segway.THETA] = np.pi / 2
        upper_limit[Segway.V] = 1.0
        upper_limit[Segway.THETA_DOT] = 3.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100 * 10.0])
        lower_limit = -torch.tensor([100 * 10.0])

        return (upper_limit, lower_limit)

    @property
    def goal_point(self):
        return torch.tensor([[2.0, 0, 0, 0]])

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # We have to avoid a bar at (0, 1) with some radius
        # Bar parameters
        bar_x = 0
        bar_y = 1
        bar_radius = 0.15
        safety_margin = 1.5

        # Get position of head of segway
        p = x[:, Segway.POS]
        theta = x[:, Segway.THETA]
        segway_head_x = p + torch.sin(theta)
        segway_head_y = torch.cos(theta)

        # Compute distance to the bar and make sure it's greater than the bar radius
        distance_to_bar = (segway_head_x - bar_x) ** 2 + (segway_head_y - bar_y) ** 2
        distance_to_bar = torch.sqrt(distance_to_bar)
        safe_mask = torch.logical_and(
            safe_mask, distance_to_bar >= safety_margin * bar_radius
        )

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have to avoid a bar at (0, 1) with some radius
        # Bar parameters
        bar_x = 0
        bar_y = 1
        bar_radius = 0.15

        # Get position of head of segway
        p = x[:, Segway.POS]
        theta = x[:, Segway.THETA]
        segway_head_x = p + torch.sin(theta)
        segway_head_y = torch.cos(theta)

        # Compute distance to the bar and make sure it's greater than the bar radius
        distance_to_bar = (segway_head_x - bar_x) ** 2 + (segway_head_y - bar_y) ** 2
        distance_to_bar = torch.sqrt(distance_to_bar)
        unsafe_mask = torch.logical_or(unsafe_mask, distance_to_bar <= bar_radius)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = (x - self.goal_point).norm(dim=-1) <= 0.3

        return goal_mask

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
        f = f.type_as(x)

        # Extract the needed parameters
        # m = params["m"]
        # and state variables
        theta = x[:, Segway.THETA]
        v = x[:, Segway.V]
        theta_dot = x[:, Segway.THETA_DOT]

        f[:, Segway.POS, 0] = v
        f[:, Segway.THETA, 0] = theta_dot
        f[:, Segway.V, 0] = (
            torch.cos(theta) * (9.8 * torch.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (theta_dot ** 2) * torch.sin(theta)
        ) / (torch.cos(theta) - 24.7)
        f[:, Segway.THETA_DOT, 0] = (
            -58.8 * v * torch.cos(theta)
            - 243.5 * v
            - torch.sin(theta) * (208.3 + (theta_dot ** 2) * torch.cos(theta))
        ) / (torch.cos(theta) ** 2 - 24.7)

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
        g = g.type_as(x)

        # Extract the needed parameters
        # m = params["m"]
        # and state variables
        theta = x[:, Segway.THETA]

        g[:, Segway.V, Segway.U] = (-1.8 * torch.cos(theta) - 10.9) / (
            torch.cos(theta) - 24.7
        )
        g[:, Segway.THETA_DOT, Segway.U] = (9.3 * torch.cos(theta) + 38.6) / (
            torch.cos(theta) ** 2 - 24.7
        )

        return g
