"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class LinearSatellite(ControlAffineSystem):
    """
    Represents a satellite through the linearized Clohessy-Wiltshire equations

    The system has state

        x = [x, y, z, xdot, ydot, zdot]

    representing the position and velocity of the chaser satellite, and it
    has control inputs

        u = [ux, uy, uz]

    representing the thrust applied in each axis. Distances are in km, and control
    inputs are measured in km/s^2.

    The task here is to get to the origin without leaving the bounding box [-5, 5] on
    all positions and [-1, 1] on velocities.

    The system is parameterized by
        a: the length of the semi-major axis of the target's orbit (e.g. 6871)
        ux_target, uy_target, uz_target: accelerations due to unmodelled effects and
                                         target control.
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 3

    # State indices
    X = 0
    Y = 1
    Z = 2
    XDOT = 3
    YDOT = 4
    ZDOT = 5
    # Control indices
    UX = 0
    UY = 1
    UZ = 2

    # Constant parameters
    MU = 398600.0  # Earth's gravitational parameter

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
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "a" in params
        valid = valid and "ux_target" in params
        valid = valid and "uy_target" in params
        valid = valid and "uz_target" in params

        # Make sure all parameters are physically valid
        valid = valid and params["a"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return LinearSatellite.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LinearSatellite.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[LinearSatellite.X] = 1.5
        upper_limit[LinearSatellite.Y] = 1.5
        upper_limit[LinearSatellite.Z] = 1.5
        upper_limit[LinearSatellite.XDOT] = 1
        upper_limit[LinearSatellite.YDOT] = 1
        upper_limit[LinearSatellite.ZDOT] = 1

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([0.03, 0.03, 0.03])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Stay within some maximum distance from the target
        distance = x[:, : LinearSatellite.Z + 1].norm(dim=-1)
        # safe_mask.logical_and_(distance <= 1.5)

        # Stay at least some minimum distance from the target
        safe_mask.logical_and_(distance >= 0.75)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Maximum distance
        distance = x[:, : LinearSatellite.Z + 1].norm(dim=-1)
        # unsafe_mask.logical_or_(distance >= 2.0)

        # Minimum distance
        unsafe_mask.logical_or_(distance <= 0.3)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = x.norm(dim=-1) <= 0.5

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
        a = params["a"]
        ux_target = params["ux_target"]
        uy_target = params["uy_target"]
        uz_target = params["uz_target"]
        # Compute mean-motion
        n = sqrt(LinearSatellite.MU / a ** 3)
        # and state variables
        x_ = x[:, LinearSatellite.X]
        z_ = x[:, LinearSatellite.Z]
        xdot_ = x[:, LinearSatellite.XDOT]
        ydot_ = x[:, LinearSatellite.YDOT]
        zdot_ = x[:, LinearSatellite.ZDOT]

        # The first three dimensions just integrate the velocity
        f[:, LinearSatellite.X, 0] = xdot_
        f[:, LinearSatellite.Y, 0] = ydot_
        f[:, LinearSatellite.Z, 0] = zdot_

        # The last three use the CHW equations
        f[:, LinearSatellite.XDOT, 0] = 3 * n ** 2 * x_ + 2 * n * ydot_
        f[:, LinearSatellite.YDOT, 0] = -2 * n * xdot_
        f[:, LinearSatellite.ZDOT, 0] = -(n ** 2) * z_

        # Add perturbations
        f[:, LinearSatellite.XDOT, 0] += ux_target
        f[:, LinearSatellite.YDOT, 0] += uy_target
        f[:, LinearSatellite.ZDOT, 0] += uz_target

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

        # The control inputs are accelerations
        g[:, LinearSatellite.XDOT, LinearSatellite.UX] = 1.0
        g[:, LinearSatellite.YDOT, LinearSatellite.UY] = 1.0
        g[:, LinearSatellite.ZDOT, LinearSatellite.UZ] = 1.0

        return g
