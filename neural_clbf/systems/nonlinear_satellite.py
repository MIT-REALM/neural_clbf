"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class NonlinearSatellite(ControlAffineSystem):
    """
    Represents a satellite using linearized Clohessy Wiltshire equations plus nonlinear
    drag terms (but neglecting J2 dynamics), assuming a circular orbit.

    The system has state

        x = [x, y, z, xdot, ydot, zdot]

    representing the position and velocity of the chaser satellite, and it
    has control inputs

        u = [ux, uy, uz]

    representing the thrust applied in each axis. Distances are in km, and control
    inputs are measured in km/s^2

    The task is to remain within the state limits and stay at least 0.2 km from the
    origin (where there is another satellite).

    The system is parameterized by
        mu: Earth's gravitational parameter (known constant)
        A: the surface area of the spacecraft (e.g. 2e-6)
        Cd: the atmospheric drag coefficient (e.g. 2)
        rho: the atmospheric density (e.g. 9.1515e-5)
        m: the mass of the satellite (e.g. 500)
        r the radius of the circular orbit (e.g. 500 + NonlinearSatellite.RE)
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
    RE = 6371.0  # Earth's radius

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
                            Requires keys ["A", "Cd", "rho", "m", "r"]
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
                    Requires keys ["A", "Cd", "rho", "m", "r"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "A" in params
        valid = valid and "Cd" in params
        valid = valid and "rho" in params
        valid = valid and "m" in params
        valid = valid and "r" in params

        # Make sure all parameters are physically valid
        valid = valid and params["A"] > 0
        valid = valid and params["Cd"] > 0
        valid = valid and params["rho"] > 0
        valid = valid and params["m"] > 0
        valid = valid and params["r"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return NonlinearSatellite.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return NonlinearSatellite.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[NonlinearSatellite.X] = 5
        upper_limit[NonlinearSatellite.Y] = 5
        upper_limit[NonlinearSatellite.Z] = 5
        upper_limit[NonlinearSatellite.XDOT] = 1
        upper_limit[NonlinearSatellite.YDOT] = 1
        upper_limit[NonlinearSatellite.ZDOT] = 1

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

        state_limit_mask = x.norm(dim=-1) <= 1.0
        safe_mask.logical_and_(state_limit_mask)

        obstacle_avoidance = x.norm(dim=-1) >= 0.4
        safe_mask.logical_and_(obstacle_avoidance)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        state_limit_mask = x.norm(dim=-1) >= 4.0
        unsafe_mask.logical_or_(state_limit_mask)

        obstacle = x.norm(dim=-1) <= 0.2
        unsafe_mask.logical_or_(obstacle)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = x.norm(dim=-1) <= 0.3

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
        A, Cd, rho, m = params["A"], params["Cd"], params["rho"], params["m"]
        r = params["r"]
        # and state variables
        x_ = x[:, NonlinearSatellite.X]
        y_ = x[:, NonlinearSatellite.Y]
        z_ = x[:, NonlinearSatellite.Z]
        xdot_ = x[:, NonlinearSatellite.XDOT]
        ydot_ = x[:, NonlinearSatellite.YDOT]
        zdot_ = x[:, NonlinearSatellite.ZDOT]

        # The first three dimensions just integrate the velocity
        f[:, NonlinearSatellite.X, 0] = xdot_
        f[:, NonlinearSatellite.Y, 0] = ydot_
        f[:, NonlinearSatellite.Z, 0] = zdot_

        # The last three use equation 7 from Axel's paper
        f[:, NonlinearSatellite.XDOT, 0] = (
            -NonlinearSatellite.MU * x_ / r ** 3 - 0.5 * Cd * A * rho / m * xdot_ ** 2
        )
        f[:, NonlinearSatellite.YDOT, 0] = (
            -NonlinearSatellite.MU * y_ / r ** 3 - 0.5 * Cd * A * rho / m * ydot_ ** 2
        )
        f[:, NonlinearSatellite.ZDOT, 0] = (
            -NonlinearSatellite.MU * z_ / r ** 3 - 0.5 * Cd * A * rho / m * zdot_ ** 2
        )

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
        g[:, NonlinearSatellite.XDOT, NonlinearSatellite.UX] = 1.0
        g[:, NonlinearSatellite.YDOT, NonlinearSatellite.UY] = 1.0
        g[:, NonlinearSatellite.ZDOT, NonlinearSatellite.UZ] = 1.0

        return g
