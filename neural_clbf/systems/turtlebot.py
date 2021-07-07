"""Define a dymamical system for TurtleBot3"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class TurtleBot(ControlAffineSystem):
    """
    Represents a two wheeled differential drive robot, the TurtleBot3.
    The system has state
        p = [x, y, theta]
    representing the x and y position and angle of orientation of the robot,
    and it has control inputs
        u = [v theta_dot]
    representing the desired linear velocity and angular velocity.
    The system is parameterized by
        R: radius of the wheels
        L: radius of rotation, or the distance between the two wheels
    """

    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1
    THETA = 2
    # Control indices
    V = 0
    THETA_DOT = 1

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
                            Requires keys ["R", "L"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios,
            use_linearized_controller=False
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid
        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["R", "L"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "R" in params
        valid = valid and "L" in params

        # Make sure all parameters are physically valid
        valid = valid and params["R"] > 0
        valid = valid and params["L"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return TurtleBot.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [TurtleBot.THETA]

    @property
    def n_controls(self) -> int:
        return TurtleBot.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[TurtleBot.X] = 2.0
        upper_limit[TurtleBot.Y] = 2.0
        upper_limit[TurtleBot.THETA] = np.pi

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        # TODO @bethlow these are relaxed for now, but eventually
        # these values should be measured on the hardware.
        upper_limit = torch.ones(self.n_controls)
        upper_limit[TurtleBot.V] = 100 * 10.0
        upper_limit[TurtleBot.THETA_DOT] = 2.0 * np.pi
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        safe_mask = x.norm(dim=-1) <= 0.8

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = x.norm(dim=-1) >= 1.5

        return unsafe_mask

    def distance_to_goal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distance from each point in x to the goal (positive for points
        outside the goal, negative for points inside the goal), normalized by the state
        limits.
        args:
            x: the points from which we calculate distance
        """
        upper_limit, _ = self.state_limits
        return x.norm(dim=-1) / upper_limit.norm()

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

        # f is a zero vector as nothing should happen when no control input is given
        f[:, TurtleBot.X, 0] = 0

        f[:, TurtleBot.Y, 0] = 0

        f[:, TurtleBot.THETA, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-dependent part of the control-affine dynamics.
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
        R, L = params["R"], params["L"]
        # and state variables
        theta = x[:, TurtleBot.THETA]

        # Tensor for wheel velocities
        v = torch.zeros((2, 2))
        v = v.type_as(x)

        # Building tensor v
        v[0, 0] = 1 / R
        v[1, 0] = 1 / R
        v[0, 1] = L / (2 * R)
        v[1, 1] = -L / (2 * R)

        # Effect on x
        g[:, TurtleBot.X, TurtleBot.V] = R / 2 * torch.cos(theta)
        g[:, TurtleBot.X, TurtleBot.THETA_DOT] = R / 2 * torch.cos(theta)

        # Effect on y
        g[:, TurtleBot.Y, TurtleBot.V] = R / 2 * torch.sin(theta)
        g[:, TurtleBot.Y, TurtleBot.THETA_DOT] = R / 2 * torch.sin(theta)

        # Effect on theta
        g[:, TurtleBot.THETA, TurtleBot.V] = -R / (2 * L)
        g[:, TurtleBot.THETA, TurtleBot.THETA_DOT] = R / (2 * L)

        g = g.matmul(v)

        return g

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        

        self.P = torch.eye(3,3)
        self.K = torch.ones(self.n_controls, self.n_dims)

        K = self.K.type_as(x)
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)
        # Clamp given the control limits
        # import pdb; pdb.set_trace()
        upper_u_lim, lower_u_lim = self.control_limits
        u = torch.clamp(u, min=lower_u_lim[0].item(), max=upper_u_lim[0].item())
        # for dim_idx in range(self.n_controls):
        #     u[:, dim_idx] = torch.clamp(
        #         u[:, dim_idx],
        #         min=lower_u_lim[dim_idx].item(),
        #         max=upper_u_lim[dim_idx].item(),
        #     ) 

        return u