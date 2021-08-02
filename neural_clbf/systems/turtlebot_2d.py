"""Define a dymamical system for a single integrator with lidar"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .planar_lidar_system import PlanarLidarSystem, Scene
from neural_clbf.systems.utils import Scenario


class TurtleBot2D(PlanarLidarSystem):
    """
    Represents a two wheeled differential drive robot, the TurtleBot3, with lidar
    sensing. The dynamics were written by @bethlow and the sensor model was added
    by @dawsonc

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
        scene: Scene,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        num_rays: int = 10,
        field_of_view: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        max_distance: float = 10.0,
    ):
        """
        Initialize the turtlebot.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["R", "L"]
            scene: the scene in which to operate
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params,
            scene,
            dt=dt,
            controller_dt=controller_dt,
            use_linearized_controller=False,
            num_rays=num_rays,
            field_of_view=field_of_view,
            max_distance=max_distance,
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
        return TurtleBot2D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [TurtleBot2D.THETA]

    @property
    def n_controls(self) -> int:
        return TurtleBot2D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[TurtleBot2D.X] = 2.0
        upper_limit[TurtleBot2D.Y] = 2.0
        upper_limit[TurtleBot2D.THETA] = 2 * np.pi

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        # These are relaxed for now, but eventually they should be measured on hardware
        upper_limit = torch.ones(self.n_controls)
        upper_limit[TurtleBot2D.V] = 2.0
        upper_limit[TurtleBot2D.THETA_DOT] = 2.0 * np.pi
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

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
        f[:, TurtleBot2D.X, 0] = 0
        f[:, TurtleBot2D.Y, 0] = 0
        f[:, TurtleBot2D.THETA, 0] = 0

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

        # Extract state variables
        theta = x[:, TurtleBot2D.THETA]

        # Effect on x
        g[:, TurtleBot2D.X, TurtleBot2D.V] = torch.cos(theta)
        g[:, TurtleBot2D.X, TurtleBot2D.THETA_DOT] = 0.0

        # Effect on y
        g[:, TurtleBot2D.Y, TurtleBot2D.V] = torch.sin(theta)
        g[:, TurtleBot2D.Y, TurtleBot2D.THETA_DOT] = 0.0

        # Effect on theta
        g[:, TurtleBot2D.THETA, TurtleBot2D.V] = 0.0
        g[:, TurtleBot2D.THETA, TurtleBot2D.THETA_DOT] = 1.0

        return g

    def planar_configuration(self, x: torch.Tensor) -> torch.Tensor:
        """Return the x, y, theta configuration of this system at the given states."""
        # The state is just the planar configuration
        return x

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters.

        args:
            x: bs x self.n_dims tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # The turtlebot linearization is not well-behaved, so we create our own
        # P and K matrices (mainly as placeholders)
        self.P = torch.eye(self.n_dims)
        self.K = torch.zeros(self.n_controls, self.n_dims)

        # This controller should navigate us towards the origin. We can do this by
        # setting a velocity proportional to the inner product of the vector
        # from the turtlebot to the origin and the vector pointing out in front of
        # the turtlebot. If the bot is pointing away from the origin, this inner product
        # will be negative, so we'll drive backwards towards the goal. If the bot
        # is pointing towards the origin, it will drive forwards.
        u = torch.zeros(x.shape[0], self.n_controls)

        v_scaling = 1.0
        bot_to_origin = -x[:, : TurtleBot2D.Y + 1].reshape(-1, 1, 2)
        theta = x[:, TurtleBot2D.THETA]
        bot_facing = torch.stack((torch.cos(theta), torch.sin(theta))).T.unsqueeze(-1)
        u[:, TurtleBot2D.V] = v_scaling * torch.bmm(bot_to_origin, bot_facing).squeeze()

        # In addition to setting the velocity towards the origin, we also need to steer
        # towards the origin. We can do this via P control on the angle between the
        # turtlebot and the vector to the origin.
        #
        # However, this angle becomes ill-defined as the bot approaches the origin, so
        # so we switch this term off if the bot is too close (and instead just control
        # theta to zero)
        phi_control_on = bot_to_origin.norm(dim=-1) >= 0.2
        phi_control_on = phi_control_on.reshape(-1)
        omega_scaling = 5.0
        angle_from_origin_to_bot = torch.atan2(x[:, TurtleBot2D.Y], x[:, TurtleBot2D.X])
        phi = theta - angle_from_origin_to_bot
        # First, wrap the angle error into [-pi, pi]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        # Now decrement any errors > pi/2 by pi and increment any errors < -pi / 2 by pi
        # Then P controlling the error to zero will drive the bot to point towards the
        # origin
        phi[phi > np.pi / 2.0] -= np.pi
        phi[phi < -np.pi / 2.0] += np.pi

        # Only apply this P control when the bot is far enough from the origin;
        # default to P control on theta
        u[:, TurtleBot2D.THETA_DOT] = -omega_scaling * theta
        u[phi_control_on, TurtleBot2D.THETA_DOT] = -omega_scaling * phi[phi_control_on]

        # Clamp given the control limits
        u_upper, u_lower = self.control_limits
        u = torch.clamp(u, u_lower, u_upper)

        return u
