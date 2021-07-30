"""Define a dymamical system for a single integrator with lidar"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .planar_lidar_system import PlanarLidarSystem, Scene
from neural_clbf.systems.utils import Scenario


class SingleIntegrator2D(PlanarLidarSystem):
    """
    Represents a 2D single integrator with lidar.

    The system has state

        x = [x, y]

    representing the 2D location, and it has control inputs

        u = [vx, vy]

    representing the velocity in either direction.

    The system has no parameters
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2

    # State indices
    PX = 0
    PY = 1
    # Control indices
    VX = 0
    VY = 1

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
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            No required keys.
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
            num_rays=num_rays,
            field_of_view=field_of_view,
            max_distance=max_distance,
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    No required keys.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        return valid

    @property
    def n_dims(self) -> int:
        return SingleIntegrator2D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return SingleIntegrator2D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SingleIntegrator2D.PX] = 5.0
        upper_limit[SingleIntegrator2D.PY] = 5.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([5.0, 5.0])
        lower_limit = -torch.tensor([5.0, 5.0])

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

        # The single integrator doesn't move unless there is a control input

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

        # The velocities are directly actuated
        g[:, SingleIntegrator2D.PX, SingleIntegrator2D.VX] = 1.0
        g[:, SingleIntegrator2D.PY, SingleIntegrator2D.VY] = 1.0

        return g

    def planar_configuration(self, x: torch.Tensor) -> torch.Tensor:
        """Return the x, y, theta configuration of this system at the given states."""
        # This system has constant heading at theta = 0.
        thetas = torch.zeros(x.shape[0], 1).type_as(x)
        # Set velocities at zero for now
        velocities = torch.zeros(x.shape[0], 3).type_as(x)
        q = torch.cat((x, thetas, velocities), dim=-1)
        return q

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
        # For this dead-simple single integrator, we just find the direction from our
        # current position to the origin and go in that direction at constant velocity
        u = self.goal_point - x
        u_upper, u_lower = self.control_limits
        u = torch.clamp(u, u_lower, u_upper)

        return u
