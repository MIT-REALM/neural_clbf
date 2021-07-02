"""Define a dynamical system for a 2D quadrotor"""
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class Quad2D(ControlAffineSystem):
    """
    Represents a planar quadrotor.

    The system has state

        x = [px, pz, theta, vx, vz, theta_dot]

    representing the position, orientation, and velocities of the quadrotor, and it
    has control inputs

        u = [u_right, u_left]

    representing the thrust at the right and left rotor.

    The system is parameterized by
        m: mass
        I: rotational inertia
        r: the distance from the center of mass to the rotors (assumed to be symmetric)
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 2

    # State indices
    PX = 0
    PZ = 1
    THETA = 2
    VX = 3
    VZ = 4
    THETA_DOT = 5
    # Control indices
    U_RIGHT = 0
    U_LEFT = 1

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "I", "r"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params, dt, controller_dt)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "I", "r"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "m" in params
        valid = valid and "I" in params
        valid = valid and "r" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["I"] > 0
        valid = valid and params["r"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Quad2D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [Quad2D.THETA]

    @property
    def n_controls(self) -> int:
        return Quad2D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Quad2D.PX] = 2.0
        upper_limit[Quad2D.PZ] = 2.0
        upper_limit[Quad2D.THETA] = np.pi
        upper_limit[Quad2D.VX] = 2.0
        upper_limit[Quad2D.VZ] = 2.0
        upper_limit[Quad2D.THETA_DOT] = 2.0 * np.pi

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.nominal_params["m"] * grav / 2.0 + torch.tensor([4.0, 4.0])
        lower_limit = self.nominal_params["m"] * grav / 2.0 - torch.tensor([4.0, 4.0])

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid
        safe_z = -0.1
        floor_mask = x[:, 1] >= safe_z
        safe_mask.logical_and_(floor_mask)

        # We also have a block obstacle to the left at ground level
        obs1_min_x, obs1_max_x = (-1.1, -0.4)
        obs1_min_z, obs1_max_z = (-0.5, 0.6)
        obs1_mask_x = torch.logical_or(x[:, 0] <= obs1_min_x, x[:, 0] >= obs1_max_x)
        obs1_mask_z = torch.logical_or(x[:, 1] <= obs1_min_z, x[:, 1] >= obs1_max_z)
        obs1_mask = torch.logical_or(obs1_mask_x, obs1_mask_z)
        safe_mask.logical_and_(obs1_mask)

        # We also have a block obstacle to the right in the air
        obs2_min_x, obs2_max_x = (-0.1, 1.1)
        obs2_min_z, obs2_max_z = (0.7, 1.5)
        obs2_mask_x = torch.logical_or(x[:, 0] <= obs2_min_x, x[:, 0] >= obs2_max_x)
        obs2_mask_z = torch.logical_or(x[:, 1] <= obs2_min_z, x[:, 1] >= obs2_max_z)
        obs2_mask = torch.logical_or(obs2_mask_x, obs2_mask_z)
        safe_mask.logical_and_(obs2_mask)

        # Also constrain to be within a norm bound
        norm_mask = x.norm(dim=-1) <= 4.5
        safe_mask.logical_and_(norm_mask)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid
        unsafe_z = -0.3
        floor_mask = x[:, 1] <= unsafe_z
        unsafe_mask.logical_or_(floor_mask)

        # We also have a block obstacle to the left at ground level
        obs1_min_x, obs1_max_x = (-1.0, -0.5)
        obs1_min_z, obs1_max_z = (-0.4, 0.5)
        obs1_mask_x = torch.logical_and(x[:, 0] >= obs1_min_x, x[:, 0] <= obs1_max_x)
        obs1_mask_z = torch.logical_and(x[:, 1] >= obs1_min_z, x[:, 1] <= obs1_max_z)
        obs1_mask = torch.logical_and(obs1_mask_x, obs1_mask_z)
        unsafe_mask.logical_or_(obs1_mask)

        # We also have a block obstacle to the right in the air
        obs2_min_x, obs2_max_x = (0.0, 1.0)
        obs2_min_z, obs2_max_z = (0.8, 1.4)
        obs2_mask_x = torch.logical_and(x[:, 0] >= obs2_min_x, x[:, 0] <= obs2_max_x)
        obs2_mask_z = torch.logical_and(x[:, 1] >= obs2_min_z, x[:, 1] <= obs2_max_z)
        obs2_mask = torch.logical_and(obs2_mask_x, obs2_mask_z)
        unsafe_mask.logical_or_(obs2_mask)

        # Also constrain with a norm bound
        norm_mask = x.norm(dim=-1) >= 7.0
        unsafe_mask.logical_or_(norm_mask)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal_xz = x[:, : Quad2D.PZ + 1].norm(dim=-1) <= 0.3
        goal_mask.logical_and_(near_goal_xz)
        near_goal_theta = x[:, Quad2D.THETA].abs() <= 1.0
        goal_mask.logical_and_(near_goal_theta)
        near_goal_xz_velocity = x[:, Quad2D.VX : Quad2D.VZ + 1].norm(dim=-1) <= 1.0
        goal_mask.logical_and_(near_goal_xz_velocity)
        near_goal_theta_velocity = x[:, Quad2D.THETA_DOT].abs() <= 1.0
        goal_mask.logical_and_(near_goal_theta_velocity)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

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

        # The derivatives of px, pz, and theta are just the velocities
        f[:, Quad2D.PX, 0] = x[:, Quad2D.VX]
        f[:, Quad2D.PZ, 0] = x[:, Quad2D.VZ]
        f[:, Quad2D.THETA, 0] = x[:, Quad2D.THETA_DOT]

        # Acceleration in x has no control-independent part
        f[:, 3, 0] = 0.0
        # Acceleration in z is affected by the relentless pull of gravity
        f[:, 4, 0] = -grav
        # Acceleration in theta has no control-independent part
        f[:, 5, 0] = 0.0

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
        m, inertia, r = params["m"], params["I"], params["r"]
        # and state variables
        theta = x[:, Quad2D.THETA]

        # Effect on x acceleration
        g[:, Quad2D.VX, Quad2D.U_RIGHT] = -torch.sin(theta) / m
        g[:, Quad2D.VX, Quad2D.U_LEFT] = -torch.sin(theta) / m

        # Effect on z acceleration
        g[:, Quad2D.VZ, Quad2D.U_RIGHT] = torch.cos(theta) / m
        g[:, Quad2D.VZ, Quad2D.U_LEFT] = torch.cos(theta) / m

        # Effect on heading from rotors
        g[:, Quad2D.THETA_DOT, Quad2D.U_RIGHT] = r / inertia
        g[:, Quad2D.THETA_DOT, Quad2D.U_LEFT] = -r / inertia

        return g

    @property
    def u_eq(self):
        u_eq = (
            torch.zeros(
                (
                    1,
                    self.n_controls,
                )
            )
            + self.nominal_params["m"] * grav / 2.0
        )
        return u_eq
