"""Define a dynamical system for a 3D quadrotor"""
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class Quad3D(ControlAffineSystem):
    """
    Represents a planar quadrotor.

    The system has state

        x = [px, py, pz, vx, vy, vz, phi, theta, psi]

    representing the position, orientation, and velocities of the quadrotor, and it
    has control inputs

        u = [f, phi_dot, theta_dot, psi_dot]

    The system is parameterized by
        m: mass

    NOTE: Z is defined as positive downwards
    """

    # Number of states and controls
    N_DIMS = 9
    N_CONTROLS = 4

    # State indices
    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5

    PHI = 6
    THETA = 7
    PSI = 8

    # Control indices
    F = 0
    PHI_DOT = 1
    THETA_DOT = 2
    PSI_DOT = 3

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
                            Requires keys ["m"]
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
        return Quad3D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [Quad3D.PHI, Quad3D.THETA, Quad3D.PSI]

    @property
    def n_controls(self) -> int:
        return Quad3D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Quad3D.PX] = 4.0
        upper_limit[Quad3D.PY] = 4.0
        upper_limit[Quad3D.PZ] = 4.0
        upper_limit[Quad3D.VX] = 8.0
        upper_limit[Quad3D.VY] = 8.0
        upper_limit[Quad3D.VZ] = 8.0
        upper_limit[Quad3D.PHI] = np.pi / 2.0
        upper_limit[Quad3D.THETA] = np.pi / 2.0
        upper_limit[Quad3D.PSI] = np.pi / 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100, 50, 50, 50])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        safe_z = 0.0
        safe_radius = 3
        safe_mask = torch.logical_and(
            x[:, Quad3D.PZ] <= safe_z, x.norm(dim=-1) <= safe_radius
        )

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        unsafe_z = 0.3
        unsafe_radius = 3.5
        unsafe_mask = torch.logical_or(
            x[:, Quad3D.PZ] >= unsafe_z, x.norm(dim=-1) >= unsafe_radius
        )

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = x.norm(dim=-1) <= 0.3
        goal_mask.logical_and_(near_goal)

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

        # Derivatives of positions are just velocities
        f[:, Quad3D.PX] = x[:, Quad3D.VX]  # x
        f[:, Quad3D.PY] = x[:, Quad3D.VY]  # y
        f[:, Quad3D.PZ] = x[:, Quad3D.VZ]  # z

        # Constant acceleration in z due to gravity
        f[:, Quad3D.VZ] = grav

        # Orientation velocities are directly actuated

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
        m = params["m"]

        # Derivatives of linear velocities depend on thrust f
        s_theta = torch.sin(x[:, Quad3D.THETA])
        c_theta = torch.cos(x[:, Quad3D.THETA])
        s_phi = torch.sin(x[:, Quad3D.PHI])
        c_phi = torch.cos(x[:, Quad3D.PHI])
        g[:, Quad3D.VX, Quad3D.F] = -s_theta / m
        g[:, Quad3D.VY, Quad3D.F] = c_theta * s_phi / m
        g[:, Quad3D.VZ, Quad3D.F] = -c_theta * c_phi / m

        # Derivatives of all orientations are control variables
        g[:, Quad3D.PHI :, Quad3D.PHI_DOT :] = torch.eye(self.n_controls - 1)

        return g

    @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls))
        u_eq[0, Quad3D.F] = self.nominal_params["m"] * grav
        return u_eq
