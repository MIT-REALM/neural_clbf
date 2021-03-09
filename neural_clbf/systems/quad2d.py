"""Define a dynamical system for a 2D quadrotor"""
import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario, lqr


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

    def __init__(self, nominal_params: Scenario):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "I", "r"]
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params)

        # Compute the LQR gain matrix for the nominal parameters
        # Linearize the system about the x = 0, u1 = u2 = mg / 2
        A = np.zeros((self.n_dims, self.n_dims))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 2] = -grav

        B = np.zeros((self.n_dims, self.n_controls))
        B[4, 0] = 1.0 / self.nominal_params["m"]
        B[4, 1] = 1.0 / self.nominal_params["m"]
        B[5, 0] = self.nominal_params["r"] / self.nominal_params["I"]
        B[5, 1] = -self.nominal_params["r"] / self.nominal_params["I"]

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self.K = torch.tensor(lqr(A, B, Q, R))

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
    def n_controls(self) -> int:
        return Quad2D.N_CONTROLS

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

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. For Quad2D, the nominal
        controller is LQR

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Compute nominal control from feedback + equilibrium control
        u_nominal = -(self.K.type_as(x) @ x.T).T
        u_eq = torch.zeros_like(u_nominal) + self.nominal_params["m"] * grav / 2.0

        return u_nominal + u_eq
