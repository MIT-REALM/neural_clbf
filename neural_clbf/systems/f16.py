"""Define a dynamical system for the F16 AeroBench model"""
from typing import Tuple, Optional

import torch
import numpy as np

import neural_clbf.setup.aerobench as aerobench_loader  # type: ignore
from aerobench.highlevel.controlled_f16 import controlled_f16  # type: ignore
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot  # type: ignore
from aerobench.lowlevel.low_level_controller import LowLevelController  # type: ignore

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario


# make sure that the import worked
assert aerobench_loader


class F16(ControlAffineSystem):
    """
    Represents an F16 aircraft

    The system has state

        x[0] = air speed, VT    (ft/sec)
        x[1] = angle of attack, alpha  (rad)
        x[2] = angle of sideslip, beta (rad)
        x[3] = roll angle, phi  (rad)
        x[4] = pitch angle, theta  (rad)
        x[5] = yaw angle, psi  (rad)
        x[6] = roll rate, P  (rad/sec)
        x[7] = pitch rate, Q  (rad/sec)
        x[8] = yaw rate, R  (rad/sec)
        x[9] = northward horizontal displacement, pn  (feet)
        x[10] = eastward horizontal displacement, pe  (feet)
        x[11] = altitude, h  (feet)
        x[12] = engine thrust dynamics lag state, pow
        x[13, 14, 15] = internal integrator states

    and control inputs, which are setpoints for a lower-level integrator

        u[0] = Z acceleration
        u[1] = stability roll rate
        u[2] = side acceleration + yaw rate (usually regulated to 0)
        u[3] = throttle command (0.0, 1.0)

    The system is parameterized by
        lag_error: the additive error in the engine lag state dynamics
    """

    # Number of states and controls
    N_DIMS = 16
    N_CONTROLS = 4

    # State indices
    VT = 0  # airspeed
    ALPHA = 1  # angle of attack
    BETA = 2  # sideslip angle
    PHI = 3  # roll angle
    THETA = 4  # pitch angle
    PSI = 5  # yaw angle
    P = 6  # roll rate
    Q = 7  # pitch rate
    R = 8  # yaw rate
    POSN = 9  # northward displacement
    POSE = 10  # eastward displacement
    H = 11  # altitude
    POW = 12  # engine thrust dynamics lag state
    # Control indices
    U_NZ = 0  # desired z acceleration
    U_SR = 1  # desired stability roll rate
    U_NYR = 2  # desired side acceleration + yaw rate
    U_THROTTLE = 3  # throttle command

    def __init__(self, nominal_params: Scenario):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["lag_error"]
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["lag_error"]
        returns:
            True if parameters are valid, False otherwise
        """
        # Make sure needed parameters were provided
        valid = "lag_error" in params

        return valid

    @property
    def n_dims(self) -> int:
        return F16.N_DIMS

    @property
    def n_controls(self) -> int:
        return F16.N_CONTROLS

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Not implemented. The F16 model can only compute f and g simultaneously using
        a linear regression.
        """
        raise NotImplementedError()

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Not implemented. The F16 model can only compute f and g simultaneously using
        a linear regression.
        """
        raise NotImplementedError()

    def control_affine_dynamics(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Default to nominal parameters
        if params is None:
            params = self.nominal_params

        # The f16 model is not batched, so we need to compute f and g for each row of x
        n_batch = x.size()[0]
        f = torch.zeros((n_batch, self.n_dims, 1)).type_as(x)
        g = torch.zeros(n_batch, self.n_dims, self.n_controls).type_as(x)

        # Convert input to numpy
        x = x.detach().cpu().numpy()
        for batch in range(n_batch):
            # Get the derivatives at each of n_controls + 1 linearly independent points
            # (plus zero) to fit control-affine dynamics
            u = np.zeros((1, self.n_controls))
            for i in range(self.n_controls):
                u_i = np.zeros((1, self.n_controls))
                u_i[0, i] = 1.0
                u = np.vstack((u, u_i))

            # Compute derivatives at each of these points
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot = np.zeros((self.n_controls + 1, self.n_dims))
            for i in range(self.n_controls + 1):
                xdot[i, :], _, _, _, _ = controlled_f16(
                    t, x[batch, :], u[i, :], llc, f16_model=model
                )

            # Run a least-squares regression to fit control-affine dynamics
            # We want a relationship of the form
            #       xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
            # Augment the inputs with a one column for the control-independent part
            regressors = np.hstack((np.ones((self.n_controls + 1, 1)), u))
            # Compute the least-squares fit and find A^T such that xdot = [1, u] A^T
            A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
            A = A.T
            # Extract the control-affine fit
            f[batch, :, 0] = torch.tensor(A[:, 0]).type_as(f)
            g[batch, :, :] = torch.tensor(A[:, 1:]).type_as(g)

            # Add in the lag error (which we're treating as bounded additive error)
            f[batch, self.POW] += params["lag_error"]

        return f, g

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u, computed using
        the underlying simulation (no control-affine approximation)

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # The F16 model is not batched, so we need to derivatives for each x separately
        n_batch = x.size()[0]
        xdot = torch.zeros_like(x).type_as(x)

        # Convert input to numpy
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        for batch in range(n_batch):
            # Compute derivatives at this point
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot_np, _, _, _, _ = controlled_f16(
                t, x_np[batch, :], u_np[batch, :], llc, f16_model=model
            )

            xdot[batch, :] = torch.tensor(xdot_np).type_as(x)

        return xdot

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. For F16, the nominal
        controller is the GCAS controller from the original AeroBench toolkit.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        gcas = GcasAutopilot()

        # The autopilot is not meant to be run on batches so we need to get control
        # inputs separately
        n_batch = x.size()[0]
        u = torch.zeros((n_batch, self.n_controls)).type_as(x)

        x_np = x.cpu().detach().numpy()
        for batch in range(n_batch):
            # The GCAS autopilot is implemented as a state machine that first rolls and
            # then pulls up. Here we unwrap the state machine logic to get a simpler
            # mapping from state to control

            # If the plane is not hurtling towards the ground, no need to do anything.
            if gcas.is_nose_high_enough(x_np[batch, :]) or gcas.is_above_flight_deck(
                x_np[batch, :]
            ):
                continue

            # If we are hurtling towards the ground and the plane isn't level, we need
            # to roll to get level
            if not gcas.is_roll_rate_low(x_np[batch, :]) or not gcas.are_wings_level(
                x_np[batch, :]
            ):
                u[batch, :] = torch.tensor(gcas.roll_wings_level(x_np[batch, :]))
                continue

            # If we are hurtling towards the ground and the plane IS level, then we need
            # to pull up
            u[batch, :] = torch.tensor(gcas.pull_nose_level()).type_as(u)

        return u
