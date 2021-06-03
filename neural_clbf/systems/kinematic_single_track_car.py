"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, lqr

import neural_clbf.setup.commonroad as commonroad_loader  # type: ignore
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2  # type: ignore


# make sure that the import worked
assert commonroad_loader


class KSCar(ControlAffineSystem):
    """
    Represents a car using the kinematic single-track model.

    The system has state defined relative to a reference path
    [x_ref, y_ref, psi_ref, v_ref, omega_ref, a_ref]

        x = [s_x - x_ref, s_y - y_ref, delta, v - v_ref, psi - psi_ref]

    where s_x and s_y are the x and y position, delta is the steering angle, v is the
    longitudinal velocity, and psi is the heading. The errors in x and y are expressed
    in the reference path frame

    The control inputs are

        u = [v_delta, a_long]

    representing the steering effort (change in delta) and longitudinal acceleration.

    The system is parameterized by a bunch of car-specific parameters, which we load
    from the commonroad model, and by the parameters of the reference point. Instead of
    viewing these as time-varying parameters, we can view them as bounded uncertainties,
    particularly in omega_ref and a_ref.
    """

    # Number of states and controls
    N_DIMS = 5
    N_CONTROLS = 2

    # State indices
    SXE = 0
    SYE = 1
    DELTA = 2
    VE = 3
    PSI_E = 4
    # Control indices
    VDELTA = 0
    ALONG = 1

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the car model.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["psi_ref", "v_ref", "a_ref",
                            "omega_ref"] (_c and _s denote cosine and sine)
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        # Get car parameters
        self.car_params = parameters_vehicle2()

        super().__init__(nominal_params, dt, controller_dt)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["psi_ref", "v_ref", "a_ref", "omega_ref"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "psi_ref" in params
        valid = valid and "v_ref" in params
        valid = valid and "a_ref" in params
        valid = valid and "omega_ref" in params

        return valid

    @property
    def n_dims(self) -> int:
        return KSCar.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [KSCar.DELTA]

    @property
    def n_controls(self) -> int:
        return KSCar.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[KSCar.SXE] = 1.5
        upper_limit[KSCar.SYE] = 1.5
        upper_limit[KSCar.DELTA] = self.car_params.steering.max
        upper_limit[KSCar.VE] = 1.5
        upper_limit[KSCar.PSI_E] = np.pi / 2

        lower_limit = -1.0 * upper_limit
        lower_limit[KSCar.DELTA] = self.car_params.steering.min

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor(
            [
                5.0,  # self.car_params.steering.v_max,
                self.car_params.longitudinal.a_max,
            ]
        )
        lower_limit = torch.tensor(
            [
                -5.0,  # self.car_params.steering.v_min,
                -self.car_params.longitudinal.a_max,
            ]
        )

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Avoid tracking errors that are too large
        max_safe_tracking_error = 0.5
        tracking_error = x
        tracking_error_small_enough = (
            tracking_error.norm(dim=-1) <= max_safe_tracking_error
        )
        safe_mask.logical_and_(tracking_error_small_enough)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Avoid angles that are too large
        # Avoid tracking errors that are too large
        max_safe_tracking_error = 1.0
        tracking_error = x
        tracking_error_too_big = tracking_error.norm(dim=-1) >= max_safe_tracking_error
        unsafe_mask.logical_or_(tracking_error_too_big)

        return unsafe_mask

    def distance_to_goal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distance from each point in x to the goal (positive for points
        outside the goal, negative for points inside the goal), normalized by the state
        limits.

        args:
            x: the points from which we calculate distance
        """
        upper_limit, _ = self.state_limits
        return (x.norm(dim=-1) - 0.25) / upper_limit.norm()

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        tracking_error = x
        near_goal = tracking_error.norm(dim=-1) <= 0.25
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

        # Extract the parameters
        v_ref = torch.tensor(params["v_ref"])
        a_ref = torch.tensor(params["a_ref"])
        omega_ref = torch.tensor(params["omega_ref"])

        # Extract the state variables and adjust for the reference
        v = x[:, KSCar.VE] + v_ref
        psi_e = x[:, KSCar.PSI_E]
        delta = x[:, KSCar.DELTA]
        sxe = x[:, KSCar.SXE]
        sye = x[:, KSCar.SYE]

        # Compute the dynamics
        wheelbase = self.car_params.a + self.car_params.b

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = v * torch.cos(psi_e) - v_ref + omega_ref * sye
        dsye_r = v * torch.sin(psi_e) - omega_ref * sxe

        f[:, KSCar.SXE, 0] = dsxe_r
        f[:, KSCar.SYE, 0] = dsye_r
        f[:, KSCar.VE, 0] = -a_ref
        f[:, KSCar.DELTA, 0] = 0.0
        f[:, KSCar.PSI_E, 0] = v / wheelbase * torch.tan(delta) - omega_ref

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

        g[:, KSCar.DELTA, KSCar.VDELTA] = 1.0
        g[:, KSCar.VE, KSCar.ALONG] = 1.0

        return g

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. For the inverted
        pendulum, the nominal controller is LQR

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        if params is None or not self.validate_params(params):
            params = self.nominal_params

        # Compute the LQR gain matrix

        # Linearize the system about the path
        wheelbase = self.car_params.a + self.car_params.b
        x0 = self.goal_point
        x0[0, KSCar.DELTA] = torch.atan(
            torch.tensor(params["omega_ref"] * wheelbase / params["v_ref"])
        )
        x0 = x0.type_as(x)
        A = np.zeros((self.n_dims, self.n_dims))
        A[KSCar.SXE, KSCar.SYE] = self.nominal_params["omega_ref"]
        A[KSCar.SXE, KSCar.VE] = 1

        A[KSCar.SYE, KSCar.SXE] = -self.nominal_params["omega_ref"]
        A[KSCar.SYE, KSCar.PSI_E] = self.nominal_params["v_ref"]

        A[KSCar.PSI_E, KSCar.VE] = torch.tan(x0[0, KSCar.DELTA]) / wheelbase
        A[KSCar.PSI_E, KSCar.DELTA] = self.nominal_params["v_ref"] / wheelbase

        A = np.eye(self.n_dims) + self.controller_dt * A

        B = np.zeros((self.n_dims, self.n_controls))
        B[KSCar.DELTA, KSCar.VDELTA] = 1.0
        B[KSCar.VE, KSCar.ALONG] = 1.0
        B = self.controller_dt * B

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self.K = torch.tensor(lqr(A, B, Q, R))

        # Compute nominal control from feedback + equilibrium control
        u_nominal = -(self.K.type_as(x) @ (x - x0).T).T
        u_eq = torch.zeros_like(u_nominal)
        u = u_nominal + u_eq

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u
