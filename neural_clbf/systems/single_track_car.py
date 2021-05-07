"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
from torch.autograd.functional import jacobian
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, lqr

import neural_clbf.setup.commonroad as commonroad_loader  # type: ignore
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2  # type: ignore


# make sure that the import worked
assert commonroad_loader


class STCar(ControlAffineSystem):
    """
    Represents a car using the single-track model.

    The system has state defined relative to a reference path
    [x_ref, y_ref, psi_ref, v_ref, omega_ref, a_ref]

        x = [s_x - x_ref, s_y - y_ref, delta, v - v_ref, psi - psi_ref,
             psi_dot - psi_ref_dot, beta]

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
    N_DIMS = 7
    N_CONTROLS = 2

    # State indices
    SXE = 0
    SYE = 1
    DELTA = 2
    VE = 3
    PSI_E = 4
    PSI_DOT = 5
    BETA = 6
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
        super().__init__(nominal_params, dt)

        # Get car parameters
        self.car_params = parameters_vehicle2()

        if controller_dt is None:
            controller_dt = dt
        self.controller_dt = controller_dt

        # Compute the LQR gain matrix for the nominal parameters
        # Linearize the system about the x = 0, u = 0
        x0 = self.goal_point
        u0 = torch.zeros((1, self.n_controls)).type_as(x0)
        dynamics = lambda x: self.closed_loop_dynamics(
            x, u0, self.nominal_params
        ).squeeze()
        A = jacobian(dynamics, self.goal_point).squeeze().cpu().numpy()
        A = np.eye(self.n_dims) + self.controller_dt * A

        B = self._g(self.goal_point, self.nominal_params).squeeze().cpu().numpy()
        B = self.controller_dt * B

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self.K = torch.tensor(lqr(A, B, Q, R))

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
        return STCar.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [STCar.DELTA]

    @property
    def n_controls(self) -> int:
        return STCar.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[STCar.SXE] = 1.0
        upper_limit[STCar.SYE] = 1.0
        upper_limit[STCar.DELTA] = self.car_params.steering.max
        upper_limit[STCar.VE] = 1.0
        upper_limit[STCar.PSI_E] = np.pi / 2
        upper_limit[STCar.PSI_DOT] = 1.0
        upper_limit[STCar.BETA] = np.pi / 3

        lower_limit = -1.0 * upper_limit
        lower_limit[STCar.DELTA] = self.car_params.steering.min

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
        max_safe_tracking_error = 0.3
        tracking_error = x[
            :,
            [
                STCar.SXE,
                STCar.SYE,
                STCar.VE,
                STCar.PSI_E,
            ],
        ]
        tracking_error_small_enough = (
            tracking_error.norm(dim=-1) <= max_safe_tracking_error
        )
        safe_mask.logical_and_(tracking_error_small_enough)

        # We need slightly lower heading angle to be "close"
        psi_err_small_enough = x[:, STCar.PSI_E].abs() <= np.pi / 3
        safe_mask.logical_and_(psi_err_small_enough)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Avoid angles that are too large
        # Avoid tracking errors that are too large
        max_safe_tracking_error = 0.5
        tracking_error = x[
            :,
            [
                STCar.SXE,
                STCar.SYE,
                STCar.VE,
                STCar.PSI_E,
            ],
        ]
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
        return x.norm(dim=-1) / upper_limit.norm()

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        tracking_error = x[
            :,
            [
                STCar.SXE,
                STCar.SYE,
                STCar.VE,
                STCar.PSI_E,
            ],
        ]
        near_goal = tracking_error.norm(dim=-1) <= 0.3
        goal_mask.logical_and_(near_goal)

        # We need slightly lower heading angle to be "close"
        near_goal_psi = x[:, STCar.PSI_E].abs() <= 0.3
        goal_mask.logical_and_(near_goal_psi)

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
        v = x[:, STCar.VE] + v_ref
        psi_e = x[:, STCar.PSI_E]
        psi_dot = x[:, STCar.PSI_DOT]
        beta = x[:, STCar.BETA]
        delta = x[:, STCar.DELTA]
        sxe = x[:, STCar.SXE]
        sye = x[:, STCar.SYE]

        # set gravity constant
        g = 9.81  # [m/s^2]

        # create equivalent bicycle parameters
        mu = self.car_params.tire.p_dy1
        C_Sf = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        C_Sr = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        lf = self.car_params.a
        lr = self.car_params.b
        m = self.car_params.m
        Iz = self.car_params.I_z

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = v * torch.cos(psi_e + beta) - v_ref + omega_ref * sye
        dsye_r = v * torch.sin(psi_e + beta) - omega_ref * sxe

        f[:, STCar.SXE, 0] = dsxe_r
        f[:, STCar.SYE, 0] = dsye_r
        f[:, STCar.VE, 0] = -a_ref
        f[:, STCar.DELTA, 0] = 0.0
        f[:, STCar.PSI_E, 0] = psi_dot - omega_ref
        # Sorry this is a mess (it's ported from the commonroad models)
        f[:, STCar.PSI_DOT, 0] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * psi_dot
            + (mu * m / (Iz * (lr + lf)))
            * (lr * C_Sr * g * lf - lf * C_Sf * g * lr)
            * beta
            + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * g * lr) * delta
        )
        f[:, STCar.BETA, 0] = (
            (
                (mu / (v ** 2 * (lr + lf))) * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
                - 1
            )
            * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * g * lf + C_Sf * g * lr) * beta
            + mu / (v * (lr + lf)) * (C_Sf * g * lr) * delta
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

        # Extract the parameters
        v_ref = torch.tensor(params["v_ref"])

        # Extract the state variables and adjust for the reference
        v = x[:, STCar.VE] + v_ref
        psi_dot = x[:, STCar.PSI_DOT]
        beta = x[:, STCar.BETA]
        delta = x[:, STCar.DELTA]

        # create equivalent bicycle parameters
        mu = self.car_params.tire.p_dy1
        C_Sf = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        C_Sr = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        lf = self.car_params.a
        lr = self.car_params.b
        h = self.car_params.h_s
        m = self.car_params.m
        Iz = self.car_params.I_z

        g[:, STCar.DELTA, STCar.VDELTA] = 1.0
        g[:, STCar.VE, STCar.ALONG] = 1.0

        g[:, STCar.PSI_DOT, STCar.ALONG] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h)
            * psi_dot
            + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta
            - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta
        )
        g[:, STCar.BETA, STCar.ALONG] = (
            (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta
            - mu / (v * (lr + lf)) * C_Sf * h * delta
        )

        return g

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. For the inverted
        pendulum, the nominal controller is LQR

        args:
            x: bs x self.n_dims tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        if params is None or not self.validate_params(params):
            params = self.nominal_params

        # Compute the LQR gain matrix for the nominal parameters
        # create equivalent bicycle parameters
        g = 9.81  # [m/s^2]
        C_Sf = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        C_Sr = -self.car_params.tire.p_ky1 / self.car_params.tire.p_dy1
        lf = self.car_params.a
        lr = self.car_params.b

        # Linearize the system about the path
        x0 = self.goal_point
        x0[0, STCar.PSI_DOT] = params["omega_ref"]
        x0[0, STCar.DELTA] = (
            (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * params["omega_ref"]
            / params["v_ref"]
        )
        x0[0, STCar.DELTA] /= lf * C_Sf * g * lr
        u0 = torch.zeros((1, self.n_controls)).type_as(x0)
        dynamics = lambda x: self.closed_loop_dynamics(x, u0, params).squeeze()
        A = jacobian(dynamics, self.goal_point).squeeze().cpu().numpy()
        A = np.eye(self.n_dims) + self.controller_dt * A

        B = self._g(self.goal_point, self.nominal_params).squeeze().cpu().numpy()
        B = self.controller_dt * B

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self.K = torch.tensor(lqr(A, B, Q, R))

        # Compute nominal control from feedback + equilibrium control
        x_goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(self.K.type_as(x) @ (x - x_goal).T).T
        u_eq = torch.zeros_like(u_nominal)

        return u_nominal + u_eq
