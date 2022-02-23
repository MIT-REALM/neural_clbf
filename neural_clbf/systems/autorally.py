"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario
from neural_clbf.systems.car_parameters import VehicleParameters


class AutoRally(ControlAffineSystem):
    """
    Represents a car using the AutoRally model.

    The system has state defined relative to a reference path
    [x_ref, y_ref, psi_ref, v_ref, psidot_ref, ax_ref]. We assume that the reference
    path is not sliding in the y direction (i.e. 2nd order dubins car dynamics).

    The relative state is

        x = [
            p_x - x_ref,
            p_y - y_ref,
            psi - psi_ref,
            delta,
            omega_front - omega_front_ref,
            omega_rear - omega_front_ref,
            vx,
            vy,
            psi_dot - psi_ref_dot,
        ]

    where p_x and p_y are the x and y position, delta is the steering angle, vx is the
    longitudinal velocity, psi is the heading, and vy is the transverse velocity.
    The errors in x and y are expressed in the reference path frame.
    Angular wheel speeds are expressed relative to the speed needed to track v_ref with
    no slip.

    The control inputs are

        u = [v_delta, omega_rear_dot]

    representing the steering effort (change in delta) and angular acceleration of
    the real wheel. In practice, omega_rear_dot will need to be converted to a throttle
    or brake command by inverting the rear wheel dynamics model.

    The system is parameterized by a bunch of car-specific parameters and by the
    parameters of the reference path. Instead of viewing these as time-varying
    parameters, we can view them as bounded uncertainties, particularly in
    psidot_ref and a_ref.
    """

    # Number of states and controls
    N_DIMS = 9
    N_CONTROLS = 2

    # State indices
    SXE = 0
    SYE = 1
    PSI_E = 2
    DELTA = 3
    OMEGA_F_E = 4
    OMEGA_R_E = 5
    VX = 6
    VY = 7
    PSI_E_DOT = 8
    # Control indices
    VDELTA = 0
    OMEGA_R_E_DOT = 1

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
                            Requires keys ["psi_ref", "v_ref", "omega_ref"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        # Get car parameters
        self.car_params = VehicleParameters()

        # Then initialize
        super().__init__(
            nominal_params, dt, controller_dt, use_linearized_controller=True
        )

        # self.P *= 0.1

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["psi_ref", "v_ref", "omega_ref"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "psi_ref" in params
        valid = valid and "v_ref" in params
        valid = valid and "omega_ref" in params

        return valid

    @property
    def n_dims(self) -> int:
        return AutoRally.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [AutoRally.DELTA]

    @property
    def n_controls(self) -> int:
        return AutoRally.N_CONTROLS

    @property
    def goal_point(self):
        goal = torch.zeros((1, self.n_dims))
        goal[:, AutoRally.VX] = self.nominal_params["v_ref"]

        return goal

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[AutoRally.SXE] = 1.0
        upper_limit[AutoRally.SYE] = 1.0
        upper_limit[AutoRally.DELTA] = 1.0  # TODO confirm with AutoRally team
        upper_limit[AutoRally.VX] = 10.0
        upper_limit[AutoRally.VY] = 3.0
        upper_limit[AutoRally.OMEGA_F_E] = 20.0  # Relative to desired
        upper_limit[AutoRally.OMEGA_R_E] = 20.0  # Relative to desired
        upper_limit[AutoRally.PSI_E] = np.pi / 2
        upper_limit[AutoRally.PSI_E_DOT] = np.pi / 2

        lower_limit = -1.0 * upper_limit
        lower_limit[AutoRally.VX] = 0.0

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = torch.ones(self.n_controls)
        upper_limit[AutoRally.VDELTA] = 5.0
        upper_limit[AutoRally.OMEGA_R_E_DOT] = 10

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Avoid tracking errors that are too large
        max_safe_tracking_error = 0.5
        tracking_error = x - self.goal_point.type_as(x)
        tracking_error = tracking_error[:, : AutoRally.PSI_E + 1]
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
        max_safe_tracking_error = 0.8
        tracking_error = x - self.goal_point.type_as(x)
        tracking_error = tracking_error[:, : AutoRally.PSI_E + 1]
        tracking_error_too_big = tracking_error.norm(dim=-1) >= max_safe_tracking_error
        unsafe_mask.logical_or_(tracking_error_too_big)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the origin
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
        omega_ref = torch.tensor(params["omega_ref"])

        # set gravity constant
        g = 9.81  # [m/s^2]

        # set parameters
        m_kg = 21.76
        Iz_kgm2 = 1.12
        Iwf_kgm2 = 0.05
        lf_m = 0.34
        lr_m = 0.23
        rf_m = 0.095
        rr_m = 0.09
        h_m = 0.12
        B = 4.0
        C = 1.0
        D = 1.0

        # Extract the state variables and adjust for the reference
        vx = x[:, AutoRally.VX]
        vy = x[:, AutoRally.VY]
        omega_f = x[:, AutoRally.OMEGA_F_E] + v_ref / rf_m
        omega_r = x[:, AutoRally.OMEGA_R_E] + v_ref / rr_m
        psi_e = x[:, AutoRally.PSI_E]
        psi_e_dot = x[:, AutoRally.PSI_E_DOT]
        psi_dot = psi_e_dot + omega_ref
        delta = x[:, AutoRally.DELTA]
        sxe = x[:, AutoRally.SXE]
        sye = x[:, AutoRally.SYE]

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = vx * torch.cos(psi_e) - vy * torch.sin(psi_e) - v_ref + omega_ref * sye
        dsye_r = vx * torch.sin(psi_e) + vy * torch.cos(psi_e) - omega_ref * sxe
        f[:, AutoRally.SXE, 0] = dsxe_r
        f[:, AutoRally.SYE, 0] = dsye_r

        f[:, AutoRally.PSI_E, 0] = psi_e_dot  # integrate
        f[:, AutoRally.OMEGA_R_E, 0] = 0.0  # actuated
        f[:, AutoRally.DELTA, 0] = 0.0  # actuated

        # Compute front and rear coefficients of friction
        # This starts with wheel speeds
        v_fx = (
            vx * torch.cos(delta)
            + vy * torch.sin(delta)
            + psi_dot * lf_m * torch.sin(delta)
        )
        v_fy = (
            vy * torch.cos(delta)
            - vx * torch.sin(delta)
            + psi_dot * lf_m * torch.cos(delta)
        )
        v_rx = vx
        v_ry = vy - psi_dot * lr_m

        # From that, get longitudinal and lateral slip
        sigma_fx = (v_fx - omega_f * rf_m) / (1e-3 + omega_f * rf_m)
        sigma_fy = v_fy / (1e-3 + omega_f * rf_m)
        sigma_rx = (v_rx - omega_r * rr_m) / (1e-3 + omega_r * rr_m)
        sigma_ry = v_ry / (1e-3 + omega_r * rr_m)
        # And slip magnitude
        sigma_f = torch.sqrt(1e-5 + sigma_fx ** 2 + sigma_fy ** 2)
        sigma_r = torch.sqrt(1e-5 + sigma_rx ** 2 + sigma_ry ** 2)

        # These let us get friction coefficients
        mu_f = D * torch.sin(C * torch.arctan(B * sigma_f))
        mu_r = D * torch.sin(C * torch.arctan(B * sigma_r))

        # Decompose friction into longitudinal and lateral
        mu_fx = -sigma_fx / (sigma_f + 1e-3) * mu_f
        mu_fy = -sigma_fy / (sigma_f + 1e-3) * mu_f
        mu_rx = -sigma_rx / (sigma_r + 1e-3) * mu_r
        mu_ry = -sigma_ry / (sigma_r + 1e-3) * mu_r

        # Compute vertical forces on the front and rear wheels
        f_fz = (m_kg * g * lr_m - m_kg * g * mu_rx * h_m) / (
            lf_m
            + lr_m
            + mu_fx * h_m * torch.cos(delta)
            - mu_fy * h_m * torch.sin(delta)
            - mu_rx * h_m
        )
        f_rz = m_kg * g - f_fz

        # Get longitudinal and lateral wheel forces from vertical forces and friction
        f_fx = f_fz * mu_fx
        f_fy = f_fz * mu_fy
        f_rx = f_rz * mu_rx
        f_ry = f_rz * mu_ry

        # Use these to compute derivatives
        f[:, AutoRally.OMEGA_F_E, 0] = -rf_m / Iwf_kgm2 * f_fx

        f[:, AutoRally.PSI_E_DOT, 0] = (
            (f_fy * torch.cos(delta) + f_fx * torch.sin(delta)) * lf_m - f_ry * lr_m
        ) / Iz_kgm2 - omega_ref

        # Changes in vx and vy have three components. One due to the dynamics of the
        # car, one due to the rotation of the car,
        # and one due to the fact that the reference frame is rotating.

        # Dynamics
        vx_dot = (f_fx * torch.cos(delta) - f_fy * torch.sin(delta) + f_rx) / m_kg
        vy_dot = (f_fx * torch.sin(delta) + f_fy * torch.cos(delta) + f_ry) / m_kg

        # Car rotation and frame rotation
        vx_dot += vy * psi_e_dot
        vy_dot += -vx * psi_e_dot

        f[:, AutoRally.VX, 0] = vx_dot
        f[:, AutoRally.VY, 0] = vy_dot

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

        # Steering angle delta and rear wheel angular acceleration are controlled,
        # everything else is pure drift dynamics
        g[:, AutoRally.DELTA, AutoRally.VDELTA] = 1.0
        g[:, AutoRally.OMEGA_R_E, AutoRally.OMEGA_R_E_DOT] = 1.0

        return g

    def nominal_simulator(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Simulate the system forward using the nominal controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            num_steps - a positive integer
        returns
            a bs x num_steps x self.n_dims tensor of simulated trajectories
        """
        # Call the simulate method using the nominal controller
        x_init = x_init + self.goal_point.type_as(x_init)
        return self.simulate(
            x_init, num_steps, self.u_nominal, guard=self.out_of_bounds_mask
        )
