"""
pusher_slider.py
Description:
    Define a dynamical system for the Pusher-Slider System (a common manipulation benchmark)
"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .hybrid_control_affine_system import HybridControlAffineSystem
from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
    lqr,
    robust_continuous_lyap,
    continuous_lyap,
)
from neural_clbf.systems.pusher_slider_parameters import PusherSliderParameters


class PusherSlider(HybridControlAffineSystem):
    """
    Represents a car using the Pusher-Slider Model developed by  model.

    The state is

        x = [
            s_x,
            s_y,
            s_theta,
            p_y
        ]

    where s_x and s_y are the x and y position of the slider, s_theta is the angle of the slider with respect to
    the x-axis, and p_y is the position of the pusher along the side of the slider.

    It is assumed that the pushing finger is always in contact with the slider object.

    The control inputs are

        u = [v_n, v_t]

    References:
        Hogan, F.R., Rodriguez, A. (2020).
            Feedback Control of the Pusher-Slider System: A Story of Hybrid and Underactuated Contact Dynamics.
            In: Goldberg, K., Abbeel, P., Bekris, K., Miller, L. (eds) Algorithmic Foundations of Robotics XII. Springer Proceedings in Advanced Robotics, vol 13. Springer, Cham. https://doi.org/10.1007/978-3-030-43089-4_51

        

    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 2
    N_MODES = 3

    # State indices
    S_X = 0
    S_Y = 1
    S_THETA = 2
    P_Y = 3
    # Control indices
    V_N = 0
    V_T = 1

    def __init__(
            self,
            rc_scenario: Scenario,
            dt: float = 0.01,
            controller_dt: Optional[float] = None,
    ):
        """
        Initialize the pusher-slider model.

        args:
            rc_scenario: a dictionary giving the parameter values for the system.
                            Requires keys ["s_x_ref", "s_y_ref", "bar_radius"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if rc_scenario are not valid for this system
        """

        self.manip_params = PusherSliderParameters()
        self.p_y = 0.0  # Make this constant

        # Then initialize
        super().__init__(
            rc_scenario, dt, controller_dt, use_linearized_controller=False, n_modes=PusherSlider.N_MODES
        )

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        self.P = torch.eye(self.n_dims)

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
        valid = valid and "s_x_ref" in params
        valid = valid and "s_y_ref" in params

        return valid

    @property
    def n_dims(self) -> int:
        return PusherSlider.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [PusherSlider.S_THETA]

    @property
    def n_controls(self) -> int:
        return PusherSlider.N_CONTROLS

    @property
    def n_modes(self) -> int:
        return PusherSlider.N_MODES

    @property
    def goal_point(self):
        goal = torch.zeros((1, self.n_dims))
        goal[:, PusherSlider.S_X] = self.nominal_params["s_x_ref"]
        goal[:, PusherSlider.S_Y] = self.nominal_params["s_y_ref"]

        return goal

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[PusherSlider.S_X] = 5.0
        upper_limit[PusherSlider.S_Y] = 5.0
        upper_limit[PusherSlider.S_THETA] = np.pi / 2
        # upper_limit[PusherSlider.P_Y] = self.manip_params.s_width/2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = torch.ones(self.n_controls)
        upper_limit[PusherSlider.V_N] = 15.0
        upper_limit[PusherSlider.V_T] = 10.0

        lower_limit = torch.zeros(self.n_controls)
        lower_limit[PusherSlider.V_T] = -upper_limit[PusherSlider.V_T]

        return upper_limit, lower_limit

    def safe_mask(self, x):
        """
        Description:
            Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # We have to avoid a bar at (0, 1) with some radius
        # Bar parameters
        bar_x = 0.0
        bar_y = 0.0
        bar_radius = self.rc_scenario['bar_radius']
        safety_factor = 1.5

        # Get position of head of segway
        s_x = x[:, PusherSlider.S_X]
        s_y = x[:, PusherSlider.S_Y]

        # Compute distance to the bar and make sure it's greater than the bar radius
        distance_to_bar = (s_x - bar_x) ** 2 + (s_y - bar_y) ** 2
        distance_to_bar = torch.sqrt(distance_to_bar)
        safe_mask = torch.logical_and(
            safe_mask, distance_to_bar >= safety_factor * bar_radius
        )

        #safe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have to avoid a bar at (0, 1) with some radius
        # Bar parameters
        bar_x = 0
        bar_y = 0
        bar_radius = self.rc_scenario['bar_radius']
        safety_factor = 1.5

        # Get position of head of segway
        s_x = x[:, PusherSlider.S_X]
        s_y = x[:, PusherSlider.S_Y]

        # Compute distance to the bar and make sure it's greater than the bar radius
        distance_to_bar = (s_x - bar_x) ** 2 + (s_y - bar_y) ** 2
        distance_to_bar = torch.sqrt(distance_to_bar)
        unsafe_mask = torch.logical_and(
            unsafe_mask, distance_to_bar <= safety_factor * bar_radius
        )

        # unsafe_mask = torch.zeros_like(x[:,0], dtype=torch.bool)

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
        goal_mask = torch.logical_and(
            goal_mask, near_goal
        )
        # goal_mask.logical_and_(near_goal)

        # The goal set has to be a subset of the safe set
        temp_safe_mask = self.safe_mask(x)
        goal_mask = torch.logical_and(
            goal_mask, temp_safe_mask
        )
        # goal_mask.logical_and_(self.safe_mask(x))  # This is technically in-place?

        # goal_mask = torch.zeros_like(x[:,0], dtype=torch.bool)

        return goal_mask

    def _f_all(self, x: torch.Tensor, u: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x self.n_modes tensor
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, self.n_modes))
        f = f.type_as(x)

        return f

    def get_motion_cone_vectors(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        get_motion_cone_vectors
        Description:
            This function should compute the boundaries that, if crossed, will cause
            the slider to move up (positive) or down (negative) along the edge of the slider.
        Returns:
            gamma_pos: bs x 1 tensor describing the upper motion cone for each of the states in the batch
            gamma_neg: bs x 1 tensor describing the lower motion cone for each of the states in the batch
        """

        # Constants
        slider_params = self.manip_params
        mu_ps = slider_params.ps_cof

        g = 9.81 # m/s^2

        f_max = slider_params.st_cof * slider_params.s_mass * g
        tau_max = slider_params.st_cof * slider_params.s_mass * g * (2.0*slider_params.s_width / 3.0)   # The last term is meant to come from
                                                                                                        # a sort of mass distribution/moment calculation.
        c = tau_max / f_max

        p_x = slider_params.p_x
        p_y = x[:, PusherSlider.P_Y]

        # Algorithm
        gamma_pos = (mu_ps * c ** 2 + (p_x * mu_ps - p_y) * p_x) / (c ** 2 - (p_x * mu_ps - p_y) * p_y)
        gamma_neg = (-mu_ps * c ** 2 + (- p_x * mu_ps - p_y) * p_x) / (c ** 2 - (p_x * mu_ps + p_y) * p_y)

        return gamma_pos, gamma_neg

    def identify_mode(self, x: torch.Tensor, u: torch.Tensor):
        """
        identify_mode
        Description:
            Determines which mode the pusher slider is going to be in.
        Usage:
            mode_tensor = self.identify_mode(x, u)
        Returns:
            mode_matrix: bs
        """
        # Constants
        batch_size = x.shape[0]
        v_n = u[:, PusherSlider.V_N]
        v_t = u[:, PusherSlider.V_T]

        # Algorithm
        gamma_pos, gamma_neg = self.get_motion_cone_vectors(x)

        # Compare the current input values with the cone vectors

        # Sticking
        sticking_mask = torch.ones((batch_size, 1 , 1), dtype=torch.bool)
        # sticking_mask = torch.ones_like(x[:, 0], dtype=torch.bool) # labels each
        sticking_mask = torch.logical_and(
            sticking_mask, v_t <= gamma_pos * v_n
        )
        sticking_mask = torch.logical_and(
            sticking_mask, v_t >= gamma_neg * v_n
        )

        # Sliding Up (Positive)
        sliding_up_mask = torch.ones((batch_size, 1 , 1), dtype=torch.bool)
        #sliding_up_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        sliding_up_mask = torch.logical_and(
            sliding_up_mask, v_t > gamma_pos * v_n
        )

        # Sliding Down (Negative)
        sliding_down_mask = torch.ones((batch_size, 1, 1), dtype=torch.bool)
        # sliding_down_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        sliding_down_mask = torch.logical_and(
            sliding_down_mask, v_t < gamma_neg * v_n
        )

        # Return
        mode_matrix = torch.zeros((batch_size, PusherSlider.N_MODES, 1))
        mode_matrix[:, :, :] = torch.hstack((sticking_mask, sliding_up_mask, sliding_down_mask))
        return mode_matrix

    def _g_all(self, x: torch.Tensor, u: torch.Tensor, params: Scenario):
        # Constants

        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g_all = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_modes))
        g_all = g_all.type_as(x)

        g = 9.81  # [m/s^2] # gravity

        st_cof = self.manip_params.st_cof
        s_mass = self.manip_params.s_mass
        s_width = self.manip_params.s_width

        f_max = st_cof * s_mass * g
        m_max = st_cof * s_mass * g * (s_width / 2.0)
        c = m_max / f_max

        p_x = self.manip_params.p_x

        # set parameters
        # (There are none right now)

        # Extract the state variables and adjust for the reference
        s_x = x[:, PusherSlider.S_X]
        s_y = x[:, PusherSlider.S_Y]
        s_theta = x[:, PusherSlider.S_THETA]
        p_y = self.p_y  # p_y = x[:, PusherSlider.P_Y]

        # Compute the elements of the product Q*P
        R_theta = torch.zeros((batch_size, 2, 2))
        R_theta[:, 0, 0] = torch.cos(s_theta)
        R_theta[:, 0, 1] = torch.sin(s_theta)
        R_theta[:, 1, 0] = -torch.sin(s_theta)
        R_theta[:, 1, 1] = torch.cos(s_theta)

        Q = torch.zeros((batch_size, 2, 2))
        Q[:, 0, 0] = c**2 + p_x**2
        Q[:, 0, 1] = p_x * p_y
        Q[:, 1, 0] = p_x * p_y
        Q[:, 1, 1] = c**2 + p_y**2

        # Compute the Mode dependent matrices
        gamma_pos, gamma_neg = self.get_motion_cone_vectors(x)

        P1 = torch.zeros((batch_size, 2, 2))
        P1[:, 0, 0] = 1.0
        P1[:, 1, 1] = 1.0

        P2 = torch.zeros((batch_size, 2, 2))
        P2[:, 0, 0] = 1.0
        P2[:, 1, 0] = gamma_pos

        P3 = torch.zeros((batch_size, 2, 2))
        P3[:, 0, 0] = 1.0
        P3[:, 1, 0] = gamma_neg

        b = torch.zeros((batch_size, 1, 2))
        b[:, 0, 0] = -p_y
        b[:, 0, 1] = p_x

        c1 = torch.zeros((batch_size, 1, 2))

        c2 = torch.zeros((batch_size, 1, 2)) # sliding up (positive)
        c2[:, 0, 0] = -gamma_pos
        c2[:, 0, 0] = 1

        c3 = torch.zeros((batch_size, 1, 2))  # sliding down (negative)
        c3[:, 0, 0] = -gamma_neg
        c3[:, 0, 0] = 1

        # Compose all of these into a single vector.
        RQ = torch.matmul(R_theta,Q)
        g1 = torch.hstack(
            ( torch.matmul(RQ, P1), (1/(c ** 2 + p_x**2 + p_y**2)) * torch.matmul(b, P1), c1)
        )

        g2 = torch.hstack(
            ( torch.matmul(RQ, P2), (1/(c ** 2 + p_x**2 + p_y**2)) * torch.matmul(b, P2), c2)
        )

        g3 = torch.hstack(
            (torch.matmul(RQ, P3), (1 / (c ** 2 + p_x ** 2 + p_y ** 2)) * torch.matmul(b, P3), c3)
        )

        # Create output
        g_all[:, :, :, 0] = g1
        g_all[:, :, :, 1] = g2
        g_all[:, :, :, 2] = g3

        return g_all

    # def _g(self, x: torch.Tensor, u: torch.Tensor, params: Scenario):
    #     """
    #     Return the control-independent part of the control-affine dynamics.
    #
    #     args:
    #         x: bs x self.n_dims tensor of state
    #         params: a dictionary giving the parameter values for the system. If None,
    #                 default to the nominal parameters used at initialization
    #     returns:
    #         g_x: bs x self.n_dims x self.n_controls tensor
    #     """
    #
    #     # Constants
    #
    #     # Algorithm
    #     mode_tensor = self.identify_mode(x, u)
    #
    #     if curr_mode == 'Sticking':
    #         return self._g1(x, u)
    #     elif curr_mode == 'SlidingUp':
    #         return self._g2(x, u)
    #     elif curr_mode == 'SlidingDown':
    #         return self._g3(x, u)
    #     else:
    #         raise (Exception("There was a problem with identifying the current mode! Unexpected mode = " + curr_mode))
    #
    #     # Extract batch size and set up a tensor for holding the result
    #     batch_size = x.shape[0]
    #     g_x = torch.zeros((batch_size, self.n_dims, self.n_controls))
    #     g_x = g_x.type_as(x)
    #
    #     # Extract the parameters
    #
    #     # constants
    #     g = 9.81  # [m/s^2] # gravity
    #
    #     st_cof = self.manip_params.st_cof
    #     s_mass = self.manip_params.s_mass
    #     s_width = self.manip_params.s_width
    #
    #     f_max = st_cof * s_mass * g
    #     m_max = st_cof * s_mass * g * (s_width / 2.0)
    #     c = m_max / f_max
    #
    #     p_x = self.manip_params.p_x
    #
    #     # set parameters
    #     # (There are none right now)
    #
    #     # Extract the state variables and adjust for the reference
    #     s_x = x[:, PusherSlider.S_X]
    #     s_y = x[:, PusherSlider.S_Y]
    #     s_theta = x[:, PusherSlider.S_THETA]
    #     p_y = self.p_y # p_y = x[:, PusherSlider.P_Y]
    #
    #     # Compute the elements of the product Q*P
    #     QP00 = (1 / (c ** 2 + p_x ** 2 + p_y ** 2)) * (c ** 2 + p_x ** 2)  # (0,0)-th element of Pu product
    #     QP01 = (1 / (c ** 2 + p_x ** 2 + p_y ** 2)) * p_x * p_y  # (0,1)-th element of Pu product
    #     QP10 = (1 / (c ** 2 + p_x ** 2 + p_y ** 2)) * p_x * p_y  # (1,0)-th element of Pu product
    #     QP11 = (1 / (c ** 2 + p_x ** 2 + p_y ** 2)) * (c ** 2 + p_y ** 2)  # (1,0)-th element of Pu product
    #
    #     # Compute the elements of the product CQP (determines the first few rows of the g() matrix
    #     g_x[:, PusherSlider.S_X, PusherSlider.V_N] = torch.cos(s_theta) * QP00 + torch.sin(
    #         s_theta) * QP10
    #     g_x[:, PusherSlider.S_X, PusherSlider.V_T] = torch.cos(s_theta) * QP01 + torch.sin(
    #         s_theta) * QP11
    #     g_x[:, PusherSlider.S_Y, PusherSlider.V_N] = -torch.sin(s_theta) * QP00 + torch.cos(
    #         s_theta) * QP10
    #     g_x[:, PusherSlider.S_Y, PusherSlider.V_T] = -torch.sin(s_theta) * QP01 + torch.cos(
    #         s_theta) * QP11
    #
    #     g_x[:, PusherSlider.S_THETA, PusherSlider.V_N] = -p_y / (c ** 2 + p_x ** 2 + p_y ** 2)
    #     g_x[:, PusherSlider.S_THETA, PusherSlider.V_T] = p_x
    #
    #     # Return g(x)
    #     return g_x

    # def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
    #     """
    #     Computes the linearized controller K and lyapunov matrix P.
    #     """
    #     # We need to compute the LQR closed-loop linear dynamics for each scenario
    #     Acl_list = []
    #     # Default to the nominal scenario if none are provided
    #     if scenarios is None:
    #         scenarios = [self.rc_scenario]
    #
    #     # For each scenario, get the LQR gain and closed-loop linearization
    #     for s in scenarios:
    #         # Compute the LQR gain matrix for the nominal parameters
    #         Act, Bct = self.linearized_ct_dynamics_matrices(s)
    #         A, B = self.linearized_dt_dynamics_matrices(s)
    #
    #         # Define cost matrices as identity
    #         Q = np.eye(self.n_dims - 1)
    #         R = np.eye(self.n_controls)
    #
    #         # Get feedback matrix
    #         A_sub, B_sub = A[:self.N_DIMS - 1, :self.N_DIMS - 1], B[:self.N_DIMS - 1, :]
    #         K_np0 = lqr(A_sub, B_sub, Q, R)
    #         K_np = np.hstack((K_np0, np.zeros((self.N_CONTROLS, 1))))
    #         # print(K_np)
    #         self.K = torch.tensor(K_np)
    #
    #         Acl_list.append(Act - Bct @ K_np)
    #
    #     # If more than one scenario is provided...
    #     # get the Lyapunov matrix by robustly solving Lyapunov inequalities
    #     if len(scenarios) > 1:
    #         self.P = torch.eye(self.N_DIMS)  # torch.tensor(robust_continuous_lyap(Acl_list, Q))
    #     else:
    #         # Otherwise, just use the standard Lyapunov equation
    #         self.P = torch.eye(self.N_DIMS)  # torch.tensor(continuous_lyap(Acl_list[0], Q))
    #
    # def nominal_simulator(self, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
    #     """
    #     Simulate the system forward using the nominal controller
    #
    #     args:
    #         x_init - bs x n_dims tensor of initial conditions
    #         num_steps - a positive integer
    #     returns
    #         a bs x num_steps x self.n_dims tensor of simulated trajectories
    #     """
    #     # Call the simulate method using the nominal controller
    #     x_init = x_init + self.goal_point.type_as(x_init)
    #     return self.simulate(
    #         x_init, num_steps, self.u_nominal, guard=self.out_of_bounds_mask
    #     )

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
            params = self.rc_scenario

        # Compute the LQR gain matrix

        # Linearize the system about the path
        x0 = self.goal_point
        # x0[0, PusherSlider.S_THETA] = torch.atan(
        #     torch.tensor(params["omega_ref"] * wheelbase / params["v_ref"])
        # )
        # x0 = x0.type_as(x)

        # Compute the LQR gain matrix for the nominal parameters
        # Act, Bct = self.linearized_ct_dynamics_matrices(x0)
        A, B = self.linearized_dt_dynamics_matrices(x0)

        # Define cost matrices as identity
        Q = np.eye(self.n_dims - 1)
        R = np.eye(self.n_controls)

        A_sub, B_sub = A[:self.N_DIMS - 1, :self.N_DIMS - 1], B[:self.N_DIMS - 1, :]
        K_np0 = lqr(A_sub, B_sub, Q, R)

        K_np1 = np.zeros((self.N_CONTROLS, 1))
        K_np = np.hstack((K_np0, K_np1))
        # print(K_np)
        self.K = torch.tensor(K_np)

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
