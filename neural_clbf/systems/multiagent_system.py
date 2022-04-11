"""Define a dynamical system for turtlebot and quadrotor multiagent system"""
# from msilib.schema import Control
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class Multiagent(ControlAffineSystem):
    """
    Represents a quadrotor and turtlebot working in tandem
    to complete a given task.

    The system has states

        x_tb = [x y theta]      turtlebot
        x_cf = [px py pz vx vy vz phi theta psi]  crazyflie
        X = [x_tb | x_cf]

    representing the pose and velocities of the turtlebot and quadrotor
    and it has control inputs 

        u = [f_turtlebot | f_crazyflie] 
        where:
            f_turtlebot = [v theta_dot]
            f_crazyflie = [F phi_dot theta_dot psi_dot]

    The system is parameterized by
        R: radius of the wheels of the turtlebot
        L: radius of rotation, or the distance between the two wheels
        m: mass of the crazyflie
    """

    #Number of states and controls 
    N_DIMS = 12
    N_CONTROLS = 6

    # State indices

    # Turtlebot
    X_T = 0
    Y_T = 1
    THETA_T = 2

    # Crazyflie
    PX = 3
    PY = 4
    PZ = 5
    VX = 6
    VY = 7
    VZ = 8
    PHI = 9
    THETA_C = 10
    PSI = 11

    # Control Indices

    #Turtlebot 
    V_T = 0
    THETA_DOT_T = 1

    # Crazyflie
    F = 2
    PHI_DOT = 3
    THETA_DOT_C = 4
    PSI_DOT = 5


    def __init__(
            self,
            nominal_params: Scenario,
            dt: float = 0.01,
            controller_dt: Optional[float] = None,
        ):
            """
            Initialize the system.

            args:
                nominal_params: a dictionary giving the parameter values for the system.
                                Requires keys ["m", "R", "L"]
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
        valid = valid and "R" in params
        valid = valid and "L" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["R"] > 0
        valid = valid and params["L"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Multiagent.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [Multiagent.THETA_T, Multiagent.PHI, Multiagent.THETA_C, Multiagent.PSI]

    @property
    def n_controls(self) -> int:
        return Multiagent.N_CONTROLS

    @property 
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)

        upper_limit[Multiagent.X_T] = 5.0
        upper_limit[Multiagent.Y_T] = 5.0
        upper_limit[Multiagent.THETA_T] = np.pi

        upper_limit[Multiagent.PX] = 4.0
        upper_limit[Multiagent.PY] = 4.0
        upper_limit[Multiagent.PZ] = 4.0
        upper_limit[Multiagent.VX] = 8.0
        upper_limit[Multiagent.VY] = 8.0
        upper_limit[Multiagent.VZ] = 8.0
        upper_limit[Multiagent.PHI] = np.pi / 2.0
        upper_limit[Multiagent.THETA_C] = np.pi / 2.0
        upper_limit[Multiagent.PSI] = np.pi / 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.N_CONTROLS)
        # TurtleBot
        upper_limit[Multiagent.V_T] = 3
        upper_limit[Multiagent.THETA_DOT_T] = 3.0 * np.pi
        # Crazyflie
        upper_limit[Multiagent.F] = 100
        upper_limit[Multiagent.PHI_DOT] = 50
        upper_limit[Multiagent.THETA_DOT_C] = 50
        upper_limit[Multiagent.PSI_DOT] = 50

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        
        For this multiagent task, the safe region for the crazyflie is
        defined as the conical region above the landing platform. 
                                ________
                                \      /
                                 \    /
                                  \__/
                where the max height is ___ above the platform 
                r_upper = ...
                r_lower = ...
                slant height = ...

                #TODO measure actual hardware, create landing platform (fixed to tb)
                # rough guess for now - 
        Safe landing is >= height of trailer platform i.e. 0 point
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # # We have a floor that we need to avoid and a radius we need to stay inside of
        safe_z = 0.1 #height of trailer
        safe_radius = 3 # cone shape - 
        safe_mask = torch.logical_and( 
            x[:, Multiagent.PZ] <= safe_z, x.norm(dim=-1) <= safe_radius
        )
        #how to add 
        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        
        TODO @bethlow Flipped safe_mask with buffer 
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        unsafe_z = 0.3
        unsafe_radius = 3.5
        unsafe_mask = torch.logical_or(
            x[:, Multiagent.PZ] >= unsafe_z, x.norm(dim=-1) >= unsafe_radius
        )

        return unsafe_mask

    def goal_point(self):
        goal_set = torch.zeros((1, self.n_dims))
        goal_set[:, Multiagent.PX] = -2.0
        goal_set[:, Multiagent.PY] = 0.0
        goal_set[:, Multiagent.PZ] = -2.0
        return goal_set

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        
        Goal region is a given point at which the turtlebot and crazflie will meet
        (x_goal, y_goal) with the crazyflie landing on trailer behind turtlebot 
        (offset x_goal by trailer length)

        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = x.norm(dim=-1) <= 0.3
        goal_mask.logical_and_(near_goal)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario): #TODO @bethlow
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

        # f is a zero vector for the turtlebot
        # as nothing should happen when no control input is given
        f[:, Multiagent.X_T, 0] = 0

        f[:, Multiagent.Y_T, 0] = 0

        f[:, Multiagent.THETA_T, 0] = 0

        # Derivatives of positions are just velocities
        f[:, Multiagent.PX] = x[:, Multiagent.VX]  # x
        f[:, Multiagent.PY] = x[:, Multiagent.VY]  # y
        f[:, Multiagent.PZ] = x[:, Multiagent.VZ]  # z

        # Constant acceleration in z due to gravity
        f[:, Multiagent.VZ] = grav

        # Orientation velocities are directly actuated

        return f

    def _g(self, x: torch.Tensor, params: Scenario): #TODO @bethlow
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

        # Extract the needed parameters
        m = params["m"]

        theta = x[:, Multiagent.THETA_T]

        # Effect on x
        g[:, Multiagent.X_T, Multiagent.V_T] = torch.cos(theta)

        # Effect on y
        g[:, Multiagent.Y_T, Multiagent.V_T] = torch.sin(theta)

        # Effect on theta
        g[:, Multiagent.THETA_T, Multiagent.THETA_DOT_T] = 1.0

        # Derivatives of linear velocities depend on thrust f
        s_theta = torch.sin(x[:, Multiagent.THETA_C])
        c_theta = torch.cos(x[:, Multiagent.THETA_C])
        s_phi = torch.sin(x[:, Multiagent.PHI])
        c_phi = torch.cos(x[:, Multiagent.PHI])
        g[:, Multiagent.VX, Multiagent.F] = -s_theta / m
        g[:, Multiagent.VY, Multiagent.F] = c_theta * s_phi / m
        g[:, Multiagent.VZ, Multiagent.F] = -c_theta * c_phi / m

        # Derivatives of all orientations are control variables
        g[:, Multiagent.PHI :, Multiagent.PHI_DOT :] = torch.eye(self.n_controls - 1)

        return g

    @property
    def u_eq(self): #TODO update? @bethlow
        u_eq = torch.zeros((1, self.n_controls))
        u_eq[0, Multiagent.F] = self.nominal_params["m"] * grav
        return u_eq

    #TODO @bethlow how to combine the two
    def u_nominal(self, x: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
        # Crazyflie linearization using LQR, but implemented here to allow 
        # turtlebot linerization override
        print("got to u_nom!")
        u = torch.zeros(x.shape[0], self.n_controls).type_as(x)

        # Insert A + B method calls
        A = Multiagent.compute_A_matrix
        B = Multiagent.compute_B_matrix
        # Add to u vector at crazyflie control indices
        Multiagent.compute_linearized_controller
        
        

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

        v_scaling = 1.0
        bot_to_origin = -x[:, : Multiagent.Y + 1].reshape(-1, 1, 2)
        theta = x[:, Multiagent.THETA]
        bot_facing = torch.stack((torch.cos(theta), torch.sin(theta))).T.unsqueeze(-1)
        u[:, Multiagent.V] = v_scaling * torch.bmm(bot_to_origin, bot_facing).squeeze()

        # In addition to setting the velocity towards the origin, we also need to steer
        # towards the origin. We can do this via P control on the angle between the
        # turtlebot and the vector to the origin.
        #
        # However, this angle becomes ill-defined as the bot approaches the origin, so
        # so we switch this term off if the bot is too close (and instead just control
        # theta to zero)
        phi_control_on = bot_to_origin.norm(dim=-1) >= 0.02
        phi_control_on = phi_control_on.reshape(-1)
        omega_scaling = 5.0
        angle_from_origin_to_bot = torch.atan2(x[:, Multiagent.Y_T], x[:, Multiagent.X_T])
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
        u[:, Multiagent.THETA_DOT] = -omega_scaling * theta
        u[phi_control_on, Multiagent.THETA_DOT] = -omega_scaling * phi[phi_control_on]

        # Clamp given the control limits
        u_upper, u_lower = self.control_limits
        u = torch.clamp(u, u_lower.type_as(u), u_upper.type_as(u))

        return u