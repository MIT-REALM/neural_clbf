"""Define a dymamical system for Crazyflies"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class Crazyflie(ControlAffineSystem):
    """
    Represents a quadcopter in 3D space, the crazyflie.
    The system has state
        p = [x, y, z, vx, vy, vz]
    representing the x,y,z positions and velocities of the crazyflie
    and it has control inputs
        u = [theta, phi, psi, f]
    representing the desired roll, pitch, yaw, and net rotor thrust
    
    phi = rotation about x axis
    theta = rotation about y axis
    psi = rotation about z axis
    
    The system is parameterized by
        m: mass
        
    Note: z is positive in the direction of gravity
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 4

    # State indices
    X = 0
    Y = 1
    Z = 2
    
    VX = 3
    VY = 4
    VZ = 5
    
    # Control indices
    F = 0
    PHI = 1
    THETA = 2
    PSI = 3

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the Crazyflie.
        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios,
            use_linearized_controller=False
        )

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
        return Crazyflie.N_DIMS
    
    # angles are a control input, so theoretically this pulls the angle values from input? Other models have angles as a state variable 
    # so I'm not so sure that this will actually work
    @property
    def angle_dims(self) -> List[int]:
        return [Crazyflie.theta, Crazyflie.phi, Crazyflie.psi]

    @property
    def n_controls(self) -> int:
        return Crazyflie.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        
        # copied the values from quad3d, but not sure on a justification for any values
        upper_limit[Crazyflie.X] = 4.0
        upper_limit[Crazyflie.Y] = 4.0
        upper_limit[Crazyflie.Z] = 4.0
        upper_limit[Crazyflie.VX] = 8.0
        upper_limit[Crazyflie.VY] = 8.0
        upper_limit[Crazyflie.VZ] = 8.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        # TODO @bethlow these are relaxed for now, but eventually
        # these values should be measured on the hardware.
        
        # unsure on justification for net force upper limit; copied from quad3d
        # upper limits: force, phi, theta, psi
        # should psi limit be either 2*pi or boundless? 
        upper_limit = torch.tensor([100, np.pi.2, np,pi/2, np.pi/2])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:,0], dtype=torch.bool)
        
        # We have a floor that we need to avoid and a radius we need to stay inside of
        safe_z = 0.0
        # safe radius probably can be modified depending on what we need; placeholder value for now that was copied from quad3d
        safe_radius = 3
        
        # note that direction of gravity is positive, so all points above the ground have negative z component
        safe_mask = torch.logical_and(
            x[:, Crazyflie.Z] <= safe_z, x.norm(dim=-1) <= safe_radius
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
        
        # note that direction of gravity is positive, so all points above the ground have negative z component
        unsafe_mask = torch.logical_or(
            x[:, Crazyflie.Z] >= unsafe_z, x.norm(dim=-1) >= unsafe_radius
        )

    def distance_to_goal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distance from each point in x to the goal (positive for points
        outside the goal, negative for points inside the goal), normalized by the state
        limits.
        args:
            x: the points from which we calculate distance
        """
        # this was for turtlebot, not sure if modification is needed for crazyflie
        upper_limit, _ = self.state_limits
        return x.norm(dim=-1) / upper_limit.norm()

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set
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

        # f is a zero vector as nothing should happen when no control input is given
        f[:, TurtleBot.X, 0] = 0

        f[:, TurtleBot.Y, 0] = 0

        f[:, TurtleBot.THETA, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
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
        R, L = params["R"], params["L"]
        # and state variables
        theta = x[:, TurtleBot.THETA]

        # Tensor for wheel velocities
        v = torch.zeros((2, 2))
        v = v.type_as(x)

        # Building tensor v
        v[0, 0] = 1 / R
        v[1, 0] = 1 / R
        v[0, 1] = L / (2 * R)
        v[1, 1] = -L / (2 * R)

        # Effect on x
        g[:, TurtleBot.X, TurtleBot.V] = R / 2 * torch.cos(theta)
        g[:, TurtleBot.X, TurtleBot.THETA_DOT] = R / 2 * torch.cos(theta)

        # Effect on y
        g[:, TurtleBot.Y, TurtleBot.V] = R / 2 * torch.sin(theta)
        g[:, TurtleBot.Y, TurtleBot.THETA_DOT] = R / 2 * torch.sin(theta)

        # Effect on theta
        g[:, TurtleBot.THETA, TurtleBot.V] = -R / (2 * L)
        g[:, TurtleBot.THETA, TurtleBot.THETA_DOT] = R / (2 * L)

        g = g.matmul(v)

        return g

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        

        self.P = torch.eye(3,3)
        self.K = torch.ones(self.n_controls, self.n_dims)

        K = self.K.type_as(x)
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)
        # Clamp given the control limits
        # import pdb; pdb.set_trace()
        upper_u_lim, lower_u_lim = self.control_limits
        u = torch.clamp(u, min=lower_u_lim[0].item(), max=upper_u_lim[0].item())
        # for dim_idx in range(self.n_controls):
        #     u[:, dim_idx] = torch.clamp(
        #         u[:, dim_idx],
        #         min=lower_u_lim[dim_idx].item(),
        #         max=upper_u_lim[dim_idx].item(),
        #     ) 

        return u
