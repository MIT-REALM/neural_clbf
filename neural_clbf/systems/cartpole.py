"""Define a dymamical system for a cartpole"""
from typing import Tuple, Optional, List

import math
import torch

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

class Cartpole(ControlAffineSystem):

    """
    Represents a Cartpole.

    The system has state

        x = [pos, vel, theta, theta_dot]

    (pos, vel) are the x-position and velocity of the cart
    (theta, theta_dot) are the angle, velocity of the pole

    and it has control inputs

        u = [u]

    representing the x-force applied to the cart.
    """
    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    CART_POS = 0
    CART_VEL = 1
    POLE_ANGLE = 2
    POLE_ANGVEL = 3

    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = True,
        scenarios: Optional[ScenarioList] = None,
    ):
        # TODO: Make all this configurable through 'params' dict
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.force_mag = 10.0

        super().__init__(nominal_params, dt, controller_dt, use_linearized_controller, scenarios)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys []
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # TODO: Implement system parameters
        return valid
    
    @property 
    def n_dims(self) -> int:
        return Cartpole.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [Cartpole.POLE_ANGLE]

    @property
    def n_controls(self) -> int:
        return Cartpole.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Cartpole.CART_POS] = 4.8
        upper_limit[Cartpole.CART_VEL] = 2.0
        upper_limit[Cartpole.POLE_ANGLE] = 24 * 2 * math.pi / 360
        upper_limit[Cartpole.POLE_ANGVEL] =  2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([100 * 10.0])
        lower_limit = -torch.tensor([100 * 10.0])

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        cart_pos = x[..., Cartpole.CART_POS]
        pole_angle = x[..., Cartpole.POLE_ANGLE]
        safe_mask = torch.logical_and(
            torch.abs(pole_angle) <= (12 * 2 * math.pi / 360),
            torch.abs(cart_pos) <= 2.4
        )
            
        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        safe_mask = self.safe_mask(x)
        return ~safe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = torch.ones_like(x.sum(dim=-1), dtype=torch.bool)
        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """ Calculate the control-independent dynamics of the cartpole """
        # Unpack state
        q = x[..., Cartpole.CART_POS] # Cart position
        qdot = x[..., Cartpole.CART_VEL] # Cart velocity
        w = x[..., Cartpole.POLE_ANGLE] # Pole angle
        wdot = x[..., Cartpole.POLE_ANGVEL] # Pole angular velocity

        dot_q = qdot
        dot_w = wdot
        # Correct
        dot_wdot = (
            (
                self.gravity * self.total_mass * torch.sin(w) 
                - self.masspole * self.length * (wdot ** 2 * torch.sin(w) * torch.cos(w))
            )
            /
            (
                (4/3) * self.length *self.total_mass 
                - self.masspole * self.length * torch.cos(w) ** 2
            )
        )  
        Ax = dot_wdot
        # Correct
        dot_qdot = (
            (self.masspole * self.length * (wdot ** 2 * torch.sin(w) - torch.cos(w) * Ax))
            / (self.total_mass)
        )

        fx = torch.stack([dot_q, dot_qdot, dot_w, dot_wdot], dim=-1)
        return fx.unsqueeze(-1)

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """ Calculate the control-dependent dynamics of the cartpole """
        # Unpack state
        q = x[...,  Cartpole.CART_POS] # Cart position
        qdot = x[..., Cartpole.CART_VEL] # Cart velocity
        w = x[..., Cartpole.POLE_ANGLE] # Pole angle
        wdot = x[..., Cartpole.POLE_ANGVEL] # Pole angular velocity

        dot_q = torch.zeros_like(q)
        dot_w = torch.zeros_like(q)
        # Correct
        dot_wdot = (
            -torch.cos(w)
            / 
            (
                (4/3) * self.length *self.total_mass 
                - self.masspole * self.length * torch.cos(w) ** 2
            )
        )  
        Bx = dot_wdot
        # Correct
        dot_qdot = (
            (1 - self.masspole * self.length * torch.cos(w) * Bx)
            / self.total_mass 
        )
        return torch.stack([dot_q, dot_qdot, dot_w, dot_wdot], dim=-1).unsqueeze(-1)
    
    def reference_dynamics(self, x: torch.Tensor, u:torch.Tensor) -> torch.Tensor:
        """ Calculate the reference dynamics of the cartpole 
        
        Reference: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py """

        # Unpack state
        q = x[..., 0] # Cart position
        q_dot = x[..., 1] # Cart velocity
        theta = x[..., 2] # Pole angle
        theta_dot = x[..., 3] # Pole angular velocity
        force = u[..., 0]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force + self.masspole * self.length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / self.total_mass

        return torch.stack([q_dot, xacc, theta_dot, thetaacc], dim=-1)
    
if __name__ == "__main__":
    # Sanity check the dynamics
    nominal_params={}
    x = torch.randn([32, 4])
    u = torch.randn([32, 1])

    cartpole = Cartpole(nominal_params)
    xdot_control_affine = cartpole.closed_loop_dynamics(x, u, nominal_params)
    xdot_reference = cartpole.reference_dynamics(x, u)
    assert torch.allclose(xdot_control_affine, xdot_reference)