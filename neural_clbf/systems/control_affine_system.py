"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Callable, Tuple, Optional, List

import torch

from neural_clbf.systems.utils import Scenario


class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamcial system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + g(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(self, nominal_params: Scenario, dt: float = 0.01):
        """
        Initialize a system.

        args:
            nominal_params: a dictionary giving the parameter values for the system
            dt: the timestep to use for simulation
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__()

        # Validate parameters, raise error if they're not valid
        if not self.validate_params(nominal_params):
            raise ValueError(f"Parameters not valid: {nominal_params}")

        self.nominal_params = nominal_params

        # Make sure the timestep is valid
        assert dt > 0.0
        self.dt = dt

    @abstractmethod
    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        pass

    @abstractproperty
    def n_dims(self) -> int:
        pass

    @abstractproperty
    def angle_dims(self) -> List[int]:
        pass

    @abstractproperty
    def n_controls(self) -> int:
        pass

    @abstractproperty
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        pass

    @abstractproperty
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        pass

    def out_of_bounds_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating whether rows are outside the state limits
        for this system

        args:
            x: a tensor of points in the state space
        """
        upper_lim, lower_lim = self.state_limits
        out_of_bounds_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
        for i_dim in range(x.shape[-1]):
            out_of_bounds_mask.logical_or_(x[:, i_dim] >= upper_lim[i_dim])
            out_of_bounds_mask.logical_or_(x[:, i_dim] <= lower_lim[i_dim])

        return out_of_bounds_mask

    @abstractmethod
    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of points in the state space
        """
        pass

    @abstractmethod
    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of points in the state space
        """
        pass

    @abstractmethod
    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of points in the state space
        """
        pass

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
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        # If no params required, use nominal params
        if params is None:
            params = self.nominal_params

        return self._f(x, params), self._g(x, params)

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x, params=params)
        # Compute state derivatives using control-affine form
        xdot = f + torch.bmm(g, u.unsqueeze(-1))
        return xdot.view(x.shape)

    @torch.no_grad()
    def simulate(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        controller: Callable[[torch.Tensor], torch.Tensor],
        controller_period: Optional[float] = None,
        guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Simulate the system for the specified number of steps using the given controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            num_steps - a positive integer
            controller - a mapping from state to control action
            controller_period - the period determining how often the controller is run
                                (in seconds). If none, defaults to self.dt
            guard - a function that takes a bs x n_dims tensor and returns a length bs
                    mask that's True for any trajectories that should be reset to x_init
        returns
            a bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
            occurs on any trajectory, the simulation of all trajectories will stop and
            the second dimension will be less than num_steps
        """
        # Create a tensor to hold the simulation results
        x_sim = torch.zeros(x_init.shape[0], num_steps, self.n_dims).type_as(x_init)
        x_sim[:, 0, :] = x_init
        u = torch.zeros(x_init.shape[0], self.n_controls).type_as(x_init)

        # Compute controller update frequency
        if controller_period is None:
            controller_period = self.dt
        controller_update_freq = int(controller_period / self.dt)

        # Run the simulation until it's over or an error occurs
        t_sim_final = 0
        for tstep in range(1, num_steps):
            try:
                # Get the current state
                x_current = x_sim[:, tstep - 1, :]
                # Get the control input at the current state if it's time
                if tstep == 1 or tstep % controller_update_freq == 0:
                    u = controller(x_current)

                # Simulate forward using the dynamics
                xdot = self.closed_loop_dynamics(x_current, u)
                x_sim[:, tstep, :] = x_current + self.dt * xdot

                # If the guard is activated for any trajectory, reset that trajectory
                if guard is not None:
                    guard_activations = guard(x_sim[:, tstep, :])
                    x_sim[guard_activations, tstep, :] = x_init[guard_activations, :]

                # Update the final simulation time if the step was successful
                t_sim_final = tstep
            except ValueError:
                break

        return x_sim[:, : t_sim_final + 1, :]

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
        return self.simulate(
            x_init, num_steps, self.u_nominal, guard=self.out_of_bounds_mask
        )

    @abstractmethod
    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        pass

    @abstractmethod
    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal (e.g. LQR or proportional) control for the nominal
        parameters

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        pass
