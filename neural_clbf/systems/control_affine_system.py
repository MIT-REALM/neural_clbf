"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Callable, Tuple, Optional, List

from matplotlib.axes import Axes
import numpy as np
import torch
from torch.autograd.functional import jacobian

from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
    lqr,
    robust_continuous_lyap,
    continuous_lyap,
)


class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamical system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + g(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = True,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize a system.

        args:
            nominal_params: a dictionary giving the parameter values for the system
            dt: the timestep to use for simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_linearized_controller: if True, linearize the system model to derive a
                                       LQR controller. If false, the system is must
                                       set self.P itself to be a tensor n_dims x n_dims
                                       positive definite matrix.
            scenarios: an optional list of scenarios for robust control
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

        if controller_dt is None:
            controller_dt = self.dt
        self.controller_dt = controller_dt

        # Compute the linearized controller
        if use_linearized_controller:
            self.compute_linearized_controller(scenarios)

    @torch.enable_grad()
    def compute_A_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        # Linearize the system about the x = 0, u = 0
        x0 = self.goal_point
        u0 = self.u_eq
        dynamics = lambda x: self.closed_loop_dynamics(x, u0, scenario).squeeze()
        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (self.n_dims, self.n_dims))

        return A

    def compute_B_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        if scenario is None:
            scenario = self.nominal_params

        # Linearize the system about the x = 0, u = 0
        B = self._g(self.goal_point, scenario).squeeze().cpu().numpy()
        B = np.reshape(B, (self.n_dims, self.n_controls))

        return B

    def linearized_ct_dynamics_matrices(
        self, scenario: Optional[Scenario] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the continuous time linear dynamics matrices, dx/dt = Ax + Bu"""
        A = self.compute_A_matrix(scenario)
        B = self.compute_B_matrix(scenario)

        return A, B

    def linearized_dt_dynamics_matrices(
        self, scenario: Optional[Scenario] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous time linear dynamics matrices, x_{t+1} = Ax_{t} + Bu
        """
        Act, Bct = self.linearized_ct_dynamics_matrices(scenario)
        A = np.eye(self.n_dims) + self.controller_dt * Act
        B = self.controller_dt * Bct

        return A, B

    def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # We need to compute the LQR closed-loop linear dynamics for each scenario
        Acl_list = []
        # Default to the nominal scenario if none are provided
        if scenarios is None:
            scenarios = [self.nominal_params]

        # For each scenario, get the LQR gain and closed-loop linearization
        for s in scenarios:
            # Compute the LQR gain matrix for the nominal parameters
            Act, Bct = self.linearized_ct_dynamics_matrices(s)
            A, B = self.linearized_dt_dynamics_matrices(s)

            # Define cost matrices as identity
            Q = np.eye(self.n_dims)
            R = np.eye(self.n_controls)

            # Get feedback matrix
            K_np = lqr(A, B, Q, R)
            self.K = torch.tensor(K_np)

            Acl_list.append(Act - Bct @ K_np)

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities
        if len(scenarios) > 1:
            self.P = torch.tensor(robust_continuous_lyap(Acl_list, Q))
        else:
            # Otherwise, just use the standard Lyapunov equation
            self.P = torch.tensor(continuous_lyap(Acl_list[0], Q))

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

    def boundary_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating regions that are neither safe nor unsafe

        args:
            x: a tensor of points in the state space
        """
        return torch.logical_not(
            torch.logical_or(
                self.safe_mask(x),
                self.unsafe_mask(x),
            )
        )

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of points in the state space
        """
        # Include a sensible default
        goal_tolerance = 0.1
        return (x - self.goal_point).norm(dim=-1) <= goal_tolerance

    @property
    def goal_point(self):
        return torch.zeros((1, self.n_dims))

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls))

    def sample_state_space(self, num_samples: int) -> torch.Tensor:
        """Sample uniformly from the state space"""
        x_max, x_min = self.state_limits

        # Sample uniformly from 0 to 1 and then shift and scale to match state limits
        x = torch.Tensor(num_samples, self.n_dims).uniform_(0.0, 1.0)
        for i in range(self.n_dims):
            x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

        return x

    def sample_with_mask(
        self,
        num_samples: int,
        mask_fn: Callable[[torch.Tensor], torch.Tensor],
        max_tries: int = 5000,
    ) -> torch.Tensor:
        """Sample num_samples so that mask_fn is True for all samples. Makes a
        best-effort attempt, but gives up after max_tries, so may return some points
        for which the mask is False, so watch out!
        """
        # Get a uniform sampling
        samples = self.sample_state_space(num_samples)

        # While the mask is violated, get violators and replace them
        # (give up after so many tries)
        for _ in range(max_tries):
            violations = torch.logical_not(mask_fn(samples))
            if not violations.any():
                break

            new_samples = int(violations.sum().item())
            samples[violations] = self.sample_state_space(new_samples)

        return samples

    def sample_safe(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the safe space. May return some points that are not
        safe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.safe_mask, max_tries)

    def sample_unsafe(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the unsafe space. May return some points that are not
        unsafe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.unsafe_mask, max_tries)

    def sample_goal(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the goal. May return some points that are not in the
        goal, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.goal_mask, max_tries)

    def sample_boundary(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the state space between the safe and unsafe regions.
        May return some points that are not in this region safe, so watch out (only a
        best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.boundary_mask, max_tries)

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

    def zero_order_hold(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        controller_dt: float,
        params: Optional[Scenario] = None,
    ) -> torch.Tensor:
        """
        Simulate dynamics forward for controller_dt, simulating at self.dt, with control
        held constant at u, starting from x

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            controller_dt: the amount of time to hold for
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            x_next: bs x self.n_dims tensor of next states
        """
        num_steps = int(controller_dt / self.dt)
        for tstep in range(1, num_steps):
            # Get the derivatives for this control input
            xdot = self.closed_loop_dynamics(x, u, params)

            # Simulate forward
            x = x + self.dt * xdot

        # Return the simulated state
        return x

    def simulate(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        controller: Callable[[torch.Tensor], torch.Tensor],
        controller_period: Optional[float] = None,
        guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        params: Optional[Scenario] = None,
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
            params - a dictionary giving the parameter values for the system. If None,
                     default to the nominal parameters used at initialization
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
                xdot = self.closed_loop_dynamics(x_current, u, params)
                x_sim[:, tstep, :] = x_current + self.dt * xdot

                # If the guard is activated for any trajectory, reset that trajectory
                # to a random state
                if guard is not None:
                    guard_activations = guard(x_sim[:, tstep, :])
                    n_to_resample = int(guard_activations.sum().item())
                    x_new = self.sample_state_space(n_to_resample).type_as(x_sim)
                    x_sim[guard_activations, tstep, :] = x_new

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

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    def plot_environment(self, ax: Axes) -> None:
        """
        Add a plot of the environment to the given figure. Defaults to do nothing
        unless overidden.

        args:
            ax: the axis on which to plot
        """
        pass
