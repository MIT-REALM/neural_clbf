"""Define a dynamical system for a neural lander"""
from typing import Tuple, List, Optional
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class FaNetwork(nn.Module):
    """Ground effect network"""

    def __init__(self):
        super(FaNetwork, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device("cpu"))
    model = FaNetwork().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


class NeuralLander(ControlAffineSystem):
    """
    Represents a neural lander (a 3D quadrotor with learned ground effect).
    """

    rho = 1.225
    gravity = 9.81
    drone_height = 0.09
    mass = 1.47

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 3

    # State indices
    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5

    # Control indices
    AX = 0
    AY = 1
    AZ = 2

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            No required keys
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params, dt, controller_dt)

        # Load the ground effect model
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.Fa_model = read_weight(
            dir_name + "/controller_data/Fa_net_12_3_full_Lip16.pth"
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    No required keys
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        return valid

    @property
    def n_dims(self) -> int:
        return NeuralLander.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return NeuralLander.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[NeuralLander.PX] = 5.0
        upper_limit[NeuralLander.PY] = 5.0
        upper_limit[NeuralLander.PZ] = 2.0
        upper_limit[NeuralLander.VX] = 1.0
        upper_limit[NeuralLander.VY] = 1.0
        upper_limit[NeuralLander.VZ] = 1.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([50, 50, 50])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        safe_z = -0.05
        safe_radius = 3
        safe_mask = torch.logical_and(
            x[:, NeuralLander.PZ] >= safe_z, x.norm(dim=-1) <= safe_radius
        )

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        unsafe_z = -0.2
        unsafe_radius = 3.5
        unsafe_mask = torch.logical_or(
            x[:, NeuralLander.PZ] <= unsafe_z, x.norm(dim=-1) >= unsafe_radius
        )

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

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

    def Fa_func(self, z, vx, vy, vz):
        if next(self.Fa_model.parameters()).device != z.device:
            self.Fa_model.to(z.device)

        bs = z.shape[0]

        # use prediction from NN as ground truth
        state = torch.zeros([bs, 1, 12]).type_as(z)
        state[:, 0, 0] = z + NeuralLander.drone_height
        state[:, 0, 1] = vx  # velocity
        state[:, 0, 2] = vy  # velocity
        state[:, 0, 3] = vz  # velocity
        state[:, 0, 7] = 1.0
        state[:, 0, 8:12] = 6508.0 / 8000
        state = state.float()

        Fa = self.Fa_model(state).squeeze(1) * torch.tensor([30.0, 15.0, 10.0]).reshape(
            1, 3
        ).type_as(z)
        return Fa.type(torch.FloatTensor)

    # We need to manually compute the state-state-derivative transfer matrix, since
    # we don't want to linearize the learned ground effect.
    def compute_A_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        A = np.zeros((self.n_dims, self.n_dims))
        A[: NeuralLander.PZ + 1, NeuralLander.VX :] = np.eye(3)

        return A

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

        # Derivatives of positions are just velocities
        f[:, NeuralLander.PX] = x[:, NeuralLander.VX]  # x
        f[:, NeuralLander.PY] = x[:, NeuralLander.VY]  # y
        f[:, NeuralLander.PZ] = x[:, NeuralLander.VZ]  # z

        # Constant acceleration in z due to gravity
        f[:, NeuralLander.VZ] = -grav

        # Add disturbance from ground effect
        _, _, z, vx, vy, vz = [x[:, i] for i in range(self.n_dims)]
        Fa = self.Fa_func(z, vx, vy, vz) / NeuralLander.mass
        f[:, NeuralLander.VX] += Fa[:, 0]
        f[:, NeuralLander.VY] += Fa[:, 1]
        f[:, NeuralLander.VZ] += Fa[:, 2]

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

        # Linear accelerations are control variables
        g[:, NeuralLander.VX :, :] = torch.eye(self.n_controls) / NeuralLander.mass

        return g

    @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls))
        u_eq[0, NeuralLander.AZ] = NeuralLander.mass * grav
        return u_eq
