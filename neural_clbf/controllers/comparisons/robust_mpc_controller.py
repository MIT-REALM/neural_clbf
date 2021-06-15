from abc import abstractmethod

import torch

import matlab  # type: ignore
import matlab.engine  # type: ignore

from neural_clbf.controllers import GenericController
from neural_clbf.systems import ControlAffineSystem
from neural_clbf.setup.robust_mpc import robust_mpc_path  # type: ignore


# Make sure the robust MPC directory was found
assert robust_mpc_path != "", "Unable to locate Robust MPC MATLAB directory!"


class RobustMPCController(GenericController):
    """Interface to the matlab robust MPC solver for a generic system"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(RobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

        # Make a connection to MATLAB to run the robust MPC
        self.eng = matlab.engine.connect_matlab()
        self.eng.cd(robust_mpc_path)

        # Linearize the system about the goal point and safe the transfer matrices
        self.A, self.B = self.dynamics_model.linearized_dt_dynamics_matrices()

    @abstractmethod
    def mpc_function(self, A, B, x_current_np):
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute and return the control input for a tensor of states x"""
        # Convert the given state into a numpy array
        x_current_np = x.cpu().detach().numpy().T
        # Prepare the dynamics matrices to be sent to MATLAB
        A = matlab.double(self.A.tolist())
        B = matlab.double(self.A.tolist())
        u_mpc = self.mpc_function(A, B, x_current_np)
        u_mpc = (A, B, x_current_np)

        # return the control input as a torch tensor.
        return torch.tensor(u_mpc).type_as(x)


class KSCarRobustMPCController(RobustMPCController):
    """Interface to the matlab robust MPC solver for a KSCar"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(KSCarRobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

    def mpc_function(self, A, B, x_current_np):
        return self.eng.mpc_kscar(A, B, x_current_np)


class STCarRobustMPCController(RobustMPCController):
    """Interface to the matlab robust MPC solver for an STCar"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(STCarRobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

    def mpc_function(self, A, B, x_current_np):
        return self.eng.mpc_stcar(A, B, x_current_np)


class SegwayRobustMPCController(RobustMPCController):
    """Interface to the matlab robust MPC solver for a segway"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(SegwayRobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

    def mpc_function(self, A, B, x_current_np):
        return self.eng.mpc_segway(A, B, x_current_np)


class Quad3DRobustMPCController(RobustMPCController):
    """Interface to the matlab robust MPC solver for a Quad3D"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(Quad3DRobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

    def mpc_function(self, A, B, x_current_np):
        return self.eng.mpc_quad3d(A, B, x_current_np)


class NeuralLanderRobustMPCController(RobustMPCController):
    """Interface to the matlab robust MPC solver for a NeuralLander"""

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        controller_period: float = 0.01,
    ):
        super(NeuralLanderRobustMPCController, self).__init__(
            dynamics_model=dynamics_model, controller_period=controller_period
        )

    def mpc_function(self, A, B, x_current_np):
        return self.eng.mpc_lander(A, B, x_current_np)
