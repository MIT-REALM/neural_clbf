from typing import Tuple, Callable

import torch
from torch.autograd.functional import jacobian

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule

from .sos_polynomials.matlab_export_kscar_d10_bf import kscar_d10_polynomial_clbf
from .sos_polynomials.matlab_export_stcar_d7_bf import stcar_d7_polynomial_clbf


class PolynomialCLBFController(NeuralCLBFController):
    """Kind of a hack-y solution: implement the Polynomial CLBF controller
    to look like the neural version, but only use one scenario and replace V with
    a polynomial
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        polynomial_clbf: Callable[[torch.Tensor], torch.Tensor],
        nominal_scenario: Scenario,
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        # Make a dummy datamodule and scenario
        scenarios = [nominal_scenario]
        initial_conditions = []
        for dim in range(dynamics_model.n_dims):
            initial_conditions.append((-1.0, 1.0))
        datamodule = EpisodicDataModule(dynamics_model, initial_conditions)

        super().__init__(
            dynamics_model,
            scenarios,
            datamodule,
            clbf_relaxation_penalty=clbf_relaxation_penalty,
            controller_period=controller_period,
        )

        self.polynomial_clbf = polynomial_clbf

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Uses the nominal controller.

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               controller
        """
        return self.dynamics_model.u_nominal(x)

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Get the value of the CLBF by evaluating the polynomial model
        V = self.polynomial_clbf(x)
        # Get the Jacobian using autograd
        JV = jacobian(self.polynomial_clbf, x)
        JV = JV.reshape(-1, 1, self.dynamics_model.n_dims)

        return V, JV


# Also define some specific controllers
class KSCarPolynomialCLBFController(PolynomialCLBFController):
    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        nominal_scenario: Scenario,
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        super().__init__(
            dynamics_model,
            kscar_d10_polynomial_clbf,
            nominal_scenario,
            clbf_relaxation_penalty=clbf_relaxation_penalty,
            controller_period=controller_period,
        )


class STCarPolynomialCLBFController(PolynomialCLBFController):
    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        nominal_scenario: Scenario,
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        super().__init__(
            dynamics_model,
            stcar_d7_polynomial_clbf,
            nominal_scenario,
            clbf_relaxation_penalty=clbf_relaxation_penalty,
            controller_period=controller_period,
        )
