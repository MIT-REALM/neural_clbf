from typing import Tuple, Callable

import torch
from torch.autograd.functional import jacobian

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


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
class InvertedPendulumPolynomialCLBFController(PolynomialCLBFController):
    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        nominal_scenario: Scenario,
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        # Hard-code the polynomial found via sum-of-squares
        def polynomial_clbf(x):
            # Barf: degree-10 SOS CLBF
            V = (
                (1.6731) * x[:, 0] ** 2
                + (-0.32563) * x[:, 0] ** 3
                + (1.0295) * x[:, 0] ** 4
                + (-0.93618) * x[:, 0] ** 5
                + (5.3214) * x[:, 0] ** 6
                + (0.052531) * x[:, 0] ** 7
                + (-0.33326) * x[:, 0] ** 8
                + (-0.00080636) * x[:, 0] ** 9
                + (0.0058499) * x[:, 0] ** 10
                + (0.55492) * x[:, 2] ** 2
                + (-0.067116) * x[:, 2] ** 3
                + (0.91323) * x[:, 2] ** 4
                + (-0.0090608) * x[:, 2] ** 5
                + (0.066627) * x[:, 2] ** 6
                + (0.00039932) * x[:, 2] ** 7
                + (0.00049954) * x[:, 2] ** 8
                + (0.56745) * x[:, 2] * x[:, 0]
                + (-0.32485) * x[:, 2] * x[:, 0] ** 2
                + (0.46896) * x[:, 2] * x[:, 0] ** 3
                + (-0.33875) * x[:, 2] * x[:, 0] ** 4
                + (0.95225) * x[:, 2] * x[:, 0] ** 5
                + (0.010862) * x[:, 2] * x[:, 0] ** 6
                + (0.0012662) * x[:, 2] * x[:, 0] ** 7
                + (0.00020097) * x[:, 2] * x[:, 0] ** 8
                + (-0.0020058) * x[:, 2] * x[:, 0] ** 9
                + (-0.36771) * x[:, 2] ** 2 * x[:, 0]
                + (0.7664) * x[:, 2] ** 2 * x[:, 0] ** 2
                + (-0.097314) * x[:, 2] ** 2 * x[:, 0] ** 3
                + (0.49295) * x[:, 2] ** 2 * x[:, 0] ** 4
                + (0.018818) * x[:, 2] ** 2 * x[:, 0] ** 5
                + (-0.063826) * x[:, 2] ** 2 * x[:, 0] ** 6
                + (-0.00079333) * x[:, 2] ** 2 * x[:, 0] ** 7
                + (0.0027058) * x[:, 2] ** 2 * x[:, 0] ** 8
                + (0.02804) * x[:, 2] ** 3 * x[:, 0]
                + (0.1082) * x[:, 2] ** 3 * x[:, 0] ** 2
                + (-0.62643) * x[:, 2] ** 3 * x[:, 0] ** 3
                + (-0.0098732) * x[:, 2] ** 3 * x[:, 0] ** 4
                + (0.050649) * x[:, 2] ** 3 * x[:, 0] ** 5
                + (0.00046285) * x[:, 2] ** 3 * x[:, 0] ** 6
                + (-0.0018219) * x[:, 2] ** 3 * x[:, 0] ** 7
                + (0.0022811) * x[:, 2] ** 4 * x[:, 0]
                + (-0.080549) * x[:, 2] ** 4 * x[:, 0] ** 2
                + (0.004127) * x[:, 2] ** 4 * x[:, 0] ** 3
                + (-0.010395) * x[:, 2] ** 4 * x[:, 0] ** 4
                + (-0.00027124) * x[:, 2] ** 4 * x[:, 0] ** 5
                + (0.00080812) * x[:, 2] ** 4 * x[:, 0] ** 6
                + (0.16189) * x[:, 2] ** 5 * x[:, 0]
                + (-0.0027617) * x[:, 2] ** 5 * x[:, 0] ** 2
                + (0.0077122) * x[:, 2] ** 5 * x[:, 0] ** 3
                + (0.00018066) * x[:, 2] ** 5 * x[:, 0] ** 4
                + (-0.0007408) * x[:, 2] ** 5 * x[:, 0] ** 5
                + (-0.00043627) * x[:, 2] ** 6 * x[:, 0]
                + (-0.0078574) * x[:, 2] ** 6 * x[:, 0] ** 2
                + (-2.8064e-05) * x[:, 2] ** 6 * x[:, 0] ** 3
                + (0.00048107) * x[:, 2] ** 6 * x[:, 0] ** 4
                + (0.00011409) * x[:, 2] ** 7 * x[:, 0]
                + (-6.5418e-06) * x[:, 2] ** 7 * x[:, 0] ** 2
                + (-0.00013131) * x[:, 2] ** 7 * x[:, 0] ** 3
                + (1.4265e-05) * x[:, 2] ** 8 * x[:, 0] ** 2
            )

            return V

        super().__init__(
            dynamics_model,
            polynomial_clbf,
            nominal_scenario,
            clbf_relaxation_penalty=clbf_relaxation_penalty,
            controller_period=controller_period,
        )
