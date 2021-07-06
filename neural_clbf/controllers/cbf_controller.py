from typing import Tuple

import torch

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.experiments import ExperimentSuite


class CBFController(CLFController):
    """
    A generic CBF-based controller, using the quadratic Lyapunov function found for
    the linearized system to construct a simple barrier function.

    For our purposes, a barrier function h(x): X -> R segments h(safe) <= 0 and
    h(unsafe) >= 0, and dh/dt <= -lambda h(x).

    This definition allows us to re-use a CLF controller. Internally, we'll rename h = V
    with some notational abuse, but let V be negative sometimes.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        cbf_lambda: float = 1.0,
        cbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            cbf_lambda: scaling factor for the CBF
            cbf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
        """
        super(CBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=cbf_lambda,
            clf_relaxation_penalty=cbf_relaxation_penalty,
            controller_period=controller_period,
        )

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CBF value and its Jacobian. Remember that we're abusing notation
        and calling our barrier function V

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # To make a simple barrier function, use the Lyapunov function shifted down
        V, JV = super().V_with_jacobian(x)
        V -= 1.0

        return V, JV
