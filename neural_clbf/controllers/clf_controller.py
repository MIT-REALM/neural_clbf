from typing import Tuple, Optional, Union

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn.functional as F

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.controller import Controller
from neural_clbf.experiments import ExperimentSuite


class CLFController(Controller):
    """
    A generic CLF-based controller, using the quadratic Lyapunov function found for
    the linearized system.

    This controller and all subclasses assumes continuous-time dynamics.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clf_lambda: convergence rate for the CLF
            clf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
        """
        super(CLFController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.clf_lambda = clf_lambda
        self.safe_level: Union[torch.Tensor, float]
        self.unsafe_level: Union[torch.Tensor, float]
        self.clf_relaxation_penalty = clf_relaxation_penalty

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamics_model.n_controls)
        clf_relaxations = []
        for scenario in self.scenarios:
            clf_relaxations.append(cp.Variable(1, nonneg=True))

        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input
        V_param = cp.Parameter(1, nonneg=True)
        Lf_V_params = []
        Lg_V_params = []
        for scenario in self.scenarios:
            Lf_V_params.append(cp.Parameter(1))
            Lg_V_params.append(cp.Parameter(self.dynamics_model.n_controls))

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self.dynamics_model.n_controls)

        # These allow us to define the constraints
        constraints = []
        for i in range(len(self.scenarios)):
            # CLF decrease constraint (with relaxation)
            constraints.append(
                Lf_V_params[i]
                + Lg_V_params[i] @ u
                + self.clf_lambda * V_param
                - clf_relaxations[i]
                <= 0
            )

        # Control limit constraints
        upper_lim, lower_lim = self.dynamics_model.control_limits
        for control_idx in range(self.dynamics_model.n_controls):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])

        # And define the objective
        objective_expression = cp.sum_squares(u - u_ref_param)
        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations
        parameters = Lf_V_params + Lg_V_params
        parameters += [V_param, u_ref_param, clf_relaxation_penalty_param]
        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CLF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # First, get the Lyapunov function value and gradient at this state
        P = self.dynamics_model.P.type_as(x)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        V = 0.5 * F.bilinear(x, x, P).squeeze()
        V = V.reshape(x.shape[0])

        # Reshape again for the gradient calculation
        P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        JV = F.linear(x, P)
        JV = JV.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CLF"""
        V, _ = self.V_with_jacobian(x)
        return V

    def V_lie_derivatives(
        self, x: torch.Tensor, scenarios: Optional[ScenarioList] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            scenarios: optional list of scenarios. Defaults to self.scenarios
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        if scenarios is None:
            scenarios = self.scenarios
        n_scenarios = len(scenarios)

        # Get the Jacobian of V for each entry in the batch
        _, gradV = self.V_with_jacobian(x)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def u_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Determine the reference control input."""
        # Here we use the nominal controller as the reference, but subclasses can
        # override this
        return self.dynamics_model.u_nominal(x)

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        Gurobi, which does not allow for backpropagation.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # To find the control input, we want to solve a QP constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # We want the objective to be to minimize
        #
        #           ||u - u_ref||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u - 2 u_ref^T u + relaxation_penalty * r^2

        n_controls = self.dynamics_model.n_controls
        n_scenarios = self.n_scenarios
        allow_relaxation = not (relaxation_penalty == float("inf"))

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, n_scenarios)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            model = gp.Model("clf_qp")
            # Create variables for control input and (optionally) the relaxations
            upper_lim, lower_lim = self.dynamics_model.control_limits
            upper_lim = upper_lim.cpu().numpy()
            lower_lim = lower_lim.cpu().numpy()
            u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
            if allow_relaxation:
                r = model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
            if allow_relaxation:
                relax_penalties = relaxation_penalty * np.ones(n_scenarios)
                objective += relax_penalties @ r

            # Now build the CLF constraints
            for i in range(n_scenarios):
                Lg_V_np = Lg_V[batch_idx, i, :].detach().cpu().numpy()
                Lf_V_np = Lf_V[batch_idx, i, :].detach().cpu().numpy()
                V_np = V[batch_idx].detach().cpu().numpy()
                clf_constraint = Lf_V_np + Lg_V_np @ u + self.clf_lambda * V_np
                if allow_relaxation:
                    clf_constraint -= r[i]
                model.addConstr(clf_constraint <= 0.0, name=f"Scenario {i} Decrease")

            # Optimize!
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                if allow_relaxation:
                    for i in range(n_scenarios):
                        r_result[batch_idx, i] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            if allow_relaxation:
                for i in range(n_scenarios):
                    r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)

    def _solve_CLF_QP_cvxpylayers(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        CVXPyLayers, which does allow for backpropagation, but is slower and less
        accurate than Gurobi.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # The differentiable solver must allow relaxation
        relaxation_penalty = min(relaxation_penalty, 1e6)

        # We've already created a parameterized QP solver, so we can use that
        result = self.differentiable_qp_solver(
            *Lf_V,
            *Lg_V,
            V,
            u_ref,
            torch.tensor([relaxation_penalty]),
            solver_args={"max_iters": 50000000},
        )

        # Extract the results
        u_result = result[0]
        r_result = torch.hstack(result[1:])

        return u_result.type_as(x), r_result.type_as(x)

    def solve_CLF_QP(
        self,
        x,
        relaxation_penalty: Optional[float] = None,
        u_ref: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation, defaults to
                                self.clf_relaxation_penalty
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. If provided, must have
                   dimensions bs x self.dynamics_model.n_controls. If not provided,
                   default to calling self.u_reference.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # Get the value of the CLF and its Lie derivatives
        V = self.V(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {self.dynamics_model.n_controls} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == self.dynamics_model.n_controls, err_message
        else:
            u_ref = self.u_reference(x)

        # Apply default penalty if needed
        if relaxation_penalty is None:
            relaxation_penalty = self.clf_relaxation_penalty

        # Figure out if we need to use a differentiable solver (determined by whether
        # the input x requires a gradient or not)
        if x.requires_grad:
            return self._solve_CLF_QP_cvxpylayers(
                x, u_ref, V, Lf_V, Lg_V, relaxation_penalty
            )
        else:
            return self._solve_CLF_QP_gurobi(
                x, u_ref, V, Lf_V, Lg_V, relaxation_penalty
            )

    def u(self, x):
        """Get the control input for a given state"""
        u, _ = self.solve_CLF_QP(x)
        return u
