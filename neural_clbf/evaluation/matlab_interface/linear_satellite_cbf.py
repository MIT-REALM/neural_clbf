from collections import OrderedDict

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn as nn

from neural_clbf.systems import LinearSatellite


def normalize(dynamics_model, x, k=1.0):
    """Normalize the state input to [-k, k]

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    x_max, x_min = dynamics_model.state_limits
    x_center = (x_max + x_min) / 2.0
    x_range = (x_max - x_min) / 2.0
    # Scale to get the input between (-k, k), centered at 0
    x_range = x_range / k
    # We shouldn't scale or offset any angle dimensions
    x_center[dynamics_model.angle_dims] = 0.0
    x_range[dynamics_model.angle_dims] = 1.0

    # Do the normalization
    return (x - x_center.type_as(x)) / x_range.type_as(x)


def normalize_with_angles(dynamics_model, x, k=1.0):
    """Normalize the input using the stored center point and range, and replace all
    angles with the sine and cosine of the angles

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Scale and offset based on the center and range
    x = normalize(dynamics_model, x, k)

    # Replace all angles with their sine, and append cosine
    angle_dims = dynamics_model.angle_dims
    angles = x[:, angle_dims]
    x[:, angle_dims] = torch.sin(angles)
    x = torch.cat((x, torch.cos(angles)), dim=-1)

    return x


# Define a pared-down class for just this example
class SatelliteCBF(nn.Module):
    def __init__(self):
        super(SatelliteCBF, self).__init__()

        self.cbf_lambda = 1.0

        # Define the dynamics model
        simulation_dt = 0.001
        controller_period = 0.01
        # Define the scenarios
        nominal_params = {
            "a": 6871,
            "ux_target": 0.0,
            "uy_target": 0.0,
            "uz_target": 0.0,
        }
        scenarios = [
            nominal_params,
        ]
        for ux in [-0.01, 0.01]:
            for uy in [-0.01, 0.01]:
                for uz in [-0.01, 0.01]:
                    scenarios.append(
                        {
                            "a": 6871,
                            "ux_target": ux,
                            "uy_target": uy,
                            "uz_target": uz,
                        }
                    )

        # Define the dynamics model
        self.dynamics_model = LinearSatellite(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
        )
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Compute and save the center and range of the state variables
        x_max, x_min = self.dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = 10.0
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Define the CLBF network, which we denote V
        self.cbf_hidden_layers = 4
        self.cbf_hidden_size = 128
        self.n_dims_extended = 6
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.cbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.cbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.cbf_hidden_size, self.cbf_hidden_size
            )
            if i < self.cbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.cbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

    def V_with_jacobian(self, x):
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x, self.k)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        # Now step through each layer in V
        V = x_norm
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        return V, JV

    def V(self, x):
        """Compute the value of the CLF"""
        V, _ = self.V_with_jacobian(x)
        return V

    def V_lie_derivatives(self, x, scenarios=None):
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

    def _solve_CLF_QP_gurobi(
        self,
        x,
        u_ref,
        V,
        Lf_V,
        Lg_V,
        relaxation_penalty,
    ):
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
            r = model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
            relax_penalties = relaxation_penalty * np.ones(n_scenarios)
            objective += relax_penalties @ r

            # Now build the CLF constraints
            for i in range(n_scenarios):
                Lg_V_np = Lg_V[batch_idx, i, :].detach().cpu().numpy()
                Lf_V_np = Lf_V[batch_idx, i, :].detach().cpu().numpy()
                V_np = V[batch_idx].detach().cpu().numpy()
                clf_constraint = Lf_V_np + Lg_V_np @ u + self.cbf_lambda * V_np - r[i]
                model.addConstr(clf_constraint <= 0.0, name=f"Scenario {i} Decrease")
                # model.addConstr(r[i] >= 1.0)

            # Optimize!
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                for i in range(n_scenarios):
                    r_result[batch_idx, i] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            for i in range(n_scenarios):
                r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)

    @torch.no_grad()
    def solve_CLF_QP(
        self,
        x,
        u_ref,
        relaxation_penalty,
    ):
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

        # Check the reference control input as well
        err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
        assert u_ref.shape[0] == x.shape[0], err_message
        err_message = f"u_ref must have {self.dynamics_model.n_controls} cols,"
        err_message += f" but got {u_ref.shape[1]}"
        assert u_ref.shape[1] == self.dynamics_model.n_controls, err_message

        # Check the penalty
        assert relaxation_penalty >= 0

        # Solve
        return self._solve_CLF_QP_gurobi(x, u_ref, V, Lf_V, Lg_V, relaxation_penalty)

    def forward(self, x):
        u_ref = torch.zeros(x.shape[0], self.dynamics_model.n_controls).type_as(x)
        u_qp, _ = self.solve_CLF_QP(x, u_ref, 1e3)
        return u_qp


sat_cbf = SatelliteCBF()
sat_cbf.load_state_dict(
    torch.load("neural_clbf/evaluation/matlab_interface/commit_318eb38_v0.nn")
)


# Define a function for taking in a state and reference control and solving the CBF QP
def cbf_qp_filter(x, u_ref, relaxation_penalty):
    """Use the CBF QP to filter a provided reference control signal

    args:
        x: an N x 6 numpy array of states (x, y, z, vx, vy, vz)
        u_ref: an N x 3 numpy array of controls (fx, fy, fz)
        relaxation_penalty: the penalty to use for CBF relaxation
    returns:
        an N x 3 numpy array of filtered controls
    """
    # Convert inputs to torch tensors
    n_dims = sat_cbf.dynamics_model.n_dims
    n_controls = sat_cbf.dynamics_model.n_controls
    x_tensor = torch.tensor(x).type_as(sat_cbf.V_nn[0].weight).reshape(-1, n_dims)
    u_ref_tensor = torch.tensor(u_ref).type_as(sat_cbf.V_nn[0].weight)
    u_ref_tensor = u_ref_tensor.reshape(-1, n_controls)

    # Check input size
    N = x_tensor.shape[0]
    assert x_tensor.shape[1] == n_dims, "x must have 6 columns"
    assert u_ref_tensor.shape[0] == N, "u_ref must have the same #rows as x"
    assert u_ref_tensor.shape[1] == n_controls, "u_ref must have 3 columns"

    # Solve the CBF QP (this function refers to a CLF because CBFs subclass CLFs in
    # our implementation). We don't need the second return tensor.
    u_filtered_tensor, r = sat_cbf.solve_CLF_QP(
        x_tensor, u_ref_tensor, relaxation_penalty
    )

    # Return the filtered tensor converted to a numpy array
    u_filtered_np = u_filtered_tensor.numpy()
    r = r.numpy()
    return u_filtered_np, r
