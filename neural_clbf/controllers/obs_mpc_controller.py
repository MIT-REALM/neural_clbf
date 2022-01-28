from typing import Tuple, Optional

import cvxpy as cp
import torch
import numpy as np

from neural_clbf.systems import ObservableSystem, PlanarLidarSystem  # noqa
from neural_clbf.controllers.controller import Controller
from neural_clbf.experiments import ExperimentSuite


class ObsMPCController(Controller):
    """
    A comparison controller that implements MPC for perception-based control
    """

    def __init__(
        self,
        dynamics_model: ObservableSystem,
        controller_period: float,
        experiment_suite: ExperimentSuite,
        validation_dynamics_model: Optional[ObservableSystem] = None,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            controller_period: the controller update period
            experiment_suite: defines the experiments to run during training
            validation_dynamics_model: optionally provide a dynamics model to use during
                                       validation
        """
        super(ObsMPCController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )

        # Define this again so that Mypy is happy
        self.dynamics_model = dynamics_model
        # And save the validation model
        self.training_dynamics_model = dynamics_model
        self.validation_dynamics_model = validation_dynamics_model

        # Save the experiments suits
        self.experiment_suite = experiment_suite

    def get_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper around the dynamics model to get the observations"""
        assert isinstance(self.dynamics_model, ObservableSystem)
        return self.dynamics_model.get_observations(x)

    def approximate_lookahead(
        self, x: torch.Tensor, o: torch.Tensor, u: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper around the dynamics model to do approximate lookeahead"""
        assert isinstance(self.dynamics_model, ObservableSystem)
        return self.dynamics_model.approximate_lookahead(x, o, u, dt)

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the control input at a given state. Computes the observations and
        barrier function value at this state before computing the control.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        """
        # Get the observations
        obs = self.get_observations(x)

        # Solve the MPC problem for each element of the batch
        batch_size = x.shape[0]
        u = torch.zeros(batch_size, self.dynamics_model.n_controls).type_as(x)
        for batch_idx in range(batch_size):
            batch_obs = obs[batch_idx, :, :]
            batch_x = x[batch_idx, :2].cpu().detach().numpy()

            # Compute an ellipsoid under-approximating the free space.
            # Parameterize the ellipsoid as the 1-sublevel set of x^T P x
            P = cp.Variable((2, 2), symmetric=True)

            # The operator >> denotes matrix inequality. We want P to be PSD
            constraints = [P >> 0]

            # For each detected point, we want x^T P x >= 1.1
            for point_idx in range(batch_obs.shape[-1]):
                o_i = batch_obs[:, point_idx].reshape(-1, 1).cpu().detach().numpy()
                constraints.append(cp.quad_form(o_i, P) >= 1.25)

            # Solve for the P with largest volume
            prob = cp.Problem(
                cp.Maximize(cp.log_det(P) - 100 * cp.trace(P)), constraints
            )
            prob.solve()
            # Skip if no solution
            if prob.status != "optimal":
                continue

            # Otherwise, continue on
            P_opt = P.value

            # Next, solve for the point inside that ellipsoid closest to the origin
            x_target = cp.Variable(2)

            # x_target and P are in the local frame, so we need a rotation to
            # compare with the global origin
            theta = x[batch_idx, 2].cpu().detach().numpy()
            rotation_mat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )

            objective = cp.sum_squares(batch_x + rotation_mat @ x_target)

            # We also want x_target to be within the ellipsoid
            constraints.append(cp.quad_form(x_target, P_opt) <= 0.75)

            # Solve for the target point
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve()
            x_target_opt = x_target.value

            # Skip if no solution
            if prob.status != "optimal":
                continue

            # and convert to the global frame
            x_target_opt = rotation_mat @ x_target_opt

            # Now navigate towards that point by offsetting x from this target point
            # (shifting the origin) and applying the nominal controller
            x_shifted = torch.tensor(-x_target_opt).type_as(x)
            x_shifted = torch.cat((x_shifted, x[batch_idx, 2].unsqueeze(-1)))
            u[batch_idx, :] = self.dynamics_model.u_nominal(
                x_shifted.reshape(1, -1), track_zero_angle=False  # type: ignore
            ).squeeze()

            # If we are not at the goal and stuck by a wall, then steer towards the
            # furthest point in the safe ellipse
            P_eigenvals, P_eigenvectors = np.linalg.eigh(P_opt)
            minor_axis_length = 1 / P_eigenvals[-1]
            stuck = minor_axis_length < 0.1
            at_goal = np.linalg.norm(batch_x) < 1e-1
            if not at_goal and stuck:
                major_axis = P_eigenvectors[:, 0]

                # Pick a direction for the major axis arbitrarily (so that we don't turn
                # around unnecessarily).
                if major_axis[0] < 0:
                    major_axis *= -1

                # Re-solve the problem steering towards this point
                objective = cp.sum_squares(x_target - major_axis)
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve()
                x_target_opt = x_target.value

                # Skip if no solution
                if prob.status != "optimal":
                    continue

                # and convert to the global frame
                x_target_opt = rotation_mat @ x_target_opt

                # Now navigate towards that point by offsetting x from this target point
                # (shifting the origin) and applying the nominal controller
                x_shifted = torch.tensor(-x_target_opt).type_as(x)
                x_shifted = torch.cat((x_shifted, x[batch_idx, 2].unsqueeze(-1)))
                u[batch_idx, :] = self.dynamics_model.u_nominal(
                    x_shifted.reshape(1, -1), track_zero_angle=False  # type: ignore
                ).squeeze()

        # Scale the velocities a bit
        u[:, 0] *= 2.0
        u_upper, u_lower = self.dynamics_model.control_limits
        u = torch.clamp(u, u_lower, u_upper)

        # Stop if goal reached
        goal_reached = x[:, :2].norm(dim=-1) < 0.75
        u[goal_reached] *= 0.0

        return u
