import torch

from neural_clbf.systems import ControlAffineSystem


def normalize(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
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


def normalize_with_angles(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
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
