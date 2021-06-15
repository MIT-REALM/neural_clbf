from abc import ABC

from neural_clbf.systems import ControlAffineSystem


class GenericController(ABC):
    """Represents a generic controller."""

    controller_period: float

    def __init__(
        self, dynamics_model: ControlAffineSystem, controller_period: float = 0.01
    ):
        super(GenericController, self).__init__()
        self.controller_period = controller_period
        self.dynamics_model = dynamics_model

    def __call__(self, x):
        """Compute and return the control input for a tensor of states x"""
        raise NotImplementedError
