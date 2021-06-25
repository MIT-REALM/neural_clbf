from abc import ABC, abstractmethod

from neural_clbf.systems import ControlAffineSystem


class Controller(ABC):
    """Represents a generic controller."""

    controller_period: float

    def __init__(
        self, dynamics_model: ControlAffineSystem, controller_period: float = 0.01
    ):
        super(Controller, self).__init__()
        self.controller_period = controller_period
        self.dynamics_model = dynamics_model

    @abstractmethod
    def u(self, x):
        """Get the control input for a given state"""
        pass
