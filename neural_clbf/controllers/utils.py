from typing import Union, TYPE_CHECKING

# We only need these imports if type checking, to avoid circular imports
if TYPE_CHECKING:
    from neural_clbf.controllers.neural_rclbf_controller import NeuralrCLBFController
    from neural_clbf.controllers.neural_qp_rclbf_controller import (
        NeuralQPrCLBFController,
    )


# Define a convenience type
Controller = Union["NeuralrCLBFController", "NeuralQPrCLBFController"]
