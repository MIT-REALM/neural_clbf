from typing import Union, TYPE_CHECKING

import torch
from torch.optim import Optimizer

# We only need these imports if type checking, to avoid circular imports
if TYPE_CHECKING:
    from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController

# Define a convenience type
Controller = Union[
    "NeuralCLBFController",
]


class SGA(Optimizer):
    """Implements Stochastic Gradient Ascent for use with primal/dual optimization.

    This optimizer performs gradient ascent but clamps all values to prevent them from
    becoming negative.
    """

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0):
        """Initialize the optimizer.

        args:
            lr - the learning rate
            weight_decay - set to a positive float to allow weight decay
        """
        # Sanity check the input
        if lr <= 0.0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")

        # Make the defaults dict for the Optimizer initializer
        defaults = dict(lr=lr, weight_decay=weight_decay)
        # Call the super initializer
        super(SGA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Based on the SGD code in pytorch (i.e. this code was copy-pasted and then
        modified to do gradient ascent instead of descent, with the original code here:
        https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py)

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        # Evaluate the closure if provided
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Perform the update for each parameter group
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            weight_decay = group["weight_decay"]
            lr = group["lr"]

            # Find out which parameters have gradients
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            # Apply the clamped gradient ascent update to all parameters with gradients
            for i, param in enumerate(params_with_grad):
                # Compute the weight update based on the negative gradient to ascend
                d_p = -1.0 * d_p_list[i]

                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                # Add the weight update (with a negative learning rate, so we run in the
                # direction that decreases weights and increases the loss)
                param.add_(d_p, alpha=-lr)
                # Clamp all parameters to zero
                param.clamp_(0.0)

        return loss
