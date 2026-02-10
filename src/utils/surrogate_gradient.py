"""
Exponential surrogate gradient for Heaviside spike function.

Used for differentiable training of spiking neurons.
"""

import torch


class ExponentialSurroGrad(torch.autograd.Function):
    """Heaviside forward with exponential surrogate in backward."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thresh: float) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return input.ge(thresh).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        grad_input = grad_output.clone()
        return grad_input * torch.exp(-torch.abs(input - thresh)), None
