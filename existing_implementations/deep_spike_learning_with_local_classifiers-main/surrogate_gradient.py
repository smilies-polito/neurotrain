import torch

class ExponentialSurroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return input.ge(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        thresh = ctx.thresh
        grad_input = grad_output.clone()
        return grad_input * torch.exp(-torch.abs(input - thresh)), None