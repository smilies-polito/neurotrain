import torch

class Rate:
    """
    Simulate rate-coded spike trains from static images over T timesteps.
    """
    def __init__(self, T):
        self.T = T

    def __call__(self, input):
        # Flatten image to vector
        input = input.view(-1)
        # Allocate spike tensor [T x features]
        output = torch.zeros((self.T, *input.shape), device=input.device)
        for t in range(self.T):
            # Probabilistic firing based on pixel intensity
            output[t] = torch.rand_like(input).le(input).float()
        return output