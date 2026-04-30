from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTrainer(ABC, nn.Module):
    """
    Abstract Base Class for all trainers.
    Ensures that all trainers implement the required methods.
    """

    @abstractmethod
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train the model on a single sample.
        Must be implemented by subclasses.
        """
        
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the trainer's state.
        Must be implemented by subclasses.
        """
        ...
