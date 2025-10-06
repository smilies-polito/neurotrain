from abc import ABC, abstractmethod
import numpy as np 
import torch
import math
import torch.nn as nn

class BaseTrainer(ABC, nn.Module):
    """
    Abstract Base Class for all trainers.
    Ensures that all trainers implement the required methods.
    """

    @abstractmethod
    def train_sample(self, data, target):
        """
        Train the model on a single sample.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the trainer's state.
        Must be implemented by subclasses.
        """
        pass