import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import itertools
import math

class LearningAlgorithms:
    """
    A utility class for training and evaluating models with different trainers.
    """

    @staticmethod
    def train_epoch(
        trainer,
        train_loader: DataLoader,
        device: str = "cuda",
        print_every: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train the model for one epoch.

        Args:
            trainer: An instance of a trainer (e.g., STSFTrainer) implementing the BaseTrainer interface.
            train_loader: DataLoader for the training dataset.
            device: Device to use for training ("cuda" or "cpu").
            print_every: Frequency of logging progress (in number of samples).

        Returns:
            A dictionary containing training loss and accuracy.
        """
        trainer.network.to(device)
        trainer.network.train()

        total_samples = 0
        total_loss = 0.0
        total_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to the device (non_blocking for faster CUDA transfer)
            non_blocking = (
                device == "cuda"
                if isinstance(device, str)
                else getattr(device, "type", None) == "cuda"
            )
            data = data.transpose(0, 1).to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            batch_size = data.size(1)

            # Reset trainer state and perform a training step
            trainer.reset()
            loss, pred = trainer.train_sample(data, target)

            # Update metrics
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += pred.eq(target.view_as(pred)).sum().item()

            # Print progress
            if print_every and total_samples % print_every == 0:
                print(
                    f"[Train] Processed: {total_samples} samples, "
                    f"Loss: {total_loss / total_samples:.4f}, "
                    f"Accuracy: {total_correct / total_samples:.4f}"
                )

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    @staticmethod
    @torch.no_grad()
    def evaluate(
        network,
        test_loader: DataLoader,
        device: str = "cuda",
        print_every: int = 1000,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the test dataset.

        Args:
            network: The network to evaluate.
            test_loader: DataLoader for the test dataset.
            device: Device to use for evaluation ("cuda" or "cpu").
            print_every: Frequency of logging progress (in number of samples).

        Returns:
            A dictionary containing test accuracy.
        """
        network.to(device)
        network.eval()

        total_samples = 0
        total_correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to the device (non_blocking for faster CUDA transfer)
            non_blocking = (
                device == "cuda"
                if isinstance(device, str)
                else getattr(device, "type", None) == "cuda"
            )
            data = data.transpose(0, 1).to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            batch_size = data.size(1)

            # Reset network state and perform a forward pass
            network.reset()
            spk_sum = None
            for t in range(data.size(0)):  # Iterate over timesteps
                out = network(data[t])
                # Support both (spk, mem) and (spk, mem, p) returns
                spk = out[0] if isinstance(out, (tuple, list)) else out
                spk_sum = spk[-1] if spk_sum is None else spk_sum + spk[-1]

            # Compute predictions
            preds = spk_sum.argmax(dim=1)
            total_correct += preds.eq(target).sum().item()
            total_samples += batch_size

            # Print progress
            if print_every and total_samples % print_every == 0:
                print(
                    f"[Eval] Processed: {total_samples} samples, "
                    f"Accuracy: {total_correct / total_samples:.4f}"
                )

        return {
            "accuracy": total_correct / total_samples,
        }