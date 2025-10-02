import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import math

class LocalLearningAlgorithm(nn.Module):
    """
    A base class to provide basic functionality for local learning algorithms implemented
    """

    def MNISTLoader(batch_size=128, data_path='/tmp/data/mnist'):
        
        # dataloader arguments
        batch_size = batch_size
        data_path = data_path

        dtype = torch.float
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Define a transform
        transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        # Create DataLoaders
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
        return train_loader, test_loader, device, dtype
    
    def learning_step(net, kwargs):

        """
        A method to be overriden with the fitting number of parameters to implement the learning step
        to be call at each epoch of the outer training loop
        """
        pass

    def train_network(trainer, train_loader, print_intermediate=False, device="cuda"):

        total_samples = 0
        loss_sum = 0.0
        acc_sum  = 0

        for data, target in train_loader:
            data, target = data.transpose(0,1).to(device), target.to(device)
            B = data.size(1)

            trainer.reset()
            loss, pred = trainer.train_sample(data, target)

            total_samples   += B
            loss_sum        += loss.item() * B
            acc_sum         += pred.eq(target.view_as(pred)).sum().item()
            if print_intermediate and (total_samples % 1000 == 0):
                print(f"TRAIN: Processed {total_samples} samples, Partial Loss: {loss_sum/total_samples:.4f}, Partial Accuracy: {acc_sum/total_samples:.4f}")
            if getattr(trainer, "stop_requested", False):
                break

        return loss_sum/total_samples, acc_sum/total_samples

    def test_network(network, test_loader, print_intermediate=False, device="cuda"):
        """
        Evaluate network accuracy over the test_loader.
        Assumes network.forward(x_t) returns *only* the final-layer spike tensor of shape [B, C].
        """
        network.to(device)
        network.eval()

        total_correct = 0
        total_samples = 0

        #print("TEST: Evaluating network...")
        # No gradients needed
        with torch.no_grad():
            for data, target in test_loader:
                # data: [T, B, …], target: [B]
                data   = data.transpose(0, 1).to(device)
                target = target.to(device)
                T, B    = data.shape[0], data.shape[1]

                network.reset()

                # 1) Unroll and sum final‐layer spikes
                spk_sum = None
                for t in range(T):
                    # if your forward returns (spikes, mem), do: spk = network(data[t])[0]
                    spk, _ = network(data[t])
                    spk_sum = spk[-1] if spk_sum is None else spk_sum + spk[-1]

                # 2) Prediction & metric
                logits = spk_sum       # optionally / T for rates
                preds  = logits.argmax(dim=1)
                total_correct += preds.eq(target).sum().item()
                total_samples += B
                if print_intermediate and (total_samples % 10000 == 0):
                    print(f"TEST: Processed {total_samples} samples, Accuracy: {total_correct/total_samples:.4f}")

        return total_correct / total_samples

    def outer_training_loop(net, num_steps, train_loader, test_loader, batch_size, device, dtype):
        num_epochs = 1
        loss_hist = []
        test_loss_hist = []
        counter = 0

        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in train_batch:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                net.train()
                spk_rec, mem_rec = net(data.view(batch_size, -1))

                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)

                # PLUG THE ACTUAL ALGORITHM HERE
                # By calling the overridden version of a traning step!

                LocalLearningAlgorithm.learning_step(net)


                # OLD:
                # Gradient calculation + weight update
                # optimizer.zero_grad()
                # loss_val.backward()
                # optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                # Test set
                with torch.no_grad():
                    net.eval()
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(device)
                    test_targets = test_targets.to(device)

                    # Test set forward pass
                    test_spk, test_mem = net(test_data.view(batch_size, -1))

                    # Test set loss
                    test_loss = torch.zeros((1), dtype=dtype, device=device)
                    for step in range(num_steps):
                        test_loss += loss(test_mem[step], test_targets)
                    test_loss_hist.append(test_loss.item())

                    # Print train/test loss/accuracy
                    if counter % 50 == 0:
                        LocalLearningAlgorithm.print_performances(epoch, iter_counter, loss_hist, test_loss_hist, net, data, targets, batch_size, test_data, test_targets)
                    counter += 1
                    iter_counter +=1
    

    def compute_batch_accuracy(net, data, targets, batch_size, train=False):
        """
        pass data into the network, sum the spikes over time
        and compare the neuron with the highest number of spikes
        with the target
        """
        
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

        return acc

    def print_performances(epoch, iter_counter, loss_hist, test_loss_hist, net, data, targets, batch_size, test_data, test_targets):
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[iter_counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[iter_counter]:.2f}")
        train_acc = LocalLearningAlgorithm.compute_batch_accuracy(net, data, targets, batch_size, train=True)
        test_acc = LocalLearningAlgorithm.compute_batch_accuracy(test_data, test_targets, train=False)
        print("\n")

    # Fixes seeds for both NumPy and PyTorch random number generators. Ensures reproducibility.
    def set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def run_outer_training():
        pass