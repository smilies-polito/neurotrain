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



class LocalLearningAlgorithm(nn.Module):
    """
    A base class to provide basic functionality for local learning algorithms implemented
    """

    def setup(batch_size=128, data_path='/tmp/data/mnist'):
        
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


    def run_outer_training():
        pass