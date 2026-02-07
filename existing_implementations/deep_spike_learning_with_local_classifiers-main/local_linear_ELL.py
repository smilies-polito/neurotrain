import torch.nn as nn
import torch
import torch.nn.functional as F

from linearFA import LinearFA
from surrogate_gradient import ExponentialSurroGrad


class LocalLossBlockLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes, threshold=None, decay=0.2, first_layer=False,
                 bias=False, random_matrix=False, fa=False,
                 print_stats=False, last_layer=False):
        super(LocalLossBlockLinear, self).__init__()
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.encoder = nn.Linear(num_in, num_out, bias=bias)
        self.random_matrix = random_matrix
        self.fa = fa
        self.decay = decay
        self.is_print_stats = print_stats
        self.thresh = threshold

        # Auxiliary classifier
        if self.fa:
            self.decoder_y = LinearFA(num_out, num_classes, bias=bias)
        else:
            self.decoder_y = nn.Linear(num_out, num_classes, bias=bias)

        if self.random_matrix:
            # Change the parameters of classifier to be not trainable
            self.decoder_y.weight.requires_grad = False
            if self.decoder_y.bias is not None:
                self.decoder_y.bias.requires_grad = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=0.0)

        self.clear_stats()

    def clear_stats(self):
        self.loss_pred = 0.0
        self.correct = 0
        self.examples = 0

    def print_stats(self):
        err = 100.0 * float(self.examples - self.correct) / self.examples
        stats = '{},loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
            self.encoder,
            self.loss_pred / self.examples,
            err,
            self.examples)
        return stats, err

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot, time_window=10):
        # Loop over time window T
        for step in range(time_window):
            # Compute the input current
            if self.first_layer:
                h = self.encoder(x)
            else:
                h = self.encoder(x[step])

             # Membrane integration and spike firing
            if step == 0:
                mem = h
                spike = ExponentialSurroGrad.apply(mem, self.thresh)
                spike_return = spike.clone()
                if self.training or self.is_print_stats:
                    y_hat_mem = self.decoder_y(spike)
                    y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem, self.thresh)
                    spike_sum = y_hat_spike  # Accumulate spike count for decision
            else:
                # Detach the membrane and spike from the computation graph because ELL doesn't consider temporal dependent gradients.
                mem = mem.detach() * self.decay + h - spike.detach() * self.thresh * self.decay
                spike = ExponentialSurroGrad.apply(mem, self.thresh)
                spike_return = torch.cat((spike_return, spike),
                                         dim=0)  # Store output spikes to be propagated to downstream neurons
                if self.training or self.is_print_stats:
                    y_hat_mem = y_hat_mem.detach() * self.decay + self.decoder_y(
                        spike) - y_hat_spike.detach() * self.thresh * self.decay
                    y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem, self.thresh)
                    spike_sum = spike_sum + y_hat_spike

            # Compute loss and update parameters during training
            if self.training or self.is_print_stats:
                loss_sup = F.mse_loss(y_hat_spike, y_onehot.detach())
                if step == 0:
                    loss = loss_sup
                else:
                    loss = loss + loss_sup
                if self.training:
                    # Update weights in this layer
                    self.optimizer.zero_grad()
                    loss_sup.backward(retain_graph=False)
                    self.optimizer.step()

        spike_return = spike_return.view(-1, *spike.size())

        if self.is_print_stats:
            self.loss_pred += loss.item() * y_onehot.size(0)
            self.correct += spike_sum.max(1)[1].eq(y).cpu().sum()
            self.examples += y_onehot.size(0)

        if self.last_layer:
            return (100.0 * float(self.examples - self.correct) / self.examples), loss.item()

        return spike_return.detach(), loss.item()
