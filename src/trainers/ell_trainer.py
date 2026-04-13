"""
ELL-style local learning trainer
================================

Implementation inspired by:

    "Deep Spike Learning With Local Classifiers"
    C. Ma, R. Yan, Z. Yu, Q. Yu
    IEEE Transactions on Cybernetics, vol. 53, no. 5, pp. 3363-3375, May 2023.
    DOI: 10.1109/TCYB.2022.3188015

This trainer wraps an external spiking network (e.g. FCSNN built with
snnTorch) and augments its hidden layers with auxiliary classifiers for
layer-local supervised learning.

Relation to the paper:
    - The trainer follows the paper's main local-learning idea: each hidden
      layer is paired with a trainable auxiliary classifier and optimized
      with a local MSE objective.
    - The output layer is treated here as the final readout and is supervised
      directly in this implementation, rather than being described through a
      separate auxiliary classifier.
    - The implementation is primarily aligned with the paper's rate-coded
      image-classification setting (Sec. III-B), where the same input frame
      is presented across timesteps.
    - Some neuron-level details depend on the wrapped snnTorch network
      implementation and may therefore differ from the exact equations used
      in the paper.

Architecture:
    The network has a main pathway of spiking layers:

        x ─▶ [Layer 0] ─spk─▶ [Layer 1] ─spk─▶ … ─▶ [Output layer]
               hidden             hidden                 n_classes
               │                  │
               └─▶ aux cls        └─▶ aux cls

    Each hidden layer is paired with a trainable auxiliary classifier used
    during training. The output layer serves as the final readout.

Paper references:
    Sec. II-B states that during inference the auxiliary classifiers are
    discarded, and decisions are made from the network output spikes.
    Sec. III-B states that the categorical decision is determined by the
    neuron with the highest spike count in the output layer. :contentReference[oaicite:0]{index=0}

Algorithmic scope:
    The trainer performs per-timestep local losses and updates in the spirit
    of ELL. It aims to suppress cross-layer gradient flow and to limit
    temporal credit assignment, but exact equivalence to the paper's
    analytical derivation depends on the state-handling semantics of the
    wrapped neuron implementation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  AUXILIARY CLASSIFIER MODULE                                             ║
# ║                                                                          ║
# ║  Paper Sec. II-B, Fig. 1 (orange units):                                ║
# ║  "The number of spiking neurons in the classifier is the same as the     ║
# ║   dimension of the desired outputs."                                     ║
# ║                                                                          ║
# ║  Each hidden layer l gets one of these.  It receives o^l[t] (the         ║
# ║  hidden layer's spike output) and produces o^a[t] (classifier spike)     ║
# ║  via a linear map + LIF neuron.                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class AuxClassifier(nn.Module):
    """
        Auxiliary classifier attached to one hidden layer.

        Architecture:
            hidden spikes → Linear → LIF → classifier spikes

        In the paper, each hidden layer is paired with an auxiliary classifier,
            and the classifier output is compared against the desired output through
            a local squared-error loss (Eq. 4). This module implements that training
            component for the hidden layers. :contentReference[oaicite:1]{index=1}

        Notes:
                - This implementation reuses the neuron configuration of the wrapped
                network for convenience.
                - The paper does not fully specify whether auxiliary classifiers must
                share exactly the same neuron hyperparameters as the corresponding
                hidden layer.
                - Exact neuron dynamics and surrogate behavior depend on snnTorch.
    """

    def __init__(
        self,
        num_in: int,
        num_classes: int,
        beta: float,
        threshold: float,
        spike_grad=None,
        reset_mechanism: str = "subtract",
        bias: bool = False,
    ):
        super().__init__()
        # w^a: auxiliary classifier weights (Eq. 5)
        self.decoder = nn.Linear(num_in, num_classes, bias=bias)
        # LIF readout neuron for the classifier
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
            init_hidden=True,
            output=True,
        )

    def reset(self) -> None:
        self.lif.reset_mem()

    def forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
            One-timestep forward.

            Args:
                spike_in: [B, num_in] — spikes from the hidden layer.

            Returns:
                y_hat_spike: [B, num_classes] — classifier spike o^a[t].
        """
        cur = self.decoder(spike_in)
        y_hat_spike, _ = self.lif(cur)
        return y_hat_spike


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ELL TRAINER — Algorithm 1                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ELLTrainer(BaseTrainer):
    """
            Layer-local supervised trainer inspired by the paper's ELL setting.

            This trainer wraps an external spiking network and adds auxiliary
            classifiers to all hidden layers. Hidden layers are trained with local
            MSE losses through their classifiers, while the output layer is supervised
            directly in this implementation.

            What is faithful to the paper:
                - hidden layers receive trainable local classifiers
                - local MSE losses are used
                - auxiliary classifiers are not used for inference
                - prediction is based on output-layer spike counts

            What is implementation-specific here:
                - the output layer is optimized directly instead of being described as
                another hidden-layer-style local classifier block
                - the exact temporal-detachment behavior depends partly on the wrapped
                neuron model
                - exact neuron equations and surrogate gradients may differ from the
                paper if the underlying snnTorch network differs

            This trainer is best viewed as a practical, paper-aligned implementation
            for rate-coded classification, rather than as a line-by-line reproduction
            of every modeling choice in the article. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 5e-4,
        batch_size: int = 100,
        **kwargs,
    ):
        """
            Args:
                network:
                    A spiking network exposing .synapses, .neurons, .n_classes,
                    and .reset(), for example an FCSNN-style model.

                lr:
                    Learning rate used by the per-layer Adam optimizers.
                    The paper uses Adam and reports task-specific learning-rate
                    schedules in its experimental setup. :contentReference[oaicite:3]{index=3}

                batch_size:
                    Stored for reference. The effective batch size is inferred from
                    the input tensor passed to train_sample().

            Notes:
                - Auxiliary classifiers are created only for hidden layers.
                - The output layer is treated as the network readout and optimized
                directly in this implementation.
                - This mirrors the paper's overall training/inference separation, but
                the exact optimization decomposition is an implementation choice.
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size

        n_layers = len(network.synapses)
        n_hidden = n_layers - 1  # all layers except the last (output)
        n_classes = network.n_classes

        # ── Identify architecture ──
        # network.synapses = [Linear(784→800), Linear(800→10)]
        # network.neurons  = [Leaky,           Leaky          ]
        #                     ^^^^hidden^^^^    ^^^^output^^^^
        # Hidden layers (0 … n_hidden-1): get aux classifiers
        # Output layer  (n_hidden):       IS the readout, no aux classifier

        # ── Create auxiliary classifiers for HIDDEN layers only ──
        # Paper Fig. 1: aux classifiers (orange) only on hidden layers (blue)
        self.aux_classifiers = nn.ModuleList()

        # Extract neuron config from the network's first neuron
        ref_neuron = network.neurons[0]
        neuron_beta = float(ref_neuron.beta)
        neuron_thresh = float(ref_neuron.threshold)
        neuron_reset = ref_neuron.reset_mechanism
        neuron_spike_grad = ref_neuron.spike_grad

        for i in range(n_hidden):
            hidden_size = network.synapses[i].out_features
            self.aux_classifiers.append(
                AuxClassifier(
                    num_in=hidden_size,
                    num_classes=n_classes,
                    beta=neuron_beta,
                    threshold=neuron_thresh,
                    spike_grad=neuron_spike_grad,
                    reset_mechanism=neuron_reset,
                )
            )

        # Move aux classifiers to same device as network
        device = next(network.parameters()).device
        self.aux_classifiers = self.aux_classifiers.to(device)

        # ── Create per-layer optimizers ──
        # Paper Sec. III-B: "We use the Adam optimizer"
        # Paper: "regularization techniques like dropout are not used"
        #
        # Each hidden layer is trained independently (Sec. II-B):
        # optimizer_i covers: synapse[i] + neuron[i] + aux_classifier[i]
        #
        # The output layer has its own optimizer:
        # optimizer covers: synapse[-1] + neuron[-1]
        self.optimizers: List[torch.optim.Adam] = []

        for i in range(n_hidden):
            params = (
                list(network.synapses[i].parameters())
                + list(network.neurons[i].parameters())
                + list(self.aux_classifiers[i].parameters())
            )
            self.optimizers.append(
                torch.optim.Adam(params, lr=lr, weight_decay=0.0)
            )

        # Output layer optimizer (no aux classifier params)
        output_params = (
            list(network.synapses[-1].parameters())
            + list(network.neurons[-1].parameters())
        )
        self.optimizers.append(
            torch.optim.Adam(output_params, lr=lr, weight_decay=0.0)
        )

        self._n_hidden = n_hidden
        self._n_layers = n_layers

    # ══════════════════════════════════════════════════════════════════════
    #  Training — Algorithm 1
    # ══════════════════════════════════════════════════════════════════════

    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Train on one mini-batch with per-timestep local losses.

            Args:
                data:
                    Tensor of shape [T, B, *in_shape].

                    This implementation is primarily intended for the paper's
                    rate-coded image-classification setting, where the same image is
                    effectively presented across timesteps. Accordingly, the trainer
                    uses data[0] as the constant input frame across the simulation
                    window. This is appropriate for that setting, but it is not a
                    general implementation of all temporal tasks studied in the paper.
                    :contentReference[oaicite:4]{index=4}

                target:
                    Tensor of shape [B] with integer class labels.

            Returns:
                avg_loss:
                    Scalar tensor containing the average of the per-layer, per-step
                    losses used for logging.

                pred:
                    Tensor of shape [B] with predicted classes, obtained from the
                    accumulated spike count of the output layer.

            Relation to the paper:
                - Hidden-layer losses follow the paper's local-classifier training
                idea and use MSE against a desired output pattern (Eq. 4). :contentReference[oaicite:5]{index=5}
                - For classification, the paper states that the target neuron is
                trained to fire while others remain silent, and decisions are made
                from output spike counts. This implementation uses a constant
                one-hot target across timesteps for that rate-based setting.
                :contentReference[oaicite:6]{index=6}

            Important implementation note:
                This function aims to emulate ELL-style local learning by blocking
                cross-layer gradients and limiting temporal gradient flow. However,
                exact equivalence to the paper's analytical ELL derivation depends on
                how the wrapped neuron model stores and reuses its internal state.
        """
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device
        n_classes = self.network.n_classes

        # ── Target: constant one-hot at every timestep ──
        # Sec. III-B: "we train the target output neuron to fire at every
        # time step while keeping the others silent."
        #
        # ACCEPTABLE DIVERGENCE #3:
        # Eq. 4 allows time-varying y[t]; we use constant one-hot for
        # image classification (Sec. III-B).
        target_onehot = torch.zeros(batch_size, n_classes, device=device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        # ── Reset all state ──
        self.network.reset()
        for aux in self.aux_classifiers:
            aux.reset()

        # ── Constant input (rate code) ──
        # Sec. III-B: "Pixel values of images are directly fed into the
        # first hidden layer."
        x_const = data[0]

        # Accumulator for output layer spike count (readout)
        # Sec. III-B: "the neuron that generates the highest spike count"
        # This reads from the OUTPUT layer, NOT from aux classifiers.
        spk_sum = torch.zeros(batch_size, n_classes, device=device)
        total_loss = 0.0

        # ── Main loop: Algorithm 1, lines 2–6 ──
        for t in range(num_timesteps):

            # ── Forward one timestep through the main pathway ──
            # Algorithm 1 line 3: "Compute output spikes o^l[t], o^a[t]"
            #
            # We step through each layer manually to:
            # (a) collect intermediate spikes for aux classifiers
            # (b) detach spikes between layers (local learning)
            # (c) detach recurrent state (ELL: no temporal gradient)

            spk = x_const.reshape(x_const.shape[0], -1)  # flatten
            layer_spikes = []  # spikes from each layer (for aux classifiers)

            for i in range(self._n_layers):
                cur = self.network.synapses[i](spk)
                spk_out, mem_out = self.network.neurons[i](cur)
                layer_spikes.append(spk_out)

                # ELL: detach recurrent state between timesteps.
                # Paper Eqs. 5–6: only direct dependencies at current step.
                # snnTorch stores membrane as .mem attribute; detach it so
                # the next timestep's backward won't see this step's graph.
                #
                # NOTE: we detach AFTER collecting spk_out (which is live
                # and part of the current step's graph for backward).
                self.network.neurons[i].mem = self.network.neurons[i].mem.detach()

                # Detach spike for input to the NEXT layer (local learning:
                # no cross-layer gradient, Fig. 1).
                # The live spk_out is kept in layer_spikes[i] for this
                # layer's own loss backward.
                if i < self._n_layers - 1:
                    spk = spk_out.detach()
                # For the output layer, spk_out is used directly for loss

            # Also detach aux classifier recurrent state
            for aux in self.aux_classifiers:
                aux.lif.mem = aux.lif.mem.detach()

            # ── Compute aux classifier outputs for hidden layers ──
            aux_spikes = []
            for i in range(self._n_hidden):
                # Feed the LIVE hidden spike (not detached) into the aux
                # classifier so gradients flow: loss → aux_cls → hidden_spike
                # → hidden_mem → synapse weights (Eq. 6).
                y_hat = self.aux_classifiers[i](layer_spikes[i])
                aux_spikes.append(y_hat)

            # ── Losses — Eq. 4 ──
            # L = ½ Σ_t ‖y[t] − o^a[t]‖²
            #
            # ACCEPTABLE DIVERGENCE #4:
            # Paper Eq. 4 uses ½·sum; F.mse_loss uses mean.
            # Constant factor absorbed into effective learning rate.

            losses = []

            # Hidden layers: MSE on aux classifier spikes
            for i in range(self._n_hidden):
                loss_i = F.mse_loss(aux_spikes[i], target_onehot)
                losses.append(loss_i)

            # Output layer: MSE on output spikes directly
            # (no aux classifier — this IS the readout)
            loss_out = F.mse_loss(layer_spikes[-1], target_onehot)
            losses.append(loss_out)

            # Accumulate scalar loss for reporting
            for loss_l in losses:
                total_loss += loss_l.item()

            # ── Backward + update, per layer ──
            # Algorithm 1 lines 4–5.
            #
            # ACCEPTABLE DIVERGENCE #5:
            # Algorithm 1 says layers can update in parallel.  We serialise
            # backward passes (reverse order) for PyTorch graph lifetime.
            # Mathematically identical.
            n_losses = len(losses)
            for idx in reversed(range(n_losses)):
                self.optimizers[idx].zero_grad()
                losses[idx].backward(retain_graph=(idx > 0))
                self.optimizers[idx].step()

            # ── Readout: accumulate OUTPUT LAYER spikes ──
            # Sec. III-B: "the neuron that generates the highest spike count"
            # Sec. II-B: "input spikes are propagated through the network to
            # generate output ones for a decision"
            # → we read from the output layer, NOT from aux classifiers.
            spk_sum += layer_spikes[-1].detach()

        # ── End of Algorithm 1 ──

        avg_loss = torch.tensor(
            total_loss / (num_timesteps * (self._n_hidden + 1)),
            device=device,
        )
        pred = spk_sum.argmax(dim=1)

        return avg_loss, pred

    # ══════════════════════════════════════════════════════════════════════
    #  Inference — Sec. II-B
    # ══════════════════════════════════════════════════════════════════════

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
            Inference on one batch using only the main network pathway.

            Auxiliary classifiers are not used at inference time. Predictions are
            obtained by accumulating output-layer spikes over time and selecting the
            neuron with the highest spike count, consistent with the paper's
            description of the inference/readout procedure. :contentReference[oaicite:7]{index=7}

            Args:
                data:
                    Tensor of shape [T, B, *in_shape].

            Returns:
                pred:
                    Tensor of shape [B, 1] with predicted classes.

            Note:
                As in train_sample(), this implementation assumes the rate-coded
                classification usage pattern and reuses data[0] across timesteps.
        """
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device
        n_classes = self.network.n_classes

        self.network.reset()
        x_const = data[0]

        spk_sum = torch.zeros(batch_size, n_classes, device=device)

        for t in range(num_timesteps):
            # Standard forward through the main network — no aux classifiers
            spk_rec, mem_rec = self.network(x_const)
            # Accumulate output layer spikes
            spk_sum += spk_rec[-1].detach()

        pred = spk_sum.argmax(dim=1, keepdim=True)
        return pred

    # ══════════════════════════════════════════════════════════════════════
    #  Utilities
    # ══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset the network and aux classifiers' recurrent state."""
        self.network.reset()
        for aux in self.aux_classifiers:
            aux.reset()

    def to(self, device) -> "ELLTrainer":
        """
            Move trainer components to the target device and rebuild optimizers.

            Rebuilding is necessary because optimizers keep references to parameter
            tensors, which change after moving modules across devices.
        """
        super().to(device)
        self.aux_classifiers = self.aux_classifiers.to(device)

        # Rebuild optimizers (old ones hold stale param references)
        self.optimizers = []
        for i in range(self._n_hidden):
            params = (
                list(self.network.synapses[i].parameters())
                + list(self.network.neurons[i].parameters())
                + list(self.aux_classifiers[i].parameters())
            )
            self.optimizers.append(
                torch.optim.Adam(params, lr=self.lr, weight_decay=0.0)
            )
        output_params = (
            list(self.network.synapses[-1].parameters())
            + list(self.network.neurons[-1].parameters())
        )
        self.optimizers.append(
            torch.optim.Adam(output_params, lr=self.lr, weight_decay=0.0)
        )
        return self


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY OF ACCEPTABLE DIVERGENCES FROM THE PAPER                        ║
# ║                                                                          ║
# ║  #1  Aux classifier LIF shares beta/threshold/surrogate with hidden.     ║
# ║      Paper does not specify distinct values.                             ║
# ║      Location: AuxClassifier.__init__                                    ║
# ║                                                                          ║
# ║  #2  data[0] instead of data.mean(dim=0) for constant input.            ║
# ║      Equivalent for rate code; safer for temporal data.                  ║
# ║      Location: ELLTrainer.train_sample                                   ║
# ║                                                                          ║
# ║  #3  Constant one-hot target (Sec. III-B image classification).          ║
# ║      Eq. 4 allows time-varying y[t]; extend for spike-train mapping.    ║
# ║      Location: ELLTrainer.train_sample                                   ║
# ║                                                                          ║
# ║  #4  F.mse_loss(mean) vs paper's ½·sum (Eq. 4).                         ║
# ║      Constant factor absorbed into learning rate.                        ║
# ║      Location: ELLTrainer.train_sample                                   ║
# ║                                                                          ║
# ║  #5  Sequential backward (reverse order) vs parallel update.             ║
# ║      Mathematically identical; serialised for PyTorch graph lifetime.    ║
# ║      Location: ELLTrainer.train_sample                                   ║
# ║                                                                          ║
# ║  #6  snnTorch LIF reset: mem = beta*mem + input - thresh*spk             ║
# ║      Paper Eq. 2: mem = decay*(mem - thresh*spk) + input                 ║
# ║      Difference: reset term not multiplied by decay in snnTorch.         ║
# ║      This is a known minor variant common in SNN implementations.        ║
# ║      Location: inherited from the external FCSNN network.                ║
# ║                                                                          ║
# ║  #7  Surrogate gradient: FCSNN uses fast_sigmoid (snnTorch default).     ║
# ║      Paper Eq. 7 uses exp(−|u−ϑ|).  Both are valid surrogates;          ║
# ║      fast_sigmoid is standard in snnTorch-based implementations.         ║
# ║      Location: inherited from the external FCSNN network.                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝