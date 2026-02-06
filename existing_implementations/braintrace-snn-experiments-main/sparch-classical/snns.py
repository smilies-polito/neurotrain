#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the Spiking Neural Network (SNN) baseline is defined using the
surrogate gradient method.
"""

import brainscale
import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from init import KaimingUniform, Orthogonal


class SpikeFunctionBoxcar(braintools.surrogate.Surrogate):
    def surrogate_grad(self, x) -> jax.Array:
        return jnp.where(jnp.abs(x) > 0.5, 0., 1.)


class SNN(brainstate.nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).

    Arguments
    ---------
    input_shape : tuple of int
        Shape of an input example.
    layer_sizes : list of int
        List of number of neurons in all hidden layers
    neuron_type : str
        Type of neuron model, either 'LIF', 'adLIF', 'RLIF' or 'RadLIF'.
    threshold : float
        Fixed threshold value for the membrane potential.
    dropout : float
        Dropout rate (must be between 0 and 1).
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    use_readout_layer : bool
        If True, the final layer is a non-spiking, non-recurrent LIF and outputs
        a cumulative sum of the membrane potential over time. The outputs have
        shape (batch, labels) with no time dimension. If False, the final layer
        is the same as the hidden layers and outputs spike trains with shape
        (batch, time, labels).
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        neuron_type: str = "LIF",
        threshold: float = 1.0,
        dropout: float = 0.0,
        inp_scale: float = 5 ** 0.5,
        rec_scale: float = 1.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
        use_readout_layer: bool = True,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = input_shape
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.use_readout_layer = use_readout_layer
        self.inp_scale = inp_scale
        self.rec_scale = rec_scale

        if neuron_type not in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            raise ValueError(f"Invalid neuron type {neuron_type}")

        # Init trainable parameters
        self.snn = self._init_layers()

    def _init_layers(self):
        snn = []
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers

        # Hidden layers
        for i in range(num_hidden_layers):
            snn.append(
                globals()[snn_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    inp_scale=self.inp_scale,
                    rec_scale=self.rec_scale,
                )
            )
            input_size = self.layer_sizes[i]

        # Readout layer
        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[-1],
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                )
            )

        return snn

    def update(self, x):
        # Process all layers
        for i, snn_lay in enumerate(self.snn):
            x = snn_lay(x)
        return x


class SNNExtractSpikes(brainstate.nn.Module):
    def __init__(self, net: SNN):
        super().__init__()
        self.net = net

    def update(self, x):
        outs = []
        layers = (
            self.net.snn[:-1]
            if self.net.use_readout_layer else
            self.net.snn
        )
        for layer in layers:
            x = layer(x)
            outs.append(x)
        return outs


class LIFLayer(brainstate.nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        threshold: float = 1.0,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
        inp_scale: float = 5 ** 0.5,
        rec_scale: float = 1.0,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar()

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size,
            self.hidden_size,
            w_init=KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = brainscale.nn.BatchNorm0d(self.hidden_size, momentum=0.05, axis_name='batch')
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = brainscale.nn.LayerNorm(self.hidden_size)
            self.normalize = True
        elif normalization == "none":
            pass
        else:
            raise ValueError("Unsupported normalization type")

        # Initialize dropout
        self.drop = brainscale.nn.Dropout(1 - dropout)

    def update(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.normalize:
            Wx = self.norm(Wx)

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, *args, **kwargs):
        self.ut = brainstate.HiddenState(brainstate.random.rand(self.hidden_size))
        self.st = brainstate.HiddenState(brainstate.random.rand(self.hidden_size))

    def _lif_cell(self, Wx):
        alpha = self.alpha.execute()
        alpha = jnp.clip(alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Compute membrane potential (LIF)
        ut = alpha * self.ut.value - alpha * self.st.value + (1 - alpha) * Wx

        # Compute spikes with surrogate gradient
        st = self.spike_fct(ut - self.threshold)
        self.ut.value = ut
        self.st.value = st
        return st


class adLIFLayer(brainstate.nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        threshold: float = 1.0,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
        inp_scale: float = 5 ** 0.5,
        rec_scale: float = 1.0,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar()

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size,
            self.hidden_size,
            w_init=KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )
        self.beta = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.beta_lim[0], self.beta_lim[1], size=self.hidden_size),
        )
        self.a = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.a_lim[0], self.a_lim[1], size=self.hidden_size),
        )
        self.b = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.b_lim[0], self.b_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = brainscale.nn.BatchNorm0d(self.hidden_size, momentum=0.05, axis_name='batch')
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = brainscale.nn.LayerNorm(self.hidden_size)
            self.normalize = True
        elif normalization == "none":
            pass
        else:
            raise ValueError("Unsupported normalization type")

        # Initialize dropout
        self.drop = brainscale.nn.Dropout(1 - dropout)

    def update(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.normalize:
            Wx = self.norm(Wx)

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, *args, **kwargs):
        self.ut = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.wt = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.st = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def _adlif_cell(self, Wx):
        # Bound values of the neuron parameters to plausible ranges
        alpha = jnp.clip(self.alpha.execute(), min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = jnp.clip(self.beta.execute(), min=self.beta_lim[0], max=self.beta_lim[1])
        a = jnp.clip(self.a.execute(), min=self.a_lim[0], max=self.a_lim[1])
        b = jnp.clip(self.b.execute(), min=self.b_lim[0], max=self.b_lim[1])

        # Compute potential (adLIF)
        wt = beta * self.wt.value + a * self.ut.value + b * self.st.value
        ut = alpha * self.ut.value - alpha * self.st.value + (1 - alpha) * (Wx - wt)

        # Compute spikes with surrogate gradient
        st = self.spike_fct(ut - self.threshold)

        self.ut.value = ut
        self.wt.value = wt
        self.st.value = st
        return st


class RLIFLayer(brainstate.nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        threshold: float = 1.0,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
        inp_scale: float = 5 ** 0.5,
        rec_scale: float = 1.0,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar()

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size,
            self.hidden_size,
            w_init=KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        # Set diagonal elements of recurrent matrix to zero
        w_mask = jnp.ones([self.hidden_size, self.hidden_size])
        w_mask = jnp.fill_diagonal(w_mask, 0, inplace=False)
        self.V = brainscale.nn.Linear(
            self.hidden_size, self.hidden_size,
            w_init=Orthogonal(rec_scale), b_init=None,
            w_mask=w_mask
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = brainscale.nn.BatchNorm0d(self.hidden_size, momentum=0.05, axis_name='batch')
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = brainscale.nn.LayerNorm(self.hidden_size)
            self.normalize = True
        elif normalization == "none":
            pass
        else:
            raise ValueError("Unsupported normalization type")

        # Initialize dropout
        self.drop = brainscale.nn.Dropout(1 - dropout)

    def update(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.normalize:
            Wx = self.norm(Wx)

        # Compute spikes via neuron dynamics
        s = self._rlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, *args, **kwargs):
        self.ut = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.st = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def _rlif_cell(self, Wx):
        # Bound values of the neuron parameters to plausible ranges
        alpha = jnp.clip(self.alpha.execute(), min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Compute membrane potential (RLIF)
        ut = alpha * self.ut.value - alpha * self.st.value + (1 - alpha) * (Wx + self.V(self.st.value))

        # Compute spikes with surrogate gradient
        st = self.spike_fct(ut - self.threshold)

        self.ut.value = ut
        self.st.value = st
        return st


class RadLIFLayer(brainstate.nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RadLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        threshold: float = 1.0,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
        inp_scale: float = 5 ** 0.5,
        rec_scale: float = 1.0,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar()

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size,
            self.hidden_size,
            w_init=KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        # Set diagonal elements of recurrent matrix to zero
        w_mask = jnp.ones([self.hidden_size, self.hidden_size])
        w_mask = jnp.fill_diagonal(w_mask, 0, inplace=False)
        self.V = brainscale.nn.Linear(
            self.hidden_size, self.hidden_size,
            w_init=Orthogonal(rec_scale), b_init=None, w_mask=w_mask
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )
        self.beta = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.beta_lim[0], self.beta_lim[1], size=self.hidden_size),
        )
        self.a = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.a_lim[0], self.a_lim[1], size=self.hidden_size),
        )
        self.b = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.b_lim[0], self.b_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = brainscale.nn.BatchNorm0d(self.hidden_size, momentum=0.05, axis_name='batch')
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = brainscale.nn.LayerNorm(self.hidden_size)
            self.normalize = True
        elif normalization == "none":
            pass
        else:
            raise ValueError("Unsupported normalization type")

        # Initialize dropout
        self.drop = brainscale.nn.Dropout(1 - dropout)

    def update(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.normalize:
            Wx = self.norm(Wx)

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)

        return s

    def init_state(self, *args, **kwargs):
        self.ut = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.wt = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.st = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def _radlif_cell(self, Wx):
        # Bound values of the neuron parameters to plausible ranges
        alpha = jnp.clip(self.alpha.execute(), min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = jnp.clip(self.beta.execute(), min=self.beta_lim[0], max=self.beta_lim[1])
        a = jnp.clip(self.a.execute(), min=self.a_lim[0], max=self.a_lim[1])
        b = jnp.clip(self.b.execute(), min=self.b_lim[0], max=self.b_lim[1])

        # Compute potential (RadLIF)
        wt = beta * self.wt.value + a * self.ut.value + b * self.st.value
        ut = alpha * self.ut.value - alpha * self.st.value + (1 - alpha) * (Wx + self.V(self.st.value) - wt)

        # Compute spikes with surrogate gradient
        st = self.spike_fct(ut - self.threshold)

        self.ut.value = ut
        self.wt.value = wt
        self.st.value = st
        return st


class ReadoutLayer(brainstate.nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        use_bias: bool = False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size, self.hidden_size,
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = brainscale.nn.BatchNorm0d(self.hidden_size, momentum=0.05, axis_name='batch')
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = brainscale.nn.LayerNorm(self.hidden_size)
            self.normalize = True
        elif normalization == "none":
            pass
        else:
            raise ValueError("Unsupported normalization type")

        # Initialize dropout
        self.drop = brainscale.nn.Dropout(1 - dropout)

    def update(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.normalize:
            Wx = self.norm(Wx)

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)
        return out

    def init_state(self, *args, **kwargs):
        self.ut = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        # self.out = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def _readout_cell(self, Wx):
        # Bound values of the neuron parameters to plausible ranges
        alpha = jnp.clip(self.alpha.execute(), min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Compute potential (LIF)
        ut = alpha * self.ut.value + (1 - alpha) * Wx
        self.ut.value = ut
        return ut
        # out = self.out.value + brainstate.functional.softmax(ut)
        # return out
