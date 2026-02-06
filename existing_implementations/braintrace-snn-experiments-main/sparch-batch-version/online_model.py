import datetime
import errno
import os
import time
from datetime import timedelta
from typing import Callable, Union, Optional, Sequence, Any

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.nn._normalizations import _canonicalize_axes, _compute_stats, _normalize, NormalizationParamState

import brainscale
from batchnorm import batch_norm
from general_utils import setup_logging, load_model_states, save_model_states
from spiking_datasets import load_dataset


def print_model_options(logger, args):
    logger.warning(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
    """.format(**vars(args))
    )


def print_training_options(logger, args):
    logger.warning(
        """
        Training Config
        ---------------
        Load experiment folder: {load_exp_folder}
        New experiment folder: {new_exp_folder}
        Dataset name: {dataset_name}
        Data folder: {data_folder}
        Save best model: {save_best}
        Batch size: {batch_size}
        Number of epochs: {nb_epochs}
        Start epoch: {start_epoch}
        Initial learning rate: {lr}
        Use data augmentation: {use_augm}
    """.format(**vars(args))
    )


class BatchNorm0d_V1(brainstate.nn.Module):
    num_spatial_dims: int = 0

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        feature_axis: brainstate.typing.Axes = -1,
        *,
        track_running_stats: bool = True,
        epsilon: float = 1e-5,
        momentum: float = 0.99,
        affine: bool = True,
        BN_type: str = 'old',
        bias_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(0.),
        scale_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(1.),
        axis_name: Optional[Union[str, Sequence[str]]] = None,
        axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        use_fast_variance: bool = True,
        name: Optional[str] = None,
        dtype: Any = None,
    ):
        super().__init__(name=name)

        # parameters
        self.BN_type = BN_type
        self.in_size = in_size
        self.out_size = in_size
        self.affine = affine
        self.bias_initializer = bias_initializer
        self.scale_initializer = scale_initializer
        self.dtype = dtype or brainstate.environ.dftype()
        self.track_running_stats = track_running_stats
        self.momentum = jnp.asarray(momentum, dtype=self.dtype)
        self.epsilon = jnp.asarray(epsilon, dtype=self.dtype)
        self.use_fast_variance = use_fast_variance

        # parameters about axis
        feature_axis = (feature_axis,) if isinstance(feature_axis, int) else feature_axis
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axis)
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        # variables
        feature_shape = tuple([(ax if i in self.feature_axes else 1) for i, ax in enumerate(self.in_size)])
        self.feature_shape = feature_shape
        if self.track_running_stats:
            self.running_mean = brainstate.BatchState(jnp.zeros(feature_shape, dtype=self.dtype))
            self.running_var = brainstate.BatchState(jnp.ones(feature_shape, dtype=self.dtype))
        else:
            self.running_mean = None
            self.running_var = None

        # parameters
        if self.affine:
            assert track_running_stats, "Affine parameters are not needed when track_running_stats is False."
            bias = braintools.init.param(self.bias_initializer, feature_shape)
            scale = braintools.init.param(self.scale_initializer, feature_shape)
            self.weight = NormalizationParamState(dict(bias=bias, scale=scale))
        else:
            self.weight = None

    def init_state(self, batch_size=None, **kwargs):
        self.total_mean = brainstate.ShortTermState(jnp.zeros(self.feature_shape, dtype=self.dtype))
        self.total_var = brainstate.ShortTermState(jnp.zeros(self.feature_shape, dtype=self.dtype))
        self.index = brainstate.ShortTermState(0)

    def update(self, x):
        fit_phase = brainstate.environ.get('fit', desc='Whether this is a fitting process. Bool.')
        self.index.value += 1
        out, self.total_mean.value, self.total_var.value = batch_norm(
            x, self.weight.value['scale'], self.weight.value['bias'],
            self.running_mean.value, self.running_var.value,
            self.total_mean.value, self.total_var.value,
            fit_phase, self.epsilon, -1
        )
        return out

    def finalize(self):
        mean = self.total_mean.value / self.index.value
        var = self.total_var.value / self.index.value
        self.running_mean.value += (1 - self.momentum) * (mean - self.running_mean.value)
        self.running_var.value += (1 - self.momentum) * (var - self.running_var.value)


class BatchNorm0d_V2(brainstate.nn.Module):
    num_spatial_dims: int = 0

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        feature_axis: brainstate.typing.Axes = -1,
        *,
        track_running_stats: bool = True,
        epsilon: float = 1e-5,
        momentum: float = 0.99,
        affine: bool = True,
        BN_type: str = 'new',
        bias_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(0.),
        scale_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(1.),
        axis_name: Optional[Union[str, Sequence[str]]] = None,
        axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        use_fast_variance: bool = True,
        name: Optional[str] = None,
        dtype: Any = None,
    ):
        super().__init__(name=name)

        # parameters
        self.BN_type = BN_type
        self.in_size = in_size
        self.out_size = in_size
        self.affine = affine
        self.bias_initializer = bias_initializer
        self.scale_initializer = scale_initializer
        self.dtype = dtype or brainstate.environ.dftype()
        self.track_running_stats = track_running_stats
        self.momentum = jnp.asarray(momentum, dtype=self.dtype)
        self.epsilon = jnp.asarray(epsilon, dtype=self.dtype)
        self.use_fast_variance = use_fast_variance

        # parameters about axis
        feature_axis = (feature_axis,) if isinstance(feature_axis, int) else feature_axis
        self.feature_axes = _canonicalize_axes(len(self.in_size), feature_axis)
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        # variables
        feature_shape = tuple([(ax if i in self.feature_axes else 1) for i, ax in enumerate(self.in_size)])
        self.feature_shape = feature_shape
        if self.track_running_stats:
            self.running_mean = brainstate.BatchState(jnp.zeros(feature_shape, dtype=self.dtype))
            self.running_var = brainstate.BatchState(jnp.ones(feature_shape, dtype=self.dtype))
        else:
            self.running_mean = None
            self.running_var = None

        # parameters
        if self.affine:
            assert track_running_stats, "Affine parameters are not needed when track_running_stats is False."
            bias = braintools.init.param(self.bias_initializer, feature_shape)
            scale = braintools.init.param(self.scale_initializer, feature_shape)
            self.weight = NormalizationParamState(dict(bias=bias, scale=scale))
        else:
            self.weight = None

    def init_state(self, batch_size=None, **kwargs):
        size = self.feature_shape if batch_size is None else (batch_size,) + self.feature_shape
        self.total_mean = brainstate.ShortTermState(jnp.zeros(size, dtype=self.dtype))
        self.index = brainstate.ShortTermState(0)

    def update(self, x, mask: Optional[jax.Array] = None):
        # input shape and batch mode or not
        if x.ndim == self.num_spatial_dims + 2:
            x_shape = x.shape[1:]
            batch = True
        elif x.ndim == self.num_spatial_dims + 1:
            x_shape = x.shape
            batch = False
        else:
            raise ValueError(
                f"expected {self.num_spatial_dims + 2}D (with batch) or "
                f"{self.num_spatial_dims + 1}D (without batch) input (got {x.ndim}D input, {x.shape})"
            )
        if self.in_size != x_shape:
            raise ValueError(f"The expected input shape is {self.in_size}, while we got {x_shape}.")

        # reduce the feature axis
        if batch:
            reduction_axes = tuple(i for i in range(x.ndim) if (i - 1) not in self.feature_axes)
        else:
            reduction_axes = tuple(i for i in range(x.ndim) if i not in self.feature_axes)

        # fitting phase
        fit_phase = brainstate.environ.get('fit', desc='Whether this is a fitting process. Bool.')
        self.index.value += 1

        # compute the running mean and variance
        if self.track_running_stats:
            if fit_phase:
                self.total_mean.value += x
                reduction_axes = tuple(i for i in range(x.ndim) if (i - 1) not in self.feature_axes)
                mean, var = _compute_stats(
                    self.total_mean.value / self.index.value,
                    reduction_axes,
                    dtype=self.dtype, axis_name=self.axis_name,
                    axis_index_groups=self.axis_index_groups,
                    use_fast_variance=self.use_fast_variance,
                )
                mean = self.momentum * self.running_mean.value + (1 - self.momentum) * mean
                var = self.momentum * self.running_var.value + (1 - self.momentum) * var
            else:
                mean = self.running_mean.value
                var = self.running_var.value
        else:
            mean, var = None, None

        # normalize
        return _normalize(
            x, mean=mean, var=var, weights=self.weight,
            reduction_axes=reduction_axes,
            feature_axes=self.feature_axes,
            dtype=self.dtype, epsilon=self.epsilon
        )

    def finalize(self):
        x = self.total_mean.value / self.index.value
        reduction_axes = tuple(i for i in range(x.ndim) if (i - 1) not in self.feature_axes)
        mean, var = _compute_stats(
            x, reduction_axes,
            dtype=self.dtype, axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
        )
        self.running_mean.value = self.momentum * self.running_mean.value + (1 - self.momentum) * mean
        self.running_var.value = self.momentum * self.running_var.value + (1 - self.momentum) * var


norm_type = BatchNorm0d_V1


# norm_type = BatchNorm0d_V2


def init(size):
    # return brainstate.random.rand(*size)
    return jnp.zeros(size)


def normalize(self, x):
    # if self.normalize:
    #     x = self.norm(x)
    # return self.W(x)

    Wx = self.W(x)
    if self.normalize:
        Wx = self.norm(Wx)
    return Wx


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
    input_shape : tuple
        Shape of an input example.
    layer_sizes : int list
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
        momentum: float = 0.9,
        relu_width: float = 1.
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
        self.momentum = momentum
        self.relu_width = relu_width

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
                    momentum=self.momentum,
                    relu_width=self.relu_width,
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
                    momentum=self.momentum,
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
        layers = self.net.snn[:-1] if self.net.use_readout_layer else self.net.snn
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
        momentum: float = 0.9,
        relu_width: float = 1.
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 250)]
        self.spike_fct = SpikeFunctionBoxcar()
        self.spike_fct = braintools.surrogate.ReluGrad(width=relu_width)

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size,
            self.hidden_size,
            w_init=braintools.init.KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = norm_type(self.hidden_size, momentum=momentum)
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
        Wx = normalize(self, x)

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.ut = brainstate.HiddenState(init(size))
        self.st = brainstate.HiddenState(init(size))

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
        momentum: float = 0.9,
        relu_width: float = 1.
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 250)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar()
        # self.spike_fct = braintools.surrogate.ReluGrad(width=relu_width)

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size, self.hidden_size,
            w_init=braintools.init.KaimingUniform(inp_scale),
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
            self.norm = norm_type(self.hidden_size, momentum=momentum)
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
        Wx = normalize(self, x)

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.ut = brainstate.HiddenState(init(size))
        self.wt = brainstate.HiddenState(init(size))
        self.st = brainstate.HiddenState(init(size))

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
        momentum: float = 0.9,
        relu_width: float = 1.
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 250)]
        self.spike_fct = SpikeFunctionBoxcar()
        self.spike_fct = braintools.surrogate.ReluGrad(width=relu_width)

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size, self.hidden_size,
            w_init=braintools.init.KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        # Set diagonal elements of recurrent matrix to zero
        w_mask = jnp.ones([self.hidden_size, self.hidden_size])
        w_mask = jnp.fill_diagonal(w_mask, 0, inplace=False)
        self.V = brainscale.nn.Linear(
            self.hidden_size, self.hidden_size,
            w_init=braintools.init.Orthogonal(rec_scale), b_init=None, w_mask=w_mask
        )
        self.alpha = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.alpha_lim[0], self.alpha_lim[1], size=self.hidden_size),
        )

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = norm_type(self.hidden_size, momentum=momentum)
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
        Wx = normalize(self, x)

        # Compute spikes via neuron dynamics
        s = self._rlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)
        return s

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.ut = brainstate.HiddenState(init(size))
        self.st = brainstate.HiddenState(init(size))

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
        momentum: float = 0.9,
        relu_width: float = 1.
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 250)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar()
        # self.spike_fct = braintools.surrogate.ReluGrad(width=relu_width)

        # Trainable parameters
        bound = 1 / self.input_size ** 0.5
        self.W = brainscale.nn.Linear(
            self.input_size, self.hidden_size,
            w_init=braintools.init.KaimingUniform(inp_scale),
            b_init=braintools.init.Uniform(-bound, bound) if use_bias else None
        )
        # Set diagonal elements of recurrent matrix to zero
        w_mask = jnp.ones([self.hidden_size, self.hidden_size])
        w_mask = jnp.fill_diagonal(w_mask, 0, inplace=False)
        self.V = brainscale.nn.Linear(
            self.hidden_size, self.hidden_size,
            w_init=braintools.init.Orthogonal(rec_scale), b_init=None, w_mask=w_mask
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
            self.norm = norm_type(self.hidden_size, momentum=momentum)
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
        Wx = normalize(self, x)

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Apply dropout
        s = self.drop(s)

        return s

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.ut = brainstate.HiddenState(init(size))
        self.wt = brainstate.HiddenState(init(size))
        self.st = brainstate.HiddenState(init(size))

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
        momentum: float = 0.9,
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
            self.norm = norm_type(self.hidden_size, momentum=momentum)
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

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.ut = brainstate.HiddenState(init(size))

    def _readout_cell(self, Wx):
        # Bound values of the neuron parameters to plausible ranges
        alpha = jnp.clip(self.alpha.execute(), min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Compute potential (LIF)
        ut = alpha * self.ut.value + (1 - alpha) * Wx
        self.ut.value = ut
        return ut
        # out = self.out.value + brainstate.functional.softmax(ut)
        # return out


class Experiment(brainstate.util.PrettyObject):
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):
        self.args = args

        # New model config
        self.net_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias

        # Training config
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.use_augm = args.use_augm

        # Initialize logging and output folders
        self.init_exp_folders()
        self.logger = setup_logging(os.path.join(self.log_dir, 'exp.log'))
        print_model_options(self.logger, args)
        print_training_options(self.logger, args)

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.trainable_weights = self.net.states(brainstate.ParamState)
        lr = braintools.optim.StepLR(self.lr, step_size=args.lr_step_size, gamma=args.lr_step_gamma)
        self.optimizer = braintools.optim.Adam(lr)
        self.optimizer.register_trainable_weights(self.trainable_weights)

    def f_train(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        # Initialize best accuracy
        best_epoch, best_acc = 0, 0

        # Loop over epochs (training + validation)
        self.logger.warning("\n------ Begin training ------\n")
        for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
            self.train_one_epoch(e)
            best_epoch, best_acc = self.valid_one_epoch(e, best_epoch, best_acc)
            self.optimizer.lr.step_epoch()
        self.logger.warning(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
        self.logger.warning("\n------ Training finished ------\n")

        # Loading best model
        if self.save_best:
            load_model_states(f"{self.checkpoint_dir}/best_model.pth", self.net)
            self.logger.warning(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")
        else:
            self.logger.warning(
                "Cannot load best model because save_best option is "
                "disabled. Model from last epoch is used for testing."
            )

        # Test trained model
        self.test_one_epoch(self.valid_loader)
        self.logger.warning("\nThis dataset uses the same split for validation and testing.\n")

    def f_test(self, n_fig=5):
        data = iter(self.valid_loader)

        for _ in range(5):
            x, y = next(data)

            # validation
            x = jnp.asarray(x)
            print(x.shape)
            outs = self._validate(x)
            outs = jax.tree.map(np.asarray, outs)

            # visualization
            fig, gs = braintools.visualize.get_figure(len(outs), n_fig, 3, 3)
            for i, out in enumerate(outs):
                for i_img in range(n_fig):
                    fig.add_subplot(gs[i, i_img])
                    spikes = out[:, i_img]
                    spikes = np.reshape(spikes, (spikes.shape[0], -1))
                    # Create a raster plot of spikes
                    neuron_indices = np.where(spikes > 0)
                    plt.scatter(neuron_indices[0], neuron_indices[1], s=1, c='black', marker='|')
                    plt.ylabel('Neuron Index')
                    plt.xlabel('Time Step')
                    plt.title(f'Sample {i_img}, Layer {i}')
            plt.show()
            plt.close()

    def _validate(self, inputs):
        inputs = self._process_input(inputs)

        # add environment context
        model = brainstate.nn.EnvironContext(SNNExtractSpikes(self.net), fit=False)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.vmap_init_all_states(model, state_tag='hidden', axis_size=n_batch)

        # forward propagation
        outs = brainstate.transform.for_loop(model, inputs)

        return outs

    def _loss(self, predictions, targets):
        return braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, targets).mean()

    def _acc(self, predictions, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(predictions, axis=1)))

    def _process_input(self, inputs):
        inputs = u.math.flatten(jnp.asarray(inputs), start_axis=2)
        inputs = inputs.transpose((1, 0, 2))  # [n_time, n_batch, n_feature]
        return inputs

    @brainstate.transform.jit(static_argnums=0)
    def predict(self, inputs: jax.Array, targets: jax.Array):
        inputs = self._process_input(inputs)

        # add environment context
        model = brainstate.nn.EnvironContext(self.net, fit=False)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.init_all_states(model, batch_size=n_batch)

        # forward propagation
        outs = brainstate.transform.for_loop(model, inputs)
        outs = outs.sum(axis=0)
        # outs = outs[-1]

        loss = self._loss(outs, targets)
        acc = self._acc(outs, targets)
        return acc, loss

    @brainstate.transform.jit(static_argnums=0)
    def online_train(self, inputs, targets):
        inputs = self._process_input(inputs)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]

        # initialize the online learning model
        model = brainstate.nn.EnvironContext(self.net, fit=True)
        if self.args.method == 'esd-rtrl':
            model = brainscale.IODimVjpAlgorithm(model,
                                                 self.args.etrace_decay,
                                                 vjp_method=self.args.vjp_method,
                                                 mode=brainstate.mixin.Batching())
        elif self.args.method == 'd-rtrl':
            model = brainscale.ParamDimVjpAlgorithm(model,
                                                    vjp_method=self.args.vjp_method,
                                                    mode=brainstate.mixin.Batching())
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        inp = jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype)
        brainstate.nn.init_all_states(self.net, batch_size=n_batch)
        model.compile_graph(inp)
        model.show_graph()

        def _etrace_grad(inp):
            out = model(inp)
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(_etrace_grad, self.trainable_weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.trainable_weights.to_dict_values())
            grads, (outs, losses) = brainstate.transform.scan(_etrace_step, grads, inputs_)
            # gradient updates
            self.optimizer.update(grads)
            # accuracy
            return losses.mean(), outs.sum(axis=0)

        loss, out_sum = _etrace_train(inputs)
        acc = self._acc(out_sum, targets)

        # Finalize batchnorm statistics after each parameter update
        for node in self.net.nodes(norm_type).values():
            node.finalize()
        return acc, loss

    def init_exp_folders(self):
        # Use given path for new model folder
        if self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        else:
            # Generate a path for new model from chosen config
            if self.args.method == 'esd-rtrl':
                outname = f'{self.args.method}_{self.args.etrace_decay}_{self.dataset_name}/'
            else:
                outname = f'{self.args.method}_{self.dataset_name}/'
            outname = outname + self.net_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_lr" + str(self.lr)
            exp_folder = f"{outname.replace('.', '_')}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

        # For a new model check that out path does not exist
        if os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder
        self.checkpoint_dir = exp_folder
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        results = load_dataset(self.args)

        self.nb_inputs = results['in_shape']
        self.nb_outputs = results['out_shape']

        self.train_loader = results['train_loader']
        self.valid_loader = results['test_loader']
        if self.use_augm:
            self.logger.warning("\nWarning: Data augmentation not implemented for SHD and SSC.\n")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.net_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            self.net = SNN(
                input_shape=self.nb_inputs,
                layer_sizes=layer_sizes,
                neuron_type=self.net_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                use_readout_layer=True,
                inp_scale=self.args.inp_scale,
                rec_scale=self.args.rec_scale,
                momentum=self.args.momentum,
                relu_width=self.args.relu_width,
            )
            self.logger.warning(f"\nCreated new spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.net_type}")

        table, _ = brainstate.nn.count_parameters(self.net, return_table=True)
        self.logger.warning('\n' + str(table))

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        losses, accs = [], []

        # Loop over batches from train set
        for step, (x, y) in enumerate(self.train_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            acc, loss = self.online_train(x, y)
            losses.append(loss)
            accs.append(acc)

        # Learning rate of whole epoch
        current_lr = self.optimizer.lr()
        self.logger.warning(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        self.logger.warning(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        self.logger.warning(f"Epoch {e}: train acc={train_acc}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        self.logger.warning(f"Epoch {e}: train elapsed time={elapsed}")

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        losses, accs = [], []

        # Loop over batches from validation set
        for step, (x, y) in enumerate(self.valid_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            acc, loss = self.predict(x, y)
            losses.append(loss)
            accs.append(acc)

        # Validation loss of whole epoch
        valid_loss = np.mean(losses)
        self.logger.warning(f"Epoch {e}: valid loss={valid_loss}")

        # Validation accuracy of whole epoch
        valid_acc = np.mean(accs)
        self.logger.warning(f"Epoch {e}: valid acc={valid_acc}")

        # # Update learning rate
        # self.scheduler.step(valid_acc)

        # Update the best epoch and accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = e

            # Save best model
            if self.save_best:
                save_model_states(
                    f"{self.checkpoint_dir}/best_model.pth", self.net, valid_acc=best_acc, epoch=best_epoch)
                self.logger.warning(f"\nBest model saved with valid acc={valid_acc}")

        self.logger.warning("\n-----------------------------\n")

        return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        losses, accs = [], []

        self.logger.warning("\n------ Begin Testing ------\n")

        # Loop over batches from test set
        for step, (x, y) in enumerate(test_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            acc, loss = self.predict(x, y)
            losses.append(loss)
            accs.append(acc)

        # Test loss
        test_loss = np.mean(losses)
        self.logger.warning(f"Test loss={test_loss}")

        # Test accuracy
        test_acc = np.mean(accs)
        self.logger.warning(f"Test acc={test_acc}")

        self.logger.warning("\n-----------------------------\n")
