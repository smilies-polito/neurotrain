# Benchmarking SNN Models

This folder provides four single-timestep SNN backbones for benchmarking:
- `FCSNN` (`fc_snn.py`)
- `RSNN` (`r_snn.py`)
- `ConvSNN` (`conv_snn.py`)
- `VG11SNN` (`vg11_snn.py`)

## Verification against common implementations

References checked:
- snnTorch Quickstart: https://snntorch.readthedocs.io/en/latest/quickstart.html
- snnTorch Tutorial 5 (FC): https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
- snnTorch Tutorial 6 (Conv): https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
- snnTorch `RLeaky` docs/source: https://snntorch.readthedocs.io/en/latest/_modules/snntorch/_neurons/rleaky.html
- torchvision VGG source (canonical VGG config pattern): https://docs.pytorch.org/vision/main/_modules/torchvision/models/vgg.html

Conclusions:
- `FCSNN`: standard feedforward SNN baseline (`Linear -> LIF`).
- `RSNN`: standard recurrent SNN baseline (`Linear -> RLeaky` recurrent core + spiking readout).
- `ConvSNN`: standard conv-SNN style using `Conv -> (BN) -> Pool -> LIF` feature blocks.
- `VG11SNN`: standard VGG-like SNN style using repeated conv blocks with interleaved pooling and optional BN.

## Shared behavior

- `forward` processes one timestep only (single batch).
- Time loops are external (trainer controls sequence iteration).
- Neuron states are persistent across calls via `init_hidden=True`, and are cleared only by `reset()`.
- Return value is always `(spk_rec, mem_rec)` as layer-ordered lists for that timestep.

---

## FCSNN (`fc_snn.py`)

### Architecture

```text
(B, *in_shape)
  -> Flatten
  -> [Linear -> Leaky] x N_hidden
  -> Linear -> Leaky (output)
```

### Parameters

- `in_shape`: input shape excluding batch dimension.
- `num_classes`: output class count.
- `hidden_sizes`: hidden layer sizes. `()` gives direct input->output.
- `beta`: LIF membrane decay.
- `threshold`: LIF firing threshold.
- `spike_grad`: surrogate gradient function. Defaults to `fast_sigmoid(slope=25)`.

### Usage examples

```python
from networks.fc_snn import FCSNN

model = FCSNN(in_shape=(1, 28, 28), num_classes=10, hidden_sizes=(256, 128))
model.reset()
spk_rec, mem_rec = model(x_t)  # x_t: (B, 1, 28, 28)
```

```python
model = FCSNN(in_shape=(784,), num_classes=10, hidden_sizes=())
for t in range(T):
    spk_rec, mem_rec = model(x_seq[t])  # x_seq[t]: (B, 784)
```

---

## RSNN (`r_snn.py`)

### Architecture

```text
(B, *in_shape)
  -> Flatten
  -> [Linear -> RLeaky(recurrent)] x N_recurrent
  -> Linear -> Leaky (output)
```

### Parameters

- `in_shape`: input shape excluding batch dimension.
- `num_classes`: output class count.
- `hidden_sizes`: recurrent hidden sizes. Example: `(256,)`, `(256, 128)`.
- `beta`: decay for recurrent and output spiking neurons.
- `threshold`: firing threshold for recurrent and output neurons.
- `spike_grad`: surrogate gradient function. Defaults to `fast_sigmoid(slope=25)`.

Notes:
- Recurrent core uses `snn.RLeaky(all_to_all=True, learn_recurrent=True)`.
- Suitable as architecture for BPTT/e-prop-style recurrent experiments.

### Usage examples

```python
from networks.r_snn import RSNN

model = RSNN(in_shape=(1, 28, 28), num_classes=10, hidden_sizes=(256,))
model.reset()
spk_rec, mem_rec = model(x_t)
```

```python
model = RSNN(in_shape=(64,), num_classes=5, hidden_sizes=(128, 64))
for t in range(T):
    spk_rec, mem_rec = model(x_seq[t])
```

---

## ConvSNN (`conv_snn.py`)

### Architecture

```text
(B, C, H, W)
  -> [Conv2d -> (optional BatchNorm2d) -> (optional Pool) -> Leaky] x N_conv
  -> Flatten
  -> [Linear -> Leaky] x N_fc_hidden
  -> Linear -> Leaky (output)
```

### Parameters

- `in_shape`: input shape `(C, H, W)`.
- `num_classes`: output class count.
- `conv_channels`: output channels for each conv layer.
- `fc_hidden_sizes`: classifier hidden layer sizes. `()` gives direct output layer.
- `use_batch_norm`: if `True`, apply `BatchNorm2d` after each conv.
- `pool_after`: bool list controlling pooling per conv block.
- `pool_kernel`, `pool_stride`: pooling hyperparameters.
- `conv_kernel_size`, `conv_stride`, `conv_padding`: convolution hyperparameters.
- `beta`: LIF membrane decay.
- `threshold`: LIF firing threshold.
- `spike_grad`: surrogate gradient function. Defaults to `fast_sigmoid(slope=25)`.

### Usage examples

```python
from networks.conv_snn import ConvSNN

model = ConvSNN(
    in_shape=(1, 28, 28),
    num_classes=10,
    conv_channels=(32, 64),
    fc_hidden_sizes=(128,),
    use_batch_norm=True,
)
model.reset()
spk_rec, mem_rec = model(x_t)  # x_t: (B, 1, 28, 28)
```

```python
model = ConvSNN(
    in_shape=(3, 32, 32),
    num_classes=10,
    conv_channels=(32, 64, 64),
    pool_after=(True, False, True),
    fc_hidden_sizes=(256, 128),
    use_batch_norm=False,
)
```

---

## VG11SNN (`vg11_snn.py`)

### Architecture

```text
(B, C, H, W)
  -> feature_cfg-driven blocks:
     int token: Conv2d -> (optional BatchNorm2d) -> (optional Pool from "M") -> Leaky
     "M" token: attach Pool to previous conv block
  -> Flatten
  -> [Linear -> Leaky] x N_classifier_hidden
  -> Linear -> Leaky (output)
```

Default `feature_cfg` follows VGG-11 stage rhythm (conv groups + interleaved pooling),
optionally scaled by `base_channels`.

### Parameters

- `in_shape`: input shape `(C, H, W)`.
- `num_classes`: output class count.
- `feature_cfg`: sequence of `int` and `"M"` tokens controlling feature extractor.
- `classifier_hidden_sizes`: classifier hidden sizes. `()` gives direct output layer.
- `base_channels`: multiplier base for integer feature tokens.
- `cfg_scale_with_base_channels`: if `True`, scales integer feature tokens by `base_channels`.
- `use_batch_norm`: if `True`, apply `BatchNorm2d` after each conv.
- `pool_kernel`, `pool_stride`: pooling hyperparameters used by `"M"` tokens.
- `beta`: LIF membrane decay.
- `threshold`: LIF firing threshold.
- `spike_grad`: surrogate gradient function. Defaults to `fast_sigmoid(slope=25)`.

### Usage examples

```python
from networks.vg11_snn import VG11SNN

model = VG11SNN(
    in_shape=(3, 32, 32),
    num_classes=10,
    base_channels=16,
    use_batch_norm=True,
)
model.reset()
spk_rec, mem_rec = model(x_t)  # x_t: (B, 3, 32, 32)
```

```python
model = VG11SNN(
    in_shape=(3, 32, 32),
    num_classes=10,
    feature_cfg=(1, "M", 2, 2, "M", 4),
    base_channels=8,
    classifier_hidden_sizes=(128, 64),
    use_batch_norm=False,
)
```
