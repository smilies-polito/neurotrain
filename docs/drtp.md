# DRTP (Direct Random Target Projection)

Direct Random Target Projection (DRTP) is a local learning rule that avoids
backpropagation through the network. Hidden layers receive fixed random
projections of the target labels, while the output layer is trained with a
local MSE loss on spike counts.

## Enable DRTP

Set the trainer name to `drtp` in your config:

```yaml
trainer:
  name: "drtp"
```

DRTP configs live alongside the other experiment configs in `configs/`:
- `configs/mnist_drtp.yaml`
- `configs/cifar10_drtp.yaml`
- `configs/mnist_drtp_fc1_500.yaml`
- `configs/mnist_drtp_fc2_500.yaml`
- `configs/cifar10_drtp_fc1_500.yaml`
- `configs/cifar10_drtp_fc2_500.yaml`
- `configs/mnist_drtp_conv_rand.yaml`
- `configs/mnist_drtp_conv_train.yaml`
- `configs/cifar10_drtp_conv_rand.yaml`
- `configs/cifar10_drtp_conv_train.yaml`

Run them the same way as other algorithms:

```bash
python3 main.py --config configs/mnist_drtp.yaml
```

## Configuration

DRTP-specific options live under the `drtp` section:

```yaml
drtp:
  loss: "mse"  # "mse", "bce", "ce"
  feedback_distribution: "kaiming_uniform"  # "kaiming_uniform", "uniform", "normal"
  feedback_scale: 1.0
  fixed_feedback: true
```

Notes:
- `loss` controls the output loss used for DRTP updates and logging.
- `feedback_distribution` matches the reference implementation default
  (`kaiming_uniform`).
- `feedback_scale` multiplies the initialized feedback matrices.
- `fixed_feedback` keeps matrices constant for the whole run (default).

To use optimizer-based updates instead of manual DRTP updates, set the training
optimizer (optional):

```yaml
training:
  optimizer: "adam"  # "adam", "sgd", "nag", "rmsprop", or null
```

## Conv Front-End (Optional)

Use `model.architecture: "conv"` and specify the conv front-end as a list of
layers. `model.layer_sizes` then defines the FC sizes after flattening:

```yaml
model:
  architecture: "conv"
  conv_layers:
    - out_channels: 32
      kernel_size: 5
      stride: 1
      padding: 2
      pool_kernel: 2
      pool_stride: 2
  layer_sizes: [1000, 10]
```

To freeze conv weights (random fixed front-end), set:

```yaml
training:
  freeze_conv: true
```

## Per-Timestep Updates (Online)

To update weights every timestep, keep the trainer update settings at the
per-timestep defaults used by the DRTP configs:

```yaml
trainer:
  update_last: false
  update_every: 1
```

## Run in Singularity

From the repository root, use the container in `./src/`:

```bash
# a) minimal sanity check (short run)
singularity exec ./src/snn-training-benchmarking.sif python3 main.py --config configs/mnist_drtp.yaml --epochs 1 --batch-size 8 --T 2

# b) MNIST DRTP with the config
singularity exec ./src/snn-training-benchmarking.sif python3 main.py --config configs/mnist_drtp.yaml

# c) CIFAR10 DRTP with the config
singularity exec ./src/snn-training-benchmarking.sif python3 main.py --config configs/cifar10_drtp.yaml
```

## Limitations / Assumptions

- Supports FC and conv+FC SNNs; recurrent DRTP is not implemented.
- Hidden-layer updates use target projections and local spike gating; no
  backpropagation through hidden layers is used.
- DRTP feedback matrices are stored in checkpoints via trainer state; resuming
  restores them when using the built-in resume flow.
