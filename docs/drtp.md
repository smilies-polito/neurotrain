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

Run them the same way as other algorithms:

```bash
python3 main.py --config configs/mnist_drtp.yaml
```

## Configuration

DRTP-specific options live under the `drtp` section:

```yaml
drtp:
  feedback_distribution: "kaiming_uniform"  # "kaiming_uniform", "uniform", "normal"
  feedback_scale: 1.0
  fixed_feedback: true
```

Notes:
- `feedback_distribution` matches the reference implementation default
  (`kaiming_uniform`).
- `feedback_scale` multiplies the initialized feedback matrices.
- `fixed_feedback` keeps matrices constant for the whole run (default).

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

- Supports `FCNetwork` architectures only (feedforward SNNs).
- Hidden-layer updates use target projections and local spike gating; no
  backpropagation through hidden layers is used.
- Checkpoints save the model and optimizer by default; fixed DRTP matrices are
  stored in the trainer state, so resuming from checkpoints will reinitialize
  them unless you persist the trainer state separately.
