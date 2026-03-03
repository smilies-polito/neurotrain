# SNN Training Benchmarking

This repository contains code for benchmarking the training of Spiking Neural Networks (SNNs). The code is structured in a modular way, clearly separating network architectures and training algorithms which can be mixed and matched to evaluate different combinations of architectures and training methods.

## Repository Structure

- `📂 configs/`: Contains configuration files for different experiments.
- `📂 docs/`: Contains documentation for the framework.
- `📂 hpc/`: Contains scripts for running benchmarks on High-Performance Computing (HPC) clusters.
- `📂 src/`: Contains the source code of the framework.
  - `📂 src/networks/`: Contains different SNN architectures.
  - `📂 src/trainers/`: Contains different trainer objects that implement various training algorithms.
  - `📂 src/datasets/`: Contains code for loading and preprocessing datasets.
  - `📂 src/utils/`: Contains utility functions for training and evaluation.
- `Makefile`: Contains commands for setting up the environment and running benchmarks.
- `main.py`: The main entry point for running single experiments.
- `benchmarking.py`: The main entry point for running benchmarking experiments.
- `reproducibility.py`: Script to launch reproducibility experiments on implemented trainers.

## Main Files

### `main.py`
This file serves as the main entry point for running single experiments.

How to call it:
...

The file starts form the main, there is reading and extraction of config both from file and command line, then some helpers are created for logging and checkpoint management then we have the most crtitical lines:
```python
# Get trainer class
trainer_class = get_trainer(config.trainer.name)
```

and: 
```python
# Run training
trainable(
    config=config,
    trainer_class=trainer_class,
    logger=logger,
    checkpoint_manager=checkpoint_manager,
    start_epoch=start_epoch,
    resume_checkpoint=resume_checkpoint,
)
```

The `get_trainer` function is a simple helper function that returns the trainer class based on the name specified in the config.

The `trainable` function is the main function that runs the training loop, it takes care of the entire training process including logging, checkpointing, and evaluation.