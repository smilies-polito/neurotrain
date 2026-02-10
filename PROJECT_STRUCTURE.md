# Project Structure

## Directory Structure

The project is organized into the following main directories:

```
CODE/
├── datasets/          # Contains dataset loaders for various datasets (e.g., MNIST, CIFAR10, etc.)
├── networks/          # Contains neural network architectures
├── trainers/          # Contains trainer classes for training models
├── utils/             # Utility functions for setup, parameter parsing, etc.
├── LearningAlgorithms # Contains algorithms for training and evaluation
└── main.py            # Entry point for the project
```

## Project Organization

### 1. **Datasets**
   - The `datasets/` directory contains loaders for different datasets such as MNIST, CIFAR10, FashionMNIST, etc.
   - Each dataset loader is implemented in a separate file (e.g., `mnist_loader.py`, `cifar10_loader.py`).
   - The `get_loader.py` file provides a unified interface to access these loaders.

### 2. **Networks**
   - The `networks/` directory contains the implementation of neural network architectures.
   - Example: `fc_network.py` for fully connected networks.

### 3. **Trainers**
   - The `trainers/` directory contains trainer classes that encapsulate the training logic.
   - Example: `stsf_trainer.py` for the STSF training strategy.

### 4. **Utilities**
   - The `utils/` directory contains helper functions and scripts for:
     - Device setup (`helpers.py`)
     - Parsing command-line arguments (`parameters.py`)
     - Experiment setup and storage management.

### 5. **Learning Algorithms**
   - The `LearningAlgorithms/` directory contains reusable training and evaluation algorithms.
   - Example: `train_epoch` and `evaluate` methods for model training and testing.

### 6. **Main Script**
   - The `main.py` file serves as the entry point for the project.
   - It handles:
     - Parsing arguments
     - Setting up the experiment
     - Initializing the trainer and dataset loaders
     - Running the training and evaluation loops.

## Notes
- The project is modular, with each component (datasets, networks, trainers) separated into its own directory for better organization and reusability.
- The `src/` folder acts as the core of the project, containing all the essential modules and scripts.
- The `main.py` file has been moved outside the `src/` folder for easier access and execution.

---

This structure ensures clarity, modularity, and ease of maintenance for the project.