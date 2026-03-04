# Test programs

In this directory there are atomic test programs that are used to test the functionality of the system. These programs are designed to be simple and focused on specific features or components of the system. They can be used for unit testing, integration testing, or performance testing.

## Tests for trainer validation

Each test consists of a simple python script that launches: **one trainer on one dataset with one networks**. It can be integrated with Optuna to perform hyperparameter optimization.
The naming convention for these files is: `[trainer]_[dataset]_[network].py`.