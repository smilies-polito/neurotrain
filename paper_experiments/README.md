# Paper Experiments — Final Results

This file collects the results reported on the paper. To showcase the capabilities of NeuroTrain the results have been obtained with an **HPO with 20 epochs, batch size 256 and 15 trials** for every trainer
benchmarked in the paper. Some results have been obtained with a different configuration that is specified in the noted.
The ranges and default parameters are stored in the `configs/default` folder. Small deviations from the default configuration are noted in the table footnotes. The results are reported as the final test accuracy obtained with the best hyperparameters found in the HPO.

---

## Legend

| Symbol | Meaning                                               |
| ------ | ----------------------------------------------------- |
| 🟢     | Experiment successful                                 |
| 🟡     | Results but with problems                             |
| 🔴     | Error while running                                   |
| ⚫     | Not supported                                         |

> **Dataset groups** — Frame-based: `MNIST` `F-MNIST` `CIFAR10` `SVHN`
> · Neuromorphic: `NMNIST` `DVSGest.` `DVSCifar10` `SHD`
>
> **Network abbreviations** — `FC` = Fully Connected · `RC` = Recurrent · `Conv` = Convolutional

**Default Network Architectures**:

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 784-256-10 | 784-800-10 | 3072-1024-512-10 | 3072-1024-512-10 | 2312-512-10 | 32768-2048-11 | 32768-1024-512-10 | 700-512-20 |
| RC      | 784-256-10 | 784-256-10 | 3072-512-256-10 | 3072-512-256-10 | 2312-256-10 | 32768-1024-11 | 32768-512-10 | 700-512-20 |

Conv is always: 12C5-MP2-32C5-MP2-FC

---

## BPTT

All the results here have been obtained with the default campaign.

| Network |   MNIST  | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :-: |
| FC      | 0.978 🟢 | 0.837 🟢 | 0.359 🟢 | 0.536 🟢 | 0.968 🟢 | 0.689 🟢 |  0.337 🟢  | 0.496 🟢 |
| RC      | 0.969 🟢 | 0.828 🟢 | 0.343 🟢 | 0.561 🟢 | 0.958 🟢 | 0.712 🟢 |  0.340 🟢  | 0.696 🟢 |
| Conv    | 0.989 🟢 | 0.829 🟢 | 0.449 🟢 | 0.838 🟢 | 0.982 🟢 | 0.636 🟢 |  0.379 🟢  |    ⚫    |

---

## DECOLLE

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.957 🟢 | 0.801 🟢 | 0.399 🟢 | 0.749 🟢 | 0.931 🟢 | 0.708 🟢 |  0.361 🟢  | 0.375 🟢 |
| RC      |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |
| Conv    | 0.971 🟢 | 0.783 🟢 | 0.352 🟢 | 0.559 🟢 | 0.955 🟢 | 0.784 🟢 [1] | 0.394 🟢 [1] |    ⚫     |

1. Results obtained with a lower batch size of 64 due to GPU memory constraints.

---

## EPROP

| Network |   MNIST   |  F-MNIST  | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :-------: | :-------: | :-----: | :--: | :----: | :------: | :--------: | :------: |
| FC      |     ⚫     |     ⚫     |    ⚫    |  ⚫   |   ⚫    |    ⚫     |     ⚫      |    ⚫     |
| RC      | 0.9783 🟢 | 0.8550 🟢 | 0.425 🟢 [1] |0.596 🟢 [1] | 0.960 🟢 [1] | 0.667 🟢 [2] | 0.253 🟢 [2]| 0.6913 🟢 |
| Conv    |     ⚫     |     ⚫     |    ⚫    |  ⚫   |   ⚫    |    ⚫     |     ⚫      |    ⚫     |

1. For a fair comparison these results have been obtained with 512 hidden units instead of 256 ran in a different campaign since the framework doesn't support different default configs per trainer in a benchmarking campaign but it is planned to be added in the near future.
2. Results obtained with a lower batch size of 32 due to GPU memory constraints.

---

## ESD_RTRL

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.956 🟢 | 0.864 🟢 | 0.426 🟢 | 0.732 🟢 | 0.958 🟢 | 0.731 🟢 |  0.379 🟢  | 0.510 🟢 |
| RC      | 0.827 🟢 | 0.678 🟢 | 0.311 🟢 | 0.253 🟢 | 0.801* 🟢 | 0.708 🟢 [1] | 0.191 🟢 [1] | 0.451 🟢 |
| Conv    | 0.963 🟢 | 0.808 🟢 | 0.569 🟢 | 0.661 🟢 | 0.949 🟢 | 0.474 🟢 |  0.233 🟢  |    ⚫     |

1. Batch size of 32.

- NMNIST on HPC `paper_esd-rtrl_r_pr-1716677`

---

## ETLP

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.932 🟢 | 0.822 🟢 | 0.249 🟢 | 0.361 🟢 | 0.888 🟢 | 0.636 🟢 [1] | 0.264 🟢 [1] | 0.260 🟢 |
| RC      | 0.913 🟢 | 0.807 🟢 | 0.261 🟢 | 0.308 🟢 | 0.901 🟢 | 0.689 🟢 [1] | 0.308 🟢     | 0.269 🟢 |
| Conv    |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |

1. Results obtained with a lower batch size of 64 due to GPU memory constraints.

---

## OSTL

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.963 🟢 | 0.837 🟢 | 0.379 🟢 | 0.618 🟢 | 0.932 🟢 | 0.712 🟢 [1] | 0.324 🟢 [1] | 0.236 🟢 |
| RC      | 0.965 🟢 | 0.832 🟢 | 0.237 🟢 | 0.279 🟢 | 0.941 🟢 | 0.712 🟢 [1] | 0.323 🟢 [1] | 0.308 🟢 |
| Conv    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |     ⚫     |    ⚫    |

1. Results obtained with a lower batch size of 32 due to GPU memory constraints.

---

## OSTTP

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.925 🟢 | 0.823 🟢 | 0.315* 🟢 | 0.300 🟢 | 0.910 🟢 | 0.693 🟢 [1] | 0.253 🟢 [1] | 0.280 🟢 |
| RC      | 0.921 🟢 | 0.810 🟢 | 0.215 🟢 | 0.217 🟢 | 0.918 🟢 | 0.655 🟢 [1] | 0.100 🟡 | 0.057 🟡 |
| Conv    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |    ⚫    |     ⚫     |    ⚫    |

1. Results obtained with a lower batch size of 16 due to GPU memory constraints.

---

## OTTT

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.932 🟢 | 0.807 🟢 | 0.349 🟢 | 0.609 🟢 | 0.870 🟢 | 0.572 🟢 |  0.297 🟢  | 0.264 🟢 |
| RC      | 0.930 🟢 | 0.810 🟢 | 0.350 🟢 | 0.583 🟢 | 0.882 🟢 | 0.606 🟢 |  0.357 🟢  | 0.412 🟢 |
| Conv    | 0.954 🟢 | 0.738 🟢 | 0.492 🟢 | 0.795 🟢 | 0.802 🟢 | 0.576 🟢 |  0.357 🟢  |    ⚫     |

---

## STSF

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.933 🟢 | 0.800 🟢 | 0.277 🟢 | 0.276 🟢 | 0.904 🟢 | 0.708 🟢 |  0.199 🟢  | 0.221 🟢 |
| RC      |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |
| Conv    |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |

---

## TP

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.975 🟢 | 0.862 🟢 | 0.339 🟢 | 0.457 🟢 | 0.963 🟢 | 0.686 🟢 |  0.307 🟢  | 0.498 🟢 |
| RC      | 0.974 🟢 | 0.859 🟢 | 0.348 🟢 | 0.562 🟢 | 0.962 🟢 | 0.701 🟢 |  0.338 🟢  | 0.576 🟢 |
| Conv    | 0.982 🟢 | 0.846 🟢 | 0.547 🟢 | 0.785 🟢 | 0.972 🟢 | 0.629 🟢 |  0.333 🟢  |    ⚫     |

# Results on VGG9 networks

The framework also supports VGG9 architectures. They are tested separately from the main experiments given the increased training time and complexity. In this section we report the results obtained on the trainer that operates on VGG nets (BPTT, OTTT, TP) on the more complex datasets (CIFAR10, SVHN, DVSCifar10, DVSGesture) using two VGG9 variants.

## VGG9 Architecture Variants

Two distinct VGG9 architectures are used in these experiments, tailored for different training paradigms. Both share the same base convolutional structure with 8 layers (64, 128, 256, 256, 512, 512, 512, 512 channels).

### VGG9 (OTTT-inspired)

- **Head**: Global linear classifier (no temporal integration)
- **Pooling**: Average pooling 2x2 after blocks 2 and 4
- **Architecture parameters**: Sigmoid surrogate function, Conv gain set to 1.0, uses scaling after LIF

### VGG9 (TP-inspired)

- **Head**: Leaky integrator with temporal pooling integration (2x2 spatial, leak 1.0)
- **Pooling**: MaxPool 2x2 after blocks 2, 4, 6; AdaptiveAvgPool 2x2 after block 8
- **Fixed architecture parameters**: Arctangent surrogate function, Conv gain set to 1.8, no scaling after LIF

| Trainer |        Network       |   CIFAR10    |     SVHN     |   DVSCifar10  |  DVSGesture  |
| ------- | -------------------- | :----------: | :----------: | :-----------: | :----------: |
| BPTT    | VGG9 (OTTT-inspired) | 0.100 🟡 [1] | 0.067 🟡 [2] | 0.085 🟡 [1] | 0.091 🟡 [1] |
| BPTT    | VGG9 (TP-inspired)   |   0.910 🟢   |   0.960 🟢   | 0.626 🟡 [3] |   0.894 🟢   |
| OTTT    | VGG9 (OTTT-inspired) | 0.525 🟡 [4] | 0.485 🟡 [5] | 0.100 🟡 [1] | 0.091 🟡 [1] |
| OTTT    | VGG9 (TP-inspired)   | 0.666 🟡 [4] | 0.224 🟡 [4] | 0.587 🟡 [6] | 0.091 🟡 [1] |
| TP      | VGG9 (OTTT-inspired) | 0.534 🟡 [7] | 0.311 🟡 [8] | 0.375 🟡 [8] |   0.920 🟢   |
| TP      | VGG9 (TP-inspired)   | 0.750 🟡 [7] |   0.944 🟢   | 0.311 🟡 [8] |   0.882 🟢   | 

1. Accuracy stayed at chance level for all 100 epochs.
2. After an initial increase in accuracy (~90%), the model collapsed to chance level for the remaining epochs.
3. Ran only on 10 epochs.
4. Good growth for first epochs (~75/80/90%) but then collapsed and stabilized to value in table.
5. Normal growth, probably needs better tuning.
6. Random at first but started learning steadily from epoch ~85. May need more epochs.
7. Good train accuracy but poor test accuracy, likely overfitting. Steady growth.
8. Good train accuracy but poor test accuracy, likely overfitting. Drop in test accuracy.