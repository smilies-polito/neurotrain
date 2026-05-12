# Paper Experiments — Final Results

This file collects the results reported on the paper. To showcase the capabilities of NeuroTrain the results have been obtained with an **HPO with 20 epochs and 15 trials** for every trainer
benchmarked in the paper. Some results have been obtained with a different configuration that is specified in the noted.

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
| Conv    | 0.971 🟢 | 0.783 🟢 | 0.352 🟢 | 0.559 🟢 | 0.955 🟢 | 0.784 🟢 | 0.394 🟢 |    ⚫     |

---

## EPROP

| Network |   MNIST   |  F-MNIST  | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :-------: | :-------: | :-----: | :--: | :----: | :------: | :--------: | :------: |
| FC      |     ⚫     |     ⚫     |    ⚫    |  ⚫   |   ⚫    |    ⚫     |     ⚫      |    ⚫     |
| RC      | 0.9783 🟢 | 0.8550 🟢 |    🔴    |  🔴   |   🔴    |    🔴     |     🔴      | 0.6913 🟢 |
| Conv    |     ⚫     |     ⚫     |    ⚫    |  ⚫   |   ⚫    |    ⚫     |     ⚫      |    ⚫     |

> Only RC results for MNIST, F-MNIST and SHD have been collected so far.
> Remaining dataset / architecture combinations are to be run.

---

## ESD_RTRL

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.956 🟢 | 0.864 🟢 | 0.426 🟢 | 0.732 🟢 | 0.958 🟢 | 0.731 🟢 |  0.379 🟢  | 0.510 🟢 |
| RC      |   🟡     |   🟡     |   🟡     |   🟡     |   🟡     |   🟡     |    🟡      |   🟡     |
| Conv    | 0.963 🟢 | 0.808 🟢 | 0.429 🟢 | 0.661 🟢 | 0.949 🟢 | 0.474 🟢 |  0.233 🟢  |    ⚫     |

> RC accuracy is at chance level for all datasets — the r_snn model consistently fails to learn with ESD_RTRL (all values in the range 0.06–0.13).

---

## ETLP

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.932 🟢 | 0.822 🟢 | 0.249 🟢 | 0.361 🟢 | 0.888 🟢 |   🔴     |     🔴     | 0.260 🟢 |
| RC      | 0.913 🟢 | 0.807 🟢 | 0.261 🟢 | 0.308 🟢 | 0.901 🟢 |   🔴     |  0.308 🟢  | 0.269 🟢 |
| Conv    |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |

---

## OSTL

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. | DVSCifar10 |   SHD    |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :--------: | :------: |
| FC      | 0.963 🟢 | 0.837 🟢 | 0.379 🟢 | 0.618 🟢 | 0.932 🟢 |    🔴    |     🔴     | 0.236 🟢 |
| RC      | 0.965 🟢 | 0.832 🟢 | 0.237 🟢 | 0.279 🟢 | 0.941 🟢 |    🔴    |     🔴     | 0.308 🟢 |
| Conv    |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |     ⚫      |    ⚫     |

---

## OSTTP

> ⚠️ **Results not yet available — to be added once 20-epoch / 15-trial HPO experiments are run.**
>
> Earlier HPO runs exist in `config/paper/README.md` but a dedicated 20-epoch / 15-trial sweep
> has not been completed yet.

---

## OTTT

> ⚠️ **Results not yet available — to be added once 20-epoch / 15-trial HPO experiments are run.**
>
> No experiments have been run for this trainer yet.

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
| TP      | VGG9 (TP-inspired)   | 0.750 🟡 [7] | 0.321 🟡 [8] | 0.311 🟡 [8] |   0.882 🟢   | 

1. Accuracy stayed at chance level for all 100 epochs.
2. After an initial increase in accuracy (~90%), the model collapsed to chance level for the remaining epochs.
3. Ran only on 10 epochs.
4. Good growth for first epochs (~75/80/90%) but then collapsed and stabilized to value in table.
5. Normal growth, probably needs better tuning.
6. Random at first but started learning steadily from epoch ~85. May need more epochs.
7. Good train accuracy but poor test accuracy, likely overfitting. Steady growth.
8. Good train accuracy but poor test accuracy, likely overfitting. Drop in test accuracy.