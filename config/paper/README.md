## Paper Config Files

In this folder are collected the configuration files used to get the results shown in the paper.
Each file contains a subset of the total values so that they can be run more efficiently.

## How we obtained them

To obtain these configurations we performed a hyperparameter search using Optuna. Here we report
the accuracy obtained by the selected configuration for each experiment on the limited number of
epochs used for exploration.

---

## Legend

| Symbol | Meaning |
| ------ | ------- |
| 🟢 | Experiment successful |
| 🔴 | Error while running |
| ⚫ | Not supported — incompatible algorithm / architecture |
| 🔵 | Not yet run |

> **Dataset groups** — Frame-based: `MNIST` `F-MNIST` `CIFAR10` `SVHN`
> · Neuromorphic: `NMNIST` `DVSGest.` `DVSCifar10` `SHD`
>
> `SHD` = Spiking Heidelberg Digits (700-channel audio spike trains).
>
> **Network abbreviations** — `FC` = Fully Connected · `RC` = Recurrent · `Conv` = Convolutional

---

## Results by Trainer

### BPTT
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.971 🟢 | 0.828 🟢 | 0.362 🟢 | 0.527 🟢 | 0.962 🟢 | 0.674 🟢 | 0.325 🟢 | 🔵 |
| RC      | 0.965 🟢 | 0.821 🟢 | 0.345 🟢 | 0.452 🟢 | 0.954 🟢 | 0.705 🟢 | 0.312 🟢 | 🔵 |
| Conv    | 0.987 🟢 | 0.808 🟢 | 0.425 🟢 | 0.818 🟢 | 0.981 🟢 | 0.595 🟢 | 0.294 🟢 | ⚫ |

---

### DECOLLE
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.913 🟢 | 0.724 🟢 | 0.381 🟢 | 0.744 🟢 | 0.919 🟢 | 0.739 🟢 | 0.360 🟢 | 🔵 |
| RC      | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| Conv    | 0.890 🟢 | 0.649 🟢 | 0.366 🟢 | 0.584 🟢 | 0.896 🟢 | 0.663 🟢 | 🔴 | ⚫ |

---

### EPROP
Commit: `968f810153ca27300c9347a7be933628302bf732`

> e-prop is designed for recurrent networks; FC and Conv are not applicable.

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| RC      | 0.967 🟢 | 0.828 🟢 | 🔴 | 🔴 | 0.953 🟢 | 🔴 | 🔴 | 🔵 |
| Conv    | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

---

### ESD_RTRL
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.969 🟢 | 0.847 🟢 | 0.400 🟢 | 0.688 🟢 | 🔵 | 0.697 🟢 | 🔵 | 🔵 |
| RC      | 0.879 🟢 | 0.735 🟢 | 0.277 🟢 | 🔴 | 🔵 | 0.553 🟢 | 🔵 | 🔵 |
| Conv    | 0.986 🟢 | 🔴 | 🔴 | 🔴 | 🔵 | 🔴 | 🔵 | ⚫ |

---

### ETLP
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.925 🟢 | 0.792 🟢 | 0.301 🟢 | 0.221 🟢 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 0.915 🟢 | 0.809 🟢 | 0.301 🟢 | 🔴 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

### OSTL

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

### OSTTP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

### OTTT

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

### STSF

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

### TP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | ⚫ |

---

## VGG9 Full Matrix (`make vgg9-matrix`)

All 24 combinations from the Makefile `vgg9-matrix` target.
Config files live in `config/vgg9/`.

> **Network variants:**
> - **TPnet** — TP-style head: leaky-integrator readout, atan surrogate, conv_gain = 1.8
> - **OTTTnet** — OTTT-style head: global linear readout, sigmoid surrogate, scale_after_lif = 2.74

| Trainer | Net variant | CIFAR10 | SVHN | DVSGesture | DVSCifar10 |
| ------- | :---------: | :-----: | :--: | :--------: | :--------: |
| TP      | TPnet       | 🔵 | 🔵 | 🔵 | 🔵 |
| TP      | OTTTnet     | 🔵 | 🔵 | 🔵 | 🔵 |
| OTTT    | TPnet       | 🔵 | 🔵 | 🔵 | 🔵 |
| OTTT    | OTTTnet     | 🔵 | 🔵 | 🔵 | 🔵 |
| BPTT    | TPnet       | 🔵 | 🔵 | 🔵 | 🔵 |
| BPTT    | OTTTnet     | 🔵 | 🔵 | 🔵 | 🔵 |
