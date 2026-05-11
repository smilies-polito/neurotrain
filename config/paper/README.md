## Paper Config Files

In this folder are collected the configuration files used to get the results shown in the paper.
Each file contains a subset of the total values so that they can be run more efficiently.

## How we obtained them

To obtain these configurations we performed a hyperparameter search using Optuna. Here we report
the accuracy obtained by the selected configuration for each experiment on the limited number of
epochs used for exploration.

---

## Legend

| Symbol | Meaning                                               |
| ------ | ----------------------------------------------------- |
| 🟢     | Experiment successful                                 |
| 🟡     | We have results but with problems                     |
| 🔴     | Error while running                                   |
| ⚫      | Not supported — incompatible algorithm / architecture |
| 🔵     | Not yet run                                           |

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

HPO results:
| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.971 🟢 | 0.828 🟢 | 0.362 🟢 | 0.527 🟢 | 0.962 🟢 | 0.674 🟢 | 0.325 🟢 | 🔵 |
| RC      | 0.965 🟢 | 0.821 🟢 | 0.345 🟢 | 0.452 🟢 | 0.954 🟢 | 0.705 🟢 | 0.312 🟢 | 🔵 |
| Conv    | 0.987 🟢 | 0.808 🟢 | 0.425 🟢 | 0.818 🟢 | 0.981 🟢 | 0.595 🟢 | 0.294 🟢 | ⚫ |

Final Results:
| Network |   MNIST  | F-MNIST  | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- |   :---:  | :-----:  | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.976 🟢 | 0.851 🟢 | 0.409 🟢 | 0.724 🟢 | 0.974 🟢 | 0.697 🟢 | 0.362 🟢 | 🔵 |
| RC      | 0.974 🟢 | 0.834 🟢 | 0.380 🟢 | 0.625 🟢 | 0.956 🟢 | 0.701 🟡 | 0.360 🟢 | 🔵 |
| Conv    | 0.987 🟡 | 0.829 🟢 | 0.518 🟢 | 0.845 🟢 | 0.977 🟡 | 0.652 🟢 | 0.328 🟢 | ⚫ |

Final Results on HPO with 20 epochs and 15 trials:
| Network |   MNIST  | F-MNIST  | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- |   :---:  | :-----:  | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.978 🟢 | 0.837 🟢 | 0.359 🟢 | 0.536 🟢 | 0.968 🟢 | 0.689 🟢 | 0.337 🟢 | 0.496 🟢 |
| RC      | 0.969 🟢 | 0.828 🟢 | 0.343 🟢 | 0.561 🟢 | 0.958 🟢 | 0.712 🟢 | 0.340 🟢 | 0.696 🟢 |
| Conv    | 0.989 🟢 | 0.829 🟢 | 0.449 🟢 | 0.838 🟢 | 0.982 🟢 | 0.636 🟢 | 0.379 🟢 | ⚫ |
---

### DECOLLE
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   |  NMNIST  | DVSGest. |  DVSCifar10  | SHD |
| ------- | :------: | :------: | :------: | :------: | :------: | :------: | :----------: | :-: |
| FC      | 0.913 🟢 | 0.724 🟢 | 0.381 🟢 | 0.744 🟢 | 0.919 🟢 | 0.739 🟢 |   0.360 🟢   | 🔵  |
| RC      |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |    ⚫     |      ⚫       |  ⚫  |
| Conv    | 0.890 🟢 | 0.649 🟢 | 0.366 🟢 | 0.584 🟢 | 0.896 🟢 | 0.663 🟢 | 0.403 🟡 [1] |  ⚫  |
##### Comments
 1. Got [[DECOLLE DVSCIFAR10 aedat error|this]] error. Very strange since it appeared after some time.

---

### EPROP
Commit: `968f810153ca27300c9347a7be933628302bf732`

| Network |  MNIST   | F-MNIST  | CIFAR10 |  SVHN  |  NMNIST  | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :-----: | :----: | :------: | :------: | :--------: | :-: |
| FC      |    ⚫     |    ⚫     |    ⚫    |   ⚫    |    ⚫     |    ⚫     |     ⚫      |  ⚫  |
| RC      | 0.967 🟢 | 0.828 🟢 | 🔴 [1]  | 🔴 [1] | 0.953 🟢 |  🔴 [2]  |   🔴 [2]   | 🔵  |
| Conv    |    ⚫     |    ⚫     |    ⚫    |   ⚫    |    ⚫     |    ⚫     |     ⚫      |  ⚫  |
1. It was trying to use a multi-layer RSNN, fixed already and waiting to be run.
2. CUDA out of memory, need to reduce batch size.

---

### ESD_RTRL
Commit: `968f810153ca27300c9347a7be933628302bf732`

HPO results:
| Network |  MNIST   | F-MNIST  | CIFAR10  |     SVHN     | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :----------: | :----: | :------: | :--------: | :-: |
| FC      | 0.969 🟢 | 0.847 🟢 | 0.400 🟢 |   0.688 🟢   |   🔵   | 0.697 🟢 |  0.334 🟢  | 🔵  |
| RC      | 0.879 🟢 | 0.735 🟢 | 0.277 🟢 | 0.097 🟡 [1] |   🔵   | 0.553 🟢 |     🔵     | 🔵  |
| Conv    | 0.986 🟢 | 0.829 🟢 | 0.470 🟢 |      🔵      |   🔵   | 0.583 🟢 |     🔵     |  ⚫  |
1. Accuracy stayed at random, better to run this again since it is very strange.

Final results:
| Network |  MNIST   | F-MNIST  | CIFAR10  |     SVHN     | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :----------: | :----: | :------: | :--------: | :-: |
| FC      | 0.974 🟢 | 0.847 🟡 | 0.449 🟢 |   0.744 🟢   |       | 0.723 🟢 |  0.351 🟢  |     |
| RC      | 0.700 🔴 | 0.315 🔴 | 0.123 🔴 |              |       | 0.697 🟢 |            |     |
| Conv    | 0.854 🔴 | 0.828 🟡 | 0.436 🟡 |              |       | 0.390 🔴 |            |      |
---

### ETLP
Commit: `968f810153ca27300c9347a7be933628302bf732`

HPO results:
| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :------: | :----: | :------: | :--------: | :-: |
| FC      | 0.925 🟢 | 0.792 🟢 | 0.301 🟢 | 0.221 🟢 | 0.824 🟢 |  🔴 [1]  | 0.140 🟡 [2] | 🔵  |
| RC      | 0.915 🟢 | 0.809 🟢 | 0.301 🟢 | 0.337 🟢 |   🔵   | 0.670 🟢 |     🔵     | 🔵  |
| Conv    |    ⚫    |    ⚫     |    ⚫     |    ⚫     |   ⚫    |    ⚫     |     ⚫      |  ⚫  |
1. It immediately failed for fully-connected because of out of memory for CUDA.
2. Very low but I think this could be the best we can get with this.

Final results:
| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :------: | :----: | :------: | :--------: | :-: |
| FC      | 0.945 🟢 | 0.830 🟢 | 0.308 🟢 | 0.481 🟢 | 0.890 🟢 |   |  |   |
| RC      | 0.914 🟡 | 0.753 🟡 | 0.331 🟢 | 0.177 🔴 |      |  |         |   |

Final Results on HPO with 20 epochs and 15 trials:
| Network |  MNIST   | F-MNIST  | CIFAR10  |   SVHN   | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :------: | :------: | :------: | :------: | :----: | :------: | :--------: | :-: |
| FC      | 0.932 🟢 | 0.822 🟢 | 0.249 🟢 | 0.361 🟢 | 0.888 🟢 |    🔴     |     🔴      | 0.260 🟢 |
| RC      | 0.913 🟢 | 0.807 🟢 | 0.261 🟢 | 0.308 🟢 | 0.901 🟢 |    🔴     |   0.308 🟢   | 0.269 🟢 |
| Conv    |    ⚫    |    ⚫     |    ⚫     |    ⚫     |   ⚫    |    ⚫     |     ⚫      |  ⚫  |

---

### OSTL

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.967 🟢 | 0.838 🟢 | 0.395 🟢 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| RC      | 0.967 🟢 |   🔵   |    0.396   | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    |   ⚫     |   ⚫   |    ⚫   | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

---

### OSTTP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      |   0.924  |   0.816   |    0.313   | 0.385 | 0.915 | 🔵 | 🔵 | 🔵 |
| RC      |   0.917  |   0.807   |    0.339   | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Conv    |   ⚫  |   ⚫   |    ⚫   | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

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
| FC      | 0.932 🟢 | 0.806 🟢 | 0.254 🟢 | 0.290 🟢 | 0.901 🟢 | 0.659 🟢 | 0.191 🟢 | 🔵 |
| RC      | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| Conv    | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

Final Results on HPO with 20 epochs and 15 trials:
| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.933 🟢 | 0.800 🟢 | 0.277 🟢 | 0.276 🟢 | 0.904 🟢 | 0.708 🟢 | 0.199 🟢 | 0.221 🟢 |
| RC      | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| Conv    | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

---

### TP

Final Results on HPO with 20 epochs and 15 trials:
| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC      | 0.975 🟢 | 0.862 🟢 | 0.339 🟢 | 0.457 🟢 | 0.963 🟢 | 0.686 🟢 | 0.307 🟢 | 0.498 🟢 |
| RC      | 0.974 🟢 | 0.859 🟢 | 0.348 🟢 | 0.562 🟢 | 0.962 🟢 | 0.701 🟢 | 0.338 🟢 | 0.576 🟢 |
| Conv    | 0.982 🟢 | 0.846 🟢 | 0.547 🟢 | 0.785 🟢 | 0.972 🟢 | 0.629 🟢 | 0.333 🟢 | ⚫ |

---

## VGG9 Paper Experiments — Individual Configs (`config/paper/vgg9/`)

Each row below has its own config file and sbatch script so they can be submitted in parallel.
Launch all 15 HPC jobs at once with `make paper-vgg9-all-hpc-split`.

| Trainer | Net     | Dataset     | Acc (200 ep) | Config file                                  |
| ------- | :-----: | :---------: | :----------: | -------------------------------------------- |
| BPTT    | TPnet   | CIFAR10     | 0.855 🟢     | `config/paper/vgg9/bptt_tpnet_cifar10.yaml`   |
| BPTT    | TPnet   | SVHN        | 0.946 🟢     | `config/paper/vgg9/bptt_tpnet_svhn.yaml`       |
| BPTT    | TPnet   | DVSGesture  | 0.765 🟡     | `config/paper/vgg9/bptt_tpnet_dvsgesture.yaml` |
| BPTT    | OTTTnet | SVHN        | 0.913 🟢     | `config/paper/vgg9/bptt_otttnet_svhn.yaml`     |
| TP      | TPnet   | CIFAR10     | 0.723 🟢     | `config/paper/vgg9/tp_tpnet_cifar10.yaml`      |
| TP      | TPnet   | SVHN        | 0.921 🟡     | `config/paper/vgg9/tp_tpnet_svhn.yaml`         |
| TP      | TPnet   | DVSGesture  | 0.666 🟡     | `config/paper/vgg9/tp_tpnet_dvsgesture.yaml`   |
| TP      | TPnet   | DVSCifar10  | 0.506 🟢     | `config/paper/vgg9/tp_tpnet_dvscifar10.yaml`   |
| TP      | OTTTnet | CIFAR10     | 0.729 🟢     | `config/paper/vgg9/tp_otttnet_cifar10.yaml`    |
| TP      | OTTTnet | SVHN        | 0.924 🟡     | `config/paper/vgg9/tp_otttnet_svhn.yaml`       |
| TP      | OTTTnet | DVSGesture  | 0.610 🟡     | `config/paper/vgg9/tp_otttnet_dvsgesture.yaml` |
| TP      | OTTTnet | DVSCifar10  | 0.451 🟢     | `config/paper/vgg9/tp_otttnet_dvscifar10.yaml` |
| OTTT    | TPnet   | CIFAR10     | 0.774 🟢     | `config/paper/vgg9/ottt_tpnet_cifar10.yaml`    |
| OTTT    | OTTTnet | CIFAR10     | 0.717 🟢     | `config/paper/vgg9/ottt_otttnet_cifar10.yaml`  |
| OTTT    | OTTTnet | SVHN        | 0.915 🟢     | `config/paper/vgg9/ottt_otttnet_svhn.yaml`     |

> Accuracy values are from the HPO exploration phase (10 epochs). Full 200-epoch results will replace these once available.

---

## VGG9 Full Matrix (`make vgg9-matrix`)

All 24 combinations from the Makefile `vgg9-matrix` target.
Config files live in `config/vgg9/`.

> **Network variants:**
> - **TPnet** — TP-style head: leaky-integrator readout, atan surrogate, conv_gain = 1.8
> - **OTTTnet** — OTTT-style head: global linear readout, sigmoid surrogate, scale_after_lif = 2.74

| Trainer | Net variant |   CIFAR10   |   SVHN   |  DVSGesture  |  DVSCifar10  |
| ------- | :---------: | :---------: | :------: | :----------: | :----------: |
| TP      |    TPnet    |  0.723 🟢   | 0.921 🟡 | 0.666 🟡 [2] |   0.506 🟢   |
| TP      |   OTTTnet   |  0.729 🟢   | 0.924 🟡 | 0.610 🟡 [2] |   0.451 🟢   |
| OTTT    |    TPnet    |   0.774    |  🔴 [4]  |    🔴 [4]    |    🔴 [4]    |
| OTTT    |   OTTTnet   |   0.717    |  0.915  |    🔴 [4]    |    🔴 [4]    |
| BPTT    |    TPnet    |  0.855 🟢   | 0.946 🟢 | 0.765 🟡 [3] | 0.639 🟡 [3] |
| BPTT    |   OTTTnet   | 0.10 🟡 [1] | 0.913 🟢 | 0.091 🔴 [2] | 0.085 🟡 [1] |
1. In these trial the accuracy always ended up at random of worse but some trial got a growing accuracy in the first epoch and then completely died.
2. Died almost immediately.
3. Extreme overfitting.
4. Error on OTTT trainer that I thought I had fixed locally but apparently not.
#### Comments on VGG Trials
###### BPTT on OTTT-net with CIFAR10
Here only one trial got some kinda good results that died completely for some reason.

Output of terminal:
```
2026-04-26 09:09:53,393 [INFO] Using device: cuda
2026-04-26 09:09:53,393 [INFO] Loading dataset: cifar10 (T=6, batch=128)
2026-04-26 09:09:54,572 [INFO] Building network: vgg9
2026-04-26 09:09:54,626 [INFO] Building trainer: bptt
2026-04-26 09:09:54,626 [INFO] Training for 10 epochs...
2026-04-26 09:11:46,760 [INFO] Epoch 1/10 — loss: 1.5965  train_acc: 0.4155  test_acc: 0.5141
2026-04-26 09:13:38,696 [INFO] Epoch 2/10 — loss: 1.2689  train_acc: 0.5546  test_acc: 0.5930
2026-04-26 09:15:30,699 [INFO] Epoch 3/10 — loss: 1.1450  train_acc: 0.6002  test_acc: 0.6024
2026-04-26 09:17:22,644 [INFO] Epoch 4/10 — loss: 1.0567  train_acc: 0.6367  test_acc: 0.6495
2026-04-26 09:19:14,577 [INFO] Epoch 5/10 — loss: 0.9861  train_acc: 0.6627  test_acc: 0.6672
2026-04-26 09:21:06,626 [INFO] Epoch 6/10 — loss: 0.9327  train_acc: 0.6818  test_acc: 0.6924
2026-04-26 09:22:58,625 [INFO] Epoch 7/10 — loss: 0.9362  train_acc: 0.6808  test_acc: 0.1000
2026-04-26 09:24:50,238 [INFO] Epoch 8/10 — loss: 2.3026  train_acc: 0.0988  test_acc: 0.1000
2026-04-26 09:26:41,890 [INFO] Epoch 9/10 — loss: 2.3026  train_acc: 0.0974  test_acc: 0.1000
2026-04-26 09:28:33,488 [INFO] Epoch 10/10 — loss: 2.3026  train_acc: 0.0991  test_acc: 0.1000
2026-04-26 09:28:33,489 [INFO] Training done in 1118.9 s. Final test accuracy: 0.1000
```

Configuration:
```yaml
name: bptt_otttnet_cifar10_opt_t1
opt: false
trainer:
  name: bptt
  supported_net_types:
  - fc
  - rec
  - conv
  lr: 4.207988669606632e-05
  loss_type: ce_rate
  grad_clip: null
  use_optimizer: true
model:
  name: vgg9
  net_type: vgg9
  in_channels: 3
  num_classes: 10
  input_shape:
  - 3
  - 32
  - 32
  head_type: global_linear
  use_tp_pool: false
  channels:
  - 64
  - 128
  - 256
  - 256
  - 512
  - 512
  - 512
  - 512
  beta: 0.4076362190319798
  threshold: 0.5580836121681995
  conv_gain: 1.0
  scale_after_lif:
        value: 2.74
        type: float
        min: 1.5
        max: 3.5
  surrogate_kind: sigmoid
  surrogate_slope: 4.0
  pool_after_blocks:
  - 2
  - 4
  algorithm_name: bptt
dataset:
  name: cifar10
  supported_net_types:
  - fc
  - rec
  - conv
  - vgg9
  T: 6
  seed: 867
  direct_coding: true
  pin_memory: false
  download: true
  data_root: null
  num_workers: 4
runtime:
  epochs: 10
  device: cuda
  seed: 42
  log_level: INFO
  neurobench: true
  batch_size: 128
  progress: false
optuna: {}
```
###### BPTT on OTTT-net with DVSCIFAR10
Again we got trials where the accuracy grows and then drops.

Terminal output:
```
2026-04-26 17:18:37,497 [INFO] Using device: cuda
2026-04-26 17:18:37,497 [INFO] Loading dataset: dvscifar10 (T=10, batch=32)
2026-04-26 17:18:37,542 [INFO] Building network: vgg9
2026-04-26 17:18:37,621 [INFO] Building trainer: bptt
2026-04-26 17:18:37,621 [INFO] Training for 10 epochs...
2026-04-26 17:25:17,360 [INFO] Epoch 1/10 — loss: 2.0481  train_acc: 0.2537  test_acc: 0.3130
2026-04-26 17:31:57,040 [INFO] Epoch 2/10 — loss: 1.8655  train_acc: 0.3462  test_acc: 0.3580
2026-04-26 17:38:36,929 [INFO] Epoch 3/10 — loss: 1.7715  train_acc: 0.3908  test_acc: 0.4230
2026-04-26 17:45:17,096 [INFO] Epoch 4/10 — loss: 1.6936  train_acc: 0.4235  test_acc: 0.4330
2026-04-26 17:51:57,244 [INFO] Epoch 5/10 — loss: 1.6325  train_acc: 0.4511  test_acc: 0.4460
2026-04-26 17:58:37,520 [INFO] Epoch 6/10 — loss: 1.6160  train_acc: 0.4504  test_acc: 0.0850
2026-04-26 18:05:15,558 [INFO] Epoch 7/10 — loss: 2.3026  train_acc: 0.1016  test_acc: 0.0850
2026-04-26 18:11:53,677 [INFO] Epoch 8/10 — loss: 2.3026  train_acc: 0.1015  test_acc: 0.0850
2026-04-26 18:18:31,791 [INFO] Epoch 9/10 — loss: 2.3026  train_acc: 0.1015  test_acc: 0.0850
2026-04-26 18:25:09,822 [INFO] Epoch 10/10 — loss: 2.3026  train_acc: 0.1015  test_acc: 0.0850
2026-04-26 18:25:09,822 [INFO] Training done in 3992.2 s. Final test accuracy: 0.0850
```

Configuration:
```yaml
name: bptt_otttnet_dvscifar10_opt_t1
opt: false
trainer:
  name: bptt
  supported_net_types:
  - fc
  - rec
  - conv
  lr: 4.207988669606632e-05
  loss_type: ce_rate
  grad_clip: null
  use_optimizer: true
model:
  name: vgg9
  net_type: vgg9
  in_channels: 2
  num_classes: 10
  input_shape:
  - 2
  - 128
  - 128
  head_type: global_linear
  use_tp_pool: false
  channels:
  - 64
  - 128
  - 256
  - 256
  - 512
  - 512
  - 512
  - 512
  beta: 0.4076362190319798
  threshold: 0.5580836121681995
  conv_gain: 1.0
  scale_after_lif:
        value: 2.74
        type: float
        min: 1.5
        max: 3.5
  surrogate_kind: sigmoid
  surrogate_slope: 4.0
  pool_after_blocks:
  - 2
  - 4
  - 6
  algorithm_name: bptt
dataset:
  name: dvscifar10
  supported_net_types:
  - fc
  - rec
  - conv
  - vgg9
  T: 10
  seed: 867
  pin_memory: false
  num_workers: 4
  train_fraction: 0.9
  data_root: null
  download: true
  use_cache: true
runtime:
  epochs: 10
  device: cuda
  seed: 42
  log_level: INFO
  neurobench: true
  batch_size: 32
  progress: false
optuna: {}
```