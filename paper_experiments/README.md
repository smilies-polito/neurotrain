# Paper Experiments — Final Results

This file collects the **Final Results on HPO with 20 epochs and 15 trials** for every trainer
benchmarked in the paper. Tables are copied from `config/paper/README.md`.

Trainers whose final-run experiments have not yet been completed are explicitly marked
**⚠️ Results not yet available — to be added once experiments are run**.

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
> **Network abbreviations** — `FC` = Fully Connected · `RC` = Recurrent · `Conv` = Convolutional

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
| RC      |   🔴     |   🔴     |   🔴     |   🔴     |   🔴     |   🔴     |    🔴      |   🔴     |
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
