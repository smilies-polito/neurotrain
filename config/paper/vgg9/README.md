# VGG9 SNN Training Benchmarks

This document contains comprehensive results and analysis for VGG9 models trained with different learning algorithms (BPTT, OTTT, TP) and architectures (OTTTNet, TPNet) on various datasets.

---

## Experiment Status

| Algorithm | Network  | Dataset      | Config file                         | State              |
|-----------|----------|--------------|-------------------------------------|--------------------|
| BPTT      | OTTTNet  | CIFAR-10     | `bptt_otttnet_cifar10.yaml`         | 🔵 DEFAULT         |
| BPTT      | OTTTNet  | DVS-CIFAR10  | `bptt_otttnet_dvscifar10.yaml`      | 🔵 DEFAULT         |
| BPTT      | OTTTNet  | DVS-Gesture  | `bptt_otttnet_dvsgesture.yaml`      | 🔵 DEFAULT         |
| BPTT      | OTTTNet  | SVHN         | `bptt_otttnet_svhn.yaml`            | 🟡 HPO RESULT      |
| BPTT      | TPNet    | CIFAR-10     | `bptt_tpnet_cifar10.yaml`           | 🟡 HPO RESULT      |
| BPTT      | TPNet    | DVS-CIFAR10  | `bptt_tpnet_dvscifar10.yaml`        | 🟡 HPO RESULT      |
| BPTT      | TPNet    | DVS-Gesture  | `bptt_tpnet_dvsgesture.yaml`        | 🟡 HPO RESULT      |
| BPTT      | TPNet    | SVHN         | `bptt_tpnet_svhn.yaml`              | 🟡 HPO RESULT      |
| OTTT      | OTTTNet  | CIFAR-10     | `ottt_otttnet_cifar10.yaml`         | 🟡 HPO RESULT      |
| OTTT      | OTTTNet  | DVS-CIFAR10  | `ottt_otttnet_dvscifar10.yaml`      | 🔵 DEFAULT         |
| OTTT      | OTTTNet  | DVS-Gesture  | `ottt_otttnet_dvsgesture.yaml`      | 🔵 DEFAULT         |
| OTTT      | OTTTNet  | SVHN         | `ottt_otttnet_svhn.yaml`            | 🟡 HPO RESULT      |
| OTTT      | TPNet    | CIFAR-10     | `ottt_tpnet_cifar10.yaml`           | 🟡 HPO RESULT      |
| OTTT      | TPNet    | DVS-CIFAR10  | `ottt_tpnet_dvscifar10.yaml`        | 🔵 DEFAULT         |
| OTTT      | TPNet    | DVS-Gesture  | `ottt_tpnet_dvsgesture.yaml`        | 🔵 DEFAULT         |
| OTTT      | TPNet    | SVHN         | `ottt_tpnet_svhn.yaml`              | 🔵 DEFAULT         |
| TP        | OTTTNet  | CIFAR-10     | `tp_otttnet_cifar10.yaml`           | 🟡 HPO RESULT      |
| TP        | OTTTNet  | DVS-CIFAR10  | `tp_otttnet_dvscifar10.yaml`        | 🟡 HPO RESULT      |
| TP        | OTTTNet  | DVS-Gesture  | `tp_otttnet_dvsgesture.yaml`        | 🟡 HPO RESULT      |
| TP        | OTTTNet  | SVHN         | `tp_otttnet_svhn.yaml`              | 🟡 HPO RESULT      |
| TP        | TPNet    | CIFAR-10     | `tp_tpnet_cifar10.yaml`             | 🟡 HPO RESULT      |
| TP        | TPNet    | DVS-CIFAR10  | `tp_tpnet_dvscifar10.yaml`          | 🟡 HPO RESULT      |
| TP        | TPNet    | DVS-Gesture  | `tp_tpnet_dvsgesture.yaml`          | 🟡 HPO RESULT      |
| TP        | TPNet    | SVHN         | `tp_tpnet_svhn.yaml`                | 🟡 HPO RESULT      |

> **State legend** — 🔵 DEFAULT: base hyperparameters, no HPO yet · 🟡 HPO RESULT: Optuna-tuned hyperparameters · 🟢 DONE: final result accepted for the paper

---

## Table of Contents

- [BPTT](#bptt)
  - [OTTTNet](#otttnet)
  - [TPNet](#tpnet)
- [OTTT](#ottt)
  - [OTTTNet](#otttnet-1)
  - [TPNet](#tpnet-1)
- [TP](#tp)
  - [OTTTNet](#otttnet-2)
  - [TPNet](#tpnet-2)
- [Summary & Comparative Analysis](#summary--comparative-analysis)
- [Notes & Metadata](#notes--metadata)

---

## BPTT

### OTTTNet

#### CIFAR-10

**Configuration:** `bptt_otttnet_cifar10.yaml`

**Status:** [To be filled - e.g., Completed, In Progress, Pending]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-CIFAR10

**Configuration:** `bptt_otttnet_dvscifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-Gesture

**Configuration:** `bptt_otttnet_dvsgesture.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### SVHN

**Configuration:** `bptt_otttnet_svhn.yaml`

**Status:** 🟠 Problematic

**Accuracy:** -

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-09 21:41:03,084 [INFO] Epoch 1/200 — loss: 1.7153  train_acc: 0.4233  test_acc: 0.5938
2026-05-09 21:45:51,042 [INFO] Epoch 2/200 — loss: 1.0202  train_acc: 0.6929  test_acc: 0.7480
2026-05-09 21:50:38,966 [INFO] Epoch 3/200 — loss: 0.7337  train_acc: 0.7861  test_acc: 0.8159
2026-05-09 21:55:26,750 [INFO] Epoch 4/200 — loss: 0.5937  train_acc: 0.8285  test_acc: 0.8546
2026-05-09 22:00:14,574 [INFO] Epoch 5/200 — loss: 0.5098  train_acc: 0.8549  test_acc: 0.8735
2026-05-09 22:05:02,210 [INFO] Epoch 6/200 — loss: 0.4506  train_acc: 0.8745  test_acc: 0.8923
2026-05-09 22:09:49,857 [INFO] Epoch 7/200 — loss: 0.4053  train_acc: 0.8881  test_acc: 0.8995
2026-05-09 22:14:37,534 [INFO] Epoch 8/200 — loss: 0.3673  train_acc: 0.9002  test_acc: 0.9039
2026-05-09 22:19:25,083 [INFO] Epoch 9/200 — loss: 0.3388  train_acc: 0.9088  test_acc: 0.9111
2026-05-09 22:24:12,744 [INFO] Epoch 10/200 — loss: 0.3148  train_acc: 0.9163  test_acc: 0.9143
2026-05-09 22:29:00,518 [INFO] Epoch 11/200 — loss: 0.2934  train_acc: 0.9229  test_acc: 0.9193
2026-05-09 22:33:48,168 [INFO] Epoch 12/200 — loss: 0.2761  train_acc: 0.9287  test_acc: 0.9156
2026-05-09 22:38:35,876 [INFO] Epoch 13/200 — loss: 0.2588  train_acc: 0.9330  test_acc: 0.9250
2026-05-09 22:43:23,464 [INFO] Epoch 14/200 — loss: 0.2451  train_acc: 0.9383  test_acc: 0.9305
2026-05-09 22:48:11,112 [INFO] Epoch 15/200 — loss: 0.3028  train_acc: 0.9151  test_acc: 0.1959
2026-05-09 22:52:58,506 [INFO] Epoch 16/200 — loss: 2.2954  train_acc: 0.1892  test_acc: 0.1959
```

**Comments:**

The run is reaching good accuracies for us of around **93%** but is then suddendly dying reverting to random accuracy that resembles a network that is not spiking.
I tried doing lr/6 and it stayed stable for longer but then loss became NaN.

---

### TPNet

#### CIFAR-10

**Configuration:** `bptt_tpnet_cifar10.yaml`

**Status:** 🟢* Good

**Accuracy:** 91.3%

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 02:13:28,279 [INFO] Epoch 195/200 — loss: 0.0094  train_acc: 0.9967  test_acc: 0.9094
2026-05-10 02:14:53,712 [INFO] Epoch 196/200 — loss: 0.0077  train_acc: 0.9976  test_acc: 0.9084
2026-05-10 02:16:19,224 [INFO] Epoch 197/200 — loss: 0.0089  train_acc: 0.9971  test_acc: 0.9111
2026-05-10 02:17:44,705 [INFO] Epoch 198/200 — loss: 0.0101  train_acc: 0.9967  test_acc: 0.9126
2026-05-10 02:19:10,191 [INFO] Epoch 199/200 — loss: 0.0095  train_acc: 0.9967  test_acc: 0.9080
2026-05-10 02:20:35,589 [INFO] Epoch 200/200 — loss: 0.0099  train_acc: 0.9966  test_acc: 0.9130
```

**Comments:**

Accuracy is good and growing in a good way but there is a bit of overfitting that could be a symptom of other problems if we find it again.

---

#### DVS-CIFAR10

**Configuration:** `bptt_tpnet_dvscifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-Gesture

**Configuration:** `bptt_tpnet_dvsgesture.yaml`

**Status:** 🟢* Good

**Accuracy:** 90%

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 02:21:30,752 [INFO] Epoch 195/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.8939
2026-05-10 02:22:58,448 [INFO] Epoch 196/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.8902
2026-05-10 02:24:26,354 [INFO] Epoch 197/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.9015
2026-05-10 02:25:54,212 [INFO] Epoch 198/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.9015
2026-05-10 02:27:21,815 [INFO] Epoch 199/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.9015
2026-05-10 02:28:49,748 [INFO] Epoch 200/200 — loss: 0.0023  train_acc: 0.9991  test_acc: 0.8939
```

**Comments:**

Similar to other cases. Converges around 20 epochs and then keeps steady in the overfitting regime. Could be an evaluation issue.

---

#### SVHN

**Configuration:** `bptt_tpnet_svhn.yaml`

**Status:** 🟢 Done

**Accuracy:** 95%

**Campaign name:** 10-05

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

Still not finished. Accuracy good here, it is consistentely around 95% already after 10 epochs. I will wait until it finishes to put terminal output.

---

## OTTT

### OTTTNet

#### CIFAR-10

**Configuration:** `ottt_otttnet_cifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-CIFAR10

**Configuration:** `ottt_otttnet_dvscifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-Gesture

**Configuration:** `ottt_otttnet_dvsgesture.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### SVHN

**Configuration:** `ottt_otttnet_svhn.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

### TPNet

#### CIFAR-10

**Configuration:** `ottt_tpnet_cifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-CIFAR10

**Configuration:** `ottt_tpnet_dvscifar10.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### DVS-Gesture

**Configuration:** `ottt_tpnet_dvsgesture.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

#### SVHN

**Configuration:** `ottt_tpnet_svhn.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

## TP

### OTTTNet

#### CIFAR-10

**Configuration:** `tp_otttnet_cifar10.yaml`

**Status:** 🟠 Problematic

**Accuracy:** ~32% (at epoch 151)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 03:12:22,134 [INFO] Epoch 15/200 — loss: 0.7811  train_acc: 0.7743  test_acc: 0.7198
2026-05-10 03:15:13,463 [INFO] Epoch 16/200 — loss: 0.7623  train_acc: 0.7809  test_acc: 0.7492
2026-05-10 03:18:04,812 [INFO] Epoch 17/200 — loss: 0.7535  train_acc: 0.7843  test_acc: 0.6880
2026-05-10 03:20:56,149 [INFO] Epoch 18/200 — loss: 0.7264  train_acc: 0.7900  test_acc: 0.7037
2026-05-10 03:23:47,490 [INFO] Epoch 19/200 — loss: 0.7162  train_acc: 0.7966  test_acc: 0.7209
2026-05-10 03:26:38,808 [INFO] Epoch 20/200 — loss: 0.6979  train_acc: 0.8009  test_acc: 0.7240
2026-05-10 03:29:30,155 [INFO] Epoch 21/200 — loss: 0.6955  train_acc: 0.8023  test_acc: 0.6735
2026-05-10 03:32:21,536 [INFO] Epoch 22/200 — loss: 0.6612  train_acc: 0.8111  test_acc: 0.7288
2026-05-10 03:35:12,873 [INFO] Epoch 23/200 — loss: 0.6494  train_acc: 0.8160  test_acc: 0.6874
2026-05-10 03:38:04,260 [INFO] Epoch 24/200 — loss: 0.6442  train_acc: 0.8169  test_acc: 0.6610
2026-05-10 03:40:55,646 [INFO] Epoch 25/200 — loss: 0.6325  train_acc: 0.8213  test_acc: 0.6307
2026-05-10 03:43:46,998 [INFO] Epoch 26/200 — loss: 0.6278  train_acc: 0.8215  test_acc: 0.5759
......
2026-05-10 09:40:40,691 [INFO] Epoch 151/200 — loss: 0.1796  train_acc: 0.9554  test_acc: 0.3078
```

**Comments:**

Large gap between training (95.5%) and test (30.8%) accuracy indicates significant overfitting. Accuracy between train and test grows otgether until around epoch 20 and 72%, then it starts dying and stabilizing around 30% test.

---

#### DVS-CIFAR10

**Configuration:** `tp_otttnet_dvscifar10.yaml`

**Status:** 🟠 Incomplete

**Accuracy:** ~42% (at epoch 27)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 07:43:02,481 [INFO] Epoch 18/200 — loss: 1.6569  train_acc: 0.5477  test_acc: 0.5160
2026-05-10 07:55:44,093 [INFO] Epoch 19/200 — loss: 1.7047  train_acc: 0.5349  test_acc: 0.4690
2026-05-10 08:08:25,943 [INFO] Epoch 20/200 — loss: 1.7714  train_acc: 0.5341  test_acc: 0.4320
2026-05-10 08:21:07,735 [INFO] Epoch 21/200 — loss: 1.6999  train_acc: 0.5485  test_acc: 0.3620
2026-05-10 08:33:49,358 [INFO] Epoch 22/200 — loss: 1.7355  train_acc: 0.5504  test_acc: 0.3030
2026-05-10 08:46:31,075 [INFO] Epoch 23/200 — loss: 1.6949  train_acc: 0.5581  test_acc: 0.2900
2026-05-10 08:59:12,900 [INFO] Epoch 24/200 — loss: 1.7457  train_acc: 0.5553  test_acc: 0.3280
2026-05-10 09:11:54,732 [INFO] Epoch 25/200 — loss: 1.6675  train_acc: 0.5657  test_acc: 0.3620
2026-05-10 09:24:36,507 [INFO] Epoch 26/200 — loss: 1.6579  train_acc: 0.5714  test_acc: 0.2790
2026-05-10 09:37:18,327 [INFO] Epoch 27/200 — loss: 1.6054  train_acc: 0.5796  test_acc: 0.4140
2026-05-10 09:50:00,450 [INFO] Epoch 28/200 — loss: 1.6106  train_acc: 0.5867  test_acc: 0.4530
```

**Comments:**

Early in training (epoch 27/200). Training accuracy ~58% and test ~41%, test accuracy is fluctuating while training accuracy is steadily increasing. Shows some learning but not very stable yet.

---

#### DVS-Gesture

**Configuration:** `tp_otttnet_dvsgesture.yaml`

**Status:** 🟠 Incomplete

**Accuracy:** ~87% (at epoch 49)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 09:40:30,627 [INFO] Epoch 49/200 — loss: 0.1745  train_acc: 0.9640  test_acc: 0.8674
```

**Comments:**

Good performance on DVS-Gesture with ~87% test accuracy and 96.4% training accuracy. Overfitting already but still growing with some isolated fluctuations. To be considered that numbers on the TP papers are on 32x32 DVSGesture, we are working on 128x128.

---

#### SVHN

**Configuration:** `tp_otttnet_svhn.yaml`

**Status:** [To be filled]

**Accuracy:** -

**Campaign name:** - 

**Terminal Output:**

```
[Paste terminal output here]
```

**Comments:**

[Your comments and analysis here]

---

### TPNet

#### CIFAR-10

**Configuration:** `tp_tpnet_cifar10.yaml`

**Status:** 🟠 Problematic

**Accuracy:** 29.1%

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-09 22:07:46,304 [INFO] Epoch 11/200 — loss: 0.9099  train_acc: 0.7305  test_acc: 0.7135
2026-05-09 22:10:37,763 [INFO] Epoch 12/200 — loss: 0.8887  train_acc: 0.7379  test_acc: 0.7256
2026-05-09 22:13:29,093 [INFO] Epoch 13/200 — loss: 0.8773  train_acc: 0.7440  test_acc: 0.7226
2026-05-09 22:16:20,391 [INFO] Epoch 14/200 — loss: 0.8545  train_acc: 0.7524  test_acc: 0.7334
2026-05-09 22:19:11,642 [INFO] Epoch 15/200 — loss: 0.8238  train_acc: 0.7575  test_acc: 0.7302
2026-05-09 22:22:02,998 [INFO] Epoch 16/200 — loss: 0.8074  train_acc: 0.7641  test_acc: 0.7376
2026-05-09 22:24:54,339 [INFO] Epoch 17/200 — loss: 0.7924  train_acc: 0.7689  test_acc: 0.7110
2026-05-09 22:27:45,711 [INFO] Epoch 18/200 — loss: 0.7810  train_acc: 0.7715  test_acc: 0.6994
2026-05-09 22:30:37,019 [INFO] Epoch 19/200 — loss: 0.7667  train_acc: 0.7751  test_acc: 0.7370
2026-05-09 22:33:28,346 [INFO] Epoch 20/200 — loss: 0.7631  train_acc: 0.7812  test_acc: 0.7265
2026-05-09 22:36:19,601 [INFO] Epoch 21/200 — loss: 0.7488  train_acc: 0.7818  test_acc: 0.7046
2026-05-09 22:39:10,942 [INFO] Epoch 22/200 — loss: 0.7279  train_acc: 0.7898  test_acc: 0.7303
2026-05-09 22:42:02,284 [INFO] Epoch 23/200 — loss: 0.7182  train_acc: 0.7941  test_acc: 0.7204
2026-05-09 22:44:53,601 [INFO] Epoch 24/200 — loss: 0.7069  train_acc: 0.7963  test_acc: 0.7184
2026-05-09 22:47:44,972 [INFO] Epoch 25/200 — loss: 0.6988  train_acc: 0.7988  test_acc: 0.6805
2026-05-09 22:50:36,313 [INFO] Epoch 26/200 — loss: 0.6911  train_acc: 0.8013  test_acc: 0.6898
2026-05-09 22:53:27,697 [INFO] Epoch 27/200 — loss: 0.6748  train_acc: 0.8042  test_acc: 0.6445
....
2026-05-10 07:04:35,255 [INFO] Epoch 199/200 — loss: 0.2850  train_acc: 0.9281  test_acc: 0.3053
2026-05-10 07:07:26,593 [INFO] Epoch 200/200 — loss: 0.2810  train_acc: 0.9283  test_acc: 0.2909
2026-05-10 07:07:26,594 [INFO] Training done in 34266.1 s. Final test accuracy: 0.2909
```

**Comments:**

Catastrophic failure on CIFAR-10 with TP trainer. Despite 92.8% training accuracy, test accuracy collapses to just 29%. Again there is an initial phase of learning where train and test grow together until around epoch 20, then it starts dying and stabilizing around 30% test accuracy.

---

#### DVS-CIFAR10

**Configuration:** `tp_tpnet_dvscifar10.yaml`

**Status:** 🟠 Problematic (Incomplete)

**Accuracy:** ~34% (at epoch 49)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 00:51:45,319 [INFO] Epoch 8/200 — loss: 1.6315  train_acc: 0.4964  test_acc: 0.4580
2026-05-10 01:04:25,831 [INFO] Epoch 9/200 — loss: 1.6164  train_acc: 0.5087  test_acc: 0.4350
2026-05-10 01:17:06,519 [INFO] Epoch 10/200 — loss: 1.6313  train_acc: 0.5066  test_acc: 0.4980
2026-05-10 01:29:47,569 [INFO] Epoch 11/200 — loss: 1.6319  train_acc: 0.5106  test_acc: 0.4330
2026-05-10 01:42:28,526 [INFO] Epoch 12/200 — loss: 1.6121  train_acc: 0.5150  test_acc: 0.3820
2026-05-10 01:55:09,172 [INFO] Epoch 13/200 — loss: 1.6543  train_acc: 0.5183  test_acc: 0.2150
2026-05-10 02:07:50,098 [INFO] Epoch 14/200 — loss: 1.9057  train_acc: 0.5102  test_acc: 0.4230
2026-05-10 02:20:31,122 [INFO] Epoch 15/200 — loss: 1.7313  train_acc: 0.5091  test_acc: 0.4430
2026-05-10 02:33:12,006 [INFO] Epoch 16/200 — loss: 1.7192  train_acc: 0.5170  test_acc: 0.4700
2026-05-10 02:45:53,036 [INFO] Epoch 17/200 — loss: 1.6766  train_acc: 0.5191  test_acc: 0.4970
2026-05-10 02:58:34,122 [INFO] Epoch 18/200 — loss: 1.7802  train_acc: 0.5136  test_acc: 0.2510
...
2026-05-10 09:31:55,155 [INFO] Epoch 49/200 — loss: 2.2412  train_acc: 0.5566  test_acc: 0.3430
```

**Comments:**

TP trainer failing on DVS-CIFAR10 with only ~34% test accuracy at epoch 49. Training accuracy ~56% suggests the model is not learning effectively. Similar failure pattern to CIFAR-10. Also here there is an initial phase of learning where train and test grow together until around epoch 20, then it starts dying and stabilizing around 30-40% test accuracy.

---

#### DVS-Gesture

**Configuration:** `tp_tpnet_dvsgesture.yaml`

**Status:** 🟢* Incomplete

**Accuracy:** ~92% (at epoch 142)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 09:41:36,887 [INFO] Epoch 142/200 — loss: 0.0000  train_acc: 1.0000  test_acc: 0.9015
```

**Comments:**

Good performance on DVS-Gesture reaching ~90%+ test accuracy with perfect training accuracy. TP trainer performs very well on this dataset, showing ~91.5% stability on average. Again remember that we are working 128x128 DVSGesture, while the TP paper numbers are on 32x32, so this is a good result.

---

#### SVHN

**Configuration:** `tp_tpnet_svhn.yaml`

**Status:** 🟠 Problematic (Incomplete)

**Accuracy:** ~71% (at epoch 61)

**Campaign name:** 10-05

**Terminal Output:**

```
2026-05-10 04:08:45,590 [INFO] Epoch 15/200 — loss: 0.3780  train_acc: 0.9252  test_acc: 0.9212
2026-05-10 04:15:55,255 [INFO] Epoch 16/200 — loss: 0.3779  train_acc: 0.9252  test_acc: 0.9173
2026-05-10 04:23:04,995 [INFO] Epoch 17/200 — loss: 0.3669  train_acc: 0.9267  test_acc: 0.9099
2026-05-10 04:30:14,735 [INFO] Epoch 18/200 — loss: 0.3531  train_acc: 0.9286  test_acc: 0.9189
2026-05-10 04:37:24,458 [INFO] Epoch 19/200 — loss: 0.3579  train_acc: 0.9311  test_acc: 0.9192
2026-05-10 04:44:34,238 [INFO] Epoch 20/200 — loss: 0.3646  train_acc: 0.9302  test_acc: 0.8089
2026-05-10 04:51:43,857 [INFO] Epoch 21/200 — loss: 0.3861  train_acc: 0.9290  test_acc: 0.6790
...
2026-05-10 09:38:05,629 [INFO] Epoch 61/200 — loss: 0.3717  train_acc: 0.9463  test_acc: 0.7066
```

**Comments:**

TP trainer shows significant overfitting on SVHN with training accuracy at 94.6% but test accuracy stuck around 71%. Performance degrades after epoch 20 where it was at 92.2% with comparable test accuracy, indicating convergence issues.
