## Versions

This file keeps a modular, append-only record of benchmark results across code versions and training algorithms.

### Conventions

- `Acc`: final test accuracy (fraction in `[0, 1]`).
- `Wall`: total wall-clock training time for the run.
- `Epoch`: average wall-clock time per epoch.

---

### v0.1a (BPTT, STSF, OTTT)

**Meta**
| Field | Value |
| --- | --- |
| Epochs | 40 |
| Algorithms | BPTT, STSF, OTTT |
| Notes | Initial recorded baseline for these three algorithms. |

**Final Summary**
| Dataset | Algo | Acc | Wall | Epoch |
| --- | --- | --- | --- | --- |
| MNIST | BPTT | 0.9758 | 371.1s | 9189ms |
| MNIST | STSF | 0.9017 | 236.9s | 5833ms |
| MNIST | OTTT | 0.7749 | 279.1s | 6886ms |
| FashionMNIST | BPTT | 0.8433 | 375.6s | 9298ms |
| FashionMNIST | STSF | 0.7512 | 238.8s | 5878ms |
| FashionMNIST | OTTT | 0.4765 | 279.9s | 6906ms |
| CIFAR10 | BPTT | 0.3657 | 443.6s | 10921ms |
| CIFAR10 | STSF | 0.3603 | 329.0s | 8052ms |
| CIFAR10 | OTTT | 0.3992 | 362.0s | 8878ms |
| SVHN | BPTT | 0.4524 | 653.7s | 15925ms |
| SVHN | STSF | 0.3357 | 501.2s | 12105ms |
| SVHN | OTTT | 0.5405 | 561.2s | 13605ms |

**NeuroBench Metrics**
| Dataset | Algo | Params | Footprint | ActSpars | Eff. MACs | Dense MACs | Savings | MemUpdates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | BPTT | 203,264 | 810.7 KB | 0.7039 | 0 | 5,081,600 | 100.0% | 6,649 |
| MNIST | STSF | 203,264 | 810.7 KB | 0.3533 | 0 | 5,081,600 | 100.0% | 6,649 |
| MNIST | OTTT | 203,264 | 810.7 KB | 0.6846 | 0 | 5,081,600 | 100.0% | 6,649 |
| FashionMNIST | BPTT | 203,264 | 810.7 KB | 0.5797 | 0 | 5,081,600 | 100.0% | 6,649 |
| FashionMNIST | STSF | 203,264 | 810.7 KB | 0.4505 | 0 | 5,081,600 | 100.0% | 6,649 |
| FashionMNIST | OTTT | 203,264 | 810.7 KB | 0.6742 | 0 | 5,081,600 | 100.0% | 6,649 |
| CIFAR10 | BPTT | 1,577,984 | 6.05 MB | 0.5637 | 0 | 39,449,600 | 100.0% | 13,043 |
| CIFAR10 | STSF | 1,577,984 | 6.05 MB | 0.3846 | 0 | 39,449,600 | 100.0% | 13,044 |
| CIFAR10 | OTTT | 1,577,984 | 6.05 MB | 0.7486 | 0 | 39,449,600 | 100.0% | 13,043 |
| SVHN | BPTT | 1,577,984 | 6.12 MB | 0.6278 | 0 | 39,449,600 | 100.0% | 11,575 |
| SVHN | STSF | 1,577,984 | 6.12 MB | 0.6116 | 0 | 39,449,600 | 100.0% | 11,567 |
| SVHN | OTTT | 1,577,984 | 6.12 MB | 0.8564 | 0 | 39,449,600 | 100.0% | 11,548 |
