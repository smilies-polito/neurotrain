
# Deep Spike Learning with Local Classifiers



This repository contains a PyTorch implementation for the paper [Deep Spike Learning with Local Classifiers](https://ieeexplore.ieee.org/document/9837810), IEEE Transactions on Cybernetics, 2022. 

### Dependencies
- torch v1.0.1
- torchvision v0.2.1
### How to run

1. Use Anaconda to install required dependencies

```sh
conda env create -f=./env.yaml -p /your_env_path
```

2. Train SNNs with local learning rules
```sh
python main_train.py --dataset CIFAR10 --dim-in-decoder=1024 --epochs 400 --lr 1e-3 --lr-decay-fact 0.1 --lr-decay-milestones 400 --model CIFARCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL
python main_train.py --dataset SVHN --dim-in-decoder 1024 --epochs 100 --lr 3e-3 --lr-decay-fact 0.2 --lr-decay-milestones 30 60 90 --model SVHNCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL
python main_train.py --dataset FashionMNIST --dim-in-decoder 1024 --epochs 150 --lr 3e-3 --lr-decay-fact 0.1 --lr-decay-milestones 80 120 --model FashionCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL
python main_train.py --dataset FashionMNIST --epochs 150 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 60 120 --model FashionDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL
python main_train.py --dataset MNIST --dim-in-decoder 1024 --epochs 150 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 60 120 --model MNISTCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL
python main_train.py --dataset MNIST --epochs 150 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 60 120 --model MNISTDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule BELL

python main_train.py --dataset CIFAR10 --dim-in-decoder=1024 --epochs 200 --lr 5e-4 --lr-decay-fact 0.1 --lr-decay-milestones 100 200 --model CIFARCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL
python main_train.py --dataset SVHN --dim-in-decoder 1024 --epochs 100 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 40 80 --model SVHNCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL
python main_train.py --dataset FashionMNIST --dim-in-decoder 1024 --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model FashionCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL
python main_train.py --dataset FashionMNIST --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model FashionDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL
python main_train.py --dataset MNIST --dim-in-decoder 1024 --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model MNISTCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL
python main_train.py --dataset MNIST --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model MNISTDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule FELL

python main_train.py --dataset CIFAR10 --dim-in-decoder=1024 --epochs 200 --lr 5e-4 --lr-decay-fact 0.1 --lr-decay-milestones 100 200 --model CIFARCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL
python main_train.py --dataset SVHN --dim-in-decoder 1024 --epochs 100 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 40 80 --model SVHNCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL
python main_train.py --dataset FashionMNIST --dim-in-decoder 1024 --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model FashionCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL
python main_train.py --dataset FashionMNIST --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model FashionDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL
python main_train.py --dataset MNIST --dim-in-decoder 1024 --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model MNISTCNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL
python main_train.py --dataset MNIST --epochs 50 --lr 5e-4 --lr-decay-fact 0.2 --lr-decay-milestones 15 30 45 --model MNISTDNN --print-stats --tau 1 --thresh 1 --time-window 10 --learning-rule ELL

```




### Citation:
If you find this work useful, please consider citing:
```
@ARTICLE{ma2022spklocal,  
    author={Ma, Chenxiang and Yan, Rui and Yu, Zhaofei and Yu, Qiang},  
    journal={IEEE Transactions on Cybernetics},   
    title={Deep Spike Learning With Local Classifiers},   
    year={2022},  
    doi={10.1109/TCYB.2022.3188015}
    }
```

### Acknowledgments
Parts of this code were derived from [anokland/local-loss](https://github.com/anokland/local-loss). Thanks to the authors for their code.  