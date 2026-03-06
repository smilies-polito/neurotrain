# ------------------ # BPTT TESTS # ------------------ #


# TRAINING
# BPTT on CIFAR-10 with VGG11 architecture
bptt_cifar10_vgg11:
	python3 bptt_cifar10_vgg11.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto
# BPTT on Fashion-MNIST with a simple convolutional architecture
bptt_fmnist_conv:
	python3 bptt_fmnist_conv.py --epochs 10 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto
# BPTT on MNIST with a simple fully connected architecture
bptt_mnist_fc:
	python3 bptt_mnist_fc.py --epochs 10 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto
# BPTT on DVS-Gesture with a simple recurrent architecture
bptt_dvsgest_r:
	python3 bptt_dvsgest_r.py --epochs 10 --batch-size 16 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto
# BPTT on N-MNIST with a simple recurrent architecture
bptt_nmnist_r:
	python3 bptt_nmnist_r.py --epochs 10 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto
# BPTT on SVHN with a simple recurrent architecture
bptt_svhn_r:
	python3 bptt_svhn_r.py --epochs 10 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto


# ------------------ # OSTTP TESTS # ------------------ #


# TRAINING
# OSTTP on SHD with the paper-style recurrent architecture
osttp_shd_repro:
	python3 osttp_shd_repro.py


# OPTUNA HYPERPARAMETER OPTIMIZATION
# BPTT on CIFAR-10 with VGG11 architecture
bptt_cifar10_vgg11_optuna:
	python3 bptt_cifar10_vgg11.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_cifar10_vgg11
# BPTT on Fashion-MNIST with a simple convolutional architecture
bptt_fmnist_conv_optuna:
	python3 bptt_fmnist_conv.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_fmnist_conv
# BPTT on MNIST with a simple fully connected architecture
bptt_mnist_fc_optuna:
	python3 bptt_mnist_fc.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_mnist_fc
# BPTT on DVS-Gesture with a simple recurrent architecture
bptt_dvsgest_r_optuna:
	python3 bptt_dvsgest_r.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_dvsgest_r
# BPTT on N-MNIST with a simple recurrent architecture
bptt_nmnist_r_optuna:
	python3 bptt_nmnist_r.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_nmnist_r
# BPTT on SVHN with a simple recurrent architecture
bptt_svhn_r_optuna:
	python3 bptt_svhn_r.py --epochs 1 --batch-size 64 --timesteps 10 --lr 1e-3 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name bptt_svhn_r

# OSTTP on SHD with the paper-style recurrent architecture
osttp_shd_repro_optuna:
	python3 osttp_shd_repro.py --epochs 1 --batch-size 64 --timesteps 100 --lr 2e-4 --beta 0.95 --threshold 1.0 --seed 42 --device auto --optuna-trials 10 --optuna-epochs 5 --study-name osttp_shd_repro


# HPC RUNS
# BPTT on CIFAR-10 with VGG11 architecture
bptt_cifar10_vgg11_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_cifar10_vgg11.sbatch
# BPTT on Fashion-MNIST with a simple convolutional architecture
bptt_fmnist_conv_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_fmnist_conv.sbatch
# BPTT on MNIST with a simple fully connected architecture
bptt_mnist_fc_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_mnist_fc.sbatch
# BPTT on DVS-Gesture with a simple recurrent architecture
bptt_dvsgest_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_dvsgest_r.sbatch
# BPTT on N-MNIST with a simple recurrent architecture
bptt_nmnist_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_nmnist_r.sbatch
# BPTT on SVHN with a simple recurrent architecture
bptt_svhn_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/bptt_svhn_r.sbatch

# OSTTP on SHD with the paper-style recurrent architecture
osttp_shd_repro_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch --export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif hpc/osttp_shd_repro.sbatch
