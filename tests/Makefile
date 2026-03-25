# ------------------ # BPTT TESTS # ------------------ #


# TRAINING
# BPTT on CIFAR-10 with VGG11 architecture
bptt_cifar10_vgg11:
	python3 bptt_cifar10_vgg11.py
# BPTT on Fashion-MNIST with a simple convolutional architecture
bptt_fmnist_conv:
	python3 bptt_fmnist_conv.py
# BPTT on MNIST with a simple fully connected architecture
bptt_mnist_fc:
	python3 bptt_mnist_fc.py
# BPTT on DVS-Gesture with VGG-9 architecture (BPTT baseline for OTTT/STLLR/TP)
bptt_dvsgest_vgg9:
	python3 bptt_dvsgest_vgg9.py
# BPTT on DVS-Gesture with a simple recurrent architecture
bptt_dvsgest_r:
	python3 bptt_dvsgest_r.py
# BPTT on N-MNIST with a simple recurrent architecture
bptt_nmnist_r:
	python3 bptt_nmnist_r.py
# BPTT on SVHN with a simple recurrent architecture
bptt_svhn_r:
	python3 bptt_svhn_r.py


# ------------------ # OSTTP TESTS # ------------------ #
# TRAINING
# OSTTP on SHD with the paper-style recurrent architecture
osttp_shd_repro:
	python3 osttp_shd_repro.py


# OPTUNA HYPERPARAMETER OPTIMIZATION
# BPTT on CIFAR-10 with VGG11 architecture
bptt_cifar10_vgg11_optuna:
	python3 bptt_cifar10_vgg11.py --optuna-trials 10 --optuna-epochs 5 --study-name bptt_cifar10_vgg11 --optuna-storage sqlite:///optuna_results/bptt_cifar10_vgg11.db
# BPTT on Fashion-MNIST with a simple convolutional architecture
bptt_fmnist_conv_optuna:
	python3 bptt_fmnist_conv.py --optuna-epochs 5 --study-name bptt_fmnist_conv --optuna-storage sqlite:///optuna_results/bptt_fmnist_conv.db
# BPTT on MNIST with a simple fully connected architecture
bptt_mnist_fc_optuna:
	python3 bptt_mnist_fc.py  --optuna-trials 20 --optuna-epochs 7 --study-name bptt_mnist_fc --optuna-storage sqlite:///optuna_results/bptt_mnist_fc.db
# BPTT on DVS-Gesture with VGG-9 architecture
bptt_dvsgest_vgg9_optuna:
	python3 bptt_dvsgest_vgg9.py --optuna-epochs 5 --study-name bptt_dvsgest_vgg9 --optuna-storage sqlite:///optuna_results/bptt_dvsgest_vgg9.db
# BPTT on DVS-Gesture with a simple recurrent architecture
bptt_dvsgest_r_optuna:
	python3 bptt_dvsgest_r.py --optuna-epochs 5 --study-name bptt_dvsgest_r --optuna-storage sqlite:///optuna_results/bptt_dvsgest_r.db
# BPTT on N-MNIST with a simple recurrent architecture
bptt_nmnist_r_optuna:
	python3 bptt_nmnist_r.py --optuna-epochs 5 --study-name bptt_nmnist_r --optuna-storage sqlite:///optuna_results/bptt_nmnist_r.db
# BPTT on SVHN with a simple recurrent architecture
bptt_svhn_r_optuna:
	python3 bptt_svhn_r.py --optuna-epochs 5 --study-name bptt_svhn_r --optuna-storage sqlite:///optuna_results/bptt_svhn_r.db

# OSTTP on SHD with the paper-style recurrent architecture
osttp_shd_repro_optuna:
	python3 osttp_shd_repro.py --optuna-trials 10 --optuna-epochs 5 --study-name osttp_shd_repro --optuna-storage sqlite:///optuna_results/osttp_shd_repro.db


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
