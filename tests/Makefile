.PHONY: \
	bptt_cifar10_vgg11 bptt_cifar10_vgg11_optuna bptt_cifar10_vgg11_sbatch \
	bptt_fmnist_conv bptt_fmnist_conv_optuna bptt_fmnist_conv_sbatch \
	bptt_mnist_fc bptt_mnist_fc_optuna bptt_mnist_fc_sbatch \
	bptt_dvsgest_r bptt_dvsgest_r_optuna bptt_dvsgest_r_sbatch \
	bptt_nmnist_r bptt_nmnist_r_optuna bptt_nmnist_r_sbatch \
	bptt_svhn_r bptt_svhn_r_optuna bptt_svhn_r_sbatch

bptt_cifar10_vgg11:
	python3 bptt_cifar10_vgg11.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto
	
bptt_cifar10_vgg11_optuna:
	python3 bptt_cifar10_vgg11.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_cifar10_vgg11

# Submit the provided sbatch script from the repo root so Slurm paths work.
bptt_cifar10_vgg11_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_cifar10_vgg11.sbatch

bptt_fmnist_conv:
	python3 bptt_fmnist_conv.py \
		--epochs 10 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto

bptt_fmnist_conv_optuna:
	python3 bptt_fmnist_conv.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_fmnist_conv

bptt_fmnist_conv_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_fmnist_conv.sbatch

bptt_mnist_fc:
	python3 bptt_mnist_fc.py \
		--epochs 10 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto

bptt_mnist_fc_optuna:
	python3 bptt_mnist_fc.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_mnist_fc

bptt_mnist_fc_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_mnist_fc.sbatch

bptt_dvsgest_r:
	python3 bptt_dvsgest_r.py \
		--epochs 10 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto

bptt_dvsgest_r_optuna:
	python3 bptt_dvsgest_r.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_dvsgest_r

bptt_dvsgest_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_dvsgest_r.sbatch

bptt_nmnist_r:
	python3 bptt_nmnist_r.py \
		--epochs 10 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto

bptt_nmnist_r_optuna:
	python3 bptt_nmnist_r.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_nmnist_r

bptt_nmnist_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_nmnist_r.sbatch

bptt_svhn_r:
	python3 bptt_svhn_r.py \
		--epochs 10 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto

bptt_svhn_r_optuna:
	python3 bptt_svhn_r.py \
		--epochs 1 \
		--batch-size 64 \
		--timesteps 10 \
		--lr 1e-3 \
		--beta 0.95 \
		--threshold 1.0 \
		--seed 42 \
		--device auto \
		--optuna-trials 10 \
		--optuna-epochs 5 \
		--study-name bptt_svhn_r

bptt_svhn_r_sbatch:
	cd .. && mkdir -p hpc/slurm_outputs && sbatch \
		--export=ALL,PROJECT_ROOT=$(abspath ..),APPTAINER_IMAGE=$(abspath ..)/hpc/snn-training-benchmarking.sif \
		hpc/bptt_svhn_r.sbatch
