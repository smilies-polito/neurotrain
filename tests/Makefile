.PHONY: bptt_cifar10_vgg11 bptt_cifar10_vgg11_optuna bptt_cifar10_vgg11_sbatch

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
