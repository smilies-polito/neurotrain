# Tooling entry points (override on CLI if needed)
PYTHON ?= python3
DEVICE ?= cuda
EPOCHS ?= 50
STOP_TEST_EPOCHS ?= 5
STOP_MIN_GAIN ?= 0.0
STOP_COMPLEX_EPOCHS ?= 10
STOP_COMPLEX_BATCH_SIZE ?= 32
STOP_COMPLEX_TIMESTEPS ?= 8
STOP_COMPLEX_LR ?= 0.005

# STOP complex backbones: per-target defaults (override from CLI if needed)
STOP_VGG11_CIFAR10_EPOCHS ?= $(STOP_COMPLEX_EPOCHS)
STOP_VGG11_CIFAR10_BATCH_SIZE ?= 32
STOP_VGG11_CIFAR10_TIMESTEPS ?= 4
STOP_VGG11_CIFAR10_LR ?= 0.001

STOP_RESNET18_CIFAR10_EPOCHS ?= $(STOP_COMPLEX_EPOCHS)
STOP_RESNET18_CIFAR10_BATCH_SIZE ?= 8
STOP_RESNET18_CIFAR10_TIMESTEPS ?= 8
STOP_RESNET18_CIFAR10_LR ?= 0.006

STOP_VGG11_SVHN_EPOCHS ?= $(STOP_COMPLEX_EPOCHS)
STOP_VGG11_SVHN_BATCH_SIZE ?= 32
STOP_VGG11_SVHN_TIMESTEPS ?= 4
STOP_VGG11_SVHN_LR ?= 0.001

STOP_RESNET18_SVHN_EPOCHS ?= $(STOP_COMPLEX_EPOCHS)
STOP_RESNET18_SVHN_BATCH_SIZE ?= 16
STOP_RESNET18_SVHN_TIMESTEPS ?= 8
STOP_RESNET18_SVHN_LR ?= 0.003



#  /$$$$$$$$ /$$   /$$ /$$       /$$             /$$$$$$$$ /$$$$$$$$  /$$$$$$  /$$$$$$$$ /$$$$$$ 
# | $$_____/| $$  | $$| $$      | $$            |__  $$__/| $$_____/ /$$__  $$|__  $$__//$$__  $$
# | $$      | $$  | $$| $$      | $$               | $$   | $$      | $$  \__/   | $$  | $$  \__/
# | $$$$$   | $$  | $$| $$      | $$               | $$   | $$$$$   |  $$$$$$    | $$  |  $$$$$$ 
# | $$__/   | $$  | $$| $$      | $$               | $$   | $$__/    \____  $$   | $$   \____  $$
# | $$      | $$  | $$| $$      | $$               | $$   | $$       /$$  \ $$   | $$   /$$  \ $$
# | $$      |  $$$$$$/| $$$$$$$$| $$$$$$$$         | $$   | $$$$$$$$|  $$$$$$/   | $$  |  $$$$$$/
# |__/       \______/ |________/|________/         |__/   |________/ \______/    |__/   \______/ 
                                                                                               
# Full test suite
complete-test-long:
	$(PYTHON) run_all_benchmarks.py --epochs 50 --device $(DEVICE)
# Focused, faster subset of tests for quick feedback
complete-test-short:
	$(PYTHON) run_all_benchmarks.py --epochs 1 --device $(DEVICE)



#  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$$$
# | $$__  $$ /$$__  $$|__  $$__//$$__  $$ /$$__  $$| $$_____/|__  $$__/
# | $$  \ $$| $$  \ $$   | $$  | $$  \ $$| $$  \__/| $$         | $$   
# | $$  | $$| $$$$$$$$   | $$  | $$$$$$$$|  $$$$$$ | $$$$$      | $$   
# | $$  | $$| $$__  $$   | $$  | $$__  $$ \____  $$| $$__/      | $$   
# | $$  | $$| $$  | $$   | $$  | $$  | $$ /$$  \ $$| $$         | $$   
# | $$$$$$$/| $$  | $$   | $$  | $$  | $$|  $$$$$$/| $$$$$$$$   | $$   
# |_______/ |__/  |__/   |__/  |__/  |__/ \______/ |________/   |__/   

# Convenience target: run all algorithms on MNIST only
all-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST $(if $(ALGORITHMS),--algorithms $(ALGORITHMS),)

# Convenience target: run all algorithms on CIFAR10 only
all-cifar10:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets CIFAR10 $(if $(ALGORITHMS),--algorithms $(ALGORITHMS),)



#   /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$$  /$$$$$$ /$$$$$$$$ /$$   /$$ /$$      /$$
#  /$$__  $$| $$       /$$__  $$ /$$__  $$| $$__  $$|_  $$_/|__  $$__/| $$  | $$| $$$    /$$$
# | $$  \ $$| $$      | $$  \__/| $$  \ $$| $$  \ $$  | $$     | $$   | $$  | $$| $$$$  /$$$$
# | $$$$$$$$| $$      | $$ /$$$$| $$  | $$| $$$$$$$/  | $$     | $$   | $$$$$$$$| $$ $$/$$ $$
# | $$__  $$| $$      | $$|_  $$| $$  | $$| $$__  $$  | $$     | $$   | $$__  $$| $$  $$$| $$
# | $$  | $$| $$      | $$  \ $$| $$  | $$| $$  \ $$  | $$     | $$   | $$  | $$| $$\  $ | $$
# | $$  | $$| $$$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$ /$$$$$$   | $$   | $$  | $$| $$ \/  | $$
# |__/  |__/|________/ \______/  \______/ |__/  |__/|______/   |__/   |__/  |__/|__/     |__/

# BPTT
bptt-mnist-fc:
	$(PYTHON) main.py --config configs/benchmarking/bptt/bptt-mnist-fc_snn.yaml --epochs 1
bptt-mnist-conv:
	$(PYTHON) main.py --config configs/benchmarking/bptt/bptt-mnist-conv_snn.yaml --epochs 1
bptt-mnist-rsnn:
	$(PYTHON) main.py --config configs/benchmarking/bptt/bptt-mnist-r_snn.yaml --epochs 1
bptt-mnist-vgg11:
	$(PYTHON) main.py --config configs/benchmarking/bptt/bptt-mnist-vg11_snn.yaml --epochs 1
bptt-mnist:
	$(MAKE) bptt-mnist-fc
	$(MAKE) bptt-mnist-conv
	$(MAKE) bptt-mnist-rsnn
	$(MAKE) bptt-mnist-vgg11

# STSF
stsf-mnist:
	$(PYTHON) main.py --config configs/mnist_default.yaml --epochs $(EPOCHS)
stsf-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stsf

# E-prop
eprop-mnist:
	$(PYTHON) main.py --config configs/mnist_eprop.yaml --epochs $(EPOCHS)
eprop-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms eprop

# DECOLLE
decolle-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms decolle
decolle-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms decolle

# OTTT
ottt-mnist-fc:
	$(PYTHON) main.py --config configs/benchmarking/ottt/ottt-mnist-fc_snn.yaml --epochs 1
ottt-mnist-conv:
	$(PYTHON) main.py --config configs/benchmarking/ottt/ottt-mnist-conv_snn.yaml --epochs 1
ottt-mnist-rsnn:
	$(PYTHON) main.py --config configs/benchmarking/ottt/ottt-mnist-r_snn.yaml --epochs 1
ottt-mnist-vgg11:
	$(PYTHON) main.py --config configs/benchmarking/ottt/ottt-mnist-vg11_snn.yaml --epochs 1
ottt-mnist:
	$(MAKE) ottt-mnist-fc
	$(MAKE) ottt-mnist-conv
	$(MAKE) ottt-mnist-rsnn
	$(MAKE) ottt-mnist-vgg11
ottt-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ottt

# DRTP
drtp-mnist:
	$(PYTHON) main.py --config configs/mnist_drtp.yaml --epochs $(EPOCHS)
drtp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms drtp

# OSTL
ostl-mnist:
	$(PYTHON) main.py --config configs/mnist_ostl.yaml --epochs $(EPOCHS)
ostl-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ostl

# ELL
ell-mnist:
	$(PYTHON) main.py --config configs/mnist_ell.yaml --epochs $(EPOCHS)
ell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ell

# FELL
fell-mnist:
	$(PYTHON) main.py --config configs/mnist_fell.yaml --epochs $(EPOCHS)
fell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms fell

# BELL
bell-mnist:
	$(PYTHON) main.py --config configs/mnist_bell.yaml --epochs $(EPOCHS)
bell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms bell

# STLLR
stllr-mnist:
	$(PYTHON) main.py --config configs/mnist_stllr.yaml --epochs $(EPOCHS)
stllr-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stllr

# STOP
stop-mnist:
	$(PYTHON) main.py --config configs/mnist_stop.yaml --epochs $(EPOCHS)
stop-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stop

# ETLP
etlp-mnist:
	$(PYTHON) main.py --config configs/mnist_etlp.yaml --epochs $(EPOCHS)
etlp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms etlp

# Trace Propagation (TP)
tp-mnist:
	$(PYTHON) main.py --config configs/mnist_tp.yaml --epochs $(EPOCHS)
tp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms tp

# ES-D-RTRL
esd-rtrl-mnist:
	$(PYTHON) main.py --config configs/mnist_esd_rtrl.yaml --epochs $(EPOCHS)
esd-rtrl-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms esd_rtrl


# STOP on complex datasets (new backbones)
stop-vgg11-cifar10:
	$(PYTHON) main.py --config configs/cifar10_stop_vgg11.yaml --epochs $(STOP_VGG11_CIFAR10_EPOCHS) --batch-size $(STOP_VGG11_CIFAR10_BATCH_SIZE) --T $(STOP_VGG11_CIFAR10_TIMESTEPS) --lr $(STOP_VGG11_CIFAR10_LR)
stop-resnet18-cifar10:
	$(PYTHON) main.py --config configs/cifar10_stop_resnet18.yaml --epochs $(STOP_RESNET18_CIFAR10_EPOCHS) --batch-size $(STOP_RESNET18_CIFAR10_BATCH_SIZE) --T $(STOP_RESNET18_CIFAR10_TIMESTEPS) --lr $(STOP_RESNET18_CIFAR10_LR)
stop-vgg11-svhn:
	$(PYTHON) main.py --config configs/svhn_stop_vgg11.yaml --epochs $(STOP_VGG11_SVHN_EPOCHS) --batch-size $(STOP_VGG11_SVHN_BATCH_SIZE) --T $(STOP_VGG11_SVHN_TIMESTEPS) --lr $(STOP_VGG11_SVHN_LR)
stop-resnet18-svhn:
	$(PYTHON) main.py --config configs/svhn_stop_resnet18.yaml --epochs $(STOP_RESNET18_SVHN_EPOCHS) --batch-size $(STOP_RESNET18_SVHN_BATCH_SIZE) --T $(STOP_RESNET18_SVHN_TIMESTEPS) --lr $(STOP_RESNET18_SVHN_LR)
stop-newnets-cifar10:
	$(MAKE) stop-vgg11-cifar10
	$(MAKE) stop-resnet18-cifar10
stop-newnets-svhn:
	$(MAKE) stop-vgg11-svhn
	$(MAKE) stop-resnet18-svhn
stop-newnets-all:
	$(MAKE) stop-newnets-cifar10
	$(MAKE) stop-newnets-svhn



#  /$$   /$$ /$$$$$$$$ /$$$$$$ /$$        /$$$$$$ 
# | $$  | $$|__  $$__/|_  $$_/| $$       /$$__  $$
# | $$  | $$   | $$     | $$  | $$      | $$  \__/
# | $$  | $$   | $$     | $$  | $$      |  $$$$$$ 
# | $$  | $$   | $$     | $$  | $$       \____  $$
# | $$  | $$   | $$     | $$  | $$       /$$  \ $$
# |  $$$$$$/   | $$    /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/    |__/   |______/|________/ \______/ 

# OSTTP on MNIST
run-osttp-mnist:
	$(PYTHON) main.py --config configs/mnist_osttp.yaml

# Remove Python bytecode and caches
clean-data:
	rm -rf src/Data
clean-experiments:
	rm -rf experiments
clean-results:
	rm -rf benchmark_results
clean-all:
	$(MAKE) clean-experiments
	$(MAKE) clean-results
	
