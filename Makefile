
# Default interpreter for all Python entrypoints.
PYTHON ?= python3



#  /$$$$$$$  /$$$$$$$$ /$$   /$$  /$$$$$$  /$$   /$$ /$$      /$$  /$$$$$$  /$$$$$$$  /$$   /$$ /$$$$$$ /$$   /$$  /$$$$$$ 
# | $$__  $$| $$_____/| $$$ | $$ /$$__  $$| $$  | $$| $$$    /$$$ /$$__  $$| $$__  $$| $$  /$$/|_  $$_/| $$$ | $$ /$$__  $$
# | $$  \ $$| $$      | $$$$| $$| $$  \__/| $$  | $$| $$$$  /$$$$| $$  \ $$| $$  \ $$| $$ /$$/   | $$  | $$$$| $$| $$  \__/
# | $$$$$$$ | $$$$$   | $$ $$ $$| $$      | $$$$$$$$| $$ $$/$$ $$| $$$$$$$$| $$$$$$$/| $$$$$/    | $$  | $$ $$ $$| $$ /$$$$
# | $$__  $$| $$__/   | $$  $$$$| $$      | $$__  $$| $$  $$$| $$| $$__  $$| $$__  $$| $$  $$    | $$  | $$  $$$$| $$|_  $$
# | $$  \ $$| $$      | $$\  $$$| $$    $$| $$  | $$| $$\  $ | $$| $$  | $$| $$  \ $$| $$\  $$   | $$  | $$\  $$$| $$  \ $$
# | $$$$$$$/| $$$$$$$$| $$ \  $$|  $$$$$$/| $$  | $$| $$ \/  | $$| $$  | $$| $$  | $$| $$ \  $$ /$$$$$$| $$ \  $$|  $$$$$$/
# |_______/ |________/|__/  \__/ \______/ |__/  |__/|__/     |__/|__/  |__/|__/  |__/|__/  \__/|______/|__/  \__/ \______/ 
# ========================================================================================================================

# The commands here uses the benchmarking.py script.
# The arguments that can be used are divided in categories.
# Config files:
# --config: path to the benchmark config file with the trainer configurations (required) 
# --networks-dir: path to the directory containing the network config files (required)
# Filters:
# --algorithms: list of algorithms to run (default: all)
# --networks: list of networks to run (default: all)
# --datasets: list of datasets to run (default: all)
# Training parameters to override:
# --epochs: number of epochs to train each network
# -- bathch-size: batch size to use for training 
# --lr: learning rate to use for training
# --timesteps: number of timesteps to use for training
# --device: device to use for training (default: cuda if available, else cpu)
# --seed: random seed to use for training (default: 0)
# --run-neurobench: whether to run the neurobench evaluation after training (default:

# Full test suite
bench-short:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --run-neurobench
# Only BPTT test suite
bptt:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --algorithms bptt --run-neurobench --datasets MNIST
# Only OTTT test suite
ottt:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --algorithms ottt
# Only DRTP test suite
drtp:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --algorithms drtp
# Only OSTL test suite
ostl:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --algorithms ostl
	
# All algorithms on each MNIST network
all-mnist-fc:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --networks fc_snn --datasets MNIST 
all-mnist-conv:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --networks conv_snn --datasets MNIST 
all-mnist-rsnn:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --networks r_snn --datasets MNIST 
all-mnist-vgg11:
	$(PYTHON) benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks --epochs 1 --networks vg11_snn --datasets MNIST 



#  /$$$$$$$  /$$$$$$$$ /$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$  /$$   /$$  /$$$$$$  /$$$$$$ /$$$$$$$  /$$$$$$ /$$       /$$$$$$ /$$$$$$$$ /$$     /$$
# | $$__  $$| $$_____/| $$__  $$| $$__  $$ /$$__  $$| $$__  $$| $$  | $$ /$$__  $$|_  $$_/| $$__  $$|_  $$_/| $$      |_  $$_/|__  $$__/|  $$   /$$/
# | $$  \ $$| $$      | $$  \ $$| $$  \ $$| $$  \ $$| $$  \ $$| $$  | $$| $$  \__/  | $$  | $$  \ $$  | $$  | $$        | $$     | $$    \  $$ /$$/ 
# | $$$$$$$/| $$$$$   | $$$$$$$/| $$$$$$$/| $$  | $$| $$  | $$| $$  | $$| $$        | $$  | $$$$$$$   | $$  | $$        | $$     | $$     \  $$$$/  
# | $$__  $$| $$__/   | $$____/ | $$__  $$| $$  | $$| $$  | $$| $$  | $$| $$        | $$  | $$__  $$  | $$  | $$        | $$     | $$      \  $$/   
# | $$  \ $$| $$      | $$      | $$  \ $$| $$  | $$| $$  | $$| $$  | $$| $$    $$  | $$  | $$  \ $$  | $$  | $$        | $$     | $$       | $$    
# | $$  | $$| $$$$$$$$| $$      | $$  | $$|  $$$$$$/| $$$$$$$/|  $$$$$$/|  $$$$$$/ /$$$$$$| $$$$$$$/ /$$$$$$| $$$$$$$$ /$$$$$$   | $$       | $$    
# |__/  |__/|________/|__/      |__/  |__/ \______/ |_______/  \______/  \______/ |______/|_______/ |______/|________/|______/   |__/       |__/   
# ==================================================================================================================================================
# Uses th reproducibility.py script to run the experiments for the reproducibility suite. The arguments that can be used are:
# --configs-dir: path to the directory containing the configuration files for the reproducibility suite (required)
# --epochs: number of epochs to train each network

# Reproducibility suite (paper configs under configs/reproducibility)
repro:
	$(PYTHON) reproducibility.py --configs-dir configs/reproducibility --epochs 1



#   /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$$  /$$$$$$ /$$$$$$$$ /$$   /$$ /$$      /$$
#  /$$__  $$| $$       /$$__  $$ /$$__  $$| $$__  $$|_  $$_/|__  $$__/| $$  | $$| $$$    /$$$
# | $$  \ $$| $$      | $$  \__/| $$  \ $$| $$  \ $$  | $$     | $$   | $$  | $$| $$$$  /$$$$
# | $$$$$$$$| $$      | $$ /$$$$| $$  | $$| $$$$$$$/  | $$     | $$   | $$$$$$$$| $$ $$/$$ $$
# | $$__  $$| $$      | $$|_  $$| $$  | $$| $$__  $$  | $$     | $$   | $$__  $$| $$  $$$| $$
# | $$  | $$| $$      | $$  \ $$| $$  | $$| $$  \ $$  | $$     | $$   | $$  | $$| $$\  $ | $$
# | $$  | $$| $$$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$ /$$$$$$   | $$   | $$  | $$| $$ \/  | $$
# |__/  |__/|________/ \______/  \______/ |__/  |__/|______/   |__/   |__/  |__/|__/     |__/
# ===========================================================================================
# Sections that showcase how to directly use the main with a configuration file.

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

# DRTP
drtp-mnist-fc:
	$(PYTHON) main.py --config configs/mnist_drtp.yaml --epochs $(EPOCHS)
drtp-mnist-conv:
	$(PYTHON) main.py --config configs/mnist_drtp_conv.yaml --epochs $(EPOCHS)
drtp-mnist:
	$(MAKE) drtp-mnist-fc
	$(MAKE) drtp-mnist-conv
drtp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms drtp

# OSTL
ostl-mnist:
	$(MAKE) ostl-mnist-fc
ostl-mnist-fc:
	$(PYTHON) main.py --config configs/benchmarking/ostl/ostl-mnist-fc_snn.yaml --epochs 1
ostl-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ostl




#  /$$   /$$ /$$$$$$$$ /$$$$$$ /$$        /$$$$$$
# | $$  | $$|__  $$__/|_  $$_/| $$       /$$__  $$
# | $$  | $$   | $$     | $$  | $$      | $$  \__/
# | $$  | $$   | $$     | $$  | $$      |  $$$$$$
# | $$  | $$   | $$     | $$  | $$       \____  $$
# | $$  | $$   | $$     | $$  | $$       /$$  \ $$
# |  $$$$$$/   | $$    /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/    |__/   |______/|________/ \______/

# Remove downloaded datasets
clean-data:
	rm -rf src/Data
	
# Remove experiments data
clean-experiments:
	rm -rf experiments
clean-results:
	rm -rf benchmark_results
clean-all:
	$(MAKE) clean-experiments
	$(MAKE) clean-results
