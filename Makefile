# Tooling entry points (override on CLI if needed)
PYTHON ?= python3
DEVICE ?= cuda
EPOCHS ?= 50
STOP_TEST_EPOCHS ?= 5
STOP_MIN_GAIN ?= 0.0



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
	


#   /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$$  /$$$$$$ /$$$$$$$$ /$$   /$$ /$$      /$$
#  /$$__  $$| $$       /$$__  $$ /$$__  $$| $$__  $$|_  $$_/|__  $$__/| $$  | $$| $$$    /$$$
# | $$  \ $$| $$      | $$  \__/| $$  \ $$| $$  \ $$  | $$     | $$   | $$  | $$| $$$$  /$$$$
# | $$$$$$$$| $$      | $$ /$$$$| $$  | $$| $$$$$$$/  | $$     | $$   | $$$$$$$$| $$ $$/$$ $$
# | $$__  $$| $$      | $$|_  $$| $$  | $$| $$__  $$  | $$     | $$   | $$__  $$| $$  $$$| $$
# | $$  | $$| $$      | $$  \ $$| $$  | $$| $$  \ $$  | $$     | $$   | $$  | $$| $$\  $ | $$
# | $$  | $$| $$$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$ /$$$$$$   | $$   | $$  | $$| $$ \/  | $$
# |__/  |__/|________/ \______/  \______/ |__/  |__/|______/   |__/   |__/  |__/|__/     |__/

# BPTT
bptt-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms bptt
bptt-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms bptt

# STSF
stsf-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms stsf
stsf-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stsf

# E-prop
eprop-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms eprop
eprop-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms eprop

# DECOLLE
decolle-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms decolle
decolle-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms decolle

# OTTT
ottt-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms ottt
ottt-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ottt

# DRTP
drtp-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms drtp
drtp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms drtp

# OSTL
ostl-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms ostl
ostl-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ostl

# ELL
ell-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms ell
ell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms ell

# FELL
fell-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms fell
fell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms fell

# BELL
bell-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms bell
bell-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms bell

# STLLR
stllr-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms stllr
stllr-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stllr

# STOP
stop-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms stop
stop-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms stop

# ETLP
etlp-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms etlp
etlp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms etlp

# Trace Propagation (TP)
tp-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms tp
tp-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms tp

# ES-D-RTRL
esd-rtrl-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST --algorithms esd_rtrl
esd-rtrl-all-datasets:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --algorithms esd_rtrl



#  /$$   /$$ /$$$$$$$$ /$$$$$$ /$$        /$$$$$$ 
# | $$  | $$|__  $$__/|_  $$_/| $$       /$$__  $$
# | $$  | $$   | $$     | $$  | $$      | $$  \__/
# | $$  | $$   | $$     | $$  | $$      |  $$$$$$ 
# | $$  | $$   | $$     | $$  | $$       \____  $$
# | $$  | $$   | $$     | $$  | $$       /$$  \ $$
# |  $$$$$$/   | $$    /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/    |__/   |______/|________/ \______/ 

# Remove Python bytecode and caches
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
