# Tooling entry points (override on CLI if needed)
PYTHON ?= python3
DEVICE ?= cuda
EPOCHS ?= 50

.PHONY: test full-test quick-test run-all-mnist clean

## FULL TESTS
# Full test suite
complete-test-long:
	$(PYTHON) run_all_benchmarks.py --epochs 50 --device $(DEVICE)
# Focused, faster subset of tests for quick feedback
complete-test-short:
	$(PYTHON) run_all_benchmarks.py --epochs 1 --device $(DEVICE)

## TESTS ON DATASETS
# Convenience target: run all algorithms on MNIST only
run-all-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST $(if $(ALGORITHMS),--algorithms $(ALGORITHMS),)

# Trace Propagation on MNIST
run-tp-mnist:
	$(PYTHON) main.py --config configs/mnist_tp.yaml

# Remove Python bytecode and caches
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
