# Tooling entry points (override on CLI if needed)
PYTHON ?= python3
DEVICE ?= cuda
EPOCHS ?= 50

.PHONY: test full-test quick-test run-all-mnist clean

# Full test suite
full-test:
	$(PYTHON) run_all_benchmarks.py --epochs 40 --device $(DEVICE)

# Focused, faster subset of tests for quick feedback
quick-test:
	$(PYTHON) run_all_benchmarks.py --epochs 1 --device $(DEVICE)

# Convenience target: run all algorithms on MNIST only
run-all-mnist:
	$(PYTHON) run_all_benchmarks.py --epochs $(EPOCHS) --device $(DEVICE) --datasets MNIST $(if $(ALGORITHMS),--algorithms $(ALGORITHMS),)

# Remove Python bytecode and caches
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
