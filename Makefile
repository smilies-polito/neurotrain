# Tooling entry points (override on CLI if needed)
PYTHON ?= python3

.PHONY: test quick-test clean

# Full test suite
full-test:
	$(PYTHON) run_all_benchmarks.py --epoch 30

# Focused, faster subset of tests for quick feedback
quick-test:
	$(PYTHON) run_all_benchmarks.py --epoch 1

# Remove Python bytecode and caches
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
