# Makefile — convenience targets for the SNN benchmarking framework.

PYTHON ?= python3

BENCH_CONFIG  ?= config/benchmarking.yaml
CUSTOM_CONFIG ?= config/experiments.yaml
EXP_NAME      ?=

# ── Main entry points ───────────────────────────────────────────────────────

## Run a benchmarking campaign
bench:
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) $(if $(EXP_NAME),--name $(EXP_NAME),)

## Run a benchmarking campaign with optuna
bench-opt:
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) --name debug_opt --inline

## Run custom experiments
custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) $(if $(EXP_NAME),--name $(EXP_NAME),)

## Dry-run: print experiment list without running
dry-bench:
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) --dry-run

dry-custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) --dry-run

# ── Testing ─────────────────────────────────────────────────────────────────

## Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run a single smoke test (inline, no subprocess overhead)
smoke:
	$(PYTHON) run_exp_campaign.py \
		--benchmarking config/benchmarking.yaml \
		--name smoke_$(shell date +%Y%m%d_%H%M%S) \
		--inline

# ── Cleanup ─────────────────────────────────────────────────────────────────

## Remove all experiment outputs
clean:
	rm -rf experiments/

## Remove Python cache files
clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

## Submit all per-trainer benchmarking jobs to SLURM (one job per algorithm)
all-opt:
	sbatch hpc/bench_bptt.sbatch
	sbatch hpc/bench_decolle.sbatch
	sbatch hpc/bench_ell.sbatch
	sbatch hpc/bench_eprop.sbatch
	sbatch hpc/bench_esd_rtrl.sbatch
	sbatch hpc/bench_etlp.sbatch
	sbatch hpc/bench_ostl.sbatch
	sbatch hpc/bench_osttp.sbatch
	sbatch hpc/bench_ottt.sbatch
	sbatch hpc/bench_stsf.sbatch
	sbatch hpc/bench_tp.sbatch

.PHONY: bench custom dry-bench dry-custom test smoke clean clean-cache all-opt
