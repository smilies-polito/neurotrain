# Makefile — convenience targets for the SNN benchmarking framework.

PYTHON ?= python3

BENCH_CONFIG  ?= config/benchmarking.yaml
CUSTOM_CONFIG ?= config/experiments.yaml
EXP_NAME      ?=
HPC_SLURM_OUT ?= hpc/slurm_outputs


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


# --─ Optimization benchmarking (SLURM) ─────────────────────────────────────────

## Ensure SLURM output directory exists
hpc-mkdir:
	mkdir -p $(HPC_SLURM_OUT)

## Submit all per-trainer benchmarking jobs to SLURM (one job per algorithm)
all-opt: hpc-mkdir
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

opt-bptt: hpc-mkdir
	sbatch hpc/bench_bptt.sbatch
opt-decolle: hpc-mkdir
	sbatch hpc/bench_decolle.sbatch
opt-ell: hpc-mkdir
	sbatch hpc/bench_ell.sbatch
opt-eprop: hpc-mkdir
	sbatch hpc/bench_eprop.sbatch
opt-esd_rtrl: hpc-mkdir
	sbatch hpc/bench_esd_rtrl.sbatch
opt-etlp: hpc-mkdir
	sbatch hpc/bench_etlp.sbatch
opt-ostl: hpc-mkdir
	sbatch hpc/bench_ostl.sbatch
opt-osttp: hpc-mkdir
	sbatch hpc/bench_osttp.sbatch
opt-ottt: hpc-mkdir
	sbatch hpc/bench_ottt.sbatch
opt-stsf: hpc-mkdir
	sbatch hpc/bench_stsf.sbatch
opt-tp: hpc-mkdir
	sbatch hpc/bench_tp.sbatch


# ── Cleanup ─────────────────────────────────────────────────────────────────

## Remove all experiment outputs and clear HPC SLURM logs
clean: clean-hpc
	rm -rf experiments/

## Empty SLURM output directory without deleting it
clean-hpc:
	mkdir -p $(HPC_SLURM_OUT)
	find $(HPC_SLURM_OUT) -mindepth 1 -delete

## Remove Python cache files
clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true