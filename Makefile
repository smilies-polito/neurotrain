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
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) --name debug_opt
## Dry-run: print experiment list without running
dry-bench:
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) --dry-run


## Run custom experiments
custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) $(if $(EXP_NAME),--name $(EXP_NAME),)
## Dry-run: print experiment list without running
dry-custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) --dry-run

## Run paper experiments (fill config/paper.yaml with best Optuna results first)
paper:
	$(PYTHON) run_exp_campaign.py --custom config/paper.yaml --name paper

bench-tp-vgg9-cifar10:
	$(PYTHON) run_exp_campaign.py --benchmarking config/benchmarking/tp_vgg9_cifar10.yaml



# ── VGG9 Full Matrix — local single-run (no Optuna) ────────────────────────
# Naming: vgg9-<trainer>-<nettype>-<dataset>
# nettype: tpnet = TP-style (leaky_integrator head, atan surrogate, conv_gain=1.8)
#          otttnet = OTTT-style (global_linear head, sigmoid surrogate, scale_after_lif=2.74)
# Config files live in config/vgg9/

## TP trainer
vgg9-tp-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_tpnet_cifar10.yaml --name vgg9_tp_tpnet_cifar10
vgg9-tp-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_tpnet_svhn.yaml --name vgg9_tp_tpnet_svhn
vgg9-tp-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_tpnet_dvsgesture.yaml --name vgg9_tp_tpnet_dvsgesture
vgg9-tp-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_tpnet_dvscifar10.yaml --name vgg9_tp_tpnet_dvscifar10
vgg9-tp-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_otttnet_cifar10.yaml --name vgg9_tp_otttnet_cifar10
vgg9-tp-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_otttnet_svhn.yaml --name vgg9_tp_otttnet_svhn
vgg9-tp-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_otttnet_dvsgesture.yaml --name vgg9_tp_otttnet_dvsgesture
vgg9-tp-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/tp_otttnet_dvscifar10.yaml --name vgg9_tp_otttnet_dvscifar10

## OTTT trainer
vgg9-ottt-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_tpnet_cifar10.yaml --name vgg9_ottt_tpnet_cifar10
vgg9-ottt-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_tpnet_svhn.yaml --name vgg9_ottt_tpnet_svhn
vgg9-ottt-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_tpnet_dvsgesture.yaml --name vgg9_ottt_tpnet_dvsgesture
vgg9-ottt-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_tpnet_dvscifar10.yaml --name vgg9_ottt_tpnet_dvscifar10
vgg9-ottt-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_otttnet_cifar10.yaml --name vgg9_ottt_otttnet_cifar10
vgg9-ottt-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_otttnet_svhn.yaml --name vgg9_ottt_otttnet_svhn
vgg9-ottt-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_otttnet_dvsgesture.yaml --name vgg9_ottt_otttnet_dvsgesture
vgg9-ottt-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/ottt_otttnet_dvscifar10.yaml --name vgg9_ottt_otttnet_dvscifar10

## BPTT trainer
vgg9-bptt-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_tpnet_cifar10.yaml --name vgg9_bptt_tpnet_cifar10
vgg9-bptt-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_tpnet_svhn.yaml --name vgg9_bptt_tpnet_svhn
vgg9-bptt-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_tpnet_dvsgesture.yaml --name vgg9_bptt_tpnet_dvsgesture
vgg9-bptt-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_tpnet_dvscifar10.yaml --name vgg9_bptt_tpnet_dvscifar10
vgg9-bptt-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_otttnet_cifar10.yaml --name vgg9_bptt_otttnet_cifar10
vgg9-bptt-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_otttnet_svhn.yaml --name vgg9_bptt_otttnet_svhn
vgg9-bptt-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_otttnet_dvsgesture.yaml --name vgg9_bptt_otttnet_dvsgesture
vgg9-bptt-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/vgg9/bptt_otttnet_dvscifar10.yaml --name vgg9_bptt_otttnet_dvscifar10

## Run the complete local VGG9 matrix (all 24 cells sequentially)
vgg9-matrix: \
	vgg9-tp-tpnet-cifar10 vgg9-tp-tpnet-svhn vgg9-tp-tpnet-dvsgesture vgg9-tp-tpnet-dvscifar10 \
	vgg9-tp-otttnet-cifar10 vgg9-tp-otttnet-svhn vgg9-tp-otttnet-dvsgesture vgg9-tp-otttnet-dvscifar10 \
	vgg9-ottt-tpnet-cifar10 vgg9-ottt-tpnet-svhn vgg9-ottt-tpnet-dvsgesture vgg9-ottt-tpnet-dvscifar10 \
	vgg9-ottt-otttnet-cifar10 vgg9-ottt-otttnet-svhn vgg9-ottt-otttnet-dvsgesture vgg9-ottt-otttnet-dvscifar10 \
	vgg9-bptt-tpnet-cifar10 vgg9-bptt-tpnet-svhn vgg9-bptt-tpnet-dvsgesture vgg9-bptt-tpnet-dvscifar10 \
	vgg9-bptt-otttnet-cifar10 vgg9-bptt-otttnet-svhn vgg9-bptt-otttnet-dvsgesture vgg9-bptt-otttnet-dvscifar10



# ── VGG9 Full Matrix — HPC Optuna sweep (sbatch) ───────────────────────────
# Uses config/vgg9/*_opt.yaml (50 trials × 10 epochs each).
# Edit runtime.epochs and optuna.n_trials in those files to change the budget.

## TP trainer
sbatch-vgg9-tp-tpnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_tp_tpnet_cifar10.sbatch
sbatch-vgg9-tp-tpnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_tp_tpnet_svhn.sbatch
sbatch-vgg9-tp-tpnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_tp_tpnet_dvsgesture.sbatch
sbatch-vgg9-tp-tpnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_tp_tpnet_dvscifar10.sbatch
sbatch-vgg9-tp-otttnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_tp_otttnet_cifar10.sbatch
sbatch-vgg9-tp-otttnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_tp_otttnet_svhn.sbatch
sbatch-vgg9-tp-otttnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_tp_otttnet_dvsgesture.sbatch
sbatch-vgg9-tp-otttnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_tp_otttnet_dvscifar10.sbatch

## OTTT trainer
sbatch-vgg9-ottt-tpnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_ottt_tpnet_cifar10.sbatch
sbatch-vgg9-ottt-tpnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_ottt_tpnet_svhn.sbatch
sbatch-vgg9-ottt-tpnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_ottt_tpnet_dvsgesture.sbatch
sbatch-vgg9-ottt-tpnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_ottt_tpnet_dvscifar10.sbatch
sbatch-vgg9-ottt-otttnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_ottt_otttnet_cifar10.sbatch
sbatch-vgg9-ottt-otttnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_ottt_otttnet_svhn.sbatch
sbatch-vgg9-ottt-otttnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_ottt_otttnet_dvsgesture.sbatch
sbatch-vgg9-ottt-otttnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_ottt_otttnet_dvscifar10.sbatch

## BPTT trainer
sbatch-vgg9-bptt-tpnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_bptt_tpnet_cifar10.sbatch
sbatch-vgg9-bptt-tpnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_bptt_tpnet_svhn.sbatch
sbatch-vgg9-bptt-tpnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_bptt_tpnet_dvsgesture.sbatch
sbatch-vgg9-bptt-tpnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_bptt_tpnet_dvscifar10.sbatch
sbatch-vgg9-bptt-otttnet-cifar10: hpc-mkdir
	sbatch hpc/vgg9_bptt_otttnet_cifar10.sbatch
sbatch-vgg9-bptt-otttnet-svhn: hpc-mkdir
	sbatch hpc/vgg9_bptt_otttnet_svhn.sbatch
sbatch-vgg9-bptt-otttnet-dvsgesture: hpc-mkdir
	sbatch hpc/vgg9_bptt_otttnet_dvsgesture.sbatch
sbatch-vgg9-bptt-otttnet-dvscifar10: hpc-mkdir
	sbatch hpc/vgg9_bptt_otttnet_dvscifar10.sbatch

## Submit the complete HPC VGG9 matrix (all 24 jobs)
opt-vgg9-matrix: \
	sbatch-vgg9-tp-tpnet-cifar10 sbatch-vgg9-tp-tpnet-svhn sbatch-vgg9-tp-tpnet-dvsgesture sbatch-vgg9-tp-tpnet-dvscifar10 \
	sbatch-vgg9-tp-otttnet-cifar10 sbatch-vgg9-tp-otttnet-svhn sbatch-vgg9-tp-otttnet-dvsgesture sbatch-vgg9-tp-otttnet-dvscifar10 \
	sbatch-vgg9-ottt-tpnet-cifar10 sbatch-vgg9-ottt-tpnet-svhn sbatch-vgg9-ottt-tpnet-dvsgesture sbatch-vgg9-ottt-tpnet-dvscifar10 \
	sbatch-vgg9-ottt-otttnet-cifar10 sbatch-vgg9-ottt-otttnet-svhn sbatch-vgg9-ottt-otttnet-dvsgesture sbatch-vgg9-ottt-otttnet-dvscifar10 \
	sbatch-vgg9-bptt-tpnet-cifar10 sbatch-vgg9-bptt-tpnet-svhn sbatch-vgg9-bptt-tpnet-dvsgesture sbatch-vgg9-bptt-tpnet-dvscifar10 \
	sbatch-vgg9-bptt-otttnet-cifar10 sbatch-vgg9-bptt-otttnet-svhn sbatch-vgg9-bptt-otttnet-dvsgesture sbatch-vgg9-bptt-otttnet-dvscifar10



# ── DEPRECATED VGG9 targets (kept for in-flight campaigns) ─────────────────
# These point at config/custom/ and config/benchmarking/ paths.
# Use the vgg9-* and sbatch-vgg9-* targets above for new experiments.

############# OTTT VGG9 Experiments #############
ottt-vgg9-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_cifar10.yaml --name ottt_vgg9_cifar10
ottt-vgg9-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_svhn.yaml --name ottt_vgg9_svhn
ottt-vgg9-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_dvsgesture.yaml --name ottt_vgg9_dvsgesture

############ TP VGG9 Experiments #############
tp-vgg9-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_cifar10.yaml --name tp_vgg9_cifar10
tp-vgg9-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_svhn.yaml --name tp_vgg9_svhn
tp-vgg9-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_dvsgesture.yaml --name tp_vgg9_dvsgesture
tp-vgg9-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_dvscifar10.yaml --name tp_vgg9_dvscifar10

########### BPTT VGG9 Experiments #############
bptt-vgg9-cifar10-ottt:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_cifar10_ottt.yaml --name bptt_vgg9_cifar10_ottt
bptt-vgg9-svhn-ottt:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_svhn_ottt.yaml --name bptt_vgg9_svhn_ottt
bptt-vgg9-dvsgesture-ottt:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_dvsgesture_ottt.yaml --name bptt_vgg9_dvsgesture_ottt
bptt-vgg9-cifar10-tp:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_cifar10_tp.yaml --name bptt_vgg9_cifar10_tp
bptt-vgg9-svhn-tp:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_svhn_tp.yaml --name bptt_vgg9_svhn_tp
bptt-vgg9-dvsgesture-tp:
	$(PYTHON) run_exp_campaign.py --custom config/custom/bptt_vgg9_dvsgesture_tp.yaml --name bptt_vgg9_dvsgesture_tp

## DEPRECATED aggregate — use vgg9-matrix instead
vgg9-all: ottt-vgg9-cifar10 ottt-vgg9-svhn ottt-vgg9-dvsgesture tp-vgg9-cifar10 tp-vgg9-svhn tp-vgg9-dvsgesture bptt-vgg9-cifar10-ottt bptt-vgg9-svhn-ottt bptt-vgg9-dvsgesture-ottt bptt-vgg9-cifar10-tp bptt-vgg9-svhn-tp bptt-vgg9-dvsgesture-tp

## DEPRECATED HPC targets — use sbatch-vgg9-* / opt-vgg9-matrix instead
opt-vgg9-all: sbatch-ottt-vgg9-cifar10 sbatch-ottt-vgg9-svhn sbatch-ottt-vgg9-dvsgesture sbatch-tp-vgg9-cifar10 sbatch-tp-vgg9-svhn sbatch-tp-vgg9-dvsgesture

sbatch-ottt-vgg9-cifar10: hpc-mkdir
	sbatch hpc/custom_ottt_vgg9_cifar10.sbatch

sbatch-ottt-vgg9-svhn: hpc-mkdir
	sbatch hpc/custom_ottt_vgg9_svhn.sbatch

sbatch-ottt-vgg9-dvsgesture: hpc-mkdir
	sbatch hpc/custom_ottt_vgg9_dvsgesture.sbatch

sbatch-tp-vgg9-cifar10: hpc-mkdir
	sbatch hpc/custom_tp_vgg9_cifar10.sbatch

sbatch-tp-vgg9-svhn: hpc-mkdir
	sbatch hpc/custom_tp_vgg9_svhn.sbatch

sbatch-tp-vgg9-dvsgesture: hpc-mkdir
	sbatch hpc/custom_tp_vgg9_dvsgesture.sbatch

## DEPRECATED — use opt-vgg9-matrix instead
opt-ottt-vgg9: hpc-mkdir
	sbatch hpc/bench_ottt_vgg9_cifar10.sbatch
	sbatch hpc/bench_ottt_vgg9_svhn.sbatch
	sbatch hpc/bench_ottt_vgg9_dvscifar10.sbatch
	sbatch hpc/bench_ottt_vgg9_dvsgesture.sbatch

## DEPRECATED — use opt-vgg9-matrix instead
opt-tp-vgg9: hpc-mkdir
	sbatch hpc/bench_tp_vgg9_cifar10.sbatch
	sbatch hpc/bench_tp_vgg9_svhn.sbatch
	sbatch hpc/bench_tp_vgg9_dvscifar10.sbatch
	sbatch hpc/bench_tp_vgg9_dvsgesture.sbatch
# ── END DEPRECATED ──────────────────────────────────────────────────────────



# ── Training runs — 300 epochs, no Optuna (HPC sbatch) ──────────────────────

## TP + VGG9 + DVS-CIFAR10 — 300 epochs
sbatch-train-tp-vgg9-dvscifar10: hpc-mkdir
	sbatch hpc/train_tp_vgg9_dvscifar10.sbatch

## TP + VGG9 + DVSGesture — 300 epochs
sbatch-train-tp-vgg9-dvsgesture: hpc-mkdir
	sbatch hpc/train_tp_vgg9_dvsgesture.sbatch

## OTTT + VGG9 + CIFAR10 — 300 epochs
sbatch-train-ottt-vgg9-cifar10: hpc-mkdir
	sbatch hpc/train_ottt_vgg9_cifar10.sbatch

## OTTT + VGG9 + SVHN — 300 epochs
sbatch-train-ottt-vgg9-svhn: hpc-mkdir
	sbatch hpc/train_ottt_vgg9_svhn.sbatch

## Submit all four training jobs
sbatch-train-vgg9-test: \
	sbatch-train-tp-vgg9-dvscifar10 \
	sbatch-train-tp-vgg9-dvsgesture \
	sbatch-train-ottt-vgg9-cifar10 \
	sbatch-train-ottt-vgg9-svhn



# ── HPC / SLURM targets ─────────────────────────────────────────────────────

## Ensure SLURM output directory exists
hpc-mkdir:
	mkdir -p $(HPC_SLURM_OUT)

## Per trainer experiments
all-opt: opt-bptt opt-decolle opt-ell opt-eprop opt-esd_rtrl opt-etlp opt-ostl opt-osttp opt-ottt opt-stsf opt-tp

opt-bptt: hpc-mkdir
	sbatch hpc/bench_bptt_mnist.sbatch
	sbatch hpc/bench_bptt_fmnist.sbatch
	sbatch hpc/bench_bptt_cifar10.sbatch
	sbatch hpc/bench_bptt_svhn.sbatch
	sbatch hpc/bench_bptt_nmnist.sbatch
	sbatch hpc/bench_bptt_dvsgesture.sbatch
	sbatch hpc/bench_bptt_dvscifar10.sbatch
opt-decolle: hpc-mkdir
	sbatch hpc/bench_decolle_mnist.sbatch
	sbatch hpc/bench_decolle_fmnist.sbatch
	sbatch hpc/bench_decolle_cifar10.sbatch
	sbatch hpc/bench_decolle_svhn.sbatch
	sbatch hpc/bench_decolle_nmnist.sbatch
	sbatch hpc/bench_decolle_dvsgesture.sbatch
	sbatch hpc/bench_decolle_dvscifar10.sbatch
opt-ell: hpc-mkdir
	sbatch hpc/bench_ell_mnist.sbatch
	sbatch hpc/bench_ell_fmnist.sbatch
	sbatch hpc/bench_ell_cifar10.sbatch
	sbatch hpc/bench_ell_svhn.sbatch
	sbatch hpc/bench_ell_nmnist.sbatch
	sbatch hpc/bench_ell_dvsgesture.sbatch
	sbatch hpc/bench_ell_dvscifar10.sbatch
opt-eprop: hpc-mkdir
	sbatch hpc/bench_eprop_mnist.sbatch
	sbatch hpc/bench_eprop_fmnist.sbatch
	sbatch hpc/bench_eprop_cifar10.sbatch
	sbatch hpc/bench_eprop_svhn.sbatch
	sbatch hpc/bench_eprop_nmnist.sbatch
	sbatch hpc/bench_eprop_dvsgesture.sbatch
	sbatch hpc/bench_eprop_dvscifar10.sbatch
opt-esd_rtrl: hpc-mkdir
	sbatch hpc/bench_esd_rtrl_mnist.sbatch
	sbatch hpc/bench_esd_rtrl_fmnist.sbatch
	sbatch hpc/bench_esd_rtrl_cifar10.sbatch
	sbatch hpc/bench_esd_rtrl_svhn.sbatch
	sbatch hpc/bench_esd_rtrl_nmnist.sbatch
	sbatch hpc/bench_esd_rtrl_dvsgesture.sbatch
	sbatch hpc/bench_esd_rtrl_dvscifar10.sbatch
opt-etlp: hpc-mkdir
	sbatch hpc/bench_etlp_mnist.sbatch
	sbatch hpc/bench_etlp_fmnist.sbatch
	sbatch hpc/bench_etlp_cifar10.sbatch
	sbatch hpc/bench_etlp_svhn.sbatch
	sbatch hpc/bench_etlp_nmnist.sbatch
	sbatch hpc/bench_etlp_dvsgesture.sbatch
	sbatch hpc/bench_etlp_dvscifar10.sbatch
opt-ostl: hpc-mkdir
	sbatch hpc/bench_ostl_mnist.sbatch
	sbatch hpc/bench_ostl_fmnist.sbatch
	sbatch hpc/bench_ostl_cifar10.sbatch
	sbatch hpc/bench_ostl_svhn.sbatch
	sbatch hpc/bench_ostl_nmnist.sbatch
	sbatch hpc/bench_ostl_dvsgesture.sbatch
	sbatch hpc/bench_ostl_dvscifar10.sbatch
opt-osttp: hpc-mkdir
	sbatch hpc/bench_osttp_mnist.sbatch
	sbatch hpc/bench_osttp_fmnist.sbatch
	sbatch hpc/bench_osttp_cifar10.sbatch
	sbatch hpc/bench_osttp_svhn.sbatch
	sbatch hpc/bench_osttp_nmnist.sbatch
	sbatch hpc/bench_osttp_dvsgesture.sbatch
	sbatch hpc/bench_osttp_dvscifar10.sbatch
opt-ottt: hpc-mkdir
	sbatch hpc/bench_ottt_mnist.sbatch
	sbatch hpc/bench_ottt_fmnist.sbatch
	sbatch hpc/bench_ottt_cifar10.sbatch
	sbatch hpc/bench_ottt_svhn.sbatch
	sbatch hpc/bench_ottt_nmnist.sbatch
	sbatch hpc/bench_ottt_dvsgesture.sbatch
	sbatch hpc/bench_ottt_dvscifar10.sbatch
opt-stsf: hpc-mkdir
	sbatch hpc/bench_stsf_mnist.sbatch
	sbatch hpc/bench_stsf_fmnist.sbatch
	sbatch hpc/bench_stsf_cifar10.sbatch
	sbatch hpc/bench_stsf_svhn.sbatch
	sbatch hpc/bench_stsf_nmnist.sbatch
	sbatch hpc/bench_stsf_dvsgesture.sbatch
	sbatch hpc/bench_stsf_dvscifar10.sbatch
opt-tp: hpc-mkdir
	sbatch hpc/bench_tp_mnist.sbatch
	sbatch hpc/bench_tp_fmnist.sbatch
	sbatch hpc/bench_tp_cifar10.sbatch
	sbatch hpc/bench_tp_svhn.sbatch
	sbatch hpc/bench_tp_nmnist.sbatch
	sbatch hpc/bench_tp_dvsgesture.sbatch
	sbatch hpc/bench_tp_dvscifar10.sbatch



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
