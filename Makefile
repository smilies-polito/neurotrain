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
## Dry-run: print experiment list without running
dry-bench:
	$(PYTHON) run_exp_campaign.py --benchmarking $(BENCH_CONFIG) --dry-run


## Run custom experiments
custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) $(if $(EXP_NAME),--name $(EXP_NAME),)
## Dry-run: print experiment list without running
dry-custom:
	$(PYTHON) run_exp_campaign.py --custom $(CUSTOM_CONFIG) --dry-run

bench-tp-vgg9-cifar10:
	TP_DEBUG_DIR="./debug_output" $(PYTHON) run_exp_campaign.py --benchmarking config/benchmarking/tp_vgg9_cifar10.yaml



# ── Custom Experiments ──────────────────────────────────────────────────────

############# OTTT VGG9 Experiments #############
ottt-vgg9-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_cifar10.yaml --name ottt_vgg9_cifar10
ottt-vgg9-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_svhn.yaml --name ottt_vgg9_svhn
ottt-vgg9-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/custom/ottt_vgg9_dvsgesture.yaml --name ottt_vgg9_dvsgesture

############ TP VGG9 Experiments #############
tp-vgg9-cifar10:
	TP_DEBUG_DIR="./debug_output" $(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_cifar10.yaml --name tp_vgg9_cifar10
tp-vgg9-svhn:
	TP_DEBUG_DIR="./debug_output" $(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_svhn.yaml --name tp_vgg9_svhn
tp-vgg9-dvsgesture:
	TP_DEBUG_DIR="./debug_output" $(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_dvsgesture.yaml --name tp_vgg9_dvsgesture
tp-vgg9-dvscifar10:
	TP_DEBUG_DIR="./debug_output" $(PYTHON) run_exp_campaign.py --custom config/custom/tp_vgg9_dvscifar10.yaml --name tp_vgg9_dvscifar10

########### BPTT VGG9 Experiments #############
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

## Run all VGG9 individual custom experiments
vgg9-all: ottt-vgg9-cifar10 ottt-vgg9-svhn ottt-vgg9-dvsgesture tp-vgg9-cifar10 tp-vgg9-svhn tp-vgg9-dvsgesture bptt-vgg9-cifar10-ottt bptt-vgg9-svhn-ottt bptt-vgg9-dvsgesture-ottt bptt-vgg9-cifar10-tp bptt-vgg9-svhn-tp bptt-vgg9-dvsgesture-tp



# ── HPC / SLURM targets ─────────────────────────────────────────────────────

## VGG9 experiments
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

opt-ottt-vgg9: hpc-mkdir
	sbatch hpc/bench_ottt_vgg9_cifar10.sbatch
	sbatch hpc/bench_ottt_vgg9_svhn.sbatch
	sbatch hpc/bench_ottt_vgg9_dvscifar10.sbatch
	sbatch hpc/bench_ottt_vgg9_dvsgesture.sbatch

opt-tp-vgg9: hpc-mkdir
	sbatch hpc/bench_tp_vgg9_cifar10.sbatch
	sbatch hpc/bench_tp_vgg9_svhn.sbatch
	sbatch hpc/bench_tp_vgg9_dvscifar10.sbatch
	sbatch hpc/bench_tp_vgg9_dvsgesture.sbatch



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