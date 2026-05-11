# Makefile — convenience targets for the SNN benchmarking framework.

PYTHON ?= python3

EXP_NAME      ?=
HPC_SLURM_OUT ?= hpc/slurm_outputs



# ▗▄▄▖  ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▄▄▖ 
# ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌
# ▐▛▀▘ ▐▛▀▜▌▐▛▀▘ ▐▛▀▀▘▐▛▀▚▖
# ▐▌   ▐▌ ▐▌▐▌   ▐▙▄▄▖▐▌ ▐▌
                         
## Run paper experiments 
paper-bptt:
	$(PYTHON) run_exp_campaign.py --custom config/paper/bptt.yaml --name paper_bptt
paper-decolle:
	$(PYTHON) run_exp_campaign.py --custom config/paper/decolle.yaml --name paper_decolle
paper-eprop:
	$(PYTHON) run_exp_campaign.py --custom config/paper/eprop.yaml --name paper_eprop
paper-esd_rtrl:
	$(PYTHON) run_exp_campaign.py --custom config/paper/esd_rtrl.yaml --name paper_esd_rtrl
paper-ell:
	$(PYTHON) run_exp_campaign.py --custom config/paper/ell.yaml --name paper_ell
paper-etlp:
	$(PYTHON) run_exp_campaign.py --custom config/paper/etlp.yaml --name paper_etlp
paper-ostl:
	$(PYTHON) run_exp_campaign.py --custom config/paper/ostl.yaml --name paper_ostl
paper-osttp:
	$(PYTHON) run_exp_campaign.py --custom config/paper/osttp.yaml --name paper_osttp
paper-ottt:
	$(PYTHON) run_exp_campaign.py --custom config/paper/ottt.yaml --name paper_ottt
paper-stsf:
	$(PYTHON) run_exp_campaign.py --custom config/paper/stsf.yaml --name paper_stsf
paper-tp:
	$(PYTHON) run_exp_campaign.py --custom config/paper/tp.yaml --name paper_tp

## Run paper aggregate VGG9 experiments
paper-vgg9-tp:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9_tp.yaml --name paper_vgg9_tp
paper-vgg9-ottt:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9_ottt.yaml --name paper_vgg9_ottt
paper-vgg9-bptt:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9_bptt.yaml --name paper_vgg9_bptt
paper-vgg9: paper-vgg9-tp paper-vgg9-ottt paper-vgg9-bptt



# ▗▄▄▖  ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▄▄▖     ▗▖  ▗▖ ▗▄▄▖ ▗▄▄▖     ▗▄▄▖▗▄▄▄▖▗▖ ▗▖▗▄▄▄▗▖  ▗▖
# ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌    ▐▌  ▐▌▐▌   ▐▌       ▐▌     █  ▐▌ ▐▌▐▌  █▝▚▞▘ 
# ▐▛▀▘ ▐▛▀▜▌▐▛▀▘ ▐▛▀▀▘▐▛▀▚▖    ▐▌  ▐▌▐▌▝▜▌▐▌▝▜▌     ▝▀▚▖  █  ▐▌ ▐▌▐▌  █ ▐▌  
# ▐▌   ▐▌ ▐▌▐▌   ▐▙▄▄▖▐▌ ▐▌     ▝▚▞▘ ▝▚▄▞▘▝▚▄▞▘    ▗▄▄▞▘  █  ▝▚▄▞▘▐▙▄▄▀ ▐▌  

# ----------- Local Execution -----------
# BPTT
paper-vgg9-bptt-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_tpnet_cifar10.yaml --name paper_vgg9_bptt_tpnet_cifar10
paper-vgg9-bptt-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_tpnet_dvsgesture.yaml --name paper_vgg9_bptt_tpnet_dvsgesture
paper-vgg9-bptt-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_tpnet_svhn.yaml --name paper_vgg9_bptt_tpnet_svhn
paper-vgg9-bptt-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_otttnet_svhn.yaml --name paper_vgg9_bptt_otttnet_svhn
# TP
paper-vgg9-tp-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_tpnet_cifar10.yaml --name paper_vgg9_tp_tpnet_cifar10
paper-vgg9-tp-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_tpnet_dvscifar10.yaml --name paper_vgg9_tp_tpnet_dvscifar10
paper-vgg9-tp-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_tpnet_dvsgesture.yaml --name paper_vgg9_tp_tpnet_dvsgesture
paper-vgg9-tp-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_tpnet_svhn.yaml --name paper_vgg9_tp_tpnet_svhn
paper-vgg9-tp-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_otttnet_cifar10.yaml --name paper_vgg9_tp_otttnet_cifar10
paper-vgg9-tp-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_otttnet_dvscifar10.yaml --name paper_vgg9_tp_otttnet_dvscifar10
paper-vgg9-tp-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_otttnet_dvsgesture.yaml --name paper_vgg9_tp_otttnet_dvsgesture
paper-vgg9-tp-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/tp_otttnet_svhn.yaml --name paper_vgg9_tp_otttnet_svhn
# OTTT
paper-vgg9-ottt-tpnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_tpnet_cifar10.yaml --name paper_vgg9_ottt_tpnet_cifar10
paper-vgg9-ottt-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_otttnet_cifar10.yaml --name paper_vgg9_ottt_otttnet_cifar10
paper-vgg9-ottt-otttnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_otttnet_svhn.yaml --name paper_vgg9_ottt_otttnet_svhn

## Paper VGG9 pending experiments — local (config TODOs: fill with best HPO results first)
paper-vgg9-bptt-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_tpnet_dvscifar10.yaml --name paper_vgg9_bptt_tpnet_dvscifar10
paper-vgg9-bptt-otttnet-cifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_otttnet_cifar10.yaml --name paper_vgg9_bptt_otttnet_cifar10
paper-vgg9-bptt-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_otttnet_dvsgesture.yaml --name paper_vgg9_bptt_otttnet_dvsgesture
paper-vgg9-bptt-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/bptt_otttnet_dvscifar10.yaml --name paper_vgg9_bptt_otttnet_dvscifar10
paper-vgg9-ottt-tpnet-svhn:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_tpnet_svhn.yaml --name paper_vgg9_ottt_tpnet_svhn
paper-vgg9-ottt-tpnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_tpnet_dvsgesture.yaml --name paper_vgg9_ottt_tpnet_dvsgesture
paper-vgg9-ottt-tpnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_tpnet_dvscifar10.yaml --name paper_vgg9_ottt_tpnet_dvscifar10
paper-vgg9-ottt-otttnet-dvsgesture:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_otttnet_dvsgesture.yaml --name paper_vgg9_ottt_otttnet_dvsgesture
paper-vgg9-ottt-otttnet-dvscifar10:
	$(PYTHON) run_exp_campaign.py --custom config/paper/vgg9/ottt_otttnet_dvscifar10.yaml --name paper_vgg9_ottt_otttnet_dvscifar10

# ----------- HPC Execution -----------
# BPTT
paper-vgg9-bptt-tpnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_tpnet_cifar10.sbatch
paper-vgg9-bptt-tpnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_tpnet_dvsgesture.sbatch
paper-vgg9-bptt-tpnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_tpnet_svhn.sbatch
paper-vgg9-bptt-otttnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_otttnet_svhn.sbatch
# TP
paper-vgg9-tp-tpnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_tpnet_cifar10.sbatch
paper-vgg9-tp-tpnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_tpnet_dvscifar10.sbatch
paper-vgg9-tp-tpnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_tpnet_dvsgesture.sbatch
paper-vgg9-tp-tpnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_tpnet_svhn.sbatch
paper-vgg9-tp-otttnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_otttnet_cifar10.sbatch
paper-vgg9-tp-otttnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_otttnet_dvscifar10.sbatch
paper-vgg9-tp-otttnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_otttnet_dvsgesture.sbatch
paper-vgg9-tp-otttnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_tp_otttnet_svhn.sbatch
# OTTT
paper-vgg9-ottt-tpnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_tpnet_cifar10.sbatch
paper-vgg9-ottt-otttnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_otttnet_cifar10.sbatch
paper-vgg9-ottt-otttnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_otttnet_svhn.sbatch

## Paper VGG9 pending experiments — HPC (fill configs first!)
paper-vgg9-bptt-tpnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_tpnet_dvscifar10.sbatch
paper-vgg9-bptt-otttnet-cifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_otttnet_cifar10.sbatch
paper-vgg9-bptt-otttnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_otttnet_dvsgesture.sbatch
paper-vgg9-bptt-otttnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_bptt_otttnet_dvscifar10.sbatch
paper-vgg9-ottt-tpnet-svhn-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_tpnet_svhn.sbatch
paper-vgg9-ottt-tpnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_tpnet_dvsgesture.sbatch
paper-vgg9-ottt-tpnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_tpnet_dvscifar10.sbatch
paper-vgg9-ottt-otttnet-dvsgesture-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_otttnet_dvsgesture.sbatch
paper-vgg9-ottt-otttnet-dvscifar10-hpc: hpc-mkdir
	sbatch hpc/paper_vgg9_ottt_otttnet_dvscifar10.sbatch

## Submit all 24 paper VGG9 jobs at once to HPC
paper-vgg9-all-hpc: hpc-mkdir \
	paper-vgg9-bptt-tpnet-cifar10-hpc paper-vgg9-bptt-tpnet-dvscifar10-hpc \
	paper-vgg9-bptt-tpnet-dvsgesture-hpc paper-vgg9-bptt-tpnet-svhn-hpc \
	paper-vgg9-bptt-otttnet-cifar10-hpc paper-vgg9-bptt-otttnet-dvscifar10-hpc \
	paper-vgg9-bptt-otttnet-dvsgesture-hpc paper-vgg9-bptt-otttnet-svhn-hpc \
	paper-vgg9-tp-tpnet-cifar10-hpc paper-vgg9-tp-tpnet-dvscifar10-hpc \
	paper-vgg9-tp-tpnet-dvsgesture-hpc paper-vgg9-tp-tpnet-svhn-hpc \
	paper-vgg9-tp-otttnet-cifar10-hpc paper-vgg9-tp-otttnet-dvscifar10-hpc \
	paper-vgg9-tp-otttnet-dvsgesture-hpc paper-vgg9-tp-otttnet-svhn-hpc \
	paper-vgg9-ottt-tpnet-cifar10-hpc paper-vgg9-ottt-tpnet-dvscifar10-hpc \
	paper-vgg9-ottt-tpnet-dvsgesture-hpc paper-vgg9-ottt-tpnet-svhn-hpc \
	paper-vgg9-ottt-otttnet-cifar10-hpc paper-vgg9-ottt-otttnet-dvscifar10-hpc \
	paper-vgg9-ottt-otttnet-dvsgesture-hpc paper-vgg9-ottt-otttnet-svhn-hpc




# ▗▖  ▗▖ ▗▄▄▖ ▗▄▄▖    ▗▖    ▗▄▖  ▗▄▄▖ ▗▄▖ ▗▖       ▗▄▄▄▖▗▖  ▗▖▗▄▄▖ 
# ▐▌  ▐▌▐▌   ▐▌       ▐▌   ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌       ▐▌    ▝▚▞▘ ▐▌ ▐▌
# ▐▌  ▐▌▐▌▝▜▌▐▌▝▜▌    ▐▌   ▐▌ ▐▌▐▌   ▐▛▀▜▌▐▌       ▐▛▀▀▘  ▐▌  ▐▛▀▘ 
#  ▝▚▞▘ ▝▚▄▞▘▝▚▄▞▘    ▐▙▄▄▖▝▚▄▞▘▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖    ▐▙▄▄▖▗▞▘▝▚▖▐▌   

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
vgg9-tp-all: \
	vgg9-tp-tpnet-cifar10 vgg9-tp-tpnet-svhn vgg9-tp-tpnet-dvsgesture vgg9-tp-tpnet-dvscifar10 \
	vgg9-tp-otttnet-cifar10 vgg9-tp-otttnet-svhn vgg9-tp-otttnet-dvsgesture vgg9-tp-otttnet-dvscifar10

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
vgg9-ottt-all: \
	vgg9-ottt-tpnet-cifar10 vgg9-ottt-tpnet-svhn vgg9-ottt-tpnet-dvsgesture vgg9-ottt-tpnet-dvscifar10 \
	vgg9-ottt-otttnet-cifar10 vgg9-ottt-otttnet-svhn vgg9-ottt-otttnet-dvsgesture vgg9-ottt-otttnet-dvscifar10

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
vgg9-bptt-all: \
	vgg9-bptt-tpnet-cifar10 vgg9-bptt-tpnet-svhn vgg9-bptt-tpnet-dvsgesture vgg9-bptt-tpnet-dvscifar10 \
	vgg9-bptt-otttnet-cifar10 vgg9-bptt-otttnet-svhn vgg9-bptt-otttnet-dvsgesture vgg9-bptt-otttnet-dvscifar10

## Run the complete local VGG9 matrix (all 24 cells sequentially)
vgg9-matrix: \
	vgg9-tp-tpnet-cifar10 vgg9-tp-tpnet-svhn vgg9-tp-tpnet-dvsgesture vgg9-tp-tpnet-dvscifar10 \
	vgg9-tp-otttnet-cifar10 vgg9-tp-otttnet-svhn vgg9-tp-otttnet-dvsgesture vgg9-tp-otttnet-dvscifar10 \
	vgg9-ottt-tpnet-cifar10 vgg9-ottt-tpnet-svhn vgg9-ottt-tpnet-dvsgesture vgg9-ottt-tpnet-dvscifar10 \
	vgg9-ottt-otttnet-cifar10 vgg9-ottt-otttnet-svhn vgg9-ottt-otttnet-dvsgesture vgg9-ottt-otttnet-dvscifar10 \
	vgg9-bptt-tpnet-cifar10 vgg9-bptt-tpnet-svhn vgg9-bptt-tpnet-dvsgesture vgg9-bptt-tpnet-dvscifar10 \
	vgg9-bptt-otttnet-cifar10 vgg9-bptt-otttnet-svhn vgg9-bptt-otttnet-dvsgesture vgg9-bptt-otttnet-dvscifar10



# ▗▖  ▗▖ ▗▄▄▖ ▗▄▄▖    ▗▖ ▗▖▗▄▄▖  ▗▄▄▖    ▗▄▄▄▖▗▖  ▗▖▗▄▄▖ 
# ▐▌  ▐▌▐▌   ▐▌       ▐▌ ▐▌▐▌ ▐▌▐▌       ▐▌    ▝▚▞▘ ▐▌ ▐▌
# ▐▌  ▐▌▐▌▝▜▌▐▌▝▜▌    ▐▛▀▜▌▐▛▀▘ ▐▌       ▐▛▀▀▘  ▐▌  ▐▛▀▘ 
#  ▝▚▞▘ ▝▚▄▞▘▝▚▄▞▘    ▐▌ ▐▌▐▌   ▝▚▄▄▖    ▐▙▄▄▖▗▞▘▝▚▖▐▌   

## Ensure SLURM output directory exists
hpc-mkdir:
	mkdir -p $(HPC_SLURM_OUT)

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
sbatch-vgg9-tp-all: \
	sbatch-vgg9-tp-tpnet-cifar10 sbatch-vgg9-tp-tpnet-svhn sbatch-vgg9-tp-tpnet-dvsgesture sbatch-vgg9-tp-tpnet-dvscifar10 \
	sbatch-vgg9-tp-otttnet-cifar10 sbatch-vgg9-tp-otttnet-svhn sbatch-vgg9-tp-otttnet-dvsgesture sbatch-vgg9-tp-otttnet-dvscifar10

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
sbatch-vgg9-ottt-all: \
	sbatch-vgg9-ottt-tpnet-cifar10 sbatch-vgg9-ottt-tpnet-svhn sbatch-vgg9-ottt-tpnet-dvsgesture sbatch-vgg9-ottt-tpnet-dvscifar10 \
	sbatch-vgg9-ottt-otttnet-cifar10 sbatch-vgg9-ottt-otttnet-svhn sbatch-vgg9-ottt-otttnet-dvsgesture sbatch-vgg9-ottt-otttnet-dvscifar10

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
sbatch-vgg9-bptt-all: \
	sbatch-vgg9-bptt-tpnet-cifar10 sbatch-vgg9-bptt-tpnet-svhn sbatch-vgg9-bptt-tpnet-dvsgesture sbatch-vgg9-bptt-tpnet-dvscifar10 \
	sbatch-vgg9-bptt-otttnet-cifar10 sbatch-vgg9-bptt-otttnet-svhn sbatch-vgg9-bptt-otttnet-dvsgesture sbatch-vgg9-bptt-otttnet-dvscifar10

## Submit the complete HPC VGG9 matrix (all 24 jobs)
sbatch-vgg9-matrix: \
	sbatch-vgg9-tp-tpnet-cifar10 sbatch-vgg9-tp-tpnet-svhn sbatch-vgg9-tp-tpnet-dvsgesture sbatch-vgg9-tp-tpnet-dvscifar10 \
	sbatch-vgg9-tp-otttnet-cifar10 sbatch-vgg9-tp-otttnet-svhn sbatch-vgg9-tp-otttnet-dvsgesture sbatch-vgg9-tp-otttnet-dvscifar10 \
	sbatch-vgg9-ottt-tpnet-cifar10 sbatch-vgg9-ottt-tpnet-svhn sbatch-vgg9-ottt-tpnet-dvsgesture sbatch-vgg9-ottt-tpnet-dvscifar10 \
	sbatch-vgg9-ottt-otttnet-cifar10 sbatch-vgg9-ottt-otttnet-svhn sbatch-vgg9-ottt-otttnet-dvsgesture sbatch-vgg9-ottt-otttnet-dvscifar10 \
	sbatch-vgg9-bptt-tpnet-cifar10 sbatch-vgg9-bptt-tpnet-svhn sbatch-vgg9-bptt-tpnet-dvsgesture sbatch-vgg9-bptt-tpnet-dvscifar10 \
	sbatch-vgg9-bptt-otttnet-cifar10 sbatch-vgg9-bptt-otttnet-svhn sbatch-vgg9-bptt-otttnet-dvsgesture sbatch-vgg9-bptt-otttnet-dvscifar10



# ▗▖ ▗▖▗▄▄▖  ▗▄▖      ▗▄▄▖ ▗▄▖ ▗▖  ▗▖▗▄▄▖  ▗▄▖ ▗▄▄▄▖ ▗▄▄▖▗▖  ▗▖ ▗▄▄▖
# ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌    ▐▌   ▐▌ ▐▌▐▛▚▞▜▌▐▌ ▐▌▐▌ ▐▌  █  ▐▌   ▐▛▚▖▐▌▐▌   
# ▐▛▀▜▌▐▛▀▘ ▐▌ ▐▌    ▐▌   ▐▛▀▜▌▐▌  ▐▌▐▛▀▘ ▐▛▀▜▌  █  ▐▌▝▜▌▐▌ ▝▜▌ ▝▀▚▖
# ▐▌ ▐▌▐▌   ▝▚▄▞▘    ▝▚▄▄▖▐▌ ▐▌▐▌  ▐▌▐▌   ▐▌ ▐▌▗▄█▄▖▝▚▄▞▘▐▌  ▐▌▗▄▄▞▘

## Per trainer experiments
all-opt: opt-bptt opt-decolle opt-eprop opt-esd_rtrl opt-etlp opt-ostl opt-osttp opt-ottt opt-stsf opt-tp

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

opt-shd: hpc-mkdir
	sbatch hpc/bench_shd.sbatch

#  ▗▄▄▖▗▖   ▗▄▄▄▖ ▗▄▖ ▗▖  ▗▖▗▖ ▗▖▗▄▄▖ 
# ▐▌   ▐▌   ▐▌   ▐▌ ▐▌▐▛▚▖▐▌▐▌ ▐▌▐▌ ▐▌
# ▐▌   ▐▌   ▐▛▀▀▘▐▛▀▜▌▐▌ ▝▜▌▐▌ ▐▌▐▛▀▘ 
# ▝▚▄▄▖▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌▐▌  ▐▌▝▚▄▞▘▐▌   

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
