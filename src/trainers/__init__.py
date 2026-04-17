"""Trainer registry for SNN training.

To add a new trainer:
  1. Create src/trainers/my_trainer.py with a class inheriting BaseTrainer.
  2. Add two lines here:
       from trainers.my_trainer import MyTrainer
       TRAINER_REGISTRY["my_trainer"] = MyTrainer
"""

from trainers.base_trainer import BaseTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.stsf_trainer import STSFTrainer
from trainers.ostl_trainer import OSTLTrainer
from trainers.osttp_trainer import OSTTPTrainer
from trainers.ottt_trainer import OTTTTrainer
from trainers.eprop_trainer import EpropTrainer
from trainers.es_d_rtrl_trainer import ESDRTRLTrainer
from trainers.etlp_trainer import ETLPTrainer
from trainers.ell_trainer import ELLTrainer
from trainers.tp_trainer import TPTrainer
from trainers.decolle_trainer import DECOLLETrainer

TRAINER_REGISTRY: dict[str, type] = {
    "bptt":     BPTTTrainer,
    "stsf":     STSFTrainer,
    "ostl":     OSTLTrainer,
    "osttp":    OSTTPTrainer,
    "ottt":     OTTTTrainer,
    "eprop":    EpropTrainer,
    "esd_rtrl": ESDRTRLTrainer,
    "etlp":     ETLPTrainer,
    "ell":      ELLTrainer,
    "tp":       TPTrainer,
    "decolle":  DECOLLETrainer,
}

__all__ = ["BaseTrainer", "TRAINER_REGISTRY"]
