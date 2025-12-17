"""Training algorithms for SNNs."""

from trainers.base_trainer import BaseTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.decolle_trainer import DECOLLETrainer
from trainers.eprop_trainer import EpropTrainer
from trainers.ottt_trainer import OTTTTrainer
from trainers.stsf_trainer import STSFTrainer

__all__ = [
    "BaseTrainer",
    "BPTTTrainer",
    "STSFTrainer",
    "BPTTTrainer",
    "EpropTrainer",
    "OTTTTrainer",
    "DECOLLETrainer",
]
