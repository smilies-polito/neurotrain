"""Reproducibility-oriented reference network variants."""

from networks.reproducibility.DRTP_convolutional_network import DRTPConvMNIST
from networks.reproducibility.ottt_vgg_sws_snntorch import OTTTVGGSWS_SNNtorch

__all__ = ["DRTPConvMNIST", "OTTTVGGSWS_SNNtorch"]
