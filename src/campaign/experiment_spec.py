"""
ExperimentSpec — the single unit passed through the campaign pipeline.

Each spec holds a fully resolved configuration (defaults merged with overrides)
for one training run: trainer + model + dataset + runtime settings.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperimentSpec:
    """Fully resolved config for one experiment run."""

    # Human-readable identifier, e.g. "bptt_fc_snn_mnist"
    name: str

    # If True, tunable attributes in the config dicts contain Optuna suggestion
    # metadata ({value, type, min, max, list}). Currently wired but not driven
    # by a study; set to False for normal runs.
    opt: bool

    # Resolved trainer config — keys match the trainer __init__ kwargs.
    # Must include "name" (e.g. "bptt") to look up the class in TRAINER_REGISTRY.
    trainer: dict = field(default_factory=dict)

    # Resolved model config — keys match get_network() kwargs.
    # Must include "name" (e.g. "fc_snn") and "algorithm" (trainer name).
    model: dict = field(default_factory=dict)

    # Resolved dataset config — keys match get_loader() kwargs.
    # Must include "name" (e.g. "MNIST").
    dataset: dict = field(default_factory=dict)

    # Runtime settings shared across all three components.
    runtime: dict = field(default_factory=lambda: {
        "epochs": 10,
        "device": "cuda",
        "seed": 42,
        "log_level": "INFO",
        "neurobench": True,
    })

    # Optuna study settings (only used when opt=True).
    # Keys: n_trials, direction, sampler, pruner, timeout, storage, seed.
    optuna: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization helpers (used to pass specs across subprocesses)       #
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ExperimentSpec":
        return cls(**json.loads(s))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentSpec":
        return cls.from_json(Path(path).read_text())
