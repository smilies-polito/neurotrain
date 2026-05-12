"""
Optuna integration for the SNN benchmarking campaign.

Centralised search-space walker and study runner.  No trainer/network/dataset
class needs to know about Optuna — the walker resolves tunable blocks in the
config dict before constructors are called.

Tunable block schema (used in YAML configs):
    lr:
        value: 1.0e-3       # used in normal runs; Optuna starting point
        type: float         # float | int | categorical | null (= not tunable)
        min: 1.0e-5         # required for float and int
        max: 1.0e-1         # required for float and int
        log: true           # optional (float only); log-scale sampling
        step: 1             # optional (int only)
        list: [a, b, c]     # required for categorical

When type is null or the block has no type key, `value` is returned unchanged.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

if TYPE_CHECKING:
    import optuna

from campaign.config_loader import is_tunable_block

log = logging.getLogger(__name__)


# ── Categorical choice serialization ───────────────────────────────────────
# Optuna's CategoricalDistribution only accepts None/bool/int/float/str.
# List-valued choices (e.g. hidden_sizes=[256]) are JSON-encoded to strings
# before sampling and decoded back to lists afterwards.

def _enc_choice(v: object) -> object:
    """Encode a choice value: lists/tuples → JSON string, others unchanged."""
    return json.dumps(v) if isinstance(v, (list, tuple)) else v


def _dec_choice(v: object) -> object:
    """Decode a sampled value: JSON strings that are lists → list, others unchanged."""
    if isinstance(v, str):
        try:
            decoded = json.loads(v)
            if isinstance(decoded, list):
                return decoded
        except (json.JSONDecodeError, ValueError):
            pass
    return v


# ── Search-space walker ────────────────────────────────────────────────────


def suggest_from_cfg(cfg: dict, trial: "optuna.Trial", prefix: str = "") -> dict:
    """
    Return a new config dict where every tunable block is replaced by a
    trial.suggest_* value.

    Optuna parameter names are dotted YAML paths (e.g. 'trainer.lr') so they
    are unique across the trainer/model/dataset sub-dicts.

    Recurses into nested dicts.  Does NOT recurse into lists — list-valued
    attributes (e.g. hidden_sizes: [256]) stay literal unless wrapped as a
    categorical block: {type: categorical, list: [[256], [512, 256]]}.
    """
    result = {}
    for k, v in cfg.items():
        param_name = f"{prefix}.{k}" if prefix else k
        if is_tunable_block(v):
            result[k] = _suggest_one(trial, param_name, v)
        elif isinstance(v, dict):
            result[k] = suggest_from_cfg(v, trial, param_name)
        else:
            result[k] = v
    return result


def _suggest_one(trial: "optuna.Trial", name: str, block: dict) -> object:
    """Dispatch a single tunable block to the appropriate trial.suggest_* call."""
    kind = block.get("type")

    if kind is None:
        return block["value"]

    if kind == "float":
        return trial.suggest_float(
            name,
            float(block["min"]),
            float(block["max"]),
            log=bool(block.get("log", False)),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(block["min"]),
            int(block["max"]),
            step=int(block.get("step", 1)),
        )
    if kind == "categorical":
        choices = block["list"]
        if not choices:
            raise ValueError(f"Categorical param '{name}' has an empty list.")
        serialized = [_enc_choice(c) for c in choices]
        return _dec_choice(trial.suggest_categorical(name, serialized))

    raise ValueError(
        f"Unknown Optuna type '{kind}' for param '{name}'. "
        "Expected: float | int | categorical | null."
    )


# ── Study runner ───────────────────────────────────────────────────────────

_DEFAULT_OPTUNA_CFG: dict = {
    "n_trials": 20,
    "direction": "maximize",
    "sampler": "tpe",
    "pruner": None,
    "timeout": None,
    "storage": None,
}


def run_study(
    spec_name: str,
    optuna_cfg: dict,
    out_dir: Path,
    objective: Callable[["optuna.Trial"], float],
) -> "optuna.Study":
    """
    Create (or load) an Optuna study, run it, and write artifacts.

    Artifacts are written to out_dir/optuna/:
        trials.csv       — all trials with params and objective value
        best_params.yaml — hyper-parameters of the best trial
        study.db         — SQLite DB usable with optuna-dashboard

    The DB is written to /tmp during the study to avoid NFS/Lustre file-locking
    issues common on HPC clusters, then copied to out_dir/optuna/study.db on
    completion.  Set storage: sqlite:///... in the YAML config to override.

    Args:
        spec_name:   human-readable experiment name (used for the study name).
        optuna_cfg:  dict with n_trials, direction, sampler, pruner, timeout,
                     storage.  Missing keys fall back to _DEFAULT_OPTUNA_CFG.
        out_dir:     experiment output directory.
        objective:   function (trial) -> float to maximise/minimise.
    """
    import shutil
    import uuid

    import optuna as op

    cfg = {**_DEFAULT_OPTUNA_CFG, **optuna_cfg}

    sampler   = _build_sampler(cfg["sampler"], cfg.get("seed"))
    pruner    = _build_pruner(cfg["pruner"])
    direction = cfg["direction"]

    # Use /tmp for the live DB to avoid NFS/Lustre locking issues; copy to
    # out_dir once the study finishes.  A user-supplied storage: value bypasses
    # this and is used directly.
    user_storage = cfg["storage"]
    if user_storage:
        tmp_db  = None
        storage = user_storage
    else:
        tmp_db  = Path(f"/tmp/optuna_{spec_name}_{uuid.uuid4().hex[:8]}.db")
        storage = f"sqlite:///{tmp_db}"

    study = op.create_study(
        study_name=spec_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    log.info(
        "Starting Optuna study '%s': %d trials, direction=%s",
        spec_name, cfg["n_trials"], direction,
    )

    try:
        study.optimize(
            objective,
            n_trials=cfg["n_trials"],
            timeout=cfg["timeout"],
            gc_after_trial=True,  # force gc.collect() between trials to prevent GPU memory accumulation
        )
    finally:
        _write_artifacts(study, out_dir)
        if tmp_db and tmp_db.exists():
            dest = out_dir / "optuna" / "study.db"
            shutil.copy2(tmp_db, dest)
            tmp_db.unlink(missing_ok=True)
            log.info("Optuna DB written to: %s", dest)

    log.info(
        "Best trial #%d: value=%.4f  params=%s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )
    return study


def _build_sampler(name: str | None, seed: int | None) -> "optuna.samplers.BaseSampler":
    import optuna.samplers as s

    name = (name or "tpe").lower()
    if name == "tpe":
        return s.TPESampler(seed=seed)
    if name == "random":
        return s.RandomSampler(seed=seed)
    if name in ("cmaes", "cma-es"):
        return s.CmaEsSampler(seed=seed)
    raise ValueError(f"Unknown Optuna sampler '{name}'. Expected: tpe | random | cmaes.")


def _build_pruner(cfg: "str | dict | None") -> "optuna.pruners.BasePruner":
    """Build an Optuna pruner from a config value.

    Accepts three forms:
      * ``null`` / ``None``  — no pruning (NopPruner).
      * A plain string name  — e.g. ``"median"``, ``"hyperband"``, ``"threshold"``.
      * A dict               — must contain a ``type`` key (same names as above)
                               plus optional keyword arguments forwarded to the
                               pruner constructor.  Example::

                                   pruner:
                                     type: threshold
                                     lower: 0.15
                                     n_warmup_steps: 1
    """
    import optuna.pruners as p

    # ------------------------------------------------------------------ dict
    if isinstance(cfg, dict):
        cfg = dict(cfg)  # copy so we can pop safely
        ptype = cfg.pop("type", None)
        if ptype is None:
            raise ValueError("Pruner dict config must include a 'type' key.")
        ptype = ptype.lower()
        if ptype == "median":
            return p.MedianPruner(**cfg)
        if ptype in ("hyperband", "hb"):
            return p.HyperbandPruner(**cfg)
        if ptype == "threshold":
            cfg.setdefault("lower", 0.20)
            cfg.setdefault("n_warmup_steps", 20)
            return p.ThresholdPruner(**cfg)
        if ptype == "none":
            return p.NopPruner()
        raise ValueError(
            f"Unknown Optuna pruner type '{ptype}' in dict config. "
            "Expected: median | hyperband | threshold | none."
        )

    # ----------------------------------------------------------------- None
    # When name is None (i.e. pruner: null in config) we must explicitly use
    # NopPruner.  Passing pruner=None to create_study() makes Optuna fall back
    # to its default MedianPruner, which would prune trials even when the user
    # has disabled pruning.
    if cfg is None:
        return p.NopPruner()

    # ---------------------------------------------------------------- string
    name = cfg.lower()
    if name == "median":
        return p.MedianPruner()
    if name in ("hyperband", "hb"):
        return p.HyperbandPruner()
    if name == "threshold":
        # Legacy default: kill trials that have not exceeded 20% test accuracy
        # by epoch 20.  Use the dict form for full control.
        return p.ThresholdPruner(lower=0.20, n_warmup_steps=20)
    if name == "none":
        return p.NopPruner()
    raise ValueError(
        f"Unknown Optuna pruner '{name}'. "
        "Expected: median | hyperband | threshold | null, "
        "or a dict with 'type' plus pruner kwargs."
    )


def _write_artifacts(study: "optuna.Study", out_dir: Path) -> None:
    """Write trials.csv and best_params.yaml to out_dir/optuna/."""
    opt_dir = out_dir / "optuna"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # trials.csv — all trials
    trials = study.trials
    if trials:
        all_params = sorted({k for t in trials for k in t.params})
        csv_path = opt_dir / "trials.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["trial", "value", "state"] + all_params,
            )
            writer.writeheader()
            for t in trials:
                row = {
                    "trial": t.number,
                    "value": t.value,
                    "state": t.state.name,
                }
                row.update({p: t.params.get(p, "") for p in all_params})
                writer.writerow(row)
        log.info("Trials written to: %s", csv_path)

    # best_params.yaml
    best_path = opt_dir / "best_params.yaml"
    with open(best_path, "w") as f:
        yaml.dump(
            {"trial": study.best_trial.number, "value": study.best_value,
             "params": study.best_params},
            f, default_flow_style=False,
        )
    log.info("Best params written to: %s", best_path)
