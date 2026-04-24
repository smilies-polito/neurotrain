"""
Config loading and merging utilities.

Responsibilities:
- Load YAML files from config/default/{trainers,models,datasets}/
- Deep-merge base configs with user overrides (override wins)
- Apply per-dataset model overrides (the fallback section logic)
- Flatten Optuna-style attribute dicts to plain values for normal runs
"""

from pathlib import Path

import yaml

# Path to the repo-level config/ directory
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_default(kind: str, name: str) -> dict:
    """
    Load config/default/{kind}/{name}.yaml.

    Args:
        kind: One of "trainers", "models", "datasets".
        name: Component name, e.g. "bptt", "fc_snn", "mnist".

    Returns:
        The raw YAML dict (may contain per-dataset sections for models).

    Raises:
        FileNotFoundError: If no default config exists for this component.
    """
    # Accept both exact names ("mnist") and case variants ("MNIST")
    lower = name.lower()
    path = _CONFIG_DIR / "default" / kind / f"{lower}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No default config for {kind}/{name}. Expected: {path}"
        )
    return load_yaml(path)


def list_defaults(kind: str) -> list[str]:
    """Return all names available in config/default/{kind}/."""
    folder = _CONFIG_DIR / "default" / kind
    return [p.stem for p in sorted(folder.glob("*.yaml"))]


def merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dicts. Values in override win.

    Nested dicts are merged recursively; all other types are replaced.
    """
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_model_for_trainer_and_dataset(
    model_cfg: dict, trainer_name: str, dataset_name: str,
) -> dict:
    """
    Apply per-dataset then per-trainer model overrides.

    Merge order: default → dataset section → trainer section.
    Dataset section sets spatial shapes and topology; trainer section applies
    algorithm-specific overrides on top (head_type, surrogate, conv_gain, etc.)
    and has the final word — this lets trainers like eprop constrain the model
    regardless of dataset.

    A model YAML can have this structure:
        default:
            beta: 0.5
        tp:
            head_type: leaky_integrator
            conv_gain: 1.8
        cifar10:
            in_channels: 3
            input_shape: [3, 32, 32]

    If no 'default' key exists, the config is returned as-is.
    """
    if "default" not in model_cfg:
        return model_cfg

    base = dict(model_cfg["default"])
    t_key = trainer_name.lower()
    d_key = dataset_name.lower()

    if d_key in model_cfg:
        base = merge(base, model_cfg[d_key])
    if t_key in model_cfg:
        base = merge(base, model_cfg[t_key])

    return base


def resolve_model_for_dataset(model_cfg: dict, dataset_name: str) -> dict:
    """
    Apply per-dataset model overrides.

    A model YAML can have this structure:
        default:
            hidden_sizes: [256]
            beta: 0.9
        mnist:
            hidden_sizes: [128]
        fmnist:
            hidden_sizes: [800]

    This function returns the 'default' section merged with the dataset-specific
    section (if present). If no 'default' key exists, the config is returned as-is
    (it is already a flat dict without per-dataset sections).
    """
    if "default" not in model_cfg:
        return model_cfg

    base = dict(model_cfg["default"])
    dataset_key = dataset_name.lower()

    if dataset_key in model_cfg:
        base = merge(base, model_cfg[dataset_key])

    return base


def is_tunable_block(v: object) -> bool:
    """Return True if v is an Optuna-style attribute block ({value, type, ...})."""
    return isinstance(v, dict) and "value" in v and "type" in v


def normalize_optuna_attrs(cfg: dict) -> dict:
    """
    Flatten Optuna-style attribute dicts to their plain values.

    When an attribute is defined with hyperparameter metadata:
        lr:
            value: 1e-3
            type: float
            min: 1e-5
            max: 1e-1

    This function replaces such dicts with just the plain value (1e-3).
    Call this for all normal (non-Optuna) runs.
    """
    result = {}
    for k, v in cfg.items():
        if is_tunable_block(v):
            result[k] = v["value"]
        elif isinstance(v, dict):
            result[k] = normalize_optuna_attrs(v)
        else:
            result[k] = v
    return result
