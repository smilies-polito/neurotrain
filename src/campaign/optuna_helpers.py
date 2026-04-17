"""
Optuna hyperparameter optimization helpers.

An attribute in a YAML config can be defined in two ways:

  Plain value:
      lr: 1e-3

  Optuna-tunable attribute:
      lr:
          value: 1e-3      # default / initial value
          type: float      # suggest type: "float", "int", "categorical"
          min: 1e-5        # lower bound (float/int)
          max: 1e-1        # upper bound (float/int)
          list: null       # choices list (categorical)

The `resolve` function handles both forms and calls the right
trial.suggest_* method when a trial is active.
"""

from typing import Any, Optional

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


def is_tunable(attr: Any) -> bool:
    """Return True if attr is an Optuna attribute block (dict with 'value' and 'type')."""
    return isinstance(attr, dict) and "value" in attr and "type" in attr


def resolve(name: str, attr: Any, trial: Optional[Any] = None) -> Any:
    """
    Resolve an attribute to its final value.

    If attr is a plain value, return it directly.
    If attr is a tunable block and trial is provided, call the appropriate
    trial.suggest_* method. Otherwise, return attr["value"].

    Args:
        name:  Parameter name (used as Optuna parameter name).
        attr:  Plain value OR tunable dict {value, type, min, max, list}.
        trial: Optuna Trial object, or None for non-Optuna runs.

    Returns:
        The resolved parameter value.
    """
    if not is_tunable(attr):
        return attr

    # Plain default when not running a study
    if trial is None:
        return attr["value"]

    opt_type = str(attr.get("type", "")).lower()

    if opt_type == "categorical":
        choices = attr.get("list") or [attr["value"]]
        return trial.suggest_categorical(name, choices)

    if opt_type == "int":
        return trial.suggest_int(name, int(attr["min"]), int(attr["max"]))

    if opt_type == "float":
        log = bool(attr.get("log", False))
        return trial.suggest_float(
            name, float(attr["min"]), float(attr["max"]), log=log
        )

    # Unknown type — fall back to default value
    return attr["value"]


def resolve_all(cfg: dict, trial: Optional[Any] = None) -> dict:
    """
    Recursively resolve all tunable attributes in a config dict.

    Equivalent to calling resolve() on every leaf value.
    """
    result = {}
    for k, v in cfg.items():
        if is_tunable(v):
            result[k] = resolve(k, v, trial)
        elif isinstance(v, dict):
            result[k] = resolve_all(v, trial)
        else:
            result[k] = v
    return result
