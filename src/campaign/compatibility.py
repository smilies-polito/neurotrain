"""
Compatibility matrix for (trainer, model, dataset) triples.

This is the single source of truth for which combinations are valid.
Rules live in the YAML config files:
  - trainers/*.yaml  →  supported_net_types: [fc, recurrent, conv, ...]
  - models/*.yaml    →  net_type: fc | recurrent | conv | ...  (in the default: section)
  - datasets/*.yaml  →  supported_net_types: [fc, recurrent, conv, ...]

A triple is valid when the model's net_type is listed in both the trainer's
and the dataset's supported_net_types. If any of these fields is absent the
combination is allowed (backwards-compatible with unconfigured components).
"""

from functools import lru_cache

from campaign.config_loader import load_default, resolve_model_for_dataset


@lru_cache(maxsize=None)
def _load(kind: str, name: str) -> dict:
    """Load a default config, returning an empty dict if the file is missing."""
    try:
        return load_default(kind, name)
    except FileNotFoundError:
        return {}


def is_valid(trainer: str, model: str, dataset: str) -> bool:
    """
    Return True if the (trainer, model, dataset) triple is compatible.

    Args:
        trainer: Trainer name, e.g. "bptt".
        model:   Model name, e.g. "fc_snn".
        dataset: Dataset name, e.g. "MNIST".

    Returns:
        True if the combination can be instantiated.
    """
    trainer_cfg = _load("trainers", trainer.lower())
    model_raw   = _load("models",   model.lower())
    dataset_cfg = _load("datasets", dataset.lower())

    # Resolve the model's default section (ignores dataset-specific overrides)
    model_cfg = resolve_model_for_dataset(model_raw, "")

    model_type      = model_cfg.get("net_type", "").lower()
    trainer_types   = {t.lower() for t in trainer_cfg.get("supported_net_types", [])}
    dataset_types   = {t.lower() for t in dataset_cfg.get("supported_net_types", [])}

    # If any field is unset, skip the check for that component (allow the combo)
    if not model_type or not trainer_types or not dataset_types:
        return True

    return model_type in trainer_types and model_type in dataset_types


def filter_combinations(
    trainers: list[str],
    models: list[str],
    datasets: list[str],
) -> list[tuple[str, str, str]]:
    """
    Return all valid (trainer, model, dataset) triples from the cartesian product.

    Invalid combinations are silently skipped. The caller is responsible for
    logging which combinations were dropped.
    """
    valid = []
    for t in trainers:
        for m in models:
            for d in datasets:
                if is_valid(t, m, d):
                    valid.append((t, m, d))
    return valid


def skipped_combinations(
    trainers: list[str],
    models: list[str],
    datasets: list[str],
) -> list[tuple[str, str, str]]:
    """Return all invalid (trainer, model, dataset) triples."""
    return [
        (t, m, d)
        for t in trainers
        for m in models
        for d in datasets
        if not is_valid(t, m, d)
    ]
