"""
Compatibility matrix for (trainer, model, dataset) triples.

This is the single source of truth for which combinations are valid.
The logic mirrors the algorithm/model constraints in src/networks/get_network.py
but lives here so the campaign builder can filter before instantiating anything.
"""

# Trainers that require a recurrent network (RSNN)
_RECURRENT_TRAINERS = {"eprop", "esd_rtrl"}

# Trainers that require the LocalClassifier network — not available in benchmarking mode
_LOCAL_CLASSIFIER_TRAINERS = {"ell"}

# Models that work with recurrent trainers
_RECURRENT_MODELS = {"r_snn"}

# Models that work with local-classifier trainers
_LOCAL_CLASSIFIER_MODELS = {"local_classifier"}

# Datasets not compatible with certain trainers (extend as needed)
_TRAINER_DATASET_BLACKLIST: dict[str, set[str]] = {
    # e.g. "decolle": {"CIFAR10"} — add as discovered
}

# Models that are convolutional and incompatible with flat datasets
_CONV_MODELS = {"conv_snn", "vg11_snn"}

# Datasets that produce flat (non-spatial) features — conv models need at least 2D
_FLAT_DATASETS = {"shd", "mackeyglass", "primateReaching", "wisdm"}


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
    t = trainer.lower()
    m = model.lower()
    d = dataset.lower()

    # Recurrent trainers need recurrent models
    if t in _RECURRENT_TRAINERS and m not in _RECURRENT_MODELS:
        return False

    # Local-classifier trainers need local-classifier models
    if t in _LOCAL_CLASSIFIER_TRAINERS and m not in _LOCAL_CLASSIFIER_MODELS:
        return False

    # Non-specialized trainers should not be paired with specialized models
    if t not in _RECURRENT_TRAINERS and m in _RECURRENT_MODELS:
        return False
    if t not in _LOCAL_CLASSIFIER_TRAINERS and m in _LOCAL_CLASSIFIER_MODELS:
        return False

    # Trainer-dataset blacklist
    if t in _TRAINER_DATASET_BLACKLIST and d in _TRAINER_DATASET_BLACKLIST[t]:
        return False

    # Convolutional models cannot handle flat/sequence datasets
    if m in _CONV_MODELS and d in _FLAT_DATASETS:
        return False

    return True


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
