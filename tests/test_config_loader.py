"""
Self-contained checks for config loading and merging logic.

Run directly:  python3 tests/test_config_loader.py
No external dependencies beyond PyYAML and the config/ directory.
"""

from pathlib import Path

import yaml

# ── Inlined implementations ────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_default(kind: str, name: str) -> dict:
    lower = name.lower()
    path = _CONFIG_DIR / "default" / kind / f"{lower}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No default config for {kind}/{name}. Expected: {path}"
        )
    return _load_yaml(path)


def list_defaults(kind: str) -> list[str]:
    folder = _CONFIG_DIR / "default" / kind
    return [p.stem for p in sorted(folder.glob("*.yaml"))]


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_model_for_dataset(model_cfg: dict, dataset_name: str) -> dict:
    if "default" not in model_cfg:
        return model_cfg
    base = dict(model_cfg["default"])
    dataset_key = dataset_name.lower()
    if dataset_key in model_cfg:
        base = merge(base, model_cfg[dataset_key])
    return base


def normalize_optuna_attrs(cfg: dict) -> dict:
    result = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and "value" in v and "type" in v:
            result[k] = v["value"]
        elif isinstance(v, dict):
            result[k] = normalize_optuna_attrs(v)
        else:
            result[k] = v
    return result


# ── Tests ──────────────────────────────────────────────────────────────────

def test_merge_flat():
    base     = {"lr": 1e-3, "epochs": 10}
    override = {"lr": 5e-4}
    result   = merge(base, override)
    assert result["lr"] == 5e-4
    assert result["epochs"] == 10


def test_merge_nested():
    base     = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 99}}
    result   = merge(base, override)
    assert result["a"]["x"] == 1
    assert result["a"]["y"] == 99


def test_merge_does_not_mutate():
    base     = {"lr": 1e-3}
    override = {"lr": 5e-4}
    merge(base, override)
    assert base["lr"] == 1e-3


def test_resolve_flat_config():
    cfg = {"hidden_sizes": [256], "beta": 0.9}
    assert resolve_model_for_dataset(cfg, "MNIST") == cfg


def test_resolve_uses_default_when_no_dataset_section():
    cfg = {"default": {"hidden_sizes": [256], "beta": 0.9}}
    result = resolve_model_for_dataset(cfg, "MNIST")
    assert result["hidden_sizes"] == [256]
    assert result["beta"] == 0.9


def test_resolve_applies_dataset_override():
    cfg = {
        "default": {"hidden_sizes": [256], "beta": 0.9},
        "mnist":   {"hidden_sizes": [128]},
    }
    result = resolve_model_for_dataset(cfg, "MNIST")
    assert result["hidden_sizes"] == [128]
    assert result["beta"] == 0.9


def test_resolve_missing_dataset_falls_back_to_default():
    cfg = {
        "default":     {"hidden_sizes": [256], "beta": 0.9},
        "fashionmnist": {"hidden_sizes": [800]},
    }
    result = resolve_model_for_dataset(cfg, "CIFAR10")
    assert result["hidden_sizes"] == [256]


def test_normalize_plain_values():
    cfg = {"lr": 1e-3, "epochs": 10}
    assert normalize_optuna_attrs(cfg) == cfg


def test_normalize_optuna_block():
    cfg = {"lr": {"value": 1e-3, "type": "float", "min": 1e-5, "max": 1e-1}}
    assert normalize_optuna_attrs(cfg)["lr"] == 1e-3


def test_normalize_nested():
    cfg = {
        "trainer": {
            "lr": {"value": 1e-3, "type": "float", "min": 1e-5, "max": 1e-1},
            "batch_size": 256,
        }
    }
    result = normalize_optuna_attrs(cfg)
    assert result["trainer"]["lr"] == 1e-3
    assert result["trainer"]["batch_size"] == 256


def test_load_default_bptt():
    cfg = load_default("trainers", "bptt")
    assert cfg.get("name") == "bptt"
    assert "lr" in cfg


def test_load_default_fc_snn():
    cfg = load_default("models", "fc_snn")
    assert "default" in cfg


def test_load_default_mnist():
    cfg = load_default("datasets", "mnist")
    assert cfg.get("name") == "MNIST"
    assert "T" in cfg


def test_load_default_missing():
    try:
        load_default("trainers", "nonexistent_trainer")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_list_defaults_trainers():
    names = list_defaults("trainers")
    assert "bptt" in names
    assert "stsf" in names


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(failed)
