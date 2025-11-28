"""Tests for configuration system."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from utils.config import (
    Config,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TrainerConfig,
    DataConfig,
    HardwareConfig,
    CheckpointConfig,
    load_config,
    create_default_config,
    validate_config,
    merge_config_with_args,
    print_config,
)


class TestConfig:
    """Test Config dataclass."""

    def test_default_config_creation(self):
        """Test creating default config."""
        config = create_default_config()
        assert config.experiment.seed == 42
        assert config.training.epochs == 100
        assert config.trainer.name == "stsf"

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = Config()
        config_dict = config.to_dict()

        assert "experiment" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert config_dict["experiment"]["seed"] == 42

    def test_config_from_dict(self):
        """Test config creation from dict."""
        config_dict = {
            "experiment": {"name": "test_exp", "seed": 123},
            "training": {"epochs": 50},
        }
        config = Config.from_dict(config_dict)

        assert config.experiment.name == "test_exp"
        assert config.experiment.seed == 123
        assert config.training.epochs == 50

    def test_config_to_flat_dict(self):
        """Test flat dict conversion for logging."""
        config = Config()
        flat = config.to_flat_dict()

        assert "experiment.seed" in flat
        assert "training.epochs" in flat
        assert flat["experiment.seed"] == 42


class TestConfigIO:
    """Test config file I/O."""

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        config = Config()
        config.experiment.name = "yaml_test"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path, format="yaml")

            loaded = load_config(path)
            assert loaded.experiment.name == "yaml_test"
            assert loaded.experiment.seed == config.experiment.seed

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        config = Config()
        config.experiment.name = "json_test"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path, format="json")

            loaded = load_config(path)
            assert loaded.experiment.name == "json_test"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestConfigValidation:
    """Test config validation."""

    def test_valid_config(self):
        """Test validation of valid config."""
        config = Config()
        issues = validate_config(config)
        assert len(issues) == 0

    def test_invalid_layer_sizes(self):
        """Test validation catches invalid layer sizes."""
        config = Config()
        config.model.layer_sizes = [784]  # Only one element

        issues = validate_config(config)
        assert any("layer_sizes" in issue for issue in issues)

    def test_invalid_beta(self):
        """Test validation catches invalid beta."""
        config = Config()
        config.model.beta = 1.5  # Should be < 1

        issues = validate_config(config)
        assert any("beta" in issue for issue in issues)

    def test_invalid_epochs(self):
        """Test validation catches invalid epochs."""
        config = Config()
        config.training.epochs = -1

        issues = validate_config(config)
        assert any("epochs" in issue for issue in issues)

    def test_invalid_dataset(self):
        """Test validation catches invalid dataset."""
        config = Config()
        config.data.dataset = "INVALID_DATASET"

        issues = validate_config(config)
        assert any("dataset" in issue for issue in issues)

