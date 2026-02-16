"""Tests for configuration system."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from utils.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    TrainerConfig,
    TrainingConfig,
    create_default_config,
    load_config,
    merge_config_with_args,
    print_config,
    validate_config,
)


class TestConfig:
    """Test Config dataclass."""

    def test_default_config_creation(self):
        """Test creating default config."""
        config = create_default_config()
        assert config.experiment.seed == 42
        assert config.training.epochs == 100
        assert config.trainer.name == "stsf"
        assert config.model.recurrent_type == "standard"

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

    def test_default_drtp_config(self):
        """Test DRTP config defaults."""
        config = Config()
        assert config.drtp.loss == "mse"
        assert config.drtp.output_mode == "mem"
        assert config.drtp.surrogate_scale == 5.0
        assert config.drtp.surrogate_type == "logistic"
        assert config.drtp.feedback_distribution == "kaiming_uniform"
        assert config.drtp.feedback_scale == 1.0
        assert config.drtp.fixed_feedback is True

    def test_default_ostl_config(self):
        """Test OSTL config defaults."""
        config = Config()
        assert config.ostl.surrogate_scale == 5.0
        assert config.ostl.grad_clip == 0.0
        assert config.ostl.output_mode == "spike"

    def test_default_stop_config(self):
        """Test STOP config defaults."""
        config = Config()
        assert config.stop.loss == "ce"
        assert config.stop.surrogate == "exp"
        assert config.stop.learn_weights is True
        assert config.stop.learn_thresholds is True
        assert config.stop.learn_leakage is True

    def test_default_osttp_config(self):
        """Test OSTTP config defaults."""
        config = Config()
        assert config.osttp.pseudo_derivative == "tanh"
        assert config.osttp.output_loss == "ce"
        assert config.osttp.output_readout == "mem"
        assert config.osttp.feedback_scale == 1.0
        assert config.osttp.grad_clip == 0.0


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

    def test_invalid_drtp_output_mode(self):
        config = Config()
        config.drtp.output_mode = "logits"

        issues = validate_config(config)
        assert any("drtp.output_mode" in issue for issue in issues)

    def test_ostl_recurrent_requires_snu_type(self):
        """Test recurrent OSTL validation enforces snu/ssnu types."""
        config = Config()
        config.trainer.name = "ostl"
        config.model.architecture = "recurrent"

        issues = validate_config(config)
        assert any("OSTL currently supports" in issue for issue in issues)

    def test_ostl_invalid_output_mode(self):
        """Test OSTL validation rejects unsupported output_mode values."""
        config = Config()
        config.ostl.output_mode = "logits"

        issues = validate_config(config)
        assert any("ostl.output_mode" in issue for issue in issues)

    def test_invalid_recurrent_type(self):
        """Test validation catches invalid recurrent model type."""
        config = Config()
        config.model.recurrent_type = "custom_rnn"

        issues = validate_config(config)
        assert any("recurrent_type" in issue for issue in issues)

    def test_eprop_recurrent_type_restriction(self):
        """Test eprop recurrent model rejects non-SRNN recurrent types."""
        config = Config()
        config.trainer.name = "eprop"
        config.model.architecture = "recurrent"
        config.model.recurrent_type = "snu"

        issues = validate_config(config)
        assert any("eprop/esd_rtrl require" in issue for issue in issues)

    def test_stop_requires_fc_or_conv_architecture(self):
        """Test STOP validation enforces FC/Conv architecture."""
        config = Config()
        config.trainer.name = "stop"
        config.model.architecture = "recurrent"

        issues = validate_config(config)
        assert any("STOP currently supports" in issue for issue in issues)

    def test_osttp_requires_fc_architecture(self):
        """Test OSTTP validation enforces FC architecture."""
        config = Config()
        config.trainer.name = "osttp"
        config.model.architecture = "recurrent"

        issues = validate_config(config)
        assert any("OSTTP currently supports" in issue for issue in issues)

    def test_osttp_bce_logits_requires_logits_readout(self):
        config = Config()
        config.osttp.output_loss = "bce_logits"
        config.osttp.output_readout = "probs"

        issues = validate_config(config)
        assert any("bce_logits" in issue for issue in issues)

    def test_osttp_bce_probs_requires_probs_readout(self):
        config = Config()
        config.osttp.output_loss = "bce_probs"
        config.osttp.output_readout = "logits"

        issues = validate_config(config)
        assert any("bce_probs" in issue for issue in issues)
