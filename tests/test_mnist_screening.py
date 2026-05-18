"""Tests for MNIST screening pipeline (CPU-only, synthetic data)."""

import json
import os
import tempfile

import pytest
import torch

from scripts.run_mnist_screening import (
    HAS_TORCHVISION,
    MnistCNN,
    MnistMLP,
    apply_init,
    get_synthetic_dataloaders,
    run_experiment,
)


class TestMnistMLP:
    """Tests for MnistMLP model."""

    def test_creation(self):
        """Model creates successfully."""
        model = MnistMLP(hidden_sizes=[256, 128], activation="relu", dropout=0.0)
        assert model is not None

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        model = MnistMLP(hidden_sizes=[256, 128])
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)

    def test_forward_flatten(self):
        """Input is properly flattened."""
        model = MnistMLP(hidden_sizes=[64])
        x = torch.randn(8, 1, 28, 28)
        output = model(x)
        # Should flatten to 784 before first linear layer
        assert output.shape == (8, 10)


class TestMnistCNN:
    """Tests for MnistCNN model."""

    def test_creation(self):
        """Model creates successfully."""
        model = MnistCNN(channels=[16, 32], kernel_size=3, fc_size=128)
        assert model is not None

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        model = MnistCNN(channels=[16, 32], kernel_size=3, fc_size=128)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)

    def test_conv_layers(self):
        """Conv layers reduce spatial dimensions correctly."""
        model = MnistCNN(channels=[16, 32], kernel_size=3, fc_size=128)
        x = torch.randn(1, 1, 28, 28)
        conv_out = model.conv(x)
        # After 2 pool ops: 28 -> 14 -> 7
        assert conv_out.shape[2] == 7
        assert conv_out.shape[3] == 7
        assert conv_out.shape[1] == 32  # out_channels


class TestInitMethods:
    """Tests for initialization methods."""

    def test_ce_n_init_applied(self):
        """CE-N init is applied to Linear layers."""
        model = MnistMLP(hidden_sizes=[64, 32])
        apply_init(model, "ce_n", kind="he", m=4)
        # Check that weights are initialized (not NaN)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.isnan(module.weight).any()

    def test_xavier_init_applied(self):
        """Xavier init is applied."""
        model = MnistMLP(hidden_sizes=[64])
        apply_init(model, "xavier")
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.isnan(module.weight).any()

    def test_he_init_applied(self):
        """He init is applied."""
        model = MnistMLP(hidden_sizes=[64])
        apply_init(model, "he")
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.isnan(module.weight).any()

    def test_cnn_ce_n_init(self):
        """CE-N init works for CNN Conv2d layers."""
        model = MnistCNN(channels=[16, 32], kernel_size=3, fc_size=64)
        apply_init(model, "ce_n", kind="he", m=4)
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                assert not torch.isnan(module.weight).any()


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_config(self):
        """Config file exists and is valid JSON."""
        config_path = "configs/mnist_screening.json"
        assert os.path.exists(config_path)
        # Check file is readable as JSON
        with open(config_path, "r") as f:
            config = json.load(f)
        assert config["experiment"]["name"] == "mnist_screening"
        assert config["data"]["batch_size"] == 128

    def test_model_configs_exist(self):
        """Model configs are present in file."""
        config_path = "configs/mnist_screening.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        assert "mlp" in config["models"]
        assert "cnn" in config["models"]
        assert config["models"]["mlp"]["hidden_sizes"] == [256, 128]


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_synthetic_loader(self):
        """Synthetic dataloaders work."""
        train_loader, test_loader = get_synthetic_dataloaders(batch_size=32, num_samples=200)

        # Check we can iterate
        for data, target in train_loader:
            assert data.shape[0] <= 32
            assert target.shape[0] <= 32
            break

    def test_synthetic_shape(self):
        """Synthetic data has correct shape."""
        train_loader, _ = get_synthetic_dataloaders(batch_size=16, num_samples=100)

        for data, target in train_loader:
            assert data.shape == (16, 1, 28, 28)
            assert target.shape == (16,)
            break


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_no_crash(self):
        """Dry-run completes without crashing."""
        config = {
            "training": {"device": "cpu", "epochs": 15, "lr": 0.001},
            "data": {"batch_size": 32, "num_workers": 0},
            "evaluation": {"convergence_threshold": 0.97},
        }
        model_config = {"hidden_sizes": [32, 16], "activation": "relu", "dropout": 0.0}
        init_method = {"name": "ce_n", "kind": "he", "m": 4}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(
                model_type="mlp",
                model_config=model_config,
                init_method=init_method,
                config=config,
                output_dir=tmpdir,
                dry_run=True,
            )

            assert result["model"] == "mlp"
            assert result["init"] == "ce_n"
            assert len(result["epochs"]) >= 1
            assert "final_accuracy" in result

    def test_dry_run_single_epoch(self):
        """Dry-run runs exactly 1 epoch."""
        config = {
            "training": {"device": "cpu", "epochs": 15, "lr": 0.001},
            "data": {"batch_size": 32, "num_workers": 0},
            "evaluation": {"convergence_threshold": 0.97},
        }
        model_config = {"hidden_sizes": [32], "activation": "relu", "dropout": 0.0}
        init_method = {"name": "xavier"}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(
                model_type="mlp",
                model_config=model_config,
                init_method=init_method,
                config=config,
                output_dir=tmpdir,
                dry_run=True,
            )

            assert len(result["epochs"]) == 1


class TestMultiSeed:
    """Tests for multi-seed functionality."""

    def test_multi_seed_creates_different_weights(self):
        """Different seeds produce different weights for Xavier init."""
        # First run with seed 42
        torch.manual_seed(42)
        model1 = MnistMLP(hidden_sizes=[32, 16])
        apply_init(model1, "xavier")
        weights1 = model1.network[0].weight.clone()

        # Second run with seed 100
        torch.manual_seed(100)
        model2 = MnistMLP(hidden_sizes=[32, 16])
        apply_init(model2, "xavier")
        weights2 = model2.network[0].weight.clone()

        # Weights should be different
        assert not torch.equal(weights1, weights2)

    def test_ce_n_deterministic_no_seed(self):
        """CE-N init produces same weights regardless of seed."""
        # First run
        model1 = MnistMLP(hidden_sizes=[32, 16])
        apply_init(model1, "ce_n", m=4)
        weights1 = model1.network[0].weight.clone()

        # Second run (no seed set)
        model2 = MnistMLP(hidden_sizes=[32, 16])
        apply_init(model2, "ce_n", m=4)
        weights2 = model2.network[0].weight.clone()

        # CE-N is deterministic - weights should be identical
        assert torch.equal(weights1, weights2)


class TestConfigMParameter:
    """Tests for m parameter in config."""

    def test_config_m_parameter(self):
        """Verify m passes through to init in ablation config."""
        config_path = "configs/mnist_m_ablation.json"
        assert os.path.exists(config_path)

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check that CE-N methods have m parameter
        ce_n_methods = [m for m in config["init_methods"] if m["name"] == "ce_n"]
        assert len(ce_n_methods) == 4

        m_values = [m["m"] for m in ce_n_methods]
        assert set(m_values) == {3, 4, 5, 6}

        # He baseline should not have m
        he_methods = [m for m in config["init_methods"] if m["name"] == "he"]
        assert len(he_methods) == 1
        assert "m" not in he_methods[0]
