#!/usr/bin/env python3
"""MNIST screening pipeline with CE-N, Sobol-N, and baseline initializations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Try to import torchvision; if not available, use synthetic data
try:
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from copeland_erdos_nets.ce_init import ce_init_
from copeland_erdos_nets.sobol_init import sobol_init_


# ============================================================================
# Models
# ============================================================================

class MnistMLP(nn.Module):
    """Simple MLP for MNIST: 784 -> hidden -> 10."""

    def __init__(
        self,
        hidden_sizes: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout_rate = dropout

        layers = []
        prev_size = 784
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(self._get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 10))

        self.network = nn.Sequential(*layers)

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class MnistCNN(nn.Module):
    """Simple CNN for MNIST: Conv->Pool->Conv->Pool->FC->10."""

    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 3,
        fc_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.activation = activation
        self.dropout_rate = dropout

        conv_layers = []
        in_channels = 1
        for out_ch in channels:
            conv_layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size, padding=kernel_size // 2)
            )
            conv_layers.append(self._get_activation())
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_ch
        self.conv = nn.Sequential(*conv_layers)

        self._flattened_size = channels[-1] * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(self._flattened_size, fc_size),
            self._get_activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_size, 10),
        )

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepMLP(nn.Module):
    """Deep MLP for MNIST: 784 -> 512 -> 512 -> 256 -> 128 -> 10."""

    def __init__(
        self,
        hidden_sizes: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        layers = []
        prev_size = 784
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(self._get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 10))

        self.network = nn.Sequential(*layers)

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


# ============================================================================
# Initialization Methods
# ============================================================================

def apply_init(
    model: nn.Module,
    init_name: str,
    kind: str = "he",
    m: int = 4,
    offset: int = 0,
    scramble_seed: int = 0,
):
    """Apply initialization to model weights.
    
    Supports CE-N, CE-U, Sobol-N, Sobol-U, Xavier, He.
    """
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        if init_name in ("ce_n", "ce_u"):
            mode = "uniform" if init_name == "ce_u" else "normal"
            ce_init_(module.weight, m=m, kind=kind, offset_blocks=offset, mode=mode)
        elif init_name in ("sobol_n", "sobol_u"):
            mode = "uniform" if init_name == "sobol_u" else "normal"
            sobol_init_(module.weight, scramble_seed=scramble_seed, kind=kind, mode=mode)
        elif init_name == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init_name == "he":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_weight_stats(model: nn.Module) -> dict:
    """Collect weight statistics for each layer."""
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data.cpu()
            stats[name] = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
            }
    return stats


def get_activation_stats(model: nn.Module, device: torch.device) -> dict:
    """Collect activation statistics for each layer (on first forward pass)."""
    stats = {}
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            act = output.detach().cpu()
            activations[name] = {
                "mean": float(act.mean()),
                "std": float(act.std()),
            }
        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass on dummy input
    with torch.no_grad():
        if isinstance(model, (MnistMLP, DeepMLP)):
            dummy = torch.randn(1, 1, 28, 28, device=device)
        else:
            dummy = torch.randn(1, 1, 28, 28, device=device)
        _ = model(dummy)

    # Remove hooks
    for h in handles:
        h.remove()

    return activations


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ============================================================================
# Data Loading
# ============================================================================

def get_mnist_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """Load MNIST dataset."""
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision not available. Use synthetic data for testing.")

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(root="datasets/", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="datasets/", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_synthetic_dataloaders(batch_size: int = 128, num_samples: int = 1000):
    """Generate synthetic MNIST-like data for testing."""
    torch.manual_seed(42)

    X_train = torch.randn(num_samples, 1, 28, 28)
    y_train = torch.randint(0, 10, (num_samples,))
    X_test = torch.randn(num_samples // 5, 1, 28, 28)
    y_test = torch.randint(0, 10, (num_samples // 5,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def run_experiment(
    model_type: str,
    model_config: dict,
    init_method: dict,
    config: dict,
    output_dir: str,
    dry_run: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    scramble_seed: int | None = None,
):
    """Run a single experiment (model + init combination)."""
    # Handle device="auto"
    device_str = config["training"].get("device", "cpu")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Get data
    if HAS_TORCHVISION and not dry_run:
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    else:
        num_samples = 200 if dry_run else 5000
        train_loader, test_loader = get_synthetic_dataloaders(
            batch_size=config["data"]["batch_size"],
            num_samples=num_samples,
        )

    # Build model
    if model_type == "mlp":
        model = MnistMLP(
            hidden_sizes=model_config["hidden_sizes"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
        )
    elif model_type == "cnn":
        model = MnistCNN(
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            fc_size=model_config["fc_size"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
        )
    elif model_type == "deep_mlp":
        model = DeepMLP(
            hidden_sizes=model_config["hidden_sizes"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Apply initialization
    init_name = init_method["name"]
    kind = init_method.get("kind", "he")
    m = init_method.get("m", 4)
    apply_init(model, init_name, kind=kind, m=m, offset=offset or 0, scramble_seed=scramble_seed or 0)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # Training loop
    epochs = 1 if dry_run else config["training"]["epochs"]
    convergence_threshold = config["evaluation"]["convergence_threshold"]

    epochs_log = []
    convergence_epoch = None

    # Collect epoch 0 (before training)
    if not dry_run:
        epoch0 = {
            "epoch": 0,
            "train_loss": None,
            "train_accuracy": None,
            "test_loss": None,
            "test_accuracy": None,
            "weight_stats": get_weight_stats(model),
            "grad_norm": None,
            "activation_stats": get_activation_stats(model, device),
        }
        epochs_log.append(epoch0)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Compute grad norm (after backward pass)
        grad_norm = compute_grad_norm(model)

        # Collect stats
        epoch_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "weight_stats": get_weight_stats(model),
            "grad_norm": grad_norm,
            "activation_stats": get_activation_stats(model, device),
        }

        epochs_log.append(epoch_entry)

        # Verbose epoch logging
        conv_mark = " ★" if convergence_epoch is None and test_acc >= convergence_threshold else ""
        print(f"  Epoch {epoch:2d}/{epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
              f"grad_norm={grad_norm:.4f}{conv_mark}", flush=True)

        if convergence_epoch is None and test_acc >= convergence_threshold:
            convergence_epoch = epoch

        if dry_run:
            break

    # Find convergence epoch
    if convergence_epoch is None:
        for entry in epochs_log:
            if entry.get("test_accuracy") and entry["test_accuracy"] >= convergence_threshold:
                convergence_epoch = entry["epoch"]
                break

    return {
        "model": model_type,
        "init": init_name,
        "m": m if init_name == "ce_n" else None,
        "seed": seed,
        "offset": offset,
        "scramble_seed": scramble_seed,
        "epochs": epochs_log,
        "final_accuracy": epochs_log[-1].get("test_accuracy", 0.0) if epochs_log else 0.0,
        "convergence_epoch": convergence_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="MNIST Screening Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output", type=str, default="results/mnist/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Run only 1 epoch for testing")
    args = parser.parse_args()

    # Load config (JSON)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model and init configs
    model_configs = config["models"]
    init_methods = config["init_methods"]
    seed_range = config.get("seed_range", [42, 43, 44, 45, 46])

    results = {"experiment": config["experiment"]["name"], "runs": []}

    # Run experiments with multi-seed/offset/scramble support
    for model_type in ["mlp", "cnn", "deep_mlp"]:
        if model_type not in model_configs:
            continue
        model_config = model_configs[model_type]

        for init_method in init_methods:
            init_name = init_method["name"]

            # Determine iteration values
            offsets = init_method.get("offsets", None)
            scramble_seeds = init_method.get("scramble_seeds", None)

            if init_name in ("ce_n", "ce_u") and offsets is not None:
                # Run once per offset
                iterations = [{"offset": o, "seed": None, "scramble_seed": None} for o in offsets]
            elif init_name in ("sobol_n", "sobol_u") and scramble_seeds is not None:
                # Run once per scramble_seed
                iterations = [{"offset": None, "seed": None, "scramble_seed": s} for s in scramble_seeds]
            elif init_name in ("ce_n", "ce_u"):
                # CE without offsets: deterministic, run once
                iterations = [{"offset": 0, "seed": None, "scramble_seed": None}]
            elif init_name in ("sobol_n", "sobol_u"):
                # Sobol without scramble_seeds: use default seed
                iterations = [{"offset": None, "seed": None, "scramble_seed": 0}]
            else:
                # Xavier/He: run for each seed
                iterations = [{"offset": None, "seed": s, "scramble_seed": None} for s in seed_range]

            for it in iterations:
                seed = it["seed"]
                offset = it["offset"]
                scramble_seed = it["scramble_seed"]

                # Set seed for non-CE-N/non-Sobol runs
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                init_desc = f"{init_name}"
                if offset is not None:
                    init_desc += f"(offset={offset})"
                if scramble_seed is not None:
                    init_desc += f"(seed={scramble_seed})"
                if seed is not None:
                    init_desc += f"[seed={seed}]"

                print(f"Running: {model_type} + {init_desc}")
                result = run_experiment(
                    model_type=model_type,
                    model_config=model_config,
                    init_method=init_method,
                    config=config,
                    output_dir=str(output_dir),
                    dry_run=args.dry_run,
                    seed=seed,
                    offset=offset,
                    scramble_seed=scramble_seed,
                )
                results["runs"].append(result)
                conv = result.get("convergence_epoch")
                conv_str = f" (converged @ epoch {conv})" if conv else ""
                print(f"  → Final accuracy: {result['final_accuracy']:.4f}{conv_str}", flush=True)
                print(flush=True)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
