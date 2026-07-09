#!/usr/bin/env python3
"""MNIST screening pipeline with CE-N, Sobol-N, and baseline initializations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import subprocess
import time

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
    assignment: str = "sequential",
    orthogonalize: bool = False,
    matrix_shaped: bool = False,
):
    """Apply initialization to model weights.
    
    Supports CE-N, CE-U, Sobol-N, Sobol-U, Xavier, He.
    """
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        if init_name in ("ce_n", "ce_u"):
            mode = "uniform" if init_name == "ce_u" else "normal"
            ce_init_(
                module.weight, 
                m=m, 
                kind=kind, 
                offset_blocks=offset, 
                mode=mode, 
                assignment=assignment,
                orthogonalize=orthogonalize
            )
        elif init_name in ("sobol_n", "sobol_u"):
            mode = "uniform" if init_name == "sobol_u" else "normal"
            sobol_init_(
                module.weight, 
                scramble_seed=scramble_seed, 
                kind=kind, 
                mode=mode,
                assignment=assignment,
                matrix_shaped=matrix_shaped
            )
        elif init_name == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init_name == "he":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================================
# GPU Safety & Monitoring
# ============================================================================

def get_gpu_temp() -> int:
    """Read GPU temperature using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return int(output.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def wait_for_gpu_cooling(threshold: int = 75, interval: int = 10):
    """Pause execution if GPU temperature is too high."""
    temp = get_gpu_temp()
    if temp == 0:
        return

    if temp >= threshold:
        print(f"  [Safety Gate] GPU Temp {temp}°C >= {threshold}°C. Cooling down...", flush=True)
        while temp > threshold - 5: # Cool down to threshold - 5
            time.sleep(interval)
            temp = get_gpu_temp()
            print(f"  [Cooling...] Current Temp: {temp}°C", end="\r", flush=True)
        print(f"\n  [Safety Gate] GPU cooled to {temp}°C. Resuming.")


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

def get_mnist_dataloaders(batch_size: int = 128, num_workers: int = 0, root: str = "datasets/"):
    """Load MNIST dataset once."""
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision not available. Use synthetic data for testing.")

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    print(f"  [loading MNIST from {root}...]")
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    dry_run: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    scramble_seed: int | None = None,
    assignment: str = "sequential",
    orthogonalize: bool = False,
    matrix_shaped: bool = False,
):
    """Run a single experiment (model + init combination)."""
    # DataLoader is now passed from main

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
    apply_init(
        model, 
        init_name, 
        kind=kind, 
        m=m, 
        offset=offset or 0, 
        scramble_seed=scramble_seed or 0,
        assignment=assignment,
        orthogonalize=orthogonalize,
        matrix_shaped=matrix_shaped
    )

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

    # Free model memory explicitly
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    import time
    time.sleep(0.5) # Let GPU breathe

    return {
        "model": model_type,
        "init": init_name,
        "m": m if init_name == "ce_n" else None,
        "seed": seed,
        "offset": offset,
        "scramble_seed": scramble_seed,
        "assignment": assignment,
        "orthogonalize": orthogonalize,
        "matrix_shaped": matrix_shaped,
        "epochs": epochs_log,
        "final_accuracy": epochs_log[-1].get("test_accuracy", 0.0) if epochs_log else 0.0,
        "convergence_epoch": convergence_epoch,
    }


def _upload_via_pydrive2(
    local_path: str,
    filename: str,
    sa_key_path: str = "/content/sa.json",
    root_folder: str = "",
    project_folder: str = "",
) -> bool:
    """Upload/update file on GDrive via PyDrive2 using Service Account key.

    Folder names are read from parameters (typically env vars) to avoid
    hardcoding private infrastructure names in public code.
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials

        scope = ["https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(sa_key_path, scope)
        gauth = GoogleAuth()
        gauth.credentials = creds
        drive = GoogleDrive(gauth)

        def find_folder(drive_client, name, parent_id=None):
            if parent_id:
                q = f"title = '{name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            else:
                q = f"title = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            file_list = drive_client.ListFile({"q": q}).GetList()
            return file_list[0]['id'] if file_list else None

        root_id = find_folder(drive, root_folder)
        if not root_id:
            print(f"[GDrive] Error: Root folder '{root_folder}' not found.", flush=True)
            return False

        project_id = find_folder(drive, project_folder, root_id)
        if not project_id:
            print(f"[GDrive] Error: Project folder '{project_folder}' not found.", flush=True)
            return False

        results_id = find_folder(drive, "results", project_id)
        if not results_id:
            print("[GDrive] Error: 'results' folder not found.", flush=True)
            return False

        q = f"title = '{filename}' and '{results_id}' in parents and trashed = false"
        existing = drive.ListFile({"q": q}).GetList()
        if existing:
            gfile = drive.CreateFile({"id": existing[0]['id']})
            gfile.SetContentFile(local_path)
            gfile.Upload()
            print(f"[GDrive] Upload SUCCESS: {filename}", flush=True)
            return True
        else:
            print(f"[GDrive] WARNING: {filename} not found — pre-create placeholder first", flush=True)
            return False
    except Exception as e:
        print(f"[GDrive] Error during upload: {e}", flush=True)
        return False


def _save_results(results: dict, output_dir: Path) -> None:
    """Atomically save results (crash-safe: write to tmp then rename)."""
    results_path = output_dir / "results.json"
    tmp_path = output_dir / "results.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(results_path)
    n = len(results.get("runs", []))
    print(f"  [saved {n} runs to {results_path}]", flush=True)

    # GDrive upload is optional — requires explicit env flag
    if os.environ.get("CE_NETS_ENABLE_GDRIVE_UPLOAD") != "1":
        return

    exp_name = results.get("experiment", "unknown_experiment")
    filename = f"{exp_name}_results.json"

    # Path B: PyDrive2 + SA key (preferred)
    sa_key_path = os.environ.get("CE_NETS_GDRIVE_SA_KEY", "/content/sa.json")
    root_folder = os.environ.get("CE_NETS_GDRIVE_ROOT_FOLDER", "")
    project_folder = os.environ.get("CE_NETS_GDRIVE_PROJECT_FOLDER", "")

    if sa_key_path and os.path.exists(sa_key_path) and root_folder and project_folder:
        print(f"[GDrive] Uploading {filename} via PyDrive2...", flush=True)
        success = _upload_via_pydrive2(str(results_path), filename, sa_key_path, root_folder, project_folder)
        if success:
            return

    # Path A: mounted GDrive fallback
    gdrive_root_str = os.environ.get("CE_NETS_GDRIVE_ROOT", "")
    if gdrive_root_str and os.path.isdir(gdrive_root_str):
        try:
            gdrive_results_dir = Path(gdrive_root_str) / "results"
            gdrive_results_dir.mkdir(parents=True, exist_ok=True)
            gdrive_path = gdrive_results_dir / filename
            gdrive_tmp_path = gdrive_results_dir / f"{filename}.tmp"
            
            with open(gdrive_tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            gdrive_tmp_path.rename(gdrive_path)
            print(f"  [GDrive] Dual-saved via mounted drive: {gdrive_path}", flush=True)
        except Exception as e:
            print(f"  [Warning] [GDrive] Failed dual-save: {e}", flush=True)


def _get_run_id(model_type, init_name, seed, offset, scramble_seed, assignment=None, orthogonalize=None, matrix_shaped=None):
    """Generate a unique ID for a run to check for existing results."""
    suffix = ""
    if assignment and assignment != "sequential":
        suffix += f"_{assignment}"
    if orthogonalize:
        suffix += "_ortho"
    if matrix_shaped:
        suffix += "_matrix"
    return f"{model_type}_{init_name}_s{seed}_o{offset}_sc{scramble_seed}{suffix}"


def _load_existing_run_ids(output_dir: Path) -> set[str]:
    """Load IDs of already completed runs from results.json."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return set()
    
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
            ids = set()
            for run in data.get("runs", []):
                ids.add(_get_run_id(
                    run.get("model"),
                    run.get("init"),
                    run.get("seed"),
                    run.get("offset"),
                    run.get("scramble_seed"),
                    run.get("assignment", "sequential"),
                    run.get("orthogonalize", False),
                    run.get("matrix_shaped", False),
                ))
            return ids
    except Exception as e:
        print(f"  [Warning] Could not load existing results: {e}")
        return set()


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
    
    # Handle device="auto"
    device_str = config["training"].get("device", "cpu")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Global Device: {device}")

    # Load data once
    if HAS_TORCHVISION and not args.dry_run:
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    else:
        num_samples = 200 if args.dry_run else 5000
        train_loader, test_loader = get_synthetic_dataloaders(
            batch_size=config["data"]["batch_size"],
            num_samples=num_samples,
        )

    # Load existing runs for resumption
    completed_ids = _load_existing_run_ids(output_dir)
    if completed_ids:
        print(f"Found {len(completed_ids)} existing runs. Resuming...")
        # Populate results with existing runs to keep incremental save working
        with open(output_dir / "results.json", "r") as f:
            existing_data = json.load(f)
            results["runs"] = existing_data.get("runs", [])

    # Run experiments with multi-seed/offset/scramble support
    for model_type in ["mlp", "cnn", "deep_mlp"]:
        if model_type not in model_configs:
            continue
        model_config = model_configs[model_type]

        for init_method in init_methods:
            # Bug fix: use "name" and check params to normalize ce/sobol detection
            raw_name = init_method["name"]
            mode = init_method.get("params", {}).get("mode", "normal")
            
            # Internal normalization for iteration logic
            if raw_name == "ce":
                init_name = "ce_u" if mode == "uniform" else "ce_n"
            elif raw_name == "sobol":
                init_name = "sobol_u" if mode == "uniform" else "sobol_n"
            else:
                init_name = raw_name

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
            # ...
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
                
                # Check for resumption
                run_id = _get_run_id(
                    model_type,
                    init_name,
                    seed,
                    offset,
                    scramble_seed,
                    assignment=init_method.get("params", {}).get("assignment", "sequential"),
                    orthogonalize=init_method.get("params", {}).get("orthogonalize", False),
                    matrix_shaped=init_method.get("params", {}).get("matrix_shaped", False),
                )
                if run_id in completed_ids:
                    # Skip verbose for efficiency
                    continue

                # Safety check: Thermal cooling
                wait_for_gpu_cooling(threshold=config["training"].get("thermal_threshold", 75))

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
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    dry_run=args.dry_run,
                    seed=seed,
                    offset=offset,
                    scramble_seed=scramble_seed,
                    assignment=init_method.get("params", {}).get("assignment", "sequential"),
                    orthogonalize=init_method.get("params", {}).get("orthogonalize", False),
                    matrix_shaped=init_method.get("params", {}).get("matrix_shaped", False),
                )
                results["runs"].append(result)
                conv = result.get("convergence_epoch")
                conv_str = f" (converged @ epoch {conv})" if conv else ""
                print(f"  → Final accuracy: {result['final_accuracy']:.4f}{conv_str}", flush=True)

                # Incremental save (crash-safe)
                _save_results(results, output_dir)
                
                # Inter-experiment cooldown
                cooldown = config["training"].get("cooldown_seconds", 15)
                if cooldown > 0:
                    time.sleep(cooldown)

    # Final save
    _save_results(results, output_dir)
    print(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
