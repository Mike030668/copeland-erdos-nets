#!/usr/bin/env python3
"""MLP-Mixer screening pipeline with CE-N, Sobol-N, and baseline initializations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision.datasets import FashionMNIST, CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# Import local initialization functions
from copeland_erdos_nets.ce_init import ce_init_
from copeland_erdos_nets.sobol_init import sobol_init_
from copeland_erdos_nets.assignment import compute_effective_rank

# ============================================================================
# MLP-Mixer Architecture
# ============================================================================

class MlpBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, num_patches: int, hidden_dim: int, token_dim: int, channel_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_patches, token_dim)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channel_dim)

    def forward(self, x):
        # Token mixing: acts on transposed spatial dimension
        out = self.ln1(x)
        out = out.transpose(1, 2)
        out = self.token_mix(out)
        out = out.transpose(1, 2)
        x = x + out
        
        # Channel mixing: acts on channel dimension
        x = x + self.channel_mix(self.ln2(x))
        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        num_classes: int,
        num_blocks: int,
        hidden_dim: int,
        token_dim: int,
        channel_dim: int,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Patch projection
        self.patch_proj = nn.Linear(patch_size * patch_size * in_channels, hidden_dim)

        # Mixer blocks
        self.blocks = nn.Sequential(*[
            MixerBlock(self.num_patches, hidden_dim, token_dim, channel_dim)
            for _ in range(num_blocks)
        ])

        # Classification head
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Split into patches and flatten: (B, num_patches, P*P*C)
        x = x.view(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, self.num_patches, -1)
        
        # Apply projection
        x = self.patch_proj(x)
        
        # Apply Mixer blocks
        x = self.blocks(x)
        
        x = self.ln(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        return self.head(x)

# ============================================================================
# Initialization Applier
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
    """Apply initialization to MLP-Mixer weights."""
    for module in model.modules():
        if not isinstance(module, nn.Linear):
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
# GPU Thermal Safety & Metrics
# ============================================================================

def get_gpu_temp() -> int:
    import subprocess
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return int(output.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def wait_for_gpu_cooling(threshold: int = 75, interval: int = 10, max_wait_seconds: int = 120):
    temp = get_gpu_temp()
    if temp == 0:
        return
    if temp >= threshold:
        print(f"  [Safety Gate] GPU Temp {temp}°C >= {threshold}°C. Cooling down...", flush=True)
        start_time = time.time()
        while temp > threshold - 3 and (time.time() - start_time) < max_wait_seconds:
            time.sleep(interval)
            temp = get_gpu_temp()
            print(f"  [Cooling...] Current Temp: {temp}°C", flush=True)
        print(f"  [Safety Gate] Cooling ended. GPU Temp: {temp}°C. Resuming.", flush=True)


def get_weight_and_spectral_stats(model: nn.Module) -> dict:
    """Collect statistics including condition number and effective rank."""
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            w = param.data.cpu()
            w_2d = w.view(w.shape[0], -1)
            try:
                s = torch.linalg.svdvals(w_2d).numpy()
                cond = float(s[0] / (s[-1] + 1e-10))
                eff_rank = compute_effective_rank(param)
            except Exception:
                cond = 1.0
                eff_rank = 1.0

            stats[name] = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
                "condition_number": cond,
                "effective_rank": eff_rank
            }
    return stats


# ============================================================================
# Data Loading
# ============================================================================

def get_dataloaders(dataset_name: str, batch_size: int, root: str = "datasets/"):
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision not available. Use synthetic data.")

    if dataset_name.lower() == "fashionmnist":
        transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
        train_dataset = FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root=root, train=False, download=True, transform=transform)
        in_channels = 1
        img_size = 28
    elif dataset_name.lower() == "cifar10":
        transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
        in_channels = 3
        img_size = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, in_channels, img_size


def get_synthetic_dataloaders(dataset_name: str, batch_size: int, num_samples: int = 1000):
    torch.manual_seed(42)
    if dataset_name.lower() == "fashionmnist":
        in_channels, img_size = 1, 28
    else:
        in_channels, img_size = 3, 32

    X_train = torch.randn(num_samples, in_channels, img_size, img_size)
    y_train = torch.randint(0, 10, (num_samples,))
    X_test = torch.randn(num_samples // 5, in_channels, img_size, img_size)
    y_test = torch.randint(0, 10, (num_samples // 5,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_channels, img_size

# ============================================================================
# Training Loops
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
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
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), correct / total

# ============================================================================
# Experiment Core
# ============================================================================

def run_experiment(
    dataset_name: str,
    init_method: dict,
    config: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    in_channels: int,
    img_size: int,
    device: torch.device,
    dry_run: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    scramble_seed: int | None = None,
    assignment: str = "sequential",
    orthogonalize: bool = False,
    matrix_shaped: bool = False,
):
    mixer_cfg = config["model"]
    model = MlpMixer(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=mixer_cfg["patch_size"],
        num_classes=10,
        num_blocks=mixer_cfg["num_blocks"],
        hidden_dim=mixer_cfg["hidden_dim"],
        token_dim=mixer_cfg["token_dim"],
        channel_dim=mixer_cfg["channel_dim"],
    )
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    epochs = 1 if dry_run else config["training"]["epochs"]
    convergence_threshold = config["evaluation"]["convergence_threshold"]

    epochs_log = []
    convergence_epoch = None

    # Epoch 0 stats (initialization)
    if not dry_run:
        epoch0 = {
            "epoch": 0,
            "train_loss": None,
            "train_accuracy": None,
            "test_loss": None,
            "test_accuracy": None,
            "weight_stats": get_weight_and_spectral_stats(model),
            "grad_norm": None,
        }
        epochs_log.append(epoch0)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Collect stats
        epoch_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "weight_stats": get_weight_and_spectral_stats(model),
        }
        epochs_log.append(epoch_entry)

        conv_mark = " ★" if convergence_epoch is None and test_acc >= convergence_threshold else ""
        print(f"  Epoch {epoch:2d}/{epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}{conv_mark}", flush=True)

        if convergence_epoch is None and test_acc >= convergence_threshold:
            convergence_epoch = epoch

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
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


def _upload_via_pydrive2(local_path: str, filename: str, sa_key_path: str = "/content/sa.json") -> bool:
    """Upload/update file on GDrive via PyDrive2 using Service Account key."""
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials

        # 1. Authenticate via sa.json (headless, no OAuth popup)
        scope = ["https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(sa_key_path, scope)
        gauth = GoogleAuth()
        gauth.credentials = creds
        drive = GoogleDrive(gauth)

        # Helper to find folder
        def find_folder(drive_client, name, parent_id=None):
            if parent_id:
                q = f"title = '{name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            else:
                q = f"title = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            file_list = drive_client.ListFile({"q": q}).GetList()
            return file_list[0]['id'] if file_list else None

        # Navigate: agent-rules-tree-control → copeland-erdos-nets_drive -> results
        root_id = find_folder(drive, "agent-rules-tree-control")
        if not root_id:
            print("[K005-B] Error: Root folder 'agent-rules-tree-control' not found.", flush=True)
            return False

        project_id = find_folder(drive, "copeland-erdos-nets_drive", root_id)
        if not project_id:
            print("[K005-B] Error: Project folder 'copeland-erdos-nets_drive' not found.", flush=True)
            return False

        results_id = find_folder(drive, "results", project_id)
        if not results_id:
            print("[K005-B] Error: 'results' folder not found.", flush=True)
            return False

        # 3. Update existing file by ID (quota bypass)
        q = f"title = '{filename}' and '{results_id}' in parents and trashed = false"
        existing = drive.ListFile({"q": q}).GetList()
        if existing:
            gfile = drive.CreateFile({"id": existing[0]['id']})
            gfile.SetContentFile(local_path)
            gfile.Upload()
            print(f"[K005-B] GDrive Direct Upload SUCCESS: {filename}", flush=True)
            return True
        else:
            print(f"[K005-B] WARNING: {filename} not found in GDrive — pre-create placeholder first", flush=True)
            return False
    except Exception as e:
        print(f"[K005-B] Error during PyDrive2 upload: {e}", flush=True)
        return False


def _save_results(results: dict, output_dir: Path) -> None:
    results_path = output_dir / "results.json"
    tmp_path = output_dir / "results.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(results_path)
    n = len(results.get("runs", []))
    print(f"  [saved {n} runs to {results_path}]", flush=True)

    exp_name = results.get("experiment", "unknown_experiment")
    filename = f"{exp_name}_results.json"

    # Path B: PyDrive2 + SA key (Preferred)
    sa_key_path = "/content/sa.json"
    if os.path.exists(sa_key_path):
        print(f"[K005] Path B active. Uploading {filename} via PyDrive2...", flush=True)
        success = _upload_via_pydrive2(str(results_path), filename, sa_key_path)
        if success:
            return

    # Path A: drive.mount / local copy (Fallback)
    gdrive_root_str = "/content/drive/MyDrive/agent-rules-tree-control/copeland-erdos-nets_drive"
    if os.path.isdir(gdrive_root_str):
        try:
            gdrive_results_dir = Path(gdrive_root_str) / "results"
            gdrive_results_dir.mkdir(parents=True, exist_ok=True)
            gdrive_path = gdrive_results_dir / filename
            gdrive_tmp_path = gdrive_results_dir / f"{filename}.tmp"
            
            with open(gdrive_tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            gdrive_tmp_path.rename(gdrive_path)
            print(f"  [K005-A] Dual-saved via mounted GDrive: {gdrive_path}", flush=True)
        except Exception as e:
            print(f"  [Warning] [K005-A] Failed dual-save to GDrive: {e}", flush=True)


def _get_run_id(dataset, init_name, seed, offset, scramble_seed, assignment=None, orthogonalize=None, matrix_shaped=None):
    suffix = ""
    if assignment and assignment != "sequential":
        suffix += f"_{assignment}"
    if orthogonalize:
        suffix += "_ortho"
    if matrix_shaped:
        suffix += "_matrix"
    return f"{dataset}_{init_name}_s{seed}_o{offset}_sc{scramble_seed}{suffix}"


def _load_existing_run_ids(output_dir: Path) -> set[str]:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return set()
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
            ids = set()
            for run in data.get("runs", []):
                ids.add(_get_run_id(
                    run.get("dataset"),
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

# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP-Mixer Screening Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output", type=str, default="results/mixer/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Run only 1-2 epochs and synthetic data")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_methods = config["init_methods"]
    seed_range = config["experiment"].get("seed_range", [42])
    datasets = config["experiment"].get("datasets", ["FashionMNIST"])

    results = {"experiment": config["experiment"]["name"], "runs": []}

    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Global Device: {device}")

    # Load existing runs
    completed_ids = _load_existing_run_ids(output_dir)
    if completed_ids:
        print(f"Found {len(completed_ids)} existing runs. Resuming...")
        with open(output_dir / "results.json", "r") as f:
            existing_data = json.load(f)
            results["runs"] = existing_data.get("runs", [])

    for dataset_name in datasets:
        # Load Data
        if HAS_TORCHVISION and not args.dry_run:
            train_loader, test_loader, in_channels, img_size = get_dataloaders(
                dataset_name=dataset_name,
                batch_size=config["data"]["batch_size"],
            )
        else:
            num_samples = 200 if args.dry_run else 1000
            train_loader, test_loader, in_channels, img_size = get_synthetic_dataloaders(
                dataset_name=dataset_name,
                batch_size=config["data"]["batch_size"],
                num_samples=num_samples,
            )

        for init_method in init_methods:
            raw_name = init_method["name"]
            mode = init_method.get("params", {}).get("mode", "normal")
            
            if raw_name == "ce":
                init_name = "ce_u" if mode == "uniform" else "ce_n"
            elif raw_name == "sobol":
                init_name = "sobol_u" if mode == "uniform" else "sobol_n"
            else:
                init_name = raw_name

            offsets = init_method.get("offsets", None)
            scramble_seeds = init_method.get("scramble_seeds", None)

            if init_name in ("ce_n", "ce_u") and offsets is not None:
                iterations = [{"offset": o, "seed": None, "scramble_seed": None} for o in offsets]
            elif init_name in ("sobol_n", "sobol_u") and scramble_seeds is not None:
                iterations = [{"offset": None, "seed": None, "scramble_seed": s} for s in scramble_seeds]
            elif init_name in ("ce_n", "ce_u"):
                iterations = [{"offset": 0, "seed": None, "scramble_seed": None}]
            elif init_name in ("sobol_n", "sobol_u"):
                iterations = [{"offset": None, "seed": None, "scramble_seed": 0}]
            else:
                iterations = [{"offset": None, "seed": s, "scramble_seed": None} for s in seed_range]

            for it in iterations:
                seed = it["seed"]
                offset = it["offset"]
                scramble_seed = it["scramble_seed"]

                assignment = init_method.get("params", {}).get("assignment", "sequential")
                orthogonalize = init_method.get("params", {}).get("orthogonalize", False)
                matrix_shaped = init_method.get("params", {}).get("matrix_shaped", False)

                run_id = _get_run_id(
                    dataset_name, init_name, seed, offset, scramble_seed,
                    assignment=assignment, orthogonalize=orthogonalize, matrix_shaped=matrix_shaped
                )

                if run_id in completed_ids:
                    continue

                # Safety check
                wait_for_gpu_cooling(threshold=config["training"].get("thermal_threshold", 75))

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

                print(f"\nRunning: {dataset_name} + {init_desc} | assignment={assignment}")
                result = run_experiment(
                    dataset_name=dataset_name,
                    init_method=init_method,
                    config=config,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    in_channels=in_channels,
                    img_size=img_size,
                    device=device,
                    dry_run=args.dry_run,
                    seed=seed,
                    offset=offset,
                    scramble_seed=scramble_seed,
                    assignment=assignment,
                    orthogonalize=orthogonalize,
                    matrix_shaped=matrix_shaped,
                )
                results["runs"].append(result)
                conv = result.get("convergence_epoch")
                conv_str = f" (converged @ epoch {conv})" if conv else ""
                print(f"  → Final accuracy: {result['final_accuracy']:.4f}{conv_str}", flush=True)

                _save_results(results, output_dir)

                cooldown = config["training"].get("cooldown_seconds", 15)
                if cooldown > 0:
                    time.sleep(cooldown)

    _save_results(results, output_dir)
    print(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
