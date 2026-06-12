#!/usr/bin/env python3
"""Diagnostic Collector for Initialization Methods.

Analyzes weight distributions, spectral properties (SVD), and activation 
propagation Statistics for CE, Sobol, and standard initializations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

from copeland_erdos_nets.ce_init import ce_init_
from copeland_erdos_nets.sobol_init import sobol_init_

# Import models from screening script to stay consistent
import sys
sys.path.append(str(Path(__file__).parent))
from run_mnist_screening import MnistMLP, MnistCNN, DeepMLP


def apply_init_variants(model: nn.Module, method_config: Dict[str, Any]):
    """Apply a specific initialization variant to the model."""
    name = method_config["name"]
    params = method_config.get("params", {})
    
    for name_p, param in model.named_parameters():
        if "weight" in name_p and param.ndim >= 2:
            if name == "ce":
                ce_init_(param, **params)
            elif name == "sobol":
                sobol_init_(param, **params)
            elif name == "he":
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif name == "xavier":
                nn.init.xavier_normal_(param)
            elif name == "orthogonal_standard":
                nn.init.orthogonal_(param)
        elif "bias" in name_p:
            nn.init.zeros_(param)


def collect_spectral_metrics(model: nn.Module) -> Dict[str, Any]:
    """Analyze SVD spectrum of all weight matrices."""
    metrics = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            # Reshape to 2D for SVD
            w = param.data.view(param.shape[0], -1)
            try:
                # Use svdvals for efficiency if we only need spectrum
                s = torch.linalg.svdvals(w).cpu().numpy()
                metrics[name] = {
                    "singular_values": s.tolist(),
                    "condition_number": float(s[0] / (s[-1] + 1e-10)),
                    "spectral_norm": float(s[0]),
                    "stable_rank": float((s**2).sum() / (s[0]**2 + 1e-10)),
                    "variance": float(param.data.var().item()),
                }
            except Exception as e:
                print(f"SVD failed for {name}: {e}")
    return metrics


def collect_activation_metrics(model: nn.Module, input_batch: torch.Tensor) -> Dict[str, Any]:
    """Analyze activation statistics across layers."""
    metrics = {}
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            act = output.detach()
            # For ReLU, count fraction of zeros (dead neurons in this batch)
            dead_fraction = float((act == 0).sum().item() / act.numel())
            activations[name] = {
                "mean": float(act.mean().item()),
                "std": float(act.std().item()),
                "dead_fraction": dead_fraction,
                "max": float(act.max().item()),
                "min": float(act.min().item()),
            }
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        _ = model(input_batch)

    for h in handles:
        h.remove()

    return activations


def run_diagnostics(config_path: str, output_dir: str):
    """Run full diagnostic suite."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seeds = config.get("seeds") or config.get("experiment", {}).get("seed_range", [42])
    results = []
    
    # Create a stable input batch for activation analysis
    torch.manual_seed(42)
    input_batch = torch.randn(128, 1, 28, 28).to(device)

    for method in config["init_methods"]:
        for seed in seeds:
            print(f"Analyzing method: {method['id']} (seed {seed})")
            
            # Update method config with current seed for reproducible randomness
            current_method = method.copy()
            if "params" not in current_method:
                current_method["params"] = {}
            
            # Map 'seed' to appropriate parameter based on method type
            if method["name"] == "ce":
                current_method["params"]["offset_blocks"] = seed * 1000 # Different stream portions
            elif method["name"] == "sobol":
                current_method["params"]["scramble_seed"] = seed
            
            for model_type in config["models"]:
                if model_type == "deep_mlp":
                    model = DeepMLP(hidden_sizes=[512, 512, 512, 512, 512], activation="relu")
                elif model_type == "mlp":
                    model = MnistMLP(hidden_sizes=[512], activation="relu")
                else:
                    continue
                
                # For non-CE/non-Sobol methods, use manual seed for the whole init
                if method["name"] not in ("ce", "sobol"):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                model = model.to(device)
                apply_init_variants(model, current_method)
                
                # Collect metrics
                spectral = collect_spectral_metrics(model)
                activations = collect_activation_metrics(model, input_batch)
                
                results.append({
                    "method": method["id"],
                    "seed": seed,
                    "model": model_type,
                    "spectral": spectral,
                    "activations": activations
                })
                
                # Clear memory explicitly
                model.to("cpu")
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # Small breathe time for GPU
                import time
                time.sleep(0.2)

    # Save results
    with open(output_path / "diagnostics_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / 'diagnostics_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/T005/")
    args = parser.parse_args()
    
    run_diagnostics(args.config, args.output)
