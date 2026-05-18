#!/usr/bin/env python3
"""Plot MNIST screening results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str):
    """Load results JSON."""
    with open(path, "r") as f:
        return json.load(f)


def plot_accuracy_curves(results: dict, output_dir: str):
    """Plot accuracy curves for each init method."""
    output_path = Path(output_dir) / "accuracy_curves.png"

    plt.figure(figsize=(10, 6))

    # Group by model type
    for model_type in ["mlp", "cnn"]:
        init_results = {}
        for run in results["runs"]:
            if run["model"] == model_type:
                init_name = run["init"]
                if init_name not in init_results:
                    init_results[init_name] = []
                init_results[init_name].append(run["epochs"])

        for init_name, epoch_lists in init_results.items():
            # Average across seeds
            avg_curve = None
            for epochs in epoch_lists:
                accuracies = [e["test_accuracy"] for e in epochs]
                if avg_curve is None:
                    avg_curve = np.array(accuracies)
                else:
                    avg_curve += np.array(accuracies)
            avg_curve /= len(epoch_lists)

            plt.plot(range(1, len(avg_curve) + 1), avg_curve, label=f"{model_type}_{init_name}")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Screening: Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_accuracy_bar(results: dict, output_dir: str):
    """Plot bar chart of final accuracy."""
    output_path = Path(output_dir) / "final_accuracy.png"

    # Group results
    groups = {}
    for run in results["runs"]:
        key = f"{run['model']}_{run['init']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(run["final_accuracy"])

    # Prepare data
    labels = list(groups.keys())
    means = [np.mean(groups[k]) for k in labels]
    stds = [np.std(groups[k]) for k in labels]

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Final Test Accuracy")
    plt.title("MNIST Screening: Final Accuracy by Model and Init")
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_speed(results: dict, output_dir: str):
    """Plot convergence speed comparison."""
    output_path = Path(output_dir) / "convergence_speed.png"

    # Group results
    groups = {}
    for run in results["runs"]:
        key = f"{run['model']}_{run['init']}"
        if key not in groups:
            groups[key] = []
        if run["convergence_epoch"] is not None:
            groups[key].append(run["convergence_epoch"])

    # Prepare data
    labels = [k for k in groups.keys() if groups[k]]
    means = [np.mean(groups[k]) for k in labels]
    stds = [np.std(groups[k]) for k in labels] if any(len(groups[k]) > 1 for k in labels) else [0] * len(labels)

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Epoch to Convergence (acc >= 0.97)")
    plt.title("MNIST Screening: Convergence Speed")
    plt.ylim(0, max(means) + 2 if means else 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MNIST Results")
    parser.add_argument("results_path", type=str, help="Path to results.json")
    parser.add_argument("--output", type=str, default="results/mnist/plots/", help="Output directory")
    args = parser.parse_args()

    results = load_results(args.results_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_curves(results, str(output_dir))
    plot_final_accuracy_bar(results, str(output_dir))
    plot_convergence_speed(results, str(output_dir))

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
