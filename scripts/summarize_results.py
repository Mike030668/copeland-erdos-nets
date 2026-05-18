#!/usr/bin/env python3
"""Summarize MNIST screening results into a table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load_results(path: str):
    """Load results JSON."""
    with open(path, "r") as f:
        return json.load(f)


def summarize(results: dict):
    """Summarize results by (model, init, m)."""
    # Group runs
    groups = defaultdict(list)
    for run in results["runs"]:
        key = (run["model"], run["init"], run.get("m"))
        groups[key].append(run)

    # Compute stats
    rows = []
    for (model, init, m), runs in sorted(groups.items()):
        accuracies = [r["final_accuracy"] for r in runs]
        convergences = [
            r["convergence_epoch"] if r["convergence_epoch"] is not None else float("nan")
            for r in runs
        ]

        acc_mean = np.mean(accuracies) * 100
        acc_std = np.std(accuracies) * 100

        # Filter out nan for convergence
        conv_valid = [c for c in convergences if not np.isnan(c)]
        if conv_valid:
            conv_mean = np.mean(conv_valid)
            conv_std = np.std(conv_valid)
        else:
            conv_mean = float("nan")
            conv_std = float("nan")

        rows.append({
            "model": model,
            "init": init,
            "m": m,
            "runs": len(runs),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "conv_mean": conv_mean,
            "conv_std": conv_std,
        })

    return rows


def print_table(rows: list[dict]):
    """Print formatted summary table."""
    # Header
    print(
        f"{'Model':<6} {'Init':<8} {'m':<4} {'Runs':<5} {'Acc (mean±std)':<16} {'Conv@97 (mean±std)':<18}"
    )
    print("-" * 6 + " " + "-" * 8 + " " + "-" * 4 + " " + "-" * 5 + " " + "-" * 16 + " " + "-" * 18)

    for row in rows:
        m_str = str(row["m"]) if row["m"] is not None else "-"
        conv_str = f"{row['conv_mean']:.1f}±{row['conv_std']:.1f}" if not np.isnan(row["conv_mean"]) else "N/A"
        print(
            f"{row['model']:<6} {row['init']:<8} {m_str:<4} {row['runs']:<5} "
            f"{row['acc_mean']:.2f}±{row['acc_std']:.2f}    {conv_str:<18}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize MNIST Results")
    parser.add_argument("results_path", type=str, help="Path to results.json")
    args = parser.parse_args()

    results = load_results(args.results_path)
    rows = summarize(results)
    print_table(rows)


if __name__ == "__main__":
    main()
