#!/usr/bin/env python3
"""R010 paired attention-init confirmation runner.

New causal path. Does not change historical run_transformer_screening.py.

Modes:
  smoke        — plumbing only; not scientific evidence
  confirmation — Phase 1: 4 methods × seeds 42–46 × 15 epochs
                 (authorized 2026-08-24 after smoke-v2 PASS)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from copeland_erdos_nets.r010_protocol import (
    METHODS,
    PRIMARY_BASELINE,
    apply_attention_intervention,
    assert_expected_allowlist_count,
    assert_t0_invariance,
    attention_allowlist,
    batch_order_records,
    build_base_state,
    clone_from_base_state,
    collect_spectral,
    derive_seeds,
    dump_json,
    epoch_index_permutations,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"
AUTHORIZED_CONFIRM_SEEDS = (42, 43, 44, 45, 46)
AUTHORIZED_CONFIRM_EPOCHS = 15


def load_historical_model_module():
    spec = importlib.util.spec_from_file_location("ce_hist_transformer", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TokenizedDataset(Dataset):
    def __init__(self, ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.chunks = [
            ids[i : i + seq_len + 1]
            for i in range(0, len(ids) - seq_len - 1, seq_len)
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class EpochPermutationSampler(Sampler[int]):
    def __init__(self, order: list[int]):
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_gpu_temp() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(output.strip().splitlines()[0])
    except Exception:
        return 0


def get_cpu_temp() -> int:
    try:
        output = subprocess.check_output(["sensors", "-u"], text=True, stderr=subprocess.DEVNULL)
        temps = []
        for line in output.splitlines():
            if "temp1_input" in line or "Package id 0" in line:
                try:
                    temps.append(float(line.split(":")[-1].strip()))
                except ValueError:
                    pass
        if temps:
            return int(max(temps))
    except Exception:
        pass
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            val = int(zone.read_text().strip())
            if val > 200:  # millidegC
                val //= 1000
            if 20 < val < 120:
                return val
        except Exception:
            continue
    return 0


def wait_for_cooling(gpu_threshold: int = 78, cpu_threshold: int = 90, interval: int = 10) -> None:
    """K001/K007 gate. Newline prints only (no \\r)."""
    while True:
        gpu = get_gpu_temp()
        cpu = get_cpu_temp()
        hot = (gpu and gpu >= gpu_threshold) or (cpu and cpu >= cpu_threshold)
        if not hot:
            return
        print(
            f"[thermal] pause gpu={gpu}C cpu={cpu}C "
            f"(limits gpu>={gpu_threshold} cpu>={cpu_threshold})",
            flush=True,
        )
        time.sleep(interval)


def write_heartbeat(path: Path, **payload) -> None:
    payload = {
        **payload,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu_temp": get_gpu_temp(),
        "cpu_temp": get_cpu_temp(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, event: dict) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def load_wikitext_splits(cfg: dict):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    raw = load_dataset(cfg["dataset"], cfg["config_name"])
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    tok.pad_token = tok.eos_token

    def flatten(split: str) -> list[int]:
        tokenized = raw[split].map(
            lambda b: tok(b["text"]), batched=True, remove_columns=raw[split].column_names
        )
        ids: list[int] = []
        for row in tokenized["input_ids"]:
            ids.extend(row)
        return ids

    seq = int(cfg["seq_len"])
    splits = {
        "train": TokenizedDataset(flatten("train"), seq),
        "validation": TokenizedDataset(flatten("validation"), seq),
        "test": TokenizedDataset(flatten("test"), seq),
    }
    return splits, tok.vocab_size, {
        "dataset": cfg["dataset"],
        "config_name": cfg["config_name"],
        "tokenizer": cfg["tokenizer"],
        "seq_len": seq,
        "n_train_chunks": len(splits["train"]),
        "n_val_chunks": len(splits["validation"]),
        "n_test_chunks": len(splits["test"]),
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paired_stats(metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seed: dict[int, dict[str, dict]] = {}
    for row in metrics:
        by_seed.setdefault(int(row["seed"]), {})[row["method"]] = row
    diffs = []
    for seed, methods in sorted(by_seed.items()):
        base = methods.get(PRIMARY_BASELINE)
        if base is None:
            continue
        base_ppl = float(base["test_ppl"])
        for method, row in methods.items():
            if method == PRIMARY_BASELINE:
                continue
            delta = float(row["test_ppl"]) - base_ppl
            diffs.append(
                {
                    "method": method,
                    "seed": seed,
                    "method_test_ppl": float(row["test_ppl"]),
                    "baseline_test_ppl": base_ppl,
                    "delta_test_ppl": delta,
                    "rel_delta": delta / base_ppl if base_ppl else "",
                    "best_val_ppl": float(row["best_val_ppl"]),
                    "final_val_ppl": float(row["final_val_ppl"]),
                    "best_epoch": int(row["best_epoch"]),
                    "final_to_best_val_ratio": (
                        float(row["final_val_ppl"]) / float(row["best_val_ppl"])
                        if float(row["best_val_ppl"])
                        else ""
                    ),
                }
            )
    summary = []
    from collections import defaultdict

    grouped: dict[str, list[float]] = defaultdict(list)
    for d in diffs:
        grouped[d["method"]].append(float(d["delta_test_ppl"]))
    for method, vals in grouped.items():
        arr = np.array(vals, dtype=float)
        n = len(arr)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            se = sd / math.sqrt(n)
            # Student-t 95% two-sided, df=n-1; tcrit for n=5 df=4 ≈ 2.776
            tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.776)
            lo, hi = mean - tcrit * se, mean + tcrit * se
        else:
            lo = hi = mean
        summary.append(
            {
                "method": method,
                "n": n,
                "mean_delta_test_ppl": mean,
                "sd_delta_test_ppl": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "rel_mean_delta": mean / np.mean([d["baseline_test_ppl"] for d in diffs if d["method"] == method]),
            }
        )
    return diffs, summary


def write_environment(path: Path, device) -> None:
    try:
        import datasets as hf_datasets
        datasets_ver = hf_datasets.__version__
    except Exception:
        datasets_ver = "unavailable"
    try:
        import transformers
        transformers_ver = transformers.__version__
    except Exception:
        transformers_ver = "unavailable"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    cuda_runtime = torch.version.cuda or "none"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = "unavailable"
    path.write_text(
        (
            f"python={os.sys.version}\n"
            f"platform={platform.platform()}\n"
            f"torch={torch.__version__}\n"
            f"numpy={np.__version__}\n"
            f"datasets={datasets_ver}\n"
            f"transformers={transformers_ver}\n"
            f"cuda_available={torch.cuda.is_available()}\n"
            f"cuda_runtime={cuda_runtime}\n"
            f"gpu_model={gpu_name}\n"
            f"driver_version={driver}\n"
            f"device={device}\n"
            f"git_commit={git_commit()}\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    heartbeat_path = out / "heartbeat.json"
    run_log = out / "run_log.jsonl"

    mode = cfg["experiment"].get("mode", "smoke")
    methods = list(cfg["experiment"]["methods"])
    if methods != list(METHODS):
        raise SystemExit(f"methods must be exactly {METHODS}")

    if mode == "smoke":
        seeds = [int(cfg["experiment"]["seed"])]
        epochs = int(cfg["training"]["epochs"])
        if args.max_epochs is not None:
            epochs = min(epochs, args.max_epochs)
        if epochs > 2:
            raise SystemExit("smoke authorization is max 2 epochs")
        scientific = False
    elif mode == "confirmation":
        seeds = [int(s) for s in cfg["experiment"]["seeds"]]
        if seeds != list(AUTHORIZED_CONFIRM_SEEDS):
            raise SystemExit(f"confirmation seeds must be exactly {AUTHORIZED_CONFIRM_SEEDS}")
        epochs = int(cfg["training"]["epochs"])
        if args.max_epochs is not None:
            epochs = min(epochs, args.max_epochs)
        if epochs != AUTHORIZED_CONFIRM_EPOCHS:
            raise SystemExit(f"confirmation must be {AUTHORIZED_CONFIRM_EPOCHS} epochs")
        scientific = True
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    device = resolve_device(cfg["training"]["device"])
    gpu_thr = int(cfg["training"].get("thermal_threshold", 78))
    cpu_thr = int(cfg["training"].get("cpu_thermal_threshold", 90))
    cooldown = int(cfg["training"].get("cooldown_seconds", 15))

    print(f"[Milestone] R010 {mode} start seeds={seeds} epochs={epochs} device={device}", flush=True)
    append_jsonl(run_log, {"event": "job_start", "mode": mode, "seeds": seeds, "epochs": epochs})
    write_heartbeat(
        heartbeat_path,
        job_id=f"r010_{mode}",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        progress="starting",
        progress_pct=0,
        current_phase="load_data",
        host=platform.node(),
    )
    write_environment(out / "environment.txt", device)

    hist = load_historical_model_module()
    splits, vocab_size, data_manifest = load_wikitext_splits(cfg["data"])
    bs = int(cfg["data"]["batch_size"])
    train_drop = bool(cfg["data"]["train_drop_last"])
    val_loader = DataLoader(
        splits["validation"], batch_size=bs, shuffle=False,
        drop_last=bool(cfg["data"]["val_drop_last"]),
    )
    test_loader = DataLoader(
        splits["test"], batch_size=bs, shuffle=False,
        drop_last=bool(cfg["data"]["test_drop_last"]),
    )

    def factory():
        return hist.DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]),
            d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]),
            max_seq_len=int(cfg["data"]["seq_len"]),
        )

    criterion = nn.CrossEntropyLoss()
    metrics_path = out / ("per_seed.csv" if scientific else "smoke_metrics.csv")
    metrics = read_csv_rows(metrics_path)
    done = {(r["method"], int(r["seed"])) for r in metrics}

    t0_spectral = read_csv_rows(out / "t0_spectral.csv")
    best_spectral = read_csv_rows(out / "best_epoch_spectral.csv")
    final_spectral = read_csv_rows(out / "final_epoch_spectral.csv")
    unchanged_rows = read_csv_rows(out / "unchanged_parameter_hashes.csv")
    changed_hash_rows = read_csv_rows(out / "changed_parameter_hashes.csv")
    batch_rows = read_csv_rows(out / "batch_order_hashes.csv")
    protocol_deviations: list[str] = []
    allow = None
    base_hashes_global = None

    total_units = len(seeds) * len(methods)
    finished_units = len(done)

    for seed in seeds:
        seeds_obj = derive_seeds(seed)
        perms = epoch_index_permutations(len(splits["train"]), epochs, seeds_obj.seed_shuffle)
        existing_epochs = {int(r["epoch"]) for r in batch_rows if int(r["seed"]) == seed}
        if not existing_epochs:
            batch_rows.extend(
                batch_order_records(perms, seed=seed, batch_size=bs, drop_last=train_drop)
            )
        base, base_hashes = build_base_state(factory, seeds_obj.seed_model, device="cpu")
        if allow is None:
            allow = attention_allowlist(base)
            assert_expected_allowlist_count(allow, n_layers=int(cfg["model"]["n_layers"]))
            dump_json(out / "parameter_allowlist.json", {"allowlist": allow})
            write_csv(
                out / "base_state_hashes.csv",
                [{"seed": seed, "name": k, "sha256": v} for k, v in sorted(base_hashes.items())],
            )
            base_hashes_global = base_hashes
        else:
            # append this seed's base hashes
            prev = read_csv_rows(out / "base_state_hashes.csv")
            if not any(int(r.get("seed", -1)) == seed for r in prev):
                prev.extend({"seed": seed, "name": k, "sha256": v} for k, v in sorted(base_hashes.items()))
                write_csv(out / "base_state_hashes.csv", prev)

        for method in methods:
            key = (method, seed)
            if key in done:
                print(f"[resume] skip {method} seed={seed}", flush=True)
                continue
            wait_for_cooling(gpu_thr, cpu_thr)
            print(f"[Milestone] method={method} seed={seed}", flush=True)
            append_jsonl(run_log, {"event": "run_start", "method": method, "seed": seed})
            write_heartbeat(
                heartbeat_path,
                job_id=f"r010_{mode}",
                status="running",
                progress=f"{finished_units}/{total_units}",
                progress_pct=100.0 * finished_units / total_units,
                current_phase=f"{method}/seed{seed}",
                host=platform.node(),
            )
            model = clone_from_base_state(base, factory)
            meta = apply_attention_intervention(
                model, method, seeds_obj, allowlist=allow,
                ce_m=int(cfg["init"]["ce_m"]),
                ce_offset_blocks=int(cfg["init"]["ce_offset_blocks"]),
            )
            unchanged, changed_rows = assert_t0_invariance(model, base_hashes, allow)
            for name, digest in unchanged.items():
                unchanged_rows.append({"method": method, "seed": seed, "name": name, "sha256": digest})
            for row in changed_rows:
                changed_hash_rows.append(
                    {
                        "method": method, "seed": seed, "name": row["name"],
                        "base_sha256": row["base_sha256"],
                        "post_intervention_sha256": row["post_intervention_sha256"],
                        "changed": row["changed"],
                    }
                )
            t0_spectral.extend(collect_spectral(model, allow, state="t0", method=method, seed=seed))
            model.to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"]["weight_decay"]),
            )
            best_val = math.inf
            best_epoch = 0
            best_path = out / "checkpoints" / f"{method}_seed{seed}_best.pt"
            last_val = math.inf
            for epoch, order in enumerate(perms, start=1):
                wait_for_cooling(gpu_thr, cpu_thr)
                usable = order
                if train_drop:
                    usable = order[: (len(order) // bs) * bs]
                loader = DataLoader(
                    splits["train"], batch_size=bs,
                    sampler=EpochPermutationSampler(usable), drop_last=False,
                )
                model.train()
                running = 0.0
                n = 0
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss.backward()
                    opt.step()
                    running += float(loss.item()) * x.size(0)
                    n += x.size(0)
                last_val = evaluate(model, val_loader, device, criterion)
                print(
                    f"  epoch {epoch}/{epochs} train_loss={running/max(n,1):.4f} val_loss={last_val:.4f}",
                    flush=True,
                )
                write_heartbeat(
                    heartbeat_path,
                    job_id=f"r010_{mode}",
                    status="running",
                    progress=f"{finished_units}/{total_units}",
                    progress_pct=100.0 * finished_units / total_units,
                    current_phase=f"{method}/seed{seed}/ep{epoch}",
                    host=platform.node(),
                )
                if last_val < best_val:
                    best_val = last_val
                    best_epoch = epoch
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "val_loss": last_val},
                        best_path,
                    )
            final_spectral.extend(
                collect_spectral(model, allow, state="final_epoch", method=method, seed=seed)
            )
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            best_spectral.extend(
                collect_spectral(model, allow, state="best_validation", method=method, seed=seed)
            )
            test_loss = evaluate(model, test_loader, device, criterion)
            metrics.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "best_val_ppl": math.exp(min(best_val, 20)),
                    "final_val_loss": last_val,
                    "final_val_ppl": math.exp(min(last_val, 20)),
                    "test_loss": test_loss,
                    "test_ppl": math.exp(min(test_loss, 20)),
                    "seed_model": meta["seed_model"],
                    "seed_attention": meta["seed_attention"],
                    "seed_attention_method": meta["seed_attention_method"],
                    "seed_shuffle": meta["seed_shuffle"],
                }
            )
            done.add(key)
            finished_units += 1
            write_csv(metrics_path, metrics)
            write_csv(out / "t0_spectral.csv", t0_spectral)
            write_csv(out / "best_epoch_spectral.csv", best_spectral)
            write_csv(out / "final_epoch_spectral.csv", final_spectral)
            write_csv(out / "unchanged_parameter_hashes.csv", unchanged_rows)
            write_csv(out / "changed_parameter_hashes.csv", changed_hash_rows)
            write_csv(out / "batch_order_hashes.csv", batch_rows)
            append_jsonl(
                run_log,
                {"event": "run_end", "method": method, "seed": seed, "test_ppl": math.exp(min(test_loss, 20))},
            )
            if cooldown > 0:
                print(f"[cooldown] {cooldown}s", flush=True)
                time.sleep(cooldown)

    dump_json(out / "resolved_config.json", cfg)
    dump_json(
        out / "dataset_manifest.json",
        {
            **data_manifest,
            "train_drop_last": train_drop,
            "val_drop_last": cfg["data"]["val_drop_last"],
            "test_drop_last": cfg["data"]["test_drop_last"],
        },
    )
    (out / "commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    seed0 = seeds[0]
    s0 = derive_seeds(seed0)
    (out / "rng_policy.md").write_text(
        f"""# R010 RNG policy

```text
seed_model     = experiment_seed + {s0.seed_model - seed0}
seed_attention = experiment_seed + {s0.seed_attention - seed0}
seed_shuffle   = experiment_seed + {s0.seed_shuffle - seed0}
```

Each method uses an isolated attention stream:
`seed_attention_method = seed_attention + METHOD_SALT * 1000003`.

Train order is a precomputed permutation from `seed_shuffle` (same for all methods of a seed).
Model/attention RNG does not advance the shuffle generator.
CE-LCG uses the historical same-offset construction (frozen).
""",
        encoding="utf-8",
    )
    (out / "protocol_deviations.md").write_text(
        "# Protocol deviations\n\n"
        + ("None.\n" if not protocol_deviations else "\n".join(f"- {d}" for d in protocol_deviations) + "\n"),
        encoding="utf-8",
    )
    if scientific:
        diffs, summary = paired_stats(metrics)
        write_csv(out / "paired_differences.csv", diffs)
        write_csv(out / "summary.csv", summary)
        write_csv(out / "validation_metrics.csv", metrics)
        write_csv(out / "test_metrics.csv", metrics)
    dump_json(
        out / "job_summary.json",
        {"mode": mode, "scientific_evidence": scientific, "primary_baseline": PRIMARY_BASELINE, "n_rows": len(metrics)},
    )
    write_heartbeat(
        heartbeat_path,
        job_id=f"r010_{mode}",
        status="complete",
        progress=f"{finished_units}/{total_units}",
        progress_pct=100.0,
        current_phase="done",
        host=platform.node(),
    )
    append_jsonl(run_log, {"event": "job_end", "completed": finished_units, "failed": 0})
    print(f"[Milestone] R010 {mode} complete rows={len(metrics)}", flush=True)


if __name__ == "__main__":
    main()
