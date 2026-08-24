#!/usr/bin/env python3
"""R010 paired attention-init confirmation runner.

New causal path. Does not change historical run_transformer_screening.py.
Authorized now: smoke = 4 methods x seed 42 x <=2 epochs.
Five-seed 15-epoch confirmation is NOT authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
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
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="hard cap (smoke <=2)")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    if cfg["experiment"].get("mode") != "smoke":
        raise SystemExit("only smoke mode is authorized on this runner right now")

    seed = int(cfg["experiment"]["seed"])
    methods = list(cfg["experiment"]["methods"])
    if methods != list(METHODS):
        raise SystemExit(f"methods must be exactly {METHODS}")
    epochs = int(cfg["training"]["epochs"])
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    if epochs > 2:
        raise SystemExit("smoke authorization is max 2 epochs")

    device = resolve_device(cfg["training"]["device"])
    hist = load_historical_model_module()
    splits, vocab_size, data_manifest = load_wikitext_splits(cfg["data"])
    seeds = derive_seeds(seed)

    bs = int(cfg["data"]["batch_size"])
    train_drop = bool(cfg["data"]["train_drop_last"])
    val_loader = DataLoader(
        splits["validation"],
        batch_size=bs,
        shuffle=False,
        drop_last=bool(cfg["data"]["val_drop_last"]),
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=bs,
        shuffle=False,
        drop_last=bool(cfg["data"]["test_drop_last"]),
    )
    perms = epoch_index_permutations(len(splits["train"]), epochs, seeds.seed_shuffle)
    batch_rows = batch_order_records(perms, seed=seed, batch_size=bs, drop_last=train_drop)

    def factory():
        return hist.DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]),
            d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]),
            max_seq_len=int(cfg["data"]["seq_len"]),
        )

    base, base_hashes = build_base_state(factory, seeds.seed_model, device="cpu")
    allow = attention_allowlist(base)
    assert_expected_allowlist_count(allow, n_layers=int(cfg["model"]["n_layers"]))

    criterion = nn.CrossEntropyLoss()
    metrics = []
    t0_spectral = []
    best_spectral = []
    final_spectral = []
    unchanged_rows = []
    changed_hash_rows = []
    protocol_deviations: list[str] = []

    for method in methods:
        print(f"=== method {method} seed={seed} ===", flush=True)
        model = clone_from_base_state(base, factory)
        meta = apply_attention_intervention(
            model,
            method,
            seeds,
            allowlist=allow,
            ce_m=int(cfg["init"]["ce_m"]),
            ce_offset_blocks=int(cfg["init"]["ce_offset_blocks"]),
        )
        unchanged, changed_rows = assert_t0_invariance(model, base_hashes, allow)
        for name, digest in unchanged.items():
            unchanged_rows.append(
                {"method": method, "seed": seed, "name": name, "sha256": digest}
            )
        for row in changed_rows:
            changed_hash_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "name": row["name"],
                    "base_sha256": row["base_sha256"],
                    "post_intervention_sha256": row["post_intervention_sha256"],
                    "changed": row["changed"],
                }
            )
        t0_spectral.extend(
            collect_spectral(model, allow, state="t0", method=method, seed=seed)
        )
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
            usable = order
            if train_drop:
                usable = order[: (len(order) // bs) * bs]
            loader = DataLoader(
                splits["train"],
                batch_size=bs,
                sampler=EpochPermutationSampler(usable),
                drop_last=False,
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
                f"  epoch {epoch} train_loss={running/max(n,1):.4f} val_loss={last_val:.4f}",
                flush=True,
            )
            if last_val < best_val:
                best_val = last_val
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": last_val},
                    best_path,
                )
        # final spectral
        final_spectral.extend(
            collect_spectral(model, allow, state="final_epoch", method=method, seed=seed)
        )
        # reload best for test + best spectral
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

    # artifacts
    dump_json(out / "resolved_config.json", cfg)
    dump_json(out / "parameter_allowlist.json", {"allowlist": allow})
    dump_json(
        out / "dataset_manifest.json",
        {
            **data_manifest,
            "train_drop_last": train_drop,
            "val_drop_last": cfg["data"]["val_drop_last"],
            "test_drop_last": cfg["data"]["test_drop_last"],
        },
    )
    write_csv(out / "base_state_hashes.csv", [{"name": k, "sha256": v} for k, v in sorted(base_hashes.items())])
    write_csv(out / "unchanged_parameter_hashes.csv", unchanged_rows)
    write_csv(out / "changed_parameter_hashes.csv", changed_hash_rows)
    write_csv(out / "batch_order_hashes.csv", batch_rows)
    write_csv(out / "smoke_metrics.csv", metrics)
    write_csv(out / "t0_spectral.csv", t0_spectral)
    write_csv(out / "best_epoch_spectral.csv", best_spectral)
    write_csv(out / "final_epoch_spectral.csv", final_spectral)
    (out / "commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    (out / "rng_policy.md").write_text(
        f"""# R010 RNG policy

```text
seed_model     = experiment_seed + {seeds.seed_model - seed}
seed_attention = experiment_seed + {seeds.seed_attention - seed}
seed_shuffle   = experiment_seed + {seeds.seed_shuffle - seed}
```

Each method uses an isolated attention stream:
`seed_attention_method = seed_attention + METHOD_SALT * 1000003`.

Train order is a precomputed permutation from `seed_shuffle` (same for all methods).
Model/attention RNG does not advance the shuffle generator.
""",
        encoding="utf-8",
    )
    (out / "protocol_deviations.md").write_text(
        "# Protocol deviations\n\n"
        + ("None.\n" if not protocol_deviations else "\n".join(f"- {d}" for d in protocol_deviations) + "\n"),
        encoding="utf-8",
    )
    import platform
    import numpy as np
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
        import subprocess as _sp
        driver = _sp.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = "unavailable"
    (out / "environment.txt").write_text(
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
    dump_json(
        out / "smoke_summary.json",
        {
            "mode": "smoke",
            "scientific_evidence": False,
            "primary_baseline": PRIMARY_BASELINE,
            "metrics": metrics,
        },
    )
    print("R010 smoke complete", flush=True)


if __name__ == "__main__":
    main()
