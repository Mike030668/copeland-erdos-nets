#!/usr/bin/env python3
"""R011 bounded smoke: A1B0 + A1B1 × seed 42 × 1 epoch. Not scientific evidence."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from copeland_erdos_nets.r010_protocol import (
    apply_attention_intervention,
    apply_embedding_intervention,
    assert_expected_allowlist_count,
    assert_t0_factorial,
    attention_allowlist,
    batch_order_records,
    build_base_state,
    clone_from_base_state,
    collect_named_tensors,
    derive_seeds,
    dump_json,
    epoch_index_permutations,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def load_conf():
    spec = importlib.util.spec_from_file_location(
        "r010_runner", ROOT / "scripts" / "run_transformer_paired_confirmation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--r010-dir", default=str(ROOT / ".tmp/r010_phase1"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["experiment"].get("mode") != "r011_smoke":
        raise SystemExit("this script is r011_smoke only")
    if int(cfg["training"]["epochs"]) != 1:
        raise SystemExit("smoke is 1 epoch only")
    seed = int(cfg["experiment"]["seed"])
    if seed != 42:
        raise SystemExit("authorized smoke seed is 42 only")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    r010 = Path(args.r010_dir)
    conf = load_conf()
    device = conf.resolve_device(cfg["training"]["device"])
    print(f"[Milestone] R011 smoke start seed={seed} device={device}", flush=True)
    conf.write_environment(out / "environment.txt", device)

    hist = conf.load_historical_model_module()
    splits, vocab, data_manifest = conf.load_wikitext_splits(cfg["data"])
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
            vocab_size=vocab,
            d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]),
            d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]),
            max_seq_len=int(cfg["data"]["seq_len"]),
        )

    seeds = derive_seeds(seed)
    perms = epoch_index_permutations(len(splits["train"]), 1, seeds.seed_shuffle)
    batch_rows = batch_order_records(perms, seed=seed, batch_size=bs, drop_last=train_drop)
    base, base_hashes = build_base_state(factory, seeds.seed_model, device="cpu")
    allow = attention_allowlist(base)
    assert_expected_allowlist_count(allow, n_layers=2)
    dump_json(out / "parameter_allowlist.json", {"allowlist": allow})
    write_csv(out / "base_state_hashes.csv", [{"name": k, "sha256": v} for k, v in sorted(base_hashes.items())])

    r010_changed = read_csv(r010 / "changed_parameter_hashes.csv")
    r010_attn = {}
    for r in r010_changed:
        if int(r["seed"]) != seed:
            continue
        r010_attn.setdefault(r["method"], {})[r["name"]] = r["post_intervention_sha256"]

    criterion = nn.CrossEntropyLoss()
    embedding_rows = []
    attn_parity = []
    unchanged_rows = []
    metrics = []
    emb_hashes = {}

    for cell in cfg["experiment"]["cells"]:
        cell_id = cell["cell"]
        emb_mode = cell["embedding_mode"]
        attn = cell["attention"]
        print(f"[Milestone] cell={cell_id} emb={emb_mode} attn={attn}", flush=True)
        model = clone_from_base_state(base, factory)
        apply_embedding_intervention(model, emb_mode, seeds)
        apply_attention_intervention(model, attn, seeds, allowlist=allow)
        _u, changed = assert_t0_factorial(model, base_hashes, allow, emb_mode)
        emb_h = tensor_sha256(model.token_emb.weight)
        emb_hashes[cell_id] = emb_h
        embedding_rows.append(
            {
                "cell": cell_id,
                "seed": seed,
                "name": "token_emb.weight",
                "base_sha256": base_hashes["token_emb.weight"],
                "post_sha256": emb_h,
                "changed": str(emb_h != base_hashes["token_emb.weight"]).lower(),
            }
        )
        params = dict(model.named_parameters())
        for name in allow:
            got = tensor_sha256(params[name])
            exp = r010_attn.get(attn, {}).get(name, "")
            attn_parity.append(
                {
                    "cell": cell_id,
                    "seed": seed,
                    "name": name,
                    "r011_sha256": got,
                    "r010_sha256": exp,
                    "match": str(got == exp).lower(),
                }
            )
            if exp and got != exp:
                raise SystemExit(f"attention hash mismatch {cell_id} {name}")
        now = collect_named_tensors(model)
        for name, tensor in now.items():
            if name in allow or name == "token_emb.weight":
                continue
            digest = tensor_sha256(tensor)
            if digest != base_hashes[name]:
                raise SystemExit(f"non-attn drift {name}")
            unchanged_rows.append({"cell": cell_id, "seed": seed, "name": name, "sha256": digest})

        model.to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        order = perms[0]
        usable = order[: (len(order) // bs) * bs] if train_drop else order
        loader = DataLoader(
            splits["train"], batch_size=bs,
            sampler=conf.EpochPermutationSampler(usable), drop_last=False,
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
        val_loss = conf.evaluate(model, val_loader, device, criterion)
        ckpt = out / "checkpoints" / f"{cell_id}_seed{seed}_best.pt"
        torch.save({"model": model.state_dict(), "epoch": 1, "val_loss": val_loss}, ckpt)
        test_loss = conf.evaluate(model, test_loader, device, criterion)
        print(f"  train={running/max(n,1):.4f} val={val_loss:.4f} test_ppl={math.exp(min(test_loss,20)):.2f}", flush=True)
        metrics.append(
            {
                "cell": cell_id,
                "seed": seed,
                "best_epoch": 1,
                "best_val_loss": val_loss,
                "best_val_ppl": math.exp(min(val_loss, 20)),
                "test_loss": test_loss,
                "test_ppl": math.exp(min(test_loss, 20)),
                "scientific_evidence": False,
            }
        )

    if emb_hashes.get("A1B0") != emb_hashes.get("A1B1"):
        raise SystemExit("A1B0/A1B1 token_emb hashes differ")

    write_csv(out / "embedding_hashes.csv", embedding_rows)
    write_csv(out / "attention_hash_parity.csv", attn_parity)
    write_csv(out / "unchanged_parameter_hashes.csv", unchanged_rows)
    write_csv(out / "batch_order_hashes.csv", batch_rows)
    write_csv(out / "smoke_metrics.csv", metrics)
    dump_json(out / "resolved_config.json", cfg)
    dump_json(out / "dataset_manifest.json", data_manifest)
    (out / "commit.txt").write_text(conf.git_commit() + "\n", encoding="utf-8")
    print("[Milestone] R011 smoke complete", flush=True)


if __name__ == "__main__":
    main()
