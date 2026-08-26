#!/usr/bin/env python3
"""R011 four-cell factorial smoke: A0B0,A0B1,A1B0,A1B1 x seed42 x 1 epoch.

One shared base_state, identical batch order, DS parity contract.
Runs local OR on a Colab VM (same code). Not scientific evidence.
"""

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

CELLS = [
    ("A0B0", "constructor", "xavier_g1.0"),
    ("A0B1", "constructor", "orthogonal"),
    ("A1B0", "historical_xavier", "xavier_g1.0"),
    ("A1B1", "historical_xavier", "orthogonal"),
]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["experiment"].get("mode") != "r011_factorial_smoke":
        raise SystemExit("mode must be r011_factorial_smoke")
    seed = int(cfg["experiment"]["seed"])
    if seed != 42:
        raise SystemExit("authorized smoke seed is 42")
    if int(cfg["training"]["epochs"]) != 1:
        raise SystemExit("smoke is 1 epoch")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    conf = load_conf()
    device = conf.resolve_device(cfg["training"]["device"])
    print(f"[Milestone] R011 factorial smoke start seed={seed} device={device}", flush=True)
    conf.write_environment(out / "environment.txt", device)

    hist = conf.load_historical_model_module()
    splits, vocab, data_manifest = conf.load_wikitext_splits(cfg["data"])
    bs = int(cfg["data"]["batch_size"])
    train_drop = bool(cfg["data"]["train_drop_last"])
    val_loader = DataLoader(splits["validation"], batch_size=bs, shuffle=False,
                            drop_last=bool(cfg["data"]["val_drop_last"]))
    test_loader = DataLoader(splits["test"], batch_size=bs, shuffle=False,
                             drop_last=bool(cfg["data"]["test_drop_last"]))

    def factory():
        return hist.DecoderOnlyTransformer(
            vocab_size=vocab, d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]), d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]), max_seq_len=int(cfg["data"]["seq_len"]),
        )

    seeds = derive_seeds(seed)
    perms = epoch_index_permutations(len(splits["train"]), 1, seeds.seed_shuffle)
    batch_rows = batch_order_records(perms, seed=seed, batch_size=bs, drop_last=train_drop)
    base, base_hashes = build_base_state(factory, seeds.seed_model, device="cpu")
    allow = attention_allowlist(base)
    assert_expected_allowlist_count(allow, n_layers=2)
    dump_json(out / "parameter_allowlist.json", {"allowlist": allow})
    write_csv(out / "base_state_hashes.csv",
              [{"name": k, "sha256": v} for k, v in sorted(base_hashes.items())])

    criterion = nn.CrossEntropyLoss()
    emb_rows, attn_rows, unchanged_rows, metrics = [], [], [], []
    emb_by_cell, attn_by_bfactor = {}, {}

    for cell_id, emb_mode, attn in CELLS:
        print(f"[Milestone] cell={cell_id} emb={emb_mode} attn={attn}", flush=True)
        model = clone_from_base_state(base, factory)
        apply_embedding_intervention(model, emb_mode, seeds)
        apply_attention_intervention(model, attn, seeds, allowlist=allow)
        assert_t0_factorial(model, base_hashes, allow, emb_mode)
        params = dict(model.named_parameters())
        emb_h = tensor_sha256(model.token_emb.weight)
        emb_by_cell[cell_id] = emb_h
        emb_rows.append({"cell": cell_id, "embedding_mode": emb_mode,
                         "base_sha256": base_hashes["token_emb.weight"],
                         "post_sha256": emb_h,
                         "changed": str(emb_h != base_hashes["token_emb.weight"]).lower()})
        for name in allow:
            h = tensor_sha256(params[name])
            attn_rows.append({"cell": cell_id, "b_factor": attn, "name": name, "sha256": h})
            attn_by_bfactor.setdefault(attn, {}).setdefault(name, set()).add(h)
        now = collect_named_tensors(model)
        for name, tensor in now.items():
            if name in allow or name == "token_emb.weight":
                continue
            d = tensor_sha256(tensor)
            if d != base_hashes[name]:
                raise SystemExit(f"non-attn drift {cell_id} {name}")
            unchanged_rows.append({"cell": cell_id, "name": name, "sha256": d})

        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                                weight_decay=float(cfg["training"]["weight_decay"]))
        order = perms[0]
        usable = order[: (len(order) // bs) * bs] if train_drop else order
        loader = DataLoader(splits["train"], batch_size=bs,
                            sampler=conf.EpochPermutationSampler(usable), drop_last=False)
        model.train()
        run = 0.0; n = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward(); opt.step()
            run += float(loss.item()) * x.size(0); n += x.size(0)
        val = conf.evaluate(model, val_loader, device, criterion)
        torch.save({"model": model.state_dict(), "epoch": 1, "val_loss": val},
                   out / "checkpoints" / f"{cell_id}_seed{seed}_best.pt")
        test = conf.evaluate(model, test_loader, device, criterion)
        print(f"  train={run/max(n,1):.4f} val={val:.4f} test_ppl={math.exp(min(test,20)):.2f}", flush=True)
        metrics.append({"cell": cell_id, "embedding_mode": emb_mode, "attention": attn,
                        "seed": seed, "best_epoch": 1, "val_ppl": math.exp(min(val, 20)),
                        "test_ppl": math.exp(min(test, 20)), "scientific_evidence": False})

    # DS parity contract
    assert emb_by_cell["A0B0"] == emb_by_cell["A0B1"] == base_hashes["token_emb.weight"], "A0 emb != base"
    assert emb_by_cell["A1B0"] == emb_by_cell["A1B1"], "A1 emb mismatch"
    assert emb_by_cell["A1B0"] != base_hashes["token_emb.weight"], "A1 emb == base"
    for bf in ("xavier_g1.0", "orthogonal"):
        for name, hset in attn_by_bfactor[bf].items():
            assert len(hset) == 1, f"B parity fail {bf} {name}"

    parity = [
        {"check": "A0B0_emb==A0B1_emb==base", "pass": str(emb_by_cell["A0B0"] == emb_by_cell["A0B1"] == base_hashes["token_emb.weight"]).lower()},
        {"check": "A1B0_emb==A1B1_emb", "pass": str(emb_by_cell["A1B0"] == emb_by_cell["A1B1"]).lower()},
        {"check": "A1_emb!=base", "pass": str(emb_by_cell["A1B0"] != base_hashes["token_emb.weight"]).lower()},
        {"check": "B0_attn_identical_A0B0_A1B0", "pass": "true"},
        {"check": "B1_attn_identical_A0B1_A1B1", "pass": "true"},
    ]
    write_csv(out / "embedding_hashes.csv", emb_rows)
    write_csv(out / "attention_factor_parity.csv", attn_rows)
    write_csv(out / "parity_summary.csv", parity)
    write_csv(out / "unchanged_parameter_hashes.csv", unchanged_rows)
    write_csv(out / "batch_order_hashes.csv", batch_rows)
    write_csv(out / "smoke_metrics.csv", metrics)
    dump_json(out / "resolved_config.json", cfg)
    dump_json(out / "dataset_manifest.json", data_manifest)
    (out / "commit.txt").write_text(conf.git_commit() + "\n", encoding="utf-8")
    (out / "DURABLE_MARKER.txt").write_text("R011_FACTORIAL_SMOKE_COMPLETE\n", encoding="utf-8")
    print("[Milestone] R011 factorial smoke complete", flush=True)


if __name__ == "__main__":
    main()
