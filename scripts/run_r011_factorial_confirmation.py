#!/usr/bin/env python3
"""R011 fresh four-cell factorial confirmation.

A0B0,A0B1,A1B0,A1B1 x seeds 42-46 x 15 epochs on ONE frozen host.
Does NOT pool R010 rows. Per-seed parity contract (DS 2026-08-26).
Resume by (cell, seed). Not merged to master.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
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
    collect_spectral,
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
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 15


def load_conf():
    spec = importlib.util.spec_from_file_location(
        "r010_runner", ROOT / "scripts" / "run_transformer_paired_confirmation.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hb(path: Path, **p):
    p.update(last_heartbeat=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
             gpu_temp=0, cpu_temp=0)
    path.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["experiment"].get("mode") != "r011_confirmation":
        raise SystemExit("mode must be r011_confirmation")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    conf = load_conf()
    device = conf.resolve_device(cfg["training"]["device"])
    conf.write_environment(out / "environment.txt", device)
    hb(out / "heartbeat.json", job_id="r011_confirmation", status="running",
       progress="starting", current_phase="load_data", host=platform.node())
    print(f"[Milestone] R011 confirmation start device={device}", flush=True)

    hist = conf.load_historical_model_module()
    splits, vocab, data_manifest = conf.load_wikitext_splits(cfg["data"])
    bs = int(cfg["data"]["batch_size"]); train_drop = bool(cfg["data"]["train_drop_last"])
    val_loader = DataLoader(splits["validation"], batch_size=bs, shuffle=False,
                            drop_last=bool(cfg["data"]["val_drop_last"]))
    test_loader = DataLoader(splits["test"], batch_size=bs, shuffle=False,
                             drop_last=bool(cfg["data"]["test_drop_last"]))

    def factory():
        return hist.DecoderOnlyTransformer(
            vocab_size=vocab, d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]), d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]), max_seq_len=int(cfg["data"]["seq_len"]))

    metrics = read_csv(out / "per_seed.csv")
    done = {(r["cell"], int(r["seed"])) for r in metrics}
    emb_rows = read_csv(out / "embedding_hashes.csv")
    attn_rows = read_csv(out / "attention_factor_parity.csv")
    unchanged = read_csv(out / "unchanged_parameter_hashes.csv")
    batch_rows = read_csv(out / "batch_order_hashes.csv")
    parity_rows = read_csv(out / "parity_summary.csv")
    t0_spec = read_csv(out / "t0_spectral.csv")
    best_spec = read_csv(out / "best_epoch_spectral.csv")
    final_spec = read_csv(out / "final_epoch_spectral.csv")
    criterion = nn.CrossEntropyLoss()
    total = len(SEEDS) * len(CELLS); finished = len(done)

    for seed in SEEDS:
        seeds = derive_seeds(seed)
        perms = epoch_index_permutations(len(splits["train"]), EPOCHS, seeds.seed_shuffle)
        if not any(int(r["seed"]) == seed for r in batch_rows):
            batch_rows.extend(batch_order_records(perms, seed=seed, batch_size=bs, drop_last=train_drop))
        base, base_hashes = build_base_state(factory, seeds.seed_model, device="cpu")
        allow = attention_allowlist(base)
        assert_expected_allowlist_count(allow, n_layers=int(cfg["model"]["n_layers"]))
        if not (out / "parameter_allowlist.json").exists():
            dump_json(out / "parameter_allowlist.json", {"allowlist": allow})
        emb_seen = {}
        attn_seen = {}
        for cell_id, emb_mode, attn in CELLS:
            key = (cell_id, seed)
            if key in done:
                print(f"[resume] skip {cell_id} seed={seed}", flush=True); continue
            print(f"[Milestone] {cell_id} seed={seed} emb={emb_mode} attn={attn}", flush=True)
            hb(out / "heartbeat.json", job_id="r011_confirmation", status="running",
               progress=f"{finished}/{total}", current_phase=f"{cell_id}/seed{seed}", host=platform.node())
            model = clone_from_base_state(base, factory)
            apply_embedding_intervention(model, emb_mode, seeds)
            apply_attention_intervention(model, attn, seeds, allowlist=allow)
            assert_t0_factorial(model, base_hashes, allow, emb_mode)
            emb_h = tensor_sha256(model.token_emb.weight)
            emb_seen[cell_id] = emb_h
            emb_rows.append({"cell": cell_id, "seed": seed, "embedding_mode": emb_mode,
                             "base_sha256": base_hashes["token_emb.weight"], "post_sha256": emb_h,
                             "changed": str(emb_h != base_hashes["token_emb.weight"]).lower()})
            params = dict(model.named_parameters())
            attn_seen.setdefault(attn, {})
            for name in allow:
                h = tensor_sha256(params[name])
                attn_rows.append({"cell": cell_id, "seed": seed, "b_factor": attn, "name": name, "sha256": h})
                attn_seen[attn].setdefault(name, set()).add(h)
            now = collect_named_tensors(model)
            for name, tensor in now.items():
                if name in allow or name == "token_emb.weight":
                    continue
                d = tensor_sha256(tensor)
                if d != base_hashes[name]:
                    raise SystemExit(f"non-attn drift {cell_id} seed{seed} {name}")
                unchanged.append({"cell": cell_id, "seed": seed, "name": name, "sha256": d})
            t0_spec.extend(collect_spectral(model, allow, state="t0", method=cell_id, seed=seed))

            model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                                    weight_decay=float(cfg["training"]["weight_decay"]))
            best_val = math.inf; best_ep = 0; last_val = math.inf
            best_path = out / "checkpoints" / f"{cell_id}_seed{seed}_best.pt"
            for ep, order in enumerate(perms, start=1):
                usable = order[: (len(order) // bs) * bs] if train_drop else order
                loader = DataLoader(splits["train"], batch_size=bs,
                                    sampler=conf.EpochPermutationSampler(usable), drop_last=False)
                model.train(); run = 0.0; n = 0
                for x, y in loader:
                    x = x.to(device); y = y.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss.backward(); opt.step()
                    run += float(loss.item()) * x.size(0); n += x.size(0)
                last_val = conf.evaluate(model, val_loader, device, criterion)
                print(f"  ep {ep}/{EPOCHS} train={run/max(n,1):.4f} val={last_val:.4f}", flush=True)
                if last_val < best_val:
                    best_val = last_val; best_ep = ep
                    torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": last_val}, best_path)
            final_spec.extend(collect_spectral(model, allow, state="final_epoch", method=cell_id, seed=seed))
            ck = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            best_spec.extend(collect_spectral(model, allow, state="best_validation", method=cell_id, seed=seed))
            test_loss = conf.evaluate(model, test_loader, device, criterion)
            metrics.append({"cell": cell_id, "embedding_mode": emb_mode, "attention": attn, "seed": seed,
                            "best_epoch": best_ep, "best_val_ppl": math.exp(min(best_val, 20)),
                            "final_val_ppl": math.exp(min(last_val, 20)),
                            "test_ppl": math.exp(min(test_loss, 20))})
            done.add(key); finished += 1
            for pth, rws in [("per_seed.csv", metrics), ("embedding_hashes.csv", emb_rows),
                             ("attention_factor_parity.csv", attn_rows),
                             ("unchanged_parameter_hashes.csv", unchanged),
                             ("batch_order_hashes.csv", batch_rows),
                             ("t0_spectral.csv", t0_spec), ("best_epoch_spectral.csv", best_spec),
                             ("final_epoch_spectral.csv", final_spec)]:
                write_csv(out / pth, rws)
            if int(cfg["training"].get("cooldown_seconds", 15)) > 0:
                time.sleep(int(cfg["training"]["cooldown_seconds"]))

        # per-seed parity (DS 7 checks)
        assert emb_seen.get("A0B0") == emb_seen.get("A0B1") == base_hashes["token_emb.weight"] or ("A0B0", seed) in done
        if all((c, seed) in done for c, _, _ in CELLS):
            checks = [
                ("A0B0_emb==A0B1_emb==base", emb_seen["A0B0"] == emb_seen["A0B1"] == base_hashes["token_emb.weight"]),
                ("A1B0_emb==A1B1_emb!=base", emb_seen["A1B0"] == emb_seen["A1B1"] != base_hashes["token_emb.weight"]),
                ("B0_attn_A0B0==A1B0", all(len(s) == 1 for s in attn_seen["xavier_g1.0"].values())),
                ("B1_attn_A0B1==A1B1", all(len(s) == 1 for s in attn_seen["orthogonal"].values())),
            ]
            for name, ok in checks:
                parity_rows.append({"seed": seed, "check": name, "pass": str(bool(ok)).lower()})
                if not ok:
                    raise SystemExit(f"parity fail seed {seed}: {name}")
            write_csv(out / "parity_summary.csv", parity_rows)

    dump_json(out / "resolved_config.json", cfg)
    dump_json(out / "dataset_manifest.json", data_manifest)
    (out / "DURABLE_MARKER.txt").write_text("R011_FACTORIAL_CONFIRMATION_COMPLETE\n", encoding="utf-8")
    hb(out / "heartbeat.json", job_id="r011_confirmation", status="complete",
       progress=f"{finished}/{total}", current_phase="done", host=platform.node())
    print(f"[Milestone] R011 confirmation complete rows={len(metrics)}", flush=True)


if __name__ == "__main__":
    main()
