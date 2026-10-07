#!/usr/bin/env python3
"""Authorized Gate BC NONCANONICAL_SMOKE runner; canonical remains fail-closed."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from copeland_erdos_nets.gate_bc_protocol import (
    CONDITIONS, HardGateError, TokenScaleMixin, atomic_checkpoint, atomic_json,
    audit_cells, batch_schedule_receipt, construct_cells, epoch_telemetry,
    execution_gate, gradient_stats, tensor_digest,
)
from copeland_erdos_nets.r010_protocol import (
    apply_attention_intervention, attention_allowlist, build_base_state,
    collect_named_tensors, derive_seeds, epoch_index_permutations, hash_int_sequence,
)

ROOT = Path(__file__).resolve().parents[1]


def load_conf():
    spec = importlib.util.spec_from_file_location("gate_bc_inherited", ROOT/"scripts/run_transformer_paired_confirmation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def runtime_gate(freeze, output):
    import datasets, transformers
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception:
        driver = "unavailable"
    got = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda or "none", "numpy": np.__version__,
           "datasets": datasets.__version__, "transformers": transformers.__version__, "driver": driver}
    wrong = [k for k in freeze if str(got.get(k)) != str(freeze[k])]
    atomic_json(output/"runtime_assertion.json", {"expected": freeze, "actual": got, "mismatches": wrong, "pass": not wrong})
    if wrong:
        raise HardGateError("runtime mismatch: "+repr(wrong))


def load_data(cfg, output):
    """Inherited exact preprocessing, with fresh content/revision receipts."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    conf = load_conf()
    from huggingface_hub import HfApi
    # Fresh resolved revisions are pinned for this attempt if exposed. They do
    # not assert continuity with an unretained historical upstream revision.
    dataset_revision, tokenizer_revision, resolution_errors = None, None, []
    try:
        dataset_revision = HfApi().dataset_info(cfg["dataset"]).sha
    except Exception as error:
        resolution_errors.append("dataset: "+type(error).__name__)
    try:
        tokenizer_revision = HfApi().model_info(cfg["tokenizer"]).sha
    except Exception as error:
        resolution_errors.append("tokenizer: "+type(error).__name__)
    raw = load_dataset(cfg["dataset"], cfg["config_name"], revision=dataset_revision)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], revision=tokenizer_revision)
    tok.pad_token = tok.eos_token
    splits, receipts = {}, {}
    for split in ("train", "validation", "test"):
        tokenized = raw[split].map(lambda b: tok(b["text"]), batched=True, remove_columns=raw[split].column_names)
        ids = [i for row in tokenized["input_ids"] for i in row]
        ds = conf.TokenizedDataset(ids, int(cfg["seq_len"]))
        splits[split] = ds
        token_array = np.asarray(ids, dtype="<i8")
        chunks = np.asarray(ds.chunks, dtype="<i8")
        receipts[split] = {"raw_rows": len(raw[split]), "token_count": len(ids), "chunk_count": len(ds),
                           "token_sha256_le_i64": hashlib.sha256(token_array.tobytes()).hexdigest(),
                           "chunk_sha256_le_i64": hashlib.sha256(chunks.tobytes()).hexdigest(),
                           "dataset_fingerprint": raw[split]._fingerprint}
    # Exposed revision only: do not convert a content fingerprint into a commit.
    manifest = {"dataset": cfg["dataset"], "config_name": cfg["config_name"],
                "dataset_revision_exposed": dataset_revision, "tokenizer_revision_exposed": tokenizer_revision or tok.init_kwargs.get("_commit_hash"),
                "revision_resolution_errors": resolution_errors,
                "historical_revision": "UNRETAINED_NO_CONTINUITY_CLAIM", "vocab": tok.vocab_size,
                "seq_len": cfg["seq_len"], "splits": receipts}
    expected = cfg["accepted_structure"]
    actual = {"vocab": tok.vocab_size, "seq_len": cfg["seq_len"],
              **{f"n_{s}_chunks": len(splits[s]) for s in splits}}
    manifest["structure_expected"] = expected
    manifest["structure_actual"] = actual
    manifest["structure_pass"] = actual == expected
    atomic_json(output/"data_manifest.json", manifest)
    structure_gate(actual, expected)
    return splits, tok.vocab_size


def structure_gate(actual, expected):
    if actual != expected:
        raise HardGateError("data structure discrepancy STOP_FOR_DS")


def config_gate(cfg):
    """Reject protocol-changing config edits before runtime/data/RNG activity."""
    fixed = {
        "model": {"d_model": 128, "n_heads": 4, "d_ff": 512, "n_layers": 2},
        "training": {"epochs": 1, "schedule_epochs": 15, "lr": 0.0005, "weight_decay": 0.01, "device": "cuda", "held_out_test": False},
        "rng_policy": {"seed_model_offset": 0, "seed_shuffle_offset": 20011, "seed_embedding_redraw_offset": 30013},
        "parity": {"rms_abs_strict": 1e-6, "cosine_min": 0.999999, "normalized_max_abs_diff_max": 1e-4, "matched_forward_rtol": 1e-6, "matched_forward_atol": 1e-7},
        "runtime_freeze": {"gpu": "Tesla T4", "python": "3.13.15", "torch": "2.11.0+cu128", "cuda": "12.8", "numpy": "2.1.3", "datasets": "4.0.0", "transformers": "5.15.1", "driver": "580.82.07"},
    }
    data = {"dataset": "Salesforce/wikitext", "config_name": "wikitext-2-raw-v1", "tokenizer": "gpt2",
            "batch_size": 32, "seq_len": 128, "train_drop_last": True, "val_drop_last": True, "test_drop_last": False,
            "accepted_structure": {"vocab": 50257, "seq_len": 128, "n_train_chunks": 18686, "n_validation_chunks": 1931, "n_test_chunks": 2213}}
    if any(cfg.get(k)!=v for k,v in fixed.items()) or cfg.get("data") != data:
        raise HardGateError("frozen config discrepancy")
    if cfg.get("experiment", {}).get("conditions") != list(CONDITIONS) or cfg.get("telemetry") != {"enabled": True, "read_only": True}:
        raise HardGateError("frozen conditions/telemetry discrepancy")


class DriveUpdates:
    """Update exact existing placeholders only; verify every upload by readback."""
    def __init__(self, out, prefix):
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials
        ga = GoogleAuth()
        ga.credentials = ServiceAccountCredentials.from_json_keyfile_name("/content/sa.json", ["https://www.googleapis.com/auth/drive"])
        self.drive, self.out, self.prefix = GoogleDrive(ga), out, prefix
        parent = None
        for title in ("agent-rules-tree-control", "research", "copeland-erdos-nets_drive", "exchange"):
            q = f"title='{title}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent is not None:
                q += f" and '{parent}' in parents"
            found = self.drive.ListFile({"q": q}).GetList()
            if len(found) != 1:
                raise HardGateError("Drive folder not unique: "+title)
            parent = found[0]["id"]
        self.parent = parent
        self.receipts = []

    def upload(self, path, suffix):
        title = self.prefix+suffix
        found = self.drive.ListFile({"q": f"title='{title}' and '{self.parent}' in parents and trashed=false"}).GetList()
        if len(found) != 1:
            raise HardGateError("missing/nonunique Drive placeholder: "+title)
        f = self.drive.CreateFile({"id": found[0]["id"]})
        f.SetContentFile(str(path))
        f.Upload()
        verify = self.out/(".readback_"+title)
        self.drive.CreateFile({"id": found[0]["id"]}).GetContentFile(str(verify))
        digest, size = sha_file(path), Path(path).stat().st_size
        passed = sha_file(verify)==digest and verify.stat().st_size==size
        verify.unlink()
        self.receipts.append({"file": title, "sha256": digest, "size": size, "readback_pass": passed, "time_unix": time.time()})
        atomic_json(self.out/"drive_durability.json", self.receipts)
        if not passed:
            raise HardGateError("Drive readback mismatch "+title)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    crit = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += float(loss)*x.size(0)
        count += x.size(0)
    if count == 0:
        raise HardGateError("empty validation loader")
    return total/count


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def train_epoch(model, optimizer, loader, device, telemetry_on, progress=None):
    """Shared production/synthetic training step; read-only telemetry optional."""
    model.train()
    start = model.token_emb.weight.detach().double().clone() if telemetry_on else None
    gradients, losses = [], []
    total, count = 0.0, 0
    crit = torch.nn.CrossEntropyLoss()
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if telemetry_on:
            gradients.append(gradient_stats(model))
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        total += value*x.size(0)
        count += x.size(0)
        if progress and (step % 32 == 0 or step == len(loader)):
            progress(step, len(loader))
    if not count:
        raise HardGateError("empty training loader")
    return total/count, losses, epoch_telemetry(model, start, gradients) if telemetry_on else None


def train_cell(model, condition, splits, perms, cfg, out, seed, device, live, update_state, drive, attempt_id, source_sha):
    conf = load_conf()
    bs = cfg["data"]["batch_size"]
    val = DataLoader(splits["validation"], batch_size=bs, shuffle=False, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    best, best_epoch = math.inf, 0
    best_path = out/"checkpoints"/(condition+"_best.pt")
    current_path = out/"checkpoints"/(condition+"_current.pt")
    for epoch in range(1, cfg["training"]["epochs"]+1):
        order = perms[epoch-1]
        used = order[:len(order)//bs*bs]
        loader = DataLoader(splits["train"], batch_size=bs, sampler=conf.EpochPermutationSampler(used), drop_last=False)
        train, losses, diag = train_epoch(model, optimizer, loader, device, True,
            lambda step,total: update_state("running", condition, epoch-1, step=step, total_steps=total))
        vl = validation_loss(model, val, device)
        if not math.isfinite(train) or not math.isfinite(vl):
            raise HardGateError("nonfinite training/validation loss")
        improved = vl < best
        if improved:
            best, best_epoch = vl, epoch
            atomic_checkpoint(best_path, {"model": model.state_dict(), "alpha": model.token_alpha,
                              "condition": condition, "seed": seed, "epoch": epoch, "validation_loss": vl,
                              "attempt_id": attempt_id, "source_sha": source_sha})
            if drive:
                drive.upload(best_path, "_best_"+condition+".pt")
        atomic_checkpoint(current_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng": rng_state(), "alpha": model.token_alpha, "epoch": epoch,
                          "condition": condition, "seed": seed, "best_epoch": best_epoch,
                          "attempt_id": attempt_id, "source_sha": source_sha,
                          "best_validation_loss": best, "status": "NONCANONICAL_SMOKE",
                          "resume_policy": "same-live-VM-only; VM death requires whole fresh seed"})
        if drive:
            drive.upload(current_path, "_checkpoint_"+condition+".pt")
        live.append({"condition": condition, "seed": seed, "epoch": epoch,
                     "actual_batch_order_sha256": hash_int_sequence(used),
                     "train_loss": train, "validation_loss": vl, "validation_ppl": math.exp(min(vl, 20)),
                     **diag, "best_epoch": best_epoch, "is_best": improved,
                     "is_final": epoch==cfg["training"]["epochs"], "status": "NONCANONICAL_SMOKE"})
        write_csv(out/"live.csv", live)
        update_state("running", condition, epoch)
        if drive:
            drive.upload(out/"live.csv", "_live.csv")
        print(f"NONCANONICAL_SMOKE {condition} epoch={epoch} complete", flush=True)
    # No held-out-test loader/evaluation or condition contrast in smoke.
    return {"condition": condition, "seed": seed, "best_epoch": best_epoch,
            "selected_checkpoint_sha256": sha_file(best_path), "current_checkpoint_sha256": sha_file(current_path),
            "selected_checkpoint_bytes": best_path.stat().st_size, "current_checkpoint_bytes": current_path.stat().st_size,
            "attempt_id": attempt_id, "source_sha": source_sha,
            "test_evaluations": 0, "status": "NONCANONICAL_SMOKE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--attempt-id")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    # Before imports/data/model/RNG manipulation: close all canonical routes.
    execution_gate(args.seed, args.mode, cfg["training"]["epochs"])
    config_gate(cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"attempt.json").exists():
        raise HardGateError("output attempt already exists; no overwrite/splice")
    (out/"checkpoints").mkdir(exist_ok=True)
    attempt = args.attempt_id or str(uuid.uuid4())
    drive = None
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() if (ROOT/".git").exists() else os.environ.get("GATE_BC_SOURCE_BRANCH", "UNEXPOSED")
    source_sha = os.environ.get("GATE_BC_SOURCE_SHA", os.environ.get("CE_GIT_SHA", "UNEXPOSED"))
    atomic_json(out/"attempt.json", {"attempt_id": attempt, "seed": args.seed, "mode": "NONCANONICAL_SMOKE",
                "branch": branch, "source_sha": source_sha, "host": platform.node(),
                "session": os.environ.get("GATE_BC_SESSION", "UNEXPOSED"), "started_unix": time.time(),
                "canonical_seeds": "67-71 RESERVED_UNTOUCHED", "cross_vm_resume": "FORBIDDEN"})
    atomic_json(out/"resolved_config.json", cfg)
    completed_conditions = []
    expected_steps = cfg["data"]["accepted_structure"]["n_train_chunks"]//cfg["data"]["batch_size"]
    def update_state(status, condition="", epoch=0, reason="", step=0, total_steps=None):
        atomic_json(out/"state.json", {"status": status, "condition": condition, "epoch": epoch,
                    "current_condition": condition, "condition_index": CONDITIONS.index(condition)+1 if condition in CONDITIONS else (5 if status=="completed" else 0),
                    "total_conditions": 5, "total_epochs": 1, "schedule_epochs": 15,
                    "step": step, "total_steps": expected_steps if total_steps is None else total_steps, "completed_conditions": list(completed_conditions),
                    "completed_condition_count": len(completed_conditions), "remaining_conditions": 5-len(completed_conditions),
                    "seed": args.seed, "attempt_id": attempt, "mode": "NONCANONICAL_SMOKE",
                    "source_sha": source_sha, "time_unix": time.time(), "reason": reason})
        if drive:
            drive.upload(out/"state.json", "_state.json")
    try:
        if os.environ.get("GATE_BC_DRIVE_LIVE") == "1":
            drive = DriveUpdates(out, "gate_bc_smoke_seed_1067")
        update_state("preflight")
        runtime_gate(cfg["runtime_freeze"], out)
        conf = load_conf()
        splits, vocab = load_data(cfg["data"], out)
        hist = conf.load_historical_model_module()
        class Model(TokenScaleMixin, hist.DecoderOnlyTransformer):
            pass
        def factory():
            return Model(vocab_size=vocab, max_seq_len=cfg["data"]["seq_len"], **cfg["model"])
        seeds = derive_seeds(args.seed)
        atomic_json(out/"rng_policy.json", {"seed": args.seed, "model": seeds.seed_model,
                    "attention": seeds.seed_attention, "shuffle": seeds.seed_shuffle, "embedding_redraw": "DISABLED"})
        atomic_json(out/"optimizer_policy.json", {"class": "AdamW", "lr": 0.0005, "weight_decay": 0.01,
                    "betas": [0.9,0.999], "eps": 1e-8, "amsgrad": False, "AMP": False,
                    "scheduler": "NONE", "gradient_clipping": "NONE", "selection": "earliest strict validation improvement",
                    "test_evaluation": "NONE_IN_SMOKE"})
        perms = epoch_index_permutations(len(splits["train"]), 15, seeds.seed_shuffle)
        batch = batch_schedule_receipt(perms, len(splits["train"]), cfg["data"]["batch_size"])
        write_csv(out/"batch_order_pre_gate.csv", batch)
        atomic_json(out/"batch_order_gate.json", {"pass": True, "epochs": 15, "conditions": 5, "rows": 75, "before_optimization": True})
        base, _ = build_base_state(factory, seeds.seed_model, device="cpu")
        apply_attention_intervention(base, "xavier_g1.0", seeds, allowlist=attention_allowlist(base))
        hashes = [{"name": n, "sha256": tensor_digest(t), "dtype": str(t.dtype), "shape": list(t.shape)} for n,t in collect_named_tensors(base).items()]
        atomic_json(out/"base_state_hashes.json", hashes)
        atomic_json(out/"full_base_state_digest.json", {"algorithm": "sha256 canonical JSON ordered named tensor exact bytes/dtype/shape receipts",
                    "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        cells, construction = construct_cells(base, vocab, cfg["model"]["d_model"])
        write_csv(out/"construction.csv", construction)
        fixture = torch.arange(2*cfg["data"]["seq_len"], dtype=torch.long).reshape(2,-1) % vocab
        gates = audit_cells(base, cells, construction, fixture)
        atomic_json(out/"parity_gates.json", gates)
        unchanged = [{"condition": c, "name": n, "sha256": tensor_digest(t)}
                     for c,m in cells.items() for n,t in collect_named_tensors(m).items() if n!="token_emb.weight"]
        atomic_json(out/"unchanged_tensor_hashes.json", unchanged)
        # CPU fixture proof imported from the checked-in testable public helper.
        equivalence = telemetry_equivalence_receipt()
        atomic_json(out/"telemetry_equivalence.json", equivalence)
        if not equivalence["pass"]:
            raise HardGateError("synthetic telemetry complete-state mismatch")
        live, metrics = [], []
        device = torch.device("cuda")
        del base
        for condition in CONDITIONS:
            update_state("running", condition, 0)
            model = cells.pop(condition).to(device)
            metrics.append(train_cell(model, condition, splits, perms, cfg, out, args.seed, device, live, update_state, drive, attempt, source_sha))
            completed_conditions.append(condition)
            update_state("running", condition, 1, step=expected_steps, total_steps=expected_steps)
            del model
            torch.cuda.empty_cache()
        write_csv(out/"checkpoint_manifest.csv", metrics)
        atomic_json(out/"completion.json", {"status": "NONCANONICAL_SMOKE_COMPLETE", "conditions": list(CONDITIONS),
                    "test_evaluations": 0, "scientific_contrasts": "NOT_COMPUTED", "canonical_seeds": "67-71 RESERVED_UNTOUCHED"})
        update_state("completed", "", 1)
    except BaseException as error:
        if hasattr(error, "receipts"):
            atomic_json(out/"parity_gates.json", error.receipts)
        atomic_json(out/"failure.json", {"status": "EVIDENCE_INCOMPLETE", "error": repr(error), "attempt_id": attempt})
        try:
            update_state("EVIDENCE_INCOMPLETE", reason=repr(error))
        except Exception as sync_error:
            atomic_json(out/"failure_sync.json", {"error": repr(sync_error)})
        raise


def _recursive_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys()==b.keys() and all(_recursive_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a)==len(b) and all(_recursive_equal(x,y) for x,y in zip(a,b))
    return a == b


def telemetry_equivalence_receipt():
    """Production train helper ON/OFF: every parameter/optimizer/RNG/checkpoint/loss."""
    conf = load_conf()
    hist = conf.load_historical_model_module()
    class Model(TokenScaleMixin, hist.DecoderOnlyTransformer):
        pass
    # Synthetic seed7 only, wrapped to preserve caller CPU/CUDA/NumPy/Python RNG.
    saved_np, saved_py = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(7)
            base = Model(vocab_size=17, d_model=8, n_heads=2, d_ff=16, n_layers=1, max_seq_len=4)
            base.token_alpha = 0.7
            ids = torch.arange(48).reshape(12,4) % 17
            loader = DataLoader(torch.utils.data.TensorDataset(ids, (ids+1)%17), batch_size=3, shuffle=False)
            outcomes = []
            for enabled in (False, True):
                torch.manual_seed(1001)
                np.random.seed(1001)
                random.seed(1001)
                m = copy.deepcopy(base)
                opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
                curve, best, selected, best_epoch = [], math.inf, None, 0
                for epoch in (1,2):
                    tr, losses, _ = train_epoch(m, opt, loader, "cpu", enabled)
                    vl = validation_loss(m, loader, "cpu")
                    curve.append((tr, losses, vl))
                    if vl < best:
                        best, best_epoch, selected = vl, epoch, copy.deepcopy(m.state_dict())
                complete = {"model": copy.deepcopy(m.state_dict()), "optimizer": copy.deepcopy(opt.state_dict()),
                            "parameter_gradients": {n:p.grad.detach().clone() if p.grad is not None else None for n,p in m.named_parameters()},
                            "rng": rng_state(), "curve": curve, "selected_model": selected, "best_epoch": best_epoch,
                            "best_loss": best, "evaluation_logits": m(ids).detach().clone()}
                outcomes.append(complete)
            fields = {k: _recursive_equal(outcomes[0][k], outcomes[1][k]) for k in outcomes[0]}
            return {"fixture_seed": 7, "training_rng_seed": 1001, "epochs": 2, "synthetic_only": True,
                    "fields_bit_identical": fields, "pass": all(fields.values())}
    finally:
        np.random.set_state(saved_np)
        random.setstate(saved_py)


if __name__ == "__main__":
    main()
