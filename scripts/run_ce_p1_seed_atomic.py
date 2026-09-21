#!/usr/bin/env python3
"""CE-P1 seed-atomic runner: A/B/C token-embedding-only comparison.

It deliberately stops before optimization if any t0 or batch-order parity
gate fails.  One invocation is one complete numerical seed, never a cell.
"""
from __future__ import annotations

import argparse, csv, importlib.util, json, math, os, platform, subprocess, tempfile, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from copeland_erdos_nets.ce_p1_protocol import CONDITIONS, embedding_for_condition, multiset_equal
from copeland_erdos_nets.r010_protocol import (
    apply_attention_intervention, build_base_state, clone_from_base_state,
    collect_named_tensors, derive_seeds, epoch_index_permutations,
    hash_int_sequence, tensor_sha256,
)
from copeland_erdos_nets.r013_protocol import all_epoch_batch_parity, assert_no_weight_tying

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True); f.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class DriveLiveSync:
    """Path-B update-only transport for pre-created CE-P1 placeholders."""
    def __init__(self, prefix: str):
        self.prefix = prefix
        if os.environ.get("CE_P1_GDRIVE_LIVE") != "1":
            raise RuntimeError("CE-P1 HARD STOP: CE_P1_GDRIVE_LIVE=1 is required on Colab")
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials
        key = os.environ.get("CE_P1_GDRIVE_SA_KEY", "/content/sa.json")
        credentials = ServiceAccountCredentials.from_json_keyfile_name(key, ["https://www.googleapis.com/auth/drive"])
        auth = GoogleAuth(); auth.credentials = credentials; self.drive = GoogleDrive(auth)
        def folder(name, parent=None):
            query = f"title='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent: query += f" and '{parent}' in parents"
            rows = self.drive.ListFile({"q": query}).GetList()
            if not rows: raise RuntimeError(f"CE-P1 HARD STOP: Drive folder missing: {name}")
            return rows[0]["id"]
        self.exchange = folder("exchange", folder("copeland-erdos-nets_drive", folder("research", folder("agent-rules-tree-control"))))

    def update(self, filename: str, local_path: Path) -> None:
        rows = self.drive.ListFile({"q": f"title='{filename}' and '{self.exchange}' in parents and trashed=false"}).GetList()
        if len(rows) != 1:
            raise RuntimeError(f"CE-P1 HARD STOP: expected one pre-created Drive placeholder {filename}, got {len(rows)}")
        remote = self.drive.CreateFile({"id": rows[0]["id"]})
        remote.SetContentFile(str(local_path)); remote.Upload()


def load_confirmation_module():
    spec = importlib.util.spec_from_file_location("ce_p1_confirmation", ROOT / "scripts" / "run_transformer_paired_confirmation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_runtime(freeze: dict, out: Path) -> None:
    import datasets, numpy, transformers
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception:
        driver = "unavailable"
    actual = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda or "none", "numpy": numpy.__version__, "datasets": datasets.__version__, "transformers": transformers.__version__, "driver": driver}
    mismatch = [key for key in freeze if str(actual.get(key)) != str(freeze[key]) and not (key == "gpu" and str(freeze[key]) in str(actual.get(key, "")))]
    (out / "runtime_assertion.log").write_text(f"target {json.dumps(freeze, sort_keys=True)}\nactual {json.dumps(actual, sort_keys=True)}\nmismatch={mismatch}\n")
    if mismatch:
        raise SystemExit(f"CE-P1 RUNTIME HARD STOP: {mismatch}")


def train_condition(model, condition, splits, permutations, batch_size, train_drop_last, device, cfg, out, seed, on_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    validation = DataLoader(splits["validation"], batch_size=batch_size, shuffle=False, drop_last=bool(cfg["data"]["val_drop_last"]))
    test = DataLoader(splits["test"], batch_size=batch_size, shuffle=False, drop_last=bool(cfg["data"]["test_drop_last"]))
    helpers = load_confirmation_module()
    best_loss, best_epoch, final_loss = math.inf, 0, math.inf
    best_checkpoint = out / "checkpoints" / f"{condition}_seed{seed}_best.pt"
    resume_checkpoint = out / "checkpoints" / f"{condition}_seed{seed}_resume.pt"
    curves = []
    for epoch, order in enumerate(permutations, 1):
        usable = order[:(len(order) // batch_size) * batch_size] if train_drop_last else order
        train = DataLoader(splits["train"], batch_size=batch_size, sampler=helpers.EpochPermutationSampler(usable), drop_last=False)
        model.train(); total, count = 0.0, 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward(); optimizer.step()
            total += float(loss.item()) * x.size(0); count += x.size(0)
        model.eval(); val_total, val_count = 0.0, 0
        with torch.no_grad():
            for x, y in validation:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_total += float(loss.item()) * x.size(0); val_count += x.size(0)
        train_loss, val_loss = total / max(count, 1), val_total / max(val_count, 1)
        curves.append({"seed": seed, "condition": condition, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20))})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_checkpoint)
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                           "best_loss": best_loss, "best_epoch": best_epoch, "final_loss": val_loss}, resume_checkpoint)
        final_loss = val_loss
        on_epoch(condition, epoch, curves, resume_checkpoint)
        print(f"[ce-p1] {condition} ep{epoch}/{len(permutations)} val={val_loss:.5f}", flush=True)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=False)["model"])
    model.eval(); test_total, test_count = 0.0, 0
    with torch.no_grad():
        for x, y in test:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            test_total += float(loss.item()) * x.size(0); test_count += x.size(0)
    return {"seed": seed, "condition": condition, "best_epoch": best_epoch, "best_val_ppl": math.exp(min(best_loss, 20)), "final_val_ppl": math.exp(min(final_loss, 20)), "test_ppl": math.exp(min(test_total / max(test_count, 1), 20)), "checkpoint": str(best_checkpoint)}, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir(exist_ok=True)
    if args.seed not in cfg["seeds"]:
        raise SystemExit(f"CE-P1 HARD STOP: seed {args.seed} is not authorized by this config")
    if tuple(cfg["experiment"]["conditions"]) != CONDITIONS:
        raise SystemExit("CE-P1 HARD STOP: condition set differs from DS binding")
    prefix = os.environ.get("CE_P1_GDRIVE_PREFIX", f"ce_p1_{cfg['experiment']['mode']}_seed_{args.seed}")
    sync = DriveLiveSync(prefix)
    assert_runtime(cfg["runtime_freeze"], out)
    conf = load_confirmation_module(); device = conf.resolve_device(cfg["training"]["device"])
    conf.write_environment(out / "environment.txt", device)
    splits, vocab, dataset_manifest = conf.load_wikitext_splits(cfg["data"])
    dimensions = cfg["model"]
    def factory():
        historical = conf.load_historical_model_module()
        return historical.DecoderOnlyTransformer(vocab_size=vocab, d_model=int(dimensions["d_model"]), n_heads=int(dimensions["n_heads"]), d_ff=int(dimensions["d_ff"]), n_layers=int(dimensions["n_layers"]), max_seq_len=int(cfg["data"]["seq_len"]))
    seeds = derive_seeds(args.seed)
    base, base_hashes = build_base_state(factory, seeds.seed_model, device="cpu")
    # R013's common attention baseline is installed once into the shared base.
    apply_attention_intervention(base, "xavier_g1.0", seeds)
    base_hashes = {name: tensor_sha256(value) for name, value in collect_named_tensors(base).items()}
    assert_no_weight_tying(base)
    epochs, batch_size = int(cfg["training"]["epochs"]), int(cfg["data"]["batch_size"])
    permutations = epoch_index_permutations(len(splits["train"]), epochs, seeds.seed_shuffle)
    models, construction, invariant_rows = {}, [], []
    for condition in CONDITIONS:
        model = clone_from_base_state(base, factory); assert_no_weight_tying(model)
        embedding, receipt = embedding_for_condition(condition, tuple(model.token_emb.weight.shape), seed_embedding=seeds.seed_embedding, seed_shuffle=seeds.seed_shuffle)
        model.token_emb.weight.data.copy_(embedding)
        models[condition] = model
        construction.append({"seed": args.seed, **receipt})
        for name, value in collect_named_tensors(model).items():
            if name == "token_emb.weight": continue
            invariant_rows.append({"seed": args.seed, "condition": condition, "name": name, "sha256": tensor_sha256(value), "matches_shared_base": str(tensor_sha256(value) == base_hashes[name]).lower()})
    b, c = models["CE_LCG_Xavier_RMS"].token_emb.weight, models["CE_SHUFFLED_Xavier_RMS"].token_emb.weight
    tolerance = float(cfg["parity_tolerance"])
    rms_ok = all(abs(float(row["realized_rms"]) - float(row["target_rms"])) <= tolerance for row in construction)
    invariants_ok = all(row["matches_shared_base"] == "true" for row in invariant_rows)
    multiset_ok = multiset_equal(b, c)
    batch_rows = []
    for condition in CONDITIONS:
        for epoch, order in enumerate(permutations, 1):
            usable = order[:(len(order) // batch_size) * batch_size] if cfg["data"]["train_drop_last"] else order
            batch_rows.append({"seed": args.seed, "condition": condition, "epoch": epoch, "batch_order_hash": hash_int_sequence(usable)})
    batch_ok = all_epoch_batch_parity(batch_rows, len(CONDITIONS))
    gates = [("authorized_seed", True), ("shared_base_state", True), ("only_token_embedding_changes", invariants_ok), ("attention_positional_lm_head_parity", invariants_ok), ("exact_target_rms", rms_ok), ("ce_value_multiset_B_equals_C", multiset_ok), ("all_epoch_batch_parity_preoptimization", batch_ok), ("no_weight_tying", True)]
    write_csv(out / "embedding_construction.csv", construction); write_csv(out / "non_embedding_invariant_hashes.csv", invariant_rows); write_csv(out / "epoch_batch_hashes.csv", batch_rows); write_csv(out / "parity_summary.csv", [{"seed": args.seed, "gate": gate, "pass": str(value).lower()} for gate, value in gates])
    (out / "base_state_sha256.txt").write_text(hashlib_sha256_state(base))
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    (out / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n")
    def publish(status, current_condition, epoch=0):
        state = {"experiment": cfg["experiment"]["name"], "mode": cfg["experiment"]["mode"], "seed": args.seed,
                 "status": status, "current_condition": current_condition, "epoch": epoch,
                 "updated_unix": time.time(), "branch_sha": os.environ.get("CE_GIT_SHA", "")}
        write_json(out / "run_state.json", state)
        sync.update(f"{prefix}_state.json", out / "run_state.json")
        if (out / "learning_curves.csv").exists(): sync.update(f"{prefix}_live.csv", out / "learning_curves.csv")
    if not all(value for _, value in gates):
        (out / "SEED_STATUS.txt").write_text("EVIDENCE_INCOMPLETE_PARITY_FAILURE\n")
        publish("parity_failed", "")
        raise SystemExit(f"CE-P1 HARD STOP: {[gate for gate, value in gates if not value]}")
    publish("parity_passed", "")
    metrics, curves = [], []
    def on_epoch(condition, epoch, partial_curves, checkpoint):
        write_csv(out / "learning_curves.csv", curves + partial_curves)
        sync.update(f"{prefix}_checkpoint_{condition}.pt", checkpoint)
        publish("running", condition, epoch)
    for condition in CONDITIONS:
        result, condition_curves = train_condition(models[condition].to(device), condition, splits, permutations, batch_size, bool(cfg["data"]["train_drop_last"]), device, cfg, out, args.seed, on_epoch)
        metrics.append(result); curves.extend(condition_curves)
        write_csv(out / "per_seed.csv", metrics); write_csv(out / "learning_curves.csv", curves)
        publish("running", condition, epochs)
    (out / "SEED_STATUS.txt").write_text(f"{cfg['experiment']['mode'].upper()} seed={args.seed} conditions=3\n")
    publish("completed", "", epochs)


def hashlib_sha256_state(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
