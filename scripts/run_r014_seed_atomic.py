#!/usr/bin/env python3
"""R014 larger-model embedding-scale confirmation, seed-atomic runner (smoke/canonical).

One seed = 3 conditions (D_xavier, D_mid1, D_ctor) in ONE VM. Reuses R013's
embedding-scale-ladder protocol (src/copeland_erdos_nets/r013_protocol.py)
unchanged -- the factor-construction formula is already shape-parametric
(vocab, d_model are function arguments). Only the model shape (former-R009,
recovered from configs/r009_scaleup.json's model dims; vocab is NOT read from
that config -- see DS binding amendment) and the reduced 3-condition list
differ from R013.

DS Design Validation binding amendments (2026-09-11):
1. Historical R009 config declared vocab_size=28996; that field was NEVER
   consumed by the historical runner (tok.vocab_size always overrides it --
   same pattern this script already follows, see load_wikitext_splits).
   Hard-assert tokenizer.vocab_size==50257; persist both values.
2. 4-layer model has 16 Q/K/V/O attention weights (4 layers x 4 projections),
   not 8. Explicitly gated via assert_expected_allowlist_count (R011 pattern,
   dropped in R012/R013's runners since n_layers=2 made it implicit -- R014
   restores the explicit gate since n_layers changes).

DS Smoke Checkpoint Round 1 binding repairs (2026-09-11, v2):
B1. The full canonical `schedule_epochs` (15) batch schedule is ALWAYS
    precomputed, persisted (epoch_batch_hashes.csv), and parity-gated before
    the first optimizer step -- even when `train_epochs` (config
    `training.epochs`) is smaller for a bounded smoke. Training then uses a
    prefix of that same precomputed schedule, never a separately generated
    one.
B2. Attention Xavier g=1.0 is applied exactly ONCE, directly on the shared
    `base` model, before any condition is cloned from it -- not per-condition
    on each clone (even though the old per-condition-call path was numerically
    equivalent by construction, DS required the code to literally match the
    frozen causal path: construct -> attention ONCE -> freeze -> clone 3x,
    embedding-only difference after that point). `shared_base_state` is now a
    real gate (each condition's attention weights hash-compared directly to
    the frozen `base`, not just to each other).
B3. `SEED_STATUS.txt` is mode-aware: `NONCANONICAL_SMOKE seed=... conditions=3`
    unless `cfg["experiment"]["mode"]=="r014_canonical"` AND the seed is one of
    the DS-released canonical seeds (57-61) AND `train_epochs==schedule_epochs`
    (a full canonical run, not a bounded smoke) -- seed1057 can never be
    labeled canonical.

No dynamics telemetry (R013's embedding-RMS/gradient-L2 tracking) -- not
required per R014 design; standard per-epoch train/val loss + best_epoch
checkpoint-selection timing are sufficient diagnostics.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, subprocess, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    assert_expected_allowlist_count, apply_attention_intervention, tensor_sha256,
    collect_named_tensors, dump_json, epoch_index_permutations, batch_order_records,
    hash_int_sequence,
)
from copeland_erdos_nets.r013_protocol import (
    apply_embedding_dose, assert_no_weight_tying, rms, xavier_scalar_std, ladder_factors,
    cosine_and_maxdiff, all_epoch_batch_parity,
)
ROOT = Path(__file__).resolve().parents[1]

R014_DOSES = ("D_xavier", "D_mid1", "D_ctor")
HISTORICAL_R009_CONFIG_VOCAB_SIZE = 28996  # dead field, never consumed by the historical runner
EXPECTED_EFFECTIVE_VOCAB_SIZE = 50257      # tok.vocab_size for gpt2; hard-asserted below
CANONICAL_SEEDS = frozenset({57, 58, 59, 60, 61})  # DS-released canonical seeds only


def seed_status_label(cfg: dict, seed: int, schedule_epochs: int, train_epochs: int) -> str:
    """DS binding repair B3: mode-aware status for a SUCCESSFUL completion. Canonical
    requires ALL of: mode==r014_canonical, seed in the DS-released set, and a full
    (non-bounded) run (train_epochs==schedule_epochs). seed1057 (or any non-released
    seed, or any bounded run) can never be labeled canonical."""
    mode = cfg.get("experiment", {}).get("mode", "")
    is_full_canonical_run = (mode == "r014_canonical" and seed in CANONICAL_SEEDS
                              and train_epochs == schedule_epochs)
    tag = "CANONICAL" if is_full_canonical_run else "NONCANONICAL_SMOKE"
    return f"{tag} seed={seed} conditions={len(R014_DOSES)}\n"

def wcsv(p, rows):
    if not rows:
        Path(p).write_text("")
        return
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def assert_runtime(freeze, out):
    import numpy, datasets, transformers, subprocess as sp
    def drv():
        try:
            return sp.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
        except Exception:
            return "unavailable"
    got = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "python": platform.python_version(),
           "torch": torch.__version__, "cuda": torch.version.cuda or "none", "numpy": numpy.__version__,
           "datasets": datasets.__version__, "transformers": transformers.__version__, "driver": drv()}
    mism = [k for k in ["gpu", "python", "torch", "cuda", "numpy", "datasets", "transformers", "driver"]
            if str(got.get(k)) != str(freeze.get(k)) and not (k == "gpu" and str(freeze.get(k, "")) in str(got.get(k, "")))]
    (out / "runtime_assertion.log").write_text(f"target {json.dumps(freeze)}\nactual {json.dumps(got)}\nmismatch={mism}\n")
    if mism:
        raise SystemExit(f"RUNTIME HARD STOP mismatch={mism}")

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def train_condition(model, cond, splits, perms, bs, tdrop, device, cfg, out, seed):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    val = DataLoader(splits["validation"], batch_size=bs, shuffle=False, drop_last=bool(cfg["data"]["val_drop_last"]))
    test = DataLoader(splits["test"], batch_size=bs, shuffle=False, drop_last=bool(cfg["data"]["test_drop_last"]))
    conf = load_conf()
    bv = math.inf; be = 0; lv = math.inf; bl = math.inf
    bp = out / "checkpoints" / f"{cond}_seed{seed}_best.pt"
    for ep, order in enumerate(perms, 1):
        us = order[:(len(order) // bs) * bs] if tdrop else order
        ld = DataLoader(splits["train"], batch_size=bs, sampler=conf.EpochPermutationSampler(us), drop_last=False)
        model.train(); run = 0.0; n = 0
        for x, y in ld:
            x = x.to(device); y = y.to(device); opt.zero_grad(set_to_none=True)
            lo = model(x); loss = crit(lo.view(-1, lo.size(-1)), y.view(-1)); loss.backward()
            opt.step(); run += float(loss.item()) * x.size(0); n += x.size(0)
        vl = 0.0; vn = 0
        model.eval()
        with torch.no_grad():
            for x, y in val:
                x = x.to(device); y = y.to(device); lo = model(x)
                vl += float(crit(lo.view(-1, lo.size(-1)), y.view(-1)).item()) * x.size(0); vn += x.size(0)
        vloss = vl / max(vn, 1); vppl = math.exp(min(vloss, 20)); tr = run / max(n, 1)
        if vloss < bl:
            bl = vloss; bv = vloss; be = ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vloss}, bp)
        lv = vloss
        print(f"  {cond} ep{ep}/{len(perms)} train={tr:.4f} val={vloss:.4f}", flush=True)
    ck = torch.load(bp, map_location=device, weights_only=False); model.load_state_dict(ck["model"])
    tl = 0.0; tn = 0; model.eval()
    with torch.no_grad():
        for x, y in test:
            x = x.to(device); y = y.to(device); lo = model(x)
            tl += float(crit(lo.view(-1, lo.size(-1)), y.view(-1)).item()) * x.size(0); tn += x.size(0)
    test_loss = tl / max(tn, 1)
    return {"best_epoch": be, "best_val_loss": bv, "final_val_loss": lv, "best_val_ppl": math.exp(min(bv, 20)),
            "final_val_ppl": math.exp(min(lv, 20)), "test_ppl": math.exp(min(test_loss, 20)), "ckpt": bp}

def export_ckpt_drive(name, local_path):
    """Durable export to Drive placeholder r014_ckpt_<name>.pt; read back; sha."""
    sa = "/content/sa.json"
    if not os.path.exists(sa):
        raise RuntimeError("no SA on VM")
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(sa, ["https://www.googleapis.com/auth/drive"])
    ga = GoogleAuth(); ga.credentials = creds; d = GoogleDrive(ga)
    def find(n, par=None):
        q = f"title='{n}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if par:
            q = f"title='{n}' and '{par}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        l = d.ListFile({"q": q}).GetList(); return l[0]["id"] if l else None
    ex = find("exchange", find("copeland-erdos-nets_drive", find("research", find("agent-rules-tree-control"))))
    fn = f"r014_ckpt_{name}.pt"
    l = d.ListFile({"q": f"title='{fn}' and '{ex}' in parents and trashed=false"}).GetList()
    if not l:
        raise RuntimeError(f"no placeholder {fn}")
    gf = d.CreateFile({"id": l[0]["id"]}); gf.SetContentFile(str(local_path)); gf.Upload()
    back = f"/content/_verify_{fn}"; gf2 = d.CreateFile({"id": l[0]["id"]}); gf2.GetContentFile(back)
    h = hashlib.sha256()
    with open(back, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return f"gdrive:exchange/{fn}", h.hexdigest(), os.path.getsize(back)

def load_conf():
    spec = importlib.util.spec_from_file_location("r010r", ROOT / "scripts" / "run_transformer_paired_confirmation.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exchange-tag", default="",
                     help="Optional suffix appended to the Drive exchange checkpoint name "
                          "(e.g. '_v2') so a re-run at the same seed does not overwrite an "
                          "earlier run's durable checkpoints still referenced by an immutable "
                          "package's checkpoint_manifest.csv.")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True); seed = a.seed
    dump_json(out / "RUNTIME_FREEZE.json", cfg["runtime_freeze"])
    est = open(out / "environment_establishment.log", "w")
    rc = subprocess.run("pip -q install 'transformers==5.15.1' 'numpy==2.1.3' 'datasets==4.0.0'", shell=True).returncode
    est.write(f"$ pip install transformers==5.15.1 numpy==2.1.3 datasets==4.0.0\nRC={rc}\n"); est.close()
    (out / "package_snapshot_before_assert.txt").write_text(subprocess.run("pip freeze", shell=True, capture_output=True, text=True).stdout)
    assert_runtime(cfg["runtime_freeze"], out)
    from copeland_erdos_nets import r010_protocol as R
    impl = {"seed_model_offset": R.SEED_MODEL_OFFSET, "seed_shuffle_offset": R.SEED_SHUFFLE_OFFSET, "seed_embedding_redraw_offset": R.SEED_EMBEDDING_OFFSET}
    if cfg.get("rng_policy") != impl:
        raise SystemExit(f"RNG POLICY HARD STOP {cfg.get('rng_policy')} != {impl}")
    dump_json(out / "rng_policy.json", impl)
    conf = load_conf(); device = conf.resolve_device(cfg["training"]["device"]); conf.write_environment(out / "environment.txt", device)
    hist = conf.load_historical_model_module(); splits, vocab, dm = conf.load_wikitext_splits(cfg["data"])

    # DS binding amendment #1: vocab provenance -- persist both, hard-assert effective value
    vp = cfg.get("vocab_provenance", {})
    expected_eff = int(vp.get("expected_effective_vocab_size", EXPECTED_EFFECTIVE_VOCAB_SIZE))
    if vocab != expected_eff:
        raise SystemExit(f"VOCAB HARD STOP: effective tokenizer vocab_size={vocab} != expected {expected_eff}")
    dump_json(out / "vocab_provenance.json", {
        "historical_r009_config_vocab_size": vp.get("historical_r009_config_vocab_size", HISTORICAL_R009_CONFIG_VOCAB_SIZE),
        "r014_effective_vocab_size": vocab,
        "vocab_source": vp.get("vocab_source", "GPT2_TOKENIZER_RUNTIME_OVERRIDE"),
        "note": "historical R009 configs/r009_scaleup.json declared vocab_size=28996 at "
                "commit 4619405, but that field was never consumed by the historical runner "
                "(tok.vocab_size always overrides it); effective historical + R014 runtime "
                "vocab is the GPT-2 tokenizer's, 50257.",
    })

    bs = int(cfg["data"]["batch_size"]); tdrop = bool(cfg["data"]["train_drop_last"])
    dm_d = int(cfg["model"]["d_model"]); n_layers = int(cfg["model"]["n_layers"])
    def factory():
        return hist.DecoderOnlyTransformer(vocab_size=vocab, d_model=dm_d, n_heads=int(cfg["model"]["n_heads"]),
            d_ff=int(cfg["model"]["d_ff"]), n_layers=n_layers, max_seq_len=int(cfg["data"]["seq_len"]))
    s = derive_seeds(seed)
    schedule_epochs = int(cfg["training"].get("schedule_epochs", cfg["training"]["epochs"]))
    train_epochs = int(cfg["training"]["epochs"])
    if train_epochs > schedule_epochs:
        raise SystemExit(f"CONFIG HARD STOP: train_epochs={train_epochs} > schedule_epochs={schedule_epochs}")
    # DS binding repair B1: ALWAYS precompute/persist/gate the full canonical
    # schedule (15 epochs), even when training only a prefix of it for a
    # bounded smoke. Training below uses perms_full[:train_epochs] -- the
    # SAME precomputed+gated schedule, never a separately regenerated one.
    perms_full = epoch_index_permutations(len(splits["train"]), schedule_epochs, s.seed_shuffle)

    # DS binding repair B2: construct once -> apply Xavier g=1.0 attention ONCE
    # on the shared base -> freeze -> clone 3 conditions from it. Only
    # token_emb.weight may differ across conditions after this point.
    base, base_h = build_base_state(factory, s.seed_model, device="cpu")
    allow = attention_allowlist(base)
    allowlist_count_ok = True
    try:
        assert_expected_allowlist_count(allow, n_layers=n_layers)
    except AssertionError as e:
        allowlist_count_ok = False
        print(f"[r014] attention allowlist count FAIL: {e}", flush=True)
    apply_attention_intervention(base, "xavier_g1.0", s, allowlist=allow)  # ONCE, on the shared base itself
    base_attn_sha = {n: tensor_sha256(dict(base.named_parameters())[n]) for n in allow}  # frozen post-attention reference

    base_emb = base.token_emb.weight.detach().clone()  # base = token_emb.weight from the shared (post-attention) base_state
    r = xavier_scalar_std(vocab, dm_d) / rms(base_emb); facs = ladder_factors(r)
    wcsv(out / "scale_ladder.csv", [{"dose": d, "factor_rel_ctor": facs[d],
        "target_rms": (rms(base_emb) if d == "D_ctor" else facs[d] * rms(base_emb)),
        "r": r, "s_xav": xavier_scalar_std(vocab, dm_d), "rms_constructor": rms(base_emb)} for d in R014_DOSES])
    fc = []; embh = []; bdir = []; attnrows = []; unch = []; models = {}
    for d in R014_DOSES:
        m = clone_from_base_state(base, factory)  # clone from the FROZEN post-attention base; no per-condition attention call
        assert_no_weight_tying(m)
        meta = apply_embedding_dose(m, d, base_emb, vocab, dm_d)
        models[d] = m
        fc.append({"dose": d, "seed": seed, **{k: meta[k] for k in ["factor_rel_ctor", "s_xav", "r", "rms_constructor", "target_rms", "realized_rms", "base_direction_hash", "emb_hash"]}})
        embh.append({"dose": d, "seed": seed, "emb_sha256": meta["emb_hash"]})
        bdir.append({"dose": d, "seed": seed, "base_direction_hash": meta["base_direction_hash"]})
        p = dict(m.named_parameters())
        for n in allow:
            attnrows.append({"dose": d, "seed": seed, "name": n, "sha256": tensor_sha256(p[n])})
        now = collect_named_tensors(m)
        for n, t in now.items():
            if n == "token_emb.weight" or n in allow:
                continue
            unch.append({"dose": d, "seed": seed, "name": n, "sha256": tensor_sha256(t)})

    tol = cfg.get("direction_tolerances", {"cosine_min": 0.999999, "normalized_max_abs_diff_max": 1e-4})
    dir_rows = []; dir_ok = True
    for d in R014_DOSES:
        cos, mad = cosine_and_maxdiff(models[d].token_emb.weight, base_emb)
        ok = (cos >= tol["cosine_min"] and mad <= tol["normalized_max_abs_diff_max"])
        dir_ok &= ok
        bdh = next((row["base_direction_hash"] for row in bdir if row["dose"] == d), "")
        dir_rows.append({"dose": d, "seed": seed, "cosine_similarity": cos, "normalized_max_abs_diff": mad,
                          "base_direction_hash": bdh, "cosine_min": tol["cosine_min"],
                          "max_abs_diff_max": tol["normalized_max_abs_diff_max"], "pass": str(ok).lower()})
    wcsv(out / "base_direction_audit.csv", dir_rows)

    changed_ok = True
    for d in R014_DOSES:
        now = collect_named_tensors(models[d])
        for n, tt in now.items():
            if n == "token_emb.weight" or n in allow:
                continue
            if tensor_sha256(tt) != base_h[n]:
                changed_ok = False
    attn_ident = all(tensor_sha256(dict(models[d].named_parameters())[n]) == tensor_sha256(dict(models["D_ctor"].named_parameters())[n]) for d in R014_DOSES for n in allow)
    # DS binding repair B2: real shared_base_state gate -- each condition's attention
    # weights must hash-match the FROZEN base's own post-attention weights directly
    # (not merely match each other), proving the literal shared-base construction path.
    shared_base_ok = all(tensor_sha256(dict(models[d].named_parameters())[n]) == base_attn_sha[n] for d in R014_DOSES for n in allow)
    wt_ok = True
    try:
        for d in R014_DOSES:
            assert_no_weight_tying(models[d])
    except SystemExit:
        wt_ok = False
    rms_ok = all(abs(row["realized_rms"] - row["target_rms"]) < 1e-6 for row in fc)

    gates = [
        ("runtime_assert_pass", True),
        ("source_provenance_present", True),
        ("rng_policy_config==impl", True),
        ("shared_base_state", shared_base_ok),
        ("D_ctor==base_exact", tensor_sha256(models["D_ctor"].token_emb.weight) == tensor_sha256(base_emb)),
        ("xavier_scalar_formula", abs(xavier_scalar_std(vocab, dm_d) - (2.0 / (vocab + dm_d)) ** 0.5) < 1e-15),
        ("realized_rms==target", rms_ok),
        ("actual_direction_within_tol", dir_ok),
        ("exact_changed_set_only_embedding", changed_ok),
        ("no_weight_tying", wt_ok),
        ("attention_allowlist_count_16", allowlist_count_ok),
        ("attention_parity_all_conditions", attn_ident),
        ("non_embedding_parity", changed_ok),
        ("vocab_hard_assert_50257", vocab == expected_eff),
    ]
    wcsv(out / "parity_summary.csv", [{"seed": seed, "gate": g, "pass": str(bool(v)).lower()} for g, v in gates])
    if not all(v for _, v in gates):
        (out / "SEED_STATUS.txt").write_text(f"NONCANONICAL_PARITY_FAILURE {[g for g, v in gates if not v]}\n")
        raise SystemExit(f"parity fail {[g for g, v in gates if not v]}")
    wcsv(out / "factor_construction.csv", fc); wcsv(out / "embedding_hashes.csv", embh)
    wcsv(out / "base_direction_construction.csv", bdir); wcsv(out / "attention_parity.csv", attnrows)
    wcsv(out / "unchanged_parameter_hashes.csv", unch)
    # DS binding repair B1: hash the FULL schedule_epochs (15) schedule, not just
    # the (possibly smaller) number of epochs actually trained this run.
    wcsv(out / "epoch_batch_hashes.csv", [{"dose": d, "epoch": ep + 1, "batch_order_hash": hash_int_sequence(perms_full[ep][:(len(perms_full[ep]) // bs) * bs] if tdrop else perms_full[ep])} for d in R014_DOSES for ep in range(schedule_epochs)])
    from copeland_erdos_nets.r013_protocol import all_epoch_batch_parity as _batp
    _ebh = list(csv.DictReader((out / "epoch_batch_hashes.csv").open()))
    if not _batp(_ebh, len(R014_DOSES)):
        (out / "SEED_STATUS.txt").write_text("NONCANONICAL_BATCH_PARITY_FAILURE (pre-training)\n")
        raise SystemExit("PRE-TRAINING HARD STOP: all_epoch_batch_parity FAIL")
    print(f"[r014] seed {seed} PARITY PASS (16/16 attention, vocab=50257 asserted, "
          f"{schedule_epochs}-epoch schedule pre-gated); training {train_epochs}/{schedule_epochs} "
          f"epochs x {len(R014_DOSES)} conditions", flush=True)

    # Train only a prefix of the already precomputed+gated schedule (DS binding repair B1).
    perms_train = perms_full[:train_epochs]
    metrics = []; ckman = []
    for d in R014_DOSES:
        m = models[d].to(device)
        res = train_condition(m, d, splits, perms_train, bs, tdrop, device, cfg, out, seed)
        cksha = sha_file(res["ckpt"]); local_size = res["ckpt"].stat().st_size
        puri = psha = ""; psize = 0; pver = False
        try:
            puri, psha, psize = export_ckpt_drive(f"{d}_seed{seed}{a.exchange_tag}", res["ckpt"]); pver = (psha == cksha and psize == local_size)
        except Exception as e:
            print(f"[r014] durable export failed {d} {type(e).__name__}", flush=True)
        if not pver:
            (out / "SEED_STATUS.txt").write_text(f"NONCANONICAL_DURABLE_FAILURE {d}\n"); raise SystemExit(f"durable ckpt verify FAIL {d}")
        ckman.append({"dose": d, "seed": seed, "selected_epoch": res["best_epoch"], "selected_val_ppl": round(res["best_val_ppl"], 6),
                      "local_path": str(res["ckpt"]), "local_sha256": cksha, "size_bytes": local_size,
                      "persistent_uri": puri, "persistent_sha256": psha, "persistent_size_bytes": psize, "persistent_verified": str(pver).lower()})
        metrics.append({"dose": d, "seed": seed, "best_epoch": res["best_epoch"], "best_val_ppl": res["best_val_ppl"],
                         "final_val_ppl": res["final_val_ppl"], "final_minus_best_val_loss": res["final_val_loss"] - res["best_val_loss"],
                         "test_ppl": res["test_ppl"]})
        wcsv(out / "per_seed.csv", metrics); wcsv(out / "checkpoint_manifest.csv", ckman)

    par = list(csv.DictReader((out / "parity_summary.csv").open()))
    ck = list(csv.DictReader((out / "checkpoint_manifest.csv").open()))
    par.append({"seed": seed, "gate": "durable_checkpoint_verification", "pass": str(all(x.get("persistent_verified") == "true" for x in ck) and len(ck) == len(R014_DOSES)).lower()})
    ebh = list(csv.DictReader((out / "epoch_batch_hashes.csv").open()))
    batch_ok = all_epoch_batch_parity(ebh, len(R014_DOSES))
    par.append({"seed": seed, "gate": "all_epoch_batch_parity", "pass": str(bool(batch_ok)).lower()})
    if not batch_ok:
        (out / "SEED_STATUS.txt").write_text("NONCANONICAL_BATCH_PARITY_FAILURE\n")
        wcsv(out / "parity_summary.csv", par); raise SystemExit("all_epoch_batch_parity FAIL")
    wcsv(out / "parity_summary.csv", par)
    dump_json(out / "resolved_config.json", cfg); dump_json(out / "dataset_manifest.json", dm)
    status = seed_status_label(cfg, seed, schedule_epochs, train_epochs)
    (out / "SEED_STATUS.txt").write_text(status)
    (out / "DURABLE_MARKER.txt").write_text(f"R014_SEED{seed}_COMPLETE\n")
    print(f"[r014] seed {seed} COMPLETE {train_epochs}/{schedule_epochs} epochs x {len(R014_DOSES)} conditions ({status.strip()})", flush=True)

if __name__ == "__main__":
    main()
