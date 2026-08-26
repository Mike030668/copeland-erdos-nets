"""R011 embedding factorial unit/static tests."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import torch

from copeland_erdos_nets.r010_protocol import (
    apply_attention_intervention,
    apply_embedding_intervention,
    assert_t0_factorial,
    attention_allowlist,
    build_base_state,
    clone_from_base_state,
    collect_named_tensors,
    derive_seeds,
    epoch_index_permutations,
    hash_int_sequence,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"
R010_DIR = ROOT / ".tmp" / "r010_phase1"


def _load_screening():
    spec = importlib.util.spec_from_file_location("ce_screening_hist", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def screening():
    return _load_screening()


def _tiny(screening):
    def factory():
        return screening.DecoderOnlyTransformer(
            vocab_size=64, d_model=32, n_heads=4, d_ff=64, n_layers=2, max_seq_len=16
        )
    return factory


def test_a0_token_emb_unchanged(screening):
    seeds = derive_seeds(42)
    factory = _tiny(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    clone = clone_from_base_state(base, factory)
    apply_embedding_intervention(clone, "constructor", seeds)
    assert tensor_sha256(clone.token_emb.weight) == hashes["token_emb.weight"]


def test_a1_token_emb_changed_only(screening):
    seeds = derive_seeds(42)
    factory = _tiny(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    clone = clone_from_base_state(base, factory)
    apply_embedding_intervention(clone, "historical_xavier", seeds)
    now = collect_named_tensors(clone)
    assert tensor_sha256(now["token_emb.weight"]) != hashes["token_emb.weight"]
    for name, tensor in now.items():
        if name == "token_emb.weight":
            continue
        assert tensor_sha256(tensor) == hashes[name], name


def test_a1b0_still_eight_attn(screening):
    seeds = derive_seeds(42)
    factory = _tiny(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    clone = clone_from_base_state(base, factory)
    apply_embedding_intervention(clone, "historical_xavier", seeds)
    apply_attention_intervention(clone, "xavier_g1.0", seeds, allowlist=allow)
    _u, changed = assert_t0_factorial(clone, hashes, allow, "historical_xavier")
    assert {r["name"] for r in changed} == set(allow)
    assert len(changed) == 8


def test_a1b0_a1b1_same_embedding(screening):
    seeds = derive_seeds(42)
    factory = _tiny(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    c0 = clone_from_base_state(base, factory)
    apply_embedding_intervention(c0, "historical_xavier", seeds)
    apply_attention_intervention(c0, "xavier_g1.0", seeds, allowlist=allow)
    c1 = clone_from_base_state(base, factory)
    apply_embedding_intervention(c1, "historical_xavier", seeds)
    apply_attention_intervention(c1, "orthogonal", seeds, allowlist=allow)
    assert tensor_sha256(c0.token_emb.weight) == tensor_sha256(c1.token_emb.weight)
    assert tensor_sha256(c0.token_emb.weight) != hashes["token_emb.weight"]


def test_seed_embedding_order_independence(screening):
    seeds = derive_seeds(42)
    factory = _tiny(screening)
    base, _ = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)

    def emb_hash(order):
        out = {}
        for method in order:
            clone = clone_from_base_state(base, factory)
            apply_embedding_intervention(clone, "historical_xavier", seeds)
            apply_attention_intervention(clone, method, seeds, allowlist=allow)
            out[method] = tensor_sha256(clone.token_emb.weight)
        return out

    assert emb_hash(["xavier_g1.0", "orthogonal"]) == emb_hash(["orthogonal", "xavier_g1.0"])


def test_historical_selective_classifies_token_emb_other(screening):
    model = _tiny(screening)()
    assert screening._classify_layer("token_emb.weight") == "other"


def test_batch_order_matches_r010_seed42_if_present():
    r010 = R010_DIR / "batch_order_hashes.csv"
    if not r010.is_file():
        pytest.skip("R010 artifacts not on disk")
    rows = list(csv.DictReader(r010.open()))
    n_items = None
    # reconstruct from recorded n_indices / drop_last 32
    # R010 used n_train_chunks; permutation of that length.
    # Compare hash of epoch 1 seed 42 from protocol vs CSV.
    from copeland_erdos_nets.r010_protocol import derive_seeds, epoch_index_permutations, batch_order_records
    # n_indices in csv for seed 42 epoch 1
    row = next(r for r in rows if int(r["seed"]) == 42 and int(r["epoch"]) == 1)
    n_usable = int(row["n_indices"])
    # drop_last true, batch 32 → n_items >= n_usable and n_usable % 32 == 0
    # recover n_items by trying common WikiText chunk count from dataset_manifest
    man = ROOT / ".tmp/r010_phase1/dataset_manifest.json"
    import json
    n_items = json.loads(man.read_text())["n_train_chunks"] if man.is_file() else n_usable
    seeds = derive_seeds(42)
    perms = epoch_index_permutations(n_items, 15, seeds.seed_shuffle)
    rec = batch_order_records(perms, seed=42, batch_size=32, drop_last=True)
    assert rec[0]["batch_order_hash"] == row["batch_order_hash"]


@pytest.mark.slow
def test_production_attention_hash_parity_seed42():
    """A1 then B0/B1 attention hashes must equal R010 A0 B0/B1 (embedding-only delta)."""
    ch = R010_DIR / "changed_parameter_hashes.csv"
    if not ch.is_file():
        pytest.skip("R010 artifacts not on disk")
    screening = _load_screening()
    vocab = 50257
    factory = lambda: screening.DecoderOnlyTransformer(
        vocab_size=vocab, d_model=128, n_heads=4, d_ff=512, n_layers=2, max_seq_len=128
    )
    seeds = derive_seeds(42)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    rows = list(csv.DictReader(ch.open()))
    r010 = {}
    for r in rows:
        if r["seed"] != "42":
            continue
        r010.setdefault(r["method"], {})[r["name"]] = r["post_intervention_sha256"]

    for method, key in (("xavier_g1.0", "xavier_g1.0"), ("orthogonal", "orthogonal")):
        clone = clone_from_base_state(base, factory)
        apply_embedding_intervention(clone, "historical_xavier", seeds)
        apply_attention_intervention(clone, method, seeds, allowlist=allow)
        params = dict(clone.named_parameters())
        for name in allow:
            assert tensor_sha256(params[name]) == r010[key][name], name
        assert tensor_sha256(clone.token_emb.weight) != hashes["token_emb.weight"]
