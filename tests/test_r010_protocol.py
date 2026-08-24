"""Unit/static tests for R010 paired attention protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from copeland_erdos_nets.r010_protocol import (
    METHODS,
    apply_attention_intervention,
    assert_expected_allowlist_count,
    assert_t0_invariance,
    attention_allowlist,
    build_base_state,
    clone_from_base_state,
    derive_seeds,
    epoch_index_permutations,
    hash_int_sequence,
    tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"


def _load_screening():
    spec = importlib.util.spec_from_file_location("ce_screening_hist", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def screening():
    return _load_screening()


def _factory(screening, vocab=64, d_model=32, n_heads=4, d_ff=64, n_layers=2, seq=16):
    def factory():
        return screening.DecoderOnlyTransformer(
            vocab_size=vocab,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_seq_len=seq,
        )
    return factory


def test_allowlist_is_eight_qualified_names(screening):
    model = _factory(screening)()
    allow = attention_allowlist(model)
    assert_expected_allowlist_count(allow, n_layers=2)
    assert all(n.startswith("blocks.") for n in allow)
    assert all(n.endswith((".attn.q_proj.weight", ".attn.k_proj.weight",
                           ".attn.v_proj.weight", ".attn.out_proj.weight")) for n in allow)


@pytest.mark.parametrize("method", list(METHODS))
def test_t0_invariance_per_method(screening, method):
    seeds = derive_seeds(42)
    factory = _factory(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    clone = clone_from_base_state(base, factory)
    apply_attention_intervention(clone, method, seeds, allowlist=allow)
    unchanged, changed_rows = assert_t0_invariance(clone, hashes, allow)
    assert {r["name"] for r in changed_rows} == set(allow)
    assert len(changed_rows) == 8
    assert "token_emb.weight" in unchanged
    assert "pos_emb" in unchanged
    assert "lm_head.weight" in unchanged
    assert any(k.endswith("mlp.0.weight") for k in unchanged)
    assert any(k.endswith("ln1.weight") for k in unchanged)
    assert any(k.endswith(".bias") for k in unchanged)


def test_token_emb_not_rewritten(screening):
    seeds = derive_seeds(42)
    factory = _factory(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    before = tensor_sha256(base.token_emb.weight)
    clone = clone_from_base_state(base, factory)
    apply_attention_intervention(clone, "ce_lcg", seeds)
    assert tensor_sha256(clone.token_emb.weight) == before == hashes["token_emb.weight"]


def test_same_base_hash_across_methods(screening):
    seeds = derive_seeds(42)
    factory = _factory(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    non_attn = [n for n in hashes if n not in allow]
    for method in METHODS:
        clone = clone_from_base_state(base, factory)
        apply_attention_intervention(clone, method, seeds, allowlist=allow)
        assert_t0_invariance(clone, hashes, allow)
        for name, param in clone.named_parameters():
            if name in allow:
                continue
            assert tensor_sha256(param) == hashes[name]
        assert non_attn


def test_method_order_independence(screening):
    seeds = derive_seeds(42)
    factory = _factory(screening)
    allow = attention_allowlist(factory())

    def hashes_for(order):
        base, _ = build_base_state(factory, seeds.seed_model)
        out = {}
        for method in order:
            clone = clone_from_base_state(base, factory)
            apply_attention_intervention(clone, method, seeds, allowlist=allow)
            out[method] = {
                name: tensor_sha256(dict(clone.named_parameters())[name])
                for name in allow
            }
        return out

    a = hashes_for(list(METHODS))
    b = hashes_for(list(reversed(METHODS)))
    assert a == b


def test_batch_order_parity_same_seed():
    p1 = epoch_index_permutations(17, 3, seed_shuffle=42 + 20011)
    p2 = epoch_index_permutations(17, 3, seed_shuffle=42 + 20011)
    assert p1 == p2
    assert hash_int_sequence(p1[0]) == hash_int_sequence(p2[0])
    p3 = epoch_index_permutations(17, 3, seed_shuffle=99)
    assert p1 != p3


def test_one_allowlisted_tensor_left_unchanged_fails(screening):
    seeds = derive_seeds(42)
    factory = _factory(screening)
    base, hashes = build_base_state(factory, seeds.seed_model)
    allow = attention_allowlist(base)
    clone = clone_from_base_state(base, factory)
    apply_attention_intervention(clone, "xavier_g1.0", seeds, allowlist=allow)
    # restore one allowlisted tensor to base_state
    victim = allow[0]
    with torch.no_grad():
        dict(clone.named_parameters())[victim].copy_(
            dict(base.named_parameters())[victim]
        )
    with pytest.raises(AssertionError, match="left unchanged"):
        assert_t0_invariance(clone, hashes, allow)


def test_historical_screening_untouched():
    text = SCREEN.read_text(encoding="utf-8")
    assert "test_accuracy" in text  # legacy 1/loss field remains in historical runner
    assert "get_wikitext2_dataloaders" in text
