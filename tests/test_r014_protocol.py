"""R014 larger-model embedding-scale confirmation unit/static tests (DS binding).

Reuses R013's protocol machinery unchanged (already shape-parametric); tests
here exercise it at the actual R014 shape (vocab=50257, d_model=256, n_heads=4,
d_ff=1024, n_layers=4) rather than a tiny stand-in, so the 16/16 attention-
allowlist count and the recomputed (not transplanted) s_xav are tested for
real, per DS's binding amendments (2026-09-11).
"""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import pytest, torch
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    assert_expected_allowlist_count, apply_attention_intervention, tensor_sha256,
    collect_named_tensors,
)
from copeland_erdos_nets.r013_protocol import (
    ladder_factors, xavier_scalar_std, rms, apply_embedding_dose, assert_no_weight_tying,
    cosine_and_maxdiff,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"
R014_DOSES = ("D_xavier", "D_mid1", "D_ctor")

# Real R014 shape (former R009, recovered from configs/r009_scaleup.json model dims;
# vocab is NOT taken from that config -- see test_vocab_provenance_hard_assert_in_source).
VOCAB = 50257
D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 4


def _load():
    s = importlib.util.spec_from_file_location("scr", SCREEN)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def screening():
    return _load()


def _r014_shape(sc):
    def f():
        return sc.DecoderOnlyTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
                                          d_ff=D_FF, n_layers=N_LAYERS, max_seq_len=16)
    return f


def _setup(sc, seed=1057):
    s = derive_seeds(seed); f = _r014_shape(sc)
    base, base_h = build_base_state(f, s.seed_model, device="cpu")
    allow = attention_allowlist(base); base_emb = base.token_emb.weight.detach().clone()
    out = {}
    for d in R014_DOSES:
        m = clone_from_base_state(base, f)
        meta = apply_embedding_dose(m, d, base_emb, VOCAB, D_MODEL)
        apply_attention_intervention(m, "xavier_g1.0", s, allowlist=allow)
        out[d] = (m, meta)
    return base, base_h, allow, base_emb, out


def test_attention_allowlist_is_16_not_8(screening):
    """DS binding amendment #2: 4-layer model has 16 Q/K/V/O weights, not 8."""
    base, base_h, allow, base_emb, out = _setup(screening)
    assert len(allow) == 16
    assert_expected_allowlist_count(allow, n_layers=N_LAYERS)  # must not raise
    with pytest.raises(AssertionError):
        assert_expected_allowlist_count(allow, n_layers=2)  # wrong n_layers must raise


def test_s_xav_recomputed_for_larger_shape_not_transplanted():
    """DS binding amendment #1 corollary: s_xav must be computed for THIS shape, ~0.0062924,
    not the 2-layer model's ~0.0063003 (different denominator: vocab+256 vs vocab+128)."""
    s_xav_r014 = xavier_scalar_std(VOCAB, D_MODEL)
    s_xav_r013 = xavier_scalar_std(VOCAB, 128)
    assert abs(s_xav_r014 - math.sqrt(2 / (VOCAB + D_MODEL))) < 1e-15
    assert s_xav_r014 != s_xav_r013
    assert abs(s_xav_r014 - 0.0062924) < 1e-6


def test_dctor_equals_base(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    assert tensor_sha256(out["D_ctor"][0].token_emb.weight) == tensor_sha256(base_emb)


def test_direction_fixed_across_conditions(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    hs = {out[d][1]["base_direction_hash"] for d in R014_DOSES}
    assert len(hs) == 1


def test_realized_rms_matches_target(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    for d in R014_DOSES:
        meta = out[d][1]
        assert abs(meta["realized_rms"] - meta["target_rms"]) < 1e-6


def test_only_embedding_changes(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    for d in R014_DOSES:
        now = collect_named_tensors(out[d][0])
        for n, t in now.items():
            if n == "token_emb.weight" or n in allow:
                continue
            assert tensor_sha256(t) == base_h[n], (d, n)


def test_attention_identical_across_conditions(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    ref = {n: tensor_sha256(dict(out["D_ctor"][0].named_parameters())[n]) for n in allow}
    for d in R014_DOSES:
        p = dict(out[d][0].named_parameters())
        for n in allow:
            assert tensor_sha256(p[n]) == ref[n]


def test_no_weight_tying(screening):
    base, _, _, _, out = _setup(screening)
    assert_no_weight_tying(out["D_ctor"][0])


def test_direction_audit_within_tolerance(screening):
    base, base_h, allow, base_emb, out = _setup(screening)
    for d in R014_DOSES:
        cos, mad = cosine_and_maxdiff(out[d][0].token_emb.weight, base_emb)
        assert cos >= 0.999999 and mad <= 1e-4


def test_forward_backward_pass_at_r014_shape(screening):
    """Model constructs and runs a forward/backward pass at the exact R014 shape
    without shape errors, on CPU, before any GPU session is requested."""
    m = _r014_shape(screening)()
    x = torch.randint(0, VOCAB, (2, 8))
    y = torch.randint(0, VOCAB, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, VOCAB)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    assert m.token_emb.weight.grad is not None


def test_vocab_provenance_hard_assert_in_source():
    """Static proof the runner hard-asserts effective vocab==50257 and persists
    the historical config's unused 28996 value separately (DS binding amendment #1)."""
    src = (ROOT / "scripts" / "run_r014_seed_atomic.py").read_text()
    assert "VOCAB HARD STOP" in src
    assert "HISTORICAL_R009_CONFIG_VOCAB_SIZE = 28996" in src
    assert "EXPECTED_EFFECTIVE_VOCAB_SIZE = 50257" in src
    assert "vocab_provenance.json" in src


def test_allowlist_count_gate_present_in_source():
    """Static proof the 16/16 count gate (dropped implicitly in R012/R013's runners
    at n_layers=2) is explicitly restored and wired into parity_summary for R014."""
    src = (ROOT / "scripts" / "run_r014_seed_atomic.py").read_text()
    assert "assert_expected_allowlist_count(allow, n_layers=n_layers)" in src
    assert "attention_allowlist_count_16" in src


def test_pregate_before_train_condition_in_source():
    """Static ordering proof: pre-training batch HARD GATE precedes first scientific
    train_condition call."""
    src = (ROOT / "scripts" / "run_r014_seed_atomic.py").read_text()
    i_gate = src.index("PRE-TRAINING HARD STOP")
    i_train_call = src.index("res = train_condition(")
    assert i_gate < i_train_call
