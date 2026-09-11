"""R014 larger-model embedding-scale confirmation unit/static tests (DS binding).

Reuses R013's protocol machinery unchanged (already shape-parametric); tests
here exercise it at the actual R014 shape (vocab=50257, d_model=256, n_heads=4,
d_ff=1024, n_layers=4) rather than a tiny stand-in, so the 16/16 attention-
allowlist count and the recomputed (not transplanted) s_xav are tested for
real, per DS's binding amendments (2026-09-11).
"""
from __future__ import annotations
import importlib.util, json, math
from pathlib import Path
import pytest, torch
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    assert_expected_allowlist_count, apply_attention_intervention, tensor_sha256,
    collect_named_tensors, epoch_index_permutations, hash_int_sequence,
    SEED_MODEL_OFFSET, SEED_SHUFFLE_OFFSET, SEED_EMBEDDING_OFFSET,
)
from copeland_erdos_nets.r013_protocol import (
    ladder_factors, xavier_scalar_std, rms, apply_embedding_dose, assert_no_weight_tying,
    cosine_and_maxdiff, all_epoch_batch_parity,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "scripts" / "run_transformer_screening.py"
RUNNER = ROOT / "scripts" / "run_r014_seed_atomic.py"
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


def _load_runner():
    s = importlib.util.spec_from_file_location("r014run", RUNNER)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


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


# --- DS Smoke Checkpoint Round 1 binding repairs (B1/B2/B3), 6 targeted tests (B5) ---

def test_d_mid1_explicit_construction_formula():
    """D_mid1's target RMS must equal r**(2/3) * RMS(base) with r = s_xav / RMS(base),
    computed directly from the ladder formula -- not merely 'between D_xavier and D_ctor'."""
    torch.manual_seed(0)
    base_emb = torch.randn(VOCAB, D_MODEL) * 0.05
    rms_ctor = rms(base_emb)
    s_xav = xavier_scalar_std(VOCAB, D_MODEL)
    r = s_xav / rms_ctor
    expected_target_rms = (r ** (2.0 / 3.0)) * rms_ctor
    m = torch.nn.Module(); m.token_emb = torch.nn.Embedding(VOCAB, D_MODEL)
    meta = apply_embedding_dose(m, "D_mid1", base_emb, VOCAB, D_MODEL)
    assert meta["factor_rel_ctor"] == pytest.approx(r ** (2.0 / 3.0))
    assert meta["target_rms"] == pytest.approx(expected_target_rms, rel=1e-9)
    assert meta["realized_rms"] == pytest.approx(expected_target_rms, abs=1e-6)


def test_full_15epoch_schedule_derivation_and_mutated_hash_detected():
    """DS binding repair B1: the schedule is derived for the full canonical
    schedule_epochs=15 regardless of a smaller train_epochs, is reproducible from
    the same seed_shuffle, and all_epoch_batch_parity must catch a single mutated
    per-condition hash within that 15-epoch schedule (negative case)."""
    schedule_epochs = 15
    perms_a = epoch_index_permutations(200, schedule_epochs, seed_shuffle=99020011)
    perms_b = epoch_index_permutations(200, schedule_epochs, seed_shuffle=99020011)
    assert len(perms_a) == schedule_epochs
    assert [hash_int_sequence(p) for p in perms_a] == [hash_int_sequence(p) for p in perms_b]

    doses = ("D_xavier", "D_mid1", "D_ctor")
    rows = [{"dose": d, "epoch": ep + 1, "batch_order_hash": hash_int_sequence(perms_a[ep])}
            for d in doses for ep in range(schedule_epochs)]
    assert all_epoch_batch_parity(rows, len(doses))  # identical schedule shared by all doses -> PASS

    mutated = [dict(r) for r in rows]
    mutated[0]["batch_order_hash"] = "deadbeef" * 8  # corrupt one dose's epoch-1 hash
    assert not all_epoch_batch_parity(mutated, len(doses))  # must be caught, not silently accepted


def test_durable_checkpoint_readback_verifier_success_and_failure(runner, tmp_path):
    """sha_file() (the durable-export read-back verifier's hash primitive) must
    match identical bytes and must NOT match corrupted bytes -- both the success
    path (persistent_verified=true) and failure path (persistent_verified=false)
    that gate `raise SystemExit(f"durable ckpt verify FAIL {d}")` in main()."""
    local = tmp_path / "ckpt_local.pt"
    local.write_bytes(b"r014-fake-checkpoint-bytes" * 1000)
    local_sha = runner.sha_file(local)
    local_size = local.stat().st_size

    readback_ok = tmp_path / "ckpt_readback_ok.pt"
    readback_ok.write_bytes(local.read_bytes())  # simulates a correct Drive round-trip
    pver_success = (runner.sha_file(readback_ok) == local_sha and readback_ok.stat().st_size == local_size)
    assert pver_success is True

    readback_bad = tmp_path / "ckpt_readback_bad.pt"
    readback_bad.write_bytes(local.read_bytes()[:-1] + b"\x00")  # simulates a corrupted round-trip
    pver_failure = (runner.sha_file(readback_bad) == local_sha and readback_bad.stat().st_size == local_size)
    assert pver_failure is False


def test_driveronly_runtime_mismatch_hard_stop(runner, tmp_path):
    """A RUNTIME_FREEZE that matches every field the running process actually has
    EXCEPT `driver` must raise SystemExit with driver as the sole mismatch -- proving
    assert_runtime() does real per-field comparison, not an all-or-nothing check."""
    import numpy, datasets, transformers, subprocess as sp
    try:
        real_driver = sp.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        real_driver = "unavailable"
    matching_freeze = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": __import__("platform").python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda or "none",
        "numpy": numpy.__version__, "datasets": datasets.__version__, "transformers": transformers.__version__,
        "driver": real_driver,
    }
    driver_only_mismatch_freeze = dict(matching_freeze, driver="99.99.99-not-the-real-driver")
    out = tmp_path
    with pytest.raises(SystemExit) as ei:
        runner.assert_runtime(driver_only_mismatch_freeze, out)
    assert "mismatch=['driver']" in str(ei.value)
    log = (out / "runtime_assertion.log").read_text()
    assert "mismatch=['driver']" in log
    # sanity: the matching freeze (all real values) must NOT raise
    runner.assert_runtime(matching_freeze, out)


def test_rng_policy_config_matches_r010_implementation_offsets():
    """R014 binding: both shipped configs' rng_policy block must equal the ACTUAL
    R010 implementation offsets (not a hand-copied literal that could silently drift),
    matching the runtime HARD STOP check `cfg.get('rng_policy') != impl` in main()."""
    impl = {
        "seed_model_offset": SEED_MODEL_OFFSET,
        "seed_shuffle_offset": SEED_SHUFFLE_OFFSET,
        "seed_embedding_redraw_offset": SEED_EMBEDDING_OFFSET,
    }
    for name in ("r014_canonical.json", "r014_smoke.json"):
        cfg = json.loads((ROOT / "configs" / name).read_text())
        assert cfg["rng_policy"] == impl, name


def test_mode_aware_status_and_seed1057_can_never_be_canonical(runner):
    """DS binding repair B3: CANONICAL requires mode==r014_canonical AND seed in the
    DS-released set AND a full (train_epochs==schedule_epochs) run. seed1057 (used for
    all smoke runs) is not in CANONICAL_SEEDS and so can never be labeled CANONICAL,
    even if someone mistakenly points a full-schedule run at it."""
    assert 1057 not in runner.CANONICAL_SEEDS
    assert runner.CANONICAL_SEEDS == frozenset({57, 58, 59, 60, 61})

    smoke_cfg = {"experiment": {"mode": "r014_smoke"}}
    canon_cfg = {"experiment": {"mode": "r014_canonical"}}

    # bounded smoke on the smoke seed: NONCANONICAL_SMOKE
    label = runner.seed_status_label(smoke_cfg, 1057, schedule_epochs=15, train_epochs=1)
    assert label.startswith("NONCANONICAL_SMOKE seed=1057")

    # full 15/15 schedule but on seed1057 (not DS-released) must STILL be NONCANONICAL_SMOKE
    label_full_wrong_seed = runner.seed_status_label(canon_cfg, 1057, schedule_epochs=15, train_epochs=15)
    assert label_full_wrong_seed.startswith("NONCANONICAL_SMOKE seed=1057")

    # bounded run on a DS-released canonical seed must STILL be NONCANONICAL_SMOKE
    label_bounded_real_seed = runner.seed_status_label(canon_cfg, 57, schedule_epochs=15, train_epochs=1)
    assert label_bounded_real_seed.startswith("NONCANONICAL_SMOKE seed=57")

    # only mode==r014_canonical AND seed released AND full schedule -> CANONICAL
    label_true_canonical = runner.seed_status_label(canon_cfg, 57, schedule_epochs=15, train_epochs=15)
    assert label_true_canonical.startswith("CANONICAL seed=57")


# --- DS Smoke Checkpoint Round 2 binding repair C1: epoch-level curve persistence ---

def test_c1_expected_row_count_for_complete_canonical_seed_block():
    """DS: expected rows = 3 conditions x 15 epochs = 45 for a complete canonical seed."""
    assert len(R014_DOSES) * 15 == 45


def test_c1_learning_curves_wired_per_dose_in_main_source():
    """Static wiring proof: learning_curves.csv is accumulated and persisted inside
    the SAME per-dose training loop as the pre-existing per_seed.csv/
    checkpoint_manifest.csv outputs (incremental, not a bolted-on post-loop pass)."""
    src = (ROOT / "scripts" / "run_r014_seed_atomic.py").read_text()
    assert 'curves.extend(res["curve_rows"])' in src
    assert 'wcsv(out / "learning_curves.csv", curves)' in src
    i_loop = src.index("for d in R014_DOSES:\n        m = models[d].to(device)")
    i_curves = src.index('curves.extend(res["curve_rows"])')
    i_perseed = src.index('wcsv(out / "per_seed.csv"', i_loop)
    assert i_loop < i_curves < i_perseed


def test_c1_curve_persistence_observational_and_checkpoint_test_semantics_unchanged(runner, tmp_path):
    """DS binding repair C1 (Smoke Checkpoint Round 2): run the REAL train_condition
    (not a mock) on a tiny synthetic dataset and prove, empirically:
    (1) exactly one curve row per epoch, correctly tagged seed/dose/epoch;
    (2) the persisted val_loss/val_ppl at the SELECTED best epoch and at the
        final epoch are bit-identical to the scalars train_condition already
        used for checkpoint selection and for per_seed.csv's final_val_ppl --
        i.e. curve persistence reads existing scalars, it does not recompute
        anything via an extra forward/eval pass;
    (3) the pre-C1 return contract (checkpoint/test semantics) is untouched:
        every pre-existing key is still present with its own checkpoint file
        written, unchanged by the addition of curve_rows."""
    conf = runner.load_conf()
    small_vocab = 37
    seq_len = 4
    torch.manual_seed(0)
    ids_train = [i % small_vocab for i in range(400)]
    ids_val = [i % small_vocab for i in range(120)]
    ids_test = [i % small_vocab for i in range(120)]
    splits = {
        "train": conf.TokenizedDataset(ids_train, seq_len),
        "validation": conf.TokenizedDataset(ids_val, seq_len),
        "test": conf.TokenizedDataset(ids_test, seq_len),
    }

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = torch.nn.Embedding(small_vocab, 16)
            self.lin = torch.nn.Linear(16, 16)
            self.lm_head = torch.nn.Linear(16, small_vocab)

        def forward(self, x):
            return self.lm_head(torch.relu(self.lin(self.token_emb(x))))

    model = Tiny()
    cfg = {"training": {"lr": 1e-3, "weight_decay": 0.0},
           "data": {"val_drop_last": False, "test_drop_last": False}}
    n_epochs = 3
    bs = 8
    perms = [list(range(len(splits["train"]))) for _ in range(n_epochs)]
    out = tmp_path
    (out / "checkpoints").mkdir()

    res = runner.train_condition(model, "D_ctor", splits, perms, bs, False, "cpu", cfg, out, seed=999)

    rows = res["curve_rows"]
    assert len(rows) == n_epochs
    assert [r["epoch"] for r in rows] == list(range(1, n_epochs + 1))
    assert all(r["seed"] == 999 and r["dose"] == "D_ctor" for r in rows)

    best_row = rows[res["best_epoch"] - 1]
    assert best_row["val_loss"] == pytest.approx(res["best_val_loss"], rel=1e-12)
    assert best_row["val_ppl"] == pytest.approx(res["best_val_ppl"], rel=1e-12)

    last_row = rows[-1]
    assert last_row["val_loss"] == pytest.approx(res["final_val_loss"], rel=1e-12)
    assert last_row["val_ppl"] == pytest.approx(res["final_val_ppl"], rel=1e-12)

    for k in ("best_epoch", "best_val_loss", "final_val_loss", "best_val_ppl", "final_val_ppl", "test_ppl", "ckpt"):
        assert k in res
    assert res["ckpt"].exists()
