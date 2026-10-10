"""Synthetic protocol/negative tests. No reserved seeds or scientific metrics."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from copeland_erdos_nets.gate_bc_protocol import (
    CONDITIONS, HardGateError, TokenScaleMixin, atomic_checkpoint, atomic_json,
    audit_cells, batch_schedule_receipt, construct_cells, execution_gate, tensor_digest,
)
from copeland_erdos_nets.r010_protocol import epoch_index_permutations

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture():
    hist = module(ROOT/"scripts/run_transformer_screening.py")
    class Model(TokenScaleMixin, hist.DecoderOnlyTransformer):
        pass
    with torch.random.fork_rng():
        torch.manual_seed(7)
        m = Model(vocab_size=17, d_model=8, n_heads=2, d_ff=16, n_layers=1, max_seq_len=4)
        with torch.no_grad():
            m.pos_emb.fill_(0.3)
    return m, torch.arange(8).reshape(2,4)


def test_five_cells_actual_forward_and_direction():
    base, ids = fixture()
    cells, rows = construct_cells(base, 17, 8)
    receipts = audit_cells(base, cells, rows, ids)
    assert all(r["pass"] for r in receipts)
    assert torch.equal(cells["B00"].token_emb.weight, cells["B01"].token_emb.weight)
    assert torch.equal(cells["B11"].token_emb.weight, base.token_emb.weight)
    assert cells["C_002"].token_alpha == 1
    # Independent model storage, including the identical stored pairs.
    assert len({m.token_emb.weight.data_ptr() for m in cells.values()}) == 5


def test_token_only_scaling_nonzero_pos_and_gradient():
    base, ids = fixture()
    alpha = 3.0
    base.token_alpha = alpha
    actual = base.pre_block(ids)
    expected = alpha*base.token_emb(ids)+base.pos_emb[:, :ids.shape[1]]
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, alpha*(base.token_emb(ids)+base.pos_emb[:, :ids.shape[1]]))
    actual.sum().backward()
    assert torch.equal(base.pos_emb.grad, torch.full_like(base.pos_emb, 2.0))
    assert torch.equal(base.token_emb.weight.grad[:8], torch.full_like(base.token_emb.weight.grad[:8], alpha))
    assert "token_alpha" not in base.state_dict()


def test_alpha1_inherited_logits_identical():
    base, ids = fixture()
    hist = module(ROOT/"scripts/run_transformer_screening.py")
    inherited = hist.DecoderOnlyTransformer(17,8,2,16,1,4)
    inherited.load_state_dict(base.state_dict())
    assert torch.equal(base(ids), inherited(ids))


@pytest.mark.parametrize("corrupt", ["stored_rms", "direction", "nonembedding", "alpha", "tie", "pair", "forward", "negative_alpha", "dtype", "alias"])
def test_negative_construction_hard_stop(corrupt):
    base, ids = fixture()
    cells, rows = construct_cells(base, 17, 8)
    with torch.no_grad():
        if corrupt == "stored_rms":
            cells["C_002"].token_emb.weight.mul_(2)
        elif corrupt == "direction":
            cells["C_002"].token_emb.weight.neg_()
        elif corrupt == "nonembedding":
            cells["B01"].pos_emb.add_(1)
        elif corrupt == "alpha":
            cells["B10"].token_alpha *= 2
        elif corrupt == "tie":
            cells["B11"].lm_head.weight = cells["B11"].token_emb.weight
        elif corrupt == "pair":
            cells["B01"].token_emb.weight[0,0] += 0.00001
        elif corrupt == "forward":
            cells["B10"].pre_block = lambda ids: 2*cells["B10"].token_emb(ids)
        elif corrupt == "negative_alpha":
            cells["C_002"].token_alpha = -1.0
        elif corrupt == "dtype":
            cells["C_002"].pos_emb.data = cells["C_002"].pos_emb.data.double()
        elif corrupt == "alias":
            cells["B01"].token_emb.weight = cells["B00"].token_emb.weight
    with pytest.raises(HardGateError):
        audit_cells(base, cells, rows, ids)


def test_full15_schedule_before_optimization():
    perms = epoch_index_permutations(13, 15, 20018)
    rows = batch_schedule_receipt(perms, 13, 3)
    assert len(rows)==75
    for epoch in range(1,16):
        rs = [r for r in rows if r["epoch"]==epoch]
        assert len({r["batch_order_sha256"] for r in rs})==1
        assert rs[0]["samples"]==12
    with pytest.raises(HardGateError):
        batch_schedule_receipt(perms[:1], 13, 3)
    perms[2][1] = perms[2][0]
    with pytest.raises(HardGateError):
        batch_schedule_receipt(perms, 13, 3)


@pytest.mark.parametrize("seed,mode,epochs", [(67,"canonical",1),(68,"smoke",1),(1067,"canonical",1),(1067,"smoke",15),(7,"smoke",1)])
def test_execution_authorization_failclosed_without_rng(seed, mode, epochs):
    before = torch.get_rng_state().clone()
    with pytest.raises(HardGateError):
        execution_gate(seed, mode, epochs)
    assert torch.equal(before, torch.get_rng_state())
    execution_gate(1067,"smoke",1)


def test_atomic_checkpoint_state_roundtrip(tmp_path):
    base, _ = fixture()
    opt = torch.optim.AdamW(base.parameters())
    payload = {"model": base.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state(), "alpha": 1.0}
    p = tmp_path/"current.pt"
    atomic_checkpoint(p, payload)
    back = torch.load(p, weights_only=False)
    assert torch.equal(back["rng"], payload["rng"])
    assert all(torch.equal(back["model"][k], v) for k,v in payload["model"].items())
    atomic_json(tmp_path/"state.json", {"status": "running"})
    assert json.loads((tmp_path/"state.json").read_text())["status"]=="running"
    assert not list(tmp_path.glob("*.pending"))


def test_telemetry_onoff_complete_state_and_rng():
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    before = torch.get_rng_state().clone()
    np_before = np.random.get_state()
    receipt = runner.telemetry_equivalence_receipt()
    assert receipt["pass"] and all(receipt["fields_bit_identical"].values())
    assert torch.equal(before, torch.get_rng_state())
    assert np.array_equal(np_before[1], np.random.get_state()[1])


def test_runtime_negative_stops_before_any_optimization(tmp_path):
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    with pytest.raises(HardGateError, match="runtime mismatch"):
        runner.runtime_gate({"gpu": "Tesla T4"}, tmp_path)
    assert not json.loads((tmp_path/"runtime_assertion.json").read_text())["pass"]


def test_drive_transport_missing_placeholder_failclosed(tmp_path):
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    class FakeList:
        def GetList(self):
            return []
    class FakeDrive:
        def ListFile(self, request):
            return FakeList()
        def CreateFile(self, request):
            pytest.fail("must not create/upload on missing placeholder")
    d = object.__new__(runner.DriveUpdates)
    d.drive, d.out, d.prefix, d.parent = FakeDrive(), tmp_path, "prefix", "parent"
    with pytest.raises(HardGateError, match="placeholder"):
        d.upload(tmp_path/"none", "_state.json")


@pytest.mark.parametrize("field", ["lr", "schedule_epochs", "held_out_test", "alpha", "vocab", "runtime"])
def test_frozen_config_negative_before_rng(field):
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    cfg = json.loads((ROOT/"configs/gate_bc_smoke.json").read_text())
    runner.config_gate(cfg)
    if field in ("lr", "schedule_epochs", "held_out_test"):
        cfg["training"][field] = 99
    elif field == "alpha":
        cfg["experiment"]["conditions"] = ["BAD"]
    elif field == "vocab":
        cfg["data"]["accepted_structure"]["vocab"] = 100
    else:
        cfg["runtime_freeze"]["gpu"] = "cpu"
    before = torch.get_rng_state().clone()
    with pytest.raises(HardGateError):
        runner.config_gate(cfg)
    assert torch.equal(before, torch.get_rng_state())


def test_data_structure_negative():
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    with pytest.raises(HardGateError, match="data structure"):
        runner.structure_gate({"vocab": 17}, {"vocab": 50257})


@pytest.mark.parametrize("tamper", [False,True])
def test_drive_transport_verified_readback(tmp_path, tamper):
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    contents = {}
    class FakeFile:
        def SetContentFile(self, path):
            contents["uploaded"] = Path(path).read_bytes()
        def Upload(self):
            contents["uploads"] = contents.get("uploads",0)+1
        def GetContentFile(self, path):
            Path(path).write_bytes(b"corrupt" if tamper else contents["uploaded"])
    class FakeList:
        def GetList(self):
            return [{"id": "existing"}]
    class FakeDrive:
        def ListFile(self, request):
            return FakeList()
        def CreateFile(self, request):
            assert request=={"id": "existing"}
            return FakeFile()
    d = object.__new__(runner.DriveUpdates)
    d.drive, d.out, d.prefix, d.parent, d.receipts = FakeDrive(), tmp_path, "prefix", "parent", []
    from types import SimpleNamespace
    d.auth = SimpleNamespace(principal="same-sa", refresh_events=[])
    atomic_json(tmp_path/"state.json", {"status": "running"})
    if tamper:
        with pytest.raises(HardGateError, match="readback mismatch"):
            d.upload(tmp_path/"state.json", "_state.json")
    else:
        d.upload(tmp_path/"state.json", "_state.json")
        assert d.receipts[0]["readback_pass"]
    assert contents["uploads"]==1
    assert not list(tmp_path.glob(".readback*"))


def test_synthetic_smoke_durability_and_no_heldout_access(tmp_path):
    runner = module(ROOT/"scripts/run_gate_bc_seed_atomic.py")
    model, _ = fixture()
    ids = torch.arange(48).reshape(12,4)%17
    dataset = torch.utils.data.TensorDataset(ids, (ids+1)%17)
    class ForbiddenHeldOut:
        def __len__(self):
            pytest.fail("smoke must never access heldout test")
    splits = {"train": dataset, "validation": dataset, "test": ForbiddenHeldOut()}
    cfg = {"data": {"batch_size": 3}, "training": {"lr": 0.0005, "weight_decay": 0.01, "epochs": 1}}
    (tmp_path/"checkpoints").mkdir()
    uploads = []
    class Drive:
        def upload(self, path, suffix):
            assert Path(path).is_file()
            uploads.append(suffix)
    states, live = [], []
    def state(status, condition, epoch, **kwargs):
        states.append((status, condition, epoch, kwargs))
    receipt = runner.train_cell(model, "B00", splits, [list(range(12))]*15,
        cfg, tmp_path, 7, "cpu", live, state, Drive(), "synthetic-attempt", "synthetic-source")
    assert receipt["test_evaluations"]==0
    assert uploads==["_best_B00.pt", "_checkpoint_B00.pt", "_live.csv"]
    current = torch.load(tmp_path/"checkpoints/B00_current.pt", weights_only=False)
    best = torch.load(tmp_path/"checkpoints/B00_best.pt", weights_only=False)
    assert current["optimizer"]["state"] and current["rng"]
    assert current["attempt_id"]==best["attempt_id"]=="synthetic-attempt"
    assert current["source_sha"]==best["source_sha"]=="synthetic-source"
    assert current["best_epoch"]==best["epoch"]==1
    assert live[0]["is_best"] and live[0]["is_final"]
    assert states[-1][0]=="running"
