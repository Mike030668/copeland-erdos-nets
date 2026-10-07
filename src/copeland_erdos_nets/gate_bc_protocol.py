"""Gate B+C primitives. Canonical execution is deliberately not authorized.

DS GATE_BC_DESIGN_DS_REVIEW.md, 2026-10-07, is the binding protocol.
All diagnostics below are detached reads; alpha is a fixed Python scalar.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .r010_protocol import collect_named_tensors, hash_int_sequence
from .r013_protocol import assert_no_weight_tying, cosine_and_maxdiff, rms, xavier_scalar_std

CONDITIONS = ("B00", "B01", "B10", "B11", "C_002")
RESERVED_SEEDS = (67, 68, 69, 70, 71)
SMOKE_SEED = 1067


class HardGateError(RuntimeError):
    """Preserve failed evidence and stop before any further optimization."""


class TokenScaleMixin:
    """Same inherited decoder path, with alpha only on token lookup output."""
    token_alpha = 1.0

    def pre_block(self, ids):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        return self.token_alpha * self.token_emb(ids) + self.pos_emb[:, :ids.shape[1], :]

    def forward(self, ids):
        x = self.pre_block(ids)
        mask = self.bias[:, :, :ids.shape[1], :ids.shape[1]]
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.ln_f(x))


def tensor_digest(t):
    """SHA-256 of exact native tensor bytes, dtype/shape separately recorded."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def construct_cells(base, vocab, d_model):
    assert_no_weight_tying(base)
    w0 = base.token_emb.weight.detach().clone()
    large, small = rms(w0), xavier_scalar_std(vocab, d_model)
    if not math.isfinite(large) or large <= 0:
        raise HardGateError("invalid constructor RMS")
    # One realized small tensor is copied to both cells: identical stored bytes.
    small_w = w0 * (small / large)
    specifications = {
        "B00": (small_w, 1.0, small, small),
        "B01": (small_w, large / small, small, large),
        "B10": (w0, small / large, large, small),
        "B11": (w0, 1.0, large, large),
        "C_002": (w0 * (0.02 / large), 1.0, 0.02, 0.02),
    }
    cells, rows = {}, []
    for name, (w, alpha, target, effective_target) in specifications.items():
        m = copy.deepcopy(base)
        with torch.no_grad():
            m.token_emb.weight.copy_(w)
        m.token_alpha = alpha
        cells[name] = m
        rows.append({"condition": name, "alpha": alpha, "small_rms": small,
                     "large_rms": large, "target_stored_rms": target,
                     "stored_rms": rms(w), "target_effective_rms": effective_target,
                     "effective_rms": abs(alpha) * rms(w), "embedding_sha256": tensor_digest(w),
                     "constructor_sha256": tensor_digest(w0),
                     "base_direction_sha256": tensor_digest(w0/large)})
    return cells, rows


def audit_cells(base, cells, construction, fixture):
    """Actual realized tensor/forward gates. Raises with all collected receipts."""
    rows = []
    def gate(name, passed, **details):
        rows.append({"gate": name, "pass": bool(passed), **details})
    gate("exact_condition_set", tuple(cells) == CONDITIONS)
    if tuple(cells) != CONDITIONS:
        raise HardGateError(json.dumps(rows))
    by = {r["condition"]: r for r in construction}
    bt = collect_named_tensors(base)
    gate("independent_cell_storage", len({m.token_emb.weight.data_ptr() for m in cells.values()})==5)
    for name, m in cells.items():
        row = by[name]
        realized = rms(m.token_emb.weight)
        effective = abs(m.token_alpha) * realized
        cos, diff = cosine_and_maxdiff(m.token_emb.weight, base.token_emb.weight)
        gate(f"{name}:stored_rms", abs(realized-row["target_stored_rms"]) < 1e-6,
             realized=realized, target=row["target_stored_rms"])
        gate(f"{name}:effective_rms", abs(effective-row["target_effective_rms"]) < 1e-6,
             realized=effective, target=row["target_effective_rms"])
        gate(f"{name}:direction", cos >= 0.999999 and diff <= 1e-4,
             cosine_similarity=cos, normalized_max_abs_diff=diff)
        now = collect_named_tensors(m)
        gate(f"{name}:nonembedding_identity", set(now)==set(bt) and all(
            now[k].dtype==bt[k].dtype and now[k].shape==bt[k].shape and
            tensor_digest(now[k])==tensor_digest(bt[k]) for k in bt if k != "token_emb.weight"))
        gate(f"{name}:fp32_parameters", all(p.dtype==torch.float32 for p in m.parameters()))
        gate(f"{name}:untied", m.token_emb.weight.data_ptr() != m.lm_head.weight.data_ptr())
        gate(f"{name}:fixed_alpha", isinstance(m.token_alpha, float) and
             "token_alpha" not in dict(m.named_parameters()) and m.token_alpha==row["alpha"] and m.token_alpha > 0)
    for a, b in (("B00", "B01"), ("B10", "B11")):
        gate(f"stored_identity:{a}:{b}", torch.equal(cells[a].token_emb.weight, cells[b].token_emb.weight))
    gate("B11:constructor_bytes", torch.equal(cells["B11"].token_emb.weight, base.token_emb.weight))
    with torch.no_grad():
        for a, b in (("B00", "B10"), ("B01", "B11")):
            ha, hb = cells[a].pre_block(fixture), cells[b].pre_block(fixture)
            gate(f"matched_forward:{a}:{b}", torch.allclose(ha, hb, rtol=1e-6, atol=1e-7),
                 max_abs_diff=float((ha-hb).abs().max()), rtol=1e-6, atol=1e-7,
                 fixture_sha256=tensor_digest(fixture))
    if not all(r["pass"] for r in rows):
        error = HardGateError("construction/forward parity failed")
        error.receipts = rows
        raise error
    return rows


def batch_schedule_receipt(permutations, n_samples, batch_size, drop_last=True):
    """Validate all 15 full permutations before optimization, emit 75 records."""
    if len(permutations) != 15 or n_samples < batch_size:
        raise HardGateError("full15 schedule required")
    rows = []
    expected = list(range(n_samples))
    for epoch, order in enumerate(permutations, 1):
        if sorted(order) != expected:
            raise HardGateError("invalid or duplicated epoch sample indices")
        used = order[:len(order)//batch_size*batch_size] if drop_last else order
        digest = hash_int_sequence(used)
        boundaries = hash_int_sequence(range(0, len(used)+1, batch_size))
        for condition in CONDITIONS:
            rows.append({"condition": condition, "epoch": epoch,
                         "permutation_sha256": hash_int_sequence(order), "batch_order_sha256": digest,
                         "batch_boundaries_sha256": boundaries, "samples": len(used),
                         "batches": (len(used)+batch_size-1)//batch_size})
    return rows


def gradient_stats(model):
    g = model.token_emb.weight.grad
    if g is None:
        raise HardGateError("missing token gradient")
    x = g.detach().double()
    return float(x.norm()), rms(x)


def epoch_telemetry(model, start, gradients):
    end = model.token_emb.weight.detach().double()
    displacement = end-start
    norm = float(start.norm())
    if norm == 0:
        raise HardGateError("zero start embedding norm")
    return {"stored_embedding_rms": rms(end),
            "effective_embedding_rms": abs(model.token_alpha)*rms(end),
            "embedding_gradient_l2_mean": sum(x[0] for x in gradients)/len(gradients),
            "embedding_gradient_rms_mean": sum(x[1] for x in gradients)/len(gradients),
            "epoch_displacement_l2": float(displacement.norm()),
            "epoch_displacement_rms": rms(displacement),
            "relative_epoch_displacement": float(displacement.norm())/norm}


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name+".pending")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def atomic_checkpoint(path, value):
    p = Path(path)
    tmp = p.with_name(p.name+".pending")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def execution_gate(seed, mode, epochs):
    if mode != "smoke" or seed != SMOKE_SEED or epochs != 1:
        raise HardGateError("only NONCANONICAL_SMOKE seed1067, one epoch is authorized; canonical closed")
