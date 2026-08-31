"""R012 embedding scale×redraw factorial primitives (DS AUTHORIZATION 2026-08-31).

Crossed construction (BINDING): single fresh_xavier draw; cells are s{0,1}*u{0,1}.
Attention fixed = same-path Xavier g=1.0. Never independently redraw S0D1/S1D1.
"""
from __future__ import annotations
import torch, torch.nn as nn
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, tensor_sha256, attention_allowlist, collect_named_tensors,
    apply_attention_intervention, clone_from_base_state,
)

R012_CELLS = ("S0D0","S0D1","S1D0","S1D1")
SEED_EMBEDDING_REDRAW_OFFSET = 30013  # matches derive_seeds().seed_embedding

def rms(t: torch.Tensor) -> float:
    return float(t.detach().double().pow(2).mean().sqrt().item())

def xavier_target_std(weight: torch.Tensor, gain: float = 1.0) -> float:
    fan_in, fan_out = weight.shape[1], weight.shape[0]
    return gain * (2.0/(fan_in+fan_out))**0.5

def build_scale_redraw_vectors(base_emb: torch.Tensor, seed_embedding_redraw: int):
    """Return (s0,u0,s1,u1, fresh_xavier). ONE fresh xavier draw, isolated RNG."""
    base = base_emb.detach().clone()
    s0 = rms(base)
    u0 = base / s0
    fresh = torch.empty_like(base, device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed_embedding_redraw) % (2**31-1))
        nn.init.xavier_normal_(fresh, gain=1.0)
    s1 = rms(fresh)
    u1 = fresh / s1
    return s0, u0, s1, u1, fresh

def cell_embedding(cell: str, s0, u0, s1, u1, base=None, fresh=None) -> torch.Tensor:
    """Anchor cells are bit-exact (S0D0==base, S1D1==fresh); off-diagonal crossed."""
    if cell=="S0D0":
        return base.clone() if base is not None else s0*u0
    if cell=="S1D1":
        return fresh.clone() if fresh is not None else s1*u1
    if cell=="S0D1":
        return s0*u1
    if cell=="S1D0":
        return s1*u0
    raise ValueError(cell)

def apply_r012_embedding(model: nn.Module, cell: str, seeds, base_emb: torch.Tensor) -> dict:
    s0,u0,s1,u1,fresh = build_scale_redraw_vectors(base_emb, seeds.seed_embedding)
    emb = cell_embedding(cell, s0,u0,s1,u1, base=base_emb, fresh=fresh).to(model.token_emb.weight.device)
    model.token_emb.weight.data.copy_(emb)
    return {"cell":cell,"s0":s0,"s1":s1,
            "u0_hash":tensor_sha256(u0),"u1_hash":tensor_sha256(u1),
            "fresh_xavier_hash":tensor_sha256(fresh),
            "emb_hash":tensor_sha256(emb),
            "seed_embedding_redraw":seeds.seed_embedding}

def embedding_scale_stats(weight: torch.Tensor) -> dict:
    w = weight.detach().double().cpu()
    return {"mean":float(w.mean()),"std":float(w.std(unbiased=True)),
            "rms":float(w.pow(2).mean().sqrt()),"l2":float(w.norm()),
            "min":float(w.min()),"max":float(w.max())}
