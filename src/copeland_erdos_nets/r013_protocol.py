"""R013 embedding scale-response primitives (DS AUTHORIZATION 2026-09-03).

Vary ONLY initial token_emb.weight RMS along a preregistered ladder; direction
fixed to constructor base direction. No Xavier random draw (scalar target only).
"""
from __future__ import annotations
import torch, torch.nn as nn
from copeland_erdos_nets.r010_protocol import tensor_sha256

# ladder factors relative to constructor RMS; r = s_xav / RMS_constructor
def xavier_scalar_std(vocab: int, d_model: int) -> float:
    return (2.0/(vocab + d_model))**0.5

def rms(t: torch.Tensor) -> float:
    return float(t.detach().double().pow(2).mean().sqrt().item())

def ladder_factors(r: float) -> dict:
    return {
        "D_below": r/2.0,
        "D_xavier": r,
        "D_mid1":  r**(2.0/3.0),
        "D_mid2":  r**(1.0/3.0),
        "D_ctor":  1.0,
    }

DOSES = ("D_below","D_xavier","D_mid1","D_mid2","D_ctor")

def assert_no_weight_tying(model: nn.Module) -> None:
    te = model.token_emb.weight
    lm = getattr(getattr(model,"lm_head",None),"weight",None)
    if lm is not None and te.data_ptr()==lm.data_ptr():
        raise SystemExit("HARD STOP: token_emb.weight shares storage with lm_head.weight (weight tying)")

def apply_embedding_dose(model: nn.Module, dose: str, base_emb: torch.Tensor,
                         vocab: int, d_model: int) -> dict:
    """D_ctor = exact base.clone(); others = base * (target_rms / RMS(base))."""
    base = base_emb.detach()
    rms_ctor = rms(base)
    s_xav = xavier_scalar_std(vocab, d_model)
    r = s_xav / rms_ctor
    facs = ladder_factors(r)
    if dose not in facs: raise ValueError(dose)
    if dose == "D_ctor":
        emb = base.clone()
        target_rms = rms_ctor
    else:
        target_rms = facs[dose]*rms_ctor  # factor is relative-to-constructor => target RMS
        emb = base * (target_rms / rms_ctor)
    model.token_emb.weight.data.copy_(emb.to(model.token_emb.weight.device))
    u = (base/rms_ctor)
    return {
        "dose": dose, "factor_rel_ctor": facs[dose],
        "s_xav": s_xav, "r": r, "rms_constructor": rms_ctor,
        "target_rms": target_rms, "realized_rms": rms(model.token_emb.weight),
        "base_direction_hash": tensor_sha256(u.float()),
        "emb_hash": tensor_sha256(model.token_emb.weight),
    }


def cosine_and_maxdiff(realized: torch.Tensor, base: torch.Tensor):
    """Actual direction audit from realized tensors."""
    a=realized.detach().double().flatten(); b=base.detach().double().flatten()
    cos=float(torch.dot(a,b).item()/(a.norm().item()*b.norm().item()+1e-30))
    an=a/(a.norm()+1e-30); bn=b/(b.norm()+1e-30)
    mad=float((an-bn).abs().max().item())
    return cos, mad
