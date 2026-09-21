"""CE-P1 RMS-controlled token-embedding constructions.

The CE-LCG and shuffled-CE conditions originate from one standardized CE
value vector.  They therefore differ only in placement, while every condition
is rescaled to the same Xavier target RMS.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from scipy.stats import norm as scipy_norm

from .assignment import apply_assignment
from .ce_stream import blocks_to_uniform, take_blocks
from .r010_protocol import tensor_sha256
from .r013_protocol import rms, xavier_scalar_std

CONDITIONS = ("Gaussian_Xavier_RMS", "CE_LCG_Xavier_RMS", "CE_SHUFFLED_Xavier_RMS")
CE_BLOCK_DIGITS = 8


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std <= 1e-12:
        raise RuntimeError("CE-P1 HARD STOP: degenerate CE value vector")
    return (values - values.mean()) / std


def _exact_rms(values: torch.Tensor, target_rms: float) -> torch.Tensor:
    return values * (float(target_rms) / rms(values))


def ce_standardized_values(n: int, *, m: int = CE_BLOCK_DIGITS) -> np.ndarray:
    """The single CE inverse-normal value multiset shared by B and C."""
    blocks = take_blocks(m=m, num_blocks=n, offset_blocks=0)
    uniform = np.asarray(blocks_to_uniform(blocks, m), dtype=np.float64)
    raw = scipy_norm.ppf(np.clip(uniform, 1e-7, 1.0 - 1e-7))
    return _standardize(raw)


def embedding_for_condition(
    condition: str, shape: tuple[int, int], *, seed_embedding: int, seed_shuffle: int,
    target_rms: float | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build a CPU float32 embedding and a reproducible construction receipt."""
    if condition not in CONDITIONS:
        raise ValueError(f"unauthorized CE-P1 condition: {condition}")
    n = math.prod(shape)
    target = float(target_rms if target_rms is not None else xavier_scalar_std(*shape))
    source_multiset_sha256 = ""
    if condition == "Gaussian_Xavier_RMS":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed_embedding) % (2**31 - 1))
            values = torch.randn(n, dtype=torch.float64)
        values = values - values.mean()
        placement = "seed_bound_gaussian"
    else:
        source = ce_standardized_values(n)
        source_multiset_sha256 = hashlib.sha256(np.sort(source).tobytes()).hexdigest()
        strategy = "lcg" if condition == "CE_LCG_Xavier_RMS" else "shuffled"
        placed = apply_assignment(source, shape, strategy=strategy, seed=int(seed_shuffle)).reshape(-1)
        values = torch.from_numpy(placed.copy())
        placement = strategy
    embedding = _exact_rms(values.reshape(shape), target).to(torch.float32)
    # Float32 conversion can perturb RMS; correct once in its final dtype.
    embedding = _exact_rms(embedding, target)
    return embedding, {
        "condition": condition,
        "target_rms": target,
        "realized_rms": rms(embedding),
        "embedding_sha256": tensor_sha256(embedding),
        "source_multiset_sha256": source_multiset_sha256,
        "placement": placement,
        "seed_embedding": int(seed_embedding),
        "seed_shuffle": int(seed_shuffle),
        "ce_block_digits": CE_BLOCK_DIGITS if condition != "Gaussian_Xavier_RMS" else "",
    }


def multiset_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact value-multiset check for the two CE placements."""
    aa = torch.sort(a.detach().cpu().reshape(-1))[0]
    bb = torch.sort(b.detach().cpu().reshape(-1))[0]
    return bool(torch.equal(aa, bb))
