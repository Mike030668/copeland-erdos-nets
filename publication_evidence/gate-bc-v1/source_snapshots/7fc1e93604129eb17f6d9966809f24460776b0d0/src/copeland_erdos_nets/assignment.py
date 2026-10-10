"""Weight assignment strategies for Copeland–Erdős and Sobol initializations.

This module provides different ways to map a stream of values (CE digits or Sobol
sequences) into weight tensors, allowing diagnostics for spatial correlation.
"""

from __future__ import annotations

from typing import Sequence, Literal
import numpy as np
import torch


def _is_prime(k: int) -> bool:
    if k < 2:
        return False
    for i in range(2, int(k**0.5) + 1):
        if k % i == 0:
            return False
    return True

def _find_prime_stride(n: int) -> int:
    import math
    # Target stride around golden ratio * n
    target = int(n * 0.618033988749895)
    if target < 3:
        target = 3
    # Search upwards for a prime coprime to n
    p = target
    while True:
        if _is_prime(p) and math.gcd(p, n) == 1:
            return p
        p += 1

def _lcg_permutation(n: int) -> np.ndarray:
    # Find next power of 2
    m = 1
    while m < n:
        m *= 2
        
    # LCG parameters (numerical recipes / glibc values)
    a = np.uint64(1664525)
    c = np.uint64(1013904223)
    
    # Generate full period
    x = np.zeros(m, dtype=np.uint64)
    val = np.uint64(0)
    for i in range(m):
        x[i] = val
        val = (val * a + c) % np.uint64(m)
        
    # Mask elements < n (cycle walking)
    mask = x < np.uint64(n)
    return x[mask].astype(np.int64)


def apply_assignment(
    data: np.ndarray,
    shape: Sequence[int],
    strategy: Literal["sequential", "shuffled", "hash_indexed", "prime_stride", "lcg"] = "sequential",
    seed: int = 42,
) -> np.ndarray:
    """Map a 1D array of values to a tensor of target shape using a strategy.

    Args:
        data: 1D array of values (e.g. from CE digit stream).
        shape: Target tensor shape.
        strategy:
            - 'sequential': Fill row by row (PyTorch default).
            - 'shuffled': Decorrelate mapping via deterministic shuffle.
            - 'hash_indexed': Map index i to data[hash(i) % len(data)].
            - 'prime_stride': Bijective permutation using coprime prime stride.
            - 'lcg': Bijective permutation using LCG and cycle walking.
        seed: Random seed for 'shuffled' strategy.

    Returns:
        Resulting array reshaped to 'shape'.
    """
    n = 1
    for s in shape:
        n *= s

    if len(data) < n:
        raise ValueError(f"Data length ({len(data)}) is less than required ({n})")

    # Take only what we need
    data = data[:n]

    if strategy == "sequential":
        return data.reshape(shape)

    elif strategy == "shuffled":
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(n)
        return data[shuffled_indices].reshape(shape)

    elif strategy == "hash_indexed":
        # Vectorized bit-mixing hash using NumPy for speed
        indices = np.arange(n, dtype=np.uint64)
        h = indices
        h ^= h >> 16
        h *= 0x85ebca6b
        h ^= h >> 13
        h *= 0xc2b2ae35
        h ^= h >> 16
        hashed_indices = (h % np.uint64(n)).astype(np.int64)
        return data[hashed_indices].reshape(shape)

    elif strategy == "prime_stride":
        if n <= 2:
            return data.reshape(shape)
        p = _find_prime_stride(n)
        indices = np.arange(n, dtype=np.int64)
        permuted_indices = (indices * p) % n
        return data[permuted_indices].reshape(shape)

    elif strategy == "lcg":
        if n <= 2:
            return data.reshape(shape)
        permuted_indices = _lcg_permutation(n)
        return data[permuted_indices].reshape(shape)

    else:
        raise ValueError(f"Unknown assignment strategy: {strategy!r}")


def apply_orthogonal(
    tensor: torch.Tensor,
    gain: float = 1.0,
) -> torch.Tensor:
    """Transform a weight tensor to be semi-orthogonal using QR decomposition.

    This preserves the 'deterministic seed' property of CE while forcing
    a controlled singular value spectrum.

    Args:
        tensor: The input tensor (filled with CE/Sobol values).
        gain: Scaling factor (e.g. math.sqrt(2.0) for ReLU).

    Returns:
        Semi-orthogonalized tensor of the same shape.
    """
    if tensor.ndim < 2:
        return tensor  # Orthogonalization not defined for 1D

    # Standard PyTorch nn.init.orthogonal_ strategy:
    # 1. Reshape to (out_channels, -1)
    flattened = tensor.view(tensor.shape[0], -1)
    
    # We want orthogonal columns if rows < cols, or orthogonal rows if cols < rows.
    q, r = torch.linalg.qr(flattened.T if flattened.shape[0] < flattened.shape[1] else flattened)
    
    # Q is orthogonal.
    if flattened.shape[0] < flattened.shape[1]:
        q = q.T
    
    res = q * gain
    return res.view(tensor.shape)


def compute_effective_rank(tensor: torch.Tensor) -> float:
    """Compute the effective rank of a matrix using singular value entropy.

    r_eff = exp(-sum(p_i * ln(p_i))), where p_i = s_i / sum(s_j).
    """
    if tensor.ndim < 2:
        return 1.0
    w = tensor.detach().view(tensor.shape[0], -1)
    try:
        s = torch.linalg.svdvals(w)
        s_sum = s.sum()
        if s_sum < 1e-10:
            return 1.0
        p = s / s_sum
        p = p[p > 0]
        entropy = -torch.sum(p * torch.log(p))
        return float(torch.exp(entropy).item())
    except Exception:
        return 1.0
