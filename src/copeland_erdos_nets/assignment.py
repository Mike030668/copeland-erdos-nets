"""Weight assignment strategies for Copeland–Erdős and Sobol initializations.

This module provides different ways to map a stream of values (CE digits or Sobol
sequences) into weight tensors, allowing diagnostics for spatial correlation.
"""

from __future__ import annotations

from typing import Sequence, Literal
import numpy as np
import torch


def apply_assignment(
    data: np.ndarray,
    shape: Sequence[int],
    strategy: Literal["sequential", "shuffled", "hash_indexed"] = "sequential",
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
    # Actually, we follow torch.nn.init.orthogonal_:
    # "If rows > cols, the first cols columns are orthogonal."
    # We use QR.
    
    # Transpose if rows < cols to get more orthogonal vectors?
    # No, let's keep it simple and follow the standard.
    
    q, r = torch.linalg.qr(flattened.T if flattened.shape[0] < flattened.shape[1] else flattened)
    
    # Q is orthogonal.
    # If we transposed, Q is (cols, rows), so Q.T is (rows, cols).
    # Correct Q to be (rows, cols)
    if flattened.shape[0] < flattened.shape[1]:
        q = q.T
    
    # Standard trick to fix signs (make it unique)
    # d = torch.diag(r, 0).sign()
    # q *= d
    
    res = q * gain
    return res.view(tensor.shape)
