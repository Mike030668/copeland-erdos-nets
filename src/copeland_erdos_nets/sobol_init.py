"""Sobol-N weight initialization for PyTorch.

Sobol-N uses the Sobol low-discrepancy sequence (instead of CE digits)
passed through the inverse normal CDF for deterministic weight initialization.

This serves as a control baseline to distinguish CE-specific effects
from general low-discrepancy sequence properties.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.stats import norm as scipy_norm
from scipy.stats import qmc

from .ce_init import _target_std


def sobol_normal_init(
    shape: Sequence[int],
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a weight tensor initialized via Sobol-N.

    Algorithm:
        1. Generate n = prod(shape) Sobol points (U(0,1))
        2. Apply inverse normal CDF: x = Φ⁻¹(u)
        3. Standardize: x = (x - mean) / std
        4. Scale by target σ (Xavier or He)

    Args:
        shape: Desired tensor shape.
        scramble_seed: Seed for Sobol scrambling (different seeds = different sequences).
        kind: 'xavier' (Glorot) or 'he' (Kaiming) scaling.
        gain: Activation gain.
        fan_mode: 'fan_in' or 'fan_out' for He init.
        dtype: Output tensor dtype.

    Returns:
        Initialized weight tensor on CPU.
    """
    n = 1
    for s in shape:
        n *= s

    if n == 0:
        return torch.empty(shape, dtype=dtype)

    # Step 1: Generate Sobol points (U(0,1))
    sobol = qmc.Sobol(d=1, scramble=True, seed=scramble_seed)
    u = sobol.random(n).flatten()

    # Step 2: Inverse normal CDF
    u_clipped = np.clip(u, 1e-7, 1.0 - 1e-7)
    x = scipy_norm.ppf(u_clipped)

    # Step 3: Standardize
    x_mean = x.mean()
    x_std = x.std()
    if x_std > 1e-10:
        x = (x - x_mean) / x_std
    else:
        x = x - x_mean

    # Step 4: Scale by target std
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    # Reshape and convert to torch
    return torch.tensor(x.reshape(shape), dtype=dtype)


def sobol_init_(
    tensor: torch.Tensor,
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
) -> torch.Tensor:
    """In-place Sobol-N initialization.

    Args:
        tensor: Weight tensor to initialize in-place.
        scramble_seed: Seed for Sobol scrambling.
        kind, gain, fan_mode: See sobol_normal_init.

    Returns:
        The input tensor (modified in-place).
    """
    with torch.no_grad():
        new_data = sobol_normal_init(
            shape=tuple(tensor.shape),
            scramble_seed=scramble_seed,
            kind=kind,
            gain=gain,
            fan_mode=fan_mode,
            dtype=tensor.dtype,
        )
        tensor.copy_(new_data)
    return tensor
