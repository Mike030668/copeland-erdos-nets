"""Sobol weight initialization for PyTorch.

Provides two deterministic initialization modes:

- **Sobol-N** (Normal): Sobol sequence → U(0,1) → Φ⁻¹(u) → standardize → scale
- **Sobol-U** (Uniform): Sobol sequence → U(0,1) → 2u-1 → standardize → scale

These serve as control baselines to distinguish CE-specific effects
from general low-discrepancy sequence properties.

IMPORTANT — Sobol balance properties:
    SciPy warns that Sobol loses balance properties when the number of
    points is not a power of 2. This implementation pads to the next
    power of 2 and truncates, ensuring correct quasi-random coverage.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.stats import norm as scipy_norm
from scipy.stats import qmc

from .ce_init import _standardize, _target_std


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _sobol_points(n: int, scramble_seed: int = 0) -> np.ndarray:
    """Generate n Sobol points in [0, 1], respecting power-of-2 constraint.

    Generates ceil_pow2(n) points via random_base2, then truncates to n.
    This ensures Sobol balance properties are preserved.

    Args:
        n: Number of points needed.
        scramble_seed: Seed for Owen scrambling.

    Returns:
        Array of shape (n,) with values in (0, 1).
    """
    n_pow2 = _next_power_of_2(n)
    m_exp = int(math.log2(n_pow2))

    sobol = qmc.Sobol(d=1, scramble=True, seed=scramble_seed)
    # Use random_base2 for proper balance properties
    u = sobol.random_base2(m=m_exp).flatten()

    # Truncate to requested size
    return u[:n]


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
        1. Generate n = prod(shape) Sobol points (U(0,1)), power-of-2 safe
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

    # Step 1: Generate Sobol points (power-of-2 safe)
    u = _sobol_points(n, scramble_seed=scramble_seed)

    # Step 2: Inverse normal CDF
    u_clipped = np.clip(u, 1e-7, 1.0 - 1e-7)
    x = scipy_norm.ppf(u_clipped)

    # Step 3: Standardize (matched per-layer scaling)
    x = _standardize(x)

    # Step 4: Scale by target std
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    # Reshape and convert to torch
    return torch.tensor(x.reshape(shape), dtype=dtype)


def sobol_uniform_init(
    shape: Sequence[int],
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a weight tensor initialized via Sobol-U (uniform, no Phi-inverse).

    Sobol-U tests the raw low-discrepancy coverage effect without
    normal matching. Comparing Sobol-U vs Sobol-N isolates the
    contribution of the inverse normal CDF transform.

    Algorithm:
        1. Generate n = prod(shape) Sobol points (U(0,1)), power-of-2 safe
        2. Map to symmetric range: x = 2u - 1 (values in (-1, 1))
        3. Standardize: x = (x - mean) / std
        4. Scale by target σ (Xavier or He)

    Args:
        shape: Desired tensor shape.
        scramble_seed: Seed for Sobol scrambling.
        kind, gain, fan_mode: See sobol_normal_init.
        dtype: Output tensor dtype.

    Returns:
        Initialized weight tensor on CPU.
    """
    n = 1
    for s in shape:
        n *= s

    if n == 0:
        return torch.empty(shape, dtype=dtype)

    # Step 1: Generate Sobol points (power-of-2 safe)
    u = _sobol_points(n, scramble_seed=scramble_seed)

    # Step 2: Symmetric range (no inverse normal CDF)
    x = 2.0 * u - 1.0

    # Step 3: Standardize (matched per-layer scaling)
    x = _standardize(x)

    # Step 4: Scale by target std
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    return torch.tensor(x.reshape(shape), dtype=dtype)


def sobol_init_(
    tensor: torch.Tensor,
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    mode: Literal["normal", "uniform"] = "normal",
) -> torch.Tensor:
    """In-place Sobol initialization.

    Args:
        tensor: Weight tensor to initialize in-place.
        scramble_seed: Seed for Sobol scrambling.
        kind, gain, fan_mode: See sobol_normal_init.
        mode: 'normal' for Sobol-N (with Phi-inverse) or 'uniform' for Sobol-U.

    Returns:
        The input tensor (modified in-place).
    """
    init_fn = sobol_normal_init if mode == "normal" else sobol_uniform_init
    with torch.no_grad():
        new_data = init_fn(
            shape=tuple(tensor.shape),
            scramble_seed=scramble_seed,
            kind=kind,
            gain=gain,
            fan_mode=fan_mode,
            dtype=tensor.dtype,
        )
        tensor.copy_(new_data)
    return tensor
