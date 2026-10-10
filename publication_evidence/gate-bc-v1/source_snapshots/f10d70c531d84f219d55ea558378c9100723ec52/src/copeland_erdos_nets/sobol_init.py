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

from .ce_init import _standardize, _target_std, _infer_fans
from .assignment import apply_assignment


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _sobol_points(n: int, d: int = 1, scramble_seed: int = 0) -> np.ndarray:
    """Generate n Sobol points in [0, 1]^d, respecting power-of-2 constraint.

    Args:
        n: Number of points (vectors) needed.
        d: Dimension of each point.
        scramble_seed: Seed for Owen scrambling.

    Returns:
        Array of shape (n, d) with values in (0, 1).
    """
    n_pow2 = _next_power_of_2(n)
    m_exp = int(math.log2(n_pow2))

    sobol = qmc.Sobol(d=d, scramble=True, seed=scramble_seed)
    # Use random_base2 for proper balance properties
    u = sobol.random_base2(m=m_exp)

    # Truncate to requested size
    return u[:n]


def sobol_normal_init(
    shape: Sequence[int],
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    dtype: torch.dtype = torch.float32,
    assignment: Literal["sequential", "shuffled"] = "sequential",
    matrix_shaped: bool = False,
) -> torch.Tensor:
    """Create a weight tensor initialized via Sobol-N.

    Algorithm:
        1. Generate n scalars (or m vectors if matrix_shaped) via Sobol
        2. Apply inverse normal CDF: x = Φ⁻¹(u)
        3. Apply assignment strategy (if not matrix_shaped)
        4. Standardize: x = (x - mean) / std
        5. Scale by target σ (Xavier or He)

    Args:
        shape: Desired tensor shape.
        scramble_seed: Seed for scrambling.
        kind, gain, fan_mode: Scaling parameters.
        dtype: Output tensor dtype.
        assignment: Strategy for 1D-to-ND mapping.
        matrix_shaped: If True, generate Sobol(d=fan_in) for each output neuron.
    """
    if matrix_shaped and len(shape) == 2:
        # Matrix shaped: generate shape[0] points of dimension shape[1]
        u = _sobol_points(shape[0], d=shape[1], scramble_seed=scramble_seed)
    else:
        n = 1
        for s in shape:
            n *= s
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        u = _sobol_points(n, d=1, scramble_seed=scramble_seed).flatten()

    # Apply Phi-inverse
    u_clipped = np.clip(u, 1e-7, 1.0 - 1e-7)
    x = scipy_norm.ppf(u_clipped)

    # Apply assignment for 1D case
    if not matrix_shaped:
        x = apply_assignment(x, shape, strategy=assignment, seed=scramble_seed + 123)
    else:
        # For matrix_shaped, we already have exact shape (shape[0], shape[1])
        x = x.reshape(shape)

    # Standardize
    x = _standardize(x)

    # Scale
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    return torch.tensor(x, dtype=dtype)


def sobol_uniform_init(
    shape: Sequence[int],
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    dtype: torch.dtype = torch.float32,
    assignment: Literal["sequential", "shuffled"] = "sequential",
    matrix_shaped: bool = False,
) -> torch.Tensor:
    """Create a weight tensor initialized via Sobol-U (uniform, no Phi-inverse)."""
    if matrix_shaped and len(shape) == 2:
        u = _sobol_points(shape[0], d=shape[1], scramble_seed=scramble_seed)
    else:
        n = 1
        for s in shape:
            n *= s
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        u = _sobol_points(n, d=1, scramble_seed=scramble_seed).flatten()

    # Symmetric range
    x = 2.0 * u - 1.0

    if not matrix_shaped:
        x = apply_assignment(x, shape, strategy=assignment, seed=scramble_seed + 123)
    else:
        x = x.reshape(shape)

    x = _standardize(x)
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    return torch.tensor(x, dtype=dtype)


def sobol_init_(
    tensor: torch.Tensor,
    scramble_seed: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    mode: Literal["normal", "uniform"] = "normal",
    assignment: Literal["sequential", "shuffled"] = "sequential",
    matrix_shaped: bool = False,
) -> torch.Tensor:
    """In-place Sobol initialization."""
    init_fn = sobol_normal_init if mode == "normal" else sobol_uniform_init
    with torch.no_grad():
        new_data = init_fn(
            shape=tuple(tensor.shape),
            scramble_seed=scramble_seed,
            kind=kind,
            gain=gain,
            fan_mode=fan_mode,
            dtype=tensor.dtype,
            assignment=assignment,
            matrix_shaped=matrix_shaped,
        )
        tensor.copy_(new_data)
    return tensor
