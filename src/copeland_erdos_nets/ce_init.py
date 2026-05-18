"""Copeland–Erdős Normal (CE-N) weight initialization for PyTorch.

Provides deterministic, seed-free initialization that replaces
Xavier/He random init with values derived from the Copeland–Erdős
digit stream passed through the inverse normal CDF (Φ⁻¹).

Key idea:
    CE digits → m-digit blocks → U(0,1) → Φ⁻¹(u) → standardize → scale by σ_Xavier/He
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.stats import norm as scipy_norm

from .ce_stream import blocks_to_uniform, take_blocks


def _infer_fans(shape: Sequence[int]) -> tuple[int, int]:
    """Infer fan_in and fan_out from weight tensor shape.

    Follows PyTorch conventions:
    - Linear (out, in): fan_in=in, fan_out=out
    - Conv2d (out, in, kH, kW): fan_in=in*kH*kW, fan_out=out*kH*kW
    - 1D tensor: fan_in=fan_out=size
    """
    ndim = len(shape)
    if ndim < 1:
        raise ValueError(f"Cannot infer fans from shape {shape}")
    if ndim == 1:
        return shape[0], shape[0]
    if ndim == 2:
        return shape[1], shape[0]
    # Conv: shape = (out_channels, in_channels, *kernel_size)
    receptive = 1
    for s in shape[2:]:
        receptive *= s
    return shape[1] * receptive, shape[0] * receptive


def _target_std(
    shape: Sequence[int],
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
) -> float:
    """Compute target standard deviation for Xavier or He init.

    Args:
        shape: Weight tensor shape.
        kind: 'xavier' (Glorot) or 'he' (Kaiming).
        gain: Activation gain. Default: 1.0 for xavier, sqrt(2) for he.
        fan_mode: 'fan_in' or 'fan_out' (only for He).

    Returns:
        Target standard deviation σ.
    """
    fan_in, fan_out = _infer_fans(shape)

    if kind == "xavier":
        g = 1.0 if gain is None else gain
        return math.sqrt(2.0 * g * g / (fan_in + fan_out))
    elif kind == "he":
        g = math.sqrt(2.0) if gain is None else gain
        fan = fan_in if fan_mode == "fan_in" else fan_out
        return g / math.sqrt(fan)
    else:
        raise ValueError(f"Unknown init kind: {kind!r}. Use 'xavier' or 'he'.")


def ce_normal_init(
    shape: Sequence[int],
    m: int = 4,
    offset_blocks: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a weight tensor initialized via Copeland–Erdős Normal (CE-N).

    Algorithm:
        1. Extract n = prod(shape) blocks of m digits from CE stream
        2. Convert to U(0,1): u = (z + 0.5) / 10^m
        3. Apply inverse normal CDF: x = Φ⁻¹(u)
        4. Standardize: x = (x - mean) / std
        5. Scale by target σ (Xavier or He)

    Args:
        shape: Desired tensor shape (e.g., (out_features, in_features)).
        m: Digits per block (higher = finer granularity). Default: 4.
        offset_blocks: Skip this many blocks for layer-wise offsets.
        kind: 'xavier' or 'he' scaling.
        gain: Activation gain override.
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

    # Step 1–2: CE blocks → uniform
    blocks = take_blocks(m=m, num_blocks=n, offset_blocks=offset_blocks)
    u = np.array(blocks_to_uniform(blocks, m), dtype=np.float64)

    # Step 3: inverse normal CDF
    # Clip to avoid ±inf at boundaries
    u_clipped = np.clip(u, 1e-7, 1.0 - 1e-7)
    x = scipy_norm.ppf(u_clipped)

    # Step 4: standardize
    x_mean = x.mean()
    x_std = x.std()
    if x_std > 1e-8:
        x = (x - x_mean) / x_std
    else:
        x = x - x_mean

    # Step 5: scale
    sigma = _target_std(shape, kind=kind, gain=gain, fan_mode=fan_mode)
    x = sigma * x

    # Reshape and convert to torch
    return torch.tensor(x.reshape(shape), dtype=dtype)


def ce_init_(
    tensor: torch.Tensor,
    m: int = 4,
    offset_blocks: int = 0,
    kind: Literal["xavier", "he"] = "he",
    gain: float | None = None,
    fan_mode: Literal["fan_in", "fan_out"] = "fan_in",
) -> torch.Tensor:
    """In-place CE-N initialization (follows PyTorch nn.init convention).

    Args:
        tensor: Weight tensor to initialize in-place.
        m, offset_blocks, kind, gain, fan_mode: See ce_normal_init.

    Returns:
        The input tensor (modified in-place).
    """
    with torch.no_grad():
        new_data = ce_normal_init(
            shape=tuple(tensor.shape),
            m=m,
            offset_blocks=offset_blocks,
            kind=kind,
            gain=gain,
            fan_mode=fan_mode,
            dtype=tensor.dtype,
        )
        tensor.copy_(new_data)
    return tensor
