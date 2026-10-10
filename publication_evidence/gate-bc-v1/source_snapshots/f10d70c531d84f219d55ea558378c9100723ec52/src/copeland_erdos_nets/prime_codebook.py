"""Prime-block codebook for weight quantization.

Provides deterministic codebook construction from m-digit primes and
projection utilities for fake quantization with straight-through estimator.
"""

from __future__ import annotations

import numpy as np
import torch


def _is_prime(n: int) -> bool:
    """Check primality via trial division (6k±1 optimization)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def list_m_digit_primes(m: int) -> list[int]:
    """Return all primes with exactly m decimal digits.

    For m=1: [2, 3, 5, 7]
    For m=2: [11, 13, 17, ..., 97] (21 primes)

    Args:
        m: Number of digits (m >= 1).

    Returns:
        Sorted list of primes p where 10^(m-1) <= p < 10^m.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    lower = 10 ** (m - 1)
    upper = 10 ** m

    primes: list[int] = []
    for n in range(lower, upper):
        if _is_prime(n):
            primes.append(n)

    return primes


def make_prime_block_codebook(m: int = 4, K: int = 16) -> np.ndarray:
    """Construct a symmetric prime-block codebook.

    Algorithm:
        1. Enumerate all m-digit primes P_m
        2. Quantile-subsample K//2 values from P_m
        3. Normalize to [0,1]: q = p / 10^m
        4. Standardize: (q - mean) / std
        5. Symmetrize: codebook = [-q_reversed, +q]
        6. Re-standardize final codebook to zero mean, unit variance
        7. Sort to ensure ascending order

    Args:
        m: Number of digits per prime block.
        K: Final codebook size (must be even).

    Returns:
        Float64 numpy array of shape (K,), sorted ascending.
    """
    if K % 2 != 0:
        raise ValueError(f"K must be even, got {K}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    # Step 1: Get all m-digit primes
    primes = list_m_digit_primes(m)

    if len(primes) < K // 2:
        raise ValueError(
            f"Not enough {m}-digit primes ({len(primes)}) for K={K} "
            f"(need at least {K // 2})"
        )

    # Step 2: Quantile-subsample K//2 values
    n_subsample = K // 2
    indices = np.linspace(0, len(primes) - 1, n_subsample, dtype=int)
    subsampled = np.array([primes[i] for i in indices], dtype=np.float64)

    # Step 3: Normalize to [0,1]
    q = subsampled / (10 ** m)

    # Step 4: Standardize
    q_mean = q.mean()
    q_std = q.std()
    if q_std > 1e-10:
        q = (q - q_mean) / q_std
    else:
        q = q - q_mean

    # Step 5: Symmetrize: [-q_reversed, +q]
    # This creates a symmetric codebook around 0
    neg_part = -q[::-1]
    codebook = np.concatenate([neg_part, q])

    # Step 6: Re-standardize to zero mean, unit variance
    cb_mean = codebook.mean()
    cb_std = codebook.std()
    if cb_std > 1e-10:
        codebook = (codebook - cb_mean) / cb_std
    else:
        codebook = codebook - cb_mean

    # Step 7: Sort to ensure ascending order
    codebook = np.sort(codebook)

    return codebook


def project_to_codebook_(
    w: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """In-place nearest-codeword projection.

    Algorithm:
        1. Normalize: z = w / (|scale| + eps)
        2. For each z: find nearest codeword idx = argmin_c |z - c|
        3. Reconstruct: w = scale * codebook[idx]

    Args:
        w: Weight tensor to project (modified in-place).
        codebook: 1D codebook tensor.
        scale: Per-element or broadcastable scale tensor.
        eps: Small constant for numerical stability.

    Returns:
        The input tensor (modified in-place).
    """
    with torch.no_grad():
        # Normalize
        z = w / (scale.abs() + eps)

        # Compute distances using broadcasting
        # z shape: [...], codebook shape: [K]
        # Expand z to [..., 1], codebook to [1, ..., K]
        z_exp = z.unsqueeze(-1)
        cb_exp = codebook.view((1,) * z.dim() + (-1,))

        # distances shape: [..., K]
        distances = (z_exp - cb_exp).pow(2)
        idx = distances.argmin(dim=-1)

        # Reconstruct: w = scale * codebook[idx]
        w.copy_(scale * codebook[idx])

    return w


class STECodebookFn(torch.autograd.Function):
    """Straight-Through Estimator for codebook quantization.

    Forward: quantize to nearest codebook entry.
    Backward: pass gradient through unchanged (optionally clipped).
    """

    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        codebook: torch.Tensor,
        scale: torch.Tensor,
        alpha_clip: float = 2.5,
    ) -> torch.Tensor:
        """Forward pass: quantize to nearest codeword."""
        eps = 1e-8

        # Normalize
        z = w / (scale.abs() + eps)

        # Find nearest codeword
        z_exp = z.unsqueeze(-1)
        cb_exp = codebook.view((1,) * z.dim() + (-1,))
        distances = (z_exp - cb_exp).pow(2)
        idx = distances.argmin(dim=-1)

        # Reconstruct
        w_quant = scale * codebook[idx]

        # Save for backward
        ctx.save_for_backward(w, w_quant, scale)
        ctx.alpha_clip = alpha_clip

        return w_quant

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Backward pass: straight-through gradient."""
        w, w_quant, scale = ctx.saved_tensors
        alpha_clip = ctx.alpha_clip

        # Straight-through: pass gradient through
        grad_w = grad_out.clone()

        # Optionally clip gradient
        if alpha_clip is not None and alpha_clip > 0:
            threshold = alpha_clip * scale.abs()
            grad_w = grad_w.clamp(-threshold, threshold)

        return grad_w, None, None, None


def codebook_soft_regularizer(
    w: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Smooth soft-min regularizer for codebook quantization.

    For each weight w_i:
        loss_i = -tau * log(sum_k exp(-||w_i/scale_i - c_k||^2 / (2*tau)))
    """
    eps = 1e-8

    # Normalize
    z = w / (scale.abs() + eps)

    # Expand for broadcasting
    z_exp = z.unsqueeze(-1)
    cb_exp = codebook.view((1,) * z.dim() + (-1,))

    # Compute squared distances to each codeword
    distances = (z_exp - cb_exp).pow(2)

    # Soft-min: -tau * log(sum(exp(-d / (2*tau))))
    exponent = -distances / (2 * tau)
    log_sum_exp = torch.logsumexp(exponent, dim=-1)
    loss = -tau * log_sum_exp

    # Mean over all elements
    return loss.mean()
