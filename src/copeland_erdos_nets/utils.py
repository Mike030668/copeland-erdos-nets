"""Shared utilities for copeland-erdos-nets."""

from __future__ import annotations

import math


def count_primes_approx(n: int) -> int:
    """Approximate number of primes ≤ n using π(n) ≈ n/ln(n).

    Args:
        n: Upper bound.

    Returns:
        Approximate prime count.
    """
    if n < 2:
        return 0
    return int(n / math.log(n))


def digits_from_primes_approx(num_digits: int) -> int:
    """Estimate how many primes are needed to produce `num_digits` CE digits.

    Each prime p contributes floor(log10(p))+1 digits.
    On average, primes near N contribute ~log10(N) digits.

    This is a rough upper bound for buffer sizing.

    Args:
        num_digits: Target number of CE digits.

    Returns:
        Estimated number of primes needed (conservative upper bound).
    """
    if num_digits <= 0:
        return 0
    # Average prime near p_k contributes ~log10(p_k) digits.
    # For safety, assume average of ~4 digits per prime (conservative for large primes)
    # and add 50% margin.
    return int(num_digits / 2) + 100
