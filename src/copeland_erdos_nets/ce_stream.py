"""Copeland–Erdős digit stream and block extraction.

The Copeland–Erdős constant is formed by concatenating the decimal
representations of successive primes: 0.2357111317192329...

This module provides:
- A prime number generator (segmented-sieve–ready simple sieve)
- An infinite digit stream from the Copeland–Erdős constant
- A block extractor that groups m consecutive digits into integers
- A converter from blocks to U(0,1) pseudo-uniform values
"""

from __future__ import annotations

import itertools
import math
from typing import Generator, Iterator


def prime_generator() -> Generator[int, None, None]:
    """Yield successive prime numbers using a simple sieve.

    Uses trial division with 6k±1 optimization.
    For the digit stream, we typically need primes up to ~10^6,
    so this is fast enough without a segmented sieve.
    """
    yield 2
    yield 3
    n = 5
    while True:
        if _is_prime(n):
            yield n
        n += 2
        if _is_prime(n):
            yield n
        n += 4


def _is_prime(n: int) -> bool:
    """Check primality via trial division (6k±1)."""
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


def ce_digit_stream() -> Generator[int, None, None]:
    """Yield digits of the Copeland–Erdős constant.

    Produces: 2, 3, 5, 7, 1, 1, 1, 3, 1, 7, 1, 9, 2, 3, 2, 9, ...
    (digits of 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...)
    """
    for p in prime_generator():
        for ch in str(p):
            yield int(ch)


def take_blocks(
    m: int,
    num_blocks: int,
    offset_blocks: int = 0,
) -> list[int]:
    """Extract integer blocks of m digits from the CE digit stream.

    Each block is an m-digit integer formed by concatenating m successive
    digits from the stream. Blocks are non-overlapping.

    Args:
        m: Number of digits per block (e.g. 4 → values in [0, 9999]).
        num_blocks: How many blocks to extract.
        offset_blocks: Number of blocks to skip before extraction.
            This provides deterministic layer-wise offsets.

    Returns:
        List of num_blocks integers, each in [0, 10^m - 1].
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
    if offset_blocks < 0:
        raise ValueError(f"offset_blocks must be >= 0, got {offset_blocks}")

    stream = ce_digit_stream()

    # Skip offset_blocks * m digits
    for _ in range(offset_blocks * m):
        next(stream)

    # Extract blocks
    blocks: list[int] = []
    for _ in range(num_blocks):
        z = 0
        for _ in range(m):
            z = 10 * z + next(stream)
        blocks.append(z)
    return blocks


def blocks_to_uniform(blocks: list[int], m: int) -> list[float]:
    """Convert integer blocks to pseudo-uniform values in (0, 1).

    Uses the midpoint rule: u = (z + 0.5) / 10^m
    This ensures u is strictly in (0, 1), never exactly 0 or 1.

    Args:
        blocks: Integer blocks from take_blocks().
        m: Number of digits per block (must match take_blocks).

    Returns:
        List of float values in (0, 1).
    """
    denom = 10 ** m
    return [(z + 0.5) / denom for z in blocks]
