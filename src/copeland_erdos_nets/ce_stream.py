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

import numpy as np


# Cache for primes to avoid redundant calculations across layers
_PRIME_CACHE: list[int] = []

def prime_generator(initial_limit: int = 10000000) -> Generator[int, None, None]:
    """Yield successive prime numbers using a segmented-ready sieve.
    Automatically extends the cache if needed.
    """
    global _PRIME_CACHE
    
    # Initial sieve if cache is empty
    if not _PRIME_CACHE:
        _extend_prime_cache(initial_limit)
    
    idx = 0
    while True:
        if idx < len(_PRIME_CACHE):
            yield _PRIME_CACHE[idx]
            idx += 1
        else:
            # Extend cache by doubling the current limit
            new_limit = len(_PRIME_CACHE) * 10 # Rough heuristic for range
            if new_limit < _PRIME_CACHE[-1] * 2:
                new_limit = _PRIME_CACHE[-1] * 2
            _extend_prime_cache(int(new_limit))
            # Continue from new elements


def _extend_prime_cache(limit: int):
    """Extend the global prime cache up to the specified limit."""
    global _PRIME_CACHE
    start = _PRIME_CACHE[-1] + 1 if _PRIME_CACHE else 0
    if start >= limit:
        return

    # Sieve of Eratosthenes
    size = limit
    sieve = np.ones(size, dtype=bool)
    sieve[0:2] = False
    # If we already have primes, we still need to mark their multiples
    # in the new range, but here we just re-sieve for simplicity 
    # as this is only called occasionally.
    for p in range(2, int(size**0.5) + 1):
        if sieve[p]:
            sieve[p*p : size : p] = False
    
    _PRIME_CACHE = np.where(sieve)[0].tolist()


def ce_digit_stream() -> Generator[int, None, None]:
    """Yield digits of the Copeland–Erdős constant.
    Uses math-based digit extraction (no string conversion).
    """
    for p in prime_generator():
        if p == 0: continue
        # Fast math-based digit extraction
        digits = []
        temp = p
        while temp:
            digits.append(temp % 10)
            temp //= 10
        for d in reversed(digits):
            yield d


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

    # Skip offset_blocks * m digits using islice for speed
    if offset_blocks > 0:
        stream = itertools.islice(stream, offset_blocks * m, None)

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
