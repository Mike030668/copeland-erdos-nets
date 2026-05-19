"""Copeland-Erdős Nets: prime-number neural network initialization."""

__version__ = "0.1.0"

from .ce_init import ce_init_, ce_normal_init
from .ce_stream import blocks_to_uniform, ce_digit_stream, prime_generator, take_blocks
from .prime_codebook import (
    STECodebookFn,
    codebook_soft_regularizer,
    list_m_digit_primes,
    make_prime_block_codebook,
    project_to_codebook_,
)
from .sobol_init import sobol_init_, sobol_normal_init

__all__ = [
    "STECodebookFn",
    "blocks_to_uniform",
    "ce_digit_stream",
    "ce_init_",
    "ce_normal_init",
    "codebook_soft_regularizer",
    "list_m_digit_primes",
    "make_prime_block_codebook",
    "prime_generator",
    "project_to_codebook_",
    "sobol_init_",
    "sobol_normal_init",
    "take_blocks",
]
