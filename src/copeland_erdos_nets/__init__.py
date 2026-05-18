"""Copeland-Erdős Nets: prime-number neural network initialization."""

__version__ = "0.1.0"

from .ce_init import ce_init_, ce_normal_init
from .ce_stream import blocks_to_uniform, ce_digit_stream, prime_generator, take_blocks

__all__ = [
    "blocks_to_uniform",
    "ce_digit_stream",
    "ce_init_",
    "ce_normal_init",
    "prime_generator",
    "take_blocks",
]
