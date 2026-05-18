"""Tests for Copeland-Erdos digit stream and block extraction."""

import pytest

from copeland_erdos_nets.ce_stream import (
    _is_prime,
    blocks_to_uniform,
    ce_digit_stream,
    prime_generator,
    take_blocks,
)


class TestIsPrime:
    def test_small_primes(self):
        assert all(_is_prime(p) for p in [2, 3, 5, 7, 11, 13])

    def test_composites(self):
        assert not any(_is_prime(n) for n in [0, 1, 4, 6, 8, 9, 10])

    def test_larger(self):
        assert _is_prime(7919)
        assert not _is_prime(7920)


class TestPrimeGenerator:
    def test_first_10(self):
        gen = prime_generator()
        first_10 = [next(gen) for _ in range(10)]
        assert first_10 == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_ascending(self):
        gen = prime_generator()
        primes = [next(gen) for _ in range(200)]
        for i in range(1, len(primes)):
            assert primes[i] > primes[i - 1]

    def test_beyond_1000(self):
        gen = prime_generator()
        primes = [next(gen) for _ in range(500)]
        assert primes[-1] > 1000


class TestCEDigitStream:
    def test_first_digits(self):
        stream = ce_digit_stream()
        first_10 = [next(stream) for _ in range(10)]
        assert first_10 == [2, 3, 5, 7, 1, 1, 1, 3, 1, 7]

    def test_all_in_range(self):
        stream = ce_digit_stream()
        digits = [next(stream) for _ in range(1000)]
        assert all(0 <= d <= 9 for d in digits)

    def test_deterministic(self):
        s1 = ce_digit_stream()
        s2 = ce_digit_stream()
        assert [next(s1) for _ in range(500)] == [next(s2) for _ in range(500)]


class TestTakeBlocks:
    def test_single_digit(self):
        assert take_blocks(m=1, num_blocks=5) == [2, 3, 5, 7, 1]

    def test_two_digit(self):
        assert take_blocks(m=2, num_blocks=2) == [23, 57]

    def test_four_digit(self):
        assert take_blocks(m=4, num_blocks=1) == [2357]

    def test_offset(self):
        no_off = take_blocks(m=2, num_blocks=3, offset_blocks=0)
        off_1 = take_blocks(m=2, num_blocks=2, offset_blocks=1)
        assert off_1 == no_off[1:]

    def test_zero_blocks(self):
        assert take_blocks(m=4, num_blocks=0) == []

    def test_deterministic(self):
        assert take_blocks(m=4, num_blocks=50) == take_blocks(m=4, num_blocks=50)

    def test_range(self):
        m = 4
        blocks = take_blocks(m=m, num_blocks=500)
        assert all(0 <= b < 10**m for b in blocks)

    def test_invalid_m(self):
        with pytest.raises(ValueError):
            take_blocks(m=0, num_blocks=1)

    def test_negative_num(self):
        with pytest.raises(ValueError):
            take_blocks(m=4, num_blocks=-1)

    def test_negative_offset(self):
        with pytest.raises(ValueError):
            take_blocks(m=4, num_blocks=1, offset_blocks=-1)


class TestBlocksToUniform:
    def test_values_in_01(self):
        blocks = take_blocks(m=4, num_blocks=100)
        u = blocks_to_uniform(blocks, m=4)
        assert all(0 < v < 1 for v in u)

    def test_midpoint_rule(self):
        # block=0, m=2 → u = (0+0.5)/100 = 0.005
        assert abs(blocks_to_uniform([0], m=2)[0] - 0.005) < 1e-10
        # block=99, m=2 → u = (99+0.5)/100 = 0.995
        assert abs(blocks_to_uniform([99], m=2)[0] - 0.995) < 1e-10

    def test_length(self):
        blocks = take_blocks(m=3, num_blocks=20)
        assert len(blocks_to_uniform(blocks, m=3)) == 20
