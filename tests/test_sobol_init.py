"""Tests for Sobol-N initialization."""

from __future__ import annotations

import pytest
import torch

from copeland_erdos_nets import sobol_init_, sobol_normal_init


class TestSobolNormalInit:
    """Tests for sobol_normal_init function."""

    def test_shape(self):
        """Output shape matches input shape."""
        shape = (10, 20)
        out = sobol_normal_init(shape)
        assert out.shape == shape

    def test_dtype(self):
        """Default dtype is float32."""
        out = sobol_normal_init((10, 10))
        assert out.dtype == torch.float32

    def test_different_dtype(self):
        """Can specify different dtype."""
        out = sobol_normal_init((10, 10), dtype=torch.float64)
        assert out.dtype == torch.float64

    def test_zero_mean(self):
        """Standardized Sobol has approximately zero mean."""
        out = sobol_normal_init((1000, 1000))
        assert abs(out.mean().item()) < 0.1

    def test_std_matches_target_he(self):
        """Std matches He target."""
        shape = (100, 100)  # fan_in = 100
        out = sobol_normal_init(shape, kind="he")
        expected_std = 1.0 / (100 ** 0.5)
        actual_std = out.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.5  # 50% tolerance

    def test_std_matches_target_xavier(self):
        """Std matches Xavier target."""
        shape = (100, 100)  # fan_in = 100, fan_out = 100
        out = sobol_normal_init(shape, kind="xavier")
        expected_std = 1.0 / ((100 + 100) ** 0.5)
        actual_std = out.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.5  # 50% tolerance

    def test_different_seeds_different_weights(self):
        """Different scramble seeds produce different weights."""
        out1 = sobol_normal_init((100, 100), scramble_seed=0)
        out2 = sobol_normal_init((100, 100), scramble_seed=1)
        assert not torch.equal(out1, out2)

    def test_same_seed_same_weights(self):
        """Same scramble seed produces same weights."""
        out1 = sobol_normal_init((100, 100), scramble_seed=42)
        out2 = sobol_normal_init((100, 100), scramble_seed=42)
        assert torch.equal(out1, out2)


class TestSobolInit:
    """Tests for in-place sobol_init_ function."""

    def test_inplace_modification(self):
        """sobol_init_ modifies tensor in-place."""
        tensor = torch.zeros(10, 10)
        result = sobol_init_(tensor, scramble_seed=0)
        assert result is tensor
        assert not torch.allclose(tensor, torch.zeros(10, 10))

    def test_different_seeds_different_weights(self):
        """Different scramble seeds produce different weights."""
        t1 = torch.zeros(100, 100)
        sobol_init_(t1, scramble_seed=0)
        t2 = torch.zeros(100, 100)
        sobol_init_(t2, scramble_seed=1)
        assert not torch.equal(t1, t2)

    def test_same_seed_same_weights(self):
        """Same scramble seed produces same weights."""
        t1 = torch.zeros(100, 100)
        sobol_init_(t1, scramble_seed=42)
        t2 = torch.zeros(100, 100)
        sobol_init_(t2, scramble_seed=42)
        assert torch.equal(t1, t2)

    def test_zero_mean(self):
        """Standardized Sobol has approximately zero mean."""
        t = torch.zeros(1000, 1000)
        sobol_init_(t)
        assert abs(t.mean().item()) < 0.1

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        t = torch.zeros(10, 20, 30)
        sobol_init_(t)
        assert t.shape == (10, 20, 30)


class TestSobolVsCE:
    """Comparison tests between Sobol and CE-N."""

    def test_both_zero_mean(self):
        """Both Sobol and CE-N have zero mean after standardization."""
        from copeland_erdos_nets import ce_init_

        sobol = sobol_normal_init((1000, 1000))
        ce = torch.zeros(1000, 1000)
        ce_init_(ce, m=4, kind="he", offset_blocks=0)

        assert abs(sobol.mean().item()) < 0.1
        assert abs(ce.mean().item()) < 0.1
