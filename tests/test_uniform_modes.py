"""Tests for CE-U and Sobol-U uniform initialization modes.

These tests verify the new uniform modes added for the CE-U vs CE-N
ablation study (Phase 0, T004).
"""

from __future__ import annotations

import math

import pytest
import torch

from copeland_erdos_nets.ce_init import (
    _standardize,
    ce_init_,
    ce_normal_init,
    ce_uniform_init,
)
from copeland_erdos_nets.sobol_init import (
    _next_power_of_2,
    _sobol_points,
    sobol_init_,
    sobol_normal_init,
    sobol_uniform_init,
)


# ============================================================================
# CE-U tests
# ============================================================================


class TestCEUniformInit:
    """Tests for ce_uniform_init (CE-U)."""

    def test_shape(self):
        w = ce_uniform_init((32, 16))
        assert w.shape == (32, 16)

    def test_dtype_float32(self):
        w = ce_uniform_init((32, 16), dtype=torch.float32)
        assert w.dtype == torch.float32

    def test_dtype_float64(self):
        w = ce_uniform_init((32, 16), dtype=torch.float64)
        assert w.dtype == torch.float64

    def test_zero_mean(self):
        """After standardization, mean should be very close to zero."""
        w = ce_uniform_init((256, 128), m=4)
        assert abs(w.mean().item()) < 0.01

    def test_std_matches_target(self):
        """Standard deviation should match He target."""
        shape = (256, 128)
        w = ce_uniform_init(shape, m=4, kind="he")
        from copeland_erdos_nets.ce_init import _target_std

        expected_std = _target_std(shape, kind="he")
        actual_std = w.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.05

    def test_deterministic(self):
        """Same args produce identical tensors."""
        w1 = ce_uniform_init((64, 32), m=4, offset_blocks=0)
        w2 = ce_uniform_init((64, 32), m=4, offset_blocks=0)
        assert torch.equal(w1, w2)

    def test_different_offsets_differ(self):
        """Different offsets produce different weights."""
        w1 = ce_uniform_init((64, 32), m=4, offset_blocks=0)
        w2 = ce_uniform_init((64, 32), m=4, offset_blocks=100)
        assert not torch.equal(w1, w2)

    def test_differs_from_ce_n(self):
        """CE-U and CE-N produce different values (different transforms)."""
        w_u = ce_uniform_init((64, 32), m=4, offset_blocks=0)
        w_n = ce_normal_init((64, 32), m=4, offset_blocks=0)
        # Same source blocks but different transforms
        assert not torch.equal(w_u, w_n)

    def test_no_nan(self):
        w = ce_uniform_init((64, 32), m=8)
        assert not torch.isnan(w).any()

    def test_m10(self):
        """m=10 works for higher granularity."""
        w = ce_uniform_init((16, 8), m=10)
        assert w.shape == (16, 8)
        assert not torch.isnan(w).any()


class TestCEInitMode:
    """Tests for ce_init_ mode parameter."""

    def test_normal_mode(self):
        t = torch.zeros(32, 16)
        ce_init_(t, m=4, mode="normal")
        ref = ce_normal_init((32, 16), m=4)
        assert torch.allclose(t, ref)

    def test_uniform_mode(self):
        t = torch.zeros(32, 16)
        ce_init_(t, m=4, mode="uniform")
        ref = ce_uniform_init((32, 16), m=4)
        assert torch.allclose(t, ref)

    def test_modes_differ(self):
        t_n = torch.zeros(32, 16)
        t_u = torch.zeros(32, 16)
        ce_init_(t_n, m=4, mode="normal")
        ce_init_(t_u, m=4, mode="uniform")
        assert not torch.equal(t_n, t_u)


# ============================================================================
# Sobol audit tests
# ============================================================================


class TestSobolAudit:
    """Tests for Sobol power-of-2 compliance."""

    def test_next_power_of_2(self):
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(100) == 128
        assert _next_power_of_2(1024) == 1024
        assert _next_power_of_2(1025) == 2048

    def test_sobol_points_length(self):
        """_sobol_points returns exactly n points."""
        for n in [10, 100, 255, 256, 257, 1000]:
            pts = _sobol_points(n, scramble_seed=42)
            assert len(pts) == n

    def test_sobol_points_range(self):
        """All Sobol points should be in (0, 1)."""
        pts = _sobol_points(1000, scramble_seed=42)
        assert pts.min() > 0
        assert pts.max() < 1

    def test_sobol_points_deterministic(self):
        """Same seed produces same points."""
        p1 = _sobol_points(100, scramble_seed=42)
        p2 = _sobol_points(100, scramble_seed=42)
        assert (p1 == p2).all()

    def test_sobol_points_different_seeds(self):
        """Different seeds produce different points."""
        p1 = _sobol_points(100, scramble_seed=0)
        p2 = _sobol_points(100, scramble_seed=1)
        assert not (p1 == p2).all()


# ============================================================================
# Sobol-U tests
# ============================================================================


class TestSobolUniformInit:
    """Tests for sobol_uniform_init (Sobol-U)."""

    def test_shape(self):
        w = sobol_uniform_init((32, 16))
        assert w.shape == (32, 16)

    def test_zero_mean(self):
        w = sobol_uniform_init((1000, 1000))
        assert abs(w.mean().item()) < 0.1

    def test_deterministic(self):
        w1 = sobol_uniform_init((100, 100), scramble_seed=42)
        w2 = sobol_uniform_init((100, 100), scramble_seed=42)
        assert torch.equal(w1, w2)

    def test_different_seeds(self):
        w1 = sobol_uniform_init((100, 100), scramble_seed=0)
        w2 = sobol_uniform_init((100, 100), scramble_seed=1)
        assert not torch.equal(w1, w2)

    def test_differs_from_sobol_n(self):
        """Sobol-U and Sobol-N produce different values."""
        w_u = sobol_uniform_init((100, 100), scramble_seed=42)
        w_n = sobol_normal_init((100, 100), scramble_seed=42)
        assert not torch.equal(w_u, w_n)

    def test_no_nan(self):
        w = sobol_uniform_init((64, 32))
        assert not torch.isnan(w).any()


class TestSobolInitMode:
    """Tests for sobol_init_ mode parameter."""

    def test_normal_mode(self):
        t = torch.zeros(32, 16)
        sobol_init_(t, scramble_seed=42, mode="normal")
        ref = sobol_normal_init((32, 16), scramble_seed=42)
        assert torch.allclose(t, ref)

    def test_uniform_mode(self):
        t = torch.zeros(32, 16)
        sobol_init_(t, scramble_seed=42, mode="uniform")
        ref = sobol_uniform_init((32, 16), scramble_seed=42)
        assert torch.allclose(t, ref)

    def test_modes_differ(self):
        t_n = torch.zeros(32, 16)
        t_u = torch.zeros(32, 16)
        sobol_init_(t_n, scramble_seed=42, mode="normal")
        sobol_init_(t_u, scramble_seed=42, mode="uniform")
        assert not torch.equal(t_n, t_u)


# ============================================================================
# Standardize tests
# ============================================================================


class TestStandardize:
    """Tests for _standardize helper."""

    def test_zero_mean(self):
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _standardize(x)
        assert abs(result.mean()) < 1e-10

    def test_unit_std(self):
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _standardize(x)
        assert abs(result.std() - 1.0) < 1e-10

    def test_constant_input(self):
        """Constant input should produce zero-mean output."""
        import numpy as np

        x = np.array([5.0, 5.0, 5.0])
        result = _standardize(x)
        assert abs(result.mean()) < 1e-10
