"""Tests for CE-N weight initialization."""

import math

import numpy as np
import pytest
import torch

from copeland_erdos_nets.ce_init import (
    _infer_fans,
    _target_std,
    ce_init_,
    ce_normal_init,
)


class TestInferFans:
    """Tests for fan_in/fan_out inference."""

    def test_linear_layer(self):
        fan_in, fan_out = _infer_fans((64, 32))
        assert fan_in == 32
        assert fan_out == 64

    def test_conv2d_layer(self):
        # (out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
        fan_in, fan_out = _infer_fans((16, 3, 3, 3))
        assert fan_in == 3 * 9  # 27
        assert fan_out == 16 * 9  # 144

    def test_bias_1d(self):
        fan_in, fan_out = _infer_fans((128,))
        assert fan_in == 128
        assert fan_out == 128

    def test_scalar_raises(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            _infer_fans(())


class TestTargetStd:
    """Tests for Xavier/He target std computation."""

    def test_he_fan_in(self):
        sigma = _target_std((64, 32), kind="he", fan_mode="fan_in")
        expected = math.sqrt(2.0) / math.sqrt(32)
        assert abs(sigma - expected) < 1e-10

    def test_xavier(self):
        sigma = _target_std((64, 32), kind="xavier")
        expected = math.sqrt(2.0 / (32 + 64))
        assert abs(sigma - expected) < 1e-10

    def test_custom_gain(self):
        sigma = _target_std((64, 32), kind="he", gain=1.0, fan_mode="fan_in")
        expected = 1.0 / math.sqrt(32)
        assert abs(sigma - expected) < 1e-10

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _target_std((64, 32), kind="kaiming")


class TestCENormalInit:
    """Tests for the CE-N initialization function."""

    def test_shape(self):
        w = ce_normal_init((32, 16))
        assert w.shape == (32, 16)

    def test_dtype_float32(self):
        w = ce_normal_init((32, 16), dtype=torch.float32)
        assert w.dtype == torch.float32

    def test_dtype_float64(self):
        w = ce_normal_init((32, 16), dtype=torch.float64)
        assert w.dtype == torch.float64

    def test_approximately_zero_mean(self):
        """After standardization, mean should be very close to zero."""
        w = ce_normal_init((256, 128), m=4)
        assert abs(w.mean().item()) < 0.01

    def test_std_matches_target(self):
        """Standard deviation should match He target."""
        shape = (256, 128)
        w = ce_normal_init(shape, m=4, kind="he")
        expected_std = _target_std(shape, kind="he")
        actual_std = w.std().item()
        # Allow 5% tolerance for finite-sample effects
        assert abs(actual_std - expected_std) / expected_std < 0.05

    def test_deterministic(self):
        """Same args produce identical tensors."""
        w1 = ce_normal_init((64, 32), m=4, offset_blocks=0)
        w2 = ce_normal_init((64, 32), m=4, offset_blocks=0)
        assert torch.equal(w1, w2)

    def test_different_offsets_differ(self):
        """Different offsets produce different weights."""
        w1 = ce_normal_init((64, 32), m=4, offset_blocks=0)
        w2 = ce_normal_init((64, 32), m=4, offset_blocks=100)
        assert not torch.equal(w1, w2)

    def test_xavier_vs_he_different_scale(self):
        """Xavier and He produce different scales."""
        shape = (64, 32)
        w_he = ce_normal_init(shape, kind="he")
        w_xavier = ce_normal_init(shape, kind="xavier")
        # They share the same underlying pattern but different scale
        assert abs(w_he.std().item() - w_xavier.std().item()) > 0.001

    def test_small_m(self):
        """m=1 still produces valid output (coarse granularity)."""
        w = ce_normal_init((16, 8), m=1)
        assert w.shape == (16, 8)
        assert not torch.isnan(w).any()

    def test_large_m(self):
        """m=6 produces finer granularity."""
        w = ce_normal_init((16, 8), m=6)
        assert w.shape == (16, 8)
        assert not torch.isnan(w).any()

    def test_conv2d_shape(self):
        """Works for conv2d weight shapes."""
        w = ce_normal_init((16, 3, 3, 3))
        assert w.shape == (16, 3, 3, 3)


class TestCEInitModule:
    """Tests for in-place initialization via ce_init_."""

    def test_linear_layer(self):
        layer = torch.nn.Linear(32, 64)
        result = ce_init_(layer.weight, m=4, kind="he")
        # Verify in-place modification
        assert result is layer.weight
        assert not torch.isnan(layer.weight).any()

    def test_multi_layer(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
        # Initialize each weight layer
        ce_init_(model[0].weight, m=4, kind="he")
        ce_init_(model[2].weight, m=4, kind="he")
        # Verify no NaN
        assert not torch.isnan(model[0].weight).any()
        assert not torch.isnan(model[2].weight).any()

    def test_deterministic(self):
        """Same init produces identical weights."""
        m1 = torch.nn.Linear(32, 64)
        m2 = torch.nn.Linear(32, 64)
        ce_init_(m1.weight, m=4)
        ce_init_(m2.weight, m=4)
        assert torch.equal(m1.weight, m2.weight)

    def test_conv2d_layer(self):
        layer = torch.nn.Conv2d(3, 16, kernel_size=3)
        result = ce_init_(layer.weight, m=4, kind="he")
        assert result is layer.weight
        assert not torch.isnan(layer.weight).any()
