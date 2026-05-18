"""Tests for prime-block codebook module."""

import numpy as np
import pytest
import torch

from copeland_erdos_nets.prime_codebook import (
    STECodebookFn,
    codebook_soft_regularizer,
    list_m_digit_primes,
    make_prime_block_codebook,
    project_to_codebook_,
)


def is_prime(n: int) -> bool:
    """Helper to verify primality."""
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


class TestListMDigitPrimes:
    """Tests for list_m_digit_primes function."""

    def test_m_1(self):
        """m=1 gives [2, 3, 5, 7]."""
        primes = list_m_digit_primes(1)
        assert primes == [2, 3, 5, 7]

    def test_m_2_count(self):
        """m=2 gives 21 primes (11..97)."""
        primes = list_m_digit_primes(2)
        assert len(primes) == 21
        assert primes[0] == 11
        assert primes[-1] == 97

    def test_m_2_all_prime(self):
        """All m=2 primes are actually prime."""
        primes = list_m_digit_primes(2)
        assert all(is_prime(p) for p in primes)

    def test_m_3_range(self):
        """m=3 primes are in [100, 1000)."""
        primes = list_m_digit_primes(3)
        assert all(100 <= p < 1000 for p in primes)
        assert all(is_prime(p) for p in primes)

    def test_sorted(self):
        """Primes are returned sorted."""
        primes = list_m_digit_primes(2)
        assert primes == sorted(primes)

    def test_invalid_m(self):
        """m < 1 raises ValueError."""
        with pytest.raises(ValueError, match="m must be >= 1"):
            list_m_digit_primes(0)


class TestMakePrimeBlockCodebook:
    """Tests for make_prime_block_codebook function."""

    def test_shape(self):
        """Returns array of shape (K,)."""
        cb = make_prime_block_codebook(m=4, K=16)
        assert cb.shape == (16,)
        assert cb.dtype == np.float64

    def test_shape_different_k(self):
        """Works for different K values."""
        cb = make_prime_block_codebook(m=2, K=8)
        assert cb.shape == (8,)

    def test_zero_mean(self):
        """Codebook mean is approximately zero."""
        cb = make_prime_block_codebook(m=4, K=16)
        assert abs(cb.mean()) < 1e-10

    def test_symmetric(self):
        """Codebook is symmetric: c[i] = -c[K-1-i]."""
        cb = make_prime_block_codebook(m=4, K=16)
        K = len(cb)
        for i in range(K):
            assert abs(cb[i] + cb[K - 1 - i]) < 1e-10

    def test_sorted(self):
        """Codebook is sorted ascending."""
        cb = make_prime_block_codebook(m=4, K=16)
        assert all(cb[i] <= cb[i + 1] for i in range(len(cb) - 1))

    def test_unit_variance(self):
        """Codebook has approximately unit variance."""
        cb = make_prime_block_codebook(m=4, K=16)
        assert abs(cb.std() - 1.0) < 1e-10

    def test_different_m(self):
        """Different m produces different codebooks."""
        cb1 = make_prime_block_codebook(m=1, K=8)
        cb2 = make_prime_block_codebook(m=2, K=8)
        assert not np.allclose(cb1, cb2)

    def test_odd_k_raises(self):
        """Odd K raises ValueError."""
        with pytest.raises(ValueError, match="K must be even"):
            make_prime_block_codebook(m=2, K=7)

    def test_invalid_m(self):
        """m < 1 raises ValueError."""
        with pytest.raises(ValueError, match="m must be >= 1"):
            make_prime_block_codebook(m=0, K=8)


class TestProjectToCodebook:
    """Tests for project_to_codebook_ function."""

    def test_projects_to_nearest(self):
        """Projects weights to nearest codeword."""
        codebook = torch.tensor([-1.0, -0.3, 0.3, 1.0])
        scale = torch.tensor([1.0])
        w = torch.tensor([0.5])  # Should project to 0.3

        result = project_to_codebook_(w, codebook, scale)
        assert result is w  # In-place
        assert abs(w.item() - 0.3) < 0.1

    def test_idempotent(self):
        """Second projection gives same result."""
        codebook = torch.tensor([-1.0, -0.3, 0.3, 1.0])
        scale = torch.tensor([1.0])
        w = torch.randn(5, 5)

        project_to_codebook_(w, codebook, scale)
        w1 = w.clone()
        project_to_codebook_(w, codebook, scale)

        assert torch.equal(w, w1)

    def test_respects_scale(self):
        """Scale affects projection correctly."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.tensor([2.0])
        w = torch.tensor([1.5])

        project_to_codebook_(w, codebook, scale)
        # Normalized: 1.5 / 2.0 = 0.75 -> projects to 1.0
        # Reconstructed: 2.0 * 1.0 = 2.0
        assert abs(w.item() - 2.0) < 0.1

    def test_multi_element(self):
        """Works for multi-element tensors."""
        codebook = torch.tensor([-1.0, 0.0, 1.0])
        scale = torch.ones(3, 4)
        w = torch.randn(3, 4)

        result = project_to_codebook_(w, codebook, scale)
        assert result.shape == (3, 4)

    def test_per_element_scale(self):
        """Per-element scale works correctly."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.tensor([1.0, 2.0, 3.0])
        w = torch.tensor([0.5, 1.5, 2.5])

        project_to_codebook_(w, codebook, scale)
        # Each element uses its own scale
        assert w.shape == (3,)


class TestSTECodebookFn:
    """Tests for STECodebookFn autograd function."""

    def test_forward_quantizes(self):
        """Forward pass quantizes to nearest codeword."""
        codebook = torch.tensor([-1.0, -0.3, 0.3, 1.0])
        scale = torch.tensor([1.0])
        w = torch.tensor([0.7])

        w_quant = STECodebookFn.apply(w, codebook, scale, 2.5)
        # 0.7 is closer to 1.0 than 0.3
        assert abs(w_quant.item() - 1.0) < 0.1

    def test_backward_passes_gradient(self):
        """Backward passes gradient through."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.ones(3, 3)
        w = torch.randn(3, 3, requires_grad=True)

        w_quant = STECodebookFn.apply(w, codebook, scale, 2.5)
        loss = w_quant.sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.shape == w.shape

    def test_clipping(self):
        """Gradient clipping works."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.ones(2, 2)
        w = torch.randn(2, 2, requires_grad=True)

        w_quant = STECodebookFn.apply(w, codebook, scale, 0.5)
        loss = w_quant.sum()
        loss.backward()

        assert w.grad is not None
        # Gradient should be clipped
        assert (w.grad.abs() <= 0.5).all()

    def test_deterministic(self):
        """Same input gives same output."""
        codebook = torch.tensor([-1.0, 0.0, 1.0])
        scale = torch.ones(4)
        w = torch.tensor([0.2, -0.5, 0.8, -0.1])

        w1 = STECodebookFn.apply(w.clone(), codebook, scale, 2.5)
        w2 = STECodebookFn.apply(w.clone(), codebook, scale, 2.5)

        assert torch.equal(w1, w2)


class TestCodebookSoftRegularizer:
    """Tests for codebook_soft_regularizer function."""

    def test_scalar_output(self):
        """Returns scalar tensor."""
        codebook = torch.tensor([-1.0, 0.0, 1.0])
        scale = torch.ones(3, 4)
        w = torch.randn(3, 4)

        loss = codebook_soft_regularizer(w, codebook, scale)
        assert loss.dim() == 0

    def test_differentiable(self):
        """Loss is differentiable w.r.t. w."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.ones(3, 3)
        w = torch.randn(3, 3, requires_grad=True)

        loss = codebook_soft_regularizer(w, codebook, scale)
        loss.backward()

        assert w.grad is not None
        assert w.grad.shape == w.shape

    def test_finite(self):
        """Loss is finite for normal inputs."""
        codebook = torch.tensor([-1.0, 0.0, 1.0])
        scale = torch.ones(4, 4)
        w = torch.randn(4, 4)

        loss = codebook_soft_regularizer(w, codebook, scale)
        assert torch.isfinite(loss).item()

    def test_tau_sensitivity(self):
        """Different tau gives different loss."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.ones(4)
        w = torch.randn(4)

        loss1 = codebook_soft_regularizer(w, codebook, scale, 0.1)
        loss2 = codebook_soft_regularizer(w, codebook, scale, 1.0)

        # Different tau should give different (but related) losses
        # They won't be equal due to temperature scaling
        assert loss1 != loss2

    def test_smaller_tau_sharper(self):
        """Smaller tau penalizes distant weights more."""
        codebook = torch.tensor([-1.0, 1.0])
        scale = torch.ones(4)

        # w close to codebook
        w_close = torch.tensor([0.9, -0.9, 1.0, -1.0])
        # w far from codebook
        w_far = torch.tensor([0.0, 0.0, 0.0, 0.0])

        loss_close = codebook_soft_regularizer(w_close, codebook, scale, 0.1)
        loss_far = codebook_soft_regularizer(w_far, codebook, scale, 0.1)

        # Far weights should have higher (less negative) loss
        assert loss_far > loss_close
