import pytest
import numpy as np
import torch
from copeland_erdos_nets.assignment import apply_assignment, apply_orthogonal

def test_sequential_assignment():
    data = np.arange(10)
    shape = (2, 5)
    res = apply_assignment(data, shape, strategy="sequential")
    assert res.shape == shape
    np.testing.assert_array_equal(res, data.reshape(shape))

def test_shuffled_assignment():
    data = np.arange(10)
    shape = (2, 5)
    res1 = apply_assignment(data, shape, strategy="shuffled", seed=42)
    res2 = apply_assignment(data, shape, strategy="shuffled", seed=42)
    res3 = apply_assignment(data, shape, strategy="shuffled", seed=43)
    
    assert res1.shape == shape
    np.testing.assert_array_equal(res1, res2)
    assert not np.array_equal(res1, res3)
    # Check that it contains the same elements
    assert set(res1.flatten()) == set(data)

def test_hash_indexed_assignment():
    data = np.arange(100)
    shape = (10, 10)
    res = apply_assignment(data, shape, strategy="hash_indexed")
    assert res.shape == shape
    # Hash indexed should be deterministic
    res2 = apply_assignment(data, shape, strategy="hash_indexed")
    np.testing.assert_array_equal(res, res2)
    # It should use more than just a few values (checking diversity)
    assert len(set(res.flatten())) > 50

def test_apply_orthogonal_square():
    # Create a 2x2 matrix
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ortho = apply_orthogonal(tensor, gain=1.0)
    
    # Check orthogonality: Q^T * Q = I
    prod = torch.mm(ortho.t(), ortho)
    eye = torch.eye(2)
    torch.testing.assert_close(prod, eye, atol=1e-5, rtol=1e-5)

def test_apply_orthogonal_rectangular():
    # 2x4 matrix (out=2, in=4)
    tensor = torch.randn(2, 4)
    ortho = apply_orthogonal(tensor, gain=1.0)
    assert ortho.shape == (2, 4)
    
    # Rows should be orthonormal: Q * Q^T = I
    prod = torch.mm(ortho, ortho.t())
    eye = torch.eye(2)
    torch.testing.assert_close(prod, eye, atol=1e-5, rtol=1e-5)

def test_apply_orthogonal_gain():
    tensor = torch.randn(3, 3)
    ortho = apply_orthogonal(tensor, gain=2.0)
    # Q^T * Q = gain^2 * I
    prod = torch.mm(ortho.t(), ortho)
    eye = torch.eye(3) * 4.0
    torch.testing.assert_close(prod, eye, atol=1e-5, rtol=1e-5)
