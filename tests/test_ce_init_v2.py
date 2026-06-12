import pytest
import torch
from copeland_erdos_nets.ce_init import ce_normal_init, ce_uniform_init

def test_ce_normal_assignment():
    shape = (10, 10)
    # Check that sequential vs shuffled results in different tensors
    t_seq = ce_normal_init(shape, assignment="sequential")
    t_shuf = ce_normal_init(shape, assignment="shuffled", offset_blocks=0)
    
    assert t_seq.shape == shape
    assert t_shuf.shape == shape
    assert not torch.equal(t_seq, t_shuf)

def test_ce_normal_orthogonalize():
    shape = (8, 8)
    t_ortho = ce_normal_init(shape, orthogonalize=True)
    
    # Check orthogonality
    prod = torch.mm(t_ortho.t(), t_ortho)
    # Gain is target sigma. For (8,8) He init, fan_in=8, gain=sqrt(2), sigma = sqrt(2)/sqrt(8) = 1/2 = 0.5
    # So prod should be eye * 0.25
    eye = torch.eye(8) * 0.25
    torch.testing.assert_close(prod, eye, atol=1e-5, rtol=1e-5)

def test_ce_uniform_assignment():
    shape = (5, 5)
    t_seq = ce_uniform_init(shape, assignment="sequential")
    t_hash = ce_uniform_init(shape, assignment="hash_indexed")
    assert not torch.equal(t_seq, t_hash)

def test_ce_init_determinism():
    shape = (4, 4)
    t1 = ce_normal_init(shape, assignment="shuffled", offset_blocks=10)
    t2 = ce_normal_init(shape, assignment="shuffled", offset_blocks=10)
    torch.testing.assert_close(t1, t2)
