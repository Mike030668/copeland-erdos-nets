import pytest
import torch
from copeland_erdos_nets.sobol_init import sobol_normal_init, sobol_uniform_init

def test_sobol_assignment():
    shape = (10, 10)
    t_seq = sobol_normal_init(shape, assignment="sequential", scramble_seed=0)
    t_shuf = sobol_normal_init(shape, assignment="shuffled", scramble_seed=0)
    assert not torch.equal(t_seq, t_shuf)

def test_sobol_matrix_shaped():
    # 2D case
    shape = (16, 32)
    t_matrix = sobol_normal_init(shape, matrix_shaped=True, scramble_seed=42)
    assert t_matrix.shape == shape
    
    # 1D case (should fallback to 1D Sobol)
    shape_1d = (64,)
    t_1d = sobol_normal_init(shape_1d, matrix_shaped=True)
    assert t_1d.shape == shape_1d

def test_sobol_uniform_matrix_shaped():
    shape = (8, 16)
    t = sobol_uniform_init(shape, matrix_shaped=True)
    assert t.shape == shape
    # check values roughly in range
    assert t.abs().max() < 5.0
