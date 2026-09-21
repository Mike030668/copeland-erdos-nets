"""Static proof of CE-P1's DS-specified construction invariants."""
from __future__ import annotations

import math
import torch

from copeland_erdos_nets.ce_p1_protocol import CONDITIONS, embedding_for_condition, multiset_equal
from copeland_erdos_nets.r013_protocol import rms, xavier_scalar_std


def _all(shape=(64, 32)):
    return {c: embedding_for_condition(c, shape, seed_embedding=30075, seed_shuffle=20073) for c in CONDITIONS}


def test_all_conditions_have_exact_xavier_target_rms():
    shape = (64, 32); target = xavier_scalar_std(*shape)
    for embedding, receipt in _all(shape).values():
        assert abs(rms(embedding) - target) < 1e-6
        assert abs(float(receipt["realized_rms"]) - target) < 1e-6


def test_ce_lcg_and_shuffled_share_one_value_multiset_but_not_placement():
    rows = _all()
    b, rb = rows["CE_LCG_Xavier_RMS"]
    c, rc = rows["CE_SHUFFLED_Xavier_RMS"]
    assert multiset_equal(b, c)
    assert rb["source_multiset_sha256"] == rc["source_multiset_sha256"]
    assert not torch.equal(b, c)


def test_gaussian_is_seed_bound_and_ce_is_deterministic_under_its_contract():
    a1, _ = embedding_for_condition("Gaussian_Xavier_RMS", (32, 16), seed_embedding=30075, seed_shuffle=20073)
    a2, _ = embedding_for_condition("Gaussian_Xavier_RMS", (32, 16), seed_embedding=30075, seed_shuffle=20073)
    a3, _ = embedding_for_condition("Gaussian_Xavier_RMS", (32, 16), seed_embedding=30076, seed_shuffle=20073)
    b1, _ = embedding_for_condition("CE_LCG_Xavier_RMS", (32, 16), seed_embedding=30075, seed_shuffle=20073)
    b2, _ = embedding_for_condition("CE_LCG_Xavier_RMS", (32, 16), seed_embedding=30076, seed_shuffle=20074)
    assert torch.equal(a1, a2) and not torch.equal(a1, a3)
    assert torch.equal(b1, b2)


def test_condition_set_is_closed_to_the_three_preregistered_cells():
    assert CONDITIONS == ("Gaussian_Xavier_RMS", "CE_LCG_Xavier_RMS", "CE_SHUFFLED_Xavier_RMS")
