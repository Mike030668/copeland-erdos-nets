"""R012 crossed scale×redraw unit/static tests (DS binding gates)."""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import pytest, torch

from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    apply_attention_intervention, tensor_sha256, collect_named_tensors,
)
from copeland_erdos_nets.r012_protocol import (
    R012_CELLS, build_scale_redraw_vectors, cell_embedding, apply_r012_embedding,
    rms, embedding_scale_stats,
)
ROOT=Path(__file__).resolve().parents[1]
SCREEN=ROOT/"scripts"/"run_transformer_screening.py"
def _load():
    s=importlib.util.spec_from_file_location("scr",SCREEN); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
@pytest.fixture(scope="module")
def screening(): return _load()
def _tiny(sc):
    def f(): return sc.DecoderOnlyTransformer(vocab_size=64,d_model=32,n_heads=4,d_ff=64,n_layers=2,max_seq_len=16)
    return f

def _cells(sc,seed=1047):
    s=derive_seeds(seed); f=_tiny(sc)
    base,base_h=build_base_state(f,s.seed_model,device="cpu")
    allow=attention_allowlist(base)
    base_emb=base.token_emb.weight.detach().clone()
    out={}
    for c in R012_CELLS:
        m=clone_from_base_state(base,f)
        meta=apply_r012_embedding(m,c,s,base_emb)
        apply_attention_intervention(m,"xavier_g1.0",s,allowlist=allow)
        out[c]=(m,meta)
    return base,base_h,allow,base_emb,out,s

def test_s0d0_equals_base(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    assert tensor_sha256(out["S0D0"][0].token_emb.weight)==tensor_sha256(base_emb)

def test_s1d1_equals_single_fresh_xavier(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    s0,u0,s1,u1,fresh=build_scale_redraw_vectors(base_emb,s.seed_embedding)
    # S1D1 = s1*u1 = s1*(fresh/s1) = fresh
    assert tensor_sha256(out["S1D1"][0].token_emb.weight)==tensor_sha256(fresh)

def test_scale_parity_rms(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    r=lambda c: rms(out[c][0].token_emb.weight)
    assert abs(r("S0D0")-r("S0D1"))<1e-6   # S0 RMS identical across D
    assert abs(r("S1D0")-r("S1D1"))<1e-6   # S1 RMS identical across D

def test_direction_parity(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    import torch
    def unit(c):
        w=out[c][0].token_emb.weight.detach().double(); return w/w.pow(2).mean().sqrt()
    # D0 direction identical across S0/S1 (within fp tolerance)
    assert torch.allclose(unit("S0D0"),unit("S1D0"),atol=1e-6)
    # D1 direction identical across S0/S1
    assert torch.allclose(unit("S0D1"),unit("S1D1"),atol=1e-6)

def test_no_independent_redraw(screening):
    # u1 (D1 direction) is the SAME single fresh draw for S0D1 and S1D1
    base,base_h,allow,base_emb,out,s=_cells(screening)
    assert out["S0D1"][1]["u1_hash"]==out["S1D1"][1]["u1_hash"]
    assert out["S0D1"][1]["fresh_xavier_hash"]==out["S1D1"][1]["fresh_xavier_hash"]

def test_attention_identical_all_cells(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    ref={n:tensor_sha256(dict(out["S0D0"][0].named_parameters())[n]) for n in allow}
    for c in R012_CELLS:
        p=dict(out[c][0].named_parameters())
        for n in allow: assert tensor_sha256(p[n])==ref[n], (c,n)

def test_non_embedding_identical(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    for c in R012_CELLS:
        now=collect_named_tensors(out[c][0])
        for n,t in now.items():
            if n=="token_emb.weight" or n in allow: continue
            assert tensor_sha256(t)==base_h[n], (c,n)

def test_scale_stats_keys(screening):
    base,base_h,allow,base_emb,out,s=_cells(screening)
    st=embedding_scale_stats(out["S1D1"][0].token_emb.weight)
    assert set(st)=={"mean","std","rms","l2","min","max"}

def test_historical_screening_untouched():
    assert "get_wikitext2_dataloaders" in SCREEN.read_text()


def test_runtime_driver_mismatch_hard_stop(tmp_path, screening):
    import importlib.util
    spec=importlib.util.spec_from_file_location("r012run", ROOT/"scripts"/"run_r012_seed_atomic.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    freeze={"gpu":"Tesla T4","python":"9.9.9","torch":"0.0.0","cuda":"0.0",
            "numpy":"0.0","datasets":"0.0","transformers":"0.0","driver":"000.00.00"}
    import pytest as _p
    with _p.raises(SystemExit):
        m.assert_runtime(freeze, tmp_path)
    log=(tmp_path/"runtime_assertion.log").read_text()
    assert "driver" in log  # driver is compared as a frozen field


def test_rng_policy_matches_config():
    import json
    from copeland_erdos_nets import r010_protocol as r
    cfg=json.loads((ROOT/"configs"/"r012_smoke.json").read_text())["rng_policy"]
    assert cfg=={"seed_model_offset":r.SEED_MODEL_OFFSET,
                 "seed_shuffle_offset":r.SEED_SHUFFLE_OFFSET,
                 "seed_embedding_redraw_offset":r.SEED_EMBEDDING_OFFSET}
