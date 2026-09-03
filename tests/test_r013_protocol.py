"""R013 scale-response unit/static tests (DS binding)."""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import pytest, torch
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    apply_attention_intervention, tensor_sha256, collect_named_tensors,
)
from copeland_erdos_nets.r013_protocol import (
    DOSES, ladder_factors, xavier_scalar_std, rms, apply_embedding_dose, assert_no_weight_tying,
)
ROOT=Path(__file__).resolve().parents[1]; SCREEN=ROOT/"scripts"/"run_transformer_screening.py"
def _load():
    s=importlib.util.spec_from_file_location("scr",SCREEN); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
@pytest.fixture(scope="module")
def screening(): return _load()
def _tiny(sc):
    def f(): return sc.DecoderOnlyTransformer(vocab_size=64,d_model=32,n_heads=4,d_ff=64,n_layers=2,max_seq_len=16)
    return f

def _setup(sc,seed=1052):
    s=derive_seeds(seed); f=_tiny(sc)
    base,base_h=build_base_state(f,s.seed_model,device="cpu")
    allow=attention_allowlist(base); base_emb=base.token_emb.weight.detach().clone()
    out={}
    for d in DOSES:
        m=clone_from_base_state(base,f)
        meta=apply_embedding_dose(m,d,base_emb,64,32)
        apply_attention_intervention(m,"xavier_g1.0",s,allowlist=allow)
        out[d]=(m,meta)
    return base,base_h,allow,base_emb,out

def test_ladder_math():
    r=0.0063; f=ladder_factors(r)
    assert f["D_ctor"]==1.0 and f["D_xavier"]==r and f["D_below"]==r/2
    assert abs(f["D_mid1"]-r**(2/3))<1e-12 and abs(f["D_mid2"]-r**(1/3))<1e-12

def test_dctor_equals_base(screening):
    base,base_h,allow,base_emb,out=_setup(screening)
    assert tensor_sha256(out["D_ctor"][0].token_emb.weight)==tensor_sha256(base_emb)

def test_direction_fixed_across_doses(screening):
    base,base_h,allow,base_emb,out=_setup(screening)
    hs={out[d][1]["base_direction_hash"] for d in DOSES}
    assert len(hs)==1  # same constructor direction for all doses

def test_realized_rms_matches_target(screening):
    base,base_h,allow,base_emb,out=_setup(screening)
    for d in DOSES:
        meta=out[d][1]
        assert abs(meta["realized_rms"]-meta["target_rms"])<1e-6

def test_only_embedding_changes(screening):
    base,base_h,allow,base_emb,out=_setup(screening)
    for d in DOSES:
        now=collect_named_tensors(out[d][0])
        for n,t in now.items():
            if n=="token_emb.weight" or n in allow: continue
            assert tensor_sha256(t)==base_h[n], (d,n)

def test_attention_identical_across_doses(screening):
    base,base_h,allow,base_emb,out=_setup(screening)
    ref={n:tensor_sha256(dict(out["D_ctor"][0].named_parameters())[n]) for n in allow}
    for d in DOSES:
        p=dict(out[d][0].named_parameters())
        for n in allow: assert tensor_sha256(p[n])==ref[n]

def test_no_weight_tying(screening):
    base,_,_,_,out=_setup(screening)
    assert_no_weight_tying(out["D_ctor"][0])  # tiny model: lm_head separate -> no raise

def test_xavier_scalar():
    import math
    assert abs(xavier_scalar_std(50257,128)-math.sqrt(2/(50257+128)))<1e-15


def test_telemetry_rng_neutral(screening):
    # embedding_grad_stats reads grad only; must not consume RNG or mutate weights
    from copeland_erdos_nets.r013_protocol import apply_embedding_dose
    import importlib.util
    spec=importlib.util.spec_from_file_location("r013run", ROOT/"scripts"/"run_r013_seed_atomic.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    f=_tiny(screening)(); 
    # random state snapshot
    import torch as T
    f.token_emb.weight.grad=T.randn_like(f.token_emb.weight)
    wbefore=f.token_emb.weight.detach().clone()
    st0=torch.random.get_rng_state()      # snapshot RIGHT BEFORE telemetry
    gl2,grm=m.embedding_grad_stats(f)
    st1=torch.random.get_rng_state()
    assert T.equal(st0,st1)              # no RNG consumed
    assert T.equal(wbefore,f.token_emb.weight.detach())  # no mutation
    assert gl2>=0 and grm>=0
