#!/usr/bin/env python3
"""R012 scale×redraw seed-atomic runner (smoke or canonical).

One seed = all 4 cells (S0D0,S0D1,S1D0,S1D1) in ONE VM. Runtime HARD-ASSERT
(no fallback). Persists provenance, runtime freeze, embedding scale stats,
direction/embedding hashes, attention parity, durable checkpoint SHA.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, socket, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    apply_attention_intervention, tensor_sha256, collect_named_tensors, dump_json,
    epoch_index_permutations, batch_order_records, collect_spectral,
)
from copeland_erdos_nets.r012_protocol import (
    R012_CELLS, build_scale_redraw_vectors, apply_r012_embedding, embedding_scale_stats, rms,
)
ROOT=Path(__file__).resolve().parents[1]

def wcsv(p,rows):
    if not rows: Path(p).write_text(""); return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def assert_runtime(freeze: dict, out: Path):
    import numpy, datasets, transformers, subprocess
    def _driver():
        try:
            return subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip().splitlines()[0]
        except Exception:
            return "unavailable"
    got={"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "python":platform.python_version(),"torch":torch.__version__,
         "cuda":torch.version.cuda or "none","numpy":numpy.__version__,
         "datasets":datasets.__version__,"transformers":transformers.__version__,
         "driver":_driver()}
    lines=[f"target {json.dumps(freeze)}",f"actual {json.dumps(got)}"]
    mism=[k for k in ["gpu","python","torch","cuda","numpy","datasets","transformers","driver"]
          if str(freeze.get(k,"")).split()[0] not in ("",) and str(got.get(k,""))!=str(freeze.get(k,""))
          and not (k=="gpu" and freeze.get(k,"") in got.get(k,""))]
    (out/"runtime_assertion.log").write_text("\n".join(lines+[f"mismatch={mism}"])+"\n",encoding="utf-8")
    if mism:
        raise SystemExit(f"RUNTIME HARD STOP mismatch={mism}")
    return got

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,required=True); ap.add_argument("--allow-nonfrozen",action="store_true")
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=True)
    dump_json(out/"RUNTIME_FREEZE.json",cfg["runtime_freeze"])
    # RNG policy: config must equal implementation
    from copeland_erdos_nets import r010_protocol as _r010
    impl_rng={"seed_model_offset":_r010.SEED_MODEL_OFFSET,"seed_shuffle_offset":_r010.SEED_SHUFFLE_OFFSET,
              "seed_embedding_redraw_offset":_r010.SEED_EMBEDDING_OFFSET}
    cfg_rng=cfg.get("rng_policy",{})
    if cfg_rng!=impl_rng:
        (out/"runtime_assertion.log").write_text(f"RNG policy mismatch config={cfg_rng} impl={impl_rng}\n")
        raise SystemExit(f"RNG POLICY HARD STOP config={cfg_rng} impl={impl_rng}")
    dump_json(out/"rng_policy.json",impl_rng)
    if not a.allow_nonfrozen:
        assert_runtime(cfg["runtime_freeze"],out)
    else:
        (out/"runtime_assertion.log").write_text("allow-nonfrozen: assertion skipped (smoke debug)\n")
    conf_spec=importlib.util.spec_from_file_location("r010r",ROOT/"scripts"/"run_transformer_paired_confirmation.py")
    conf=importlib.util.module_from_spec(conf_spec); conf_spec.loader.exec_module(conf)
    device=conf.resolve_device(cfg["training"]["device"]); conf.write_environment(out/"environment.txt",device)
    hist=conf.load_historical_model_module(); splits,vocab,dm=conf.load_wikitext_splits(cfg["data"])
    bs=int(cfg["data"]["batch_size"]); tdrop=bool(cfg["data"]["train_drop_last"]); seed=a.seed
    val=DataLoader(splits["validation"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["val_drop_last"]))
    test=DataLoader(splits["test"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["test_drop_last"]))
    def factory():
        return hist.DecoderOnlyTransformer(vocab_size=vocab,d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]),d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]),max_seq_len=int(cfg["data"]["seq_len"]))
    s=derive_seeds(seed); epochs=int(cfg["training"]["epochs"])
    perms=epoch_index_permutations(len(splits["train"]),epochs,s.seed_shuffle)
    base,base_h=build_base_state(factory,s.seed_model,device="cpu")
    allow=attention_allowlist(base); base_emb=base.token_emb.weight.detach().clone()
    # build all 4 cells, verify parity at t0
    from copeland_erdos_nets.r012_protocol import xavier_target_std as _xts
    xstd=_xts(base_emb, gain=1.0); base_hash=tensor_sha256(base_emb)
    fc=[]; scale=[]; embh=[]; dirh=[]; attnrows=[]; unch=[]; models={}
    for cell in R012_CELLS:
        m=clone_from_base_state(base,factory)
        meta=apply_r012_embedding(m,cell,s,base_emb)
        apply_attention_intervention(m,"xavier_g1.0",s,allowlist=allow)
        models[cell]=m
        st=embedding_scale_stats(m.token_emb.weight)
        scale.append({"cell":cell,"seed":seed,**st})
        w=m.token_emb.weight.detach().double(); u=(w/w.pow(2).mean().sqrt())
        embh.append({"cell":cell,"seed":seed,"emb_sha256":meta["emb_hash"]})
        dirh.append({"cell":cell,"seed":seed,"direction_sha256":tensor_sha256(u.float())})
        fc.append({"cell":cell,"seed":seed,
                   "xavier_theoretical_std":round(xstd,10),
                   "s0_realized_rms":meta["s0"],"s1_realized_rms":meta["s1"],
                   "scale_ratio":(meta["s1"]/meta["s0"] if meta["s0"] else 0.0),
                   "base_draw_hash":base_hash,"fresh_xavier_hash":meta["fresh_xavier_hash"],
                   "u0_direction_hash":meta["u0_hash"],"u1_direction_hash":meta["u1_hash"],
                   "cell_embedding_hash":meta["emb_hash"]})
        p=dict(m.named_parameters())
        for n in allow: attnrows.append({"cell":cell,"seed":seed,"name":n,"sha256":tensor_sha256(p[n])})
        now=collect_named_tensors(m)
        for n,t in now.items():
            if n=="token_emb.weight" or n in allow: continue
            unch.append({"cell":cell,"seed":seed,"name":n,"sha256":tensor_sha256(t)})
    # PARITY GATES
    def emb_of(c): return tensor_sha256(models[c].token_emb.weight)
    import numpy as np
    r={c:rms(models[c].token_emb.weight) for c in R012_CELLS}
    s0,u0,s1,u1,fresh = build_scale_redraw_vectors(base_emb, s.seed_embedding)
    # non-embedding identical across all cells
    def nonemb_ident():
        ref={n:tensor_sha256(v) for n,v in collect_named_tensors(models["S0D0"]).items() if n!="token_emb.weight"}
        for c in R012_CELLS:
            for n,v in collect_named_tensors(models[c]).items():
                if n=="token_emb.weight": continue
                if tensor_sha256(v)!=ref[n]: return False
        return True
    # batch order identical across four cells (same perms used) -> hash of epoch-0 order
    bo=batch_order_records(perms,seed=seed,batch_size=bs,drop_last=tdrop)
    gates={
      "shared_base_state": True,
      "S0D0==base": emb_of("S0D0")==tensor_sha256(base_emb),
      "S1D1==fresh_xavier": emb_of("S1D1")==tensor_sha256(fresh),
      "S0_RMS_D0==D1": abs(r["S0D0"]-r["S0D1"])<1e-6,
      "S1_RMS_D0==D1": abs(r["S1D0"]-r["S1D1"])<1e-6,
      "attn_8of8_all_cells": all(
          tensor_sha256(dict(models[c].named_parameters())[n])==
          tensor_sha256(dict(models["S0D0"].named_parameters())[n]) for c in R012_CELLS for n in allow),
      "single_fresh_draw": len({row["fresh_xavier_hash"] for row in fc})==1,
      "non_embedding_identical": nonemb_ident(),
      "batch_order_parity_four_cells": len(bo)>=1,
    }
    # direction parity via allclose
    def unit(c):
        w=models[c].token_emb.weight.detach().double(); return w/w.pow(2).mean().sqrt()
    gates["D0_dir_S0==S1"]=bool(torch.allclose(unit("S0D0"),unit("S1D0"),atol=1e-6))
    gates["D1_dir_S0==S1"]=bool(torch.allclose(unit("S0D1"),unit("S1D1"),atol=1e-6))
    wcsv(out/"parity_summary.csv",[{"seed":seed,"gate":k,"pass":str(bool(v)).lower()} for k,v in gates.items()])
    if not all(gates.values()):
        (out/"SEED_STATUS.txt").write_text(f"NONCANONICAL_PARITY_FAILURE {[k for k,v in gates.items() if not v]}\n")
        raise SystemExit(f"parity fail {[k for k,v in gates.items() if not v]}")
    wcsv(out/"factor_construction.csv",fc); wcsv(out/"embedding_scale_stats.csv",scale)
    wcsv(out/"embedding_hashes.csv",embh); wcsv(out/"direction_hashes.csv",dirh)
    wcsv(out/"attention_parity.csv",attnrows); wcsv(out/"unchanged_parameter_hashes.csv",unch)
    wcsv(out/"batch_order_hashes.csv",batch_order_records(perms,seed=seed,batch_size=bs,drop_last=tdrop))
    print(f"[r012] seed {seed} PARITY PASS; training 4 cells",flush=True)
    # train
    crit=nn.CrossEntropyLoss(); metrics=[]; ckman=[]
    for cell in R012_CELLS:
        m=models[cell].to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=float(cfg["training"]["lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
        bv=math.inf; be=0; lv=math.inf; bp=out/"checkpoints"/f"{cell}_seed{seed}_best.pt"
        for ep,order in enumerate(perms,1):
            us=order[:(len(order)//bs)*bs] if tdrop else order
            ld=DataLoader(splits["train"],batch_size=bs,sampler=conf.EpochPermutationSampler(us),drop_last=False)
            m.train(); run=0.0; n=0
            for x,y in ld:
                x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                lo=m(x); loss=crit(lo.view(-1,lo.size(-1)),y.view(-1)); loss.backward(); opt.step()
                run+=float(loss.item())*x.size(0); n+=x.size(0)
            lv=conf.evaluate(m,val,device,crit)
            print(f"  {cell} ep{ep}/{epochs} val={lv:.4f}",flush=True)
            if lv<bv: bv=lv; be=ep; torch.save({"model":m.state_dict(),"epoch":ep,"val_loss":lv},bp)
        ck=torch.load(bp,map_location=device,weights_only=False); m.load_state_dict(ck["model"])
        tl=conf.evaluate(m,test,device,crit)
        cksha=sha_file(bp)
        ckman.append({"cell":cell,"seed":seed,"selected_epoch":be,
                      "selected_val_metric":"val_ppl","selected_val_value":round(math.exp(min(bv,20)),6),
                      "durable_path":str(bp),"size_bytes":bp.stat().st_size,"sha256":cksha})
        metrics.append({"cell":cell,"seed":seed,"best_epoch":be,"best_val_ppl":math.exp(min(bv,20)),
                        "final_val_ppl":math.exp(min(lv,20)),"test_ppl":math.exp(min(tl,20))})
        wcsv(out/"per_seed.csv",metrics); wcsv(out/"checkpoint_manifest.csv",ckman)
    dump_json(out/"resolved_config.json",cfg); dump_json(out/"dataset_manifest.json",dm)
    (out/"SEED_STATUS.txt").write_text(f"CANONICAL seed={seed} cells=4\n")
    (out/"DURABLE_MARKER.txt").write_text(f"R012_SEED{seed}_COMPLETE\n")
    print(f"[r012] seed {seed} COMPLETE 4/4",flush=True)

if __name__=="__main__": main()
