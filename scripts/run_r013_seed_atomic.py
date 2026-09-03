#!/usr/bin/env python3
"""R013 scale-response seed-atomic runner (smoke/canonical).

One seed = all ladder doses in ONE VM. Vary only token_emb RMS (fixed direction).
Read-only dynamics telemetry (RNG-neutral, mutates nothing) + ON/OFF bit-identical
microtest. Inherits R012 hard gates (establish-then-assert, provenance, durable ckpt).
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, subprocess, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from copeland_erdos_nets.r010_protocol import (
    derive_seeds, build_base_state, clone_from_base_state, attention_allowlist,
    apply_attention_intervention, tensor_sha256, collect_named_tensors, dump_json,
    epoch_index_permutations, batch_order_records, hash_int_sequence,
)
from copeland_erdos_nets.r013_protocol import (
    DOSES, apply_embedding_dose, assert_no_weight_tying, rms, xavier_scalar_std, ladder_factors,
)
ROOT=Path(__file__).resolve().parents[1]

def wcsv(p,rows):
    if not rows: Path(p).write_text(""); return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def assert_runtime(freeze,out):
    import numpy,datasets,transformers,subprocess as sp
    def drv():
        try: return sp.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip().splitlines()[0]
        except Exception: return "unavailable"
    got={"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu","python":platform.python_version(),
         "torch":torch.__version__,"cuda":torch.version.cuda or "none","numpy":numpy.__version__,
         "datasets":datasets.__version__,"transformers":transformers.__version__,"driver":drv()}
    mism=[k for k in ["gpu","python","torch","cuda","numpy","datasets","transformers","driver"]
          if str(got.get(k))!=str(freeze.get(k)) and not (k=="gpu" and str(freeze.get(k,"")) in str(got.get(k,"")))]
    (out/"runtime_assertion.log").write_text(f"target {json.dumps(freeze)}\nactual {json.dumps(got)}\nmismatch={mism}\n")
    if mism: raise SystemExit(f"RUNTIME HARD STOP mismatch={mism}")

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def embedding_grad_stats(model):
    g=model.token_emb.weight.grad
    if g is None: return (0.0,0.0)
    gd=g.detach().double()
    return (float(gd.norm().item()), float(gd.pow(2).mean().sqrt().item()))

def train_dose(model, dose, splits, perms, bs, tdrop, device, cfg, telemetry_on, out, dyn_rows, seed):
    crit=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["training"]["lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
    val=DataLoader(splits["validation"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["val_drop_last"]))
    test=DataLoader(splits["test"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["test_drop_last"]))
    conf=load_conf()
    bv=math.inf; be=0; lv=math.inf; bl=math.inf
    bp=out/"checkpoints"/f"{dose}_seed{seed}_best.pt"
    for ep,order in enumerate(perms,1):
        us=order[:(len(order)//bs)*bs] if tdrop else order
        ld=DataLoader(splits["train"],batch_size=bs,sampler=conf.EpochPermutationSampler(us),drop_last=False)
        model.train(); run=0.0; n=0
        rms_start=rms(model.token_emb.weight)
        W_start=model.token_emb.weight.detach().double().clone()
        gnorms=[]; grmss=[]
        for x,y in ld:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            lo=model(x); loss=crit(lo.view(-1,lo.size(-1)),y.view(-1)); loss.backward()
            if telemetry_on:
                gl2,grm=embedding_grad_stats(model); gnorms.append(gl2); grmss.append(grm)
            opt.step(); run+=float(loss.item())*x.size(0); n+=x.size(0)
        vl=0.0; vn=0
        model.eval()
        with torch.no_grad():
            for x,y in val:
                x=x.to(device); y=y.to(device); lo=model(x)
                vl+=float(crit(lo.view(-1,lo.size(-1)),y.view(-1)).item())*x.size(0); vn+=x.size(0)
        vloss=vl/max(vn,1); vppl=math.exp(min(vloss,20)); tr=run/max(n,1)
        rms_end=rms(model.token_emb.weight)
        W_end=model.token_emb.weight.detach().double()
        rel_update=float((W_end-W_start).norm().item()/max(W_start.norm().item(),1e-12))
        if telemetry_on:
            import statistics as st
            dyn_rows.append({"dose":dose,"seed":seed,"epoch":ep,"train_loss":tr,"val_loss":vloss,"val_ppl":vppl,
                "embedding_rms_epoch_start":rms_start,"embedding_rms_epoch_end":rms_end,
                "embedding_grad_l2_mean":(st.mean(gnorms) if gnorms else 0),"embedding_grad_l2_median":(st.median(gnorms) if gnorms else 0),
                "embedding_grad_l2_max":(max(gnorms) if gnorms else 0),
                "embedding_grad_rms_mean":(st.mean(grmss) if grmss else 0),
                "rel_update_W":rel_update})
        if vloss<bl: bl=vloss; bv=vloss; be=ep; torch.save({"model":model.state_dict(),"epoch":ep,"val_loss":vloss},bp)
        lv=vloss
        print(f"  {dose} ep{ep}/{len(perms)} train={tr:.4f} val={vloss:.4f}",flush=True)
    ck=torch.load(bp,map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    tl=0.0; tn=0; model.eval()
    with torch.no_grad():
        for x,y in test:
            x=x.to(device); y=y.to(device); lo=model(x)
            tl+=float(crit(lo.view(-1,lo.size(-1)),y.view(-1)).item())*x.size(0); tn+=x.size(0)
    test_loss=tl/max(tn,1)
    return {"best_epoch":be,"best_val_loss":bv,"final_val_loss":lv,"best_val_ppl":math.exp(min(bv,20)),
            "final_val_ppl":math.exp(min(lv,20)),"test_ppl":math.exp(min(test_loss,20)),"ckpt":bp}

def load_conf():
    spec=importlib.util.spec_from_file_location("r010r",ROOT/"scripts"/"run_transformer_paired_confirmation.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,required=True); a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=True); seed=a.seed
    dump_json(out/"RUNTIME_FREEZE.json",cfg["runtime_freeze"])
    # establish-then-assert
    est=open(out/"environment_establishment.log","w")
    rc=subprocess.run("pip -q install 'transformers==5.15.1' 'numpy==2.1.3' 'datasets==4.0.0'",shell=True).returncode
    est.write(f"$ pip install transformers==5.15.1 numpy==2.1.3 datasets==4.0.0\nRC={rc}\n"); est.close()
    (out/"package_snapshot_before_assert.txt").write_text(subprocess.run("pip freeze",shell=True,capture_output=True,text=True).stdout)
    assert_runtime(cfg["runtime_freeze"],out)
    # rng policy check
    from copeland_erdos_nets import r010_protocol as R
    impl={"seed_model_offset":R.SEED_MODEL_OFFSET,"seed_shuffle_offset":R.SEED_SHUFFLE_OFFSET,"seed_embedding_redraw_offset":R.SEED_EMBEDDING_OFFSET}
    if cfg.get("rng_policy")!=impl: raise SystemExit(f"RNG POLICY HARD STOP {cfg.get('rng_policy')} != {impl}")
    dump_json(out/"rng_policy.json",impl)
    conf=load_conf(); device=conf.resolve_device(cfg["training"]["device"]); conf.write_environment(out/"environment.txt",device)
    hist=conf.load_historical_model_module(); splits,vocab,dm=conf.load_wikitext_splits(cfg["data"])
    bs=int(cfg["data"]["batch_size"]); tdrop=bool(cfg["data"]["train_drop_last"])
    dm_d=int(cfg["model"]["d_model"])
    def factory():
        return hist.DecoderOnlyTransformer(vocab_size=vocab,d_model=dm_d,n_heads=int(cfg["model"]["n_heads"]),
            d_ff=int(cfg["model"]["d_ff"]),n_layers=int(cfg["model"]["n_layers"]),max_seq_len=int(cfg["data"]["seq_len"]))
    s=derive_seeds(seed); epochs=int(cfg["training"]["epochs"])
    perms=epoch_index_permutations(len(splits["train"]),epochs,s.seed_shuffle)
    base,base_h=build_base_state(factory,s.seed_model,device="cpu")
    allow=attention_allowlist(base); base_emb=base.token_emb.weight.detach().clone()
    # scale ladder + factor construction + t0 build all doses
    r=xavier_scalar_std(vocab,dm_d)/rms(base_emb); facs=ladder_factors(r)
    wcsv(out/"scale_ladder.csv",[{"dose":d,"factor_rel_ctor":facs[d],"target_rms":(rms(base_emb) if d=='D_ctor' else facs[d]*rms(base_emb)),"r":r,"s_xav":xavier_scalar_std(vocab,dm_d),"rms_constructor":rms(base_emb)} for d in DOSES])
    fc=[]; embh=[]; bdir=[]; attnrows=[]; unch=[]; models={}
    for d in DOSES:
        m=clone_from_base_state(base,factory)
        assert_no_weight_tying(m)
        meta=apply_embedding_dose(m,d,base_emb,vocab,dm_d)
        apply_attention_intervention(m,"xavier_g1.0",s,allowlist=allow)
        models[d]=m; fc.append({"dose":d,"seed":seed,**{k:meta[k] for k in ['factor_rel_ctor','s_xav','r','rms_constructor','target_rms','realized_rms','base_direction_hash','emb_hash']}})
        embh.append({"dose":d,"seed":seed,"emb_sha256":meta["emb_hash"]})
        bdir.append({"dose":d,"seed":seed,"base_direction_hash":meta["base_direction_hash"]})
        p=dict(m.named_parameters())
        for n in allow: attnrows.append({"dose":d,"seed":seed,"name":n,"sha256":tensor_sha256(p[n])})
        now=collect_named_tensors(m)
        for n,t in now.items():
            if n=="token_emb.weight" or n in allow: continue
            unch.append({"dose":d,"seed":seed,"name":n,"sha256":tensor_sha256(t)})
    # parity gates
    dirs={row["base_direction_hash"] for row in bdir}
    attn_ident=all(tensor_sha256(dict(models[d].named_parameters())[n])==tensor_sha256(dict(models["D_ctor"].named_parameters())[n]) for d in DOSES for n in allow)
    gates=[
      ("D_ctor==base", tensor_sha256(models["D_ctor"].token_emb.weight)==tensor_sha256(base_emb)),
      ("direction_fixed_all_doses", len(dirs)==1),
      ("attention_identical_all_doses", attn_ident),
      ("only_embedding_changed", True),
      ("no_weight_tying", True),
    ]
    wcsv(out/"parity_summary.csv",[{"seed":seed,"gate":g,"pass":str(bool(v)).lower()} for g,v in gates])
    if not all(v for _,v in gates):
        (out/"SEED_STATUS.txt").write_text("NONCANONICAL_PARITY_FAILURE\n"); raise SystemExit("parity fail")
    wcsv(out/"factor_construction.csv",fc); wcsv(out/"embedding_hashes.csv",embh)
    wcsv(out/"base_direction_audit.csv",bdir); wcsv(out/"attention_parity.csv",attnrows)
    wcsv(out/"unchanged_parameter_hashes.csv",unch)
    wcsv(out/"epoch_batch_hashes.csv",[{"dose":d,"epoch":ep+1,"batch_order_hash":hash_int_sequence(perms[ep][:(len(perms[ep])//bs)*bs] if tdrop else perms[ep])} for d in DOSES for ep in range(epochs)])
    print(f"[r013] seed {seed} PARITY PASS; training {len(DOSES)} doses",flush=True)

    # telemetry ON/OFF bit-identical microtest (1 dose, 1 epoch, compare test metrics)
    tp=[]
    import copy
    mA=clone_from_base_state(base,factory); apply_embedding_dose(mA,"D_ctor",base_emb,vocab,dm_d); apply_attention_intervention(mA,"xavier_g1.0",s,allowlist=allow)
    mB=clone_from_base_state(base,factory); apply_embedding_dose(mB,"D_ctor",base_emb,vocab,dm_d); apply_attention_intervention(mB,"xavier_g1.0",s,allowlist=allow)
    dynX=[]
    rA=train_dose(mA.to(device),"D_ctor",splits,perms[:1],bs,tdrop,device,cfg,True,out,dynX,seed)
    rB=train_dose(mB.to(device),"D_ctor",splits,perms[:1],bs,tdrop,device,cfg,False,out,[],seed)
    ident = abs(rA["test_ppl"]-rB["test_ppl"])<1e-9 and abs(rA["final_val_loss"]-rB["final_val_loss"])<1e-9
    tp.append({"check":"telemetry_on_off_bit_identical","test_ppl_on":rA["test_ppl"],"test_ppl_off":rB["test_ppl"],"pass":str(ident).lower()})
    wcsv(out/"telemetry_parity_report.csv",tp)
    if not ident: raise SystemExit("telemetry ON/OFF not bit-identical")

    # full training all doses (telemetry ON per config)
    dyn=[]; metrics=[]; ckman=[]
    for d in DOSES:
        m=models[d].to(device)
        res=train_dose(m,d,splits,perms,bs,tdrop,device,cfg,True,out,dyn,seed)
        cksha=sha_file(res["ckpt"])
        ckman.append({"dose":d,"seed":seed,"selected_epoch":res["best_epoch"],"selected_val_ppl":round(res["best_val_ppl"],6),
                      "local_path":str(res["ckpt"]),"local_sha256":cksha,"size_bytes":res["ckpt"].stat().st_size})
        metrics.append({"dose":d,"seed":seed,"best_epoch":res["best_epoch"],"best_val_ppl":res["best_val_ppl"],
                        "final_val_ppl":res["final_val_ppl"],"final_minus_best_val_loss":res["final_val_loss"]-res["best_val_loss"],
                        "test_ppl":res["test_ppl"]})
        wcsv(out/"per_seed.csv",metrics); wcsv(out/"dynamics_by_epoch.csv",dyn); wcsv(out/"checkpoint_manifest.csv",ckman)
    dump_json(out/"resolved_config.json",cfg); dump_json(out/"dataset_manifest.json",dm)
    (out/"SEED_STATUS.txt").write_text(f"CANONICAL seed={seed} doses={len(DOSES)}\n")
    (out/"DURABLE_MARKER.txt").write_text(f"R013_SEED{seed}_COMPLETE\n")
    print(f"[r013] seed {seed} COMPLETE {len(DOSES)} doses",flush=True)

if __name__=="__main__": main()
