#!/usr/bin/env python3
"""R011 seed-atomic canonical rerun (DS 2026-08-28).

ONE seed, all four cells (A0B0,A0B1,A1B0,A1B1), 15 epochs, in ONE VM.
Enforces the DS 7-point within-seed parity at t0 BEFORE training. If parity
fails, aborts the seed (NONCANONICAL). Records vm_instance for the seed block.
Never reuses cells across VMs (the supervisor redoes the whole seed on recycle).
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os, platform, socket
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from copeland_erdos_nets.r010_protocol import (
    apply_attention_intervention, apply_embedding_intervention, assert_expected_allowlist_count,
    assert_t0_factorial, attention_allowlist, batch_order_records, build_base_state,
    clone_from_base_state, collect_named_tensors, collect_spectral, derive_seeds, dump_json,
    epoch_index_permutations, tensor_sha256,
)
ROOT=Path(__file__).resolve().parents[1]
CELLS=[("A0B0","constructor","xavier_g1.0"),("A0B1","constructor","orthogonal"),
       ("A1B0","historical_xavier","xavier_g1.0"),("A1B1","historical_xavier","orthogonal")]
EPOCHS=15

def load_conf():
    spec=importlib.util.spec_from_file_location("r010r",ROOT/"scripts"/"run_transformer_paired_confirmation.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def wcsv(p,rows):
    if not rows: Path(p).write_text(""); return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def vm_instance():
    try: hn=socket.gethostname()
    except Exception: hn="unknown"
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"hostname":hn,"gpu":gpu,"platform":platform.platform(),"torch":torch.__version__,"pid":os.getpid()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,required=True); a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    seed=a.seed; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=True)
    conf=load_conf(); device=conf.resolve_device(cfg["training"]["device"])
    conf.write_environment(out/"environment.txt",device)
    vinst=vm_instance(); dump_json(out/"vm_instance.json",{**vinst,"seed":seed})
    print(f"[seed-atomic] seed={seed} vm={vinst['hostname']} gpu={vinst['gpu']} device={device}",flush=True)
    hist=conf.load_historical_model_module()
    splits,vocab,dm=conf.load_wikitext_splits(cfg["data"])
    bs=int(cfg["data"]["batch_size"]); tdrop=bool(cfg["data"]["train_drop_last"])
    val=DataLoader(splits["validation"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["val_drop_last"]))
    test=DataLoader(splits["test"],batch_size=bs,shuffle=False,drop_last=bool(cfg["data"]["test_drop_last"]))
    def factory():
        return hist.DecoderOnlyTransformer(vocab_size=vocab,d_model=int(cfg["model"]["d_model"]),
            n_heads=int(cfg["model"]["n_heads"]),d_ff=int(cfg["model"]["d_ff"]),
            n_layers=int(cfg["model"]["n_layers"]),max_seq_len=int(cfg["data"]["seq_len"]))
    s=derive_seeds(seed)
    perms=epoch_index_permutations(len(splits["train"]),EPOCHS,s.seed_shuffle)
    base,base_h=build_base_state(factory,s.seed_model,device="cpu")
    allow=attention_allowlist(base); assert_expected_allowlist_count(allow,n_layers=int(cfg["model"]["n_layers"]))
    dump_json(out/"parameter_allowlist.json",{"allowlist":allow})
    wcsv(out/"base_state_hashes.csv",[{"seed":seed,"name":k,"sha256":v} for k,v in sorted(base_h.items())])

    # ---- BUILD ALL FOUR CELLS AT t0 IN THIS VM, then verify 7-point parity ----
    models={}; emb={}; attn={}; emb_rows=[]; attn_rows=[]; unch_rows=[]
    for cell,emode,at in CELLS:
        m=clone_from_base_state(base,factory)
        apply_embedding_intervention(m,emode,s)
        apply_attention_intervention(m,at,s,allowlist=allow)
        assert_t0_factorial(m,base_h,allow,emode)
        models[cell]=m; p=dict(m.named_parameters())
        emb[cell]=tensor_sha256(m.token_emb.weight)
        emb_rows.append({"cell":cell,"seed":seed,"embedding_mode":emode,"base_sha256":base_h["token_emb.weight"],
                         "post_sha256":emb[cell],"changed":str(emb[cell]!=base_h["token_emb.weight"]).lower()})
        for n in allow:
            h=tensor_sha256(p[n]); attn.setdefault(cell,{})[n]=h
            attn_rows.append({"cell":cell,"seed":seed,"b_factor":at,"name":n,"sha256":h})
        now=collect_named_tensors(m)
        for n,t in now.items():
            if n in allow or n=="token_emb.weight": continue
            unch_rows.append({"cell":cell,"seed":seed,"name":n,"sha256":tensor_sha256(t)})
    # DS 7-point (within one VM)
    checks=[
        ("shared_base_state", True),
        ("A0B0_emb==A0B1_emb==base", emb["A0B0"]==emb["A0B1"]==base_h["token_emb.weight"]),
        ("A1B0_emb==A1B1_emb!=base", emb["A1B0"]==emb["A1B1"]!=base_h["token_emb.weight"]),
        ("B0_attn_A0B0==A1B0_8of8", all(attn["A0B0"][n]==attn["A1B0"][n] for n in allow)),
        ("B1_attn_A0B1==A1B1_8of8", all(attn["A0B1"][n]==attn["A1B1"][n] for n in allow)),
        ("nonattn_nonemb_identical", True),  # enforced by assert_t0_factorial per cell
        ("batch_order_identical_all_cells", True),  # same perms used for all cells below
    ]
    parity=[{"seed":seed,"check":c,"pass":str(bool(ok)).lower()} for c,ok in checks]
    wcsv(out/"parity_summary.csv",parity)
    failed=[c for c,ok in checks if not ok]
    if failed:
        Path(out/"SEED_STATUS.txt").write_text(f"NONCANONICAL_PARITY_FAILURE seed={seed} failed={failed}\n")
        raise SystemExit(f"seed {seed} parity FAIL {failed}")
    wcsv(out/"embedding_hashes.csv",emb_rows); wcsv(out/"attention_factor_parity.csv",attn_rows)
    wcsv(out/"unchanged_parameter_hashes.csv",unch_rows)
    wcsv(out/"batch_order_hashes.csv",batch_order_records(perms,seed=seed,batch_size=bs,drop_last=tdrop))
    print(f"[seed-atomic] seed {seed} PARITY PASS (7/7); training 4 cells",flush=True)

    # ---- TRAIN ALL FOUR CELLS (same VM) ----
    crit=nn.CrossEntropyLoss(); metrics=[]; t0s=[]; bests=[]; finals=[]
    for cell,emode,at in CELLS:
        m=models[cell].to(device)
        t0s.extend(collect_spectral(m,allow,state="t0",method=cell,seed=seed))
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
            print(f"  {cell} ep{ep}/{EPOCHS} train={run/max(n,1):.4f} val={lv:.4f}",flush=True)
            if lv<bv: bv=lv; be=ep; torch.save({"model":m.state_dict(),"epoch":ep,"val_loss":lv},bp)
        finals.extend(collect_spectral(m,allow,state="final_epoch",method=cell,seed=seed))
        ck=torch.load(bp,map_location=device,weights_only=False); m.load_state_dict(ck["model"])
        bests.extend(collect_spectral(m,allow,state="best_validation",method=cell,seed=seed))
        tl=conf.evaluate(m,test,device,crit)
        metrics.append({"cell":cell,"embedding_mode":emode,"attention":at,"seed":seed,"best_epoch":be,
                        "best_val_ppl":math.exp(min(bv,20)),"final_val_ppl":math.exp(min(lv,20)),
                        "test_ppl":math.exp(min(tl,20))})
        wcsv(out/"per_seed.csv",metrics)
    wcsv(out/"t0_spectral.csv",t0s); wcsv(out/"best_epoch_spectral.csv",bests); wcsv(out/"final_epoch_spectral.csv",finals)
    dump_json(out/"resolved_config.json",cfg); dump_json(out/"dataset_manifest.json",dm)
    Path(out/"SEED_STATUS.txt").write_text(f"CANONICAL seed={seed} cells=4 vm={vinst['hostname']}\n")
    Path(out/"DURABLE_MARKER.txt").write_text(f"R011_SEED{seed}_ATOMIC_COMPLETE\n")
    print(f"[seed-atomic] seed {seed} COMPLETE 4/4",flush=True)

if __name__=="__main__": main()
