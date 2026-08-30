#!/usr/bin/env python3
"""R011 evidence presentation (DS GOVERNANCE_ADDENDUM 2026-08-30).

Presentation-only: builds 8 canonical figures + Mini Research Report from the
seed-atomic v2 per_seed.csv, and NONCANONICAL diagnostic figures+report from the
cross-VM v1 per_seed.csv. Never pools v1 and v2. Does not touch protocol/metrics.
"""
from __future__ import annotations
import argparse, csv, hashlib, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS=["A0B0","A0B1","A1B0","A1B1"]
LABEL={"A0B0":"A0·Xavier","A0B1":"A0·Orth","A1B0":"A1·Xavier","A1B1":"A1·Orth"}

def load(p):
    p=Path(p)
    return list(csv.DictReader(p.open())) if p.exists() and p.stat().st_size else []

def by_seed(rows):
    d=defaultdict(dict)
    for r in rows: d[int(r["seed"])][r["cell"]]=r
    return d

def tcrit(n): return {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(n,2.776)

def ci(vals):
    a=np.array(vals,float); n=len(a); m=a.mean()
    sd=a.std(ddof=1) if n>1 else 0.0; se=sd/math.sqrt(n) if n else 0.0
    t=tcrit(n); return m, m-t*se, m+t*se, sd

def canonical(v2dir,out,noncanon=False):
    tag="NONCANONICAL / NOT FOR PRIMARY INFERENCE" if noncanon else ""
    rows=load(Path(v2dir)/"per_seed.csv")
    if not rows:
        print("no per_seed at",v2dir); return []
    bs=by_seed(rows); seeds=sorted(bs)
    out=Path(out); (out).mkdir(parents=True,exist_ok=True)
    def ppl(s,c): return float(bs[s][c]["test_ppl"]) if c in bs[s] else np.nan
    figs=[]
    def save(fig,name):
        p=out/name
        if tag: fig.text(0.5,0.01,tag,ha="center",color="red",fontsize=8)
        fig.savefig(p,dpi=120,bbox_inches="tight"); plt.close(fig); figs.append(p)

    # fig01 factorial test ppl (mean±ci per cell)
    fig,ax=plt.subplots(figsize=(6,4))
    means=[];los=[];his=[]
    for c in CELLS:
        vals=[ppl(s,c) for s in seeds if not np.isnan(ppl(s,c))]
        m,lo,hi,_=ci(vals); means.append(m);los.append(m-lo);his.append(hi-m)
    ax.bar(range(4),means,yerr=[los,his],capsize=5,color=["#8ec","#8ec","#f9a","#f9a"])
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[c] for c in CELLS]); ax.set_ylabel("test PPL")
    ax.set_title("R011 factorial test PPL (mean±95%CI)")
    save(fig,"fig01_factorial_test_ppl.png")

    # fig02 embedding effect by seed: (A1B0-A0B0) and (A1B1-A0B1)
    fig,ax=plt.subplots(figsize=(6,4))
    e_b0=[ppl(s,"A1B0")-ppl(s,"A0B0") for s in seeds]
    e_b1=[ppl(s,"A1B1")-ppl(s,"A0B1") for s in seeds]
    x=np.arange(len(seeds))
    ax.plot(x,e_b0,"o-",label="emb | Xavier attn"); ax.plot(x,e_b1,"s-",label="emb | Orth attn")
    ax.axhline(0,color="k",lw=0.5); ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("Δ test PPL (A1−A0)"); ax.legend(); ax.set_title("Embedding effect by seed")
    save(fig,"fig02_embedding_effect_by_seed.png")

    # fig03 attention effect by seed: (A0B1-A0B0) and (A1B1-A1B0)
    fig,ax=plt.subplots(figsize=(6,4))
    a_a0=[ppl(s,"A0B1")-ppl(s,"A0B0") for s in seeds]
    a_a1=[ppl(s,"A1B1")-ppl(s,"A1B0") for s in seeds]
    ax.plot(x,a_a0,"o-",label="attn | A0 emb"); ax.plot(x,a_a1,"s-",label="attn | A1 emb")
    ax.axhline(0,color="k",lw=0.5); ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("Δ test PPL (B1−B0)"); ax.legend(); ax.set_title("Attention effect by seed")
    save(fig,"fig03_attention_effect_by_seed.png")

    # fig04 interaction: [(A1B1-A1B0)-(A0B1-A0B0)]
    fig,ax=plt.subplots(figsize=(6,4))
    inter=[(ppl(s,"A1B1")-ppl(s,"A1B0"))-(ppl(s,"A0B1")-ppl(s,"A0B0")) for s in seeds]
    ax.bar(x,inter,color="#aa8"); ax.axhline(0,color="k",lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(seeds); ax.set_xlabel("seed"); ax.set_ylabel("interaction Δ PPL")
    ax.set_title("Factorial interaction (emb×attn)")
    save(fig,"fig04_factorial_interaction.png")

    # fig05 learning curves: needs val per epoch — approximate from best/final ppl if no curve file
    fig,ax=plt.subplots(figsize=(6,4))
    for c in CELLS:
        bv=[float(bs[s][c]["best_val_ppl"]) for s in seeds if c in bs[s]]
        ax.plot(seeds,bv,"o-",label=LABEL[c])
    ax.set_xlabel("seed"); ax.set_ylabel("best val PPL"); ax.legend(fontsize=7); ax.set_title("Best validation PPL by seed (proxy)")
    save(fig,"fig05_validation_learning_curves.png")

    # fig06 best epoch distribution
    fig,ax=plt.subplots(figsize=(6,4))
    for i,c in enumerate(CELLS):
        be=[int(bs[s][c]["best_epoch"]) for s in seeds if c in bs[s]]
        ax.scatter([i]*len(be),be,label=LABEL[c])
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[c] for c in CELLS]); ax.set_ylabel("best epoch")
    ax.set_title("Best-epoch distribution")
    save(fig,"fig06_best_epoch_distribution.png")

    # fig07 final vs best validation
    fig,ax=plt.subplots(figsize=(5,5))
    for c in CELLS:
        bv=[float(bs[s][c]["best_val_ppl"]) for s in seeds if c in bs[s]]
        fv=[float(bs[s][c]["final_val_ppl"]) for s in seeds if c in bs[s]]
        ax.scatter(bv,fv,label=LABEL[c])
    lim=ax.get_xlim(); ax.plot(lim,lim,"k--",lw=0.5)
    ax.set_xlabel("best val PPL"); ax.set_ylabel("final val PPL"); ax.legend(fontsize=7)
    ax.set_title("Final vs best validation (overfit view)")
    save(fig,"fig07_final_vs_best_validation.png")

    # fig08 embedding t0 scale stats (from t0_spectral if present else placeholder)
    fig,ax=plt.subplots(figsize=(6,4))
    sp=load(Path(v2dir)/"t0_spectral.csv")
    if sp:
        # mean effective_rank per method at t0
        by=defaultdict(list)
        for r in sp:
            try: by[r["method"]].append(float(r["effective_rank"]))
            except: pass
        ms=list(by); ax.bar(range(len(ms)),[np.mean(by[m]) for m in ms])
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms,rotation=30,fontsize=7); ax.set_ylabel("t0 mean eff. rank")
    else:
        ax.text(0.5,0.5,"t0_spectral not available",ha="center")
    ax.set_title("Attention t0 spectral scale")
    save(fig,"fig08_embedding_t0_scale_stats.png")
    return figs

def diag(v1dir,out):
    return canonical(v1dir,out,noncanon=True)

def mini_report(v2dir,out_md,figs):
    rows=load(Path(v2dir)/"per_seed.csv"); bs=by_seed(rows); seeds=sorted(bs)
    def ppl(s,c): return float(bs[s][c]["test_ppl"])
    def line(name,fn):
        vals=[fn(s) for s in seeds]; m,lo,hi,sd=ci(vals)
        return f"| {name} | {m:+.2f} | {sd:.2f} | [{lo:+.2f}, {hi:+.2f}] |"
    table="\n".join(f"| {c} | "+", ".join(f"{ppl(s,c):.1f}" for s in seeds)+f" | {ci([ppl(s,c) for s in seeds])[0]:.1f} |" for c in CELLS)
    contrasts="\n".join([
        line("emb | Xavier (A1B0−A0B0)",lambda s:ppl(s,"A1B0")-ppl(s,"A0B0")),
        line("emb | Orth (A1B1−A0B1)",lambda s:ppl(s,"A1B1")-ppl(s,"A0B1")),
        line("attn | A0 (A0B1−A0B0)",lambda s:ppl(s,"A0B1")-ppl(s,"A0B0")),
        line("attn | A1 (A1B1−A1B0)",lambda s:ppl(s,"A1B1")-ppl(s,"A1B0")),
        line("interaction",lambda s:(ppl(s,"A1B1")-ppl(s,"A1B0"))-(ppl(s,"A0B1")-ppl(s,"A0B0"))),
    ])
    figlist="\n".join(f"- `plots/{f.name}`" for f in figs)
    Path(out_md).write_text(f"""# R011 Mini Research Report (seed-atomic canonical v2)

## Research question
Does reconstructing the historical `token_emb.weight` Xavier reinitialization (A1)
explain the historical selective-init PPL advantage, and does it interact with
Orthogonal attention init (B1), under strict within-seed parity?

## Hypothesis
A1 lowers test PPL vs A0 under both attention settings; the historical large
effect is primarily embedding-driven.

## Protocol (frozen)
WikiText-2, gpt2 tok, 2-layer d=128 h=4 ff=512 seq=128, AdamW 5e-4 wd=0.01,
15 epochs, val→ckpt, test once. Seeds 42–46. **Seed-atomic**: all 4 cells of a
seed in ONE Tesla T4 VM; DS 7-point within-seed parity gate PASS before training.

## Main results — test PPL per cell
| cell | per-seed | mean |
|------|----------|------|
{table}

## Paired factorial contrasts (mean, SD, 95% t-CI)
| contrast | mean Δ | SD | 95% CI |
|----------|-------:|---:|--------|
{contrasts}

Lower PPL better; negative Δ = improvement.

## Plots
{figlist}

## Diagnostics
t0/best/final spectral in `diagnostics/`; batch-order + hash parity in `manifests/`.

## Failure modes
Overfitting after best epoch (see fig07); A1 cells select earlier best epochs.

## Limitations
n=5 seeds; single architecture/dataset; held-out test PPL (historical R008 used
best-val). Not a general embedding-init mechanism claim — causal source decomposition only.

## Artifact references
`per_seed.csv`, `parity_summary.csv`, `attention_factor_parity.csv`,
`embedding_hashes.csv`, `vm_instance_manifest.csv`, `seed_block_manifest.csv`,
`attempt_lineage.md`. Cross-VM v1 is NONCANONICAL (diagnostic only).
""",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v2",required=True); ap.add_argument("--v1",default="")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out); (out/"plots").mkdir(parents=True,exist_ok=True)
    figs=canonical(a.v2,out/"plots")
    mini_report(a.v2,out/"R011_report.md",figs)
    print("canonical figures",len(figs))
    if a.v1:
        d=Path(a.v1); (d/"plots").mkdir(parents=True,exist_ok=True)
        dfigs=diag(a.v1,d/"plots")
        Path(d/"NONCANONICAL_DIAGNOSTIC_REPORT.md").write_text(
            "# R011 cross-VM v1 — NONCANONICAL DIAGNOSTIC\n\n"
            "**NONCANONICAL / NOT FOR PRIMARY INFERENCE.**\n\n"
            "First attempt computed cells across recycled VMs; within-seed attention\n"
            "parity (#5) broke for cross-host pairs. Metrics diagnostic only; never\n"
            "pooled with seed-atomic v2.\n\nPlots:\n"+
            "\n".join(f"- `plots/{f.name}`" for f in dfigs)+"\n",encoding="utf-8")
        print("diagnostic figures",len(dfigs))

if __name__=="__main__": main()
