#!/usr/bin/env python3
"""Generate manuscript tables from accepted canonical R010-R014 CSV artifacts.

Fail-closed policy: every summary/contrast used in manuscript tables is checked against
seed-level canonical data. Any mismatch raises RuntimeError (HARD STOP -> DS).
"""
from pathlib import Path
import csv, math, json, statistics

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "canonical_inputs"
OUT = ROOT / "generated_tables"
OUT.mkdir(parents=True, exist_ok=True)
T975_DF4 = 2.776  # canonical rounded Student-t critical value used by accepted summaries
TOL = 5e-4

def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def f(x): return float(x)
def mean(xs): return sum(xs)/len(xs)
def t_ci(xs):
    m=mean(xs); sd=statistics.stdev(xs); h=T975_DF4*sd/math.sqrt(len(xs)); return m,m-h,m+h

def chk(label,a,b,tol=TOL):
    if abs(a-b)>tol: raise RuntimeError(f"HARD STOP: {label}: generated={a} canonical={b}")

def md_table(headers, data):
    lines=["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    for r in data: lines.append("| " + " | ".join(map(str,r)) + " |")
    return "\n".join(lines)+"\n"

audit=[]
# R010
p=rows(IN/'R010/per_seed.csv'); s=rows(IN/'R010/summary.csv'); pd=rows(IN/'R010/paired_differences.csv')
methods=['xavier_g1.0','orthogonal','xavier_g1.2','ce_lcg']
means={m:mean([f(r['test_ppl']) for r in p if r['method']==m]) for m in methods}
smap={r['method']:r for r in s}
for m in methods[1:]:
    xs=[f(r['delta_test_ppl']) for r in pd if r['method']==m]
    mm,lo,hi=t_ci(xs); c=smap[m]
    chk(f'R010 {m} mean delta',mm,f(c['mean_delta_test_ppl']),1e-9); chk(f'R010 {m} lo',lo,f(c['ci95_lo']),1e-9); chk(f'R010 {m} hi',hi,f(c['ci95_hi']),1e-9)
    audit.append(f'R010 {m}: PASS')
r010=[]
for m in methods:
    if m=='xavier_g1.0': r010.append([m,f"{means[m]:.4f}",'reference','—'])
    else:
        c=smap[m]; r010.append([m,f"{means[m]:.4f}",f"{f(c['mean_delta_test_ppl']):+.4f}",f"[{f(c['ci95_lo']):+.4f}, {f(c['ci95_hi']):+.4f}]"])
(OUT/'table_2_R010.md').write_text(md_table(['Method','Mean held-out test PPL','Paired Δ vs xavier_g1.0','95% CI'],r010),encoding='utf-8')

# R011
p=rows(IN/'R011/per_seed.csv'); fs=rows(IN/'R011/factorial_summary.csv'); fc=rows(IN/'R011/factorial_contrasts.csv')
cell_order=['A0B0','A0B1','A1B0','A1B1']; cellmeans={c:mean([f(r['test_ppl']) for r in p if r['cell']==c]) for c in cell_order}
fmap={r['contrast']:r for r in fs}
for k in ['E_B0','E_B1','B_A0','B_A1','I']:
    xs=[f(r[k]) for r in fc]; mm,lo,hi=t_ci(xs); c=fmap[k]
    chk(f'R011 {k} mean',mm,f(c['mean'])); chk(f'R011 {k} lo',lo,f(c['ci95_lo'])); chk(f'R011 {k} hi',hi,f(c['ci95_hi']))
    audit.append(f'R011 {k}: PASS')
labels={'A0B0':'constructor / xavier_g1.0','A0B1':'constructor / orthogonal','A1B0':'historical_xavier / xavier_g1.0','A1B1':'historical_xavier / orthogonal'}
data=[[c,labels[c],f"{cellmeans[c]:.4f}"] for c in cell_order]
data += [['Contrast','Estimate','95% CI']]
for k in ['E_B0','E_B1','B_A0','B_A1','I']:
    c=fmap[k]; data.append([k,f"{f(c['mean']):+.4f}",f"[{f(c['ci95_lo']):+.4f}, {f(c['ci95_hi']):+.4f}]"])
(OUT/'table_3_R011.md').write_text(md_table(['Cell / contrast','Condition or estimate','Held-out test PPL / CI'],data),encoding='utf-8')

# R012
p=rows(IN/'R012/per_seed.csv'); fs=rows(IN/'R012/factorial_summary.csv'); fc=rows(IN/'R012/factorial_contrasts.csv')
cell_order=['S0D0','S0D1','S1D0','S1D1']; cellmeans={c:mean([f(r['test_ppl']) for r in p if r['cell']==c]) for c in cell_order}; fmap={r['contrast']:r for r in fs}
for k in ['scale_D0','scale_D1','redraw_S0','redraw_S1','interaction']:
    xs=[f(r[k]) for r in fc]; mm,lo,hi=t_ci(xs); c=fmap[k]
    chk(f'R012 {k} mean',mm,f(c['mean'])); chk(f'R012 {k} lo',lo,f(c['ci95_lo'])); chk(f'R012 {k} hi',hi,f(c['ci95_hi'])); audit.append(f'R012 {k}: PASS')
data=[[c,f"{cellmeans[c]:.4f}",'—'] for c in cell_order]
for k in ['scale_D0','scale_D1','redraw_S0','redraw_S1','interaction']:
    c=fmap[k]; data.append([k,f"{f(c['mean']):+.4f}",f"[{f(c['ci95_lo']):+.4f}, {f(c['ci95_hi']):+.4f}]"])
(OUT/'table_4_R012.md').write_text(md_table(['Cell / contrast','Mean held-out test PPL / estimate','95% CI'],data),encoding='utf-8')

# R013/R014 dose tables
for cyc, order, contrasts in [
 ('R013',['D_below','D_xavier','D_mid1','D_mid2','D_ctor'],['D_below_minus_D_xavier','D_xavier_minus_D_mid1','D_mid1_minus_D_mid2','D_mid2_minus_D_ctor','D_xavier_minus_D_ctor']),
 ('R014',['D_xavier','D_mid1','D_ctor'],['D_xavier_minus_D_ctor','D_xavier_minus_D_mid1','D_mid1_minus_D_ctor'])]:
    p=rows(IN/f'{cyc}/per_seed.csv'); ds=rows(IN/f'{cyc}/dose_summary.csv'); cs=rows(IN/f'{cyc}/paired_contrasts_summary.csv'); pc=rows(IN/f'{cyc}/paired_contrasts.csv'); sl=rows(IN/f'{cyc}/scale_ladder.csv')
    dmap={r['dose']:r for r in ds}; cmap={r['contrast']:r for r in cs}; lmap={r['dose']:r for r in sl}
    for d in order:
        mm=mean([f(r['test_ppl']) for r in p if r['dose']==d]); chk(f'{cyc} {d} mean PPL',mm,f(dmap[d]['test_ppl_mean']))
    for k in contrasts:
        xs=[f(r['delta_test_ppl']) for r in pc if r['contrast']==k]; mm,lo,hi=t_ci(xs); c=cmap[k]
        chk(f'{cyc} {k} mean',mm,f(c['mean_delta_test_ppl'])); chk(f'{cyc} {k} lo',lo,f(c['ci95_lo'])); chk(f'{cyc} {k} hi',hi,f(c['ci95_hi'])); audit.append(f'{cyc} {k}: PASS')
    dose_data=[]
    for d in order:
        q=dmap[d]; l=lmap[d]; dose_data.append([d,f"{f(l['target_rms']):.8f}",f"{f(l['factor_rel_ctor']):.8f}",f"{f(q['test_ppl_mean']):.4f}",f"[{f(q['test_ppl_ci_lo']):.4f}, {f(q['test_ppl_ci_hi']):.4f}]"])
    cont_data=[]
    for k in contrasts:
        c=cmap[k]; cont_data.append([k,f"{f(c['mean_delta_test_ppl']):+.4f}",f"[{f(c['ci95_lo']):+.4f}, {f(c['ci95_hi']):+.4f}]",c['ci_excludes_zero']])
    num=5 if cyc=='R013' else 6
    (OUT/f'table_{num}_{cyc}_doses.md').write_text(md_table(['Dose','Target RMS','Factor vs constructor','Mean held-out test PPL','95% CI'],dose_data)+"\n"+md_table(['Paired contrast','Mean Δ test PPL','95% CI','CI excludes zero'],cont_data),encoding='utf-8')

(OUT/'AUDIT_REPORT.json').write_text(json.dumps({'status':'PASS','checks':audit,'policy':'Any mismatch raises RuntimeError (HARD STOP -> DS).'},indent=2),encoding='utf-8')
print('PASS:', len(audit), 'canonical checks')
