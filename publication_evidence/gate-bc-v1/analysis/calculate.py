"""Deterministic analysis of accepted CSVs only; stdlib, no model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

T_CRITICAL = 2.7764451051977987
CELLS = ('B00','B01','B10','B11','C_002')


def calculate(root):
    inputs, values, per_seed = [], {}, []
    for seed in range(67,72):
        p=root/'data'/str(seed)/'metrics.csv'
        inputs.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        with p.open(newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==5 and {r['condition'] for r in rows}==set(CELLS)
        assert all(int(r['seed'])==seed and int(r['test_evaluations'])==1 and r['status']=='CANONICAL' for r in rows)
        x={r['condition']:float(r['test_ppl']) for r in rows}
        assert all(math.isfinite(v) for v in x.values())
        for cell in CELLS: values.setdefault(cell,[]).append(x[cell])
        b00,b01,b10,b11,c=(x[k] for k in CELLS)
        per_seed.append({'seed':seed,'Delta_P':(b10+b11-b00-b01)/2,
                         'Delta_F':(b01+b11-b00-b10)/2,'I':b11-b10-b01+b00,
                         'Delta_002':c-b00,'P_at_small_F':b10-b00,'P_at_large_F':b11-b01,
                         'F_at_small_P':b01-b00,'F_at_large_P':b11-b10})
    def summarize(xs):
        mean=statistics.mean(xs);sd=statistics.stdev(xs);margin=T_CRITICAL*sd/math.sqrt(5)
        return dict(mean=mean,sample_sd=sd,ci_lower=mean-margin,ci_upper=mean+margin,
                    positive=sum(x>0 for x in xs),negative=sum(x<0 for x in xs),n=5,df=4)
    contrasts={k:summarize([r[k] for r in per_seed]) for k in per_seed[0] if k!='seed'}
    # Decision is externally fixed by DS, not assigned by this calculation.
    return {'input_hashes':inputs,'t_critical':T_CRITICAL,'conditions':{k:summarize(v) for k,v in values.items()},
            'contrasts':contrasts,'per_seed':per_seed,
            'ds_classification':{'GateB':'INTERACTION_RESOLVED','GateC':'C002_WORSE'},
            'scientific_scope':'fixed implementation / WikiText-2 / fixed training regime'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=calculate(args.root)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('conditions',result['conditions']),('contrasts',result['contrasts'])]:
        with (args.out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['label',*next(iter(rows.values())).keys()]);w.writeheader()
            for k,v in rows.items():w.writerow({'label':k,**v})
    with (args.out/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['per_seed'][0]));w.writeheader();w.writerows(result['per_seed'])
    print('ANALYSIS_COMPLETE: 25 accepted endpoints; no training/evaluation')


if __name__=='__main__':main()
