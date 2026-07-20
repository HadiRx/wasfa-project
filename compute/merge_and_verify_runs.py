"""Merge sharded paired evaluations and produce robust verification metrics."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np


def load_many(pattern: str):
  records=[]
  for path in sorted(glob.glob(pattern)):
    payload=json.loads(Path(path).read_text())
    records.extend(payload.get('records',[]))
  by={str(r['segment']):r for r in records}
  return by


def trimmed(v,f):
  x=np.sort(np.asarray(v,float)); k=int(len(x)*f)
  return float(x[k:len(x)-k].mean()) if len(x)>2*k else float(x.mean())


def metrics(v,lat,jerk):
  return {
    'mean':float(v.mean()),'median':float(np.median(v)),'p75':float(np.percentile(v,75)),
    'p90':float(np.percentile(v,90)),'p95':float(np.percentile(v,95)),
    'p99':float(np.percentile(v,99)),'worst':float(v.max()),
    'trimmed_mean_1pct':trimmed(v,.01),'trimmed_mean_5pct':trimmed(v,.05),
    'mean_lataccel_cost':float(lat.mean()),'mean_jerk_cost':float(jerk.mean()),
  }


def bootstrap(delta,n,seed):
  rng=np.random.default_rng(seed); m=len(delta); means=np.empty(n,float)
  for i in range(n): means[i]=delta[rng.integers(0,m,m)].mean()
  return {'samples':n,'mean_delta':float(delta.mean()),
          'ci95':[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],
          'probability_improvement':float(np.mean(means<0))}


def main(a):
  b=load_many(a.baseline_glob); c=load_many(a.candidate_glob)
  segs=sorted(set(b)&set(c))
  if len(segs)!=a.expected: raise SystemExit(f'expected {a.expected} paired segments, got {len(segs)}')
  bv=np.array([b[s]['total_cost'] for s in segs],float); cv=np.array([c[s]['total_cost'] for s in segs],float)
  bl=np.array([b[s]['lataccel_cost'] for s in segs],float); cl=np.array([c[s]['lataccel_cost'] for s in segs],float)
  bj=np.array([b[s]['jerk_cost'] for s in segs],float); cj=np.array([c[s]['jerk_cost'] for s in segs],float)
  d=cv-bv; bm=metrics(bv,bl,bj); cm=metrics(cv,cl,cj)
  report={'scope':a.scope,'segments':len(segs),'baseline':bm,'candidate':cm,
    'change_percent':{k:float((cm[k]/bm[k]-1)*100) for k in ('mean','median','p90','p95','p99','worst','mean_lataccel_cost','mean_jerk_cost')},
    'paired':{'improved':int(np.count_nonzero(d< -a.epsilon)),'regressed':int(np.count_nonzero(d>a.epsilon)),
      'unchanged':int(np.count_nonzero(np.abs(d)<=a.epsilon)),'bootstrap_mean_delta':bootstrap(d,a.bootstrap,a.seed),
      'largest_improvements':[{'segment':segs[i],'delta':float(d[i])} for i in np.argsort(d)[:20]],
      'largest_regressions':[{'segment':segs[i],'delta':float(d[i])} for i in np.argsort(d)[-20:][::-1]]},
    'gate':{'mean_better':bool(cm['mean']<bm['mean']),'median_better':bool(cm['median']<bm['median']),
      'worst_better':bool(cm['worst']<bm['worst']),'bootstrap_ci_below_zero':bool(bootstrap(d,a.bootstrap,a.seed)['ci95'][1]<0)},
    'records':[{'segment':s,'baseline_total_cost':float(bv[i]),'candidate_total_cost':float(cv[i]),'delta':float(d[i]),
      'baseline_lataccel_cost':float(bl[i]),'candidate_lataccel_cost':float(cl[i]),
      'baseline_jerk_cost':float(bj[i]),'candidate_jerk_cost':float(cj[i])} for i,s in enumerate(segs)]}
  report['gate']['passed']=all(report['gate'].values())
  out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({k:report[k] for k in ('scope','segments','baseline','candidate','change_percent','paired','gate') if k!='records'},sort_keys=True))
  if a.require_pass and not report['gate']['passed']: raise SystemExit('large-scale verification gate failed')

if __name__=='__main__':
  p=argparse.ArgumentParser(); p.add_argument('--baseline-glob',required=True); p.add_argument('--candidate-glob',required=True)
  p.add_argument('--expected',type=int,required=True); p.add_argument('--scope',required=True); p.add_argument('--output',required=True)
  p.add_argument('--bootstrap',type=int,default=10000); p.add_argument('--seed',type=int,default=20260723)
  p.add_argument('--epsilon',type=float,default=1e-9); p.add_argument('--require-pass',action='store_true'); main(p.parse_args())
