"""Verify frozen V13 against V10 on a large fresh split.

Produces distribution metrics, paired bootstrap confidence intervals, trimmed
means, cost-component comparisons, and quantitative profiles for improved and
regressed segments using only the public route telemetry.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def load_records(path):
  payload=json.loads(Path(path).read_text())
  records=payload.get('records') or payload.get('results') or payload.get('segments')
  if isinstance(records,dict): records=list(records.values())
  if not isinstance(records,list):
    raise RuntimeError(f'no records list in {path}')
  return {str(r['segment']):r for r in records}


def percentiles(v):
  return {f'p{p}':float(np.percentile(v,p)) for p in (50,75,90,95,99)}


def trimmed_mean(v,fraction):
  v=np.sort(np.asarray(v,float)); k=int(len(v)*fraction)
  return float(v[k:len(v)-k].mean()) if len(v)>2*k else float(v.mean())


def bootstrap(delta,n,seed):
  rng=np.random.default_rng(seed); size=len(delta); samples=np.empty(n,float)
  for i in range(n): samples[i]=delta[rng.integers(0,size,size)].mean()
  return {'mean_delta':float(delta.mean()),'ci95':[float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))],'probability_improvement':float(np.mean(samples<0)),'samples':n}


def route_features(csv_path):
  df=pd.read_csv(csv_path)
  def col(name,default=0.0):
    return pd.to_numeric(df[name],errors='coerce').fillna(default).to_numpy(float) if name in df else np.full(len(df),default,float)
  speed=col('v_ego'); accel=col('a_ego'); target=col('target_lataccel'); roll=col('roll_lataccel'); steer=col('steer_command')
  dtarget=np.diff(target,prepend=target[0] if len(target) else 0); dsteer=np.diff(steer,prepend=steer[0] if len(steer) else 0)
  return {'mean_speed':float(np.mean(speed)),'max_speed':float(np.max(speed)),'mean_abs_target':float(np.mean(np.abs(target))),'max_abs_target':float(np.max(np.abs(target))),'target_variability':float(np.std(target)),'mean_abs_target_change':float(np.mean(np.abs(dtarget))),'max_abs_roll':float(np.max(np.abs(roll))),'mean_abs_accel':float(np.mean(np.abs(accel))),'teacher_action_variability':float(np.std(steer)),'teacher_action_reversals':int(np.sum(np.sign(dsteer[1:])*np.sign(dsteer[:-1])<0))}


def aggregate(rows):
  if not rows:return {}
  keys=rows[0].keys(); return {k:float(np.mean([r[k] for r in rows])) for k in keys}


def main(a):
  b=load_records(a.baseline); c=load_records(a.candidate); segments=sorted(set(b)&set(c))
  if len(segments)<a.minimum_segments: raise RuntimeError(f'only {len(segments)} paired segments')
  bv=np.array([b[s]['total_cost'] for s in segments],float); cv=np.array([c[s]['total_cost'] for s in segments],float); delta=cv-bv
  blat=np.array([b[s].get('lataccel_cost',np.nan) for s in segments],float); clat=np.array([c[s].get('lataccel_cost',np.nan) for s in segments],float); bj=np.array([b[s].get('jerk_cost',np.nan) for s in segments],float); cj=np.array([c[s].get('jerk_cost',np.nan) for s in segments],float)
  improved=[s for s,d in zip(segments,delta) if d< -a.epsilon]; regressed=[s for s,d in zip(segments,delta) if d>a.epsilon]; unchanged=[s for s,d in zip(segments,delta) if abs(d)<=a.epsilon]
  feature_rows={s:route_features(Path(a.data_dir)/f'{s}.csv') for s in segments}
  report={'scope':'v13-frozen-verification-500','segments':len(segments),'baseline':{'mean':float(bv.mean()),'median':float(np.median(bv)),'worst':float(bv.max()),**percentiles(bv),'trimmed_mean_1pct':trimmed_mean(bv,.01),'trimmed_mean_5pct':trimmed_mean(bv,.05),'mean_lataccel_cost':float(np.nanmean(blat)),'mean_jerk_cost':float(np.nanmean(bj))},'candidate':{'mean':float(cv.mean()),'median':float(np.median(cv)),'worst':float(cv.max()),**percentiles(cv),'trimmed_mean_1pct':trimmed_mean(cv,.01),'trimmed_mean_5pct':trimmed_mean(cv,.05),'mean_lataccel_cost':float(np.nanmean(clat)),'mean_jerk_cost':float(np.nanmean(cj))},'change_percent':{'mean':float((cv.mean()/bv.mean()-1)*100),'median':float((np.median(cv)/np.median(bv)-1)*100),'p90':float((np.percentile(cv,90)/np.percentile(bv,90)-1)*100),'p95':float((np.percentile(cv,95)/np.percentile(bv,95)-1)*100),'p99':float((np.percentile(cv,99)/np.percentile(bv,99)-1)*100),'worst':float((cv.max()/bv.max()-1)*100),'lataccel':float((np.nanmean(clat)/np.nanmean(blat)-1)*100),'jerk':float((np.nanmean(cj)/np.nanmean(bj)-1)*100)},'paired':{'improved':len(improved),'regressed':len(regressed),'unchanged':len(unchanged),'largest_improvements':[{'segment':s,'delta':float(delta[segments.index(s)])} for s in sorted(improved,key=lambda x:delta[segments.index(x)])[:20]],'largest_regressions':[{'segment':s,'delta':float(delta[segments.index(s)])} for s in sorted(regressed,key=lambda x:delta[segments.index(x)],reverse=True)[:20]],'bootstrap_mean_delta':bootstrap(delta,a.bootstrap,a.seed)},'route_profiles':{'improved_mean':aggregate([feature_rows[s] for s in improved]),'regressed_mean':aggregate([feature_rows[s] for s in regressed]),'difference_regressed_minus_improved':{}},'records':[{'segment':s,'baseline_total_cost':float(bv[i]),'candidate_total_cost':float(cv[i]),'delta':float(delta[i]),'baseline_lataccel_cost':float(blat[i]),'candidate_lataccel_cost':float(clat[i]),'baseline_jerk_cost':float(bj[i]),'candidate_jerk_cost':float(cj[i]),**feature_rows[s]} for i,s in enumerate(segments)]}
  im=report['route_profiles']['improved_mean']; rg=report['route_profiles']['regressed_mean']; report['route_profiles']['difference_regressed_minus_improved']={k:float(rg[k]-im[k]) for k in im if k in rg}
  Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({k:report[k] for k in ('segments','baseline','candidate','change_percent','paired')},sort_keys=True))

if __name__=='__main__':
  p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True); p.add_argument('--data-dir',default='official/data'); p.add_argument('--output',default='checkpoints/general/v13-verification-500.json'); p.add_argument('--minimum-segments',type=int,default=500); p.add_argument('--bootstrap',type=int,default=10000); p.add_argument('--seed',type=int,default=20260720); p.add_argument('--epsilon',type=float,default=1e-9); main(p.parse_args())
