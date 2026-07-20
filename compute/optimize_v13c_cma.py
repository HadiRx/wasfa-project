"""Hard-case guarded CMA-ES refinement around frozen V13-A.

Every candidate is scored on a rotating fresh batch plus a fixed guard set of
known regressions. Candidates that worsen any guard case are penalized heavily.
"""
from __future__ import annotations
import argparse,json,os,shutil,sys,tempfile
from pathlib import Path
import numpy as np
from cmaes import CMA


def scalar(a,key,default):
  return float(np.asarray(a[key]).reshape(-1)[0]) if key in a else float(default)


def pack(z,b):
  a={k:v.copy() for k,v in b.items()}; w=np.asarray(b['linear_weights'],float).copy()
  for g in range(4): w[g*4:(g+1)*4]+=z[g]*.018*(np.abs(w[g*4:(g+1)*4])+.08)
  a['linear_weights']=w.astype(np.float32)
  a['linear_bias']=np.asarray([scalar(b,'linear_bias',0.0)+.018*z[4]],np.float32)
  out=scalar(b,'output_scale',1.0); a['output_scale']=np.asarray([np.clip(out*np.exp(.025*z[5]),.20,1.0)],np.float32)
  specs=[('kp',.012,.04,.45),('ki',.010,.005,.24),('kd',.010,-.24,.12),('kpreview',.012,0,.36),('inverse_scale',.040,.02,1.15),('action_delta_limit',.20,.20,4.0)]
  defaults={'kp':.195,'ki':.1,'kd':-.053,'kpreview':.1,'inverse_scale':.5,'action_delta_limit':4.0}
  for i,(name,scale,lo,hi) in enumerate(specs):
    center=scalar(b,'controller_'+name,defaults[name]); a['controller_'+name]=np.asarray([np.clip(center+scale*z[6+i],lo,hi)],np.float32)
  decay=scalar(b,'controller_integral_decay',1.0); limit=scalar(b,'controller_integral_limit',8.0)
  a['controller_integral_decay']=np.asarray([np.clip(decay+.008*z[12],.94,1.0)],np.float32)
  a['controller_integral_limit']=np.asarray([np.clip(limit*np.exp(.08*z[13]),3.0,16.0)],np.float32)
  return a


def objective(c,b,gc,gb):
  d=np.maximum(c-b,0); gd=np.maximum(gc-gb,0)
  guard_breach=float(np.count_nonzero(gd>1e-9))
  return float(c.mean()+.30*np.percentile(c,90)+.12*c.max()+.65*d.mean()+.22*d.max()+.05*np.count_nonzero(d>1e-9)+1.20*gd.mean()+.80*gd.max()+8.0*guard_breach)


def run(path,segs,m,sim,C):
  os.environ['RIYADH_GENERAL_MODEL']=str(Path(path).resolve()); vals=[]; records=[]
  for seg in segs:
    r=sim(m,f'data/{seg}.csv',C(),debug=False).rollout(); vals.append(r['total_cost']); records.append({'segment':seg,**r})
  return np.asarray(vals),records


def main(a):
  root=Path.cwd(); off=(root/a.official_dir).resolve(); src=(root/a.controller_source).resolve(); basep=(root/a.baseline_model).resolve(); inv=(root/a.inverse_model).resolve(); out=(root/a.output_model).resolve(); rep=(root/a.output_report).resolve()
  shutil.copy2(src,off/'controllers'/'riyadh_general.py'); os.environ['RIYADH_GENERAL_INVERSE_MODEL']=str(inv); sys.path.insert(0,str(off))
  with np.load(basep,allow_pickle=False) as x: base={k:x[k] for k in x.files}
  old=Path.cwd(); os.chdir(off)
  from tinyphysics import TinyPhysicsModel,TinyPhysicsSimulator
  from controllers.riyadh_general import Controller
  model=TinyPhysicsModel('models/tinyphysics.onnx',debug=False); segs=[f'{i:05d}' for i in range(a.start,a.start+a.count)]; guards=[x.strip() for x in a.guard_segments.split(',') if x.strip()]
  bv,br=run(basep,segs,model,TinyPhysicsSimulator,Controller); gb,gbr=run(basep,guards,model,TinyPhysicsSimulator,Controller)
  opt=CMA(mean=np.zeros(14),sigma=a.sigma,population_size=a.population,seed=a.seed); hist=[]; pool=[]
  with tempfile.TemporaryDirectory() as td:
    td=Path(td)
    for g in range(a.generations):
      start=(g*a.batch)%len(segs); batch=[segs[(start+i)%len(segs)] for i in range(a.batch)]; idx=[segs.index(x) for x in batch]; bb=bv[idx]; sols=[]; rows=[]
      for j in range(a.population):
        z=opt.ask(); p=td/f'{g}-{j}.npz'; np.savez_compressed(p,**pack(z,base)); cv,_=run(p,batch,model,TinyPhysicsSimulator,Controller); cg,_=run(p,guards,model,TinyPhysicsSimulator,Controller); o=objective(cv,bb,cg,gb); sols.append((z,o)); rows.append({'objective':o,'z':z.tolist(),'guard_worst_delta':float(np.max(cg-gb)),'guard_regressed':int(np.count_nonzero(cg>gb+1e-9))})
      opt.tell(sols); rows.sort(key=lambda x:x['objective']); pool+=rows[:3]; hist.append({'generation':g+1,**rows[0]}); print('V13C',g+1,rows[0],flush=True)
    finals=[]
    for i,row in enumerate(sorted(pool,key=lambda x:x['objective'])[:a.finalists]):
      p=td/f'final-{i}.npz'; arrays=pack(row['z'],base); np.savez_compressed(p,**arrays); cv,cr=run(p,segs,model,TinyPhysicsSimulator,Controller); cg,cgr=run(p,guards,model,TinyPhysicsSimulator,Controller); finals.append({'objective':objective(cv,bv,cg,gb),'mean':float(cv.mean()),'median':float(np.median(cv)),'p90':float(np.percentile(cv,90)),'worst':float(cv.max()),'regressed':int(np.count_nonzero(cv>bv+1e-9)),'guard_regressed':int(np.count_nonzero(cg>gb+1e-9)),'guard_worst_delta':float(np.max(cg-gb)),'z':row['z'],'records':cr,'guard_records':cgr,'arrays':arrays})
    finals.sort(key=lambda x:(x['guard_regressed']>0,x['guard_worst_delta']>0,x['objective'])); best=finals[0]; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**best.pop('arrays'))
  os.chdir(old); payload={'scope':'v13c-hard-guard-cma','dimensions':14,'guard_segments':guards,'baseline':{'mean':float(bv.mean()),'median':float(np.median(bv)),'p90':float(np.percentile(bv,90)),'worst':float(bv.max()),'records':br},'guard_baseline_records':gbr,'selected':{k:v for k,v in best.items() if k not in ('records','guard_records')},'selected_records':best['records'],'selected_guard_records':best['guard_records'],'history':hist}; rep.parent.mkdir(parents=True,exist_ok=True); rep.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
  p=argparse.ArgumentParser(); p.add_argument('--official-dir',default='official'); p.add_argument('--controller-source',default='compute/riyadh_v13_cma.py'); p.add_argument('--inverse-model',required=True); p.add_argument('--baseline-model',required=True); p.add_argument('--output-model',default='checkpoints/general/models/v13c-candidate.npz'); p.add_argument('--output-report',default='checkpoints/general/v13c-search.json'); p.add_argument('--start',type=int,default=16000); p.add_argument('--count',type=int,default=96); p.add_argument('--batch',type=int,default=16); p.add_argument('--population',type=int,default=12); p.add_argument('--generations',type=int,default=16); p.add_argument('--finalists',type=int,default=6); p.add_argument('--sigma',type=float,default=.16); p.add_argument('--seed',type=int,default=20260722); p.add_argument('--guard-segments',default='15692,15338,15377,15273,15297,15029,15199,15166'); main(p.parse_args())