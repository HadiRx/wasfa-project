"""Fine CMA-ES refinement around a frozen V13 candidate.

Unlike V13-A, every parameter is centered on the supplied V13 artifact. Two
additional integral-dynamics parameters are introduced conservatively. Search
uses the official TinyPhysics closed-loop cost with regression penalties.
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
  # Four local residual groups, now centered on the proven V13 weights.
  for g in range(4): w[g*4:(g+1)*4]+=z[g]*.025*(np.abs(w[g*4:(g+1)*4])+.10)
  a['linear_weights']=w.astype(np.float32)
  a['linear_bias']=np.asarray([scalar(b,'linear_bias',0.0)+.025*z[4]],np.float32)
  out=scalar(b,'output_scale',1.0); a['output_scale']=np.asarray([np.clip(out*np.exp(.035*z[5]),.20,1.0)],np.float32)
  specs=[
    ('kp',.018,.04,.45),('ki',.014,.005,.24),('kd',.014,-.24,.12),
    ('kpreview',.018,0,.36),('inverse_scale',.055,.02,1.15),('action_delta_limit',.30,.20,4.0),
  ]
  defaults={'kp':.195,'ki':.1,'kd':-.053,'kpreview':.1,'inverse_scale':.5,'action_delta_limit':4.0}
  for i,(name,scale,lo,hi) in enumerate(specs):
    center=scalar(b,'controller_'+name,defaults[name])
    a['controller_'+name]=np.asarray([np.clip(center+scale*z[6+i],lo,hi)],np.float32)
  # New conservative dynamics. Decay near 1.0 and a finite integral bound.
  decay_center=scalar(b,'controller_integral_decay',1.0)
  limit_center=scalar(b,'controller_integral_limit',8.0)
  a['controller_integral_decay']=np.asarray([np.clip(decay_center+.012*z[12],.92,1.0)],np.float32)
  a['controller_integral_limit']=np.asarray([np.clip(limit_center*np.exp(.12*z[13]),2.0,20.0)],np.float32)
  return a


def score(c,b):
  d=np.maximum(c-b,0)
  return float(c.mean()+.30*np.percentile(c,90)+.12*c.max()+.55*d.mean()+.18*d.max()+.05*np.count_nonzero(d>1e-9))


def run(path,segs,m,sim,C):
  os.environ['RIYADH_GENERAL_MODEL']=str(Path(path).resolve()); vals=[]; records=[]
  for seg in segs:
    result=sim(m,f'data/{seg}.csv',C(),debug=False).rollout(); vals.append(result['total_cost']); records.append({'segment':seg,**result})
  return np.asarray(vals),records


def main(a):
  root=Path.cwd(); off=(root/a.official_dir).resolve(); src=(root/a.controller_source).resolve(); basep=(root/a.baseline_model).resolve(); inv=(root/a.inverse_model).resolve(); out=(root/a.output_model).resolve(); rep=(root/a.output_report).resolve()
  shutil.copy2(src,off/'controllers'/'riyadh_general.py'); os.environ['RIYADH_GENERAL_INVERSE_MODEL']=str(inv); sys.path.insert(0,str(off))
  with np.load(basep,allow_pickle=False) as x: base={k:x[k] for k in x.files}
  old=Path.cwd(); os.chdir(off)
  from tinyphysics import TinyPhysicsModel,TinyPhysicsSimulator
  from controllers.riyadh_general import Controller
  model=TinyPhysicsModel('models/tinyphysics.onnx',debug=False); segs=[f'{i:05d}' for i in range(a.start,a.start+a.count)]; baseline_values,baseline_records=run(basep,segs,model,TinyPhysicsSimulator,Controller)
  opt=CMA(mean=np.zeros(14),sigma=a.sigma,population_size=a.population,seed=a.seed); history=[]; pool=[]
  with tempfile.TemporaryDirectory() as td:
    td=Path(td)
    for generation in range(a.generations):
      start=(generation*a.batch)%len(segs); batch=[segs[(start+i)%len(segs)] for i in range(a.batch)]; idx=[segs.index(x) for x in batch]; batch_base=baseline_values[idx]; solutions=[]; rows=[]
      for j in range(a.population):
        z=opt.ask(); path=td/f'{generation}-{j}.npz'; np.savez_compressed(path,**pack(z,base)); values,_=run(path,batch,model,TinyPhysicsSimulator,Controller); objective=score(values,batch_base); solutions.append((z,objective)); rows.append({'objective':objective,'z':z.tolist()})
      opt.tell(solutions); rows.sort(key=lambda x:x['objective']); pool+=rows[:3]; history.append({'generation':generation+1,'best':rows[0]['objective']}); print('V13B',generation+1,rows[0]['objective'],flush=True)
    finalists=[]
    for i,row in enumerate(sorted(pool,key=lambda x:x['objective'])[:a.finalists]):
      path=td/f'final-{i}.npz'; arrays=pack(row['z'],base); np.savez_compressed(path,**arrays); values,records=run(path,segs,model,TinyPhysicsSimulator,Controller); finalists.append({'objective':score(values,baseline_values),'mean':float(values.mean()),'median':float(np.median(values)),'p90':float(np.percentile(values,90)),'worst':float(values.max()),'regressed':int(np.count_nonzero(values>baseline_values+1e-9)),'z':row['z'],'records':records,'arrays':arrays})
    finalists.sort(key=lambda x:x['objective']); best=finalists[0]; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**best.pop('arrays'))
  os.chdir(old)
  report={'scope':'v13b-fine-cma','dimensions':14,'baseline':{'mean':float(baseline_values.mean()),'median':float(np.median(baseline_values)),'p90':float(np.percentile(baseline_values,90)),'worst':float(baseline_values.max()),'records':baseline_records},'selected':{k:v for k,v in best.items() if k!='records'},'selected_records':best['records'],'history':history}
  rep.parent.mkdir(parents=True,exist_ok=True); rep.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--official-dir',default='official');p.add_argument('--controller-source',default='compute/riyadh_v13_cma.py');p.add_argument('--inverse-model',required=True);p.add_argument('--baseline-model',required=True);p.add_argument('--output-model',default='checkpoints/general/models/v13b-cma.npz');p.add_argument('--output-report',default='checkpoints/general/v13b-cma-search.json');p.add_argument('--start',type=int,default=15500);p.add_argument('--count',type=int,default=96);p.add_argument('--batch',type=int,default=16);p.add_argument('--population',type=int,default=14);p.add_argument('--generations',type=int,default=20);p.add_argument('--finalists',type=int,default=6);p.add_argument('--sigma',type=float,default=.22);p.add_argument('--seed',type=int,default=20260721);main(p.parse_args())