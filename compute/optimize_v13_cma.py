import argparse,json,os,shutil,sys,tempfile
from pathlib import Path
import numpy as np
from cmaes import CMA

def pack(z,b):
 a={k:v.copy() for k,v in b.items()};w=np.asarray(b['linear_weights'],float).copy()
 for g in range(4):w[g*4:(g+1)*4]+=z[g]*.08*(np.abs(w[g*4:(g+1)*4])+.15)
 a['linear_weights']=w.astype(np.float32);a['linear_bias']=np.asarray([float(np.asarray(b['linear_bias']).reshape(-1)[0])+.08*z[4]],np.float32)
 out=float(np.asarray(b.get('output_scale',np.asarray([1.]))).reshape(-1)[0]);a['output_scale']=np.asarray([np.clip(out*np.exp(.1*z[5]),.2,1.)],np.float32)
 names=['kp','ki','kd','kpreview','inverse_scale','action_delta_limit'];d=[.195,.1,-.053,.1,.5,4.];s=[.04,.03,.03,.04,.15,.75];lo=[.05,.01,-.2,0,.1,.25];hi=[.4,.2,.1,.3,1.,4.]
 for i,n in enumerate(names):a['controller_'+n]=np.asarray([np.clip(d[i]+s[i]*z[6+i],lo[i],hi[i])],np.float32)
 return a

def score(c,b):
 d=np.maximum(c-b,0);return float(c.mean()+.25*np.percentile(c,90)+.1*c.max()+.35*d.mean()+.1*d.max()+.03*np.count_nonzero(d>1e-9))

def run(path,segs,m,sim,C):
 os.environ['RIYADH_GENERAL_MODEL']=str(Path(path).resolve());v=[];r=[]
 for q in segs:
  x=sim(m,f'data/{q}.csv',C(),debug=False).rollout();v.append(x['total_cost']);r.append({'segment':q,**x})
 return np.asarray(v),r

def main(a):
 root=Path.cwd();off=(root/a.official_dir).resolve();src=(root/a.controller_source).resolve();basep=(root/a.baseline_model).resolve();inv=(root/a.inverse_model).resolve();out=(root/a.output_model).resolve();rep=(root/a.output_report).resolve()
 shutil.copy2(src,off/'controllers'/'riyadh_general.py');os.environ['RIYADH_GENERAL_INVERSE_MODEL']=str(inv);sys.path.insert(0,str(off))
 with np.load(basep,allow_pickle=False) as x:b={k:x[k] for k in x.files}
 old=Path.cwd();os.chdir(off)
 from tinyphysics import TinyPhysicsModel,TinyPhysicsSimulator
 from controllers.riyadh_general import Controller
 m=TinyPhysicsModel('models/tinyphysics.onnx',debug=False);segs=[f'{i:05d}' for i in range(a.start,a.start+a.count)];bv,br=run(basep,segs,m,TinyPhysicsSimulator,Controller);opt=CMA(mean=np.zeros(12),sigma=a.sigma,population_size=a.population,seed=a.seed);hist=[];pool=[]
 with tempfile.TemporaryDirectory() as td:
  td=Path(td)
  for g in range(a.generations):
   batch=[segs[(g*a.batch+i)%len(segs)] for i in range(a.batch)];idx=[segs.index(x) for x in batch];bb=bv[idx];sol=[];rows=[]
   for j in range(a.population):
    z=opt.ask();p=td/f'{g}-{j}.npz';np.savez_compressed(p,**pack(z,b));v,_=run(p,batch,m,TinyPhysicsSimulator,Controller);o=score(v,bb);sol.append((z,o));rows.append({'o':o,'z':z.tolist()})
   opt.tell(sol);rows.sort(key=lambda x:x['o']);pool+=rows[:2];hist.append({'generation':g+1,'best':rows[0]['o']});print('V13',g+1,rows[0]['o'],flush=True)
  finals=[]
  for i,row in enumerate(sorted(pool,key=lambda x:x['o'])[:a.finalists]):
   p=td/f'f{i}.npz';arr=pack(row['z'],b);np.savez_compressed(p,**arr);v,r=run(p,segs,m,TinyPhysicsSimulator,Controller);finals.append({'objective':score(v,bv),'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'worst':float(v.max()),'regressed':int(np.count_nonzero(v>bv+1e-9)),'z':row['z'],'records':r,'arr':arr})
  finals.sort(key=lambda x:x['objective']);best=finals[0];out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**best.pop('arr'))
 os.chdir(old);payload={'scope':'v13-cma','dimensions':12,'baseline':{'mean':float(bv.mean()),'median':float(np.median(bv)),'p90':float(np.percentile(bv,90)),'worst':float(bv.max()),'records':br},'selected':{k:v for k,v in best.items() if k!='records'},'selected_records':best['records'],'history':hist};rep.parent.mkdir(parents=True,exist_ok=True);rep.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--official-dir',default='official');p.add_argument('--controller-source',default='compute/riyadh_v13_cma.py');p.add_argument('--inverse-model',required=True);p.add_argument('--baseline-model',required=True);p.add_argument('--output-model',default='checkpoints/general/models/v13-cma.npz');p.add_argument('--output-report',default='checkpoints/general/v13-cma-search.json');p.add_argument('--start',type=int,default=13500);p.add_argument('--count',type=int,default=48);p.add_argument('--batch',type=int,default=12);p.add_argument('--population',type=int,default=10);p.add_argument('--generations',type=int,default=12);p.add_argument('--finalists',type=int,default=5);p.add_argument('--sigma',type=float,default=.65);p.add_argument('--seed',type=int,default=20260720);main(p.parse_args())