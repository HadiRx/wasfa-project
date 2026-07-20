from __future__ import annotations
import os
from pathlib import Path
import numpy as np
try:
  from . import BaseController
except ImportError:
  class BaseController: pass
STEER_LIMIT=2.0; FEATURE_COUNT=16; INVERSE_FEATURE_COUNT=17
DEFAULT_MODEL=Path(__file__).resolve().parent.parent/"models"/"riyadh_general_policy.npz"
DEFAULT_INVERSE_MODEL=Path(__file__).resolve().parent.parent/"models"/"riyadh_inverse_linear.npz"
def _mean(v,n,f):
  a=np.asarray(v[:n],dtype=np.float64); return float(a.mean()) if a.size else float(f)
def policy_features(target,current,state,plan,integral,prev_error,prev_action):
  t=float(target); c=float(current); e=t-c; fl=list(plan.lataccel); fr=list(plan.roll_lataccel); p1=float(fl[0]) if fl else t
  return np.asarray([t,c,e,e-float(prev_error),float(np.clip(integral,-8,8)),float(state.roll_lataccel),float(state.v_ego)/40,float(state.a_ego)/5,float(prev_action),p1-t,_mean(fl,5,t)-t,_mean(fl,10,t)-t,_mean(fl,20,t)-t,_mean(fl,40,t)-t,_mean(fr,10,state.roll_lataccel)-float(state.roll_lataccel),abs(e)],dtype=np.float32)
def inverse_features(target,state,plan):
  t=float(target); fl=list(plan.lataccel); fr=list(plan.roll_lataccel)
  def at(v,i,f): return float(v[i]) if len(v)>i else float(f)
  t1=at(fl,0,t); t2=at(fl,1,t1); t5=at(fl,4,t2); t10=at(fl,9,t5); t20=at(fl,19,t10); r1=at(fr,0,state.roll_lataccel); s=float(state.v_ego); net=t1-r1
  return np.asarray([t,t1,t2,t5,t10,t20,t1-t,t5-t,float(state.roll_lataccel),r1,s,float(state.a_ego),net/(s*s+1),net/(s+1),abs(t1),t1*s,float(state.roll_lataccel)*s],dtype=np.float32)
class Residual:
  def __init__(self,path):
    self.available=False; self.params={}
    if not Path(path).exists(): return
    with np.load(path,allow_pickle=False) as a:
      self.mean=a["feature_mean"].astype(np.float32); self.scale=a["feature_scale"].astype(np.float32); self.weights=a["linear_weights"].astype(np.float32); self.bias=float(np.asarray(a["linear_bias"]).reshape(-1)[0]); self.limit=float(np.asarray(a["residual_limit"]).reshape(-1)[0]); self.output=float(np.asarray(a["output_scale"]).reshape(-1)[0]) if "output_scale" in a.files else 1.0
      for n in ("kp","ki","kd","kpreview","inverse_scale","action_delta_limit"):
        k=f"controller_{n}"; self.params[n]=float(np.asarray(a[k]).reshape(-1)[0]) if k in a.files else None
    self.available=True
  def predict(self,x):
    if not self.available:return 0.0
    z=(x-self.mean)/self.scale; return float(np.tanh(float(z@self.weights+self.bias))*self.limit*self.output)
class Inverse:
  def __init__(self,path):
    self.available=False
    if not Path(path).exists(): return
    with np.load(path,allow_pickle=False) as a: self.mean=a["feature_mean"].astype(np.float32); self.scale=a["feature_scale"].astype(np.float32); self.weights=a["weights"].astype(np.float32); self.bias=float(np.asarray(a["bias"]).reshape(-1)[0])
    self.available=True
  def predict(self,x):
    if not self.available:return 0.0
    return float(np.clip(((x-self.mean)/self.scale)@self.weights+self.bias,-STEER_LIMIT,STEER_LIMIT))
class Controller(BaseController):
  def __init__(self):
    self.policy=Residual(Path(os.getenv("RIYADH_GENERAL_MODEL",str(DEFAULT_MODEL)))); self.inverse=Inverse(Path(os.getenv("RIYADH_GENERAL_INVERSE_MODEL",str(DEFAULT_INVERSE_MODEL)))); self.integral=0.; self.prev_error=0.; self.prev_action=0.
    def p(name,default):
      v=self.policy.params.get(name) if self.policy.available else None
      return float(default if v is None else v)
    self.kp=p("kp",.195); self.ki=p("ki",.100); self.kd=p("kd",-.053); self.kpreview=p("kpreview",.10); self.inverse_scale=p("inverse_scale",.50); self.action_delta=p("action_delta_limit",4.0)
  def update(self,target_lataccel,current_lataccel,state,future_plan):
    e=float(target_lataccel-current_lataccel); self.integral+=e; f=policy_features(target_lataccel,current_lataccel,state,future_plan,self.integral,self.prev_error,self.prev_action); d=e-self.prev_error; base=self.kp*e+self.ki*self.integral+self.kd*d+self.kpreview*float(f[11]); inv=np.clip(self.inverse.predict(inverse_features(target_lataccel,state,future_plan)),-1,1); req=base+self.inverse_scale*inv+self.policy.predict(f); action=float(np.clip(req,self.prev_action-self.action_delta,self.prev_action+self.action_delta)); action=float(np.clip(action,-STEER_LIMIT,STEER_LIMIT)); self.prev_error=e; self.prev_action=action; return action