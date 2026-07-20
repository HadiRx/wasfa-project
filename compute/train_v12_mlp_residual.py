"""Train a compact nonlinear residual policy from public warmup telemetry.

Uses a small vectorized NumPy MLP inspired by Karpathy's micrograd teaching
style. Only public warmup steer commands are used as supervised targets; route
identity is never exposed to the policy.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np

@contextmanager
def working_directory(path):
  previous = Path.cwd(); os.chdir(path)
  try: yield
  finally: os.chdir(previous)

class RecordingPolicy:
  def __init__(self): self.features = []
  def predict(self, features):
    self.features.append(np.asarray(features, dtype=np.float32).copy())
    return 0.0

def collect_examples(model, simulator_type, controller_type, segments, residual_limit, sample_start, sample_stop):
  xs, ys, records = [], [], []
  for segment in segments:
    controller = controller_type(); recorder = RecordingPolicy(); controller.policy = recorder
    simulator = simulator_type(model, f"data/{segment}.csv", controller, debug=False)
    seg_x, seg_y = [], []
    while simulator.step_idx < len(simulator.data):
      step_idx = simulator.step_idx
      simulator.step()
      if sample_start <= step_idx < sample_stop and recorder.features:
        features = recorder.features[-1]
        baseline_action = float(controller.previous_action)
        teacher_action = float(simulator.data["steer_command"].iloc[step_idx])
        residual = float(np.clip(teacher_action - baseline_action, -residual_limit, residual_limit))
        if np.isfinite(features).all() and np.isfinite(residual):
          seg_x.append(features); seg_y.append([residual])
    records.append({"segment": segment, **simulator.compute_cost()})
    if seg_x:
      xs.append(np.asarray(seg_x, dtype=np.float32)); ys.append(np.asarray(seg_y, dtype=np.float32))
  if not xs: raise RuntimeError("no finite V12 warmup examples collected")
  x, y = np.concatenate(xs), np.concatenate(ys)
  if x.size == 0 or y.size == 0: raise RuntimeError("empty V12 training matrix")
  return x, y, records

def rollout_summary(records):
  v = np.asarray([r["total_cost"] for r in records], dtype=np.float64)
  return {"segments": int(len(v)), "mean_total_cost": float(v.mean()), "median_total_cost": float(np.median(v)), "p90_total_cost": float(np.percentile(v,90)), "worst_total_cost": float(v.max())}

def prediction_metrics(labels, predictions):
  labels = np.asarray(labels, dtype=np.float64).reshape(-1); predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
  if labels.size == 0 or predictions.size != labels.size: raise RuntimeError("invalid V12 metric inputs")
  e = predictions - labels
  corr = float(np.corrcoef(labels, predictions)[0,1]) if labels.std() > 1e-12 and predictions.std() > 1e-12 else 0.0
  return {"samples": int(labels.size), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "correlation": corr}

class TinyMLP:
  def __init__(self, input_dim, hidden1, hidden2, rng):
    self.params = {
      "w1": (rng.randn(input_dim,hidden1)/np.sqrt(input_dim)).astype(np.float32), "b1": np.zeros(hidden1,np.float32),
      "w2": (rng.randn(hidden1,hidden2)/np.sqrt(hidden1)).astype(np.float32), "b2": np.zeros(hidden2,np.float32),
      "w3": (rng.randn(hidden2,1)/np.sqrt(hidden2)).astype(np.float32), "b3": np.zeros(1,np.float32)}
  def forward(self,x):
    h1=np.tanh(x@self.params["w1"]+self.params["b1"]); h2=np.tanh(h1@self.params["w2"]+self.params["b2"]); out=np.tanh(h2@self.params["w3"]+self.params["b3"])
    return out,(x,h1,h2,out)
  def backward(self,cache,grad_out,wd):
    x,h1,h2,out=cache; gr=grad_out*(1-out**2)
    g={"w3":h2.T@gr+wd*self.params["w3"],"b3":gr.sum(0)}
    gz2=(gr@self.params["w3"].T)*(1-h2**2); g["w2"]=h1.T@gz2+wd*self.params["w2"]; g["b2"]=gz2.sum(0)
    gz1=(gz2@self.params["w2"].T)*(1-h1**2); g["w1"]=x.T@gz1+wd*self.params["w1"]; g["b1"]=gz1.sum(0)
    return g

def main(args):
  root=Path.cwd(); official=(root/args.official_dir).resolve(); source=(root/args.controller_source).resolve(); inverse=(root/args.inverse_model).resolve(); output=(root/args.output).resolve()
  shutil.copy2(source,official/"controllers"/"riyadh_general.py"); os.environ["RIYADH_GENERAL_INVERSE_MODEL"]=str(inverse); os.environ["RIYADH_GENERAL_MODEL"]=str(official/"models"/"disabled-v12.npz"); sys.path.insert(0,str(official))
  train_segments=[f"{x:05d}" for x in range(args.train_start,args.train_start+args.train_count)]; val_segments=[f"{x:05d}" for x in range(args.validation_start,args.validation_start+args.validation_count)]
  if set(train_segments)&set(val_segments): raise SystemExit("V12 train/validation overlap")
  with working_directory(official):
    from tinyphysics import TinyPhysicsModel,TinyPhysicsSimulator
    from controllers.riyadh_general import Controller,FEATURE_COUNT
    model=TinyPhysicsModel("models/tinyphysics.onnx",debug=False)
    train_x,train_y,train_records=collect_examples(model,TinyPhysicsSimulator,Controller,train_segments,args.residual_limit,args.sample_start,args.sample_stop)
    val_x,val_y,val_records=collect_examples(model,TinyPhysicsSimulator,Controller,val_segments,args.residual_limit,args.sample_start,args.sample_stop)
  mean=train_x.mean(0).astype(np.float32); scale=np.maximum(train_x.std(0).astype(np.float32),1e-4)
  train_x=((train_x-mean)/scale).astype(np.float32); val_x=((val_x-mean)/scale).astype(np.float32); train_t=(train_y/args.residual_limit).astype(np.float32); val_t=(val_y/args.residual_limit).astype(np.float32)
  if mean.shape!=(FEATURE_COUNT,): raise RuntimeError(f"V12 feature mismatch {mean.shape}")
  for name,a in {"train_x":train_x,"train_t":train_t,"val_x":val_x,"val_t":val_t}.items():
    if a.size==0 or not np.isfinite(a).all(): raise RuntimeError(f"non-finite or empty {name}")
  rng=np.random.RandomState(args.seed); net=TinyMLP(FEATURE_COUNT,args.hidden1,args.hidden2,rng); m={k:np.zeros_like(v) for k,v in net.params.items()}; vv={k:np.zeros_like(v) for k,v in net.params.items()}; best={k:v.copy() for k,v in net.params.items()}; best_val=float("inf"); history=[]; step=0
  for epoch in range(args.epochs):
    losses=[]; order=rng.permutation(len(train_x))
    for start in range(0,len(order),args.batch_size):
      idx=order[start:start+args.batch_size]; xb,yb=train_x[idx],train_t[idx]; pred,cache=net.forward(xb); err=pred-yb; loss=float(np.mean(err**2))
      if not np.isfinite(loss): raise RuntimeError(f"non-finite training loss epoch {epoch+1}")
      grads=net.backward(cache,(2.0/len(xb))*err,args.weight_decay); norm=float(np.sqrt(sum(np.sum(g*g) for g in grads.values()))); clip=min(1.0,args.gradient_clip/max(norm,1e-12)); step+=1
      for k,g in grads.items():
        g*=clip; m[k]=args.beta1*m[k]+(1-args.beta1)*g; vv[k]=args.beta2*vv[k]+(1-args.beta2)*(g*g); mh=m[k]/(1-args.beta1**step); vh=vv[k]/(1-args.beta2**step); net.params[k]-=args.learning_rate*mh/(np.sqrt(vh)+1e-8)
      losses.append(loss)
    vp,_=net.forward(val_x); vl=float(np.mean((vp-val_t)**2))
    if not np.isfinite(vl): raise RuntimeError(f"non-finite validation loss epoch {epoch+1}")
    if vl<best_val: best_val=vl; best={k:v.copy() for k,v in net.params.items()}
    history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":vl}); print(f"V12_EPOCH epoch={epoch+1} train={np.mean(losses):.8f} val={vl:.8f}",flush=True)
  net.params=best; train_pred=net.forward(train_x)[0]*args.residual_limit; val_pred=net.forward(val_x)[0]*args.residual_limit
  output.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(output,feature_mean=mean,feature_scale=scale,w1=best["w1"],b1=best["b1"],w2=best["w2"],b2=best["b2"],w3=best["w3"],b3=best["b3"],residual_limit=np.asarray([args.residual_limit],np.float32),output_scale=np.asarray([args.output_scale],np.float32),ood_gate_start=np.asarray([args.ood_gate_start],np.float32),ood_gate_width=np.asarray([args.ood_gate_width],np.float32))
  meta={"scope":"v12-public-warmup-supervised-causal-mlp-residual","architecture":[FEATURE_COUNT,args.hidden1,args.hidden2,1],"seed":args.seed,"train_segments":train_segments,"validation_segments":val_segments,"sample_window":[args.sample_start,args.sample_stop],"training":{"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"weight_decay":args.weight_decay,"gradient_clip":args.gradient_clip,"residual_limit":args.residual_limit,"output_scale":args.output_scale,"ood_gate_start":args.ood_gate_start,"ood_gate_width":args.ood_gate_width},"teacher_baseline_train":rollout_summary(train_records),"teacher_baseline_validation":rollout_summary(val_records),"train_prediction":prediction_metrics(train_y,train_pred),"validation_prediction":prediction_metrics(val_y,val_pred),"best_validation_loss":best_val,"history":history,"model":str(output)}
  output.with_suffix(".json").write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
  print(json.dumps({"architecture":meta["architecture"],"train_prediction":meta["train_prediction"],"validation_prediction":meta["validation_prediction"],"model":str(output)},sort_keys=True,allow_nan=False))

if __name__=="__main__":
  p=argparse.ArgumentParser(); p.add_argument("--official-dir",default="official"); p.add_argument("--controller-source",default="compute/riyadh_general.py"); p.add_argument("--inverse-model",required=True); p.add_argument("--output",default="checkpoints/general/models/v12-mlp-residual.npz"); p.add_argument("--train-start",type=int,default=12000); p.add_argument("--train-count",type=int,default=200); p.add_argument("--validation-start",type=int,default=12300); p.add_argument("--validation-count",type=int,default=50); p.add_argument("--sample-start",type=int,default=20); p.add_argument("--sample-stop",type=int,default=80); p.add_argument("--hidden1",type=int,default=32); p.add_argument("--hidden2",type=int,default=16); p.add_argument("--epochs",type=int,default=24); p.add_argument("--batch-size",type=int,default=2048); p.add_argument("--learning-rate",type=float,default=.002); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--gradient-clip",type=float,default=1.0); p.add_argument("--residual-limit",type=float,default=.15); p.add_argument("--output-scale",type=float,default=.5); p.add_argument("--ood-gate-start",type=float,default=4.0); p.add_argument("--ood-gate-width",type=float,default=2.0); p.add_argument("--beta1",type=float,default=.9); p.add_argument("--beta2",type=float,default=.999); p.add_argument("--seed",type=int,default=20260720); main(p.parse_args())