"""Train a compact nonlinear residual policy from causal closed-loop rollouts.

The implementation follows the small, inspectable MLP training style popularized
by Andrej Karpathy's micrograd/zero-to-hero material, but uses vectorized NumPy
for practical speed.  Logged steer commands are used only as a supervised
teacher; segment identity is never exposed to the policy.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


@contextmanager
def working_directory(path):
  previous = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(previous)


class RecordingPolicy:
  def __init__(self):
    self.features = []

  def predict(self, features):
    self.features.append(np.asarray(features, dtype=np.float32).copy())
    return 0.0


def collect_examples(model, simulator_type, controller_type, segments, residual_limit):
  xs, ys = [], []
  records = []
  for segment in segments:
    controller = controller_type()
    recorder = RecordingPolicy()
    controller.policy = recorder
    simulator = simulator_type(model, f"data/{segment}.csv", controller, debug=False)
    cost = simulator.rollout()
    records.append({"segment": segment, **cost})

    # update() is called from step 20.  Scored control is steps 100:500,
    # therefore recorder features 80:480 align exactly with those steps.
    features = np.asarray(recorder.features[80:480], dtype=np.float32)
    baseline_actions = np.asarray(simulator.action_history[100:500], dtype=np.float32)
    teacher_actions = simulator.data["steer_command"].values[100:500].astype(np.float32)
    count = min(len(features), len(baseline_actions), len(teacher_actions))
    if count == 0:
      continue
    target_residual = np.clip(
      teacher_actions[:count] - baseline_actions[:count],
      -residual_limit,
      residual_limit,
    )
    finite = np.isfinite(features[:count]).all(axis=1) & np.isfinite(target_residual)
    xs.append(features[:count][finite])
    ys.append(target_residual[finite, None])
  if not xs:
    raise RuntimeError("no finite V12 training examples collected")
  return np.concatenate(xs), np.concatenate(ys), records


def rollout_summary(records):
  values = np.asarray([item["total_cost"] for item in records], dtype=np.float64)
  return {
    "segments": int(len(values)),
    "mean_total_cost": float(values.mean()),
    "median_total_cost": float(np.median(values)),
    "p90_total_cost": float(np.percentile(values, 90)),
    "worst_total_cost": float(values.max()),
  }


def prediction_metrics(labels, predictions):
  error = predictions - labels
  return {
    "samples": int(labels.size),
    "rmse": float(np.sqrt(np.mean(error ** 2))),
    "mae": float(np.mean(np.abs(error))),
    "correlation": float(np.corrcoef(labels.ravel(), predictions.ravel())[0, 1]),
  }


class TinyMLP:
  def __init__(self, input_dim, hidden1, hidden2, rng):
    # Fan-in scaling keeps tanh activations out of saturation at initialization.
    self.params = {
      "w1": (rng.randn(input_dim, hidden1) / np.sqrt(input_dim)).astype(np.float32),
      "b1": np.zeros(hidden1, dtype=np.float32),
      "w2": (rng.randn(hidden1, hidden2) / np.sqrt(hidden1)).astype(np.float32),
      "b2": np.zeros(hidden2, dtype=np.float32),
      "w3": (rng.randn(hidden2, 1) / np.sqrt(hidden2)).astype(np.float32),
      "b3": np.zeros(1, dtype=np.float32),
    }

  def forward(self, x):
    z1 = x @ self.params["w1"] + self.params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ self.params["w2"] + self.params["b2"]
    h2 = np.tanh(z2)
    raw = h2 @ self.params["w3"] + self.params["b3"]
    out = np.tanh(raw)
    return out, (x, h1, h2, out)

  def backward(self, cache, grad_out, weight_decay):
    x, h1, h2, out = cache
    grad_raw = grad_out * (1.0 - out ** 2)
    grads = {}
    grads["w3"] = h2.T @ grad_raw + weight_decay * self.params["w3"]
    grads["b3"] = grad_raw.sum(axis=0)
    grad_h2 = grad_raw @ self.params["w3"].T
    grad_z2 = grad_h2 * (1.0 - h2 ** 2)
    grads["w2"] = h1.T @ grad_z2 + weight_decay * self.params["w2"]
    grads["b2"] = grad_z2.sum(axis=0)
    grad_h1 = grad_z2 @ self.params["w2"].T
    grad_z1 = grad_h1 * (1.0 - h1 ** 2)
    grads["w1"] = x.T @ grad_z1 + weight_decay * self.params["w1"]
    grads["b1"] = grad_z1.sum(axis=0)
    return grads


def main(args):
  root = Path.cwd()
  official = (root / args.official_dir).resolve()
  source = (root / args.controller_source).resolve()
  inverse_model = (root / args.inverse_model).resolve()
  output = (root / args.output).resolve()
  shutil.copy2(source, official / "controllers" / "riyadh_general.py")
  os.environ["RIYADH_GENERAL_INVERSE_MODEL"] = str(inverse_model)
  os.environ["RIYADH_GENERAL_MODEL"] = str(official / "models" / "disabled-v12.npz")
  sys.path.insert(0, str(official))

  train_segments = [f"{x:05d}" for x in range(args.train_start, args.train_start + args.train_count)]
  validation_segments = [
    f"{x:05d}" for x in range(args.validation_start, args.validation_start + args.validation_count)
  ]
  if set(train_segments) & set(validation_segments):
    raise SystemExit("V12 train and validation splits overlap")

  with working_directory(official):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.riyadh_general import Controller, FEATURE_COUNT

    simulator_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    train_x, train_y, train_records = collect_examples(
      simulator_model, TinyPhysicsSimulator, Controller, train_segments, args.residual_limit
    )
    val_x, val_y, val_records = collect_examples(
      simulator_model, TinyPhysicsSimulator, Controller, validation_segments, args.residual_limit
    )

  feature_mean = train_x.mean(axis=0).astype(np.float32)
  feature_scale = np.maximum(train_x.std(axis=0).astype(np.float32), 1e-4)
  train_x = ((train_x - feature_mean) / feature_scale).astype(np.float32)
  val_x = ((val_x - feature_mean) / feature_scale).astype(np.float32)
  train_target = (train_y / args.residual_limit).astype(np.float32)
  val_target = (val_y / args.residual_limit).astype(np.float32)

  if feature_mean.shape != (FEATURE_COUNT,):
    raise RuntimeError(f"V12 feature mismatch: {feature_mean.shape}")

  rng = np.random.RandomState(args.seed)
  network = TinyMLP(FEATURE_COUNT, args.hidden1, args.hidden2, rng)
  adam_m = {name: np.zeros_like(value) for name, value in network.params.items()}
  adam_v = {name: np.zeros_like(value) for name, value in network.params.items()}
  best_params = {name: value.copy() for name, value in network.params.items()}
  best_val = float("inf")
  history = []
  step = 0

  for epoch in range(args.epochs):
    order = rng.permutation(len(train_x))
    epoch_losses = []
    for start in range(0, len(order), args.batch_size):
      indices = order[start:start + args.batch_size]
      xb, yb = train_x[indices], train_target[indices]
      prediction, cache = network.forward(xb)
      error = prediction - yb
      loss = float(np.mean(error ** 2))
      grad_out = (2.0 / len(xb)) * error
      grads = network.backward(cache, grad_out, args.weight_decay)
      norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
      clip = min(1.0, args.gradient_clip / max(norm, 1e-12))
      step += 1
      for name, grad in grads.items():
        grad = grad * clip
        adam_m[name] = args.beta1 * adam_m[name] + (1.0 - args.beta1) * grad
        adam_v[name] = args.beta2 * adam_v[name] + (1.0 - args.beta2) * (grad * grad)
        m_hat = adam_m[name] / (1.0 - args.beta1 ** step)
        v_hat = adam_v[name] / (1.0 - args.beta2 ** step)
        network.params[name] -= args.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
      epoch_losses.append(loss)

    val_prediction, _ = network.forward(val_x)
    val_loss = float(np.mean((val_prediction - val_target) ** 2))
    if val_loss < best_val:
      best_val = val_loss
      best_params = {name: value.copy() for name, value in network.params.items()}
    history.append({
      "epoch": epoch + 1,
      "train_loss": float(np.mean(epoch_losses)),
      "validation_loss": val_loss,
    })
    print(
      f"V12_EPOCH epoch={epoch + 1} train={np.mean(epoch_losses):.8f} val={val_loss:.8f}",
      flush=True,
    )

  network.params = best_params
  train_prediction = network.forward(train_x)[0] * args.residual_limit
  val_prediction = network.forward(val_x)[0] * args.residual_limit

  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    w1=best_params["w1"].astype(np.float32),
    b1=best_params["b1"].astype(np.float32),
    w2=best_params["w2"].astype(np.float32),
    b2=best_params["b2"].astype(np.float32),
    w3=best_params["w3"].astype(np.float32),
    b3=best_params["b3"].astype(np.float32),
    residual_limit=np.asarray([args.residual_limit], dtype=np.float32),
    output_scale=np.asarray([args.output_scale], dtype=np.float32),
    ood_gate_start=np.asarray([args.ood_gate_start], dtype=np.float32),
    ood_gate_width=np.asarray([args.ood_gate_width], dtype=np.float32),
  )
  metadata = {
    "scope": "v12-supervised-causal-mlp-residual",
    "inspiration": [
      "karpathy/micrograd: small transparent MLP and explicit backpropagation",
      "DLR-RM/stable-baselines3: strict evaluation separation and reproducible baselines",
      "commaai/controls_challenge: official TinyPhysics closed-loop simulator",
    ],
    "architecture": [FEATURE_COUNT, args.hidden1, args.hidden2, 1],
    "seed": args.seed,
    "train_segments": train_segments,
    "validation_segments": validation_segments,
    "training": {
      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "learning_rate": args.learning_rate,
      "weight_decay": args.weight_decay,
      "gradient_clip": args.gradient_clip,
      "residual_limit": args.residual_limit,
      "output_scale": args.output_scale,
      "ood_gate_start": args.ood_gate_start,
      "ood_gate_width": args.ood_gate_width,
    },
    "teacher_baseline_train": rollout_summary(train_records),
    "teacher_baseline_validation": rollout_summary(val_records),
    "train_prediction": prediction_metrics(train_y, train_prediction),
    "validation_prediction": prediction_metrics(val_y, val_prediction),
    "best_validation_loss": best_val,
    "history": history,
    "model": str(output),
  }
  output.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
    "architecture": metadata["architecture"],
    "train_prediction": metadata["train_prediction"],
    "validation_prediction": metadata["validation_prediction"],
    "model": str(output),
  }, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--controller-source", default="compute/riyadh_general.py")
  parser.add_argument("--inverse-model", required=True)
  parser.add_argument("--output", default="checkpoints/general/models/v12-mlp-residual.npz")
  parser.add_argument("--train-start", type=int, default=20500)
  parser.add_argument("--train-count", type=int, default=200)
  parser.add_argument("--validation-start", type=int, default=21000)
  parser.add_argument("--validation-count", type=int, default=50)
  parser.add_argument("--hidden1", type=int, default=32)
  parser.add_argument("--hidden2", type=int, default=16)
  parser.add_argument("--epochs", type=int, default=24)
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--learning-rate", type=float, default=0.002)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--gradient-clip", type=float, default=1.0)
  parser.add_argument("--residual-limit", type=float, default=0.15)
  parser.add_argument("--output-scale", type=float, default=0.50)
  parser.add_argument("--ood-gate-start", type=float, default=4.0)
  parser.add_argument("--ood-gate-width", type=float, default=2.0)
  parser.add_argument("--beta1", type=float, default=0.9)
  parser.add_argument("--beta2", type=float, default=0.999)
  parser.add_argument("--seed", type=int, default=20260720)
  main(parser.parse_args())
