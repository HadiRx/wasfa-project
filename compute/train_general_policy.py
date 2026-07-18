"""Distill verified per-segment actions into a route-agnostic NumPy policy.

Teacher rollouts are split by segment before sample generation.  Validation
segments never contribute gradients, and neither route IDs nor fingerprints
are policy inputs.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import numpy as np

from riyadh_general import (
  CONTROL_CONTEXT_CALLS,
  FEATURE_COUNT,
  feedback_action,
  policy_features,
)


@contextmanager
def working_directory(path):
  previous = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(previous)


def load_actions(path):
  if path.suffix == ".b64":
    raw = base64.b64decode(path.read_text(encoding="ascii"))
    actions = np.load(BytesIO(raw), allow_pickle=False)
  else:
    actions = np.load(path, allow_pickle=False)
  actions = np.asarray(actions, dtype=np.float32)
  if actions.shape != (400,) or not np.isfinite(actions).all():
    raise ValueError(f"invalid teacher actions in {path}: {actions.shape}")
  return actions


def action_files(action_dir):
  root = Path(action_dir)
  found = {}
  for pattern in ("*.npy", "*.npy.b64"):
    for path in root.glob(pattern):
      segment = path.name.split(".")[0]
      found[segment] = path
  return [found[key] for key in sorted(found)]


def split_segments(paths, validation_fraction, seed):
  paths = list(paths)
  if len(paths) < 3:
    raise ValueError("at least three teacher segments are required")
  rng = np.random.RandomState(seed)
  order = rng.permutation(len(paths))
  validation_count = max(1, int(round(len(paths) * validation_fraction)))
  validation_count = min(validation_count, len(paths) - 2)
  validation_ids = set(order[:validation_count].tolist())
  train = [path for index, path in enumerate(paths) if index not in validation_ids]
  validation = [path for index, path in enumerate(paths) if index in validation_ids]
  if {p.name for p in train} & {p.name for p in validation}:
    raise AssertionError("segment leakage in train/validation split")
  return train, validation


class TeacherRecorder:
  def __init__(self, actions):
    self.actions = actions
    self.calls = 0
    self.error_integral = 0.0
    self.previous_error = 0.0
    self.previous_action = 0.0
    self.features = []
    self.residuals = []

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = float(target_lataccel - current_lataccel)
    self.error_integral += error
    features = policy_features(
      target_lataccel,
      current_lataccel,
      state,
      future_plan,
      self.error_integral,
      self.previous_error,
      self.previous_action,
    )
    error_diff = error - self.previous_error
    base = feedback_action(
      error,
      self.error_integral,
      error_diff,
      float(target_lataccel),
      float(features[11]),
      float(state.roll_lataccel),
    )
    if CONTROL_CONTEXT_CALLS <= self.calls < CONTROL_CONTEXT_CALLS + len(self.actions):
      index = self.calls - CONTROL_CONTEXT_CALLS
      action = float(self.actions[index])
      self.features.append(features)
      self.residuals.append(action - base)
    elif self.calls >= CONTROL_CONTEXT_CALLS + len(self.actions):
      action = float(self.actions[-1])
    else:
      action = base
    self.previous_error = error
    self.previous_action = action
    self.calls += 1
    return action


def import_simulator(official_dir):
  official = str(Path(official_dir).resolve())
  if official not in sys.path:
    sys.path.insert(0, official)
  from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
  return TinyPhysicsModel, TinyPhysicsSimulator


def collect_dataset(paths, official_dir):
  paths = [Path(path).resolve() for path in paths]
  TinyPhysicsModel, TinyPhysicsSimulator = import_simulator(official_dir)
  features, residuals, costs = [], [], {}
  with working_directory(official_dir):
    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    for source in paths:
      segment = source.name.split(".")[0]
      controller = TeacherRecorder(load_actions(source))
      simulator = TinyPhysicsSimulator(
        model, f"data/{segment}.csv", controller, debug=False
      )
      costs[segment] = simulator.rollout()
      if len(controller.features) != 400:
        raise RuntimeError(f"teacher rollout for {segment} has bad horizon")
      features.append(np.asarray(controller.features, dtype=np.float32))
      residuals.append(np.asarray(controller.residuals, dtype=np.float32)[:, None])
  return np.concatenate(features), np.concatenate(residuals), costs


class MLP:
  def __init__(self, inputs, hidden, seed):
    rng = np.random.RandomState(seed)
    self.params = {
      "w1": (rng.randn(inputs, hidden) * np.sqrt(2.0 / inputs)).astype(np.float32),
      "b1": np.zeros(hidden, dtype=np.float32),
      "w2": (rng.randn(hidden, hidden) * np.sqrt(1.0 / hidden)).astype(np.float32),
      "b2": np.zeros(hidden, dtype=np.float32),
      "w3": (rng.randn(hidden, 1) * np.sqrt(1.0 / hidden)).astype(np.float32),
      "b3": np.zeros(1, dtype=np.float32),
    }

  def forward(self, x):
    h1 = np.tanh(x @ self.params["w1"] + self.params["b1"])
    h2 = np.tanh(h1 @ self.params["w2"] + self.params["b2"])
    prediction = h2 @ self.params["w3"] + self.params["b3"]
    return prediction, (x, h1, h2)

  def gradients(self, cache, prediction, target):
    x, h1, h2 = cache
    dy = 2.0 * (prediction - target) / len(x)
    gradients = {
      "w3": h2.T @ dy,
      "b3": dy.sum(axis=0),
    }
    dh2 = (dy @ self.params["w3"].T) * (1.0 - h2 * h2)
    gradients["w2"] = h1.T @ dh2
    gradients["b2"] = dh2.sum(axis=0)
    dh1 = (dh2 @ self.params["w2"].T) * (1.0 - h1 * h1)
    gradients["w1"] = x.T @ dh1
    gradients["b1"] = dh1.sum(axis=0)
    return gradients


def train_model(train_x, train_y, val_x, val_y, args):
  model = MLP(FEATURE_COUNT, args.hidden, args.seed)
  moment1 = {key: np.zeros_like(value) for key, value in model.params.items()}
  moment2 = {key: np.zeros_like(value) for key, value in model.params.items()}
  rng = np.random.RandomState(args.seed + 1)
  best = None
  best_val = float("inf")
  step = 0
  for epoch in range(args.epochs):
    for start in range(0, len(train_x), args.batch_size):
      indices = rng.randint(0, len(train_x), size=min(args.batch_size, len(train_x)))
      x, y = train_x[indices], train_y[indices]
      prediction, cache = model.forward(x)
      gradients = model.gradients(cache, prediction, y)
      step += 1
      for key, gradient in gradients.items():
        moment1[key] = 0.9 * moment1[key] + 0.1 * gradient
        moment2[key] = 0.999 * moment2[key] + 0.001 * gradient * gradient
        corrected1 = moment1[key] / (1.0 - 0.9 ** step)
        corrected2 = moment2[key] / (1.0 - 0.999 ** step)
        model.params[key] -= args.learning_rate * corrected1 / (
          np.sqrt(corrected2) + 1e-8
        )
    val_prediction, _ = model.forward(val_x)
    val_mse = float(np.mean((val_prediction - val_y) ** 2))
    if val_mse < best_val:
      best_val = val_mse
      best = {key: value.copy() for key, value in model.params.items()}
    if epoch % 25 == 0 or epoch + 1 == args.epochs:
      train_prediction, _ = model.forward(train_x)
      train_mse = float(np.mean((train_prediction - train_y) ** 2))
      print(
        f"GENERAL_TRAIN epoch={epoch + 1} train_mse={train_mse:.8f} "
        f"val_mse={val_mse:.8f}", flush=True
      )
  model.params = best
  return model, best_val


def main(args):
  paths = action_files(args.action_dir)
  train_paths, val_paths = split_segments(
    paths, args.validation_fraction, args.seed
  )
  print("TRAIN_SEGMENTS", [p.name.split(".")[0] for p in train_paths])
  print("VALIDATION_SEGMENTS", [p.name.split(".")[0] for p in val_paths])
  train_x, train_y, train_costs = collect_dataset(train_paths, args.official_dir)
  val_x, val_y, val_costs = collect_dataset(val_paths, args.official_dir)
  feature_mean = train_x.mean(axis=0).astype(np.float32)
  feature_scale = train_x.std(axis=0).astype(np.float32)
  feature_scale = np.maximum(feature_scale, 1e-4)
  normalized_train = (train_x - feature_mean) / feature_scale
  normalized_val = (val_x - feature_mean) / feature_scale
  residual_limit = float(np.percentile(np.abs(train_y), 99.5))
  clipped_train_y = np.clip(train_y, -residual_limit, residual_limit)
  clipped_val_y = np.clip(val_y, -residual_limit, residual_limit)
  model, best_val = train_model(
    normalized_train,
    clipped_train_y,
    normalized_val,
    clipped_val_y,
    args,
  )

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    residual_limit=np.asarray([residual_limit], dtype=np.float32),
    **model.params,
  )
  metadata = {
    "policy": "route_agnostic_residual_mlp",
    "feature_count": FEATURE_COUNT,
    "train_segments": [p.name.split(".")[0] for p in train_paths],
    "validation_segments": [p.name.split(".")[0] for p in val_paths],
    "train_samples": int(len(train_x)),
    "validation_samples": int(len(val_x)),
    "best_validation_residual_mse": best_val,
    "residual_limit": residual_limit,
    "teacher_train_costs": train_costs,
    "teacher_validation_costs": val_costs,
    "seed": args.seed,
  }
  metadata_path = output.with_suffix(".json")
  metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(f"GENERAL_POLICY_SAVED model={output} metadata={metadata_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--action-dir", default="checkpoints/pilot")
  parser.add_argument("--output", default="models/riyadh_general_policy.npz")
  parser.add_argument("--validation-fraction", type=float, default=0.2)
  parser.add_argument("--hidden", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=250)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--seed", type=int, default=2026)
  main(parser.parse_args())
