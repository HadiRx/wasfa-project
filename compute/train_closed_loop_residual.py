"""Train a compact route-agnostic residual directly on closed-loop cost.

The optimizer uses deterministic simulator rollouts on an explicit training
split.  It never exposes segment identity to the policy and evaluates the
selected candidate once on a disjoint validation split.
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


class LinearResidualPolicy:
  def __init__(
    self, feature_mean, feature_scale, parameters, residual_limit,
    ood_gate_start=float("inf"), ood_gate_width=1.0,
  ):
    self.feature_mean = feature_mean
    self.feature_scale = feature_scale
    self.weights = np.asarray(parameters[:-1], dtype=np.float32)
    self.bias = float(parameters[-1])
    self.residual_limit = float(residual_limit)
    self.ood_gate_start = float(ood_gate_start)
    self.ood_gate_width = float(ood_gate_width)

  def predict(self, features):
    normalized = (features - self.feature_mean) / self.feature_scale
    value = float(normalized @ self.weights + self.bias)
    gate = 1.0
    if np.isfinite(self.ood_gate_start):
      distance = float(np.max(np.abs(normalized)))
      gate = float(np.clip(
        (self.ood_gate_start + max(self.ood_gate_width, 1e-6) - distance)
        / max(self.ood_gate_width, 1e-6),
        0.0,
        1.0,
      ))
    return float(np.tanh(value) * self.residual_limit * gate)


def summary(costs):
  values = np.asarray(costs, dtype=np.float64)
  return {
    "mean_total_cost": float(values.mean()),
    "median_total_cost": float(np.median(values)),
    "p90_total_cost": float(np.percentile(values, 90)),
    "worst_total_cost": float(values.max()),
  }


def objective(costs, baseline, tail_weight, regression_weight):
  costs = np.asarray(costs, dtype=np.float64)
  baseline = np.asarray(baseline, dtype=np.float64)
  regression = np.maximum(costs - baseline, 0.0).mean()
  return float(
    costs.mean()
    + tail_weight * np.percentile(costs, 90)
    + regression_weight * regression
  )


def evaluate(model, simulator_type, controller_type, policy, segments):
  records = []
  for segment in segments:
    controller = controller_type()
    controller.policy = policy() if callable(policy) else policy
    simulator = simulator_type(
      model, f"data/{segment}.csv", controller, debug=False
    )
    records.append({"segment": segment, **simulator.rollout()})
  return records


def costs(records):
  return np.asarray([record["total_cost"] for record in records])


def main(args):
  root = Path.cwd()
  official = (root / args.official_dir).resolve()
  source = (root / args.controller_source).resolve()
  inverse_model = (root / args.inverse_model).resolve()
  output = (root / args.output).resolve()
  shutil.copy2(source, official / "controllers" / "riyadh_general.py")
  os.environ["RIYADH_GENERAL_INVERSE_MODEL"] = str(inverse_model)
  os.environ["RIYADH_GENERAL_MODEL"] = str(
    official / "models" / "disabled-closed-loop-residual.npz"
  )
  sys.path.insert(0, str(official))
  train_segments = [
    f"{value:05d}" for value in range(args.train_start, args.train_start + args.train_count)
  ]
  validation_segments = [
    f"{value:05d}"
    for value in range(args.validation_start, args.validation_start + args.validation_count)
  ]
  overlap = set(train_segments) & set(validation_segments)
  if overlap:
    raise SystemExit(f"training and validation overlap: {sorted(overlap)}")

  with working_directory(official):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.riyadh_general import Controller, FEATURE_COUNT

    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    recorded = []
    baseline_train_records = []
    for segment in train_segments:
      controller = Controller()
      recorder = RecordingPolicy()
      controller.policy = recorder
      simulator = TinyPhysicsSimulator(
        model, f"data/{segment}.csv", controller, debug=False
      )
      baseline_train_records.append({"segment": segment, **simulator.rollout()})
      # Controller.update is called from row 20, while closed-loop control and
      # cost start at row 100.  Exclude the 80 ignored warmup calls so feature
      # normalization represents the policy's actual operating distribution.
      recorded.extend(recorder.features[80:])
    feature_array = np.asarray(recorded, dtype=np.float32)
    feature_mean = feature_array.mean(axis=0).astype(np.float32)
    feature_scale = np.maximum(
      feature_array.std(axis=0).astype(np.float32), 1e-4
    )
    if feature_mean.shape != (FEATURE_COUNT,):
      raise RuntimeError(f"bad recorded feature shape: {feature_mean.shape}")
    baseline_train_costs = costs(baseline_train_records)
    baseline_objective = objective(
      baseline_train_costs,
      baseline_train_costs,
      args.tail_weight,
      args.regression_weight,
    )

    rng = np.random.RandomState(args.seed)
    dimension = FEATURE_COUNT + 1
    center = np.zeros(dimension, dtype=np.float64)
    sigma = np.full(dimension, args.initial_sigma, dtype=np.float64)
    best_parameters = center.copy()
    best_costs = baseline_train_costs.copy()
    best_objective = baseline_objective
    generations = []
    print(
      f"CLOSED_LOOP_BASELINE mean={baseline_train_costs.mean():.9f} "
      f"objective={baseline_objective:.9f}", flush=True
    )
    for generation in range(args.generations):
      samples = [center.copy()]
      while len(samples) < args.population:
        direction = rng.randn(dimension)
        samples.append(np.clip(center + sigma * direction, -args.parameter_limit, args.parameter_limit))
        if len(samples) < args.population:
          samples.append(np.clip(center - sigma * direction, -args.parameter_limit, args.parameter_limit))
      evaluated = []
      for candidate_index, parameters in enumerate(samples):
        policy_factory = lambda p=parameters: LinearResidualPolicy(
          feature_mean, feature_scale, p, args.residual_limit,
          args.ood_gate_start, args.ood_gate_width,
        )
        records = evaluate(
          model, TinyPhysicsSimulator, Controller, policy_factory, train_segments
        )
        candidate_costs = costs(records)
        candidate_objective = objective(
          candidate_costs,
          baseline_train_costs,
          args.tail_weight,
          args.regression_weight,
        )
        evaluated.append((candidate_objective, parameters.copy(), candidate_costs))
        print(
          f"CLOSED_LOOP_TRIAL generation={generation + 1} "
          f"candidate={candidate_index + 1} objective={candidate_objective:.9f} "
          f"mean={candidate_costs.mean():.9f}", flush=True
        )
      evaluated.sort(key=lambda item: item[0])
      elite = evaluated[:args.elite]
      elite_parameters = np.asarray([item[1] for item in elite])
      center = elite_parameters.mean(axis=0)
      sigma = np.maximum(elite_parameters.std(axis=0), args.minimum_sigma)
      if elite[0][0] < best_objective:
        best_objective, best_parameters, best_costs = elite[0]
      generations.append({
        "generation": generation + 1,
        "best_objective": float(elite[0][0]),
        "best_mean_total_cost": float(elite[0][2].mean()),
        "sigma_mean": float(sigma.mean()),
      })

    def selected_policy():
      return LinearResidualPolicy(
        feature_mean, feature_scale, best_parameters, args.residual_limit,
        args.ood_gate_start, args.ood_gate_width,
      )

    baseline_validation_records = evaluate(
      model, TinyPhysicsSimulator, Controller, RecordingPolicy,
      validation_segments,
    )
    selected_validation_records = evaluate(
      model, TinyPhysicsSimulator, Controller, selected_policy,
      validation_segments,
    )

  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    linear_weights=best_parameters[:-1].astype(np.float32),
    linear_bias=np.asarray([best_parameters[-1]], dtype=np.float32),
    residual_limit=np.asarray([args.residual_limit], dtype=np.float32),
    ood_gate_start=np.asarray([args.ood_gate_start], dtype=np.float32),
    ood_gate_width=np.asarray([args.ood_gate_width], dtype=np.float32),
  )
  metadata = {
    "scope": "closed-loop-residual-training",
    "policy": "route-agnostic-linear-residual",
    "seed": args.seed,
    "train_segments": train_segments,
    "validation_segments": validation_segments,
    "configuration": {
      "population": args.population,
      "elite": args.elite,
      "generations": args.generations,
      "initial_sigma": args.initial_sigma,
      "minimum_sigma": args.minimum_sigma,
      "parameter_limit": args.parameter_limit,
      "residual_limit": args.residual_limit,
      "tail_weight": args.tail_weight,
      "regression_weight": args.regression_weight,
      "ood_gate_start": args.ood_gate_start,
      "ood_gate_width": args.ood_gate_width,
    },
    "baseline_train": summary(baseline_train_costs),
    "selected_train": summary(best_costs),
    "baseline_validation": summary(costs(baseline_validation_records)),
    "selected_validation": summary(costs(selected_validation_records)),
    "best_objective": float(best_objective),
    "baseline_objective": float(baseline_objective),
    "generations": generations,
    "baseline_validation_records": baseline_validation_records,
    "selected_validation_records": selected_validation_records,
    "model": str(output),
  }
  output.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
    "baseline_train": metadata["baseline_train"],
    "selected_train": metadata["selected_train"],
    "baseline_validation": metadata["baseline_validation"],
    "selected_validation": metadata["selected_validation"],
    "model": str(output),
  }, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--controller-source", default="compute/riyadh_general.py")
  parser.add_argument("--inverse-model", required=True)
  parser.add_argument("--output", default="checkpoints/general/models/v10-closed-loop.npz")
  parser.add_argument("--train-start", type=int, default=16000)
  parser.add_argument("--train-count", type=int, default=6)
  parser.add_argument("--validation-start", type=int, default=16200)
  parser.add_argument("--validation-count", type=int, default=12)
  parser.add_argument("--population", type=int, default=10)
  parser.add_argument("--elite", type=int, default=3)
  parser.add_argument("--generations", type=int, default=6)
  parser.add_argument("--initial-sigma", type=float, default=0.05)
  parser.add_argument("--minimum-sigma", type=float, default=0.01)
  parser.add_argument("--parameter-limit", type=float, default=0.30)
  parser.add_argument("--residual-limit", type=float, default=0.15)
  parser.add_argument("--tail-weight", type=float, default=0.10)
  parser.add_argument("--regression-weight", type=float, default=0.50)
  parser.add_argument("--ood-gate-start", type=float, default=float("inf"))
  parser.add_argument("--ood-gate-width", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=20260718)
  main(parser.parse_args())
