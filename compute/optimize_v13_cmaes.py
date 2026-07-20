"""Low-dimensional direct policy search over the frozen V10 controller.

CMA-ES optimizes the official TinyPhysics closed-loop objective rather than
teacher-action imitation. Search subsets rotate between generations, while a
fixed monitor split selects the final search candidate. Development and holdout
splits remain outside this script and are evaluated by the workflow gates.
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
from cmaes import CMA


@contextmanager
def working_directory(path):
  previous = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(previous)


PARAMETERS = (
  # name, baseline, search scale, lower bound, upper bound
  ("kp", 0.195, 0.055, 0.08, 0.34),
  ("ki", 0.100, 0.050, 0.01, 0.22),
  ("kd", -0.053, 0.050, -0.18, 0.08),
  ("kpreview", 0.100, 0.070, 0.00, 0.28),
  ("inverse_scale", 0.500, 0.180, 0.05, 0.95),
  ("action_delta_limit", 4.000, 1.000, 0.75, 4.00),
  ("output_scale", 1.000, 0.180, 0.45, 1.00),
)


def decode(vector):
  values = {}
  for raw, (name, baseline, scale, lower, upper) in zip(vector, PARAMETERS):
    values[name] = float(np.clip(baseline + scale * float(raw), lower, upper))
  return values


def robust_objective(candidate, baseline, regression_weight, worst_weight):
  candidate = np.asarray(candidate, dtype=np.float64)
  baseline = np.asarray(baseline, dtype=np.float64)
  regressions = np.maximum(candidate - baseline, 0.0)
  return float(
    candidate.mean()
    + 0.25 * np.percentile(candidate, 90)
    + 0.10 * candidate.max()
    + regression_weight * regressions.mean()
    + worst_weight * regressions.max()
  )


def summarize(candidate, baseline):
  candidate = np.asarray(candidate, dtype=np.float64)
  baseline = np.asarray(baseline, dtype=np.float64)
  delta = candidate - baseline
  return {
    "mean": float(candidate.mean()),
    "median": float(np.median(candidate)),
    "p90": float(np.percentile(candidate, 90)),
    "worst": float(candidate.max()),
    "improved_segments": int(np.sum(delta < -1e-9)),
    "regressed_segments": int(np.sum(delta > 1e-9)),
    "worst_regression": float(max(delta.max(), 0.0)),
    "mean_delta_vs_v10": float(delta.mean()),
  }


def main(args):
  root = Path.cwd()
  official = (root / args.official_dir).resolve()
  source = (root / args.controller_source).resolve()
  base_model = (root / args.base_model).resolve()
  inverse_model = (root / args.inverse_model).resolve()
  output_model = (root / args.output_model).resolve()
  output_report = (root / args.output_report).resolve()

  shutil.copy2(source, official / "controllers" / "riyadh_general.py")
  os.environ["RIYADH_GENERAL_MODEL"] = str(base_model)
  os.environ["RIYADH_GENERAL_INVERSE_MODEL"] = str(inverse_model)
  sys.path.insert(0, str(official))

  search_pool = [f"{x:05d}" for x in range(args.search_start, args.search_start + args.search_count)]
  monitor_segments = [f"{x:05d}" for x in range(args.monitor_start, args.monitor_start + args.monitor_count)]
  if set(search_pool) & set(monitor_segments):
    raise SystemExit("V13 search and monitor splits overlap")

  rng = np.random.RandomState(args.seed)
  optimizer = CMA(
    mean=np.zeros(len(PARAMETERS), dtype=np.float64),
    sigma=args.sigma,
    population_size=args.population_size,
    seed=args.seed,
    bounds=np.asarray([[-2.5, 2.5]] * len(PARAMETERS), dtype=np.float64),
  )

  with working_directory(official):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.riyadh_general import Controller

    physics = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    baseline_cache = {}

    def rollout(segment, params=None):
      controller = Controller()
      if params:
        controller.kp = params["kp"]
        controller.ki = params["ki"]
        controller.kd = params["kd"]
        controller.kpreview = params["kpreview"]
        controller.inverse_scale = params["inverse_scale"]
        controller.action_delta_limit = params["action_delta_limit"]
        controller.policy.output_scale = params["output_scale"]
      return TinyPhysicsSimulator(
        physics, f"data/{segment}.csv", controller, debug=False
      ).rollout()["total_cost"]

    def baseline_costs(segments):
      result = []
      for segment in segments:
        if segment not in baseline_cache:
          baseline_cache[segment] = float(rollout(segment))
        result.append(baseline_cache[segment])
      return np.asarray(result, dtype=np.float64)

    def candidate_costs(vector, segments):
      params = decode(vector)
      return np.asarray([rollout(segment, params) for segment in segments], dtype=np.float64)

    monitor_baseline = baseline_costs(monitor_segments)
    best_vector = np.zeros(len(PARAMETERS), dtype=np.float64)
    best_monitor_costs = monitor_baseline.copy()
    best_monitor_objective = robust_objective(
      monitor_baseline, monitor_baseline, args.regression_weight, args.worst_weight
    )
    generations = []

    for generation in range(args.generations):
      if args.subset_count >= len(search_pool):
        subset = list(search_pool)
      else:
        start = (generation * args.subset_count) % len(search_pool)
        subset = [search_pool[(start + i) % len(search_pool)] for i in range(args.subset_count)]
        # Periodically reshuffle the pool while preserving deterministic replay.
        if generation and generation % max(1, len(search_pool) // args.subset_count) == 0:
          rng.shuffle(search_pool)
      subset_baseline = baseline_costs(subset)
      solutions = []
      generation_records = []
      for _ in range(args.population_size):
        vector = optimizer.ask()
        costs = candidate_costs(vector, subset)
        objective = robust_objective(
          costs, subset_baseline, args.regression_weight, args.worst_weight
        )
        solutions.append((vector, objective))
        generation_records.append({
          "objective": objective,
          "parameters": decode(vector),
          "metrics": summarize(costs, subset_baseline),
        })
      optimizer.tell(solutions)
      generation_records.sort(key=lambda item: item["objective"])
      generation_best_vector = min(solutions, key=lambda item: item[1])[0]
      generation_monitor_costs = candidate_costs(generation_best_vector, monitor_segments)
      generation_monitor_objective = robust_objective(
        generation_monitor_costs, monitor_baseline,
        args.regression_weight, args.worst_weight,
      )
      if generation_monitor_objective < best_monitor_objective:
        best_monitor_objective = generation_monitor_objective
        best_vector = generation_best_vector.copy()
        best_monitor_costs = generation_monitor_costs.copy()
      record = {
        "generation": generation + 1,
        "search_segments": subset,
        "generation_best": generation_records[0],
        "monitor_objective": generation_monitor_objective,
        "monitor_metrics": summarize(generation_monitor_costs, monitor_baseline),
        "global_best_monitor_objective": best_monitor_objective,
      }
      generations.append(record)
      print(
        "V13_GENERATION "
        f"generation={generation + 1} "
        f"search_best={generation_records[0]['objective']:.6f} "
        f"monitor={generation_monitor_objective:.6f} "
        f"global_monitor={best_monitor_objective:.6f}",
        flush=True,
      )

  selected_params = decode(best_vector)
  with np.load(base_model, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  for name in ("kp", "ki", "kd", "kpreview", "inverse_scale", "action_delta_limit"):
    arrays[f"controller_{name}"] = np.asarray([selected_params[name]], dtype=np.float32)
  arrays["output_scale"] = np.asarray([selected_params["output_scale"]], dtype=np.float32)
  output_model.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_model, **arrays)

  report = {
    "scope": "v13-low-dimensional-cmaes-direct-policy-search",
    "seed": args.seed,
    "parameterization": [
      {"name": n, "baseline": b, "scale": s, "lower": lo, "upper": hi}
      for n, b, s, lo, hi in PARAMETERS
    ],
    "optimizer": {
      "population_size": args.population_size,
      "generations": args.generations,
      "sigma": args.sigma,
      "subset_count": args.subset_count,
    },
    "objective": {
      "formula": "mean + 0.25*p90 + 0.10*worst + regression_weight*mean_positive_delta + worst_weight*max_positive_delta",
      "regression_weight": args.regression_weight,
      "worst_weight": args.worst_weight,
    },
    "search_pool": search_pool,
    "monitor_segments": monitor_segments,
    "selected_parameters": selected_params,
    "monitor_baseline": summarize(monitor_baseline, monitor_baseline),
    "monitor_candidate": summarize(best_monitor_costs, monitor_baseline),
    "monitor_objective_baseline": robust_objective(
      monitor_baseline, monitor_baseline, args.regression_weight, args.worst_weight
    ),
    "monitor_objective_candidate": best_monitor_objective,
    "generations": generations,
    "output_model": str(output_model),
  }
  output_report.parent.mkdir(parents=True, exist_ok=True)
  output_report.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
    "selected_parameters": selected_params,
    "monitor_baseline": report["monitor_baseline"],
    "monitor_candidate": report["monitor_candidate"],
    "output_model": str(output_model),
  }, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--controller-source", default="compute/riyadh_general.py")
  parser.add_argument("--base-model", required=True)
  parser.add_argument("--inverse-model", required=True)
  parser.add_argument("--output-model", default="checkpoints/general/models/v13-cmaes.npz")
  parser.add_argument("--output-report", default="checkpoints/general/v13-cmaes-search.json")
  parser.add_argument("--search-start", type=int, default=13200)
  parser.add_argument("--search-count", type=int, default=120)
  parser.add_argument("--monitor-start", type=int, default=13400)
  parser.add_argument("--monitor-count", type=int, default=40)
  parser.add_argument("--subset-count", type=int, default=24)
  parser.add_argument("--population-size", type=int, default=10)
  parser.add_argument("--generations", type=int, default=10)
  parser.add_argument("--sigma", type=float, default=0.35)
  parser.add_argument("--regression-weight", type=float, default=0.60)
  parser.add_argument("--worst-weight", type=float, default=0.20)
  parser.add_argument("--seed", type=int, default=20260720)
  main(parser.parse_args())
