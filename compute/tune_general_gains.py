"""Coordinate-tune one global controller parameter set on training segments.

No route identifier or per-route parameter is exposed to the controller.  The
tuner evaluates the unmodified stock simulator and emits every accepted and
rejected global candidate for reproducibility.
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


SEARCH_SPACE = {
  "kp": [0.1755, 0.1950, 0.2145],
  "ki": [0.0850, 0.1000, 0.1150],
  "kd": [-0.0800, -0.0530, -0.0300],
  "kpreview": [0.0500, 0.1000, 0.1500],
  "kff": [-0.0300, 0.0000, 0.0300],
  "kroll": [-0.0300, 0.0000, 0.0300],
}

DEFAULTS = {
  "kp": 0.195,
  "ki": 0.100,
  "kd": -0.053,
  "kpreview": 0.100,
  "kff": 0.0,
  "kroll": 0.0,
}


def configure(controller, parameters):
  for name, value in parameters.items():
    setattr(controller, name, float(value))


def evaluate(model, simulator_type, controller_type, segments, parameters, tail_weight):
  records = []
  for segment in segments:
    controller = controller_type()
    configure(controller, parameters)
    simulator = simulator_type(
      model, f"data/{segment}.csv", controller, debug=False
    )
    records.append({"segment": segment, **simulator.rollout()})
  totals = np.asarray([record["total_cost"] for record in records])
  mean = float(totals.mean())
  p95 = float(np.percentile(totals, 95))
  return {
    "parameters": dict(parameters),
    "mean_total_cost": mean,
    "median_total_cost": float(np.median(totals)),
    "p95_total_cost": p95,
    "worst_total_cost": float(totals.max()),
    "objective": mean + tail_weight * p95,
    "records": records,
  }


def main(args):
  root = Path.cwd()
  official = (root / args.official_dir).resolve()
  source = (root / args.controller_source).resolve()
  shutil.copy2(source, official / "controllers" / "riyadh_general.py")
  sys.path.insert(0, str(official))
  segments = [
    f"{segment:05d}"
    for segment in range(args.start, args.start + args.count)
  ]
  parameters = dict(DEFAULTS)
  trials = []
  with working_directory(official):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.riyadh_general import Controller

    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    baseline = evaluate(
      model, TinyPhysicsSimulator, Controller, segments, parameters,
      args.tail_weight,
    )
    baseline["stage"] = "baseline"
    trials.append(baseline)
    print(
      f"GLOBAL_TUNE baseline mean={baseline['mean_total_cost']:.9f} "
      f"p95={baseline['p95_total_cost']:.9f} "
      f"objective={baseline['objective']:.9f}", flush=True
    )

    for pass_index in range(args.passes):
      for name, values in SEARCH_SPACE.items():
        candidates = []
        for value in values:
          candidate_parameters = dict(parameters)
          candidate_parameters[name] = value
          result = evaluate(
            model, TinyPhysicsSimulator, Controller, segments,
            candidate_parameters, args.tail_weight,
          )
          result.update({
            "stage": "coordinate",
            "pass": pass_index + 1,
            "parameter": name,
            "candidate": value,
          })
          trials.append(result)
          candidates.append(result)
          print(
            f"GLOBAL_TUNE pass={pass_index + 1} {name}={value:.6f} "
            f"mean={result['mean_total_cost']:.9f} "
            f"p95={result['p95_total_cost']:.9f} "
            f"objective={result['objective']:.9f}", flush=True
          )
        best = min(candidates, key=lambda item: item["objective"])
        parameters = dict(best["parameters"])
        print(
          f"GLOBAL_TUNE_ACCEPT {name}={parameters[name]:.6f} "
          f"objective={best['objective']:.9f}", flush=True
        )

  final = min(
    (trial for trial in trials if trial["parameters"] == parameters),
    key=lambda item: item["objective"],
  )
  summary = {
    "scope": "global-controller-train",
    "segments": segments,
    "tail_weight": args.tail_weight,
    "passes": args.passes,
    "selected_parameters": parameters,
    "selected_metrics": {
      key: final[key] for key in (
        "mean_total_cost", "median_total_cost", "p95_total_cost",
        "worst_total_cost", "objective",
      )
    },
    "trials": trials,
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print("GLOBAL_TUNE_DONE", json.dumps(summary["selected_parameters"], sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--controller-source", default="compute/riyadh_general.py")
  parser.add_argument("--start", type=int, default=400)
  parser.add_argument("--count", type=int, default=20)
  parser.add_argument("--passes", type=int, default=1)
  parser.add_argument("--tail-weight", type=float, default=0.05)
  parser.add_argument("--output", default="checkpoints/general/v2-train.json")
  main(parser.parse_args())
