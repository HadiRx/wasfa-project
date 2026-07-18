"""Evaluate the general controller on explicitly held-out segments."""

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


def main(args):
  root = Path.cwd()
  official = (root / args.official_dir).resolve()
  source = (root / args.controller_source).resolve()
  metadata = None
  if args.metadata:
    metadata_path = (root / args.metadata).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  validation = metadata["validation_segments"] if metadata else []
  if args.unseen_count:
    validation = [
      f"{segment:05d}"
      for segment in range(args.unseen_start, args.unseen_start + args.unseen_count)
    ]
    teacher_segments = set()
    if metadata:
      teacher_segments = set(metadata["train_segments"]) | set(
        metadata["validation_segments"]
      )
    overlap = teacher_segments & set(validation)
    if overlap:
      raise SystemExit(f"unseen evaluation overlaps teacher segments: {sorted(overlap)}")
  elif args.segments:
    requested = [item.strip() for item in args.segments.split(",") if item.strip()]
    if metadata and not set(requested).issubset(validation):
      raise SystemExit("evaluation segments must be held out by the training split")
    validation = requested
  elif not validation:
    raise SystemExit("provide --metadata, --segments, or --unseen-count")

  controller_target = official / "controllers" / "riyadh_general.py"
  model_target = official / "models" / "riyadh_general_policy.npz"
  shutil.copy2(source, controller_target)
  if args.disable_model:
    os.environ["RIYADH_GENERAL_MODEL"] = str(
      official / "models" / "disabled-general-policy.npz"
    )
  else:
    model = (root / args.model).resolve()
    shutil.copy2(model, model_target)
  sys.path.insert(0, str(official))
  with working_directory(official):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.riyadh_general import Controller
    from controllers.pid import Controller as PIDController

    simulator_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    records = []
    pid_records = []
    for segment in validation:
      simulator = TinyPhysicsSimulator(
        simulator_model,
        f"data/{segment}.csv",
        Controller(),
        debug=False,
      )
      cost = simulator.rollout()
      records.append({"segment": segment, **cost})
      if not args.quiet:
        print(f"GENERAL_EVAL segment={segment} total={cost['total_cost']:.9f}")
      if args.compare_pid:
        pid_cost = TinyPhysicsSimulator(
          simulator_model,
          f"data/{segment}.csv",
          PIDController(),
          debug=False,
        ).rollout()
        pid_records.append({"segment": segment, **pid_cost})
        if not args.quiet:
          print(f"PID_EVAL segment={segment} total={pid_cost['total_cost']:.9f}")

  totals = np.asarray([record["total_cost"] for record in records])
  summary = {
    "scope": "held_out_segments",
    "controller_variant": "analytic_feedback" if args.disable_model else "residual_policy",
    "segments": len(records),
    "mean_total_cost": float(totals.mean()),
    "median_total_cost": float(np.median(totals)),
    "worst_total_cost": float(totals.max()),
    "records": records,
  }
  if pid_records:
    pid_totals = np.asarray([record["total_cost"] for record in pid_records])
    summary["pid_baseline"] = {
      "mean_total_cost": float(pid_totals.mean()),
      "median_total_cost": float(np.median(pid_totals)),
      "worst_total_cost": float(pid_totals.max()),
      "records": pid_records,
    }
    summary["mean_improvement_vs_pid_percent"] = float(
      (pid_totals.mean() - totals.mean()) / pid_totals.mean() * 100.0
    )
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if args.quiet:
    print(json.dumps({
      key: value for key, value in summary.items()
      if key not in {"records", "pid_baseline"}
    }, sort_keys=True))
  else:
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-dir", default="official")
  parser.add_argument("--controller-source", default="compute/riyadh_general.py")
  parser.add_argument("--model", default="models/riyadh_general_policy.npz")
  parser.add_argument("--metadata", default="")
  parser.add_argument("--disable-model", action="store_true")
  parser.add_argument("--compare-pid", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--segments", default="")
  parser.add_argument("--unseen-start", type=int, default=100)
  parser.add_argument("--unseen-count", type=int, default=0)
  parser.add_argument("--output", default="checkpoints/general/heldout-summary.json")
  main(parser.parse_args())
