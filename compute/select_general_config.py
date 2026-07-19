"""Select a robust controller configuration using development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path: str):
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  records = payload.get("records", [])
  if not records:
    raise ValueError(f"no records in {path}")
  values = np.asarray([float(item["total_cost"]) for item in records], dtype=np.float64)
  return payload, values


def metrics(values):
  return {
    "mean": float(values.mean()),
    "median": float(np.median(values)),
    "p90": float(np.percentile(values, 90)),
    "worst": float(values.max()),
  }


def parse_candidate(value):
  parts = value.split("=", 2)
  if len(parts) != 3:
    raise argparse.ArgumentTypeError("candidate must be LABEL=RESULT_JSON=CONFIG_JSON")
  return tuple(parts)


def main(args):
  _, baseline_values = load_run(args.baseline)
  base = metrics(baseline_values)
  candidates = []

  for label, result_path, config_path in args.candidate:
    _, values = load_run(result_path)
    if values.shape != baseline_values.shape:
      raise ValueError(f"record count mismatch for {label}")
    current = metrics(values)
    changes = {
      key: float((current[key] - base[key]) / base[key] * 100.0)
      for key in base
    }
    regressed = int(np.sum(values > baseline_values + args.segment_epsilon))
    checks = {
      "mean": changes["mean"] <= args.allow_mean_regression_percent,
      "median": changes["median"] <= args.allow_median_regression_percent,
      "p90": changes["p90"] <= args.allow_p90_regression_percent,
      "worst": changes["worst"] <= args.allow_worst_regression_percent,
      "regressed_segments": (
        args.max_regressed_segments < 0 or regressed <= args.max_regressed_segments
      ),
    }
    objective = float(
      current["mean"]
      + args.p90_weight * current["p90"]
      + args.worst_weight * current["worst"]
      + args.regression_weight * np.maximum(values - baseline_values, 0.0).mean()
    )
    candidates.append({
      "label": label,
      "result": result_path,
      "config": config_path,
      "metrics": current,
      "change_percent": changes,
      "regressed_segments": regressed,
      "checks": checks,
      "passed": all(checks.values()),
      "objective": objective,
    })

  passing = [item for item in candidates if item["passed"]]
  if not passing:
    report = {"baseline": args.baseline, "baseline_metrics": base, "candidates": candidates, "passed": False}
    Path(args.output_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit("no V11 configuration passed development gates")

  selected = min(passing, key=lambda item: (item["objective"], item["metrics"]["mean"]))
  config = json.loads(Path(selected["config"]).read_text(encoding="utf-8"))
  env_lines = [f"export {key}={json.dumps(str(value))}" for key, value in sorted(config.items())]
  Path(args.output_env).write_text("\n".join(env_lines) + "\n", encoding="utf-8")
  report = {
    "selection_scope": "development-only-v11",
    "baseline": args.baseline,
    "baseline_metrics": base,
    "candidates": candidates,
    "selected": selected,
    "selected_config": config,
    "passed": True,
    "weights": {
      "p90": args.p90_weight,
      "worst": args.worst_weight,
      "regression": args.regression_weight,
    },
  }
  Path(args.output_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({"selected": selected["label"], "config": config, "objective": selected["objective"]}, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", required=True)
  parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
  parser.add_argument("--output-env", required=True)
  parser.add_argument("--output-report", required=True)
  parser.add_argument("--allow-mean-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-median-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-p90-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-worst-regression-percent", type=float, default=0.0)
  parser.add_argument("--max-regressed-segments", type=int, default=-1)
  parser.add_argument("--segment-epsilon", type=float, default=1e-9)
  parser.add_argument("--p90-weight", type=float, default=0.15)
  parser.add_argument("--worst-weight", type=float, default=0.02)
  parser.add_argument("--regression-weight", type=float, default=1.0)
  main(parser.parse_args())
