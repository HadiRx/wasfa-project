"""Compare closed-loop controller runs and enforce robust promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path):
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  records = payload.get("records", [])
  result = {str(record["segment"]): float(record["total_cost"]) for record in records}
  if not result or len(result) != len(records):
    raise ValueError(f"missing or duplicate segment records in {path}")
  return result


def compare(baseline_path, candidate_path, tolerances):
  baseline = load_records(baseline_path)
  candidate = load_records(candidate_path)
  if set(baseline) != set(candidate):
    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    raise ValueError(f"segment mismatch: missing={missing}, extra={extra}")
  segments = sorted(baseline)
  before = np.asarray([baseline[segment] for segment in segments])
  after = np.asarray([candidate[segment] for segment in segments])
  regressed = after > before + tolerances["segment_epsilon"]

  def metric(values):
    return {
      "mean": float(values.mean()),
      "median": float(np.median(values)),
      "worst": float(values.max()),
    }

  baseline_metrics = metric(before)
  candidate_metrics = metric(after)
  changes = {
    name: float((candidate_metrics[name] - value) / value * 100.0)
    for name, value in baseline_metrics.items()
  }
  checks = {
    "mean": changes["mean"] <= tolerances["mean_percent"],
    "median": changes["median"] <= tolerances["median_percent"],
    "worst": changes["worst"] <= tolerances["worst_percent"],
    "regressed_segments": (
      tolerances["max_regressed_segments"] < 0
      or int(regressed.sum()) <= tolerances["max_regressed_segments"]
    ),
  }
  return {
    "baseline": str(baseline_path),
    "candidate": str(candidate_path),
    "segments": len(segments),
    "baseline_metrics": baseline_metrics,
    "candidate_metrics": candidate_metrics,
    "change_percent": changes,
    "regressed_segments": int(regressed.sum()),
    "worst_regression": {
      "segment": segments[int(np.argmax(after - before))],
      "delta": float(np.max(after - before)),
    },
    "checks": checks,
    "passed": all(checks.values()),
  }


def main(args):
  tolerances = {
    "mean_percent": args.allow_mean_regression_percent,
    "median_percent": args.allow_median_regression_percent,
    "worst_percent": args.allow_worst_regression_percent,
    "max_regressed_segments": args.max_regressed_segments,
    "segment_epsilon": args.segment_epsilon,
  }
  report = compare(args.baseline, args.candidate, tolerances)
  report["tolerances"] = tolerances
  serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
  print(serialized, end="")
  return 0 if report["passed"] else 2


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", required=True)
  parser.add_argument("--candidate", required=True)
  parser.add_argument("--output", default="")
  parser.add_argument("--allow-mean-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-median-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-worst-regression-percent", type=float, default=0.0)
  parser.add_argument("--max-regressed-segments", type=int, default=-1)
  parser.add_argument("--segment-epsilon", type=float, default=1e-9)
  raise SystemExit(main(parser.parse_args()))
