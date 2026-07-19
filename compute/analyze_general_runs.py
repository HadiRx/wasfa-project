"""Generate compact diagnostics for route-agnostic controller evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict[str, dict]:
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  records = payload.get("records", [])
  result = {str(item["segment"]): item for item in records}
  if not result or len(result) != len(records):
    raise ValueError(f"missing or duplicate records in {path}")
  return result


def metrics(records: dict[str, dict]) -> dict[str, float]:
  values = np.asarray([float(item["total_cost"]) for item in records.values()])
  return {
    "mean": float(values.mean()),
    "median": float(np.median(values)),
    "p90": float(np.percentile(values, 90)),
    "worst": float(values.max()),
  }


def summarize(baseline_path: str, candidate_path: str, top_n: int) -> dict:
  baseline = load(baseline_path)
  candidate = load(candidate_path)
  if set(baseline) != set(candidate):
    raise ValueError("baseline and candidate segment sets differ")

  rows = []
  for segment in sorted(baseline):
    before = float(baseline[segment]["total_cost"])
    after = float(candidate[segment]["total_cost"])
    rows.append({
      "segment": segment,
      "baseline_total_cost": before,
      "candidate_total_cost": after,
      "delta": after - before,
      "candidate_jerk_cost": float(candidate[segment].get("jerk_cost", 0.0)),
      "candidate_lataccel_cost": float(candidate[segment].get("lataccel_cost", 0.0)),
    })

  by_candidate = sorted(rows, key=lambda row: row["candidate_total_cost"], reverse=True)
  by_regression = sorted(rows, key=lambda row: row["delta"], reverse=True)
  by_improvement = sorted(rows, key=lambda row: row["delta"])
  deltas = np.asarray([row["delta"] for row in rows])

  return {
    "baseline": baseline_path,
    "candidate": candidate_path,
    "segments": len(rows),
    "baseline_metrics": metrics(baseline),
    "candidate_metrics": metrics(candidate),
    "improved_segments": int((deltas < -1e-9).sum()),
    "unchanged_segments": int((np.abs(deltas) <= 1e-9).sum()),
    "regressed_segments": int((deltas > 1e-9).sum()),
    "top_worst_candidate": by_candidate[:top_n],
    "top_regressions": by_regression[:top_n],
    "top_improvements": by_improvement[:top_n],
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", required=True)
  parser.add_argument("--candidate", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--top-n", type=int, default=10)
  args = parser.parse_args()

  report = summarize(args.baseline, args.candidate, args.top_n)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
