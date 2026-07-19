"""Select one deployment-scale candidate on development data only.

The selector applies the same zero-regression aggregate gates used by CI, then
chooses the passing candidate with the lowest mean total cost. The untouched
test set must only be evaluated after this script has frozen the selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from compare_general_runs import compare


def parse_candidate(value: str) -> tuple[float, Path, Path]:
  try:
    scale_text, report_text, model_text = value.split("=", 2)
    scale = float(scale_text)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
      "candidate must be SCALE=REPORT_JSON=MODEL_NPZ"
    ) from exc
  if not 0.0 < scale <= 1.0:
    raise argparse.ArgumentTypeError("candidate scale must be in (0, 1]")
  return scale, Path(report_text), Path(model_text)


def main(args) -> int:
  tolerances = {
    "mean_percent": args.allow_mean_regression_percent,
    "median_percent": args.allow_median_regression_percent,
    "worst_percent": args.allow_worst_regression_percent,
    "max_regressed_segments": args.max_regressed_segments,
    "segment_epsilon": args.segment_epsilon,
  }
  evaluated = []
  for scale, report_path, model_path in args.candidate:
    result = compare(args.baseline, report_path, tolerances)
    result["scale"] = scale
    result["model"] = str(model_path)
    evaluated.append(result)

  passing = [item for item in evaluated if item["passed"]]
  if not passing:
    payload = {
      "passed": False,
      "reason": "no development candidate passed the strict promotion gates",
      "baseline": str(args.baseline),
      "candidates": evaluated,
      "tolerances": tolerances,
    }
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2

  selected = min(
    passing,
    key=lambda item: (
      item["candidate_metrics"]["mean"],
      item["candidate_metrics"]["p90"] if "p90" in item["candidate_metrics"] else 0.0,
      item["candidate_metrics"]["worst"],
      item["scale"],
    ),
  )
  selected_model = Path(selected["model"])
  output_model = Path(args.output_model)
  output_model.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(selected_model, output_model)
  Path(args.output_scale).parent.mkdir(parents=True, exist_ok=True)
  Path(args.output_scale).write_text(f"{selected['scale']:.6g}\n", encoding="utf-8")

  payload = {
    "passed": True,
    "selection_scope": "development-only",
    "baseline": str(args.baseline),
    "selected_scale": selected["scale"],
    "selected_source_model": str(selected_model),
    "selected_output_model": str(output_model),
    "selected_metrics": selected["candidate_metrics"],
    "selected_change_percent": selected["change_percent"],
    "candidates": evaluated,
    "tolerances": tolerances,
  }
  Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
  Path(args.output_report).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", required=True, type=Path)
  parser.add_argument(
    "--candidate", action="append", required=True, type=parse_candidate,
    help="SCALE=REPORT_JSON=MODEL_NPZ; repeat for each candidate",
  )
  parser.add_argument("--output-model", required=True)
  parser.add_argument("--output-scale", required=True)
  parser.add_argument("--output-report", required=True)
  parser.add_argument("--allow-mean-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-median-regression-percent", type=float, default=0.0)
  parser.add_argument("--allow-worst-regression-percent", type=float, default=0.0)
  parser.add_argument("--max-regressed-segments", type=int, default=-1)
  parser.add_argument("--segment-epsilon", type=float, default=1e-9)
  raise SystemExit(main(parser.parse_args()))
