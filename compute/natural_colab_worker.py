"""Resumable Google Colab worker for honest per-segment optimization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import onnxruntime as ort

from optimize_natural_cem import optimize as optimize_cem
from optimize_natural_hybrid import optimize as optimize_hybrid


def append_jsonl(path: Path, record: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")


def main(args) -> None:
  available = ort.get_available_providers()
  if args.provider not in available:
    raise SystemExit(
      f"{args.provider} is unavailable. Providers={available}. "
      "In Colab select Runtime > Change runtime type > GPU, then rerun setup."
    )

  data_dir = Path(args.data_dir)
  out_dir = Path(args.out_dir)
  log_path = out_dir / "progress.jsonl"
  deadline = time.time() + args.time_budget_minutes * 60
  completed = 0

  for segment_id in range(args.start, args.end):
    if time.time() >= deadline:
      print("Time budget reached safely between segments.", flush=True)
      break
    stem = f"{segment_id:05d}"
    data_path = data_dir / f"{stem}.csv"
    out_path = out_dir / f"{stem}.npy"
    if not data_path.exists():
      raise FileNotFoundError(data_path)
    if out_path.exists() and not args.refine:
      print(f"SKIP completed segment {stem}", flush=True)
      continue

    config = SimpleNamespace(
      data_path=str(data_path),
      model_path=args.model_path,
      out=str(out_path),
      provider=args.provider,
      population=args.population,
      iterations=args.iterations,
      elite_fraction=args.elite_fraction,
      initial_std=args.initial_std,
      min_std=args.min_std,
      momentum=args.momentum,
      rho=args.rho,
      seed=args.seed + segment_id,
      resume=args.refine,
      hybrid_iterations=args.hybrid_iterations,
      hybrid_min_scale=args.hybrid_min_scale,
      accept_epsilon=args.accept_epsilon,
      spacings=args.spacings,
      scales=args.scales,
    )
    try:
      optimizer = optimize_hybrid if args.method == "hybrid" else optimize_cem
      result = optimizer(config)
      result["status"] = "completed"
      append_jsonl(log_path, result)
      completed += 1
    except Exception as exc:
      append_jsonl(log_path, {
        "segment": stem,
        "status": "failed",
        "method": args.method,
        "error": f"{type(exc).__name__}: {exc}",
        "time": time.time(),
      })
      raise

  print(
    f"COLAB_WORKER_DONE method={args.method} newly_completed={completed} out={out_dir}",
    flush=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default="data")
  parser.add_argument("--model_path", default="models/tinyphysics.onnx")
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--end", type=int, default=5000)
  parser.add_argument("--time_budget_minutes", type=int, default=600)
  parser.add_argument("--provider", default="CUDAExecutionProvider")
  parser.add_argument("--method", choices=["cem", "hybrid"], default="cem")
  parser.add_argument("--population", type=int, default=512)
  parser.add_argument("--iterations", type=int, default=30)
  parser.add_argument("--elite_fraction", type=float, default=0.08)
  parser.add_argument("--initial_std", type=float, default=0.18)
  parser.add_argument("--min_std", type=float, default=0.003)
  parser.add_argument("--momentum", type=float, default=0.25)
  parser.add_argument("--rho", type=float, default=0.92)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--hybrid_iterations", type=int, default=12)
  parser.add_argument("--hybrid_min_scale", type=float, default=0.001)
  parser.add_argument("--accept_epsilon", type=float, default=1e-9)
  parser.add_argument("--spacings", default="80,40,20,10,5,2")
  parser.add_argument("--scales", default="0.030,0.025,0.020,0.015,0.010,0.0075")
  parser.add_argument("--refine", action="store_true")
  main(parser.parse_args())
