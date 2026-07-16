"""Original multiscale action-space optimizer for the stock comma simulator.

The search starts with low-dimensional smooth steering corrections and then
progressively increases temporal resolution. Every accepted sequence is scored
with the same fixed-RNG rollout used by the unmodified simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimize_natural_cem import (
  HORIZON,
  BatchedEvaluator,
  stock_cost,
  warmstart,
)


def linear_basis(spacing: int) -> np.ndarray:
  knots = list(range(0, HORIZON, spacing))
  if knots[-1] != HORIZON - 1:
    knots.append(HORIZON - 1)
  knots = np.asarray(knots, dtype=np.float32)
  timeline = np.arange(HORIZON, dtype=np.float32)
  eye = np.eye(len(knots), dtype=np.float32)
  return np.stack([
    np.interp(timeline, knots, eye[i]).astype(np.float32)
    for i in range(len(knots))
  ])


def optimize(args):
  evaluator = BatchedEvaluator(args.data_path, args.model_path, args.provider)
  pid = warmstart(args.data_path, args.model_path)
  starts = np.stack([pid, np.zeros(HORIZON, dtype=np.float32)])
  start_costs, _ = evaluator.evaluate(starts)
  base = starts[int(np.argmin(start_costs))].copy()
  best_cost = float(np.min(start_costs))
  rng = np.random.RandomState(args.seed)

  spacings = [int(x) for x in args.spacings.split(",") if x]
  scales = [float(x) for x in args.scales.split(",") if x]
  if len(spacings) != len(scales):
    raise ValueError("spacings and scales must have equal lengths")

  for stage, (spacing, scale) in enumerate(zip(spacings, scales), start=1):
    basis = linear_basis(spacing)
    dims = basis.shape[0]
    mean = np.zeros(dims, dtype=np.float32)
    std = np.full(dims, scale, dtype=np.float32)
    elite_count = max(2, int(args.population * args.elite_fraction))

    for iteration in range(args.iterations):
      coeff = rng.normal(mean, std, size=(args.population, dims)).astype(np.float32)
      coeff[0] = 0.0
      population = np.clip(base[None, :] + coeff @ basis, -2.0, 2.0)
      costs, _ = evaluator.evaluate(population)
      order = np.argsort(costs)
      if float(costs[order[0]]) < best_cost:
        best_cost = float(costs[order[0]])
        base = population[order[0]].copy()
        # Re-center corrections around the newly accepted action.
        coeff = coeff - coeff[order[0]][None, :]
      elite = coeff[order[:elite_count]]
      mean = args.momentum * mean + (1.0 - args.momentum) * elite.mean(axis=0)
      std = args.momentum * std + (1.0 - args.momentum) * (
        elite.std(axis=0) + args.min_std
      )
      print(
        f"stage={stage}/{len(spacings)} spacing={spacing} dims={dims} "
        f"iter={iteration + 1}/{args.iterations} best={best_cost:.6f}",
        flush=True,
      )

  exact = float(stock_cost(args.data_path, args.model_path, base))
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.stem + ".tmp.npy")
  np.save(tmp, base.astype(np.float32))
  tmp.replace(out)
  print(
    f"NATURAL_MULTISCALE_RESULT batched={best_cost:.9f} stock={exact:.9f} "
    f"parity_delta={exact - best_cost:.9f} out={out}",
    flush=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default="data/00000.csv")
  parser.add_argument("--model_path", default="models/tinyphysics.onnx")
  parser.add_argument("--out", default="natural_actions/multiscale_00000.npy")
  parser.add_argument("--provider", default="CUDAExecutionProvider")
  parser.add_argument("--population", type=int, default=128)
  parser.add_argument("--iterations", type=int, default=8)
  parser.add_argument("--elite_fraction", type=float, default=0.10)
  parser.add_argument("--momentum", type=float, default=0.25)
  parser.add_argument("--min_std", type=float, default=0.002)
  parser.add_argument("--spacings", default="40,20,10,5,2")
  parser.add_argument("--scales", default="0.30,0.18,0.10,0.06,0.03")
  parser.add_argument("--seed", type=int, default=2026)
  optimize(parser.parse_args())
