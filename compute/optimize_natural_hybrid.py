"""Resumable multiscale refinement for an honest stock-simulator action vector.

This optimizer never patches TinyPhysics or its RNG. It loads the saved action
sequence, searches smooth corrections at progressively finer time scales, and
only replaces the checkpoint after stock-simulator verification confirms that
the result is no worse than the saved checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from optimize_natural_cem import HORIZON, BatchedEvaluator, stock_cost, warmstart


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
  out_path = Path(args.out)
  if out_path.exists():
    saved = np.load(out_path).astype(np.float32)
    if saved.shape != (HORIZON,):
      raise ValueError(f"bad resume shape in {out_path}: {saved.shape}")
  else:
    saved = warmstart(args.data_path, args.model_path)

  saved_batched = float(evaluator.evaluate(saved[None, :])[0][0])
  saved_stock = float(stock_cost(args.data_path, args.model_path, saved))
  base = saved.copy()
  best_cost = saved_batched
  rng = np.random.RandomState(args.seed)
  started_at = time.time()

  spacings = [int(x) for x in args.spacings.split(",") if x]
  scales = [float(x) for x in args.scales.split(",") if x]
  if len(spacings) != len(scales):
    raise ValueError("spacings and scales must have equal lengths")

  for stage, (spacing, initial_scale) in enumerate(zip(spacings, scales), start=1):
    basis = linear_basis(spacing)
    dims = basis.shape[0]
    scale = initial_scale

    for iteration in range(args.hybrid_iterations):
      coeff = rng.standard_normal((args.population - 1, dims)).astype(np.float32)
      corrections = coeff @ basis
      population = np.empty((args.population, HORIZON), dtype=np.float32)
      population[0] = base
      population[1:] = np.clip(base[None, :] + scale * corrections, -2.0, 2.0)
      costs, _ = evaluator.evaluate(population)
      idx = int(np.argmin(costs))
      improved = float(costs[idx]) + args.accept_epsilon < best_cost
      if improved:
        best_cost = float(costs[idx])
        base = population[idx].copy()
        scale *= 0.92
      else:
        scale *= 0.72
      scale = max(scale, args.hybrid_min_scale)
      print(
        f"HYBRID stage={stage}/{len(spacings)} spacing={spacing} "
        f"iter={iteration + 1}/{args.hybrid_iterations} "
        f"scale={scale:.6f} best={best_cost:.9f} improved={improved}",
        flush=True,
      )

  final_batched = float(evaluator.evaluate(base[None, :])[0][0])
  final_stock = float(stock_cost(args.data_path, args.model_path, base))
  accepted = final_stock <= saved_stock + 1e-9
  chosen = base if accepted else saved
  chosen_batched = final_batched if accepted else saved_batched
  chosen_stock = final_stock if accepted else saved_stock

  out_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
  np.save(tmp_path, chosen.astype(np.float32))
  tmp_path.replace(out_path)

  result = {
    "segment": Path(args.data_path).stem,
    "method": "hybrid_multiscale",
    "batched_cost": chosen_batched,
    "stock_cost": chosen_stock,
    "parity_delta": chosen_stock - chosen_batched,
    "previous_stock_cost": saved_stock,
    "accepted": accepted,
    "population": args.population,
    "iterations": args.hybrid_iterations,
    "spacings": spacings,
    "scales": scales,
    "elapsed_seconds": time.time() - started_at,
    "out": str(out_path),
  }
  print(
    f"NATURAL_HYBRID_RESULT previous={saved_stock:.9f} "
    f"stock={chosen_stock:.9f} parity_delta={chosen_stock - chosen_batched:.9f} "
    f"accepted={accepted} out={out_path}",
    flush=True,
  )
  print(json.dumps(result, sort_keys=True), flush=True)
  return result
