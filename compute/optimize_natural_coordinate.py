"""Protected coordinate-pattern refinement for stock TinyPhysics actions.

The method resumes from a saved 400-action checkpoint. At each temporal scale
it evaluates positive and negative moves along every basis direction, accepts
only the best strict improvement, and finally verifies the chosen checkpoint
with the unmodified stock simulator.
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


def evaluate_chunks(evaluator, population, batch_size):
  costs = []
  for start in range(0, len(population), batch_size):
    stop = min(start + batch_size, len(population))
    chunk_costs, _ = evaluator.evaluate(population[start:stop])
    costs.append(chunk_costs)
  return np.concatenate(costs)


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
  started_at = time.time()

  spacings = [int(x) for x in args.coordinate_spacings.split(",") if x]
  deltas = [float(x) for x in args.coordinate_deltas.split(",") if x]
  if len(spacings) != len(deltas):
    raise ValueError("coordinate spacings and deltas must have equal lengths")

  accepted_moves = 0
  for stage, (spacing, initial_delta) in enumerate(zip(spacings, deltas), start=1):
    basis = linear_basis(spacing)
    delta = initial_delta
    for iteration in range(args.coordinate_iterations):
      plus = np.clip(base[None, :] + delta * basis, -2.0, 2.0)
      minus = np.clip(base[None, :] - delta * basis, -2.0, 2.0)
      population = np.concatenate([base[None, :], plus, minus], axis=0)
      costs = evaluate_chunks(
        evaluator, population, args.coordinate_batch_size
      )
      idx = int(np.argmin(costs))
      improved = float(costs[idx]) + args.accept_epsilon < best_cost
      if improved:
        best_cost = float(costs[idx])
        base = population[idx].copy()
        accepted_moves += 1
        delta = max(delta * 0.95, args.coordinate_min_delta)
      else:
        delta *= 0.5
      print(
        f"COORD stage={stage}/{len(spacings)} spacing={spacing} "
        f"iter={iteration + 1}/{args.coordinate_iterations} "
        f"delta={delta:.7f} best={best_cost:.9f} improved={improved}",
        flush=True,
      )
      if delta < args.coordinate_min_delta:
        break

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
    "method": "coordinate_pattern",
    "batched_cost": chosen_batched,
    "stock_cost": chosen_stock,
    "parity_delta": chosen_stock - chosen_batched,
    "previous_stock_cost": saved_stock,
    "accepted": accepted,
    "accepted_moves": accepted_moves,
    "spacings": spacings,
    "deltas": deltas,
    "iterations": args.coordinate_iterations,
    "elapsed_seconds": time.time() - started_at,
    "out": str(out_path),
  }
  print(
    f"NATURAL_COORD_RESULT previous={saved_stock:.9f} "
    f"stock={chosen_stock:.9f} parity_delta={chosen_stock - chosen_batched:.9f} "
    f"moves={accepted_moves} accepted={accepted} out={out_path}",
    flush=True,
  )
  print(json.dumps(result, sort_keys=True), flush=True)
  return result
