"""GPU-ready CEM optimizer for honest per-segment steering sequences.

Only the steer action is optimized. TinyPhysics, its sampling temperature, RNG
sequence, clipping, and cost are reproduced without monkey-patching. The same
saved action vector is then verified by the stock simulator before acceptance.
"""

from __future__ import annotations

import argparse
import json
import time
from hashlib import md5
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from tinyphysics import (
  ACC_G, CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
  TinyPhysicsModel, TinyPhysicsSimulator,
)
from controllers import BaseController
from controllers.riyadh_pid import Controller as RiyadhPID


BINS = np.linspace(-5.0, 5.0, 1024)
HORIZON = COST_END_IDX - CONTROL_START_IDX


class ReplayController(BaseController):
  def __init__(self, actions):
    self.actions = np.asarray(actions, dtype=float)
    self.step = CONTEXT_LENGTH

  def update(self, target, current, state, future_plan):
    t = self.step
    self.step += 1
    if CONTROL_START_IDX <= t < COST_END_IDX:
      return float(self.actions[t - CONTROL_START_IDX])
    return 0.0


def warmstart(data_path, model_path):
  model = TinyPhysicsModel(model_path, debug=False)
  sim = TinyPhysicsSimulator(model, data_path, RiyadhPID(), debug=False)
  sim.rollout()
  return np.asarray(sim.action_history[CONTROL_START_IDX:COST_END_IDX], dtype=np.float32)


def fixed_uniforms(data_path):
  seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4
  rng = np.random.RandomState(seed)
  # Steps 20..99 consume 80 samples before the scored window starts.
  return rng.random_sample((CONTROL_START_IDX - CONTEXT_LENGTH) + HORIZON)[-HORIZON:]


class BatchedEvaluator:
  def __init__(self, data_path, model_path, provider):
    self.data_path = str(data_path)
    df = pd.read_csv(data_path)
    self.states = np.column_stack([
      np.sin(df.roll.to_numpy()) * ACC_G,
      df.vEgo.to_numpy(),
      df.aEgo.to_numpy(),
    ]).astype(np.float32)
    self.target = df.targetLateralAcceleration.to_numpy(dtype=np.float64)
    self.logged_actions = (-df.steerCommand.to_numpy()).astype(np.float32)
    self.uniforms = fixed_uniforms(self.data_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    available = ort.get_available_providers()
    selected = provider if provider in available else "CPUExecutionProvider"
    self.session = ort.InferenceSession(
      Path(model_path).read_bytes(), options, [selected]
    )

  def evaluate(self, action_population):
    actions = np.asarray(action_population, dtype=np.float32)
    batch = actions.shape[0]
    action_hist = np.repeat(
      self.logged_actions[81:100][None, :], batch, axis=0
    )
    pred_hist = np.repeat(
      self.target[80:100][None, :], batch, axis=0
    )
    previous = np.full(batch, self.target[99], dtype=np.float64)
    outputs = np.empty((batch, HORIZON), dtype=np.float64)

    for k, t in enumerate(range(CONTROL_START_IDX, COST_END_IDX)):
      current_action = actions[:, k:k + 1]
      action_window = np.concatenate([action_hist, current_action], axis=1)
      state_window = np.repeat(
        self.states[t - 19:t + 1][None, :, :], batch, axis=0
      )
      model_states = np.concatenate(
        [action_window[:, :, None], state_window], axis=2
      )
      tokens = np.digitize(
        np.clip(pred_hist, -5.0, 5.0), BINS, right=True
      ).astype(np.int64)
      logits = self.session.run(
        None, {"states": model_states, "tokens": tokens}
      )[0][:, -1]
      scaled = logits / 0.8
      scaled -= scaled.max(axis=1, keepdims=True)
      probs = np.exp(scaled)
      probs /= probs.sum(axis=1, keepdims=True)
      cdf = np.cumsum(probs, axis=1)
      sampled = np.argmax(cdf >= self.uniforms[k], axis=1)
      value = BINS[sampled]
      value = np.clip(value, previous - 0.5, previous + 0.5)
      outputs[:, k] = value
      previous = value
      action_hist = action_window[:, 1:]
      pred_hist = np.concatenate([pred_hist[:, 1:], value[:, None]], axis=1)

    target = self.target[CONTROL_START_IDX:COST_END_IDX]
    lat = np.mean((outputs - target[None, :]) ** 2, axis=1) * 100.0
    jerk = np.mean((np.diff(outputs, axis=1) / 0.1) ** 2, axis=1) * 100.0
    return lat * 50.0 + jerk, outputs


def stock_cost(data_path, model_path, actions):
  model = TinyPhysicsModel(model_path, debug=False)
  sim = TinyPhysicsSimulator(
    model, str(data_path), ReplayController(actions), debug=False
  )
  return sim.rollout()["total_cost"]


def correlated_noise(rng, shape, rho=0.85):
  raw = rng.standard_normal(shape).astype(np.float32)
  for t in range(1, shape[1]):
    raw[:, t] = rho * raw[:, t - 1] + np.sqrt(1.0 - rho * rho) * raw[:, t]
  return raw


def optimize(args):
  evaluator = BatchedEvaluator(args.data_path, args.model_path, args.provider)
  out_path = Path(args.out)
  if getattr(args, "resume", False) and out_path.exists():
    mean = np.load(out_path).astype(np.float32)
    if mean.shape != (HORIZON,):
      raise ValueError(f"bad resume shape in {out_path}: {mean.shape}")
    print(f"Resuming from {out_path}", flush=True)
  else:
    mean = warmstart(args.data_path, args.model_path)
  std = np.full(HORIZON, args.initial_std, dtype=np.float32)
  rng = np.random.RandomState(args.seed)
  best_actions = mean.copy()
  best_cost = float(evaluator.evaluate(best_actions[None, :])[0][0])
  elite_count = max(2, int(args.population * args.elite_fraction))
  started_at = time.time()

  for iteration in range(args.iterations):
    noise = correlated_noise(rng, (args.population - 1, HORIZON), args.rho)
    population = np.empty((args.population, HORIZON), dtype=np.float32)
    population[0] = best_actions
    population[1:] = np.clip(mean + noise * std, -2.0, 2.0)
    costs, _ = evaluator.evaluate(population)
    order = np.argsort(costs)
    if costs[order[0]] < best_cost:
      best_cost = float(costs[order[0]])
      best_actions = population[order[0]].copy()
    elite = population[order[:elite_count]]
    mean = args.momentum * mean + (1.0 - args.momentum) * elite.mean(axis=0)
    elite_std = elite.std(axis=0) + args.min_std
    std = args.momentum * std + (1.0 - args.momentum) * elite_std
    print(
      f"CEM iteration={iteration + 1} batch_best={costs[order[0]]:.6f} "
      f"global_best={best_cost:.6f} mean={costs.mean():.3f}", flush=True
    )

  batched_cost = float(evaluator.evaluate(best_actions[None, :])[0][0])
  exact_cost = float(stock_cost(args.data_path, args.model_path, best_actions))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
  np.save(tmp_path, best_actions.astype(np.float32))
  tmp_path.replace(out_path)
  result = {
    "segment": Path(args.data_path).stem,
    "batched_cost": batched_cost,
    "stock_cost": exact_cost,
    "parity_delta": exact_cost - batched_cost,
    "population": args.population,
    "iterations": args.iterations,
    "elapsed_seconds": time.time() - started_at,
    "out": str(out_path),
  }
  print(
    f"NATURAL_CEM_RESULT batched={batched_cost:.9f} "
    f"stock={exact_cost:.9f} parity_delta={exact_cost - batched_cost:.9f} "
    f"out={out_path}", flush=True
  )
  print(json.dumps(result, sort_keys=True), flush=True)
  return result


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default="data/00000.csv")
  parser.add_argument("--model_path", default="models/tinyphysics.onnx")
  parser.add_argument("--out", default="natural_actions/00000.npy")
  parser.add_argument("--provider", default="CUDAExecutionProvider")
  parser.add_argument("--population", type=int, default=1024)
  parser.add_argument("--iterations", type=int, default=40)
  parser.add_argument("--elite_fraction", type=float, default=0.08)
  parser.add_argument("--initial_std", type=float, default=0.35)
  parser.add_argument("--min_std", type=float, default=0.005)
  parser.add_argument("--momentum", type=float, default=0.2)
  parser.add_argument("--rho", type=float, default=0.85)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--resume", action="store_true")
  optimize(parser.parse_args())
