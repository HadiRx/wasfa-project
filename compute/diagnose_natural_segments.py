"""Diagnose saved Natural-20 steering checkpoints with the stock simulator."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from io import BytesIO
from pathlib import Path

import numpy as np

from tinyphysics import (
  CONTROL_START_IDX,
  COST_END_IDX,
  CONTEXT_LENGTH,
  DEL_T,
  TinyPhysicsModel,
  TinyPhysicsSimulator,
)


HORIZON = COST_END_IDX - CONTROL_START_IDX


class ReplayController:
  def __init__(self, actions):
    self.actions = np.asarray(actions, dtype=np.float32)
    if self.actions.shape != (HORIZON,):
      raise ValueError(f"expected {(HORIZON,)}, got {self.actions.shape}")
    self.step = CONTEXT_LENGTH

  def update(self, target, current, state, future_plan):
    step = self.step
    self.step += 1
    if CONTROL_START_IDX <= step < COST_END_IDX:
      return float(self.actions[step - CONTROL_START_IDX])
    return 0.0


def load_actions(actions_dir: Path, stem: str) -> np.ndarray:
  npy_path = actions_dir / f"{stem}.npy"
  b64_path = actions_dir / f"{stem}.npy.b64"
  if npy_path.exists():
    return np.load(npy_path).astype(np.float32)
  if b64_path.exists():
    raw = base64.b64decode(b64_path.read_text(encoding="utf-8"))
    return np.load(BytesIO(raw)).astype(np.float32)
  raise FileNotFoundError(f"missing checkpoint for {stem} in {actions_dir}")


def rolling_windows(steps, error, target, pred, action, data, width=20, limit=5):
  candidates = []
  for start in range(0, len(error) - width + 1):
    stop = start + width
    err = error[start:stop]
    score = float(np.mean(err ** 2))
    candidates.append((score, start, stop))
  selected = []
  occupied = np.zeros(len(error), dtype=bool)
  for score, start, stop in sorted(candidates, reverse=True):
    if occupied[start:stop].any():
      continue
    occupied[start:stop] = True
    selected.append({
      "start_step": int(steps[start]),
      "end_step": int(steps[stop - 1]),
      "rmse": float(np.sqrt(score)),
      "mean_abs_error": float(np.mean(np.abs(error[start:stop]))),
      "max_abs_error": float(np.max(np.abs(error[start:stop]))),
      "target_min": float(np.min(target[start:stop])),
      "target_max": float(np.max(target[start:stop])),
      "pred_min": float(np.min(pred[start:stop])),
      "pred_max": float(np.max(pred[start:stop])),
      "action_min": float(np.min(action[start:stop])),
      "action_max": float(np.max(action[start:stop])),
      "speed_mean": float(np.mean(data["v_ego"].values[start:stop])),
      "abs_roll_mean": float(np.mean(np.abs(data["roll_lataccel"].values[start:stop]))),
      "abs_a_ego_mean": float(np.mean(np.abs(data["a_ego"].values[start:stop]))),
    })
    if len(selected) == limit:
      break
  return selected


def classify(cost, max_row, windows):
  total = max(float(cost["total_cost"]), 1e-9)
  jerk_share = float(cost["jerk_cost"]) / total
  max_abs_target = abs(float(max_row["target_lataccel"]))
  speed = float(max_row["v_ego"])
  abs_roll = abs(float(max_row["roll_lataccel"]))
  if jerk_share >= 0.35:
    primary = "jerk_dominated"
  elif max_abs_target >= 2.0:
    primary = "sharp_lateral_tracking"
  elif speed >= 32.0:
    primary = "high_speed_tracking"
  elif abs_roll >= 0.5:
    primary = "roll_coupled_tracking"
  else:
    primary = "tracking_error"
  return {
    "primary_failure_mode": primary,
    "jerk_share": jerk_share,
    "worst_window_start": windows[0]["start_step"] if windows else None,
    "worst_window_end": windows[0]["end_step"] if windows else None,
  }


def diagnose_segment(data_dir: Path, model_path: str, actions_dir: Path, stem: str):
  actions = load_actions(actions_dir, stem)
  data_path = data_dir / f"{stem}.csv"
  if not data_path.exists():
    raise FileNotFoundError(data_path)

  model = TinyPhysicsModel(model_path, debug=False)
  sim = TinyPhysicsSimulator(
    model, str(data_path), ReplayController(actions), debug=False
  )
  cost = sim.rollout()
  target = np.asarray(sim.target_lataccel_history, dtype=np.float64)
  pred = np.asarray(sim.current_lataccel_history, dtype=np.float64)
  steps = np.arange(CONTROL_START_IDX, COST_END_IDX)
  scored = slice(CONTROL_START_IDX, COST_END_IDX)
  target_scored = target[scored]
  pred_scored = pred[scored]
  error = target_scored - pred_scored
  abs_error = np.abs(error)
  data = sim.data.iloc[CONTROL_START_IDX:COST_END_IDX].reset_index(drop=True)
  action = actions.astype(np.float64)
  jerk = np.concatenate([[np.nan], np.diff(pred_scored) / DEL_T])

  max_idx = int(np.argmax(abs_error))
  max_row = {
    "step": int(steps[max_idx]),
    "target_lataccel": float(target_scored[max_idx]),
    "current_lataccel": float(pred_scored[max_idx]),
    "error": float(error[max_idx]),
    "abs_error": float(abs_error[max_idx]),
    "action": float(action[max_idx]),
    "v_ego": float(data["v_ego"].iloc[max_idx]),
    "a_ego": float(data["a_ego"].iloc[max_idx]),
    "roll_lataccel": float(data["roll_lataccel"].iloc[max_idx]),
  }
  max_jerk_idx = int(np.nanargmax(np.abs(jerk)))
  max_jerk = {
    "step": int(steps[max_jerk_idx]),
    "jerk": float(jerk[max_jerk_idx]),
    "target_lataccel": float(target_scored[max_jerk_idx]),
    "current_lataccel": float(pred_scored[max_jerk_idx]),
  }
  windows = rolling_windows(
    steps, error, target_scored, pred_scored, action, data
  )
  summary = {
    "segment": stem,
    "stock_cost": float(cost["total_cost"]),
    "lataccel_cost": float(cost["lataccel_cost"]),
    "jerk_cost": float(cost["jerk_cost"]),
    "rmse_lataccel": float(np.sqrt(cost["lataccel_cost"] / 100.0)),
    "mean_abs_error": float(np.mean(abs_error)),
    "max_abs_error": float(np.max(abs_error)),
    "max_error": max_row,
    "max_jerk": max_jerk,
    "windows": windows,
  }
  summary.update(classify(cost, max_row, windows))
  rows = []
  for i, step in enumerate(steps):
    rows.append({
      "step": int(step),
      "target_lataccel": float(target_scored[i]),
      "current_lataccel": float(pred_scored[i]),
      "error": float(error[i]),
      "abs_error": float(abs_error[i]),
      "action": float(action[i]),
      "jerk": None if np.isnan(jerk[i]) else float(jerk[i]),
      "v_ego": float(data["v_ego"].iloc[i]),
      "a_ego": float(data["a_ego"].iloc[i]),
      "roll_lataccel": float(data["roll_lataccel"].iloc[i]),
    })
  return summary, rows


def write_csv(path: Path, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def md_table(rows, keys):
  lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
  for row in rows:
    values = []
    for key in keys:
      value = row.get(key)
      if isinstance(value, float):
        values.append(f"{value:.6f}")
      else:
        values.append(str(value))
    lines.append("| " + " | ".join(values) + " |")
  return "\n".join(lines)


def write_segment_markdown(path: Path, summary):
  windows = summary["windows"]
  path.parent.mkdir(parents=True, exist_ok=True)
  text = f"""# Segment {summary['segment']} diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/{summary['segment']}.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | {summary['stock_cost']:.9f} |
| lataccel_cost | {summary['lataccel_cost']:.9f} |
| jerk_cost | {summary['jerk_cost']:.9f} |
| rmse_lataccel | {summary['rmse_lataccel']:.9f} |
| mean_abs_error | {summary['mean_abs_error']:.9f} |
| max_abs_error | {summary['max_abs_error']:.9f} |
| jerk_share | {summary['jerk_share']:.6f} |

Primary failure mode: `{summary['primary_failure_mode']}`.

## Worst point

{md_table([summary['max_error']], ['step', 'target_lataccel', 'current_lataccel', 'error', 'abs_error', 'action', 'v_ego', 'a_ego', 'roll_lataccel'])}

## Largest jerk

{md_table([summary['max_jerk']], ['step', 'jerk', 'target_lataccel', 'current_lataccel'])}

## Worst 20-step windows

{md_table(windows, ['start_step', 'end_step', 'rmse', 'mean_abs_error', 'max_abs_error', 'target_min', 'target_max', 'pred_min', 'pred_max', 'action_min', 'action_max', 'speed_mean', 'abs_roll_mean', 'abs_a_ego_mean'])}
"""
  path.write_text(text, encoding="utf-8")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default="data")
  parser.add_argument("--model_path", default="models/tinyphysics.onnx")
  parser.add_argument("--actions_dir", default="checkpoints/pilot")
  parser.add_argument("--segments", default="00002,00003,00006,00008,00009,00010")
  parser.add_argument("--out_dir", default="knowledge/Diagnosis")
  parser.add_argument("--trace_dir", default="checkpoints/diagnosis")
  args = parser.parse_args()

  data_dir = Path(args.data_dir)
  actions_dir = Path(args.actions_dir)
  out_dir = Path(args.out_dir)
  trace_dir = Path(args.trace_dir)
  segments = [x.strip().zfill(5) for x in args.segments.split(",") if x.strip()]

  summaries = []
  for stem in segments:
    summary, rows = diagnose_segment(data_dir, args.model_path, actions_dir, stem)
    summaries.append(summary)
    write_csv(trace_dir / f"{stem}-trace.csv", rows)
    write_segment_markdown(out_dir / f"route-{stem}-analysis.md", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)

  ranked = sorted(summaries, key=lambda row: row["stock_cost"], reverse=True)
  aggregate = {
    "segments": segments,
    "mean_stock_cost": float(np.mean([row["stock_cost"] for row in summaries])),
    "worst_segment": ranked[0]["segment"],
    "worst_cost": ranked[0]["stock_cost"],
    "under_20": int(sum(row["stock_cost"] < 20.0 for row in summaries)),
    "summaries": summaries,
  }
  trace_dir.mkdir(parents=True, exist_ok=True)
  (trace_dir / "summary.json").write_text(
    json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  overview_rows = [{
    "segment": row["segment"],
    "stock_cost": row["stock_cost"],
    "rmse_lataccel": row["rmse_lataccel"],
    "max_abs_error": row["max_abs_error"],
    "jerk_share": row["jerk_share"],
    "primary_failure_mode": row["primary_failure_mode"],
    "worst_window_start": row["worst_window_start"],
    "worst_window_end": row["worst_window_end"],
  } for row in ranked]
  overview = f"""# Hard-segment diagnosis

Source: stock TinyPhysics rollouts using saved steering checkpoints in `checkpoints/pilot/*.npy.b64`.

{md_table(overview_rows, ['segment', 'stock_cost', 'rmse_lataccel', 'max_abs_error', 'jerk_share', 'primary_failure_mode', 'worst_window_start', 'worst_window_end'])}
"""
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "Hard-Segment-Diagnosis.md").write_text(overview, encoding="utf-8")


if __name__ == "__main__":
  main()
