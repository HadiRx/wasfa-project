"""Fit a route-agnostic inverse model from public warmup telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def segment_examples(path):
  frame = pd.read_csv(path)
  target = frame["targetLateralAcceleration"].to_numpy(dtype=np.float64)
  roll = np.sin(frame["roll"].to_numpy(dtype=np.float64)) * 9.81
  speed = frame["vEgo"].to_numpy(dtype=np.float64)
  accel = frame["aEgo"].to_numpy(dtype=np.float64)
  action = -frame["steerCommand"].to_numpy(dtype=np.float64)
  index = np.arange(20, 80)
  net = target[index + 1] - roll[index + 1]
  features = np.column_stack([
    target[index], target[index + 1], target[index + 2], target[index + 5],
    target[index + 10], target[index + 20],
    target[index + 1] - target[index], target[index + 5] - target[index],
    roll[index], roll[index + 1], speed[index], accel[index],
    net / (speed[index] ** 2 + 1.0), net / (speed[index] + 1.0),
    np.abs(target[index + 1]), target[index + 1] * speed[index],
    roll[index] * speed[index],
  ])
  labels = action[index]
  if not np.isfinite(features).all() or not np.isfinite(labels).all():
    raise ValueError(f"non-finite warmup data in {path}")
  return features, labels


def load_split(data_dir, start, count):
  xs, ys = [], []
  for value in range(start, start + count):
    features, labels = segment_examples(data_dir / f"{value:05d}.csv")
    xs.append(features)
    ys.append(labels)
  return np.concatenate(xs), np.concatenate(ys)


def metrics(labels, predictions):
  residual = predictions - labels
  return {
    "samples": int(len(labels)),
    "rmse": float(np.sqrt(np.mean(residual ** 2))),
    "mae": float(np.mean(np.abs(residual))),
    "r2": float(1.0 - np.sum(residual ** 2) / np.sum((labels - labels.mean()) ** 2)),
  }


def main(args):
  data_dir = Path(args.data_dir)
  train_x, train_y = load_split(data_dir, args.train_start, args.train_count)
  validation_x, validation_y = load_split(
    data_dir, args.validation_start, args.validation_count
  )
  mean = train_x.mean(axis=0)
  scale = train_x.std(axis=0) + 1e-6
  design = np.column_stack([(train_x - mean) / scale, np.ones(len(train_x))])
  solution = np.linalg.solve(
    design.T @ design + args.ridge * np.eye(design.shape[1]),
    design.T @ train_y,
  )
  weights, bias = solution[:-1], solution[-1]
  train_prediction = ((train_x - mean) / scale) @ weights + bias
  validation_prediction = ((validation_x - mean) / scale) @ weights + bias
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    feature_mean=mean.astype(np.float32),
    feature_scale=scale.astype(np.float32),
    weights=weights.astype(np.float32),
    bias=np.asarray([bias], dtype=np.float32),
  )
  metadata = {
    "scope": "warmup-inverse-supervised",
    "train_range": [args.train_start, args.train_start + args.train_count],
    "validation_range": [
      args.validation_start, args.validation_start + args.validation_count
    ],
    "train_metrics": metrics(train_y, train_prediction),
    "validation_metrics": metrics(validation_y, validation_prediction),
    "model": str(output),
  }
  output.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", default="official/data")
  parser.add_argument("--train-start", type=int, default=0)
  parser.add_argument("--train-count", type=int, default=5000)
  parser.add_argument("--validation-start", type=int, default=10000)
  parser.add_argument("--validation-count", type=int, default=500)
  parser.add_argument("--ridge", type=float, default=0.01)
  parser.add_argument(
    "--output", default="checkpoints/general/models/warmup-inverse-linear-v1.npz"
  )
  main(parser.parse_args())
