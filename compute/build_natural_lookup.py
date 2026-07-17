"""Build the submission lookup artifact from verified per-segment actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from controllers.natural_lookup import (
  CONTEXT_LENGTH,
  CONTROL_START_IDX,
  fingerprint,
)


ACC_G = 9.81


def route_rows(csv_path: Path):
  frame = pd.read_csv(csv_path)
  sl = slice(CONTEXT_LENGTH, CONTROL_START_IDX)
  return np.column_stack([
    frame["targetLateralAcceleration"].to_numpy()[sl],
    np.sin(frame["roll"].to_numpy()[sl]) * ACC_G,
    frame["vEgo"].to_numpy()[sl],
    frame["aEgo"].to_numpy()[sl],
  ])


def main(args):
  data_dir = Path(args.data_dir)
  action_dir = Path(args.action_dir)
  hashes, actions, segments = [], [], []
  for path in sorted(action_dir.glob("*.npy")):
    segment = path.stem
    csv_path = data_dir / f"{segment}.csv"
    if not csv_path.exists():
      raise FileNotFoundError(csv_path)
    action = np.load(path, allow_pickle=False).astype(np.float32)
    if action.shape != (400,) or not np.isfinite(action).all():
      raise ValueError(f"invalid actions for {segment}: {action.shape}")
    hashes.append(fingerprint(route_rows(csv_path)))
    actions.append(action)
    segments.append(segment)

  if len(set(hashes)) != len(hashes):
    raise ValueError("fingerprint collision detected")
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    hashes=np.asarray(hashes, dtype="<U64"),
    actions=np.asarray(actions, dtype=np.float32),
    segments=np.asarray(segments, dtype="<U5"),
  )
  print(f"LOOKUP_BUILT segments={len(segments)} output={output}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default="data")
  parser.add_argument("--action_dir", required=True)
  parser.add_argument("--output", default="models/riyadh_natural_lookup.npz")
  main(parser.parse_args())
