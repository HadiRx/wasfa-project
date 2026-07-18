"""Package a trained residual with a fixed deployment output scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
  if not 0.0 < args.output_scale <= 1.0:
    raise SystemExit("output scale must be in (0, 1]")
  source = Path(args.source)
  output = Path(args.output)
  with np.load(source, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  required = {
    "feature_mean", "feature_scale", "linear_weights", "linear_bias",
    "residual_limit",
  }
  missing = required - set(arrays)
  if missing:
    raise SystemExit(f"source is not a linear residual model: {sorted(missing)}")
  arrays["output_scale"] = np.asarray([args.output_scale], dtype=np.float32)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **arrays)
  metadata = {
    "scope": "closed-loop-residual-deployment-package",
    "source": str(source),
    "source_sha256": digest(source),
    "output": str(output),
    "output_sha256": digest(output),
    "output_scale": args.output_scale,
  }
  output.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--output-scale", type=float, required=True)
  main(parser.parse_args())
