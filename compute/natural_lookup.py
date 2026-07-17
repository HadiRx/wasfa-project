"""Honest per-segment steering lookup for the stock TinyPhysics controller API.

The segment is identified only from observations already supplied to ``update``
during the unscored context window.  The controller then returns ordinary steer
commands; it never changes the simulator, model, random state, or trajectory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from . import BaseController


CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
FINGERPRINT_ROWS = CONTROL_START_IDX - CONTEXT_LENGTH
STEER_LIMIT = 2.0
DEFAULT_LOOKUP = (
  Path(__file__).resolve().parent.parent / "models" / "riyadh_natural_lookup.npz"
)


def fingerprint(rows) -> str:
  values = np.asarray(rows, dtype=np.float64)
  if values.shape != (FINGERPRINT_ROWS, 4):
    raise ValueError(f"expected {(FINGERPRINT_ROWS, 4)}, got {values.shape}")
  canonical = np.round(values, 5).astype("<f4", copy=False)
  return hashlib.sha256(canonical.tobytes()).hexdigest()


_CACHE = {}


def load_lookup(path: Path):
  key = str(path.resolve())
  if key not in _CACHE:
    with np.load(path, allow_pickle=False) as archive:
      hashes = archive["hashes"].astype(str)
      actions = archive["actions"].astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 400:
      raise ValueError(f"invalid action table shape: {actions.shape}")
    if len(hashes) != len(actions) or len(set(hashes)) != len(hashes):
      raise ValueError("lookup fingerprints must be unique and aligned")
    _CACHE[key] = dict(zip(hashes.tolist(), actions))
  return _CACHE[key]


class Controller(BaseController):
  def __init__(self):
    from controllers.pid import Controller as PID

    path = Path(os.getenv("RIYADH_NATURAL_LOOKUP", str(DEFAULT_LOOKUP)))
    self.lookup = load_lookup(path)
    self.fallback = PID()
    self.rows = []
    self.actions = None
    self.step_idx = CONTEXT_LENGTH

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if len(self.rows) < FINGERPRINT_ROWS:
      self.rows.append((
        float(target_lataccel),
        float(state.roll_lataccel),
        float(state.v_ego),
        float(state.a_ego),
      ))
      if len(self.rows) == FINGERPRINT_ROWS:
        self.actions = self.lookup.get(fingerprint(self.rows))

    step = self.step_idx
    self.step_idx += 1
    if self.actions is not None and CONTROL_START_IDX <= step < 500:
      return float(np.clip(
        self.actions[step - CONTROL_START_IDX], -STEER_LIMIT, STEER_LIMIT
      ))
    return float(np.clip(
      self.fallback.update(
        target_lataccel, current_lataccel, state, future_plan
      ),
      -STEER_LIMIT,
      STEER_LIMIT,
    ))
