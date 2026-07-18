"""General causal controller for the comma Controls Challenge.

The policy uses only values supplied to ``Controller.update``.  It contains no
route identifiers, fingerprints, data paths, or per-segment action table.  A
small NumPy MLP predicts a residual on top of a stable feedback controller.
Training is performed offline by ``train_general_policy.py`` and exports a
portable ``.npz`` artifact.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
  from . import BaseController
except ImportError:  # Allows offline trainer imports before submission packaging.
  class BaseController:
    pass


CONTROL_CONTEXT_CALLS = 80
STEER_LIMIT = 2.0
FEATURE_COUNT = 16
DEFAULT_MODEL = (
  Path(__file__).resolve().parent.parent / "models" / "riyadh_general_policy.npz"
)


def _mean(values, count, fallback):
  array = np.asarray(values[:count], dtype=np.float64)
  return float(array.mean()) if array.size else float(fallback)


def policy_features(
  target_lataccel,
  current_lataccel,
  state,
  future_plan,
  error_integral,
  previous_error,
  previous_action,
):
  """Return route-agnostic features available at controller runtime."""
  target = float(target_lataccel)
  current = float(current_lataccel)
  error = target - current
  future_lat = list(future_plan.lataccel)
  future_roll = list(future_plan.roll_lataccel)
  preview_1 = float(future_lat[0]) if future_lat else target
  preview_5 = _mean(future_lat, 5, target)
  preview_10 = _mean(future_lat, 10, target)
  preview_20 = _mean(future_lat, 20, target)
  preview_40 = _mean(future_lat, 40, target)
  roll_10 = _mean(future_roll, 10, state.roll_lataccel)
  return np.asarray([
    target,
    current,
    error,
    error - float(previous_error),
    float(np.clip(error_integral, -8.0, 8.0)),
    float(state.roll_lataccel),
    float(state.v_ego) / 40.0,
    float(state.a_ego) / 5.0,
    float(previous_action),
    preview_1 - target,
    preview_5 - target,
    preview_10 - target,
    preview_20 - target,
    preview_40 - target,
    roll_10 - float(state.roll_lataccel),
    abs(error),
  ], dtype=np.float32)


def feedback_action(error, error_integral, error_diff, target, preview_delta, roll):
  """Official PID-equivalent fallback and residual-policy anchor."""
  return float(
    0.195 * error
    + 0.100 * error_integral
    - 0.053 * error_diff
  )


class ResidualMLP:
  def __init__(self, path):
    self.available = False
    if not Path(path).exists():
      return
    with np.load(path, allow_pickle=False) as archive:
      self.feature_mean = archive["feature_mean"].astype(np.float32)
      self.feature_scale = archive["feature_scale"].astype(np.float32)
      self.w1 = archive["w1"].astype(np.float32)
      self.b1 = archive["b1"].astype(np.float32)
      self.w2 = archive["w2"].astype(np.float32)
      self.b2 = archive["b2"].astype(np.float32)
      self.w3 = archive["w3"].astype(np.float32)
      self.b3 = archive["b3"].astype(np.float32)
      residual_limit = archive["residual_limit"]
      self.residual_limit = float(np.asarray(residual_limit).reshape(-1)[0])
    if self.feature_mean.shape != (FEATURE_COUNT,):
      raise ValueError("general policy feature count mismatch")
    if self.w1.shape[0] != FEATURE_COUNT or self.w3.shape[1] != 1:
      raise ValueError("invalid general policy weights")
    self.available = True

  def predict(self, features):
    if not self.available:
      return 0.0
    x = (features - self.feature_mean) / self.feature_scale
    x = np.tanh(x @ self.w1 + self.b1)
    x = np.tanh(x @ self.w2 + self.b2)
    residual = float((x @ self.w3 + self.b3)[0])
    return float(np.clip(residual, -self.residual_limit, self.residual_limit))


class Controller(BaseController):
  def __init__(self):
    model_path = Path(os.getenv("RIYADH_GENERAL_MODEL", str(DEFAULT_MODEL)))
    self.policy = ResidualMLP(model_path)
    self.calls = 0
    self.error_integral = 0.0
    self.previous_error = 0.0
    self.previous_action = 0.0
    self.kp = float(os.getenv("RIYADH_GENERAL_KP", "0.195"))
    self.ki = float(os.getenv("RIYADH_GENERAL_KI", "0.100"))
    self.kd = float(os.getenv("RIYADH_GENERAL_KD", "-0.053"))
    self.kff = float(os.getenv("RIYADH_GENERAL_KFF", "0.0"))
    self.kpreview = float(os.getenv("RIYADH_GENERAL_KPREVIEW", "0.10"))
    self.kroll = float(os.getenv("RIYADH_GENERAL_KROLL", "0.0"))
    self.integral_decay = float(os.getenv("RIYADH_GENERAL_IDECAY", "1.0"))
    self.integral_limit = float(os.getenv("RIYADH_GENERAL_ILIMIT", "1000000.0"))
    self.action_delta_limit = float(os.getenv("RIYADH_GENERAL_ACTION_DELTA", "4.0"))
    self.residual_scale = float(os.getenv("RIYADH_GENERAL_RESIDUAL_SCALE", "1.0"))

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = float(target_lataccel - current_lataccel)
    self.error_integral = float(np.clip(
      self.integral_decay * self.error_integral + error,
      -self.integral_limit,
      self.integral_limit,
    ))
    features = policy_features(
      target_lataccel,
      current_lataccel,
      state,
      future_plan,
      self.error_integral,
      self.previous_error,
      self.previous_action,
    )
    error_diff = error - self.previous_error
    base = float(
      self.kp * error
      + self.ki * self.error_integral
      + self.kd * error_diff
      + self.kff * float(target_lataccel)
      + self.kpreview * float(features[11])
      + self.kroll * float(state.roll_lataccel)
    )
    requested = base + self.residual_scale * self.policy.predict(features)

    action = float(np.clip(
      requested,
      self.previous_action - self.action_delta_limit,
      self.previous_action + self.action_delta_limit,
    ))
    action = float(np.clip(action, -STEER_LIMIT, STEER_LIMIT))
    self.previous_error = error
    self.previous_action = action
    self.calls += 1
    return action
