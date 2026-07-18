"""Route-agnostic causal controller for comma Controls Challenge v2.

The controller uses only ``Controller.update`` inputs and causal memory.  It
contains no route identifier, fingerprint, data path, or action lookup table.
V8 blends stable PID-plus-preview feedback with a compact inverse feedforward
model trained on the public warmup rows. V10 optionally adds a bounded linear
residual trained on closed-loop rollouts and packaged with a conservative
deployment scale. With no residual artifact, behavior remains V8-compatible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
  from . import BaseController
except ImportError:
  class BaseController:
    pass


STEER_LIMIT = 2.0
FEATURE_COUNT = 16
INVERSE_FEATURE_COUNT = 17
DEFAULT_MODEL = (
  Path(__file__).resolve().parent.parent / "models" / "riyadh_general_policy.npz"
)
DEFAULT_INVERSE_MODEL = (
  Path(__file__).resolve().parent.parent / "models" / "riyadh_inverse_linear.npz"
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


def inverse_features(target_lataccel, state, future_plan):
  target = float(target_lataccel)
  future_target = list(future_plan.lataccel)
  future_roll = list(future_plan.roll_lataccel)

  def at(values, index, fallback):
    return float(values[index]) if len(values) > index else float(fallback)

  target_1 = at(future_target, 0, target)
  target_2 = at(future_target, 1, target_1)
  target_5 = at(future_target, 4, target_2)
  target_10 = at(future_target, 9, target_5)
  target_20 = at(future_target, 19, target_10)
  roll_1 = at(future_roll, 0, state.roll_lataccel)
  speed = float(state.v_ego)
  net = target_1 - roll_1
  return np.asarray([
    target,
    target_1,
    target_2,
    target_5,
    target_10,
    target_20,
    target_1 - target,
    target_5 - target,
    float(state.roll_lataccel),
    roll_1,
    speed,
    float(state.a_ego),
    net / (speed * speed + 1.0),
    net / (speed + 1.0),
    abs(target_1),
    target_1 * speed,
    float(state.roll_lataccel) * speed,
  ], dtype=np.float32)


def deadband(value, width):
  magnitude = max(abs(float(value)) - max(float(width), 0.0), 0.0)
  return float(np.copysign(magnitude, value))


class ResidualMLP:
  def __init__(self, path):
    self.available = False
    if not Path(path).exists():
      return
    with np.load(path, allow_pickle=False) as archive:
      self.feature_mean = archive["feature_mean"].astype(np.float32)
      self.feature_scale = archive["feature_scale"].astype(np.float32)
      self.linear = "linear_weights" in archive.files
      if self.linear:
        self.linear_weights = archive["linear_weights"].astype(np.float32)
        self.linear_bias = float(
          np.asarray(archive["linear_bias"]).reshape(-1)[0]
        )
      else:
        self.w1 = archive["w1"].astype(np.float32)
        self.b1 = archive["b1"].astype(np.float32)
        self.w2 = archive["w2"].astype(np.float32)
        self.b2 = archive["b2"].astype(np.float32)
        self.w3 = archive["w3"].astype(np.float32)
        self.b3 = archive["b3"].astype(np.float32)
      self.residual_limit = float(np.asarray(archive["residual_limit"]).reshape(-1)[0])
      self.output_scale = (
        float(np.asarray(archive["output_scale"]).reshape(-1)[0])
        if "output_scale" in archive.files else 1.0
      )
      archived_ood_start = (
        float(np.asarray(archive["ood_gate_start"]).reshape(-1)[0])
        if "ood_gate_start" in archive.files else float("inf")
      )
      archived_ood_width = (
        float(np.asarray(archive["ood_gate_width"]).reshape(-1)[0])
        if "ood_gate_width" in archive.files else 1.0
      )
      self.ood_gate_start = float(os.getenv(
        "RIYADH_GENERAL_OOD_GATE_START", str(archived_ood_start)
      ))
      self.ood_gate_width = float(os.getenv(
        "RIYADH_GENERAL_OOD_GATE_WIDTH", str(archived_ood_width)
      ))
    if self.feature_mean.shape != (FEATURE_COUNT,):
      raise ValueError("general policy feature count mismatch")
    if self.linear and self.linear_weights.shape != (FEATURE_COUNT,):
      raise ValueError("invalid linear residual weights")
    if not self.linear and (
      self.w1.shape[0] != FEATURE_COUNT or self.w3.shape[1] != 1
    ):
      raise ValueError("invalid general policy weights")
    self.available = True

  def predict(self, features):
    if not self.available:
      return 0.0
    x = (features - self.feature_mean) / self.feature_scale
    if self.linear:
      value = float(x @ self.linear_weights + self.linear_bias)
      gate = 1.0
      if np.isfinite(self.ood_gate_start):
        distance = float(np.max(np.abs(x)))
        gate = float(np.clip(
          (self.ood_gate_start + max(self.ood_gate_width, 1e-6) - distance)
          / max(self.ood_gate_width, 1e-6),
          0.0,
          1.0,
        ))
      return float(
        np.tanh(value) * self.residual_limit * self.output_scale * gate
      )
    x = np.tanh(x @ self.w1 + self.b1)
    x = np.tanh(x @ self.w2 + self.b2)
    residual = float((x @ self.w3 + self.b3)[0])
    return float(
      np.clip(residual, -self.residual_limit, self.residual_limit)
      * self.output_scale
    )


class InverseLinear:
  def __init__(self, path):
    self.available = False
    if not Path(path).exists():
      return
    with np.load(path, allow_pickle=False) as archive:
      self.feature_mean = archive["feature_mean"].astype(np.float32)
      self.feature_scale = archive["feature_scale"].astype(np.float32)
      self.weights = archive["weights"].astype(np.float32)
      self.bias = float(np.asarray(archive["bias"]).reshape(-1)[0])
    if self.feature_mean.shape != (INVERSE_FEATURE_COUNT,):
      raise ValueError("inverse model feature count mismatch")
    if self.weights.shape != (INVERSE_FEATURE_COUNT,):
      raise ValueError("invalid inverse model weights")
    self.available = True

  def predict(self, features):
    if not self.available:
      return 0.0
    normalized = (features - self.feature_mean) / self.feature_scale
    return float(np.clip(
      normalized @ self.weights + self.bias, -STEER_LIMIT, STEER_LIMIT
    ))


class Controller(BaseController):
  def __init__(self):
    self.policy = ResidualMLP(Path(os.getenv(
      "RIYADH_GENERAL_MODEL", str(DEFAULT_MODEL)
    )))
    self.inverse = InverseLinear(Path(os.getenv(
      "RIYADH_GENERAL_INVERSE_MODEL", str(DEFAULT_INVERSE_MODEL)
    )))
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
    self.inverse_scale = float(os.getenv("RIYADH_GENERAL_INVERSE_SCALE", "0.50"))
    self.inverse_limit = float(os.getenv("RIYADH_GENERAL_INVERSE_LIMIT", "1.0"))

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
    inverse = inverse_features(target_lataccel, state, future_plan)
    error_diff = error - self.previous_error
    preview_signal = deadband(features[11], 0.0)
    base = float(
      self.kp * error
      + self.ki * self.error_integral
      + self.kd * error_diff
      + self.kff * float(target_lataccel)
      + self.kpreview * preview_signal
      + self.kroll * float(state.roll_lataccel)
    )
    inverse_action = float(np.clip(
      self.inverse.predict(inverse), -self.inverse_limit, self.inverse_limit
    ))
    requested = (
      base
      + self.inverse_scale * inverse_action
      + self.residual_scale * self.policy.predict(features)
    )
    action = float(np.clip(
      requested,
      self.previous_action - self.action_delta_limit,
      self.previous_action + self.action_delta_limit,
    ))
    action = float(np.clip(action, -STEER_LIMIT, STEER_LIMIT))
    self.previous_error = error
    self.previous_action = action
    return action
