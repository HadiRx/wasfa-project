"""A preview-aware, anti-windup lateral controller for the comma challenge.

The controller deliberately stays small and auditable.  It combines feedback
on the measured lateral-acceleration error with a short look-ahead term and a
leaky/clamped integral.  Gains can be overridden with RPL_* environment
variables so the supplied tuning script can reproduce an experiment without
editing this file.
"""

from . import BaseController

import os
import numpy as np


def _gain(name, default):
  return float(os.getenv(f"RPL_{name}", default))


class Controller(BaseController):
  def __init__(self):
    self.kp = _gain("KP", 0.22)
    self.ki = _gain("KI", 0.100)
    self.kd = _gain("KD", 0.0)
    self.kff = _gain("KFF", 0.05)
    self.kpreview = _gain("KPREVIEW", 0.22)
    self.preview_steps = max(1, int(_gain("PREVIEW_STEPS", 8)))
    self.kroll = _gain("KROLL", 0.0)
    self.integral_decay = _gain("IDECAY", 1.0)
    self.integral_limit = _gain("ILIMIT", 1000000.0)
    self.action_limit = 2.0
    self.error_integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral = np.clip(
      self.integral_decay * self.error_integral + error,
      -self.integral_limit,
      self.integral_limit,
    )
    error_diff = error - self.prev_error
    self.prev_error = error

    # A short horizon reacts early to curves while averaging model noise.
    future = np.asarray(future_plan.lataccel[:self.preview_steps], dtype=float)
    preview = float(np.mean(future) - target_lataccel) if future.size else 0.0

    action = (
      self.kp * error
      + self.ki * self.error_integral
      + self.kd * error_diff
      + self.kff * target_lataccel
      + self.kpreview * preview
      + self.kroll * state.roll_lataccel
    )
    return float(np.clip(action, -self.action_limit, self.action_limit))
