import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPUTE = ROOT / "compute"


def load_controller_module():
  controllers = types.ModuleType("controllers")
  controllers.BaseController = object
  sys.modules.setdefault("controllers", controllers)
  package = types.ModuleType("compute_controllers")
  package.__path__ = [str(COMPUTE)]
  package.BaseController = object
  sys.modules["compute_controllers"] = package
  spec = importlib.util.spec_from_file_location(
    "compute_controllers.riyadh_general", COMPUTE / "riyadh_general.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class State:
  roll_lataccel = 0.1
  v_ego = 25.0
  a_ego = -0.2


class Plan:
  lataccel = [0.2] * 50
  roll_lataccel = [0.1] * 50
  v_ego = [25.0] * 50
  a_ego = [-0.2] * 50


class GeneralControllerTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.module = load_controller_module()

  def test_feature_shape_and_finiteness(self):
    features = self.module.policy_features(
      0.2, 0.1, State(), Plan(), 0.0, 0.0, 0.0
    )
    self.assertEqual(features.shape, (self.module.FEATURE_COUNT,))
    self.assertTrue(np.isfinite(features).all())

  def test_controller_has_no_route_identity_mechanism(self):
    source = (COMPUTE / "riyadh_general.py").read_text(encoding="utf-8").lower()
    forbidden = ("fingerprint(", "data_path", "segment_id", "natural_lookup")
    for token in forbidden:
      self.assertNotIn(token, source)

  def test_controller_output_is_bounded(self):
    controller = self.module.Controller()
    outputs = [controller.update(5.0, -5.0, State(), Plan()) for _ in range(120)]
    self.assertLessEqual(max(outputs), 2.0)
    self.assertGreaterEqual(min(outputs), -2.0)

  def test_missing_model_uses_general_feedback(self):
    policy = self.module.ResidualMLP(ROOT / "does-not-exist.npz")
    self.assertFalse(policy.available)
    self.assertEqual(policy.predict(np.zeros(self.module.FEATURE_COUNT)), 0.0)

  def test_linear_closed_loop_policy_loads_and_is_bounded(self):
    count = self.module.FEATURE_COUNT
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "linear-residual.npz"
      np.savez_compressed(
        path,
        feature_mean=np.zeros(count, dtype=np.float32),
        feature_scale=np.ones(count, dtype=np.float32),
        linear_weights=np.ones(count, dtype=np.float32),
        linear_bias=np.asarray([0.0], dtype=np.float32),
        residual_limit=np.asarray([0.2], dtype=np.float32),
        output_scale=np.asarray([0.8], dtype=np.float32),
      )
      policy = self.module.ResidualMLP(path)
      self.assertTrue(policy.available)
      self.assertLessEqual(policy.predict(np.ones(count)), 0.160001)
      self.assertGreaterEqual(policy.predict(-np.ones(count)), -0.160001)

  def test_inverse_features_are_finite(self):
    features = self.module.inverse_features(0.2, State(), Plan())
    self.assertEqual(features.shape, (self.module.INVERSE_FEATURE_COUNT,))
    self.assertTrue(np.isfinite(features).all())

  def test_missing_inverse_model_is_safe(self):
    inverse = self.module.InverseLinear(ROOT / "missing-inverse.npz")
    self.assertFalse(inverse.available)
    self.assertEqual(
      inverse.predict(np.zeros(self.module.INVERSE_FEATURE_COUNT)), 0.0
    )

  def test_v8_defaults(self):
    controller = self.module.Controller()
    self.assertEqual(controller.inverse_scale, 0.5)
    self.assertEqual(controller.inverse_limit, 1.0)

  def test_deadband_is_symmetric(self):
    self.assertEqual(self.module.deadband(0.01, 0.02), 0.0)
    self.assertAlmostEqual(self.module.deadband(0.05, 0.02), 0.03)
    self.assertAlmostEqual(self.module.deadband(-0.05, 0.02), -0.03)


if __name__ == "__main__":
  unittest.main()
