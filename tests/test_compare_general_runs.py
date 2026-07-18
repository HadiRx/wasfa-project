import json
from pathlib import Path
import tempfile
import unittest

from compute.compare_general_runs import compare, load_records


class GeneralRunComparisonTests(unittest.TestCase):
  def write_run(self, directory, name, costs):
    path = Path(directory) / name
    payload = {
      "records": [
        {"segment": segment, "total_cost": cost}
        for segment, cost in costs.items()
      ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

  def test_accepts_uniform_improvement(self):
    with tempfile.TemporaryDirectory() as directory:
      baseline = self.write_run(directory, "before.json", {"1": 10, "2": 20})
      candidate = self.write_run(directory, "after.json", {"1": 9, "2": 18})
      report = compare(baseline, candidate, {
        "mean_percent": 0,
        "median_percent": 0,
        "worst_percent": 0,
        "max_regressed_segments": 0,
        "segment_epsilon": 1e-9,
      })
      self.assertTrue(report["passed"])

  def test_rejects_hidden_worst_case_regression(self):
    with tempfile.TemporaryDirectory() as directory:
      baseline = self.write_run(directory, "before.json", {"1": 100, "2": 100})
      candidate = self.write_run(directory, "after.json", {"1": 1, "2": 150})
      report = compare(baseline, candidate, {
        "mean_percent": 0,
        "median_percent": 0,
        "worst_percent": 0,
        "max_regressed_segments": -1,
        "segment_epsilon": 1e-9,
      })
      self.assertFalse(report["passed"])
      self.assertFalse(report["checks"]["worst"])

  def test_rejects_segment_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      baseline = self.write_run(directory, "before.json", {"1": 10})
      candidate = self.write_run(directory, "after.json", {"2": 9})
      with self.assertRaises(ValueError):
        compare(baseline, candidate, {
          "mean_percent": 0,
          "median_percent": 0,
          "worst_percent": 0,
          "max_regressed_segments": -1,
          "segment_epsilon": 1e-9,
        })

  def test_rejects_duplicate_records(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "duplicate.json"
      path.write_text(json.dumps({"records": [
        {"segment": "1", "total_cost": 10},
        {"segment": "1", "total_cost": 9},
      ]}), encoding="utf-8")
      with self.assertRaises(ValueError):
        load_records(path)


if __name__ == "__main__":
  unittest.main()
