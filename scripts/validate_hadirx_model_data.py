#!/usr/bin/env python3
"""Validate governed HadiRx model JSONL datasets.

This validator intentionally favors false positives for PHI-like patterns. It does
not prove de-identification; it blocks obvious identifiers and governance errors
before data enters training or evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


PHONE_RE = re.compile(r"(?<!\d)(?:\+?966[- ]?)?0?5\d{8}(?!\d)")
SAUDI_ID_RE = re.compile(r"(?<!\d)[12]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
MRN_RE = re.compile(r"\b(?:mrn|medical\s*record|رقم\s*(?:الملف|الهوية)|هوية)\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.I)
DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]?(?:0[1-9]|1[0-2])[-/]?(?:0[1-9]|[12]\d|3[01])\b")


def iter_strings(value: Any):
  if isinstance(value, str):
    yield value
  elif isinstance(value, dict):
    for child in value.values():
      yield from iter_strings(child)
  elif isinstance(value, list):
    for child in value:
      yield from iter_strings(child)


def phi_hits(record: dict[str, Any]) -> list[str]:
  joined = "\n".join(iter_strings(record))
  hits = []
  for label, pattern in (
    ("phone", PHONE_RE),
    ("saudi_id", SAUDI_ID_RE),
    ("email", EMAIL_RE),
    ("mrn_or_named_identifier", MRN_RE),
    ("full_date", DATE_RE),
  ):
    if pattern.search(joined):
      hits.append(label)
  return hits


def validate_file(path: Path, validator: Draft202012Validator) -> tuple[list[str], list[dict[str, Any]]]:
  errors: list[str] = []
  records: list[dict[str, Any]] = []
  seen_ids: set[str] = set()

  with path.open(encoding="utf-8") as handle:
    for line_no, raw in enumerate(handle, 1):
      if not raw.strip():
        continue
      try:
        record = json.loads(raw)
      except json.JSONDecodeError as exc:
        errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
        continue

      schema_errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
      for exc in schema_errors:
        location = ".".join(str(p) for p in exc.path) or "<root>"
        errors.append(f"{path}:{line_no}:{location}: {exc.message}")

      case_id = record.get("case_id")
      if case_id in seen_ids:
        errors.append(f"{path}:{line_no}: duplicate case_id {case_id}")
      if isinstance(case_id, str):
        seen_ids.add(case_id)

      hits = phi_hits(record)
      if hits:
        errors.append(f"{path}:{line_no}: PHI-like pattern(s): {', '.join(hits)}")

      records.append(record)

  return errors, records


def cross_record_checks(records: list[dict[str, Any]]) -> list[str]:
  errors: list[str] = []
  split_counts = Counter(r.get("split") for r in records)
  signal_counts = Counter((r.get("split"), r.get("gold", {}).get("signal")) for r in records)

  scenario_splits: dict[str, set[str]] = defaultdict(set)
  source_splits: dict[str, set[str]] = defaultdict(set)
  for record in records:
    governance = record.get("governance", {})
    split = record.get("split")
    scenario = governance.get("scenario_family")
    source = governance.get("source_family")
    if scenario and split:
      scenario_splits[scenario].add(split)
    if source and split:
      source_splits[source].add(split)

  for scenario, splits in scenario_splits.items():
    if "test" in splits and "train" in splits:
      errors.append(f"scenario leakage: {scenario!r} appears in train and test")

  for source, splits in source_splits.items():
    if "test" in splits and "train" in splits:
      errors.append(f"source-family leakage: {source!r} appears in train and test")

  if records:
    print("Dataset summary")
    print("---------------")
    for split in ("train", "dev", "test"):
      print(f"{split}: {split_counts[split]}")
      for signal in ("context_present", "documentation_gap", "requires_review"):
        print(f"  {signal}: {signal_counts[(split, signal)]}")

  return errors


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("paths", nargs="+", type=Path)
  parser.add_argument("--schema", type=Path, default=Path("model_data/schema/hadirx_case.schema.json"))
  args = parser.parse_args()

  schema = json.loads(args.schema.read_text(encoding="utf-8"))
  validator = Draft202012Validator(schema, format_checker=FormatChecker())

  all_errors: list[str] = []
  all_records: list[dict[str, Any]] = []
  for path in args.paths:
    errors, records = validate_file(path, validator)
    all_errors.extend(errors)
    all_records.extend(records)

  all_errors.extend(cross_record_checks(all_records))
  if all_errors:
    print("\nVALIDATION FAILED", file=sys.stderr)
    for error in all_errors:
      print(f"- {error}", file=sys.stderr)
    return 1

  print("\nVALIDATION PASSED")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
