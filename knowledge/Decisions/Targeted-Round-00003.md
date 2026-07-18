---
type: decision
scope: targeted-pilot
status: planned
segments: [00002, 00003, 00006, 00008, 00009, 00010]
source: checkpoints/pilot/refined-progress.jsonl
---

# Targeted Round 00003

Do not scale to the official 5000-segment evaluation yet. The latest verified
pilot evidence still has a hard outlier: segment `00006` at `101.277932657`
from `checkpoints/pilot/refined-progress.jsonl:6`.

## Target Segments

| Segment | Latest stock cost | Previous stock cost | Evidence |
|---|---:|---:|---|
| 00002 | 42.433692992 | 57.042733627 | `checkpoints/pilot/refined-progress.jsonl:2` |
| 00003 | 25.603151337 | 57.536241556 | `checkpoints/pilot/refined-progress.jsonl:3` |
| 00006 | 101.277932657 | 111.587081274 | `checkpoints/pilot/refined-progress.jsonl:6` |
| 00008 | 42.297873661 | 66.421435638 | `checkpoints/pilot/refined-progress.jsonl:8` |
| 00009 | 37.379313858 | 70.083236479 | `checkpoints/pilot/refined-progress.jsonl:9` |
| 00010 | 37.582516026 | 48.607901746 | `checkpoints/pilot/refined-progress.jsonl:10` |

## Rule

Run diagnosis before another optimization pass. For every target segment, record
the worst error windows, lataccel cost, jerk cost, and exact replay path used for
stock parity.

## Execution Path

Use `.github/workflows/natural-target-diagnosis.yml` to produce diagnosis
artifacts for the six target segments. The first local diagnosis for `00006`
is recorded in `knowledge/Diagnosis/Route-00006-Analysis.md`.
