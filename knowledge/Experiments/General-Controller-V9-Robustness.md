---
type: experiment
scope: general-controller-v9
status: rejected
date: 2026-07-18
baseline: V8
---

# General Controller V9 Robustness

## Outcome

No V9 candidate was promoted. V8 remains the active general controller. Every
candidate was causal and route-agnostic, but each failed closed-loop robustness
on at least one held-out set. This is a negative result, not an official score.

## Method

- Kept the stock TinyPhysics simulator and official cost.
- Used no route ID, data path, fingerprint, or per-route action lookup.
- Reproduced V8 on segments `16500`–`16599` before comparing candidates.
- Selected parameters only on a small calibration subset.
- Evaluated the leading candidate on all `16500`–`16599` and then on untouched
  segments `17500`–`17599`.
- Added `compute/compare_general_runs.py` so promotion decisions are computed
  from the raw per-segment records rather than summary claims.

## Verified closed-loop results

| Set | Controller | Mean | Median | Worst | Result |
|---|---:|---:|---:|---:|---|
| `16500`–`16599` | V8 | 55.310090 | 48.867802 | 267.000881 | baseline |
| `16500`–`16599` | tuned candidate | 55.084657 | 49.465651 | 228.759758 | rejected: median +1.223% |
| `17500`–`17599` | V8 | 81.309627 | 54.698406 | 1668.227953 | untouched baseline |
| `17500`–`17599` | tuned candidate | 87.117913 | 53.239588 | 1646.177007 | rejected: mean +7.143% |

The tuned candidate reduced the development mean by `0.408%` and worst case by
`14.322%`, but regressed 60/100 development segments. On the untouched set it
regressed 50/100 segments; segment `17511` alone increased by `1148.3043`.

## Other rejected candidates

- Full warmup window (`20`–`480`): supervised validation improved slightly,
  while closed-loop development mean rose to `55.486970` and worst to
  `271.458`.
- Speed-expert inverse model: supervised validation improved, but the hard
  closed-loop subset regressed, so the full test was not run.
- Autoregressive inverse model: validation R² reached `0.997661`, yet the hard
  10-segment closed-loop mean exploded to `7399.404290`. This is distribution
  shift caused by feeding model-generated actions back into a model trained on
  logged actions.
- Integral cap `8`: development mean improved to `55.069041`, but the separate
  `17000`–`17099` mean rose from `56.405032` to `57.309426` and worst rose from
  `434.832444` to `525.271885`.
- Fixed gain, feedforward scale/cap, and speed-gated gain sweeps all failed at
  least one mean, median, or worst-case gate.

## Evidence

- `checkpoints/general/inverse-v8-local-test-100.json`
- `checkpoints/general/v9-robust-candidate-dev-16500.json`
- `checkpoints/general/v9-robust-gate-dev.json`
- `checkpoints/general/v8-baseline-test-17500.json`
- `checkpoints/general/v9-robust-candidate-test-17500.json`
- `checkpoints/general/v9-robust-gate-test.json`
- `checkpoints/general/v9-autoregressive-hard-10.json`
- `checkpoints/general/v9-ilimit8-test-17000-17099.json`

## Next technical step

The remaining failure mode is not solved by one global gain vector. The next
controller iteration should learn a closed-loop residual or short-horizon
predictive policy using simulator rollouts, then pass the same two-set promotion
gate before any 5000-segment run.
