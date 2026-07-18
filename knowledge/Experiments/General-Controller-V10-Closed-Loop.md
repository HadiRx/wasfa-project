---
type: experiment
scope: general-controller-v10
status: awaiting-ci
date: 2026-07-18
baseline: V8
---

# General Controller V10 Closed-Loop Residual

## Outcome

V10 is a provisional promotion candidate. It passed local development and a
separate final 100-segment test, but remains `awaiting-ci` until GitHub Actions
reproduces both gates. This is a pilot result, not the official 5000 score.

## Controller

- V8 PID-plus-preview and inverse feedforward remain the safety baseline.
- A 17-parameter linear residual consumes the same 16 causal policy features
  plus bias.
- Parameters were optimized directly against stock TinyPhysics closed-loop
  `total_cost` using antithetic evolution search.
- The search objective combines mean cost, p90 tail cost, and a penalty for
  per-segment regression against V8.
- Residual output is bounded to `0.05`, then packaged with deployment scale
  `0.8` for an effective maximum correction of `0.04`.
- No route ID, path, fingerprint, or per-segment action table is available to
  the controller.

## Split discipline

| Purpose | Segments | Used for |
|---|---|---|
| Training | `16000`–`16011` | evolution-search objective |
| Validation | `16200`–`16219` | accept/reject trained weights |
| Development gate | `16500`–`16599` | comparison with prior V8 evidence |
| Scale stress set | selected hard cases from `18000`–`18099` | choose deployment scale only |
| Final test | `18500`–`18599` | final untouched local gate |

The first six-segment, residual-`0.15` run was rejected: training improved but
validation mean rose from `113.442795` to `152.875824` and worst rose from
`634.068576` to `1204.425197`.

The conservative run used 12 training segments, residual limit `0.05`, and a
stronger regression penalty. On its 20 validation segments it improved mean
from `85.928789` to `77.911101`, median from `45.456942` to `41.756307`, p90
from `101.952956` to `83.053907`, and worst from `634.068576` to `628.011145`.

An out-of-distribution gate was tested after failures on the scale stress set.
It was rejected because its transition introduced additional closed-loop
instability. A constant residual scale of `0.8` was selected instead.

## Local promotion gates

| Set | Controller | Mean | Median | Worst |
|---|---:|---:|---:|---:|
| `16500`–`16599` | V8 | 55.310090 | 48.867802 | 267.000881 |
| `16500`–`16599` | V10 | 53.305525 | 48.223051 | 236.129071 |
| `18500`–`18599` | V8 | 64.016017 | 53.558044 | 867.930844 |
| `18500`–`18599` | V10 | 58.748875 | 51.465297 | 598.065093 |

V10 improved development mean by `3.624%` and final-test mean by `8.228%`.
Final-test worst case improved by `31.093%`. It regressed 19/100 development
segments and 14/100 final-test segments, but the largest individual regression
was only `5.793210`; all aggregate promotion checks passed.

## Evidence

- `checkpoints/general/models/v10-closed-loop-conservative.json`
- `checkpoints/general/models/v10-closed-loop-deploy.json`
- `checkpoints/general/v10-scale08-dev-16500.json`
- `checkpoints/general/v10-scale08-gate-dev.json`
- `checkpoints/general/v8-baseline-test-18500.json`
- `checkpoints/general/v10-scale08-test-18500.json`
- `checkpoints/general/v10-scale08-gate-test.json`
- `checkpoints/general/v10-package-parity.json`

## Next gate

GitHub Actions must rebuild the inverse model, run both 100-segment comparisons,
and archive the deploy model plus raw JSON. Only then may V10 become the active
general-controller candidate. The 5000-segment evaluation remains blocked.
