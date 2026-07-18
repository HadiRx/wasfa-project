---
type: diagnosis
scope: general-controller-v10
status: active
date: 2026-07-18
---

# V10 Extreme Failure Diagnosis

## Scope

This diagnosis uses V8 closed-loop traces from `17565`, `17511`, `17558`,
`16537`, and `16523`. It is a development diagnosis, not an official result.

## Findings

| Segment | Total cost | Lataccel cost | Jerk cost | Max error | Max integral | Action saturation |
|---|---:|---:|---:|---:|---:|---:|
| 17565 | 1668.227953 | 24.050285 | 465.713678 | 1.858717 | 10.967327 | 0 |
| 17511 | 497.872709 | 7.689192 | 113.413114 | 1.998412 | 13.648343 | 2 |
| 17558 | 426.224235 | 6.240532 | 114.197614 | 1.488171 | 7.478776 | 0 |
| 16537 | 267.000881 | 4.381844 | 47.908701 | 1.084843 | 4.987263 | 0 |
| 16523 | 242.934284 | 3.551786 | 65.344977 | 1.437293 | 11.251678 | 0 |

The failures are not primarily steering saturation. Four of the five cases
have zero saturated control steps, yet all show large tracking error and rapid
lataccel changes. The two most extreme target ranges are `[-3.466, 4.135]` on
`17565` and `[-4.878, 0.151]` on `17511`.

## Rejected causal rules

Two route-agnostic rules were tested on a hard mixed subset:

1. Reset the error integral after a sufficiently large target sign reversal.
   It left the worst case unchanged and increased mean by `1.298%`.
2. Increase or decrease inverse feedforward only at high target magnitude.
   Positive and negative boosts both created large regressions; tested worst
   cases ranged from `1643.069` to `3573.599`.

Both rules were removed. The active controller remains byte-equivalent to V8.

## Conclusion

The next credible approach is closed-loop policy learning or short-horizon
predictive control trained and selected with simulator rollouts. Logged-action
supervision and global PID/feedforward sweeps do not model how policy errors
change the future state distribution.

Raw trace summary: `checkpoints/general/v10-failure-diagnosis.json`.
