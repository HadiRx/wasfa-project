---
type: experiment
scope: unseen-100-pilot
status: accepted-for-next-iteration
controller: compute/riyadh_general.py
source: checkpoints/general/preview-v1-test-100.json
---

# General Controller V1

This experiment starts the route-agnostic controller track. The runtime policy
contains no route ID, fingerprint, CSV path, or per-segment action table. It
uses the ordinary controller inputs: target/current lateral acceleration,
vehicle state, future plan, and causal controller memory.

## Split integrity

- Teacher-action distillation prototype: eight train segments and two held-out
  segments from `00001`–`00010`.
- Global preview-gain search: segments `00200`–`00229`.
- Model selection validation: segments `00100`–`00199`.
- Final untouched V1 test: segments `00300`–`00399`.

These scopes are engineering pilots, not the official 5000-segment score.

## Accepted V1 result

V1 is the official PID structure plus one route-agnostic ten-step target
preview term with `Kpreview=0.10`. Source:
`checkpoints/general/preview-v1-test-100.json`.

| Metric | General V1 | Official PID |
|---|---:|---:|
| unseen segments | 100 | 100 |
| mean total cost | 87.916127347 | 90.571683636 |
| median total cost | 70.799595691 | 70.492342192 |
| worst total cost | 320.285413495 | 408.226125114 |
| mean improvement | 2.931994% | — |

The mean and worst case improved on the untouched test set, while the median
regressed slightly. V1 is therefore accepted only as a baseline for the next
general-controller iteration, not as submission-ready.

## Rejected experiments retained

### Residual MLP V0

The offline residual imitation model achieved validation residual MSE
`0.00191705`, but closed-loop evaluation on held-out segments `00004` and
`00008` produced mean total cost `217.939410333`. This is a distribution-shift
failure. Artifacts are under `checkpoints/general/rejected/residual-mlp-v0*`.

### Analytic feedback V0

The first hand-tuned feedback variant looked better on two held-out routes but
failed on a broader unseen-100 set: mean `143.632498979` versus PID
`105.619910958`, with worst case `2368.230227088`. Source:
`checkpoints/general/analytic-v0-unseen-100.json`. It was rejected and the V1
base was restored to PID-equivalent behavior before adding preview.

## Next hypothesis

Keep V1 protected and train a closed-loop policy with dataset aggregation or
model-predictive targets. Any learned residual must beat V1 on a fresh
segment-level test split before it can be packaged. The official 5000-segment
evaluation remains blocked.
