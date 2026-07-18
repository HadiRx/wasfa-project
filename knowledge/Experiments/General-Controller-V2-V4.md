---
type: experiment
scope: general-controller-pilots
status: rejected
controller: compute/riyadh_general.py
---

# General Controller V2–V4 Rejections

V1 remains the protected general-controller baseline. These experiments use
one parameter set and one causal policy for every route; none uses route IDs,
fingerprints, file paths, or action lookup.

## V2 — global gain coordinate search

Training scope: segments `00400`–`00419`. The selected candidate changed only
`Kpreview` from `0.10` to `0.15` and `Kroll` from `0.00` to `-0.03`.

On validation segments `00500`–`00599`, V2 improved mean cost from
`118.917669814` to `118.178722023`, and worst cost from `1034.784759040` to
`977.591457810`.

On the untouched test segments `00600`–`00699`, it regressed:

| Metric | Protected V1 | V2 candidate |
|---|---:|---:|
| mean total cost | 119.658040137 | 122.812591194 |
| median total cost | 70.491362087 | 72.183097378 |
| worst total cost | 2012.552118945 | 2319.429424794 |

V2 is rejected. Sources: `checkpoints/general/v2-train.json`,
`v1-validation-500-599.json`, `v2-validation-500-599.json`,
`v1-test-600-699.json`, and `v2-test-600-699.json`.

## V3 — anti-windup and action-rate limits

Training scope: segments `00700`–`00719`.

- Integral limits `8` and `12` worsened mean cost.
- Integral limits `16`, `20`, and the effectively unlimited baseline were
  identical on this scope.
- Action-delta limits `0.25`, `0.50`, `1.00`, and `4.00` were identical,
  showing that ordinary V1 action changes were already below these limits.

V3 produced no promotable candidate and is rejected. Sources:
`checkpoints/general/v3-train-*.json`.

## V4 — conservative learned residual

The residual MLP was retrained against the corrected PID-plus-preview base.
Teacher split: eight train segments and held-out segments `00002`, `00003`.
Best supervised validation residual MSE was `0.00351996`.

On new training segments `01000`–`01019`, residual scales `0.02`, `0.05`,
`0.10`, and `0.20` did not beat scale `0.00` on official mean cost:

| residual scale | mean total cost |
|---:|---:|
| 0.00 | 80.540949016 |
| 0.02 | 80.844803774 |
| 0.05 | 80.627664430 |
| 0.10 | 80.590195625 |
| 0.20 | 83.435259144 |

V4 is rejected before validation. The low supervised loss does not establish
closed-loop improvement. Sources: `checkpoints/general/models/residual-mlp-v1*`
and `checkpoints/general/v4-train-scale-*.json`.

## Outlier evidence

V1 improves the dominant outliers relative to PID but does not solve them:

- Segment `00582`: PID `1497.946337581`, V1 `1034.784759040`.
- Segment `00633`: PID `2909.893953791`, V1 `2012.552118945`.

Both contain large lateral-acceleration error and jerk. The next learned-policy
iteration needs closed-loop labels (for example dataset aggregation with an
oracle that can relabel visited states) or a model-based online correction.
More open-loop behavior cloning or shallow global-gain sweeps are not promoted.
