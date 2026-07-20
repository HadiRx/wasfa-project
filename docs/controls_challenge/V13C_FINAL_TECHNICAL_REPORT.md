# Riyadh Natural-20 V13-C — Final Technical Report

**Author:** Hadi Ameayr (HadiRx)  
**Challenge:** comma Controls Challenge v2  
**Controller:** `riyadh_v13c`  
**Status:** Frozen after successful 5,000-segment verification

## Executive result

V13-C is a deterministic closed-loop controller built from an analytic feedback controller, a learned linear inverse model, and a compact linear residual policy. Its controller parameters and residual weights were refined with low-dimensional CMA-ES against the official TinyPhysics closed-loop objective.

The final paired verification against frozen V13-A produced:

| Evaluation | Metric | V13-A | V13-C | Change |
|---|---:|---:|---:|---:|
| 1,000 fresh | Mean | 61.5602 | 58.0490 | -5.70% |
| 1,000 fresh | Median | 50.3704 | 49.5878 | -1.55% |
| 1,000 fresh | P99 | 415.3046 | 290.3373 | -30.09% |
| 1,000 fresh | Worst | 1489.4588 | 1190.9254 | -20.04% |
| 5,000 final | Mean | 59.8251 | 57.3149 | -4.20% |
| 5,000 final | Median | 50.9009 | 49.9827 | -1.80% |
| 5,000 final | P90 | 85.8758 | 83.5956 | -2.66% |
| 5,000 final | P99 | 288.4970 | 279.1656 | -3.23% |
| 5,000 final | Worst | 6948.9661 | 5709.0308 | -17.84% |

On the final 5,000 segments, V13-C improved 3,585 segments and regressed on 1,415. Mean lataccel cost improved 4.40%, and mean jerk cost improved 3.77%. A 30,000-sample paired bootstrap estimated a mean delta of -2.5102 with a 95% interval of [-3.7317, -1.3482], giving an estimated improvement probability of 99.9967%.

## Development path

### V8 — analytic baseline

A hand-designed controller established a deterministic baseline and the closed-loop evaluation infrastructure.

### V10 — learned residual

A compact linear residual policy was added around the analytic controller. It improved the earlier baseline and became the base model for direct policy search.

### V12 — nonlinear imitation attempt

A small MLP was trained from warm-up teacher data. It fit the supervised objective but worsened closed-loop Development performance by roughly 26%, demonstrating that teacher imitation was misaligned with the competition objective. V12 was rejected.

### V13-A — direct policy search

CMA-ES optimized a low-dimensional vector covering residual blocks, bias, output scale, and controller parameters. V13-A produced repeatable improvements on Development, Holdout, and 500-segment verification.

### V13-B — fine refinement

V13-B improved mean and median on Development but slightly worsened the worst case and caused a large regression on segment 15692. The automatic gate rejected it and retained V13-A.

### V13-C — hard-case guarded refinement

V13-C incorporated a fixed hard-case guard set during search. Each candidate was evaluated on a changing search batch plus known difficult segments. The search objective heavily penalized guard regressions. V13-C passed Development and fresh Holdout gates, then passed independent 1,000- and 5,000-segment verifications.

## Controller structure

The submitted controller combines:

1. Proportional, integral, derivative, and preview feedback.
2. A linear inverse model trained on official public data.
3. A normalized 16-feature residual policy with a bounded tanh output.
4. Action rate limiting and absolute steering limits.
5. Integral decay and integral limiting.

The runtime controller is deterministic. It does not use route IDs, segment-specific branches, random sampling, hidden evaluation data, simulator modification, or external network access.

## Data separation

Distinct ranges were used for inverse-model training, V10 selection, V13-A search, V13-B search and gates, V13-C search and gates, 1,000-segment verification, and 5,000-segment final verification. The final 5,000 range was not used during controller optimization.

## Reproduction

The official repository currently instructs entrants to run:

```bash
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller riyadh_v13c --baseline_controller pid
```

The submission package contains the controller, both required model files, the generated `report.html`, independent verification evidence, a manifest, and SHA-256 checksums.

## Interpretation

V13-C is the strongest verified controller in this project. The evidence supports a real aggregate improvement over V13-A, including better central tendency, tail metrics, worst-case performance, lataccel tracking, and jerk. It is not better on every segment, and the large individual regressions remain documented rather than hidden. The frozen submission therefore represents a validated aggregate improvement, not a claim of universal dominance or a global leaderboard position.
