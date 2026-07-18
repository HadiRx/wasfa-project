---
type: diagnosis
segment: 00006
scope: single-segment-stock-replay
source_checkpoint: checkpoints/pilot/00006.npy.b64
source_progress: checkpoints/pilot/refined-progress.jsonl:6
generated_by: compute/diagnose_natural_segment.py
---

# Route 00006 Analysis

This diagnosis replays the saved `00006` action checkpoint through stock
TinyPhysics using the same relative data path used by the GitHub refinement
job: `data/00006.csv`. The path string matters because the stock simulator
derives its random seed from `data_path`.

## Verified Cost

- Total cost: `101.277932657`
- Lataccel cost: `1.214334910`
- Jerk cost: `40.561187169`
- RMSE lataccel: `0.110196865` m/s^2
- Max absolute error: `1.022644721` m/s^2
- Max jerk: `5.000000000` m/s^3
- Evidence scope: single-segment stock replay, not an official 5000-segment score.

## Worst Error Windows

| Rank | Steps | RMSE | Max abs error | Target range | Pred range | Mean vEgo | Target delta p95 |
|---:|---|---:|---:|---|---|---:|---:|
| 1 | 103-122 | 0.437430 | 1.022645 | -0.307..-0.022 | -0.455..0.721 | 32.341 | 0.043096 |
| 2 | 102-121 | 0.436222 | 1.022645 | -0.312..-0.030 | -0.455..0.721 | 32.341 | 0.043096 |
| 3 | 101-120 | 0.434936 | 1.022645 | -0.313..-0.045 | -0.455..0.721 | 32.340 | 0.045398 |
| 4 | 100-119 | 0.431669 | 1.022645 | -0.313..-0.090 | -0.455..0.721 | 32.338 | 0.044139 |

## Diagnosis

The dominant local failure is immediately after control starts. The route asks
for mild negative lateral acceleration near steps `100-122`, while the replayed
trajectory first swings positive up to `0.721` and then clips down to `-0.455`.
This makes `00006` an early-window alignment problem more than a late sharp-turn
problem.

The high total cost is not only lateral tracking. Jerk contributes
`40.561187169`, so the next targeted run should penalize early-window action
changes and inspect steps `100-126` before trying global 400-step changes.

## Next Experiment Target

Prioritize a protected refinement focused on the first 30 scored steps:

- Segment: `00006`
- Window: scored indices `0-30`, simulator steps `100-130`
- Candidate basis: dense local basis over the early window plus conservative
  action smoothing
- Acceptance rule: stock cost must improve and parity delta must remain zero
