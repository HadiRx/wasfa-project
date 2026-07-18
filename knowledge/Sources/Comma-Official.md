---
type: source
authority: primary
---

# Comma Official

- Challenge repository: https://github.com/commaai/controls_challenge
- Leaderboard: https://comma.ai/leaderboard
- Required evaluation:
  `python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <name> --baseline_controller pid`
- Competitive publication gate documented by comma: `total_cost < 100`.

These sources govern [[Submission/Checklist]] and override social-media claims.

## Leaderboard check — 2026-07-18

The official board listed `6.880` first overall using per-segment direct
quadratic optimization, `17.791` for PGTO plus behavioral-cloning distillation,
and `110.254` for the stock PID baseline. The `6.880` author's public write-up
labels its simulator-injection controllers as not real controllers and reports
an approximate metric floor of `6.18`: https://github.com/RyanL2/commacontrol.

Riyadh Natural-20 will not patch the simulator, reset its RNG, or inject a
precomputed lateral-acceleration trajectory merely to claim first place.
