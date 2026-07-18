---
type: decision
status: active
date: 2026-07-18
scope: general-controller-v9
---

# Reject V9 Candidates

V9 is not promoted. Keep V8 as the active general controller.

Reason: the leading candidate improved development mean and worst case but
regressed development median, then increased untouched-set mean by `7.143%`.
The other candidates failed earlier closed-loop gates. High supervised R² is
explicitly insufficient evidence for controller promotion.

The project may continue to V10 with simulator-rollout training. It must not run
the official 5000-segment evaluation or claim completion until a single causal,
route-agnostic controller passes development and untouched-set gates.

See [[Experiments/General-Controller-V9-Robustness]].
