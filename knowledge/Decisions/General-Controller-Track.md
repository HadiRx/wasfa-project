---
type: decision
status: active
scope: controller-design
---

# General Controller Track

The submission target is a single causal controller that operates without
route IDs, fingerprints, file paths, or per-segment action lookup. Existing
optimized action checkpoints remain useful as offline teacher/oracle data, but
the final general controller must be evaluated on unseen segments.

Acceptance gates:

1. Preserve the stock TinyPhysics simulator and official cost.
2. Split data by segment before training.
3. Reject models that improve supervised loss but regress closed-loop cost.
4. Compare every promoted controller with official PID on the same unseen set.
5. Run an untouched test set after selecting parameters.
6. Do not call any pilot result an official or leaderboard score.

The first implementation and evidence are in
[[Experiments/General-Controller-V1]].
