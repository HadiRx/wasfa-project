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

The rebuilt inverse-feedforward candidate and its recovery limitations are in
[[Experiments/General-Controller-V8-Recovery]].

The V9 robustness campaign and its rejected candidates are in
[[Experiments/General-Controller-V9-Robustness]]. V8 remains active until a
later candidate passes both development and untouched-set promotion gates.
The formal promotion decision is [[Decisions/Reject-V9-Candidates]].

The extreme-failure trace analysis and rejected V10 causal rules are recorded
in [[Diagnosis/V10-Extreme-Failures]].

The first closed-loop residual to pass local development and final-test gates
is documented in [[Experiments/General-Controller-V10-Closed-Loop]]. Its
provisional promotion rule is [[Decisions/Promote-V10-After-CI]].
