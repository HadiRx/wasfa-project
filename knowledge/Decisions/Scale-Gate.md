---
type: decision
status: active
---

# Scale Gate

Do not launch 5000-segment optimization merely because infrastructure works.
Scale only after a representative pilot demonstrates acceptable mean, tail
risk, throughput, and exact parity with the stock simulator.

The first ten-segment mean of `48.3352` failed the Natural-20 target, while the
worst segment exceeded `111`. Therefore the honest decision was to refine the
algorithm rather than multiply a weak configuration across 5000 routes.

## Promotion requirements

1. Stock simulator parity delta is zero or within declared tolerance.
2. No segment checkpoint regresses during protected refinement.
3. Pilot mean and worst-tail trend improve materially.
4. Runtime estimate fits the selected compute budget.
5. The lookup controller reproduces offline actions through `update()` only.

Related: [[Experiments/Pilot-00001-00010]], [[Diagnosis/Hard-Segment-Diagnosis]],
[[Decisions/Targeted-Round-00003]], [[Diagnosis/Route-00006-Analysis]],
[[Submission/Controller-Design]].
