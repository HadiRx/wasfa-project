---
type: decision
status: active
date: 2026-07-18
scope: general-controller-v10
---

# Promote V10 After CI

Promote the bounded closed-loop residual to the active general-controller
candidate. GitHub Actions run `29652798809` reproduced both strict gates on
`16500`–`16599` and `18500`–`18599` and archived the deploy artifact.

Do not promote the rejected residual-`0.15` model or the OOD-gated variants.
Do not yet run or claim the official 5000 score. First run a broader preflight
set and review its mean, median, tail, worst case, and regression distribution.

See [[Experiments/General-Controller-V10-Closed-Loop]].
