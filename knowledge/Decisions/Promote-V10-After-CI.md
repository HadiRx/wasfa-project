---
type: decision
status: provisional
date: 2026-07-18
scope: general-controller-v10
---

# Promote V10 After CI

Promote the bounded closed-loop residual to the active general-controller
candidate only if GitHub Actions reproduces both strict gates on
`16500`–`16599` and `18500`–`18599`.

Do not promote the rejected residual-`0.15` model or the OOD-gated variants.
Do not run or claim the official 5000 score from local evidence alone.

See [[Experiments/General-Controller-V10-Closed-Loop]].
