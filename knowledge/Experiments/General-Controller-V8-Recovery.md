---
type: experiment
scope: general-controller-recovery
status: awaiting-ci
controller: compute/riyadh_general.py
---

# General Controller V8 Recovery

## Hypothesis

Public warmup steering can identify a compact route-agnostic inverse model.
Blending that feedforward prediction with stable PID-plus-preview feedback can
reduce lag without using route identity or an action lookup.

## Configuration

- Linear ridge inverse model with 17 causal current/future features.
- Training segments: `00000`–`04999`, warmup rows only.
- Supervised validation segments: `10000`–`10499`.
- Closed-loop CI segments: `16500`–`16599`.
- Feedforward scale: `0.50`; prediction cap: `1.00`.
- Stock simulator, stock cost, no RNG reset, no simulator modification.

## Recovery note

The first local V8 commit and its raw pilot files were lost when the transient
workspace was reset before the approved push. They were never published, so
their SHA values cannot be pushed or treated as durable evidence. The
implementation was rebuilt from the reviewed design, and the workflow now
trains the tiny inverse artifact deterministically during CI to avoid depending
on an uncommitted binary.

## Promotion gate

The rebuilt V8 remains `awaiting-ci`. Do not cite the pre-reset local numbers as
published evidence. Promote only after the new GitHub Action succeeds and its
held-out JSON artifact is archived. This remains a pilot, not official-5000.
