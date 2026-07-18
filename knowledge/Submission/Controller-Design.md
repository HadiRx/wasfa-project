---
type: design
status: implemented-pilot
---

# Controller Design

## General controller track

`compute/riyadh_general.py` is the active route-agnostic research controller.
It uses only ordinary `update()` inputs and causal memory; it contains no
fingerprint, route ID, data path, or per-segment action table. General V1 adds
a globally selected preview term to the official PID structure. Its current
evidence is an unseen-100 engineering test, documented in
[[Experiments/General-Controller-V1]]. It is not submission-ready.

## Per-segment lookup track

`controllers.natural_lookup.Controller` identifies a dataset segment from the
80 observations provided during steps 20–99, before the scored control window.
It then returns the pre-optimized 400 steering commands through the ordinary
`update()` interface.

## Integrity constraints

- No simulator patch.
- No RNG replacement or reset.
- No lateral-acceleration or trajectory injection.
- Steering remains clipped to the official `[-2, 2]` range.
- Unknown fingerprints fall back to the stock PID controller.
- The lookup builder rejects malformed actions and fingerprint collisions.

The implementation lives in `compute/natural_lookup.py` and
`compute/build_natural_lookup.py`. Promotion requires the official evaluation
described in [[Submission/Checklist]].
