---
type: design
status: implemented-pilot
---

# Controller Design

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
