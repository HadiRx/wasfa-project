# V12 Research Basis

## Goal

Build a compact, route-agnostic, causal nonlinear residual controller that can improve the frozen V10 baseline without using route identity, lookup tables, or test-set tuning.

## External repositories reviewed

### karpathy/micrograd
- Use: design inspiration for a very small, transparent MLP and explicit backpropagation concepts.
- Do not vendor as runtime dependency; micrograd is scalar-valued and educational.
- License: MIT.

### vwxyzjn/cleanrl
- Use: reproducible experiment structure, explicit seeding, compact single-file training style, continuous-control baselines, logging discipline, and separation of training/evaluation.
- Do not import CleanRL as a library; adapt the research pattern only.
- License: MIT.

### acados/acados
- Use later as an MPC benchmark or teacher policy after a reliable dynamics approximation is available.
- Not selected for the first V12 implementation because the current TinyPhysics challenge interface and GitHub runner budget favor a compact learned residual first.
- License: BSD-2-Clause.

### do-mpc/do-mpc
- Use as a reference for robust MPC and moving-horizon estimation concepts.
- Not selected as a runtime dependency because it is heavier and LGPL-3.0, and integration would add solver/runtime complexity before proving value.

### google-deepmind/acme and google-deepmind/dm_control
- Use as references for continuous-control evaluation, reproducibility, and environment design.
- Not selected as dependencies because the challenge already supplies TinyPhysics and a dedicated simulator.

### commaai/controls_challenge and commaai/openpilot
- Remain the authoritative simulator, data, scoring, and control-domain references.
- Relevant design lessons include causal temporal inputs, past-curvature/history for smoother lateral control, model-based feedforward, conservative fallback behavior, and strict held-out evaluation.

## V12 architecture decision

1. Keep frozen V10 as the safety baseline and fallback.
2. Add a compact MLP residual rather than replacing the controller.
3. Inputs remain causal and route-agnostic.
4. Add a short temporal history summary for error/action oscillation and phase lag.
5. Bound the residual output and apply an out-of-distribution gate.
6. Train with deterministic seeds and explicit train/validation/development/holdout splits.
7. Promote only if mean, median, P90, worst-case, and regression gates pass against V10.
8. Record model hashes, data ranges, configuration, and complete evaluation summaries.

## Planned implementation stages

1. Add temporal feature extraction and tests.
2. Add supervised warm-start training for a compact MLP using official warmup telemetry.
3. Add closed-loop fine-tuning or derivative-free search on a limited training split.
4. Select architecture and residual scale on development only.
5. Freeze candidate and evaluate once on a fresh holdout.
6. Expand to 500, 1,000, and then 5,000 segments only after passing earlier gates.

## Non-goals

- No route IDs, segment fingerprints, or per-segment action tables.
- No direct use on a real vehicle.
- No claim of leaderboard performance until an official-style full evaluation is complete.
