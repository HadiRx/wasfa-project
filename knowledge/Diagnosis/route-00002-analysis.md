# Segment 00002 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00002.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 42.433692992 |
| lataccel_cost | 0.466649569 |
| jerk_cost | 19.101214527 |
| rmse_lataccel | 0.068311754 |
| mean_abs_error | 0.053067205 |
| max_abs_error | 0.238245761 |
| jerk_share | 0.450143 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 391 | 0.470454 | 0.708700 | -0.238246 | 0.238246 | 0.184342 | 31.239285 | -0.028222 | 0.483380 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 483 | -1.270772 | 0.525792 | 0.415445 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 378 | 397 | 0.152363 | 0.138047 | 0.238246 | 0.458610 | 0.704900 | 0.532747 | 0.747801 | 0.067732 | 0.453562 | 31.235298 | 0.481678 | 0.049871 |
| 122 | 141 | 0.097625 | 0.082363 | 0.186559 | -0.152951 | 0.125413 | -0.141740 | 0.102639 | -0.146757 | 0.190780 | 31.206678 | 0.256343 | 0.039185 |
| 475 | 494 | 0.083337 | 0.066315 | 0.183576 | 0.150548 | 0.602300 | 0.063539 | 0.620723 | 0.070565 | 0.285451 | 31.171350 | 0.471150 | 0.046557 |
| 242 | 261 | 0.082894 | 0.071103 | 0.148992 | -0.141752 | 0.082409 | -0.141740 | 0.024438 | -0.054795 | 0.053462 | 31.223807 | 0.291409 | 0.031296 |
| 357 | 376 | 0.082861 | 0.070074 | 0.172079 | 0.467513 | 0.752328 | 0.464321 | 0.718475 | 0.336785 | 0.490554 | 31.196100 | 0.464728 | 0.046600 |
