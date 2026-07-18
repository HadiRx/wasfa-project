# Segment 00010 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00010.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 37.582516026 |
| lataccel_cost | 0.376235778 |
| jerk_cost | 18.770727114 |
| rmse_lataccel | 0.061338061 |
| mean_abs_error | 0.048973890 |
| max_abs_error | 0.210287507 |
| jerk_share | 0.499454 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 474 | -0.870112 | -0.659824 | -0.210288 | 0.210288 | -0.054440 | 30.354170 | 0.027747 | -0.391377 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 178 | 1.173021 | 0.931495 | 0.982405 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 456 | 475 | 0.097611 | 0.081788 | 0.210288 | -0.877811 | -0.415266 | -0.728250 | -0.405670 | -0.082365 | 0.132873 | 30.314619 | 0.360336 | 0.053875 |
| 157 | 176 | 0.093559 | 0.085194 | 0.163509 | 0.596813 | 0.942085 | 0.522972 | 0.826002 | 0.138785 | 0.400621 | 30.402588 | 0.456623 | 0.028670 |
| 129 | 148 | 0.085294 | 0.075517 | 0.140544 | 0.734388 | 0.814182 | 0.630499 | 0.923754 | 0.151919 | 0.286702 | 30.365792 | 0.481545 | 0.034258 |
| 276 | 295 | 0.076402 | 0.063009 | 0.157310 | 0.210540 | 0.424312 | 0.102639 | 0.581623 | 0.025781 | 0.170865 | 30.291646 | 0.319044 | 0.111550 |
| 109 | 128 | 0.068479 | 0.058303 | 0.130451 | 0.777944 | 0.809407 | 0.650049 | 0.904203 | 0.066216 | 0.195294 | 30.341229 | 0.477411 | 0.039969 |
