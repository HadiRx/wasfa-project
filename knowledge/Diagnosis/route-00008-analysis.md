# Segment 00008 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00008.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 42.297873661 |
| lataccel_cost | 0.516044821 |
| jerk_cost | 16.495632605 |
| rmse_lataccel | 0.071836260 |
| mean_abs_error | 0.049247002 |
| max_abs_error | 0.405723068 |
| jerk_share | 0.389987 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 105 | -0.058704 | 0.347019 | -0.405723 | 0.405723 | -0.238226 | 17.274054 | 1.032233 | 0.339635 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 104 | 2.932551 | -0.063976 | 0.259042 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 104 | 123 | 0.200793 | 0.143390 | 0.405723 | -0.063976 | 0.025255 | -0.092864 | 0.347019 | -0.435002 | -0.125305 | 18.137120 | 0.348337 | 0.997295 |
| 194 | 213 | 0.114621 | 0.103969 | 0.211391 | -1.236414 | -0.479114 | -1.148583 | -0.650049 | -0.710762 | -0.068389 | 24.392255 | 0.052458 | 0.060384 |
| 124 | 143 | 0.086931 | 0.063561 | 0.186439 | -0.004088 | 0.035350 | -0.171065 | 0.073314 | -0.346152 | -0.256775 | 20.038825 | 0.326200 | 0.923450 |
| 372 | 391 | 0.075029 | 0.059012 | 0.159702 | 0.165836 | 0.277613 | 0.131965 | 0.434995 | -0.112847 | 0.113436 | 24.731842 | 0.621188 | 0.040307 |
| 242 | 261 | 0.072874 | 0.062243 | 0.131588 | -0.102003 | 0.117603 | -0.131965 | 0.200391 | -0.169293 | 0.030495 | 24.582535 | 0.486574 | 0.051198 |
