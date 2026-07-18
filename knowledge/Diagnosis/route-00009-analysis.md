# Segment 00009 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00009.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 37.379313858 |
| lataccel_cost | 0.355951748 |
| jerk_cost | 19.581726461 |
| rmse_lataccel | 0.059661692 |
| mean_abs_error | 0.041501019 |
| max_abs_error | 0.444362593 |
| jerk_share | 0.523865 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 102 | -1.943385 | -1.499022 | -0.444363 | 0.444363 | -0.488892 | 17.267160 | -0.205161 | -0.396350 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 102 | 5.000000 | -1.943385 | -1.499022 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 102 | 121 | 0.166633 | 0.108773 | 0.444363 | -1.943385 | -0.104472 | -1.549365 | -0.043988 | -0.688874 | -0.049905 | 17.075268 | 0.178119 | 0.192461 |
| 438 | 457 | 0.078665 | 0.069696 | 0.135801 | -0.006001 | 0.088098 | -0.141740 | 0.161290 | -0.026174 | 0.105963 | 21.278888 | 0.231350 | 0.180744 |
| 363 | 382 | 0.076302 | 0.066287 | 0.139821 | 0.046209 | 0.115383 | -0.063539 | 0.112414 | -0.052782 | 0.101394 | 22.171882 | 0.237971 | 0.068234 |
| 210 | 229 | 0.074465 | 0.064391 | 0.151332 | -0.178731 | 0.072319 | -0.122190 | -0.014663 | -0.052171 | 0.147109 | 17.946147 | 0.386927 | 0.396208 |
| 476 | 495 | 0.055301 | 0.046111 | 0.100883 | -0.083078 | 0.035104 | -0.112414 | 0.024438 | -0.160699 | -0.045908 | 21.131809 | 0.280762 | 0.093475 |
