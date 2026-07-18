# Segment 00006 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00006.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 101.277932657 |
| lataccel_cost | 1.214334910 |
| jerk_cost | 40.561187169 |
| rmse_lataccel | 0.110196865 |
| mean_abs_error | 0.054080106 |
| max_abs_error | 1.022644721 |
| jerk_share | 0.400494 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 104 | -0.301237 | 0.721408 | -1.022645 | 1.022645 | -0.163822 | 32.307446 | -0.163662 | 0.501759 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 103 | 5.000000 | -0.281483 | 0.221408 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 103 | 122 | 0.437430 | 0.303821 | 1.022645 | -0.306722 | -0.021902 | -0.454545 | 0.721408 | -0.534375 | -0.013519 | 32.341181 | 0.513218 | 0.063654 |
| 287 | 306 | 0.089949 | 0.071678 | 0.193827 | -0.097512 | 0.022761 | -0.171065 | 0.073314 | -0.398769 | -0.266389 | 31.833955 | 0.529630 | 0.065797 |
| 192 | 211 | 0.083883 | 0.071190 | 0.157416 | -0.047125 | 0.008912 | -0.180841 | 0.122190 | -0.295513 | -0.148592 | 32.295132 | 0.515501 | 0.148317 |
| 408 | 427 | 0.062659 | 0.053140 | 0.127871 | -0.044866 | 0.094657 | -0.102639 | 0.161290 | -0.105484 | 0.172465 | 32.805399 | 0.501408 | 0.227401 |
| 123 | 142 | 0.060409 | 0.049446 | 0.136342 | -0.060351 | 0.111271 | -0.141740 | 0.131965 | -0.194656 | -0.085233 | 32.371871 | 0.532044 | 0.097018 |
