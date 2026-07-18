# Segment 00003 diagnosis

Source: stock TinyPhysics rollout using the saved steering checkpoint in `checkpoints/pilot/00003.npy.b64`.

## Cost

| metric | value |
| --- | ---: |
| total_cost | 25.603151337 |
| lataccel_cost | 0.297581485 |
| jerk_cost | 10.724077062 |
| rmse_lataccel | 0.054551030 |
| mean_abs_error | 0.037391065 |
| max_abs_error | 0.279227712 |
| jerk_share | 0.418858 |

Primary failure mode: `jerk_dominated`.

## Worst point

| step | target_lataccel | current_lataccel | error | abs_error | action | v_ego | a_ego | roll_lataccel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 413 | 0.029961 | -0.249267 | 0.279228 | 0.279228 | -0.026993 | 19.082010 | -0.379349 | 0.410493 |

## Largest jerk

| step | jerk | target_lataccel | current_lataccel |
| --- | --- | --- | --- |
| 398 | -1.564027 | -0.264393 | -0.474096 |

## Worst 20-step windows

| start_step | end_step | rmse | mean_abs_error | max_abs_error | target_min | target_max | pred_min | pred_max | action_min | action_max | speed_mean | abs_roll_mean | abs_a_ego_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 395 | 414 | 0.155674 | 0.134476 | 0.279228 | -1.168371 | 0.081121 | -0.962854 | -0.131965 | -0.620009 | 0.036989 | 19.088103 | 0.367187 | 0.229613 |
| 103 | 122 | 0.071928 | 0.055980 | 0.166510 | -0.044321 | 0.143488 | -0.102639 | 0.122190 | -0.330241 | -0.080241 | 12.313319 | 0.287790 | 1.173214 |
| 360 | 379 | 0.070367 | 0.053893 | 0.158932 | -0.027293 | 0.214327 | 0.014663 | 0.298143 | 0.012085 | 0.098870 | 18.427787 | 0.284141 | 0.100968 |
| 415 | 434 | 0.069588 | 0.054183 | 0.161036 | 0.080760 | 0.387607 | 0.014663 | 0.434995 | 0.076983 | 0.255482 | 18.306113 | 0.331521 | 0.911445 |
| 175 | 194 | 0.052280 | 0.043891 | 0.094087 | 0.019080 | 0.106448 | -0.034213 | 0.200391 | -0.179113 | -0.092183 | 16.814054 | 0.408024 | 0.157044 |
