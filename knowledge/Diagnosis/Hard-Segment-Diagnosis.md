# Hard-segment diagnosis

Source: stock TinyPhysics rollouts using saved steering checkpoints in `checkpoints/pilot/*.npy.b64`.

| segment | stock_cost | rmse_lataccel | max_abs_error | jerk_share | primary_failure_mode | worst_window_start | worst_window_end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00006 | 101.277933 | 0.110197 | 1.022645 | 0.400494 | jerk_dominated | 103 | 122 |
| 00002 | 42.433693 | 0.068312 | 0.238246 | 0.450143 | jerk_dominated | 378 | 397 |
| 00008 | 42.297874 | 0.071836 | 0.405723 | 0.389987 | jerk_dominated | 104 | 123 |
| 00010 | 37.582516 | 0.061338 | 0.210288 | 0.499454 | jerk_dominated | 456 | 475 |
| 00009 | 37.379314 | 0.059662 | 0.444363 | 0.523865 | jerk_dominated | 102 | 121 |
| 00003 | 25.603151 | 0.054551 | 0.279228 | 0.418858 | jerk_dominated | 395 | 414 |

Detailed notes: [[Diagnosis/route-00002-analysis]],
[[Diagnosis/route-00003-analysis]], [[Diagnosis/route-00006-analysis]],
[[Diagnosis/route-00008-analysis]], [[Diagnosis/route-00009-analysis]],
[[Diagnosis/route-00010-analysis]].
