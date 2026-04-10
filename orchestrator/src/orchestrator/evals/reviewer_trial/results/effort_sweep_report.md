# Reviewer Panel Trial Report

Total trial cost: **$22.21**

## Variant Leaderboard

| Rank | Variant | Reviewers | Mean F1 | Blocking Recall | Cost | F1/$ | BR/$ |
|------|---------|-----------|---------|-----------------|------|------|------|
| 1 | variant_a | 1 | 0.472 | 0.878 | $5.21 | 0.091 | 0.169 |
| 2 | variant_a_medium | 1 | 0.436 | 0.789 | $6.76 | 0.064 | 0.117 |
| 3 | variant_a_max | 1 | 0.416 | 0.789 | $10.24 | 0.041 | 0.077 |

## Per-Language Breakdown

### variant_a

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.423 | 0.639 | 0.322 | 0.917 |
| rust | 0.571 | 0.778 | 0.454 | 0.833 |
| typescript | 0.371 | 0.500 | 0.317 | 0.889 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.315 | 0.389 | 0.278 | 0.500 |
| synthetic | 0.511 | 0.736 | 0.398 | 0.972 |

### variant_a_medium

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.386 | 0.722 | 0.269 | 0.917 |
| rust | 0.529 | 0.861 | 0.385 | 0.833 |
| typescript | 0.348 | 0.500 | 0.292 | 0.444 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.411 | 0.556 | 0.345 | 0.167 |
| synthetic | 0.442 | 0.778 | 0.313 | 0.944 |

### variant_a_max

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.456 | 0.861 | 0.315 | 1.000 |
| rust | 0.469 | 0.806 | 0.334 | 0.750 |
| typescript | 0.227 | 0.389 | 0.162 | 0.444 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.406 | 0.667 | 0.293 | 0.333 |
| synthetic | 0.418 | 0.764 | 0.292 | 0.903 |

## Per-Diff F1 Scores

| Diff | variant_a | variant_a_medium | variant_a_max |
|------|------|------|------|
| py_always_passes | 0.571 | 0.500 | 0.444 |
| py_deep_shallow | 0.333 | 0.222 | 0.400 |
| py_missing_await | 0.333 | 0.250 | 0.571 |
| py_obvious_copy | 0.571 | 0.500 | 0.500 |
| py_real_01 | 0.444 | 0.400 | 0.600 |
| py_swapped_args | 0.286 | 0.444 | 0.222 |
| rs_heavy_mutation_copy | 0.800 | 0.667 | 0.667 |
| rs_iter_vs_recurse | 0.727 | 0.667 | 0.615 |
| rs_leaked_resource | 0.400 | 0.308 | 0.286 |
| rs_missing_error_prop | 0.750 | 0.600 | 0.600 |
| rs_off_by_one | 0.750 | 0.600 | 0.364 |
| rs_real_01 | 0.000 | 0.333 | 0.286 |
| ts_obvious_copy | 0.250 | 0.364 | 0.182 |
| ts_real_01 | 0.500 | 0.500 | 0.333 |
| ts_shallow_mutate | 0.364 | 0.182 | 0.167 |

## Head-to-Head Comparisons

### variant_a_medium vs variant_a

- **variant_a_medium wins**: 3 diffs (py_swapped_args, rs_real_01, ts_obvious_copy)
- **variant_a wins**: 11 diffs (py_always_passes, py_deep_shallow, py_missing_await, py_obvious_copy, py_real_01, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_shallow_mutate)
- **Ties**: 1 diffs

### variant_a_medium vs variant_a_max

- **variant_a_medium wins**: 9 diffs (py_always_passes, py_swapped_args, rs_iter_vs_recurse, rs_leaked_resource, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_real_01, ts_shallow_mutate)
- **variant_a_max wins**: 3 diffs (py_deep_shallow, py_missing_await, py_real_01)
- **Ties**: 3 diffs

### variant_a vs variant_a_max

- **variant_a wins**: 11 diffs (py_always_passes, py_obvious_copy, py_swapped_args, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_obvious_copy, ts_real_01, ts_shallow_mutate)
- **variant_a_max wins**: 4 diffs (py_deep_shallow, py_missing_await, py_real_01, rs_real_01)
- **Ties**: 0 diffs

## Issues Missed by ALL Variants

- `py_pass_2`
- `rs_leak_3`
- `rs_real_dup_1`
- `ts_copy_2`
- `ts_shallow_1`
