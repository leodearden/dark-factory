# Reviewer Panel Trial Report

Total trial cost: **$53.82**

## Variant Leaderboard

| Rank | Variant | Reviewers | Mean F1 | Blocking Recall | Cost | F1/$ | BR/$ |
|------|---------|-----------|---------|-----------------|------|------|------|
| 1 | variant_a | 1 | 0.457 | 0.844 | $5.21 | 0.088 | 0.162 |
| 2 | variant_b | 2 | 0.456 | 0.778 | $8.08 | 0.056 | 0.096 |
| 3 | variant_c | 3 | 0.358 | 0.878 | $12.05 | 0.030 | 0.073 |
| 4 | variant_d | 3 | 0.341 | 0.856 | $10.79 | 0.032 | 0.079 |
| 5 | baseline | 5 | 0.243 | 0.844 | $17.69 | 0.014 | 0.048 |

## Per-Language Breakdown

### variant_a

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.386 | 0.583 | 0.294 | 0.833 |
| rust | 0.571 | 0.778 | 0.454 | 0.833 |
| typescript | 0.371 | 0.500 | 0.317 | 0.889 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.241 | 0.278 | 0.222 | 0.333 |
| synthetic | 0.511 | 0.736 | 0.398 | 0.972 |

### variant_b

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.407 | 0.722 | 0.283 | 0.833 |
| rust | 0.571 | 0.819 | 0.474 | 0.833 |
| typescript | 0.325 | 0.556 | 0.233 | 0.556 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.227 | 0.278 | 0.208 | 0.000 |
| synthetic | 0.513 | 0.840 | 0.385 | 0.972 |

### variant_c

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.375 | 1.000 | 0.234 | 1.000 |
| rust | 0.400 | 0.903 | 0.262 | 0.917 |
| typescript | 0.237 | 0.556 | 0.153 | 0.556 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.246 | 0.667 | 0.151 | 0.667 |
| synthetic | 0.385 | 0.924 | 0.248 | 0.931 |

### variant_d

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.337 | 0.944 | 0.207 | 0.917 |
| rust | 0.371 | 0.903 | 0.235 | 1.000 |
| typescript | 0.287 | 0.611 | 0.194 | 0.444 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.335 | 0.722 | 0.224 | 0.500 |
| synthetic | 0.342 | 0.896 | 0.213 | 0.944 |

### baseline

| Language | F1 | Recall | Precision | Blocking Recall |
|----------|----|--------|-----------|-----------------|
| python | 0.243 | 1.000 | 0.139 | 1.000 |
| rust | 0.271 | 0.917 | 0.159 | 0.833 |
| typescript | 0.187 | 0.611 | 0.111 | 0.556 |

| Source | F1 | Recall | Precision | Blocking Recall |
|--------|----|--------|-----------|-----------------|
| real_world | 0.209 | 0.667 | 0.125 | 0.333 |
| synthetic | 0.251 | 0.944 | 0.146 | 0.972 |

## Per-Diff F1 Scores

| Diff | variant_a | variant_b | variant_c | variant_d | baseline |
|------|------|------|------|------|------|
| py_always_passes | 0.571 | 0.545 | 0.429 | 0.353 | 0.231 |
| py_deep_shallow | 0.333 | 0.571 | 0.500 | 0.364 | 0.286 |
| py_missing_await | 0.333 | 0.286 | 0.364 | 0.400 | 0.267 |
| py_obvious_copy | 0.571 | 0.571 | 0.250 | 0.267 | 0.191 |
| py_real_01 | 0.222 | 0.182 | 0.375 | 0.308 | 0.273 |
| py_swapped_args | 0.286 | 0.286 | 0.333 | 0.333 | 0.210 |
| rs_heavy_mutation_copy | 0.800 | 1.000 | 0.500 | 0.500 | 0.286 |
| rs_iter_vs_recurse | 0.727 | 0.462 | 0.300 | 0.286 | 0.320 |
| rs_leaked_resource | 0.400 | 0.364 | 0.210 | 0.250 | 0.250 |
| rs_missing_error_prop | 0.750 | 0.500 | 0.600 | 0.429 | 0.353 |
| rs_off_by_one | 0.750 | 0.600 | 0.429 | 0.400 | 0.261 |
| rs_real_01 | 0.000 | 0.500 | 0.364 | 0.364 | 0.154 |
| ts_obvious_copy | 0.250 | 0.667 | 0.500 | 0.429 | 0.200 |
| ts_real_01 | 0.500 | 0.000 | 0.000 | 0.333 | 0.200 |
| ts_shallow_mutate | 0.364 | 0.308 | 0.210 | 0.100 | 0.160 |

## Head-to-Head Comparisons

### baseline vs variant_a

- **baseline wins**: 2 diffs (py_real_01, rs_real_01)
- **variant_a wins**: 13 diffs (py_always_passes, py_deep_shallow, py_missing_await, py_obvious_copy, py_swapped_args, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_obvious_copy, ts_real_01, ts_shallow_mutate)
- **Ties**: 0 diffs

### baseline vs variant_b

- **baseline wins**: 2 diffs (py_real_01, ts_real_01)
- **variant_b wins**: 13 diffs (py_always_passes, py_deep_shallow, py_missing_await, py_obvious_copy, py_swapped_args, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_shallow_mutate)
- **Ties**: 0 diffs

### baseline vs variant_c

- **baseline wins**: 3 diffs (rs_iter_vs_recurse, rs_leaked_resource, ts_real_01)
- **variant_c wins**: 12 diffs (py_always_passes, py_deep_shallow, py_missing_await, py_obvious_copy, py_real_01, py_swapped_args, rs_heavy_mutation_copy, rs_missing_error_prop, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_shallow_mutate)
- **Ties**: 0 diffs

### baseline vs variant_d

- **baseline wins**: 2 diffs (rs_iter_vs_recurse, ts_shallow_mutate)
- **variant_d wins**: 12 diffs (py_always_passes, py_deep_shallow, py_missing_await, py_obvious_copy, py_real_01, py_swapped_args, rs_heavy_mutation_copy, rs_missing_error_prop, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_real_01)
- **Ties**: 1 diffs

### variant_a vs variant_b

- **variant_a wins**: 9 diffs (py_always_passes, py_missing_await, py_real_01, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_real_01, ts_shallow_mutate)
- **variant_b wins**: 4 diffs (py_deep_shallow, rs_heavy_mutation_copy, rs_real_01, ts_obvious_copy)
- **Ties**: 2 diffs

### variant_a vs variant_c

- **variant_a wins**: 9 diffs (py_always_passes, py_obvious_copy, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_real_01, ts_shallow_mutate)
- **variant_c wins**: 6 diffs (py_deep_shallow, py_missing_await, py_real_01, py_swapped_args, rs_real_01, ts_obvious_copy)
- **Ties**: 0 diffs

### variant_a vs variant_d

- **variant_a wins**: 9 diffs (py_always_passes, py_obvious_copy, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, ts_real_01, ts_shallow_mutate)
- **variant_d wins**: 6 diffs (py_deep_shallow, py_missing_await, py_real_01, py_swapped_args, rs_real_01, ts_obvious_copy)
- **Ties**: 0 diffs

### variant_b vs variant_c

- **variant_b wins**: 10 diffs (py_always_passes, py_deep_shallow, py_obvious_copy, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_shallow_mutate)
- **variant_c wins**: 4 diffs (py_missing_await, py_real_01, py_swapped_args, rs_missing_error_prop)
- **Ties**: 1 diffs

### variant_b vs variant_d

- **variant_b wins**: 11 diffs (py_always_passes, py_deep_shallow, py_obvious_copy, rs_heavy_mutation_copy, rs_iter_vs_recurse, rs_leaked_resource, rs_missing_error_prop, rs_off_by_one, rs_real_01, ts_obvious_copy, ts_shallow_mutate)
- **variant_d wins**: 4 diffs (py_missing_await, py_real_01, py_swapped_args, ts_real_01)
- **Ties**: 0 diffs

### variant_c vs variant_d

- **variant_c wins**: 8 diffs (py_always_passes, py_deep_shallow, py_real_01, rs_iter_vs_recurse, rs_missing_error_prop, rs_off_by_one, ts_obvious_copy, ts_shallow_mutate)
- **variant_d wins**: 4 diffs (py_missing_await, py_obvious_copy, rs_leaked_resource, ts_real_01)
- **Ties**: 3 diffs

## Issues Missed by ALL Variants

- `ts_shallow_1`
