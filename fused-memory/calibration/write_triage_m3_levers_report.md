# Write-triage M3 lever replay

Rubric v2.1; baseline arm `f3p`; provenance shipped_prompt_commit `326b773226`, shipped_prompt_sha256 `897885879ce30793ba614b837d1ba28f4732ae8ff27dea41eda65bb1d8fbded5`.
Soak M3 (population): 0.3790 (0.2686–0.4673), 39 of 371 sampled CONFLICT writes hidden.

## Arms

The cost columns include the calls of the arms each arm builds on.

| arm | M3 | M3 95% | ΔM3 vs base | Δ 95% | M1 | M5 | M2 contested | M2 amended | false contests added | M4 | harm Δ | calls/write | p50 s | p95 s | fail-open | USD/1k judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f3p | 0.4141 | 0.3104–0.5018 | — | — | 0.0529 | 0.8686 | 0.8686 | 0.4022–0.4022 | — | 0.4287–0.4295 | 0.1175–0.1180 | 0.8782 | 2.9110 | 4.5960 | 0 | 2.8888 |
| f3p-repeat | 0.3945 | 0.2934–0.4844 | -0.0196 | -0.0989–0.0560 | 0.0419 | 0.8700 | 0.8700 | 0.4066 | 1.0000 | 0.4198 | 0.1013 | 0.8782 | 2.8570 | 4.6290 | 0 | 2.8857 |
| a1-partial-dispute | 0.2809 | 0.1861–0.3746 | -0.1332 | -0.2224–-0.0467 | 0.0529 | 0.8022 | 0.8022 | 0.4246 | 46.2200 | 0.3882 | 0.0626 | 0.8782 | 3.0280 | 5.0110 | 0 | 3.0978 |
| c1-dispute-marker | 0.2505 | 0.1614–0.3407 | -0.1636 | -0.2551–-0.0715 | 0.0088 | 0.6523 | 0.6523 | 0.4296 | 161.3800 | 0.3776 | 0.1624 | 1.0698 | 3.1240 | 7.7510 | 0 | 3.4856 |
| d1-stale-detail-check | 0.1987 | 0.1222–0.2837 | -0.2154 | -0.3149–-0.1104 | 0.0088 | 0.6673 | 0.6673 | 0.4389 | 161.8800 | 0.3762 | 0.1369 | 0.8782 | 3.0410 | 5.1670 | 0 | 3.2976 |
| e1-mark-refused-contests | 0.3734 | 0.2656–0.4641 | -0.0407 | -0.0609–-0.0242 | 0.0529 | 0.8572 | 0.8572 | 0.4022–0.4022 | 8.2500 | 0.4243–0.4250 | 0.1125–0.1130 | 0.8782 | 2.9110 | 4.5960 | 0 | 2.8888 |
| d1-e1-combined | 0.0813 | 0.0435–0.1360 | -0.3327 | -0.4297–-0.2241 | 0.0088 | 0.6559 | 0.6559 | 0.4389 | 205.6100 | 0.3765 | 0.1553 | 0.8782 | 3.0410 | 5.1670 | 0 | 3.2976 |
| a1-e1-combined | 0.2136 | 0.1231–0.3073 | -0.2005 | -0.2935–-0.1104 | 0.0529 | 0.7954 | 0.7954 | 0.4246 | 59.4700 | 0.3821 | 0.0535 | 0.8782 | 3.0280 | 5.0110 | 0 | 3.0978 |
| b1-sol-medium | 0.4379 | 0.3359–0.5232 | 0.0238 | -0.0515–0.1033 | 0.0749 | 0.8336 | 0.8336 | 0.3924–0.3924 | 11.9900 | 0.4272–0.4279 | 0.1433–0.1439 | 0.8782 | 3.1820 | 6.1040 | 0 | 3.0575 |
| b2-sol-high | 0.4393 | 0.3365–0.5233 | 0.0252 | -0.0487–0.1049 | 0.0639 | 0.8327 | 0.8327 | 0.3889–0.3889 | 12.2400 | 0.4384–0.4391 | 0.1535–0.1540 | 0.8782 | 4.0830 | 9.7140 | 0 | 3.6103 |

## Own cost of the arms built on a parent

| arm | parent | calls/write | p50 s | p95 s | fail-open | USD/1k judged |
|---|---|---|---|---|---|---|
| c1-dispute-marker | f3p | 0.1917 | 3.4950 | 6.0010 | 0 | 2.7344 |
| e1-mark-refused-contests | f3p | 0.0000 | — | — | 0 | — |
| d1-e1-combined | d1-stale-detail-check | 0.0000 | — | — | 0 | — |
| a1-e1-combined | a1-partial-dispute | 0.0000 | — | — | 0 | — |

## Harness checks

| arm | agreement with the soak | 95% | n | fail-open |
|---|---|---|---|---|
| psi-repro | 0.7626 | 0.7262–0.7955 | 577 | 0 |

| arm | verdict agreement with the baseline | n |
|---|---|---|
| f3p-repeat | 0.9611 | 591 |
| a1-partial-dispute | 0.9323 | 591 |
| d1-stale-detail-check | 0.8697 | 591 |
| b1-sol-medium | 0.9543 | 591 |
| b2-sol-high | 0.9577 | 591 |

## Arms left partial

| arm | rows | cases | USD so far | projected USD |
|---|---|---|---|---|

## Classification of the soak M3 hits

39 hits, population weight 202.6100.

### action_changing

| action_changing | n | population |
|---|---|---|
| no | 10 | 45.4700 |
| tied | 5 | 17.2400 |
| yes | 24 | 139.9000 |

### cos_parent_bucket

| cos_parent_bucket | n | population |
|---|---|---|
| 0.60-0.70 | 5 | 39.2200 |
| 0.70-0.80 | 8 | 75.9400 |
| 0.80-0.90 | 17 | 65.2100 |
| <0.60 | 1 | 12.2400 |
| >=0.90 | 7 | 8.7500 |
| unknown | 1 | 1.2500 |

### cross_category

| cross_category | n | population |
|---|---|---|
| false | 25 | 152.1400 |
| true | 14 | 50.4700 |

### dispute_in_child_core

| dispute_in_child_core | n | population |
|---|---|---|
| false | 15 | 117.6600 |
| true | 24 | 84.9500 |

### dispute_kind

| dispute_kind | n | population |
|---|---|---|
| contradiction | 9 | 33.2300 |
| newer_measurement | 11 | 46.7200 |
| outdated_state | 11 | 57.7100 |
| status_overtaken | 8 | 64.9500 |

### home

| home | n | population |
|---|---|---|
| no | 16 | 140.8900 |
| yes | 23 | 61.7200 |

### project

| project | n | population |
|---|---|---|
| dark_factory | 10 | 78.4400 |
| reify | 5 | 17.2400 |
| solar_challenge | 19 | 67.7100 |
| solar_challenge_platform | 5 | 39.2200 |

### route

| route | n | population |
|---|---|---|
| force_stored | 5 | 6.2500 |
| judge | 32 | 193.8600 |
| supersession | 2 | 2.5000 |

### scope

| scope | n | population |
|---|---|---|
| central | 13 | 27.2400 |
| secondary | 25 | 174.1200 |
| tied | 1 | 1.2500 |

### self_declared_correction

| self_declared_correction | n | population |
|---|---|---|
| false | 32 | 182.8700 |
| true | 7 | 19.7400 |

### soak_band

| soak_band | n | population |
|---|---|---|
| judge | 15 | 172.6100 |
| none | 1 | 1.2500 |
| restated | 23 | 28.7500 |

### stratum

| stratum | n | population |
|---|---|---|
| amended-sample | 14 | 171.3600 |
| restated-sample | 25 | 31.2500 |

### via_sibling

| via_sibling | n | population |
|---|---|---|
| false | 19 | 144.6400 |
| none | 1 | 1.2500 |
| true | 19 | 56.7200 |
