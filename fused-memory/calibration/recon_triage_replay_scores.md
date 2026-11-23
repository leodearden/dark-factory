# Recon triage replay scores (rubric v2.2)

Headline children: 160; report-only: 12; touched (live recon state known): 14.
Sampled per stratum: amended|dark_factory 10, amended|reify 10, amended|solar_challenge 10, amended|solar_challenge_platform 10, contested|dark_factory 20, contested|reify 20, contested|solar_challenge 20, contested|solar_challenge_platform 20, restated|dark_factory 10, restated|reify 10, restated|solar_challenge 10, restated|solar_challenge_platform 10.
Stratum weights: amended|dark_factory 33.048, amended|reify 34.272, amended|solar_challenge 36.72, amended|solar_challenge_platform 17.136, contested|dark_factory 6.25, contested|reify 5.8, contested|solar_challenge 6.6, contested|solar_challenge_platform 4.9, restated|dark_factory 1.75, restated|reify 2.375, restated|solar_challenge 6.75, restated|solar_challenge_platform 1.625.
Bootstrap: 10000 resamples, seed 7161.

## Headline

| metric | n | hits | rate | Wilson 95% | population rate | population 95% | correct contests detached |
|---|---|---|---|---|---|---|---|
| action_accuracy | 160 | 99 | 0.6188 | [0.5415, 0.6904] | 0.5809 | [0.4828, 0.6788] | — |
| contest_resolution_accuracy | 80 | 45 | 0.5625 | [0.4534, 0.6659] | 0.557 | [0.4478, 0.6663] | — |
| judge_wrong_home_all | 160 | 63 | 0.3937 | [0.3214, 0.4711] | 0.462 | [0.3583, 0.5675] | — |
| judge_wrong_home_contested | 80 | 21 | 0.2625 | [0.1786, 0.3682] | 0.26 | [0.1668, 0.3576] | — |
| misfile_detection | 63 | 40 | 0.6349 | [0.5115, 0.7428] | 0.4423 | [0.2905, 0.6076] | — |
| wrong_detach_rate | 97 | 12 | 0.1237 | [0.0722, 0.2039] | 0.0747 | [0.0392, 0.118] | 12 |

## Action accuracy by outcome

| outcome | n | hits | rate | Wilson 95% | population rate | population 95% |
|---|---|---|---|---|---|---|
| amended | 40 | 23 | 0.575 | [0.4219, 0.7149] | 0.5697 | [0.4253, 0.7131] |
| contested | 80 | 45 | 0.5625 | [0.4534, 0.6659] | 0.557 | [0.4488, 0.665] |
| restated | 40 | 31 | 0.775 | [0.625, 0.8768] | 0.78 | [0.615, 0.918] |

## By project

| project | metric | n | hits | rate | Wilson 95% | population rate | population 95% | correct contests detached |
|---|---|---|---|---|---|---|---|---|
| dark_factory | action_accuracy | 40 | 22 | 0.55 | [0.3983, 0.6929] | 0.4507 | [0.2374, 0.6794] | — |
| dark_factory | contest_resolution_accuracy | 20 | 11 | 0.55 | [0.3421, 0.7418] | 0.55 | [0.35, 0.75] | — |
| dark_factory | judge_wrong_home_all | 40 | 17 | 0.425 | [0.2851, 0.5781] | 0.5546 | [0.3355, 0.7436] | — |
| dark_factory | judge_wrong_home_contested | 20 | 3 | 0.15 | [0.0524, 0.3604] | 0.15 | [0.0, 0.3] | — |
| dark_factory | misfile_detection | 17 | 8 | 0.4706 | [0.2616, 0.6904] | 0.1898 | [0.0322, 0.4752] | — |
| dark_factory | wrong_detach_rate | 23 | 4 | 0.1739 | [0.0698, 0.3714] | 0.1187 | [0.0253, 0.2725] | 4 |
| reify | action_accuracy | 40 | 29 | 0.725 | [0.5716, 0.8389] | 0.83 | [0.6639, 0.9371] | — |
| reify | contest_resolution_accuracy | 20 | 13 | 0.65 | [0.4329, 0.8188] | 0.65 | [0.45, 0.85] | — |
| reify | judge_wrong_home_all | 40 | 9 | 0.225 | [0.1232, 0.375] | 0.271 | [0.082, 0.4912] | — |
| reify | judge_wrong_home_contested | 20 | 4 | 0.2 | [0.0807, 0.416] | 0.2 | [0.05, 0.4] | — |
| reify | misfile_detection | 9 | 6 | 0.6667 | [0.3542, 0.8794] | 0.6754 | [0.1664, 1.0] | — |
| reify | wrong_detach_rate | 31 | 0 | 0.0 | [0.0, 0.1103] | 0.0 | [0.0, 0.0] | 0 |
| solar_challenge | action_accuracy | 40 | 21 | 0.525 | [0.375, 0.6706] | 0.4593 | [0.2535, 0.6707] | — |
| solar_challenge | contest_resolution_accuracy | 20 | 9 | 0.45 | [0.2582, 0.6579] | 0.45 | [0.25, 0.65] | — |
| solar_challenge | judge_wrong_home_all | 40 | 17 | 0.425 | [0.2851, 0.5781] | 0.5179 | [0.3111, 0.7123] | — |
| solar_challenge | judge_wrong_home_contested | 20 | 7 | 0.35 | [0.1812, 0.5671] | 0.35 | [0.15, 0.55] | — |
| solar_challenge | misfile_detection | 17 | 10 | 0.5882 | [0.3601, 0.7839] | 0.4321 | [0.1587, 0.7602] | — |
| solar_challenge | wrong_detach_rate | 23 | 5 | 0.2174 | [0.0966, 0.419] | 0.1208 | [0.0347, 0.2514] | 5 |
| solar_challenge_platform | action_accuracy | 40 | 27 | 0.675 | [0.5202, 0.7992] | 0.6171 | [0.4171, 0.8028] | — |
| solar_challenge_platform | contest_resolution_accuracy | 20 | 12 | 0.6 | [0.3866, 0.7812] | 0.6 | [0.4, 0.8] | — |
| solar_challenge_platform | judge_wrong_home_all | 40 | 20 | 0.5 | [0.352, 0.648] | 0.5199 | [0.3142, 0.7142] | — |
| solar_challenge_platform | judge_wrong_home_contested | 20 | 7 | 0.35 | [0.1812, 0.5671] | 0.35 | [0.15, 0.55] | — |
| solar_challenge_platform | misfile_detection | 20 | 16 | 0.8 | [0.584, 0.9193] | 0.7032 | [0.4119, 0.9632] | — |
| solar_challenge_platform | wrong_detach_rate | 20 | 3 | 0.15 | [0.0524, 0.3604] | 0.1072 | [0.0, 0.2537] | 3 |

## Confusion (expected repair by recon repair)

| expected | contest_upheld | detach | keep | not_a_contest | relabel->amended | relabel->contested | superseded_parent |
|---|---|---|---|---|---|---|---|
| contest_upheld | 28 | 12 | 0 | 13 | 0 | 0 | 0 |
| detach | 1 | 36 | 13 | 7 | 1 | 0 | 0 |
| keep | 0 | 0 | 27 | 0 | 1 | 0 | 0 |
| not_a_contest | 1 | 0 | 0 | 3 | 0 | 0 | 0 |
| relabel->amended | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| relabel->contested | 0 | 4 | 1 | 0 | 3 | 3 | 0 |
| superseded_parent | 1 | 0 | 0 | 0 | 1 | 1 | 1 |

## Harm per write (weights: provisional)

| scope | figure | n priced | judge | recon | status quo | recon − judge | recon − judge 95% |
|---|---|---|---|---|---|---|---|
| all | sample | 159 | 0.9686 | 0.6226 | 0.6981 | -0.3459 | [-0.5692, -0.1258] |
| all | population | 159 | 0.8167 | 0.5708 | 0.4657 | -0.246 | [-0.3935, -0.103] |
| dark_factory | sample | 40 | 0.775 | 0.6375 | 0.775 | -0.1375 | [-0.4875, 0.2] |
| dark_factory | population | 40 | 0.6826 | 0.6341 | 0.4235 | -0.0485 | [-0.2768, 0.1562] |
| reify | sample | 39 | 0.7179 | 0.5128 | 0.7051 | -0.2051 | [-0.6923, 0.2821] |
| reify | population | 39 | 0.4409 | 0.2815 | 0.3883 | -0.1593 | [-0.4408, 0.1106] |
| solar_challenge | sample | 40 | 1.225 | 0.8 | 0.725 | -0.425 | [-0.8875, 0.0125] |
| solar_challenge | population | 40 | 1.1064 | 0.7724 | 0.5256 | -0.334 | [-0.6366, -0.0564] |
| solar_challenge_platform | sample | 40 | 1.15 | 0.5375 | 0.5875 | -0.6125 | [-1.0625, -0.15] |
| solar_challenge_platform | population | 40 | 1.0915 | 0.5487 | 0.5459 | -0.5428 | [-0.9244, -0.1512] |

Weights: attached_supersession 1, digest_conflict 2, duplicate_clutter 0.5, false_conflict 2, hidden_conflict 3, hidden_detail 1, orphaned_conflict 1.5, wrong_home 1.

## Missing and unpriced

| kind | count | children |
|---|---|---|
| no_decision | 0 |  |
| unpriced | 1 | 9b472ee2-23ef-4ab7-8773-97784a958086 (not_a_contest, incomplete) |

## Touched children (recon or an agent acted on them live)

| entry | project | outcome | headline | live recon state | truth cells | expected | recon | correct | judge harm | recon harm |
|---|---|---|---|---|---|---|---|---|---|---|
| 024ca326-c8da-492f-b14e-3f8ddf7b95e0 | solar_challenge | contested | no | relinked | — | contest_upheld | detach | no | 0 | 1.5 |
| 092c0560-ac75-40fe-a634-e35bacd973f3 | solar_challenge_platform | contested | yes | missing | — | contest_upheld | not_a_contest | no | 0 | 2 |
| 3df2714b-cedd-44c6-8e5d-fa5eb6a21cd7 | solar_challenge | restated | no | relinked | wrong_home | detach | detach | yes | 1 | 0 |
| 58959f97-206c-4789-94e0-4477359745b8 | solar_challenge | contested | no | relinked | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| 5b5850ac-5732-4523-80a2-8304aeb3b594 | solar_challenge | contested | no | relinked | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| 61fe3b21-af0c-4358-b5b8-4711d9cb56d3 | solar_challenge | contested | no | missing | — | contest_upheld | detach | no | 0 | 1.5 |
| 81711e8a-b5db-4683-a7bf-8a03d3efc41a | solar_challenge | contested | no | relinked | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| 911ee809-bed1-4f90-93a1-3e1bbcbbff0d | solar_challenge | restated | yes | relinked | wrong_home | detach | detach | yes | 1 | 0 |
| 95a00908-8611-4dc3-a67f-c1fc04a2a489 | solar_challenge | restated | no | missing | hidden_conflict, wrong_home | relabel->contested | detach | no | 4 | 1.5 |
| b6b9d450-457b-4632-8fa6-de0ce1a8ea17 | dark_factory | contested | no | missing | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| de109f51-4da2-4ba5-a24d-5bf3925e7128 | solar_challenge | contested | no | relinked | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| e15ddb6f-0096-44d1-9e2e-69fafe6b786b | solar_challenge | contested | no | relinked | false_conflict, wrong_home | detach | detach | yes | 3 | 0 |
| e511b7ae-c368-40b4-8f44-bbef521c46af | solar_challenge | contested | no | relinked | — | contest_upheld | not_a_contest | no | 0 | 3 |
| f57607c7-243a-41e5-9461-3c51404d540c | solar_challenge | contested | no | relinked | — | contest_upheld | contest_upheld | yes | 0 | 0 |
