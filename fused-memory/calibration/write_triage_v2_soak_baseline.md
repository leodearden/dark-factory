# Write-triage reader report (rubric v2)

Items: 553 (decided 552, unclear 1). By outcome: amended 40, contested 473, restated 40, stored 0.
Sampled per stratum: amended-sample 40, contested-census 473, restated-sample 40.
Stratum populations: amended-sample 1224, contested-census 473, restated-sample 125.
Bootstrap: 10000 resamples, seed 7157.

## Metrics

| metric | n | hits | rate | Wilson 95% | population rate | population 95% | unclear |
|---|---|---|---|---|---|---|---|
| M1 | 39 | 18 | 0.4615 | [0.3157, 0.6143] | 0.4697 | [0.3158, 0.6263] | 0 |
| M2 amended | 40 | 25 | 0.625 | [0.4703, 0.7578] | 0.625 | [0.475, 0.775] | 0 |
| M2 contested | 472 | 315 | 0.6674 | [0.6237, 0.7084] | 0.6674 | [0.625, 0.7091] | 1 |
| M2 restated | 40 | 13 | 0.325 | [0.2008, 0.4798] | 0.325 | [0.175, 0.475] | 0 |
| M2 stored | 0 | 0 | — | — | — | — | 0 |
| M3 | 340 | 15 | 0.0441 | [0.0269, 0.0715] | 0.3618 | [0.1728, 0.4986] | 1 |
| M4 | 552 | 127 | 0.2301 | [0.1969, 0.267] | 0.2974 | [0.204, 0.3979] | 1 |
| M4 direct | 426 | 57 | 0.1338 | [0.1047, 0.1694] | 0.1787 | [0.0907, 0.2831] | 0 |
| M4 via_sibling | 122 | 68 | 0.5574 | [0.4688, 0.6424] | 0.6732 | [0.4652, 0.8272] | 1 |
| M4 unknown route | 4 | 2 | 0.5 | [0.15, 0.85] | 0.6735 | [0.0, 1.0] | 0 |
| M4 same category | 363 | 87 | 0.2397 | [0.1986, 0.2862] | 0.3467 | [0.2229, 0.4755] | 0 |
| M4 cross category | 189 | 40 | 0.2116 | [0.1594, 0.2753] | 0.1874 | [0.0698, 0.3402] | 1 |
| M4 unknown category | 0 | 0 | — | — | — | — | 0 |
| M5 | 472 | 325 | 0.6886 | [0.6454, 0.7287] | 0.6886 | [0.6469, 0.7294] | 1 |
| M6 | 13 | 9 | — | — | — | — | 0 |

- M1: items of unknown band (outside M1): 4.
- M3 by ending: restated 10, amended 5, stored_unlinked 0, contested 325, stored_linked 0.
- M5 by scope: central 166, secondary 156, unknown 3.
- M6: n is attach-time-rated items, hits the drifted ones (28bd2462-f06f-4194-9cac-a13c461cf4ec, 3c88f869-8247-419b-a985-30c966d68afd, 51f55850-df72-4eda-8f03-77c24dc7a75b, 5b5850ac-5732-4523-80a2-8304aeb3b594, 6d648544-f6a7-4a5c-a68c-ec785825466a, 777f6cb1-8a75-4761-8717-7465e6dfb875, 8f46293e-67ad-41d1-a0a0-da9186f05b72, f57607c7-243a-41e5-9461-3c51404d540c, fa50625d-8f99-4041-85eb-3a00a559fc59); parent edited after attach 13, of which unrated at attach time 0.

## Cells

| cell | triage | status quo | weight | rule |
|---|---|---|---|---|
| hidden_detail | 8 | 0 | 1 | A restated write adds something a reader acting on its record would need (HOME, LOSS, no CONFLICT), and a sighting is only counted, never shown. |
| hidden_conflict | 10 | 0 | 3 | A restated write disputes its record (CONFLICT), and the dispute sits in a sighting a reader of the record never sees. |
| digest_conflict | 5 | 0 | 2 | An amended write disputes its record (CONFLICT), and a reader of the record meets the dispute only as an untagged 240-character digest. |
| wrong_home | 127 | 0 | 1 | An attached write is not about its record's claim (HOME=NO); a contest whose dispute is true is never a wrong home (ruling 5b). |
| false_conflict | 147 | 0 | 2 | A contested write does not dispute its record (CONFLICT=NO), so a reader is told a sound record is in doubt. |
| duplicate_clutter | 0 | 20 | 0.5 | A standalone write restates an earlier record and adds nothing a reader needs (HOME, no LOSS, no CONFLICT). |
| orphaned_conflict | 0 | 340 | 1.5 | A standalone write disputes an earlier record (CONFLICT) that carries no superseded-by mark for it, so a reader of that record never learns of it. |
| attached_supersession | 12 | 0 | 1 | A write declaring that it supersedes its record was attached under it; a supersession is a new standalone record plus a mark, never an attach (ruling 5c). |

## Harm per write (weights: provisional)

| figure | triage | status quo | delta | delta 95% |
|---|---|---|---|---|
| per write (sample) | 0.8714 | 0.942 | -0.0707 | [-0.2228, 0.0824] |
| per write (population) | 0.701 | 0.4569 | 0.2441 | [0.1154, 0.3789] |

Over 552 decided writes; unclear writes left out: 1.

## Rater agreement (primary vs second)

Floor: kappa >= 0.7 on at least 60 doubly rated pairs.

| binary | n overlap | kappa | floor met |
|---|---|---|---|
| home | 3 | 0.0 | no |
| loss | 80 | 0.3962 | no |
| conflict | 74 | 0.5826 | no |
| scope | 18 | 0.4906 | no |
| digest | 48 | -0.036 | no |
