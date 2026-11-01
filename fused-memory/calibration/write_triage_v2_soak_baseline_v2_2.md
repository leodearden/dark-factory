# Write-triage reader report (rubric v2.2)

Items: 673 (decided 671, unclear 2). By outcome: amended 100, contested 473, restated 100, stored 0.
Sampled per stratum: amended-sample 100, contested-census 473, restated-sample 100.
Stratum populations: amended-sample 1224, contested-census 473, restated-sample 125.
Bootstrap: 10000 resamples, seed 7157.

## Metrics

| metric | n | hits | rate | Wilson 95% | population rate | population 95% | unclear |
|---|---|---|---|---|---|---|---|
| M1 | 91 | 28 | 0.3077 | [0.2223, 0.4087] | 0.3084 | [0.2149, 0.4054] | 1 |
| M1 as amended | 91 | 29 | 0.3187 | [0.232, 0.4201] | 0.3172 | [0.2225, 0.4157] | 1 |
| M2 amended | 100 | 31 | 0.31 | [0.2278, 0.4063] | 0.31 | [0.22, 0.4] | 0 |
| M2 contested | 472 | 322 | 0.6822 | [0.6389, 0.7226] | 0.6822 | [0.6385, 0.7246] | 1 |
| M2 restated | 99 | 29 | 0.2929 | [0.2124, 0.389] | 0.2929 | [0.202, 0.3878] | 1 |
| M2 stored | 0 | 0 | — | — | — | — | 0 |
| M3 | 371 | 39 | 0.1051 | [0.0779, 0.1405] | 0.379 | [0.2686, 0.4673] | 2 |
| M4 | 671 | 234 | 0.3487 | [0.3136, 0.3856] | 0.5422 | [0.4767, 0.6048] | 2 |
| M4 direct | 503 | 127 | 0.2525 | [0.2165, 0.2922] | 0.4658 | [0.3882, 0.5418] | 1 |
| M4 via_sibling | 164 | 105 | 0.6402 | [0.5644, 0.7097] | 0.7772 | [0.6712, 0.8629] | 1 |
| M4 unknown route | 4 | 2 | 0.5 | [0.15, 0.85] | 0.5294 | [0.0, 1.0] | 0 |
| M4 same category | 445 | 157 | 0.3528 | [0.3098, 0.3983] | 0.5474 | [0.4663, 0.6216] | 1 |
| M4 cross category | 226 | 77 | 0.3407 | [0.282, 0.4047] | 0.5314 | [0.4125, 0.6395] | 1 |
| M4 unknown category | 0 | 0 | — | — | — | — | 0 |
| M5 | 472 | 332 | 0.7034 | [0.6607, 0.7428] | 0.7034 | [0.661, 0.7442] | 1 |
| M6 | 20 | 11 | — | — | — | — | 0 |

- M1: items of unknown band (outside M1): 4. M1 as amended reads every band attach as if it were amended (report-only).
- M3 by ending: restated 25, amended 14, stored_unlinked 0, contested 332, stored_linked 0.
- M5 by scope: central 158, secondary 140, unknown 34.
- M5 by action-changing: yes 273, no 32, unknown 27.
- M6: n is attach-time-rated items, hits the drifted ones (0944c8af-4c77-46f0-802f-fef0d93b4e7f, 28bd2462-f06f-4194-9cac-a13c461cf4ec, 3c88f869-8247-419b-a985-30c966d68afd, 434bbd68-0ad2-4b69-9b8d-37e0955e303e, 51f55850-df72-4eda-8f03-77c24dc7a75b, 56c08b07-48f9-487d-8b87-27e74f597415, 5b5850ac-5732-4523-80a2-8304aeb3b594, 6d648544-f6a7-4a5c-a68c-ec785825466a, 777f6cb1-8a75-4761-8717-7465e6dfb875, 8f46293e-67ad-41d1-a0a0-da9186f05b72, fa50625d-8f99-4041-85eb-3a00a559fc59); parent edited after attach 20, of which unrated at attach time 0.

## Cells

| cell | triage | status quo | weight | rule |
|---|---|---|---|---|
| hidden_detail | 7 | 0 | 1 | A restated write adds something a reader acting on its record would need (HOME, LOSS, no CONFLICT), and a sighting is only counted, never shown. |
| hidden_conflict | 25 | 0 | 3 | A restated write disputes its record (CONFLICT), and the dispute sits in a sighting a reader of the record never sees. |
| digest_conflict | 14 | 0 | 2 | An amended write disputes its record (CONFLICT), and a reader of the record meets the dispute only as an untagged 240-character digest. |
| wrong_home | 234 | 0 | 1 | An attached write is not about its record's claim (HOME=NO); a contest whose dispute is true is never a wrong home (ruling 5b). |
| false_conflict | 140 | 0 | 2 | A contested write does not dispute its record (CONFLICT=NO), so a reader is told a sound record is in doubt. |
| duplicate_clutter | 0 | 51 | 0.5 | A standalone write restates an earlier record and adds nothing a reader needs (HOME, no LOSS, no CONFLICT). |
| orphaned_conflict | 0 | 371 | 1.5 | A standalone write disputes an earlier record (CONFLICT) that carries no superseded-by mark for it, so a reader of that record never learns of it. |
| attached_supersession | 12 | 0 | 1 | A write declaring that it supersedes its record was attached under it; a supersession is a new standalone record plus a mark, never an attach (ruling 5c). |

## Harm per write (weights: provisional)

| figure | triage | status quo | delta | delta 95% |
|---|---|---|---|---|
| per write (sample) | 0.9478 | 0.8674 | 0.0805 | [-0.0507, 0.2146] |
| per write (population) | 0.9476 | 0.5061 | 0.4415 | [0.3421, 0.5397] |

Over 671 decided writes; unclear writes left out: 2.

## Rater agreement (primary vs second, consumed answers)

Floor: kappa >= 0.7 on at least 60 consumed pairs, for home, conflict, loss; every other binary is report-only.

| binary | n consumed | kappa | floor met | floor applies |
|---|---|---|---|---|
| home | 223 | 0.9371 | yes | yes |
| loss | 81 | 0.8652 | yes | yes |
| conflict | 423 | 0.8372 | yes | yes |
| scope | 286 | 0.844 | yes | no |
| action_changing | 286 | 0.7744 | yes | no |
| digest | 4 | — | no | no |
