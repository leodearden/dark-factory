# Load-threshold calibration — cut 2026-11-03T05:49+00:00

Corpus: `/home/leo/src/dark-factory/data/load-samples.db` (read-only).

## Percentiles

| metric | n | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| `runqueue_ratio` | 504567 | 2.781 | 6.875 | 11.75 | 19.812 | 43.344 |

## Candidate thresholds (PRD D11)

Target: the gate holds on <= 20% of ticks. The longest hold RUN is reported beside the fraction because the same fraction delivered as one long block and as isolated blips are opposite verdicts for a dispatch throttle. It is a hold streak, not a floor-occupancy measurement — in-flight floor occupancy is not in this corpus.

### `runqueue_ratio` — runnable threads per CPU (ratio)

Configured value in force: 4.0

Coverage: readable on 504571/504571 corpus ticks (100.0%), present on 504571/504571

| candidate | hold fraction | <= target | longest hold run |
|---|---|---|---|
| 1.0 | 93.0% | NO | 1598 ticks (2h18m) |
| 1.5 | 82.9% | NO | 929 ticks (1h18m) |
| 2.0 | 70.4% | NO | 605 ticks (52m) |
| 2.5 | 56.9% | NO | 409 ticks (34m) |
| 3.0 | 45.5% | NO | 346 ticks (30m) |
| 4.0 | 29.4% | NO | 209 ticks (18m) |
| 5.0 | 19.2% | yes | 202 ticks (17m) |
| 6.0 | 13.0% | yes | 182 ticks (16m) |
| 8.0 | 8.0% | yes | 103 ticks (9m) |


## Config drift against the peer project

Local: `/home/leo/src/dark-factory/dark-factory-orchestrator.yaml`  ·  Peer: `/home/leo/src/reify/dark-factory-orchestrator.yaml`

No drift: both blocks set the same leaves to the same values.

## Leaves that merely restate the shipped code default

PRD §6.2 writes three values that are already the code defaults, so one fact gains three homes with no reconciler (INV-9). This section names them; it never edits either file.

- **local**: restates the default: `io_some_avg10`, `mem_full_avg10`, `mem_some_avg10`; not in the model's schema: none
- **peer**: restates the default: `io_some_avg10`, `mem_full_avg10`, `mem_some_avg10`; not in the model's schema: none

## Degradations

_None._
