# dark-factory's warm-lane bash tests

These are dark-factory's **own** copies of the project-agnostic warm-lane
bash tests, running against dark-factory's **own** script copies in
`orchestrator/scripts/warm-lane/` (relocated by task 3072, leaf α).

Ported by **task 3073**, PRD `plans/warm-lane-infra-repatriation-prd.md`
leaf α2 (Phase 1).

`.sh` files are not collected by pytest, so each of these is driven as one
parametrized pytest item by `orchestrator/tests/test_warm_lane_bash_suite.py`.
That driver also carries the two non-vacuity guards (`PORTED_TESTS` manifest,
`SCRIPT_COVERAGE` map) that keep a dropped or mis-globbed port from reporting
a vacuous green.

## Provenance

Source repo: **reify** (`/home/leo/src/reify`), path `tests/infra/<name>`.
Copied at reify HEAD `8489b49bfaefddd4abbe875a970661220dacbd57`
(2026-07-30; `tests/infra/` clean at that sha). Per-file last-touching commit
at that HEAD:

| File | lines | reify commit | date |
|---|---|---|---|
| `test_helpers.sh` | 459 | `2ac7b723b7` | 2026-07-28 |
| `test_warm_lane_disk_guard.sh` | 432 | `3662006952` | 2026-07-11 |
| `test_thin_warm_lane.sh` | 440 | `2ac7b723b7` | 2026-07-28 |
| `test_warm_lane_degenerate_ref.sh` | 550 | `5b8a44ad6e` | 2026-07-05 |
| `test_warm_lane_sizing_lifecycle.sh` | 672 | `62c0f188c5` | 2026-07-26 |
| `test_provision_warm_lane_fs.sh` | 1140 | `b37e00eaa6` | 2026-07-11 |
| `test_warm_lane_gc_sweep.sh` | 1230 | `973fde7955` | 2026-07-28 |
| `test_warm_lane_gc.sh` | 1939 | `973fde7955` | 2026-07-28 |
| `test_warm_lane_audit.sh` | 1999 | `973fde7955` | 2026-07-28 |

## Documented deltas from the reify sources

*(finalised in step-19 — see task 3073 plan)*

## Measured wall-clock

*(finalised in step-19 — see task 3073 plan)*

## Duplication window

reify's originals stay in place and green until PRD leaf **κ**, which deletes
them. This repo does not touch reify. Keeping the ported copies diffable
against their reify sources is the only cheap drift check available for the
whole α→κ window, which is why deltas are enumerated above rather than
absorbed.
