# Task 5416: cross-check parity window and OFF (closing note, 2026-10-04)

Executes Leo's ruling R11 as amended on 2026-10-01 (esc-5416-2): the window was shortened to one
day, the OFF step on both projects was pre-approved, and backstop task 6134 was added. Written by the
dark_factory L2 watcher from read-only probes of each project's `data/orchestrator/runs.db`
(`event_type LIKE 'verdict_parity%'`; timestamps UTC).

## Window: 2026-10-01T05:01:29Z to the OFF

Paired cross-check verdicts (payload `local_runner` / `remote_runner` / `passed`):

| project | verdicts | passed | mismatches | first | last |
|---|---|---|---|---|---|
| dark_factory | 13 | 13 | 0 | 10-01 05:42:06Z (task 4715) | 10-02 04:20:17Z (task 5177) |
| reify | 7 | 7 | 0 | 10-01 08:37:51Z (task 8094) | 10-02 05:54:39Z (task 7680) |

There were no `verdict_parity_mismatch` events on either project. Neither reached the original
30-verdict floor, which the one-day window superseded.

Reify also emitted three `verdict_parity_ok` rows of a different shape, with keys `coarse`,
`cold_test_count`, `merge_commit` and `shadow_compare`, and `passed` null (10-02 05:41Z, 10-03 04:59Z,
10-04 07:59Z). These come from the cold-test-count shadow comparison, not from the remote cross-check,
and are excluded above.

## OFF

`verify_cross_check_remote_green: false` was committed on both projects:

- dark_factory `5a217969e4` (2026-10-02 05:51:32Z)
- reify `dedd36e0cc` (2026-10-02 05:49:47Z)

The OFF took effect on both projects: neither has a cross-check-shaped verdict after
10-02 05:54:39Z. That last reify verdict landed five minutes after the reify commit, which fits a
verify that was already in flight.

## First post-OFF drift checks

Drift verdicts carry `local_category` / `remote_category` in the payload:

| project | drift verdicts since OFF | result | first |
|---|---|---|---|
| dark_factory | 4 (tasks 5209, 4076, 3886, 6193) | all passed / passed | 10-03 09:13:47Z (task 5209) |
| reify | 1 (task 7284) | passed / passed | 10-03 19:26:08Z (task 7284) |

The drift check does run with the cross-check OFF, but it skips a sampled landing when both verify hosts
are busy. That skip, and its DEBUG-only logging, are task 5349's scope; this note does not close it.

## Disposition

The ruling's steps are complete: the window was classified, the OFF was executed on both projects, the
first post-OFF drift checks passed on both, and this note is recorded. 5416 and esc-5416-2 close
operational-verified. 6134 is already done.
