# Task 4528 — `metadata.modules` → `metadata.files` migration: run evidence

Run 2026-08-21 from branch `task/4528` against the live fused-memory server at
`http://127.0.0.1:8002`, covering all seven known project roots. This file is
the durable record of the run; the script is
`scripts/migrate_metadata_modules_to_files.py` and is idempotent, so every
figure below is re-derivable by re-running the dry-run phase.

The run is PART OF THE TASK, not a follow-up. It is recorded here rather than
only in the PR description so a later reader can recover the numbers.

## 0. Commands

Every phase was one invocation carrying all seven `--project-root` flags, run
in the foreground:

```bash
uv run --project shared python scripts/migrate_metadata_modules_to_files.py [--dry-run] \
  --project-root /home/leo/src/dark-factory \
  --project-root /home/leo/src/reify \
  --project-root /home/leo/src/autopilot-video \
  --project-root /home/leo/src/know-live \
  --project-root /home/leo/src/pump-web-ui \
  --project-root /home/leo/src/solar-challenge \
  --project-root /home/leo/src/solar-challenge-platform
```

Phases: (1) `--dry-run` — before; (2) live; (3) `--dry-run` — after; then a
targeted retry of the transient failures and (4) a final `--dry-run`.

A **1-write canary** was run live on `solar-challenge` (the smallest corpus
with a pending action) between phases 1 and 2, deliberately, so that a
systematic gate rejection would surface having touched one record rather than
437. It succeeded in 1.8s.

## 1. Before (phase 1 dry-run)

| project | visited | copied | sanitized_empty | dropped |
|---|---|---|---|---|
| dark-factory | 904 | 0 | 4 | 160 |
| reify | 692 | 0 | 7 | 265 |
| autopilot-video | 6 | 0 | 0 | 0 |
| know-live | 5 | 0 | 0 | 0 |
| pump-web-ui | 4 | 0 | 0 | 0 |
| solar-challenge | 7 | 0 | 0 | 1 |
| solar-challenge-platform | 3 | 0 | 0 | 0 |
| **total** | **1621** | **0** | **11** | **426** |

`copied: 0` is the headline and it is the predicted result, not a surprise.
**All 11 copy-branch tasks across all seven corpora are 100% directory-shaped**,
so every one of them sanitizes to the empty list through
`shared.locking.strip_directory_locks` and lands as `sanitized_empty`: `files`
is left empty and scope is deferred to the architect. Pre-fix, all 11 would
have been REJECTED by `_reject_directory_locks_in_update_metadata`. There is no
file-shaped copy anywhere in the live data — that branch is covered by test
only.

Divergence from the planning baseline (measured ~4h earlier): `visited` 1621
identical; `dropped` 426 vs 427 (reify 265 vs 266). One reify task reached a
terminal status in the interval. This is live, continuously-mutated state; the
divergence is immaterial and was not forced.

## 2. Live run (phase 2)

431 of 436 pending writes landed. 3m25s wall clock. Five failures, in two
classes — **the reply classifier added by this task is what made them visible
at all**; pre-fix, `call_tool` raised only on a JSON-RPC-level error, so every
one of these would have printed as a success.

### Class B — transient, 4 tasks (RESOLVED)

`dark-factory` 3791, 3792, 3794 and `reify` 5810, each reported as
`update_task raised:` with an empty exception message — consistent with the
30s client-side read timeout under 437 back-to-back writes.

Re-measured rather than assumed: all four succeeded on a targeted retry
(dark-factory's three in 4.4s total, reify 5810 in 11s). Phase 3 confirmed none
of them had partially applied — their `modules` was still present before the
retry, so there is no ghost/partial write.

### Class A — deterministic, 1 task (NOT RESOLVED, escalated)

`reify` **5050** is rejected every time, reproduced twice:

```
[error][/home/leo/src/reify] task=5050 action=copy-sanitized-empty update_task rejected:
  server returned an error: TypeError: shared.task_metadata.Milestone() argument
  after ** must be a mapping, not bool
```

The cause is a PRE-EXISTING malformed field on that record, not anything this
migration does: `metadata.milestone` is the bool `True`, where the server's
deserializer expects the dated/delayed mapping documented in
`docs/task-authoring.md`. Confirmed by reading the record with `get_task`
(`"milestone": true`, `updatedAt` 2026-07-19).

**The record is therefore unwritable by ANY `update_task` caller**, not merely
by this script — the failure is on the metadata deserialization path, before
this migration's payload is ever considered. The read path is unaffected, which
is why `get_tasks`/`get_task` return it fine.

Census of the class across all seven corpora (read-only probe):

| project | non-mapping `milestone`, total | of which NON-TERMINAL |
|---|---|---|
| reify | 216 | **1** (id 5050) |
| all six others | 0 | 0 |

So the live blast radius is exactly one task. The other 215 are terminal
(done/cancelled/deferred) and are already never written; they are a latent
landmine for a future tool, not an active one.

## 3. After (phase 4 final dry-run)

| project | visited | copied | sanitized_empty | dropped | pending actions |
|---|---|---|---|---|---|
| dark-factory | 904 | 0 | 0 | 0 | **0** |
| reify | 692 | 0 | 1 | 0 | 1 (id 5050) |
| autopilot-video | 6 | 0 | 0 | 0 | **0** |
| know-live | 5 | 0 | 0 | 0 | **0** |
| pump-web-ui | 4 | 0 | 0 | 0 | **0** |
| solar-challenge | 7 | 0 | 0 | 0 | **0** |
| solar-challenge-platform | 3 | 0 | 0 | 0 | **0** |
| **total** | **1621** | **0** | **1** | **0** | **1** |

**Six of seven roots are at zero pending. 435 of 436 writes landed.** The one
remainder is reify 5050, blocked by the malformed `milestone` above and
escalated rather than forced.

## 4. Residual carriers by status

Unchanged before and after, by design: `done`/`cancelled`/`deferred` are
skipped deliberately. PRD decision 1 keeps `modules` on terminal tasks as the
only in-record trace of their original scope, and `update_task` will not write
them anyway. The migration now COUNTS them so the claim is checkable.

| project | done | cancelled | deferred |
|---|---|---|---|
| dark-factory | 1457 | 78 | 5 |
| reify | 2297 | 245 | 11 |
| autopilot-video | 416 | 26 | — |
| know-live | 39 | — | — |
| pump-web-ui | 2 | — | — |
| solar-challenge | 34 | — | — |
| solar-challenge-platform | 110 | — | — |
| **total** | **4355** | **349** | **16** |

`merge-deferred` is NOT in the skip set and IS processed — the set is an exact
match, and `deferred` being a substring of `merge-deferred` makes a
`startswith`/substring refactor a live hazard. Pinned by
`test_merge_deferred_is_processed_and_never_skipped`.

Every remaining carrier outside these three statuses is accounted for: exactly
one, reify 5050.
