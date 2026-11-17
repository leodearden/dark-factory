# Instruments honesty: the factory's stores, gauges and reports say what they measure

Status: active, authored 2026-11-05 from the portfolio-review follow-up
`plans/portfolio-review-2026-11-01/followups/instruments.md` (seats and critic under
`followups/instruments-seats/`; substrate verified by seats V1–V3 there). Leo's rulings of 2026-11-05 bound
every design decision below: (1) the return brief covers the active projects; (2) landing velocity is
successful merges per unit time, meaning items removed from the merge queue that advanced main, whether by
merge commit or fast-forward; (3) `created_at` is a new column on the tasks table; (4) every L2 close stamps
`resolution_class`, and `ruled_by_human` is set only when an actual human made a ruling on that specific
issue; (5) merge-halt events are wanted; (6) one PRD.

**Code anchors** are cited by symbol (`path/to/module.py::symbol`); main moves fast, re-locate at
implementation time.

## Goal

An operator, an agent or a later review reads a factory instrument and gets the number it names. Concretely,
when this PRD lands:

- `data/return-brief.md` §2–§5 cover every active project, and "Landed" counts landings on main, not
  orchestrator runs that finished `done`.
- `get_scheduler_state` (fused-memory MCP), the dashboard paused list and `scripts/clear_leaked_claimants.py`
  read a snapshot written at every pause, resume, restore and shutdown, and the reader says how old the
  snapshot is and whether it agrees with the authoritative pause row in runs.db.
- Every task row carries `created_at` with its provenance; `get_task`/`get_tasks` and the dashboard task view
  expose it; a 95-day-old task cannot read as one day old.
- Every resolved or dismissed L2 record carries a non-null `resolution_class`, and `ruled_by_human` is true
  exactly on the closes a human ruled; the dashboard resolver-tier chart and the shadow-ruling agreement report
  compare against human rulings, not against the seat that typed the close.
- `task_results.review_cycles` is the task-lifetime count of blocking review rounds; `runs.paused_for_cap`
  and `cap_pause_secs` are written from the gate's clock.
- runs.db carries `merge_halted` / `merge_unhalted` events, and the dashboard merge-halt badge shows halt hours
  over the last 7 days, or "no halt event recorded yet" when the producer has not written.
- Digest windows do not overlap after a restart, and the EWA breaker's numerator and denominator span the same
  window.
- `docs/data-map.md` tells the next review's seat brief where each counting rule lives, by pointer.

## Background

The 2026-11-01 portfolio review §3.6 listed fifteen instruments whose raw numbers mislead. The follow-up
measured each (figures and rules in `followups/instruments.md` §1–§2). Six feed a decision today: the return
brief's roots list (a systemd drop-in REPLACES `DASHBOARD_KNOWN_PROJECT_ROOTS`, so §2–§5 show two of ten
projects); `scheduler_state.json` never written on pause (`Scheduler.flush_state_snapshot` has no caller;
pump-web-ui read "running" for six days of cost-ceiling pause); no task creation time (age proxies disagree
88× on task 3263); four velocity counts, none a landing count (29% of merged tasks never get a `done` event;
the digest's $/done denominator misses 31% of landings); `review_cycles` meaning "replans in this dispatch"
(max 1, reset per dispatch; the reviewer false-negative corpus miner reads it as "review passed"); and a digest
window end held in memory (each restart's first digest re-covers 24 h; sums +32%; the EWA breaker divides
since-restart submissions by a 24 h done count).

Owners that already exist and are left in place: flake_debt eager close (3792, 5875); sampler
`verify_concurrency` (7858, amended here with the cause: `sampler/src/sampler/metrics.py::classify_process`
requires argv[0] to be `verify.sh`, which is `bash`); cgroup-blind `own_cpu_some10` (3394, then 5220);
verify-log retention (5672); quality-metrics second snapshot (6698, 8463/8464); dashboard one-pass predicate
unification (8391); landing datum (quality-dashboard α, 8457); executor and coordinator identities with
`resolution_class` required on executor closes (decision-stream α/γ, 8422/8424); preparer cap ledger and
brief scope over scanned queues (8253).

The review's R7 proposed one "ledger-hygiene PRD" with ten bullets; four were already owned or already true,
two were wrong as written (account_events cannot be de-duplicated at write because nine processes write nine
ledgers; `resolution_class` cannot be required before the callers stamp it). This PRD is what survives:
the two live mis-statements, the one structural gap, the halt events Leo asked for, and the counting-rule
documentation, with the owned items referenced rather than re-decomposed.

## Sketch of approach

Six small producers and one documentation home, landed as one batch so the integration gate can prove them
against the live stores together:

1. **Fleet activity and landings for the brief** (`scripts/sitting/fleet_state.py`,
   `scripts/sitting/return_brief.py`): `build` measures §2–§5 over
   `inventory.queue_project_roots(sources.queues_scanned)` (the roots the decision registry has seen, which
   is already how §1 scans), every root's `ProjectState` gains activity fields, and the existing `landed`
   measurement counts landings through the quality-dashboard PRD's
   `shared/src/shared/quality_series/landings.py::landings` (8457). 8253 Part C keeps the preparer trial's
   14-day and lifetime windows; its root-source half moves here (G4 table).
2. **Scheduler snapshot truth** (`orchestrator/src/orchestrator/harness.py`,
   `shared/src/shared/scheduler_state.py::SchedulerStateSnapshot`, fused-memory
   `mcp_tools/scheduler_state.py`): `flush_state_snapshot()` at the four state changes; the reader serves
   `snapshot_age_s` and cross-checks the runs.db `scheduler_state` row, which is the pause fact's home.
3. **`tasks.created_at`** (fused-memory `backends/sqlite_task_backend.py`, `task_projection.py`, a backfill
   script, dashboard task view): v5→v6 additive column plus `created_at_source`; stamped in `add_task`;
   backfilled by a script with the earliest usable of the runs.db event, a non-backfilled `pending_since`
   and `updated_at` (decision 3).
4. **Ledger counters** (orchestrator `harness.py`, `workflow.py`, `run_store.py`): lifetime `review_cycles`;
   cap-pause fields set before `finish_run`; a persisted digest window end.
5. **Merge-halt events** (`event_store.py::EventType`, `merge_lane/halt.py::HaltState`,
   `merge_lane/worker.py`, dashboard `data/merge_halt.py`): edge-detected `merge_halted` / `merge_unhalted`
   with lane, reason, owner and `rehydrated`; a 7-day halt-hours datum.
6. **Resolution provenance** (`escalation/src/escalation/{models,queue,server,classify,shadow_ruling}.py`,
   `skills/escalation-watcher/SKILL.md`, `scripts/sitting/payloads.py::resolve_issue_payload`,
   `scripts/sitting/prepare_sitting.py::_cmd_resolve_answers`, dashboard `escalation_analytics.py`):
   `resolution_class` required on level-2 closes with a typed refusal; `ruled_by_human: bool | None` on the
   record, refused from automated identities; resolver tiers split human-ruled from attended-seat.
7. **Counting rules beside the stores** (`docs/data-map.md`, DDL comments, OPERATIONS.md corrections, a
   `measured_snapshots` view in burndown.db, the review-all report header and Phase 0 refresh).

## Resolved design decisions

1. **Active project = derived, not listed; the roots come from the decision registry.** The brief's §2–§5
   measure every root in `inventory.queue_project_roots(sources.queues_scanned)`, the same source §1 already
   scans, so no environment list can omit a project again (the 10-25 drop-in REPLACED
   `DASHBOARD_KNOWN_PROJECT_ROOTS`; that variable stays only for a project the registry has never seen, as
   8253 already says). Each root's `ProjectState` gains `unit_active` (systemd `ActiveState`),
   `pause_reason` (the runs.db `scheduler_state` row, which operator halts also write through
   `harness.py::force_halt_scheduler`), `landings` and `new_escalations` for the window. A project is
   **active** and gets its §2–§5 rows when `unit_active` or it had any landing or new escalation in the
   window; a paused or halted active project keeps its rows with a `scheduler paused: <reason>` flag so a
   cost-ceiling pause is visible rather than hidden. A project with `unit_active=False` and no activity is
   **inactive** and appears once in a footer line naming it with the reason. Rationale: Leo's ruling 1 names
   "the active ones"; a derived predicate with a visible remainder cannot silently omit a project (INV-11),
   and a halt must never make a project disappear from the page that reports halts.
2. **Landing = the quality-dashboard PRD's datum.** `landings(runs_db, repo, since_cursor)` (8457) returns
   `merge_finalized state=done` ∪ `train_merged` members, ordered by the event timestamp, with a sha cursor.
   This PRD does not re-derive it: `fleet_state.landed(runs_db, window, *, now, repo) -> Measurement[int]`
   keeps its envelope, gains the `repo` keyword `landings()` needs for its git cross-check (`measure_projects`
   holds the root), and counts `landings(runs_db, repo, since_cursor=None)` rows whose `landed_at` falls in
   the window; 8457 is amended to promise `since_cursor=None` means the full history (G4 table). The digest's
   done-in-window calls the same function through `asyncio.to_thread`, because `landings()` runs a git
   first-parent cross-check and `_maybe_write_digest` is a coroutine on the loop thread (INV-8). The former counts
   (`task_completed` done events; `task_results` done rows) are kept where they serve a purpose and
   relabelled "orchestrator runs that finished done (excludes hand carries and merge-deferred landings)".
   The dashboard burndown "Completed / day" becomes "Closed / day (any path: gates, recon and bulk closes
   included)" and the forecast band is captioned as backlog drain, not landings. $/done in
   `digest.py::query_model_role_rollup` is relabelled "per done run"; per-landing cost is the quality
   dashboard's series, not duplicated here (INV-9).
3. **`created_at` is a nullable TEXT column with a sibling `created_at_source` column**, values
   `stamped | first_event | pending_since | updated_at`. Rationale for the sibling column over a metadata
   key: every age reader needs provenance in the same row; `pending_since_backfilled` as a metadata key made
   the 2026-11-02 follow-up parse JSON per row to find it; a closed enum column is countable in SQL and needs
   no vocabulary blessing. Nullable because `ALTER TABLE ADD COLUMN` cannot add NOT NULL without a default,
   and a reader on a read-only connection may meet an unmigrated store (V1): readers guard with
   `'created_at' in row_keys`, the pattern `candidate_key` already uses. Stamped in
   `SqliteTaskBackend.add_task` from the `now` it already reads for `pending_since` and `updated_at`; that is
   the only INSERT into `tasks` (six callers, all through it). Backfill is a separate idempotent script, not
   the open-time migration: the migration is lazy per store and must not open another store's runs.db.
   Backfill rule (amended by task 8558 review, 2026-11-12): the backfill stores the EARLIEST usable of the
   task's earliest runs.db event (`first_event`; 5,747 of 8,515 df tasks have one), its un-back-filled
   `metadata.pending_since` (`pending_since`), and its `updated_at` or the v4→v5 back-filled `pending_since`
   copy of it (`updated_at`); ties go in that order. Each proxy is an upper bound on the filing time, and a
   long-waiting task's first event is often the starvation watchdog's alarm, so the event cannot simply win.
   Rows with no usable proxy are skipped and counted. Timestamps are normalised to the task store's `Z`
   millisecond form.
4. **The watchdog's clock is not changed.** `eligible_since` measures time as a dispatch candidate, which is
   what a dispatch-fairness alarm should measure (task 3821's decision). Age since filing is a different
   question and `created_at` answers it on the dashboard and in `get_task`. Every age column a reader shows
   is labelled with its proxy.
5. **The pause fact's home is the runs.db `scheduler_state` row; the JSON snapshot is a cache.**
   `flush_state_snapshot()` runs after `Harness.pause_scheduler`, `resume_scheduler`,
   `_load_persisted_scheduler_pause` (the restore path calls `scheduler.pause` directly, so a flush inside
   `pause_scheduler` alone would miss it) and in `run()`'s shutdown `finally:`. One reader,
   `shared.scheduler_state.read_scheduler_state_checked`, homed beside the view type in
   `shared/src/shared/scheduler_state.py`, loads the snapshot, adds `snapshot_age_s`, reads the runs.db row
   read-only, and when the two disagree serves the row's verdict with `pause_disagreement=true`. The
   fused-memory MCP tool (`mcp_tools/scheduler_state.py`) and the dashboard paused list
   (`dashboard/src/dashboard/data/scheduler.py`) both call it (the dashboard imports only `shared`, never
   `fused_memory`, and must not gain that edge), and the dashboard serves the reader's `snapshot_at` and
   `pause_disagreement` instead of its own cache-fill time. INV-9: one home, one reader, the cache points at
   the row.
6. **Digest window end persists in its own single-row table**, `digest_window(window_end_iso)` in run_store,
   on the `main_sweep_cold_control` pattern; `scheduler_state` cannot hold it because that row is deleted on
   resume (V2). `harness.py::_init_digest_state` seeds `_last_digest_window_end_iso` from it (the test helper
   already delegates to that method); the first digest after a long outage covers "last end to now", which is
   correct, and the EWA step then sees matching spans.
7. **Halt events are edge-detected.** `HaltState` gains `on_transition(kind, lane, reason, owner_esc_id)`
   beside `on_resume`, fired only when a lane's halted state changes (`unhalt_all_lanes` fires `on_resume`
   unconditionally today, so emission cannot ride it). The worker wires it to `event_store.emit` as
   `merge_halted` / `merge_unhalted` with `data = {lane, reason, owner_escalation_id, rehydrated, run_id}`;
   `_rehydrate_merge_halt` emits with `rehydrated=true`; `release_lanes_for_shutdown` emits `merge_unhalted`
   with `reason='shutdown'` so every interval closes inside its `run_id`. A reader pairs within `run_id` and
   treats a `rehydrated` halt as a continuation. `EventType` keeps name == value
   (`test_event_store.py::test_all_event_types_valid`).
8. **`resolution_class` is required on level-2 closes; the refusal is typed.** `queue.py::EscalationQueue.resolve`
   refuses a level-2 resolve or dismiss whose stamped class is None with `resolution_class_required`;
   cascade members inherit the parent's class as today. Levels 0 and 1 keep their defaults (the reaper tier's
   `benign`; the L1 skill already stamps). Callers that must change in the same leaf:
   `skills/escalation-watcher/SKILL.md` (zero `resolution_class` mentions today),
   `scripts/sitting/payloads.py::resolve_issue_payload` and `scripts/sitting/prepare_sitting.py::_cmd_resolve_answers`
   (the sitting resolve path). The decision-stream PRD requires the class on executor closes (8422 α,
   8424 γ); this leaf depends on 8422 so the two server edits do not collide, and extends the requirement from
   one identity to the level. Tests assert the refusal contract (`error == 'resolution_class_required'`,
   record unchanged), never the class enum, which has grown before.
9. **`ruled_by_human` is asserted by the caller; the server enforces the negative.** New `Escalation` field
   `ruled_by_human: bool | None` (None = not asserted; legacy records read as None through `from_dict`'s
   filter, no migration). `resolve_issue(ruled_by_human=True)` is refused with `ruled_by_human_forbidden`
   when the server-attributed identity (the `X-Escalation-Identity` header) is the executor or the L1
   auto-watcher; header-less callers keep their self-declared `resolved_by` and are trusted as today, since
   identity is server-attributed only from a header. The skill text states the rule: true only when a human
   ruled on this record or its sitting docket item, and the resolution cites that ruling. Rationale: no
   machine-readable "a human ruled" fact exists at resolve time (V3); the `x_agreed` marker and
   DecisionRecords are indirect, so the caller asserts and the decision-stream audit (10/wk) checks. Cascade
   members inherit the flag with the class.
10. **Resolver tiers split human-ruled from attended-seat.** `classify.py::classify_resolver_tier(resolved_by)`
    keeps its string signature (three callers pass a bare string); a new `classify.py::resolver_tier(record)`
    returns `human-ruled` when `ruled_by_human` is true, `attended-seat` for a `_HUMAN_RESOLVERS` identity
    without it, and otherwise delegates to the string function. `escalation/src/escalation/shadow_ruling.py`
    compares agreement against `human-ruled` only; `escalation_analytics.py`'s lifespan and workflow blocks
    route through `resolver_tier` and relabel the chart, keeping the stamped/inferred split (legacy nulls stay
    inferred, and the inferred share now decays). 8422 owns the membership of `_HUMAN_RESOLVERS` (it adds the
    coordinator identity); this PRD adds the record-based tier on top and depends on it.
11. **`review_cycles` becomes the lifetime blocking-round count** from
    `TaskArtifacts.get_review_cycles_total()` at the `TaskReport` build site, which is reachable there and
    survives re-dispatch (V2). No backfill: `review_completed` events exist only from 2026-10-21. The column's
    DDL comment names the cutover date; the dashboard one-pass predicate is 8391's (it depends on this leaf);
    `mining.py::mine_fn_candidates`' docstring is corrected to "reviewer blocked at least once" and its
    predicate left for the eval owner (open question 1); the canary runbook notes the baseline shift.
12. **Cap-pause fields are read from the gate before `finish_run`**, and labelled in the DDL as "gate-level
    pause seconds, clean-exit runs only". The `workflow_verify` duration stays inside `data.duration_ms`:
    `event_store.py` already documents that verify-shaped rows carry durations in `data` because the
    recording double keeps only `(event_type, {task_id, data})`, and `merge_verify` follows that rule; making
    one of three verify-shaped types populate the column would break the uniformity (heuristic 10). ζ
    records "verify durations live in `data.duration_ms`; the column is NULL by convention" at that comment.
13. **Dead gauges are repaired, not removed.** The sampler classifier fix is 7858's (amended with the cause);
    cap-pause is decision 12. Nothing in this PRD deletes a column or a tile.
14. **Counting rules live beside the store they govern, and `docs/data-map.md` points at them** (INV-9, no
    restating): the `account_events` rule at `shared/src/shared/cost_store.py::_SCHEMA` (one row per observing
    gate; count physical caps as distinct `(account_name, details.resets_at)` or merged intervals); the
    `workflow_verify` field vintage at `event_store.py`'s existing comment; `BURNDOWN_SCHEMA` gains
    `CREATE VIEW IF NOT EXISTS measured_snapshots AS SELECT * FROM snapshots WHERE state IS NULL OR state = 'value'`,
    the predicate `dashboard/src/dashboard/data/burndown.py` applies when narrowing a row to `_Measurement`
    (legacy rows carry a NULL state); OPERATIONS.md
    §"Reading the verify artefact archive" corrected (task-role summaries are green and red, merge-role red
    only, the 500 MB cap binds at ~5 days); the review-all report header and JSON `method` gain "dispositions
    as of `<as_of_sha>`; current standing is the task store by `x_finding_key`", and `skills/review-all/SKILL.md`
    Phase 0 refreshes dispositions by key before classifying, as `skills/hotspot-survey/SKILL.md` does. The
    data map lists, per question (merge halts, main red, fix-forward tasks, scheduler pauses, landings, task
    age, cap events, verify duration), the store, the key's home symbol and the known gap, with the timestamp
    type of each store.
15. **Deploy is one gate, and the drop-in goes only after ε lands.** Without the drop-in,
    `scripts/_task_db_scan.py::discover_project_roots` falls back to the dark-factory root alone, so deleting
    it before ε would shrink the brief from two roots to one. η therefore depends on ε (and on everything
    else in the batch): it deletes the drop-in, removes `/home/leo/src/autotrade` from the two units' root
    lists, and runs the drained fleet sweep that brings the restart-tier producers (snapshot flush, digest
    table, halt events, review_cycles source) to production. Units are patched surgically, never re-rendered.

## Pre-conditions for activating

- **8457** (quality-dashboard α, pending): `shared/src/shared/quality_series/landings.py::landings` does not
  exist on main (V3). Leaf ε depends on it.
- **8422** (decision-stream α, in progress): executor and coordinator identities, `_HUMAN_RESOLVERS`
  membership and the executor-side `resolution_class` requirement in `server.py`. Leaf δ depends on it.
- No other novel substrate: the version ladder (`_migrate_v1_to_v2` template, `add_column_if_missing`),
  `flush_state_snapshot`, `RunStore.save/load/clear_scheduler_pause`, `add_column_if_missing_sync`,
  `HaltState.on_resume`, `emit(duration_ms=)`, `get_review_cycles_total`, `inventory.queue_project_roots`,
  the `Escalation` `to_dict`/`from_dict` filter all exist and are wired (V1–V3).

## Cross-PRD relationship (G4)

| other PRD / task | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/quality-dashboard-prd.md` (8457 α) | consumes | `quality_series/landings.py::landings`; 8457 amended to promise `since_cursor=None` ⇒ full history | that PRD | queued |
| `plans/decision-stream-prd.md` (8422 α, 8424 γ) | extends | `server.py::resolve_issue` class requirement (executor → level 2); `_HUMAN_RESOLVERS` membership; executor skill stamps the class | identities and executor requirement: that PRD; level-2 requirement and `ruled_by_human`: this PRD (δ) | 8422 in progress |
| task 8253 (preparer Part C) | splits | root source for `return_brief.build` (§2–§5 over `queue_project_roots`) and `ProjectState` activity fields move to ε; 8253 keeps Part B (cap ledger) and the trial's 14-day/lifetime windows, amended to depend on ε | this PRD (ε) for roots and activity; 8253 for the trial windows | pending |
| task 8391 (one-pass predicate) | consumed by | `task_results.review_cycles` lifetime semantics | source: this PRD (β2); predicate unification: 8391 (amended to depend on β2) | pending |
| task 5353 (task_results counters) | amends | `review_cycles` cause and fix move to β2; 5353 keeps `steward_invocations`, `block_reason`, `verify_attempts` | 5353 | pending |
| task 7858 (sampler verify_concurrency) | amends | cause recorded: `classify_process` argv[0] match | 7858 | pending |
| `plans/escalation-lifecycle-dashboard-prd.md` | extends | `resolution_class` null-allowed → required at level 2; tier vocabulary | this PRD (δ) | landed PRD |
| `plans/flake-ledger-prd.md` (3792 θ) | none | eager close of flake_debt | 3792 | pending, unchanged |
| `plans/startup-hardening-prd.md` (8267–8274) | none | no overlap on the snapshot or digest paths (V2) | — | — |

## Contract (B + H)

### Fleet activity and landings (`scripts/sitting/fleet_state.py`, `scripts/sitting/return_brief.py`)

```
ProjectState += {unit_active: bool, pause_reason: str|None, new_escalations: int,
                 active: bool, inactive_reason: 'unit inactive, no activity'|None}
measure_projects(roots=inventory.queue_project_roots(sources.queues_scanned), window) -> Mapping[str, ProjectState]
landed(runs_db, window, *, now, repo) -> Measurement[int]   # repo (the project root) supplied by measure_projects;
                                                            # value = count of landings(runs_db, repo, since_cursor=None) rows
                                                            # with landed_at in the window; gap when runs.db or landings() is unreadable
```
Invariants: every scanned root appears exactly once in `measure_projects`; `active` is true iff
`unit_active or landings + new_escalations > 0`; a row with `pause_reason` set is rendered with a `scheduler
paused: <reason>` flag, never dropped; every `active=False` row is named in one footer line; `landed` never
reads `task_completed`; the `Measurement` envelope distinguishes a zero from an unreadable store.

### Scheduler state reader (`shared/src/shared/scheduler_state.py`; callers `fused-memory/src/fused_memory/mcp_tools/scheduler_state.py`, `dashboard/src/dashboard/data/scheduler.py`)

```
SchedulerStateSnapshot view += {snapshot_age_s: float|None, pause_row: {reason, pause_at, pause_class}|None,
                                pause_disagreement: bool}
shared.scheduler_state.read_scheduler_state_checked(root, *, now) -> that view
    # snapshot from scheduler_state.json; pause_row from runs.db scheduler_state, mode=ro, pause_class as a raw string
```
Invariants: `is_paused` served = `pause_row is not None` when the row is readable, else the snapshot's value
with `snapshot_age_s` set; `pause_disagreement` is true iff both are readable and differ. The orchestrator
calls `flush_state_snapshot()` after every pause, resume, restore and before shutdown returns.

### Task creation time (`fused-memory/src/fused_memory/backends/sqlite_task_backend.py`)

```
tasks.created_at TEXT NULL            -- task-store 'Z' millisecond form, stamped in add_task
tasks.created_at_source TEXT NULL     -- 'stamped' | 'first_event' | 'pending_since' | 'updated_at'
_SCHEMA_VERSION = 6; _migrate_v5_to_v6 adds both columns (add_column_if_missing)
scripts/backfill_task_created_at.py --project-root R [--dry-run] -> counts per source, rows skipped (no usable proxy)
```
Invariants: every row inserted after the migration has `created_at` and `created_at_source='stamped'`;
backfill never overwrites a `stamped` row; `get_task`/`get_tasks` carry both keys (`task_projection.py::TASK_ROW_KEYS`)
and omit them on a connection whose v6 ALTER never landed rather than failing; the dashboard task view shows "Filed <date> (<source>)".

### Merge-halt events (`orchestrator/src/orchestrator/event_store.py`, `merge_lane/halt.py`)

```
EventType.MERGE_HALTED = 'merge_halted'; EventType.MERGE_UNHALTED = 'merge_unhalted'
HaltState.on_transition: Callable[[kind: 'halted'|'unhalted', lane: str, reason: str, owner_esc_id: str|None], None] | None
data = {lane, reason, owner_escalation_id, rehydrated: bool, run_id}
dashboard/src/dashboard/data/merge_halt.py::halt_hours(project, days=7) -> Datum[float]   # 'no halt event recorded yet' when none
```
Invariants: one `merge_halted` per lane transition into halted, one `merge_unhalted` per transition out, within
one `run_id`; a `rehydrated=true` halt is a continuation; `release_lanes_for_shutdown` closes every open
interval with `reason='shutdown'`; the datum distinguishes zero hours from no producer (INV-13).

### Resolution provenance (`escalation/src/escalation/{models,queue,server,classify}.py`)

```
Escalation.ruled_by_human: bool | None
resolve_issue(..., resolution_class: str|None, ruled_by_human: bool|None=None)
  level 2 and stamped class is None                                   -> {'error': 'resolution_class_required'}
  ruled_by_human=True and header identity is executor or auto-watcher -> {'error': 'ruled_by_human_forbidden'}
classify_resolver_tier(resolved_by: str|None) -> <existing tiers>     # unchanged
resolver_tier(record) -> 'human-ruled' | 'attended-seat' | classify_resolver_tier(record.resolved_by)
```
Invariants: cascade members inherit class and flag; level 0/1 defaults unchanged; `shadow_ruling` agreement
denominator = records with tier `human-ruled`; the analytics stamped/inferred split is unchanged in meaning.

### Digest window (`orchestrator/src/orchestrator/run_store.py`)

```
digest_window(window_end_iso TEXT)   -- single row; RunStore.save_digest_window_end / load_digest_window_end
```
Invariant: `_last_digest_window_end_iso` is seeded from the row at startup; a digest's window starts at the
previous end, never at a fixed 24 h lookback, except on a store with no row.

## Boundary-test sketch (B + H): the integration gate θ's signal

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | brief renders after deploy (live) | ten roots in the decision registry, reify scheduler halted, pump-web-ui as live | §2–§5 list every root that is `active`; reify appears with `scheduler paused: <reason>`; one footer line names each inactive root with its reason; no scanned root missing |
| 2 | Landed vs landings (live) | a week with a fast-forward landing and a train in df runs.db | brief "Landed" for df equals the count of `landings()` rows in the window; the digest done-in-window agrees; the old count is shown only under its new label |
| 3 | pause then read (fixture) | call `pause_scheduler` on a fixture orchestrator, read the JSON file and `get_scheduler_state` | file `is_paused=true` within one write; reader `snapshot_age_s` < 60, `pause_disagreement=false` |
| 4 | stale snapshot (fixture) | snapshot says running, runs.db row says paused | reader serves `is_paused=true`, `pause_disagreement=true`; dashboard paused list includes the project |
| 5 | restore path (fixture) | restart with a persisted pause row | snapshot rewritten at restore; shutdown rewrites once more |
| 6 | new task | `submit_task` on a migrated store | `get_task` carries `created_at` (= `pending_since`) and `created_at_source='stamped'` |
| 7 | backfill | a store with rows having an event, a non-backfilled `pending_since`, only `updated_at`, blank `updated_at`, a row whose first event is later than its `updated_at` | sources counted per rule; each row takes its earliest usable proxy; no `created_at` later than its row's `updated_at`; `stamped` rows untouched; second run changes nothing |
| 8 | unmigrated read | `get_tasks` on a connection whose v6 ALTER never landed (the store reports v6 without the two columns). A v5 store read through fused-memory is not this case: `_get_read_access` awaits `_get_write_access`, so that read migrates it (amended by task 8558 review, 2026-11-12) | rows served without the two keys, no error; a migrated row the backfill has not reached serves both keys as None |
| 9 | review rounds | a task blocked twice then passed | `task_results.review_cycles=2`; `review_completed` max cycle agrees |
| 10 | cap pause | a run whose gate recorded `total_pause_secs > 0` and finished cleanly | `runs.paused_for_cap=1`, `cap_pause_secs>0` on the row |
| 11 | verify duration rule | the `event_store.py` comment and `docs/data-map.md` | both say verify durations live in `data.duration_ms` and the column is NULL by convention; a reader coalescing to 0 is named as the error |
| 12 | halt and unhalt | halt a lane, unhalt it | exactly one `merge_halted` and one `merge_unhalted` with the same `run_id` and lane; dashboard datum equals the interval |
| 13 | rehydrated halt | restart with a preserved halt | one `merge_halted` with `rehydrated=true`; the interval continues; no duplicate on a second rehydrate |
| 14 | shutdown while halted | stop the unit | `merge_unhalted reason='shutdown'` closes the interval |
| 15 | no halt producer | a project store with no halt events | datum reads "no halt event recorded yet", not 0 h |
| 16 | L2 close without class | `resolve_issue` on a level-2 record with no class from the watcher identity | refused `resolution_class_required`; record unchanged |
| 17 | L2 close with class and ruling | same, with `resolution_class='actionable', ruled_by_human=True` | record carries both; tier `human-ruled`; cascade members inherit |
| 18 | executor asserts ruling | executor identity passes `ruled_by_human=True` | refused `ruled_by_human_forbidden` |
| 19 | legacy record | an archived record without the field | reads as `ruled_by_human=None`, tier `attended-seat` for the watcher identity |
| 20 | digest after restart | restart with a `digest_window` row | the next digest's window starts at the row's value; no 24 h lookback |
| 21 | data map | `docs/data-map.md` | every pointer resolves to an existing symbol or doc section (a test greps each cited symbol) |

## Decomposition plan

| label | title | modules | observable signal | prereqs |
|---|---|---|---|---|
| α | `tasks.created_at` + `created_at_source`: v5→v6 migration, stamp in `add_task`, projection keys, dashboard "Filed" column, backfill script | fused-memory backend, task_projection, dashboard tasks view + tests, `scripts/backfill_task_created_at.py` | rows 6, 7, 8: `get_task` shows `created_at` on a new task; the backfill prints per-source counts and is idempotent | — |
| β1 | Scheduler snapshot truth: `flush_state_snapshot()` at pause, resume, restore and shutdown; `shared.scheduler_state.read_scheduler_state_checked` (snapshot age + runs.db cross-check) called by the fused-memory tool and the dashboard paused list | orchestrator harness; shared/scheduler_state.py (the reader's home); fused-memory mcp_tools/scheduler_state.py; dashboard data/scheduler.py + tests | rows 3, 4, 5 | — |
| β2 | Run-store counters: lifetime `review_cycles` (+ `mining.py` docstring, canary runbook note); cap-pause before `finish_run`; `digest_window` table seeded at init | orchestrator harness, run_store, workflow, evals docstring/runbook + tests | rows 9, 10, 20 | — |
| γ | Merge-halt events and the 7-day halt-hours datum | event_store, merge_lane/halt.py, merge_lane/worker.py, harness rehydrate, dashboard merge_halt.py | rows 12–15 | — |
| δ | Resolution provenance: `ruled_by_human`, level-2 class requirement, `resolver_tier(record)`, skill and sitting-path stamping, analytics and shadow-ruling relabel | escalation models/queue/server/classify/shadow_ruling, skills/escalation-watcher, scripts/sitting payloads + prepare_sitting, dashboard escalation_analytics | rows 16–19; the skill contract test runs each resolve example against the fixture server (INV-10) | 8422 |
| ε | Brief over the registry's roots with activity fields; `landed` over `landings()`; digest done-in-window via `to_thread`; relabel the former counts in digest and dashboard burndown | scripts/sitting/fleet_state.py + return_brief.py, orchestrator digest.py, dashboard burndown labels + tests | rows 1, 2 | 8457 |
| ζ | Counting rules beside the stores and `docs/data-map.md`; burndown `measured_snapshots` view; OPERATIONS.md corrections; review-all header and Phase 0 refresh | docs/, shared/cost_store.py, event_store.py comment, dashboard burndown schema, OPERATIONS.md, skills/review-all | rows 11, 21; `select count(*) from measured_snapshots` on burndown.db returns the value-row count | — |
| η | Deploy gate (OPERATOR GATE, deterministic): delete the return-brief drop-in, drop the autotrade root from two units, drained fleet sweep so β/γ reach production; amend 8253, 8391, 5353, 7858, 8457 as the G4 table says | installed units, task store amendments | the brief's next render lists the active projects; `journalctl` shows the units started on the new code | α β1 β2 γ δ ε ζ |
| θ | Integration gate: run boundary rows 1–21 (rows 3–5 on a fixture orchestrator, the rest on the live df and reify stores) and record observed postconditions in this PRD | tests + PRD record | every live row's postcondition observed on live stores, not seeded rows (INV-13) | η |
| ι | Observation 7 days after θ (`complexity='simple'`, dated milestone): producers wrote in production (new tasks carry `created_at`; a real pause flushed the snapshot; halt events exist for a real halt; L2 closes since deploy all carry a class; the brief covered the active set each night); record appended to the PRD | PRD record | the appended record with counts and rules | θ (+7 days) |

G7 walk (advisory, author mode): INV-9 one-fact-one-home (decisions 5, 12, 14); INV-11 no-silent-fail-soft (decision 1 footer and paused flag; reader `pause_disagreement`; `Measurement` envelope on `landed`); INV-13 readers-prove-their-producer (halt datum, ι); INV-8 loop-thread-occupancy-bounded (decision 2 `to_thread`); INV-10 guards-exercise-behaviour (δ's skill contract test runs the examples); INV-1 contracts-machine-checked (typed refusals in δ; `test_all_event_types_valid`). No waiver needed.

## Out of scope

- Re-deriving the landing datum (quality-dashboard PRD), changing the watchdog's clock (3821), the executor
  and coordinator identities (decision-stream PRD), the flake_debt eager sweep (3792), the sampler classifier
  fix (7858), verify-log retention (5672), the one-pass predicate (8391).
- Backfilling `review_cycles` or `workflow_verify.duration_ms` for rows before their producers existed.
- Deleting `account_events` duplicates or de-duplicating at write (no shared writer exists).
- The scoping-intake README status line (a scoping_intake task, filed separately) and the ventures STATUS doc.
- A task-age column in the return brief.

## Open questions (tactical)

1. **`mining.py::mine_fn_candidates` predicate.** With lifetime `review_cycles`, `>= 1` means "blocked at
   least once, then passed". The false-negative corpus wants "passed, then a bug surfaced"; whether that is
   `== 0` or all done rows is the reviewer-trial owner's call. **Suggested resolution:** correct the docstring
   in β, leave the predicate, note it for the eval owner.
2. **Backfill timestamp for tasks whose first event predates runs.db history (2026-04-09).** **Suggested
   resolution:** `first_event` is an upper bound on the filing time, like the other two proxies. For a task
   filed before runs.db history began, the earliest of the three is the tightest bound available, and the
   value is recorded as such.
3. **Halt-hours window.** 7 days matches the brief; the quality dashboard may want landings-windowed halts
   later. Decide in γ.
