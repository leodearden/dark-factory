# Scheduler scout: priority, age, pins (facts for the (tier, age) PRD)

Read on main at `ec46ac6670` (2026-11-13). `git status --short` is clean for every source path cited
(scheduler.py, config.py, pending_age.py, eligibility_clock.py, overrides.py, pin_reservation.py,
lock_table.py, verify_admission_gate.py, verify.py, merge_lane/, fused-memory/src, shared/src).
Task-store facts are from a read-only open of `.taskmaster/tasks/tasks.db` at about 15:50Z, 2026-11-13.
Branch facts are from `task/9908` at `e18d22c403`.

Two clocks are easy to confuse, so this report names them. **pending_since** is the task-row stamp
fused-memory writes when a task lands in `pending`. **eligible_since** is the starvation watchdog's anchor,
stamped the first tick a task is a dispatch candidate. A third, **legacy age anchor**, is a newer-task
count used only by the fallback formula.

---

## A. How priority is computed today

A1. **Tier vocabulary and bases.** `orchestrator/src/orchestrator/config.py::PRIORITY_TIERS` =
`('critical','high','medium','low','polish')`. `::PRIORITY_RANK` maps tier to index, so critical=0 and
polish=4, and lower is stronger. `::DEFAULT_TIER` = `'medium'`. `::TIER_WIDTH` = 1000. `::TIER_BASE` =
critical 16000, high 8000, medium 4000, low 2000, polish 1000. The bases are not evenly spaced; the
rule is only that each gap is at least TIER_WIDTH. `::coerce_tier` maps None to medium silently, and an
unknown string to medium with a warning that fires once per value.

A2. **Effective tier** (`scheduler.py::Scheduler._compute_effective_priorities`, a staticmethod called from
`::_phase_compute_priorities`). effective(t) = the strongest rank among t's own `priority`, t's override
`boost_tier` (if any), and effective(d) for every dependent d whose status is not done or cancelled. In
other words a tier is inherited UPSTREAM, from a dependent to its prerequisites. It is a memoised DFS over
every task in the tick's active fetch; on a cycle the task keeps its own priority and a warning is logged.
The output is a tier STRING per task (`dict[str, str]`), not a number. Only `boost_tier` enters this walk:
`_phase_compute_priorities` builds `override_boosts` from `row.boost_tier` and from nothing else, so
`pinned` plays no part.

A3. **Active scoring formula, C2** (`scheduler.py::Scheduler._compute_score`), keyword-only inputs:
```
score = TIER_BASE[tier]
      + cpm_budget * log1p(min(D, cpm_d_cap)) / log1p(cpm_d_cap)       # D = undone transitive dependents
      + age_budget * a / (a + age_half_secs)                           # a = now - metadata.pending_since, seconds
      + (continuity_credit if has_work_product else 0)
```
The defaults (`config.py::CPM_BUDGET`, `::CPM_D_CAP`, `::AGE_BUDGET`, `::AGE_HALF_SECS`,
`::CONTINUITY_CREDIT`) are 300, 32, 500, 259200 s (72 h) and 120. The age term is a hyperbola that
approaches 500 and never reaches it: 250 at 72 h, 455 at 720 h. **Age today is bounded by its budget;
it is not uncapped.** D comes from `::_compute_transitive_counts`, a BFS over undone descendants.
has_work_product comes from `self.work_product` (a git for-each-ref probe refreshed on a TTL, gated on no
`zero_progress_requeue` event, and lapsing after `continuity_credit_max_age_secs`, default 7 days).

A4. **Fallback formula** (`scheduler.py::Scheduler._compute_score_legacy`):
`TIER_BASE[tier] + min(age_alpha*age + cpm_beta*log1p(D), TIER_WIDTH-1)`. Here age =
`::_compute_age` = `max_id - anchor`, a count of newer tasks, not seconds. `age_alpha`=10 and
`cpm_beta`=100 (`config.py::OrchestratorConfig`). The age part saturates at about 100 newer tasks.

A5. **Which formula runs** (`scheduler.py::Scheduler._score_candidates`). One formula is chosen for the
whole tick. C2 runs when the share of candidates with a usable pending_since
(`pending_age.py::read_pending_age`, status PRESENT) is at least
`ScoringConfig.new_scoring_min_anchor_coverage` (0.90, `::_note_anchor_coverage` via
`anchor_coverage_gate.py`). Otherwise the legacy formula runs. Live coverage among pending rows is
3167/3171 (measured), so C2 is armed.

A6. **The single-integer invariant** (`config.py::ScoringConfig._reject_budget_sum_crossing_tier`):
`cpm_budget + age_budget + continuity_credit <= TIER_WIDTH - 1`, enforced by a pydantic validator.
`::_reject_continuity_above_cpm_d4` adds `continuity_credit < cpm(D=4)`. The legacy formula keeps the
same bound with `min(..., TIER_WIDTH-1)`. That is the whole reason a single float is strict across tiers.

A7. **Sort and pick** (`scheduler.py::Scheduler._score_candidates`, `::_phase_select_scored`). The sort
key is `(-score, *id_tie_break(task_id))`: older numeric id first, non-numeric ids last. The scored loop
tries `try_acquire` in that order (with an EASY-backfill retry through parks). The first success dispatches
and ends the tick. Only `scored[0]` (top_id) accrues skips and fairness parks
(`::_bump_skip_and_maybe_park`).

A8. **Phase order** (`scheduler.py::Scheduler._TICK_PHASE_ORDER`): ... `override_snapshot_gc`,
`compute_priorities`, `reserve_now`, `override_diff`, `build_candidates`, ... `starvation`, ..., `psi_gate`,
`select_pins`, `select_scored`. Pins are tried before scoring every tick.

A9. **Per-tier slot caps do not exist.** `acquire_next`'s docstring says "per-tier slot caps reserve
headroom", and `event_store.py::EventType.scheduler_tier_cap_idle` exists. No code emits that event and
nothing implements a cap (grep). `Scheduler._dispatched_priority` is written at both dispatch sites and
popped at release, but no code reads it.

## B. The age anchors: lifecycle, writers, readers

B1. **pending_since writer.** `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::stamp_pending_since`
is the one implementation. It is called from three backend write paths (`add_task`, `set_task_status`,
`set_status_and_stamp_audit`) and the task interceptor's batch commit (`task_interceptor.py`, one clock for
a planning commit). Its rules:
- any transition to `pending` with the key absent stamps now;
- `cancelled → pending` overwrites with now (PRD D3, the only intended reset);
- any other transition to `pending` with the key present leaves it unchanged;
- leaving `pending` never clears it.

It is stored in the task row's `metadata` JSON, so it survives an orchestrator restart. The v4→v5
migration `::_migrate_v4_to_v5` back-filled `pending_since = updated_at` plus `pending_since_backfilled`
for rows that were pending at the time.

B2. **pending_since is a first-PENDING stamp, not a first-ELIGIBLE stamp.** It is stamped on landing in
`pending` whether or not dependencies are satisfied, so a task filed behind unmet dependencies accrues C2
age during that wait. A task filed `deferred` (planning mode) is stamped at commit.

B3. **pending_since does NOT survive to done in practice (measured; mechanism found).**
- Measured: of 27 done tasks with id ≥ 9000, 26 have no pending_since, and 15 of 18 in-progress tasks
  have none.
- Mechanism: `sqlite_task_backend.py::SqliteTaskBackend.update_task` runs every caller payload through
  `::strip_machine_authored_metadata`, which removes `pending_since` and `pending_since_backfilled` in every
  mode. Under `metadata_mode='replace'` the row then takes the payload verbatim, so the stored anchor is
  deleted. The backend comment in `update_task` says so: "metadata_mode='replace' still drops the stored
  anchor, unchanged (design decision 8)".
- Orchestrator replace-mode writers that do a full read-modify-write:
  - `workflow.py::TaskWorkflow._clear_merge_phase_entered`, called at durable merge enqueue and at merge
    success whenever `merge_phase_liveness` is present;
  - `workflow.py::TaskWorkflow._clear_merge_retry_pending`;
  - `harness.py::Harness._clear_merge_retry_pending_for_restart`;
  - a further replace writer documented in `workflow.py` (docstring near the merge-retry stamp helpers).
- The sampled done rows carry neither `merge_phase_liveness` nor `pending_since`, which fits the first
  writer.
- Consequence: after such a write, a later `* → pending` (a requeue, a derailed train member flipped to
  pending, a blocked re-pend) finds the key absent and stamps NOW. That resets C2 age, against D3's
  intent.

The measured loss is fact. That the merge-phase clear is the dominant cause is inference; I did not trace
a single task's write journal.

B4. **pending_since readers.**
- `orchestrator/src/orchestrator/pending_age.py::read_pending_age`, returning an `AnchorReading(age_secs,
  status)`; absent or malformed reads as 0, and a future value reads as 0 but counts as covered.
  `::pending_age_secs` is its float wrapper.
- `scheduler.py::_pending_since`, which the legacy anchor uses to detect a reset.
- `anchor_coverage_gate.py`, the arming gate.
- `dashboard/src/dashboard/data/prd_view.py` and `task_snapshot.py`, display only.
- `fused-memory/.../middleware/followup_packing.py`, a rider-age check.
- `shared/src/shared/task_created_at.py::CreatedAtSource.PENDING_SINCE`, a proxy label for back-filled
  created_at.

B5. **Legacy age anchor** (`scheduler.py::Scheduler._update_age_anchors`, `::_compute_age`). It holds one
int per task in the active fetch (`_pending_anchor`, `_anchor_pending_since`), defaulting to the task's own
numeric id. It is persisted in a `guard_state.PersistentMap` at
`guard_path(project_root, 'scheduler_age_anchors.json')` with a 30-day TTL (`_RESURRECTION_GUARD_TTL`).
It resets to the current max_id when the task's pending_since changes (`_AgeAnchor.is_reset_by`), and
the reset emits `age_anchor_reset`. It is released in `::_phase_stale_sweep` when a task goes terminal or
leaves the fetch. It survives restart and re-pend, and it is NOT dropped at dispatch: in-progress tasks
are in the active fetch.

B6. **eligible_since** (`orchestrator/src/orchestrator/eligibility_clock.py::EligibilityClock`, store
`::EligibilityClockStore`). The SQLite file `scheduler_eligibility.db` (`config.eligibility_clock_db_path`;
live at `data/orchestrator/scheduler_eligibility.db`) has table `eligibility_anchors(project_root,
task_id, eligible_since)`.
- Writer: `scheduler.py::Scheduler._apply_starvation_watchdog` → `observe(candidates, now)`, an
  INSERT OR IGNORE for each dispatch candidate.
- Dropped: at dispatch (`::_settle_starvation_on_dispatch` → `forget([task_id])`, at BOTH dispatch sites),
  and whenever a task's status is not `pending` (`::_forget_starvation_inputs`, each tick).
- It survives a restart and a candidacy gap while the task stays pending.
- Reader: `CandidateFacts.eligible_age_secs` for `starvation_watchdog.py::classify`, plus delivery order
  (oldest first).
- It is NOT read by scoring. It is reset by dispatch and by every re-pend.

B7. **Answer to "is there any stamp today that records first eligibility and survives dispatch and
re-pend?": No.**
- eligible_since records first candidacy but dies at dispatch and at any non-pending status.
- pending_since records first pending, which is not first eligibility, and is lost at the first
  replace-mode metadata write (B3).
- The legacy anchor is a task-count, not wall time.

Task 8261's 2026-11-13 amendment states the same conclusion ("does not exist today").

B8. **PRD vs code drift.** PRD C4 specifies the watchdog clock as `now − max(pending_since,
deps_satisfied_at)`. The code instead uses the separate eligible_since store (B6).

## C. Consumers of the integer score and of TIER_BASE / TIER_WIDTH

C1. **The score float has exactly one consumer: the sort in `scheduler.py::Scheduler._score_candidates`.**
There is also a DEBUG log of the top 3 in `::_phase_select_scored`. No event carries a score (grep for
`'score'` and `dispatch_score` in orchestrator, dashboard and fused-memory finds none for dispatch), and
nothing persists it. **This consumer needs only an ordering.**

C2. **TIER_BASE and TIER_WIDTH in production:** only `config.py` (definitions and the `ScoringConfig`
validator) and `scheduler.py::_compute_score` / `::_compute_score_legacy`. Prose mentions appear in
`sqlite_task_backend.py::strip_machine_authored_metadata`'s docstring and in the `AGE_BUDGET`/`TIER_WIDTH`
comments of `fused-memory/tests/test_pending_since_anchor.py`.

C3. **Tests that pin score values or the band invariant:**
- `orchestrator/tests/test_scheduler.py`: `score == float(TIER_BASE['medium'])`, `bonus <= TIER_WIDTH-1`,
  `unlock_many < TIER_BASE['high']`;
- `orchestrator/tests/test_scheduler_scoring.py`: 10 references, including
  `_compute_score_legacy('medium', 1_000_000, 0) - TIER_BASE['medium'] == 999.0` and saturation checks;
- `orchestrator/tests/test_config_scoring_reload.py`: the budget-sum validator against `TIER_WIDTH - 1`.

These reach private methods (`_compute_score*`), which is the interface smell S17 is retiring (C/H5).

C4. **Consumers of the effective tier STRING, which needs a rank or an ordering, not a score:**
- the fairness skip threshold per tier: `config.py::FairnessConfig.skip_threshold_for`, with defaults
  `_DEFAULT_SKIP_THRESHOLD` critical 0, high 1, medium 2, low 4, polish 9999. `dark-factory-orchestrator.yaml`
  has no top-level `fairness:` block, so the defaults apply; its `skip_threshold: 150` belongs to
  `starvation_watchdog`, the watchdog's own gate;
- the park rank via `pin_reservation.py::park_rank` → `PRIORITY_RANK`, ordered by `lock_table.py::_ParkKey`
  `(rank, id tie-break)`;
- `reserve_now` (`scheduler.py::_phase_reserve_now` installs parks at the effective tier);
- `::_complete_earned_fairness_parks`;
- events `lock_acquired` (priority=tier), `task_skipped` (priority, threshold) and `reserve_now_consumed`
  (priority);
- `SchedulerStateSnapshot.effective_priorities`, which goes to `scheduler_state.json`, then to fused-memory
  `get_scheduler_state` and the dashboard (`dashboard/src/dashboard/data/scheduler.py`).

C5. **`_park_hold_refusal` compares no score and no tier** (`scheduler.py::Scheduler._park_hold_refusal`).
It is an owner-state predicate: terminal, missing, not pending, or a dependency known to be unsatisfied.

C6. **Pin rank band depends on `PRIORITY_RANK[PRIORITY_TIERS[0]] == 0`.**
`pin_reservation.py::_BEST_TIER_RANK`, `::is_pin_rank` (`rank < 0`), `::PinOrder.rank` (band
`[-1_000_000, -1]`) and `::rank_payload` (`rank >= len(PRIORITY_TIERS)` raises) all encode "tiers are
0..4, pins are negative". Park stacks are in memory only (no persistence in `lock_table.py`), so
renumbering ranks needs no migration. However, the `reservation_*` event payloads and
`SchedulerStateSnapshot.park_stacks` entries name either `tier` or `pin_order` by source.

C7. **The starvation watchdog** uses eligible_age_secs, skip_count and lock refusals
(`starvation_watchdog.py::classify`). It reads neither score nor tier.

## D. Pins

D1. **Store** (`orchestrator/src/orchestrator/overrides.py::_SCHEMA`), file
`<project_root>/data/orchestrator/scheduler_overrides.db`, table `overrides(project_root, task_id,
boost_tier TEXT, pinned INTEGER, pin_order INTEGER, reserve_now INTEGER, ttl_until TEXT, created_at,
updated_at, source TEXT DEFAULT 'operator')`, PK `(project_root, task_id)`, index
`(project_root, pinned, pin_order)`.
- `OverrideSource` has two values: `operator` and `starvation_watchdog`.
- `OverrideRow` is the frozen read projection.
- fused-memory hand-mirrors the schema in `fused-memory/src/fused_memory/server/tools.py::_OVERRIDE_SCHEMA`
  and migrates the same file.

D2. **Store API** (`overrides.py::OverrideStore`):
- `set_override`: always stamps source=operator; auto-assigns pin_order `MAX+1` under BEGIN IMMEDIATE when
  pinning without an order; preserves an existing order; raises `PinOrderCollision`; rejects pin_order
  unless pinned=True;
- `clear_override(field)`;
- `reorder_pin_queue`;
- `create_watchdog_pin`: inserts only when no row exists, at tail `MAX+1`, with a ttl;
- `release_watchdog_pin`: deletes only when source=watchdog;
- `count_pins`, `clear_terminal`, `clear_expired`, `get_overrides`, `get_pin_queue`.

D3. **How a pin is selected** (`scheduler.py::Scheduler._phase_select_pins`, `::_pin_queue`).
- Every tick, before scoring, each pinned row whose task is a dispatch candidate (`::_dispatch_candidacy`:
  `_eligible_for_dispatch` plus not in `gated_ids`) is queued.
- The queue is sorted by `pin_order` ascending, with a missing order read as 0.
- Pins are tried in order under the PSI hold for heavy tasks. The first whose `try_acquire` succeeds is
  dispatched (`::_dispatch_pin`) and the tick ends.
- Pins bypass score entirely. A pin that fails falls through and still gets its normal turn in the scored
  loop.

D4. **What else a pin does.**
- (a) **Pin reservation:** the first `pin_reservation_max_active` (default 1) lock-blocked, non-deterministic
  pins (the "heads") install parks at `PinOrder(pin_order)` rank via `::_complete_parks`. That rank shadows
  every tier park, critical included, and never preempts a held lock (`pin_reservation.py` docstring,
  `::_bound_pin_reservations`). Kill switch: `pin_reservations_enabled`.
- (b) **`pin_blocked` events:** `{task_id, pin_order, head, blockers}`, on the transition and then every
  `pin_blocked_emit_interval_secs` (`pin_reservation.py::PinBlockedLimiter`, `::_note_pin_blocked`).
- (c) **Override diff events** in `::_emit_override_diff_events`: `task_pinned`, `task_unpinned`,
  `pin_queue_reordered`, `priority_override_set` and `priority_override_cleared` (for boost_tier) and
  `reserve_now_armed`. They are observation-derived; task 4709 would make them write-through.
- (d) **`reserve_now`** (`::_phase_reserve_now`) is a one-shot flag, consumed the same tick. It installs
  fairness parks at the owner's EFFECTIVE TIER (not a pin rank) if `::_park_hold_refusal` accepts the
  owner, else emits `reserve_now_refused`.
- (e) **Watchdog settle:** a pin dispatch releases the task's watchdog pin.

D5. **A pin changes neither tier nor score.** `_compute_effective_priorities` never sees `pinned`, and
`_PinCandidate.tier` is the ordinary effective tier. A pinned task dispatched via `_dispatch_pin` books
`_dispatched_priority` and `lock_acquired.priority` as its tier.

D6. **What a pin does that `boost_tier` does not:**
- (i) it gives an absolute position (`pin_order`) ahead of ALL scored candidates, every tier included;
- (ii) its lock-blocked head reserves at a rank above every tier;
- (iii) it emits `pin_blocked`;
- (iv) it can be auto-created by the watchdog and carries a source and owner.

**What `boost_tier` does that a pin does not:** it PROPAGATES UPSTREAM. The boost enters the
effective-tier walk, so every unfinished prerequisite of a boosted task inherits the boosted tier (module
memory: a critical boost made a prerequisite a month-long critical park holder).

D7. **A pin does NOT propagate to the pinned task's unfinished prerequisites today.** `_pin_queue` admits a
pinned task only when it is itself a dispatch candidate, so a pinned task with unmet dependencies never
enters the queue, and its prerequisites get nothing from the pin. **A tier value WOULD propagate**,
because `_compute_effective_priorities` takes the strongest of own, boost and dependents' effective rank.
A "pin tier" placed into that walk would therefore lift every unfinished prerequisite to the pin tier.
That is a behaviour change, and it competes with the bounded-reservation exception (PRD §7 annotation
6040; at most `pin_reservation_max_active` heads).

D8. **Watchdog pins** (`scheduler.py::Scheduler._apply_starvation_watchdog`, `::_deliver_starvation_findings`,
`::_starvation_pin`):
- A pin-gate finding (gate `dual` or `lock_refused`; `park_blocked` never pins) on a task with no override
  row gets `create_watchdog_pin` at the tail with ttl `auto_pin_ttl_secs` (default 86400).
- At most `auto_pin_max_concurrent` (default 3) watchdog-owned pins can exist. Beyond the cap the task
  waits in the in-memory `_starvation_capped` set, and one sentinel L1 names the cohort.
- An existing operator row is never written.
- Release: at dispatch (`::_settle_starvation_on_dispatch` → `::_release_watchdog_pin`), when the task is
  no longer pending (`::_release_unpending_watchdog_pins`), by TTL, or when the task is terminal.

D9. **GC** (`scheduler.py::Scheduler._phase_override_snapshot_gc`, each tick):
`clear_terminal(terminal ∪ missing-from-fetch)`, then `_release_unpending_watchdog_pins`, then
`clear_expired(now)`. A pin's park reservation is expired by park GC through `_park_hold_refusal`, or
released by `_bound_pin_reservations` with a `PinReleaseReason`.

D10. **MCP tools** (fused-memory `server/tools.py`):
- `set_task_priority_override(project_root, task_id, boost_tier, pinned, pin_order, reserve_now,
  ttl_secs)`: validates boost_tier against `_PRIORITY_TIERS`;
- `clear_task_priority_override`: fields `_VALID_CLEAR_FIELDS = {boost_tier, pinned, reserve_now, ttl}`;
- `reorder_pin_queue(project_root, ordered_task_ids)`: the set must equal the current pinned set;
- `get_pin_queue(project_root)`: read-only;
- `get_scheduler_state`;
- `request_park_eviction`.

These write SQL directly on the same file through aiosqlite. Task 6718 (S14 θ) moves the store to
`shared/scheduler_overrides.py`.

D11. **Not the same "pin":** escalation-record pins (`escalation/src/escalation/pins.py`, `declared_pins.py`,
the MCP `declare_pin`, dashboard `pins_recovery.js`) protect escalation records and recovery. They are
unrelated to scheduler pins.

## E. Tier vocabulary enumerations

E1. **Enumerated separately in production code, not one table:**
1. `orchestrator/src/orchestrator/config.py::PRIORITY_TIERS`, `::PRIORITY_RANK`, `::DEFAULT_TIER`, plus
   `::TIER_BASE` and `::_DEFAULT_SKIP_THRESHOLD` keyed by tier;
2. `fused-memory/src/fused_memory/server/tools.py::_PRIORITY_TIERS` (boost_tier validation; "intentional
   duplicate");
3. `fused-memory/src/fused_memory/middleware/task_curator.py::_PRIORITY_RANK` (plus
   `curator_types.py::DEFAULT_PRIORITY`);
4. `dashboard/src/dashboard/app.py::_VALID_BOOST_TIERS`;
5. `client-feed/src/client_feed/policy.py` (`TaskPriority` enum, re-declared);
6. `scripts/check_run_completion.py` (completion weights);
7. `fused-memory/scripts/audit_duplicate_tasks.py::_PRIORITY_RANK`.

The submit_task docstring (`tools.py`) also lists them.

E2. **Validators keyed on the vocabulary:**
- `config.py::FairnessConfig._validate_skip_threshold` (rejects unknown tier keys);
- `overrides.py::OverrideStore.set_override` (boost_tier in `PRIORITY_RANK`);
- fused-memory `set_task_priority_override` (boost_tier in `_PRIORITY_TIERS`);
- `config.py::coerce_tier` (unknown values coerce to medium).

E3. **The task store does not validate a task's own `priority`.** `tasks.priority` is free TEXT
(`sqlite_task_backend.py`; `::_wire_priority` returns `raw or 'medium'`), and I found no enumerated check
on submit_task or update_task (grep). Live values: critical 82, high 1876, medium 3109, low 4791,
polish 18.

E4. **Adding a tier above critical touches every list in E1 that must accept it**, plus TIER_BASE,
`_DEFAULT_SKIP_THRESHOLD`, the skip_threshold validator, and the pin-band arithmetic in
`pin_reservation.py`, which assumes the best tier rank is 0 (C6).
- Whether the new tier is meant to be a stored task priority, or only an override-derived tier, decides
  whether E1 items 2 to 7 must change.
- Tasks 6642 and 6718 own folding the vocabulary into `shared/task_priority.py` (H).

## F. How code outside the scheduler reads a task's live tier today

F1. **`scheduler.py::SchedulerFacade`** is the task-store back-channel `TaskWorkflow` uses (`set_task_status`,
`set_task_claimant`, `mark_done`, `handle_blast_radius_expansion`, `acquire_uncontested`, `get_status`,
`get_task`, `update_task`, `dispatch_tool`, `submit_task`, `get_tasks`, `get_statuses`, `tasks_by_train`,
`clear_requeue_count`, `note_merge_phase_entered`, `clear_merge_phase`, `blocked_writes_suppressed`). **It
has no priority, tier, pin or age method.**

F2. **`scheduler.py::Scheduler.get_state_snapshot`** returns `shared/src/shared/scheduler_state.py::SchedulerStateSnapshot`
with these fields:
- `effective_priorities` (dict task_id → tier string, from `_last_effective_priorities`, the last tick's
  walk over the whole active fetch, so in-progress, blocked and deferred tasks are included);
- `pin_queue` (task_id, order);
- `overrides` (boost_tier, pinned, reserve_now, ttl_until);
- skip_counts, parks, park_stacks, pin_reservations, current_holders and others.

There is no age and no score. It is written to `data/orchestrator/scheduler_state.json` (currently 97.5
KB) and read back by fused-memory `get_scheduler_state` and the dashboard.

F3. **On main nothing hands the merge lane or verify a priority lookup.** `TaskAssignment.task` gives the
workflow the task dict as fetched at dispatch: its stored `priority` and any `metadata.pending_since`
present then.

F4. **Task 9908 has NOT landed.** `git merge-base --is-ancestor task/9908 main` is false; the branch has 5
commits, its worktree is `.worktrees/carry-9908`, and the task is `deferred` (held by a hand-carry
session). Its last intake.py commit on main is `cae18e8ccf` (task 8523). What the branch adds:
- `orchestrator/src/orchestrator/task_priority.py` with frozen `TaskPriority(tier: str|None,
  pinned: bool|None)`, `UNKNOWN_PRIORITY`, the Protocol `TaskPriorityLookup.__call__(task_id) ->
  TaskPriority`, and `read_task_priority(lookup, task_id)`, which never raises;
- `Scheduler.task_priority(task_id)`: tier from `_last_effective_priorities`, pinned from
  `_prev_overrides_snapshot`, None before the first tick;
- `Harness._task_priority`, injected into `TaskWorkflow` (via `build_workflow(task_priority=...)`) and the
  merge worker;
- record-only use on `merge_queued`/`merge_dequeued`/`merge_verify` events (`merge_pool_facts.py`).

**It carries no age field and changes no ordering.** Its docstring names 8261 and "the merge-pool pick" as
the intended reusers.

F5. **Merge-lane order today** (`merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`):
- lanes are ordered high before normal (lane comes from the request, i.e. `metadata.merge_lane` or a
  relane, not the scheduler tier);
- within a lane, "clique-minimal" FIFO on `_aging_key = (merge_first_enqueued_at or enqueued_at,
  request_id)` over `suffix_conflict_graph` neighbours.

No tier, pin or eligibility age enters the merge pick.

## G. Verify admission

G1. **`orchestrator/src/orchestrator/verify_admission_gate.py`:** `AdmissionOrder =
Literal['unordered','arrival']`; `AdmissionPolicy(slots, order, promote_after_secs)` validated in
`__post_init__`.
- Under 'arrival', `AdmissionGate._pick` takes `min` over waiters of
  `(0 if promoted else role_class(role), request.plan_arrival, seq)`.
- `ROLE_CLASSES` = {task: 0, background: 1}; promotion happens after
  `verify_admission_background_promote_secs` (default 7200).
- `AdmissionRequest` carries role, plan_id, plan_arrival (monotonic), task_id, module, scope_kind. **It
  has no tier and no age beyond plan arrival.**

G2. **The policy constructor is `verify.py::admission_policy(config)`, public.** There is no
`_admission_policy`; the brief's name does not exist. It is read at every grant. Config:
`verify_admission_order` default 'unordered' (`config.py`); dark_factory sets `arrival` and
`verify_admission_task_slots: 2`. These keys are in `RELOADABLE_FIELDS`.

G3. **Task 8261** (pending, high; no dependencies; dependent: 9705, S16 ζ4a). It adds
`AdmissionOrder 'priority_aged'` with key `(class, arrival + tier_rank × K, seq)`, where K =
`verify_admission_priority_age_secs` (hot-reloadable). The tier is read through a lookup at each pick,
not stamped at arrival, and the tier is recorded on `verify_admission_plan`. Its **2026-11-13 amendment**:
- priority is to be STRICT across tiers at verify and merge, ordered (tier, age), with a long-wait
  promotion as the only backstop;
- K is the backstop (suggested 12 h, not 2 h);
- the age term is to become the scheduler's first-eligible-to-done age, which "does not exist today and is
  NOT in this task's scope", so 8261 keeps `arrival` but takes age through the same injected lookup;
- reuse 9908's lookup if it has landed, otherwise build it in a shape 9908 can adopt;
- pins as a tier above critical are scheduler work and out of scope for 8261.

Note: 8261's key `arrival + tier_rank×K` is a blend with K as the exchange rate. It is not a
lexicographic (tier, age) tuple, and it is strict only inside a window of K × tier gap.

## H. Rulings, owners, collisions

H1. **`plans/scheduler-dispatch-scoring-and-lock-layer-prd.md`** (status active, 2026-08-06). Clauses a
(tier, age) tuple or a pin tier would change:
- **§4 D1** "Sub-budgets, not a clipped sum … the sum of budgets is validated below `TIER_WIDTH`."
- **§4 D2** wall-clock age from a durable `pending_since`, not id-distance.
- **§4 D3** "`pending_since` is reset ONLY on `cancelled → pending` … a task that has already waited must
  not lose its accrued wait because the machine dropped it." This is consistent with the new rule, and
  violated in practice by B3.
- **§4 D9** the watchdog consequence is an auto-pin, bounded and owned. Amended by 9636 for the cap L1.
- **§4 D12** arming by anchor coverage, with legacy fallback.
- **§5 C1** the pending_since write table and the reader contract.
- **§5 C2** the formula, with **I-1 no tier crossing** (`CPM_BUDGET + AGE_BUDGET + CONTINUITY_CREDIT ≤
  TIER_WIDTH − 1`), **I-2 non-saturating** ("`age` approaches its budget asymptotically and is never
  clipped"), **I-4 tie-break**, **I-5 locality** (score is a pure function of the task) and **I-6
  arming**. The worked arithmetic says "the age budget deliberately dominates in the long run" within a
  tier.
- **§5 C4** the watchdog clock (now drifted, B8), auto-pin at the queue tail, and the `park_blocked` gate.
- **§7 rejected:** "Below-rank-1 park INSTALLATION" (reserving for every starving task: −4 to −11%
  throughput, gridlock), with the 6040 annotation (a bounded pin-head reservation is an exception, not a
  reversal); "Preemption / force-release of a HELD lock"; "Lexicographic CPM buckets (S3)", which "lets …
  displace … on a bucket boundary … aging is structurally defeated"; "Percentile / within-set
  normalization".
- **§7 Amendment (task 7799):** the park order is one total key `(rank, id tie-break)` and "Branch progress
  is deliberately not in the key".
- **§13 Q1:** one global `AGE_HALF_SECS`.

A pin tier inside the effective-tier walk would interact with the §7 rows, because tier-ranked fairness
parks would then form at the pin tier for every prerequisite that becomes top-skipped (C4, D7).

H2. **`plans/review-all-s17-scheduler-seams-and-tick-state-prd.md`** (authored 2026-11-11, rulings
2026-11-13). It does NOT redesign scoring or overrides, but it restructures the ground the change edits.
All 22 leaves are filed and pending (9664–9685). Those that collide:
- **η 9679** (medium): `dispatch_ledger.py::DispatchLedger` of frozen `TaskRecord`s replaces 14 task-keyed
  containers, including `_dispatched_priority` and `_skip_count`. It depends on 6821, which moves
  `_pending_anchor`/`_anchor_pending_since` into `age_anchors.py`.
- **ε 9678** (medium): `tick_phases.py` with `TickPhase(name, run, reads, writes)` and a required-field
  `TickContext`. `_TICK_PHASE_ORDER` is deleted. It edits `test_scheduler_scoring.py`.
- **β 9665** (high): the `SchedulerScenario` test builder seeding through `OverrideStore.set_override`,
  and `get_state_snapshot().tasks`. It depends on 6718.
- **ι1/ι2/ι3 9666–9668, ω 9683:** tests move off private names, so tests pinning `_compute_score*` (C3)
  are slated to move or be deleted.
- The `scheduler.py` chain is κ1 → θ → η → ε → λ → δ2 → δ1 → γ → ζ → α → β.

Any scoring, age or pin edit to `scheduler.py` serialises on that file with this chain.

H3. **`plans/starvation-cap-study-2026-11-12.md`**, operator rulings of 2026-11-13:
- the auto-pin cap stays at 3, with no cap-8 trial;
- interim `starvation_watchdog.skip_threshold` 50 → 150, to be reverted when 9636 lands;
- 9636 filed (a change-driven cap L1 and restart-safe watchdog inputs; amends D9);
- narrower file lists dropped;
- releasing locks during the merge phase ruled out.

The study measured that pins rarely act (2.5% of (task, dispatch) pairs; 9 of 18 watchdog pins expired
unstarted), that 41% of slot time is spent in the merge phase, and that starts are slot-bound.

H4. **Operator rulings in memory files:**
- `feedback_locks_stay_held_until_the_merge_lands.md`: never release locks before landing.
- `project_three_gates_scheduling_study_2026_11_12.md`: priority is read at the scheduler only; "No owner
  exists for: merge queue ordered by urgency … recording eligibility history"; 8261 is pinned at 26, merge
  lane high.
- `project_merge_queue_urgency_study_2026_11_12.md`: "Urgency definition recommended: task 8261's, shared
  by both gates"; an urgency pick replaces task 1891's same-files rule.
- `feedback_a_priority_boost_below_the_inherited_tier_is_a_noop.md`: boosts at or below the inherited
  tier do nothing.
- `project_high_pins_never_park_2026_09_28.md`: boosts propagate upstream; only strict-top earns parks;
  task 6040 created the pin reservation.
- `feedback_deterministic_vs_llm_single_partition.md`: one decision partition per area.

No memory file records the 11-13 (tier, age) or pin-tier ruling; only task 8261's amendment does.

H5. **Tasks that own or collide** (status as read, 2026-11-13):

| id | what | status | relation |
|---|---|---|---|
| 8261 | verify admission 'priority_aged'; 11-13 amendment names (tier, age) and a shared lookup | pending, high | consumer; must take age through the lookup this PRD defines; its blend key differs from a lexicographic tuple |
| 9908 | merge-lane recording plus `TaskPriority(tier, pinned)` lookup | deferred (hand-carry; branch not landed) | the lookup to extend with age; must land first or be adopted |
| 6821 | extract the age-anchor bookkeeping into `age_anchors.py` | pending, low; dependent 9679 | owns the legacy anchor code this PRD may replace. Its description is stale: it names `_age_reset_owed`/`_age_reset_anchors`, which no longer exist on main |
| 9679 (S17 η) | dispatch ledger, one record per task | pending, medium; deps 9678, 6821 | collides on `scheduler.py` task state; an eligibility-age field would sit on its `TaskRecord` or beside it |
| 9678 (S17 ε) | tick phase table | pending, medium | collides on phase wiring (`compute_priorities`, `select_pins`, `select_scored`) and `test_scheduler_scoring.py` |
| 9665–9668, 9683 (S17 β, ι, ω) | test builder and private-reach migration | pending, high | the score-pinning tests (C3) will move |
| 6718 (S14 θ) | moves `OverrideStore` to `shared/scheduler_overrides.py` and creates `shared/task_priority.py` (PRIORITY_TIERS/RANK) | pending, medium; deps done; dependents 6719, 6722, 6724, 4709, 6726, 9665, 9739 | owns the pin store's and the tier vocabulary's new homes; a pin tier changes both |
| 6642 | one tier vocabulary in shared (fk-f56fced3bfea) | pending, low | adding a tier is exactly its four-file lockstep; should precede or be subsumed |
| 6722 (S14 κ) | dashboard adopts `shared.task_priority` for boost validation | pending, medium | touches `_VALID_BOOST_TIERS` |
| 7009 | type `park_stacks` entries by source once the vocabulary is shared | pending, low | the pin and tier park payload shape |
| 9636 | watchdog cap L1 change-driven and inputs restart-safe; reverts skip_threshold 150 | pending, medium; dependent 9727 | watchdog pins coexist with a pin tier; touches `_starvation_capped`/LockRefusals persistence beside eligible_since |
| 6328 | a critical candidate behind a blocked critical never earns a reservation | pending, medium | design option (a) is a "bounded automatic pin"; overlaps pin-as-tier |
| 4709 | override events written through from the MCP tools | pending, low; depends on 6718 | event vocabulary for pin and boost changes |
| 3824 / 3825 / 3826 | scoring PRD θ (durable fairness), ι (passed-over counting), κ (integration gate) | pending | 3826 drives pending_since into the score end to end, so its test shape changes with a tuple |
| 6658 | derailed train members lose their place on flip-to-pending | pending, high | a direct victim of B3's re-stamp; a first-eligible stamp would change its premise |
| 8480 | main-health heals cannot dispatch when slots are full of merge-waiters | pending, high | named by the starvation study as the structural lever; no direct collision |
| 5139 | merge and mainprobe admission role gate | pending, high | 8261's named neighbour; not a collision |
| 8526 / 8527 / 8528 | merge-lane policy δ/ε/ζ | pending, high | the merge pick change sequences after them (memory: 139 worker.py tasks parked behind 8528) |

---

## What does not exist (needed by the three changes)

1. A durable first-eligible stamp that survives dispatch, re-pend and restart. eligible_since dies at
   dispatch and re-pend, pending_since is first-pending and is wiped by replace-mode writes, and the
   legacy anchor is a task count.
2. Any age that keeps accruing after dispatch through verify and merge on a stamp that is kept. C2 age is
   computed only for dispatch candidates, and is bounded at `age_budget` (asymptotic 500), not uncapped.
3. A protection for pending_since against `metadata_mode='replace'` writes. The backend deletes it by
   design ("design decision 8").
4. A tuple or lexicographic priority type anywhere. The score is a float built on the TIER_WIDTH band
   invariant, and the effective tier is a string.
5. One lookup returning (tier, age), or (tier, pinned, age). 9908's `TaskPriority` has tier and pinned
   only, and is unlanded.
6. A pin represented as a tier. Pins are a separate queue (`_phase_select_pins`) plus a negative park-rank
   band.
7. Pin propagation to prerequisites. Only `boost_tier` propagates.
8. A tier above critical in any vocabulary or validator (E1, E2), or rank arithmetic that allows a tier
   rank below 0 (C6).
9. A shared tier vocabulary module. `shared/task_priority.py` is planned in 6718 and does not exist.
10. Per-tier slot caps (A9). Only a stale docstring and an unused event enum remain.
11. Any tier, pin or age input to the merge pick (F5) or to verify admission (G1) on main.
12. A scheduler method other code can call for live priority. `SchedulerFacade` has none; only the
    snapshot JSON and the unlanded `Scheduler.task_priority` exist.

## Unknowns

- U1. Whether `_clear_merge_phase_entered` is the dominant pending_since eraser, or whether other replace
  writers matter more (B3). I did not trace per-task write history, so the loss is measured and the
  attribution is inferred.
- U2. Whether the operator's "first became eligible" means dependencies satisfied (eligible_since
  semantics) or first pending (pending_since semantics). The code supports either; they differ by the
  dependency wait.
- U3. Whether a "pin tier" is meant to propagate to prerequisites through `_compute_effective_priorities`.
  That would be new behaviour; today a pin does not propagate (D7).
- U4. Whether pin_order survives as the intra-pin-tier order, or is replaced by age within the pin tier.
- U5. How CPM and continuity, which today order within a tier, fit a (tier, age) tuple: dropped, or a
  third key. The ruling as relayed names only (tier, age).
- U6. How the legacy-formula fallback and the anchor-coverage arming gate (D12, I-6) are meant to behave
  under a tuple.
- U7. (Resolved: dark_factory has no `fairness:` block in its yaml, so the per-tier defaults apply.)
- U8. Whether 9908 will land before this PRD's decompose. It is held by a live hand-carry session.
- U9. The task store has no priority validation that I could find. A stored priority above critical
  (if the pin tier became a task priority) would coerce to medium in the scheduler unless config.py
  changes.
