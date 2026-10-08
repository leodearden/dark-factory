# Deferral records: every `deferred` task says how it ends

**Status:** authored 2026-10-07 (`/team` + `/prd` author mode), not yet decomposed.
**Type:** new contract at the task-status write choke point, one new orchestrator sweep,
two readers, one migration. No new store and no schema migration: the record is task metadata.
**Approach:** B+H. G5 applies on several counts: five packages, the persistence choke point,
and three or more consumers.
**Code anchors** verified against main `5b645d822a` (2026-10-07). Cite-by-symbol; re-locate
at implementation time.
**Ruling (Leo, 2026-10-07):** "a transition to deferred must be supplied with a reason,
enforced deterministically (perhaps the MCP); optionally a deferral can be of limited
duration so that when it expires the task goes back to pending; the use case is a deferral
while an agent session works on a task, which is meant to be temporary but can end up
forever if the session dies." The brief's recommendations 1–4 were accepted ("LGTM").
**Amended 2026-10-08 (Leo, ruling on D-e):** "no notice so long as the session is live" —
sessions can legitimately sit for days while a complex question waits its turn. The 24 h
idle-holder notice is gone; a live holder is never notified, flipped or counted as needing
a human. Only holder death files a notice and, after the grace window, flips the task.
§3 records where this PRD departs from their wording and why.

## 1. Goal

Every task in `deferred` carries exactly one structured record, `metadata.deferral`. It
names what ends the hold, who ends it, and, optionally, when it lapses. The server refuses a
write into `deferred` that has no valid record. A hold whose owner can die (an interactive
session), or that was given a lapse time, returns to `pending` without anyone acting, and a
human is told when a holder dies — never while it lives. Observable when it lands:

- `set_task_status(id=…, status="deferred")` with no `deferral` returns
  `{'success': False, 'error': 'deferral_required', …}`, and `get_task` shows the row's status
  unchanged. A malformed record returns `error: 'deferral_invalid'`, naming `reason_code`,
  `field` and the offending `value`.
- When a `held_by_session` holder process exits, a human-queue INFO notice names the dead
  session and every task it held. Each task returns to `pending` once the sweep has seen the
  holder dead for `grace_secs` (default 7,200 s; at most one 60 s pass later).
- An `until_condition` deferral with `expires_at` returns to `pending` within one sweep pass
  after `expires_at`.
- The dashboard tasks tab shows each deferred row as "parked: <what ends it>" with its age.
  The tab also counts the deferrals that need a human.
- `scripts/deferral_census.py --check` exits 0 for each migrated project. It is run by
  `/review-briefing --validate` check 6. A zero exit means no deferred row lacks a valid
  record. The census also lists the `legacy_unknown` rows left for human triage.

## 2. Background (measured 2026-10-07, read-only, live stores)

- **Population.** dark_factory had 339 deferred tasks of 1,665 non-terminal tasks (6,094
  total) at the start of authoring. By the critic pass it had 335, and reify had 238,
  know_live 10, autopilot_video 1. In dark_factory, 160 of the 339 carried
  `metadata.x_coalesced_into`. All 160 came from the 2026-09-09 backlog sweep and point at 67
  distinct carriers. By carrier status: 45 `done`, 113 `pending`, 2 `in-progress`. Ten of the
  339 are `[auto-eval redo]` copies. Only 2 carry a checkable flip condition in metadata:
  2217 (`deferred_watch`/`trigger`) and 4827 (`x_armed_by`). Another 12 mention "after N
  lands" in prose, and 154 record no reason at all. 82 have dependency edges, and in 58 of
  those every dependency is terminal. None carries a milestone, a claimant, or
  `pending_since`, which is cleared on leaving `pending`. These are "at authoring" figures;
  the ε signal recounts.
- **No history.** The store has no status-history table and no `created_at`. Any metadata
  edit bumps `updated_at`. So two things cannot be measured today: how long a task has been
  in `deferred`, and which planning-born tasks were never committed. 52 deferred rows carry
  `human_decomposed`, which every planning birth sets. Nothing distinguishes an uncommitted
  birth among them from a committed task deferred later. This PRD stamps both facts from now
  on.
- **Producers (non-test code).** Two write paths enter `deferred`.
  - Transitions go through `fused-memory/src/fused_memory/server/tools.py::set_task_status` →
    `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._apply_status_transition`.
    Table A (`shared/src/shared/task_transitions.py::_UNION`, enforce-mode live) admits three
    edges: PENDING→, IN_PROGRESS→ and BLOCKED→DEFERRED.
  - Births go through `submit_task(planning_mode=True)` → `TaskInterceptor._submit_task_planning_mode`.
    This is the only `tm.add_task(status='deferred')` call, and it bypasses Table A.

  Births have four callers:
  - `/prd` decompose and other skill batches.
  - The orchestrator's auto-eval redo (`orchestrator/src/orchestrator/harness.py`). It flips
    the task to `pending` seconds later and, on failure, only logs "left deferred".
  - The de-flake filer (`orchestrator/src/orchestrator/flake_ledger.py`). Its
    `_owner_liveness` treats *any* deferred owner as a half-filed birth and calls
    `commit_planning` on it.
  - Recon Stage-2 decomposition (`fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`).

  `commit_planning(target_status='deferred')` is a third entry. The transition writers are:
  - humans and interactive sessions: the `/unblock` hand hold, the hand-carry guard, backlog
    sweeps;
  - the steward agent role, through its own MCP call (`StewardTerminalDecision`).

  No orchestrator, escalation or shared production code writes `deferred`.
  `SqliteTaskBackend.update_task` refuses `update_task(status=…)` (`StatusWriteAuthorityError`).
  One more path exits `deferred` without the choke point: the candidate-key self-heal in
  `sqlite_task_backend.py` cancels rows with raw SQL.
- **Nobody polls `deferred`.** The stranded reconciler, resume re-pend, warm-lane reclaim and
  auto-eval supersede all skip it (`harness.py::_RECONCILE_SWEEP_STATUSES`,
  `_RESUME_REPEND_STATUSES`, `_WARM_LANE_RECLAIM_PROTECTED_STATUSES`,
  `_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES`). The scheduler dispatches only `pending`. That is
  correct for a deliberate park, and it is also why a park whose reason is lost strands
  forever.
- **The rule exists only as prose.** `review/briefing.yaml` conventions ("Deferred-task
  invariant") and `skills/review-briefing/SKILL.md` check 6 state it, but nothing enforces it.
  Ad-hoc conventions grew in its place:
  - `x_deferred_reason` (2026-07-29);
  - `activation_trigger`;
  - `x_armed_by`;
  - `deferred_watch`/`trigger_met` on pending tasks;
  - `reopen_reason` misused as a lift condition.

## 3. Rulings executed, and departures from the brief's wording

These rulings are executed as given:
- A reason is mandatory, and the server enforces it.
- A hold may lapse back to `pending`.
- The hand-carry hold is a lease. It returns the task to `pending`, and tells a human, when
  its session dies.
- The sweep may only flip `deferred → pending`. It never reaps, cancels or reverts.
- The migration stamps what it can read and marks the rest `legacy_unknown`.
- `merge-deferred` is untouched.

The departures below are the lead's call, under "pause only at a gate the brief cannot
resolve". An adversarial critic seat (opus) judged D-a, D-b, D-c, D-d and D-f justified.
Leo confirmed D-e on 2026-10-08 with one change: no notice of any kind while the holder lives.

| # | Brief | This PRD | Why |
|---|---|---|---|
| D-a | kind `until_task(ref)` | kind `carried_by(carrier_task_id)` | Heuristic 1, informative names. The briefing invariant says "wait for task X, then run me" is a dependency edge on a `pending` task. `until_task` reads as exactly that and would invite misuse. `carried_by` says the carrier absorbed the work and a human closes this row when the carrier ends (decision 8). |
| D-b | kind `until_time` (reuse milestone + sweep) | no `until_time` kind; optional `expires_at` on `until_condition` | Heuristic 3: *what ends a hold* and *whether it lapses* are independent axes. Leo's words ("optionally a deferral can be of limited duration") describe a bound on a human-owned hold, not a kind. A gate that is *only* time belongs on a `pending` task with `metadata.milestone {mode: 'dated'}`, which is landed and restart-safe and needs no status flip (heuristic 11: one "not before T" mechanism). The milestone gate could not be reused for a `deferred` row anyway, because `Scheduler._milestone_time_gated` withholds `pending` tasks and never writes a status. |
| D-c | four kinds | adds `planning`, which the server stamps on every `planning_mode` birth | Four birth paths write `deferred`, and no caller on them can supply a reason. Births must not be refused, and an abandoned batch must be visible rather than silent. `commit_planning` may release only this kind (decision 3). |
| D-d | field `flip` | field `deferral` | The record carries the reason and the owner as well as the flip. `flip` also collides with the write-triage "flip" vocabulary already in the corpus. |
| D-e | "TTL renewed by heartbeat; on expiry flip + escalate"; lease mirrors the claimant predicates | **the lease is the holder process's life.** A live holder never lapses. Death starts a grace window (`grace_secs`, default 2 h). Only death plus grace flips the task, and the notice is filed at death. A live holder is never notified, however long it holds (Leo, 2026-10-08). Claimant columns are untouched. | The brief itself names the hazard a fixed TTL creates: "if the session is alive but slow, a flip to pending lets the scheduler dispatch an agent onto the same files". The available heartbeat is turn-granular: a session working through one long turn sends none (measured, §7). So any timeout on a *live* holder would release work that is still being written. The claimant PRD's D2/B2 forbid overloading `claimant_run_id`. **Ruled (Leo, 2026-10-08):** confirmed, without the idle notice — "sometimes complex questions get pushed to the back of the queue and take days to get answered. This is fine." |
| D-f | "rejected with a typed `SetTaskStatusRejected` subclass" | the server returns error dicts with stable codes; the orchestrator client maps them to a new `DeferralRejection` subclass | `SetTaskStatusRejected` is a client-side class in `orchestrator/src/orchestrator/scheduler.py`. Every server gate returns `{'success': False, 'error': <code>, …, 'hint'}`. |

## 4. Resolved design decisions

### 1. One record, one home

`metadata.deferral` is the only home of "why is this deferred, and what ends it" (INV-9). It
is a typed sub-model defined in a new module, `shared/src/shared/task_deferral.py`. That
module registers it with `register_metadata_submodel('deferral', …, cardinality='dict')`,
the way `shared/src/shared/deploy_state.py` and `capability_manifest.py` register their own
keys. A discriminated union needs a `RootModel` wrapper, because the registry takes a
`BaseModel`. The module owns:
- the kind vocabulary (`DeferralKind`);
- the caller-input and stored models;
- the pure predicates every consumer uses: `lapse_cause`, `needs_human`.

The server gate, the sweep, the dashboard and the census all import it, and none restates a
kind list (INV-5). `docs/task-authoring.md` §8 documents the key.

The invariant has both directions: a row carries a record **iff** its status is `deferred`.
Entering `deferred` writes the record in the same transaction as the status. Leaving
`deferred` by any path removes it in the same transaction (decision 4). The census reports a
violation either way (decision 10).

### 2. Kinds: what ends the hold, who ends it, what bounds it

| Kind | Caller fields | Exit owner (INV-7) | Bound (INV-7) | Who may write it |
|---|---|---|---|---|
| `carried_by` | `carrier_task_id` | a human: once the carrier is terminal, they close the row (`done` with `found_on_main`, or `cancelled`) or re-pend it | carrier `done` ⇒ **closable**; carrier `cancelled` ⇒ **orphaned** (the absorbed work never landed: re-pend or re-carry). While the carrier is live, the scheduler owns the carrier and so bounds the hold; a deferred carrier has its own record. | any caller |
| `until_condition` | `condition` (text, ≤ 500 chars); optional `expires_at` | a human who judges the condition | `expires_at` ⇒ the sweep flips to `pending`. Without it, the hold is **stale** after 30 days and is listed for review. | any caller |
| `held_by_session` | `holder_pid` (the caller's `$CLAUDE_PID`); optional `grace_secs` | the holding session, which lands or releases it | holder death ⇒ one INFO notice; death observed for `grace_secs` ⇒ flip to `pending`; while the holder lives, nothing (decision 6) | any caller whose pid is a live `claude` process on the fused-memory host |
| `planning` | none | the planner, via `commit_planning` | **stale batch** after 24 h; never lapses, because an unwired batch must not auto-release | the server only, at `planning_mode` birth |
| `legacy_unknown` | none | the migration triage gate (ζ) | listed with a count until triaged | the migration only (decision 9) |

Every kind accepts an optional free-text `reason` (≤ 2,000 chars). Nothing parses it
(heuristic 12): routing reads `kind` and the typed fields, never prose.

Three other shapes are not deferrals, and the `deferral_required` hint and the new
`docs/task-authoring.md` recipe "Deferring a task" name all three:
- A hold that is only a dependency is a `pending` task with `add_dependency`.
- A hold that is only time is a `pending` task with `metadata.milestone {mode: 'dated'}`.
- A hold that is a human judgment needed at dispatch is the briefing's escalate-on-dispatch
  shape: a pure gate (`execution_class='operational'`, a born-at-L2 escalation with a
  supervised consumer).

### 3. Write paths

All deferral logic lives behind **one call** in `_apply_status_transition`:
`deferral_gate.plan(before, status, request, expected_stamped_at, …) -> GateRefusal | DeferralWrite`.
The call sits before the same-status no-op guard. `DeferralWrite` carries four things: the
`audit_fields` delta, the keys to remove, the event fields, and whether this is a re-stamp.
The interceptor applies `DeferralWrite`; it does not interpret deferrals (heuristics 4 and 9).

- **Entering `deferred`** via `set_task_status(status='deferred')` from `pending`,
  `in-progress` or `blocked`. A `deferral` is required.
  - The gate refuses inside the project write lock, before any state changes.
  - The stored record rides in `audit_fields`, so `SqliteTaskBackend.set_status_and_stamp_audit`
    commits the status and the record in one transaction.
  - The CSV path applies the same record to every id.
- **Re-stamping a deferred row** via `set_task_status(status='deferred', deferral=…)` on a
  row already in `deferred`. This replaces the record. It is the one way to:
  - change a hold's kind;
  - move a lease to a new session (after `/resume`);
  - turn a hand-carried task into `carried_by(<carrier>)` when its merge moves to a carrier task;
  - give a planning-born task a real reason.

  `deferred_at` (first entry) is preserved; `stamped_at` and `stamped_by` are new. A
  `deferred` write without a `deferral` on a `deferred` row stays today's no-op.
- **`commit_planning`** gains `agent_id`.
  - `commit_planning(target_status='pending'|'cancelled')` flips only rows whose record is
    `planning`, or absent (the pre-migration window). Any other kind is refused
    (`reason_code='not_planning_hold'`), so a human hold or a live hand-carry can never be
    released by a batch commit.
  - `commit_planning(target_status='deferred')` requires `deferral` (a caller kind) and
    forwards it as a re-stamp.
  - `orchestrator/src/orchestrator/flake_ledger.py` completes its owner only when the owner's
    record is `planning`; otherwise it treats the owner as held.
- **`submit_task(planning_mode=True)`** stamps `{kind: 'planning'}` itself, attributed to the
  submitting `agent_id`. Any `submit_task` carrying a caller-supplied `metadata.deferral` is
  refused in `TaskInterceptor.submit_task` before the ticket/planning split. The backend's
  `add_task` stays neutral, because it cannot tell the interceptor from a caller.
- **`update_task(metadata=…)`** may not add, alter or remove `deferral`, in any metadata
  mode. A byte-equal echo of the stored record passes, so read-modify-write writers that
  return the blob unchanged keep working. This clones the `done_provenance` write floor in
  `SqliteTaskBackend.update_task`, with a new `DeferralWriteAuthorityError` in
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`.
- **A `deferral` with any status other than `deferred`** is refused (`status_not_deferred`).

### 4. Leaving `deferred` clears the record, and says what it cleared

Every edge out of `deferred` (→ `pending`, `done`, `blocked`, `cancelled`) goes through
`_apply_status_transition`. So `DeferralWrite` removes the key there, in the same transaction
as the status write. That one site covers recon's `_sweep_cancel_orphan` and
`_sweep_block_orphan`, a hand-carry's `deferred → done`, `commit_planning`, and the sweep's
flip, with no per-caller code.

The one raw-SQL exit, the candidate-key self-heal's cancel in `sqlite_task_backend.py`, drops
`deferral` in its own metadata write.

The `task_status_changed` event gains two fields: `deferral` (the record written, on entry)
and `deferral_cleared` (the record removed, on exit). The journal therefore keeps the only
history of holds this store will have (INV-2). A re-stamp emits `deferral_restamped` with the
old and new records.

### 5. Validation corroborates what it can

- **`carried_by`.** The carrier exists in the same project and is not the task itself. A
  terminal carrier is accepted; the row is immediately closable or orphaned.
- **`until_condition`.** `condition` must be non-empty after strip. `expires_at`, when given,
  must parse as ISO-8601 UTC and lie in the future. A tz-naive value is read as UTC, as for
  `Milestone.at`.
- **`held_by_session`.**
  - The server reads `/proc/<holder_pid>` on its own host. The process must be the Claude CLI
    (`/proc/<pid>/comm` is `claude`). A missing process is refused with `holder_not_live`. A
    different process is refused with `holder_not_session`: a shell or session leader can
    outlive a crashed `claude`, and would otherwise pin the hold forever.
  - The server stores a pid-reuse- and reboot-safe identity `{pid, start_ticks, boot_id, host}`
    (new `shared/src/shared/process_identity.py`). This corroborates the claim at write time
    (INV-3).
  - `grace_secs` defaults to 7,200 and must lie in [600, 86,400]. The default was chosen to
    match the session registry's `LEASE_HEARTBEAT_TTL`, but the two are separate policies.
- **Server-only kinds.** `planning` and `legacy_unknown` from a caller are refused
  (`server_only_kind`), except `legacy_unknown` in decision 9's migration shape.
- **Refusals.** Every refusal names `reason_code`, `field` and `value` (INV-2). The `hint`
  lists the accepted kinds and the three not-a-deferral shapes of decision 2.

### 6. The `held_by_session` lease

**Identity.** The caller passes `holder_pid`, which is its `$CLAUDE_PID`: the Claude CLI
process. The session uuid is not a key, because `/clear` and compaction re-mint it. `/team`
subagents and in-process forks see the parent's `$CLAUDE_PID`. The hold then belongs to the
parent session, which is the process whose death matters.

**Liveness** comes from the shared `process_identity.liveness(identity)`, the one liveness
test everywhere:
- **`ALIVE`** means the same host, the same boot, and a process with that pid and start ticks
  exists.
- **`DEAD`** means the same host and the identity no longer exists: the pid is gone, its start
  ticks differ, or the boot id differs. A **reboot is death**.
- **`OTHER_HOST`** means another host. Liveness is unknown, so the sweep holds the row and surfaces
  it, and never flips it.

**Lapse.** `lapse_cause(record, now, dead_since)` returns `holder_lost` when the sweep has
observed the holder dead continuously for at least `grace_secs`. `dead_since` is the time this
sweep process first saw it dead. It is held in memory and reset when the orchestrator
restarts. A restart can only lengthen the grace window, never shorten it, so a hold is never
released early. The grace window exists so that a crashed session the human `/resume`s, which
comes back as a new process, can re-stamp the hold with its new pid before the task is
released. Grace is anchored on observed death, not on a heartbeat file, for two reasons.
Third parties rewrite a dead session's `record.json`
(`session_registry.py::_mark_exited_if_still_non_terminal`). And a heartbeat already old at
death would leave no grace at all.

**A live holder never lapses.** This holds however long the holder is silent. A session
blocked for three hours in one build sends no heartbeat (the hooks fire per turn end), and
flipping its hold would let the scheduler dispatch a second writer onto the same files.

**Live means silent (Leo, 2026-10-08).** While the holder lives, nothing is filed, nothing is
flipped, and the hold is not in `needs_human`, however long it lasts: a session may wait days
for its human. The dashboard still draws the row "held by session <pid> (live)" with its age
— a display, not a notice. So no part of the design reads a session heartbeat: liveness is
`process_identity` alone, everywhere. The sweep's only use of the session registry is a
best-effort `session_registry.resolve_session_slug_for_pid` to name a dead holder in its
notice.

**A merge in flight.** Before a `holder_lost` flip, the sweep checks its own harness's merge
queue. If an entry names the task, it holds and surfaces `merge_in_flight`. When a hand-carry
moves the merge to a separate carrier task, the hand-carry guidance re-stamps the real task
`carried_by(<carrier>)`, because the carrier now carries the work.

### 7. The expiry sweep

- **Host.** A new `BackgroundService` named `deferral-sweep` is registered in
  `orchestrator/src/orchestrator/harness.py`, like `stranded-reconcile`. Its logic lives in a
  new module, `orchestrator/src/orchestrator/deferral_sweep.py`. Background services keep
  running while the scheduler is halted; phases hosted in `Scheduler.acquire_next` do not.
  Config keys (green-tier, like the other sweep intervals):
  - `deferral_sweep_enabled` (default true);
  - `deferral_sweep_interval_secs` (default 60);
  - `deferral_flip_failure_streak` (default 5).

  Every project's orchestrator runs it for its own project. A project with no running
  orchestrator gets no sweep: its holds are surfaced, never flipped.
- **Pass.** The sweep reads the project's `deferred` rows (`get_tasks(statuses=['deferred'])`)
  and parses each record with the shared model. One pure function decides,
  `decide(record, now, liveness, dead_since, merge_in_flight) -> Hold | Notify | Flip(cause)`:

  | Row | Decision |
  |---|---|
  | `until_condition` with `expires_at ≤ now` | `Flip('expired')` |
  | `held_by_session`, holder dead, no merge in flight | `Notify` at first sight of death, then `Flip('holder_lost')` when `lapse_cause` fires |
  | anything else | `Hold` |
  | missing or unparseable record | `Hold`, surfaced as `invalid_record`; never a flip |

  The last row fails safe toward holding, as the milestone gate fails toward withholding.
- **Notices.** A notice is one born-at-L2 INFO escalation per holder. It is filed against a
  stable sentinel task id derived from the holder identity
  (`deferral-holder:<host>:<boot_id8>:<pid>:<start_ticks>`), following the
  `_DIRTY_TREE_ESCALATION_SENTINEL` precedent, with `agent_role='orchestrator-deferral-sweep'`
  (a harness sentinel role).
  - It names the holder (pid, best-effort slug), every task it holds, and what will happen
    ("released to `pending` at <t>" / "released").
  - Dedup is by sentinel id in any status, so one dead holder is reported once, across
    restarts.
  - Filing it against a sentinel rather than the re-pended task leaves that task's dispatch and
    pins untouched.
  - The L2 watcher closes it as it does other sentinel notices (`close_only`).
  - The notice is filed **before** the flip, so a crash between the two cannot lose it; the
    next pass finds the escalation and completes the flip.
  - `expires_at` lapses file nothing, because the human asked for that outcome. They emit only
    `deferral_expired` events.
- **Guarded write.** A flip calls `set_task_status(status='pending',
  expected_stamped_at=<record.stamped_at>)` with:
  - `agent_id='orchestrator-deferral-sweep'`;
  - `client_op_id=f'deferral-flip:{task_id}:{stamped_at}'`, so a transient-retry replays the
    recorded outcome instead of reading its own success as a conflict.

  Inside the write lock, the server compares `expected_stamped_at` with the current record. It
  refuses with `deferral_changed` if someone re-stamped in the meantime. This is INV-3: the
  re-check sits at the choke point, not in the sweep's stale read.

  The sweep also skips any row for which
  `shared/src/shared/task_claimant.py::has_live_claimant(task, now, ttl)` holds, with `ttl`
  equal to the orchestrator's `claimant_liveness_ttl_secs` (the dispatch gate's value). A live
  claimant on a `deferred` row means a workflow teardown is in flight (claimant PRD D2/B10).
  This is not `fused-memory/src/fused_memory/middleware/live_task_write_guard.py::has_live_claimant`,
  a different, in-progress-only function of the same name. The sweep writes only `pending`.
- **Paused scheduler.** While `Scheduler.is_paused` (the operator or automatic dispatch halt):
  - The sweep still files notices, saying "will be released when the scheduler resumes".
  - It flips nothing. The pass event records `would_flip`. Nothing is re-pended into a halted
    scheduler, so this PRD adds no "re-pended while paused" signal; the stranding PRD's
    `repend_while_paused` (task 5311) stays the only one.
  - The first unpaused pass flips.
  - A halted scheduler cannot dispatch onto the holder's files, so the delay carries no
    two-writer risk.
  - Other pauses, such as the usage cap, gate capacity rather than intent, and the sweep
    ignores them. A task flipped during a usage-cap pause waits like any `pending` task.
- **Storm escape (INV-4).** A failed flip is retried on the next pass and counted per task.
  After `deferral_flip_failure_streak` consecutive failures, the sweep files one blocking
  escalation for that task and stops retrying it until its record changes. A pass event
  `deferral_sweep_pass {evaluated, flipped, would_flip, notified, surfaced, failed}` is emitted
  only when some count is non-zero.
- **Per-pass cost (INV-8).** Each pass makes one `get_tasks` call, two small `/proc` reads per held row, one merge-queue lookup per lapsing row, and awaited MCP writes. That
  was 335 rows at authoring.

### 8. Why `carried_by` is not a dependency edge

A dependency edge gates the dispatch of a task that still has its own work to do: when the
prerequisite lands, the dependent runs. A carried task has **no work of its own left**,
because the carrier absorbed it. Releasing it to the scheduler when the carrier lands would
dispatch the same work twice. So the carrier's termination makes the row closable or
orphaned. That is a human decision (did the carrier really cover it?), and it never
dispatches the row. Anything that should *run* after X is a `pending` task with
`add_dependency(X)`, and the hint and the recipe say so.

### 9. Migration and `legacy_unknown`

`legacy_unknown` is accepted only as a re-stamp of a `deferred` row that has **no** record.
This is a data-shape rule, not an actor rule; task-status-authority D5 forbids keying writes
on actor identity. Every new entry into `deferred` must carry a caller kind, so once a
project's rows are all stamped, no write can produce `legacy_unknown` there again. A re-stamp
of an unrecorded row also sets `deferred_at = null`, meaning the entry time is unknown. `null`
is legal only in that case.

`scripts/migrate_deferrals.py` is dry-run by default and writes only with `--apply`. It takes
one or more `--project-root` values and re-stamps through the MCP, idempotently: rows that
already have a valid record are skipped. Its mapping:
- `x_coalesced_into` → `carried_by(carrier_task_id)`;
- `deferred_watch` with `trigger`, or `x_armed_by` → `until_condition(condition=<that text>)`;
- everything else → `legacy_unknown`. The `reason` notes "human_decomposed: possibly an
  uncommitted planning birth" where that flag is set, because `pending_since` cannot tell the
  two apart (§2).

Its report lists the closable and orphaned `carried_by` rows (45 closable in dark_factory at
authoring). Applying the migration and triaging what it leaves is a human gate (ζ).
`x_coalesced_into` stays on the rows as the 2026-09-09 sweep's historical stamp; the live
claim is `deferral.carrier_task_id`.

### 10. Surfaces

`task_deferral.needs_human(record, now, carrier_status, liveness)` defines, once, what a human
must act on:
- `legacy_unknown`;
- `carried_by` whose carrier is `done` (closable) or `cancelled` (orphaned);
- `planning` older than 24 h (a stale batch; an unknown age counts as stale);
- `until_condition` without `expires_at` older than 30 days (stale);
- `until_condition` past `expires_at` by more than two sweep intervals (the sweep is not
  running);
- `held_by_session` whose holder is dead (release pending), or on another host;
- any missing or invalid record.

A `held_by_session` row with a live holder is never in `needs_human` (decision 6).

- **Dashboard.** `dashboard/src/dashboard/data/active_tasks.py::_build_task_row` and the tasks
  tab (`dashboard/src/dashboard/static/redux/tab_tasks.jsx`) show a deferred row as one of:
  - "parked: carried by #N (closable | orphaned)";
  - "parked until <condition> (lapses <date>)";
  - "held by session <pid> (live | lost, releasing)";
  - "planning batch";
  - "legacy: no recorded reason".

  Each row also shows its age, and the tab header counts deferrals needing a human. A deferred
  row with no record reads "no deferral recorded", a state distinct from every kind (INV-13).
  The dashboard is multi-project, so every project's rows render. The lanes, PRD grouping and
  the status-review redesign are out of scope.
- **Census.** `scripts/deferral_census.py --project-root …` classifies every deferred row with
  the shared model over the MCP read path (`get_tasks(statuses=['deferred'])`). It also checks
  non-deferred rows for a stray `deferral`.
  - `--check` exits non-zero on any missing or invalid record, any stray record, or any
    `expires_at` lapse left unflipped for more than two sweep intervals.
  - It lists `legacy_unknown`, closable, orphaned, stale and lost-holder rows without failing.
  - If a project has deferred rows but no record anywhere, it says "no deferral producer has
    written" rather than "all clean".
  - `/review-briefing --validate` check 6 runs it in place of today's prose check.

### 11. Enforcement is redundant and uniform (heuristic 10)

There is one definition, `shared/src/shared/task_deferral.py`, and it is enforced at four
points:
- **Construction:** the pydantic models.
- **The status choke point:** entry, re-stamp, the `commit_planning` scope, CAS, and
  clear-on-exit.
- **Every other metadata writer:** the `update_task` floor, the `submit_task` refusal, and the
  raw-SQL self-heal.
- **Every reader:** the sweep, dashboard and census parse with the same model and report a
  missing or invalid record as such, never as "no hold".

The census's `--check` is the standing proof that the first three points left no gap.

### 12. Enforced from landing, with no warn-mode window

The memory-vocabulary lesson (strict rejection of an open population fails) is about
re-validating an existing population. That does not apply here. The existing rows are never
re-validated on write; the migration stamps them. The writer set is closed and known
(decision 3 and §2 producers), and every refusal carries a hint complete enough for an agent
to correct its own call (the INV-1 house pattern). β lands the enforcement together with
every repo-tracked writer's instructions.

The gate is server-wide, so it binds reify, know_live and every other project at once. The
`/unblock`, `/do` and `/prd` skill files live in this repo, and `~/.claude/skills` links to
them, so β's edits reach every project. Any project-local prose elsewhere is corrected by the
hint.

The live effect needs a fused-memory restart onto β's code. The orchestrator side (γ) needs
an orchestrator restart.

### 13. Placement (heuristics 9, 13, 14)

The hosts are very large: `harness.py` 18,218 lines, `tools.py` 10,565, `scheduler.py` 9,883,
`task_interceptor.py` 7,206, `session_registry.py` 5,772. New logic therefore lives in new
modules, each statable in one sentence and readable alone:
`shared/.../task_deferral.py` (pure), `shared/.../process_identity.py` (the `/proc` reads),
`fused-memory/.../middleware/deferral_gate.py` (the request → write-plan decision) and
`orchestrator/.../deferral_sweep.py` (the pass).

The hosts gain thin wiring only:
- `_apply_status_transition`: one `plan()` call and the application of its result;
- `tools.py`: argument pass-throughs;
- `harness.py`: one service registration;
- `scheduler.py`: one error mapping.

`session_registry.py` is read, not edited.

## 5. Contract

### 5.1 The record (`shared/src/shared/task_deferral.py`)

```
class DeferralKind(StrEnum):
    CARRIED_BY = 'carried_by'
    UNTIL_CONDITION = 'until_condition'
    HELD_BY_SESSION = 'held_by_session'
    PLANNING = 'planning'              # server-only
    LEGACY_UNKNOWN = 'legacy_unknown'  # migration-only

CALLER_KINDS = frozenset({CARRIED_BY, UNTIL_CONDITION, HELD_BY_SESSION})

# caller input (the MCP `deferral` argument), discriminated on `kind`
DeferralRequest =
    {kind: 'carried_by',      carrier_task_id: int,             reason?: str}
  | {kind: 'until_condition', condition: str, expires_at?: str, reason?: str}
  | {kind: 'held_by_session', holder_pid: int, grace_secs?: int, reason?: str}
  | {kind: 'legacy_unknown',  reason?: str}       # accepted only per decision 9

# stored as metadata.deferral (frozen, heuristic 8)
Deferral = <the request's kind fields, with holder_pid replaced by
            holder: ProcessIdentity and grace_secs filled with its default>
         + deferred_at: datetime | None   # first entry; None only for a re-stamped unrecorded row
         + stamped_at: datetime           # this record's write; the CAS key
         + stamped_by: str | None         # agent_id; None renders "unattributed"

DEFAULT_GRACE_SECS = 7200; GRACE_BOUNDS = (600, 86400)
STALE_PLANNING_AFTER_SECS = 86400; STALE_CONDITION_AFTER_SECS = 30 * 86400

class Liveness(StrEnum): ALIVE, DEAD, OTHER_HOST
def lapse_cause(record, now, dead_since: datetime | None) -> LapseCause | None
def needs_human(record, now, carrier_status, liveness) -> NeedsHumanReason | None
```

The predicates take liveness and times as arguments and do no I/O (heuristic 7).

### 5.2 `shared/src/shared/process_identity.py`

```
@dataclass(frozen=True)
class ProcessIdentity: pid: int; start_ticks: int; boot_id: str; host: str
def capture(pid: int) -> ProcessIdentity | None   # None: no such process
def command_name(pid: int) -> str | None          # /proc/<pid>/comm
def liveness(identity: ProcessIdentity) -> Liveness
    # OTHER_HOST iff host differs; DEAD iff same host and (boot_id differs, pid absent,
    # or start_ticks differ); else ALIVE
```

### 5.3 MCP surface (fused-memory)

```
set_task_status(id, status, project_root, tag=None, done_provenance=None,
                reopen_reason=None, claimant_run_id=…, heartbeat_at=…,
                agent_id=None, client_op_id=None,
                deferral: dict | None = None,              # NEW
                expected_stamped_at: str | None = None)    # NEW, single id only
commit_planning(project_root, task_ids, target_status='pending',
                deferral: dict | None = None,              # NEW, required iff target_status='deferred'
                agent_id: str | None = None)               # NEW
```

Refusals are dicts: nothing is persisted, and each is journalled with `success=False`.

| `error` | When | Typed fields |
|---|---|---|
| `deferral_required` | entering `deferred`, or `commit_planning` to `deferred`, without `deferral` | `task_id`, `from_status`, `accepted_kinds`, `alternatives`, `hint` |
| `deferral_invalid` | any validation failure | `task_id`, `reason_code`, `field`, `value`, `hint` (`reason_code` values below the table) |
| `deferral_changed` | `expected_stamped_at` does not match the current record's `stamped_at`, or the row has left `deferred` | `task_id`, `expected_stamped_at`, `current_stamped_at`, `current_status` |
| `deferral_write_authority` | `update_task` or `submit_task` metadata adds, alters or removes `deferral` | `task_id` (if any), `path`, `hint` |

`reason_code` for `deferral_invalid` is one of: `unknown_kind`, `missing_field`,
`field_not_allowed`, `server_only_kind`, `holder_not_live`, `holder_not_session`,
`carrier_not_found`, `carrier_is_self`, `expires_at_not_future`, `grace_out_of_range`,
`text_too_long`, `status_not_deferred`, `not_planning_hold`, `cas_needs_single_id`.

Events: `task_status_changed` gains `deferral` (on entry) and `deferral_cleared` (on exit),
and a new event `deferral_restamped {task_id, old, new}` covers re-stamps.

### 5.4 Orchestrator

- **`orchestrator/src/orchestrator/scheduler.py`.**
  - `DeferralRejection(SetTaskStatusRejected)` carries `reason_code`, and
    `Scheduler.set_task_status` raises it for the four `deferral_*` codes.
  - `Scheduler.set_task_status` forwards `expected_stamped_at`.
- **`deferral_sweep.py`.** It contains `decide(...)` (pure) and `run_deferral_sweep_pass(...)`
  (the `BackgroundService` pass function). Events: `deferral_sweep_pass`, and
  `deferral_expired {task_id, kind, cause, record}`.
- **`flake_ledger.py`.** The de-flake owner is completed only when its record is `planning`
  (decision 3).

## 6. Boundary-test sketch

All rows run against a real SQLite task store in a temp `project_root`. They drive the public
tool and interceptor functions with real processes and never patch private names (Tests
stance, `docs/code-quality.md`). The owning leaf is in brackets.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Entry without a record [β] | task `pending` | `set_task_status(deferred)` → `deferral_required`; `get_task` still `pending`; journal row `success=False` |
| 2 | Each caller kind accepted [β] | `pending`, `in-progress` and `blocked` tasks; a live `claude`-named helper process | `deferred` with a valid record for each kind; `metadata.deferral` round-trips through the shared model; `stamped_by` = caller agent_id |
| 3 | Malformed record [β] | `until_condition` with `condition: "  "` | `deferral_invalid`, `reason_code='missing_field'`, `field='condition'` |
| 4 | Holder corroboration [β] | a live `sleep` (comm ≠ `claude`); an unused pid | `holder_not_session`; `holder_not_live` |
| 5 | Exit clears [β] | a deferred row | `→ pending`, `→ done (found_on_main)`, `→ cancelled` and `→ blocked` each leave no `metadata.deferral`; the event carries `deferral_cleared` |
| 6 | Re-stamp [β] | `planning`-born row | `set_task_status(deferred, deferral=until_condition)` replaces the record; `deferred_at` kept; `deferral_restamped` emitted; a `deferred` write without `deferral` is still a no-op |
| 7 | Planning birth and commit scope [β] | `submit_task(planning_mode=True)`; a second task deferred `until_condition` | birth row has `kind='planning'`; `commit_planning(pending)` clears it; `commit_planning(pending)` on the second → `not_planning_hold`, row unchanged; `commit_planning(deferred)` without `deferral` → `deferral_required` |
| 8 | Floor [β] | deferred row | `update_task(metadata={'deferral': <altered>})` in merge, additive and replace modes → `deferral_write_authority`; replace without the key → refused; an echo of the stored record → accepted, record unchanged; `submit_task(metadata={'deferral': …})` → refused |
| 9 | CAS and replay [β] | deferred row re-stamped after a read | `set_task_status(pending, expected_stamped_at=<old>)` → `deferral_changed`; with the current value → `pending`; the same call replayed with its `client_op_id` returns the recorded success |
| 10 | Legacy shape [β] | a row inserted `deferred` with no record | `legacy_unknown` re-stamp accepted with `deferred_at=null`; the same kind on a recorded row → `server_only_kind` |
| 11 | Raw-SQL exit [β] | deferred row hit by the candidate-key self-heal | cancelled, no `metadata.deferral` |
| 12 | Liveness [α] | a spawned process; its identity with a forged old `boot_id`; with another `host` | `ALIVE`; `DEAD` (reboot is death); `OTHER_HOST` |
| 13 | Holder lost [γ] | `claude`-named helper process holds a row (`grace_secs=600`), then is killed; fake clock | first pass after death → one sentinel INFO notice, row still `deferred`; pass 601 s after first-seen-dead → row `pending`, `deferral_expired{cause:'holder_lost'}`; no second notice |
| 14 | Live holder is silent [γ] | live holder; fake clock advanced 7 days | `Hold` on every pass; no escalation filed; not in `needs_human`; never flipped |
| 15 | `expires_at` lapses [γ] | `until_condition`, `expires_at` in the past | one pass → `pending`; no escalation; `deferral_expired{cause:'expired'}` |
| 16 | Paused [γ] | row as in 13, scheduler paused | notice filed ("released on resume"); no flip; `would_flip=1`; first unpaused pass flips |
| 17 | Race [γ] | row as in 15; re-stamp between the sweep's read and its write | `deferral_changed`; row keeps the new record; counted as a benign skip, not a failure |
| 18 | Guards [γ] | row as in 15 with a fresh claimant; a held row whose task id is in the merge queue | both held; the second surfaced `merge_in_flight` |
| 19 | Failure streak [γ] | server refusing writes for the row | after 5 passes, one blocking escalation; no retries until the record changes |
| 20 | Restart [γ] | holder dead, notice filed, sweep restarted | no duplicate notice (sentinel dedup); grace restarts from the new first-seen-dead |
| 21 | Client mapping and de-flake [γ] | stub server returning each `deferral_*` code; a de-flake owner deferred `until_condition` | `DeferralRejection` with the right `reason_code`; the ledger leaves the owner alone |
| 22 | Census [ε] | fixture: each kind, one unrecorded deferred row, one stray record on a `pending` row | `--check` exits non-zero naming both; after the fixes it exits 0 and lists `legacy_unknown`, closable and orphaned rows |
| 23 | Migration [ε] | fixture with `x_coalesced_into`, `x_armed_by`, `deferred_watch` and bare rows across two project roots | dry-run prints the mapping per project; `--apply` stamps it; a second `--apply` writes nothing |
| 24 | Dashboard [δ] | the row-22 fixture | each row renders its "parked: …" text and age; the unrecorded row reads "no deferral recorded"; the header count equals `deferral_census.py`'s needs-human count over the same store |

## 7. Pre-conditions (G3): substrate verified on `5b645d822a`

| Capability | Evidence |
|---|---|
| Single status choke point; validators run before mutation | `task_interceptor.py::TaskInterceptor._apply_status_transition` (gate chain under `_write_lock`; `_validate_done_provenance` at step 2b; same-status guard with the `done → done` repair seam `_repair_done_provenance_same_status`) |
| Atomic status + metadata write | `sqlite_task_backend.py::SqliteTaskBackend.set_status_and_stamp_audit` (one `_txn`, shallow merge, `_write_status_and_verify`). There are no key-deletion semantics yet, so β adds the narrow removal |
| Write-authority floor to clone | `SqliteTaskBackend.update_task` `done_provenance` floor, `_assert_done_provenance_passthrough`, `task_backend_errors.py::DoneProvenanceWriteAuthorityError` |
| Only one birth path | `TaskInterceptor._submit_task_planning_mode` is the only `tm.add_task(status='deferred')` call; the curator and `task_knowledge_sync` births default to `pending` |
| `commit_planning` goes through the choke point | `tools.py::commit_planning` → `TaskInterceptor.set_task_status` (CSV); it has no `agent_id` today |
| Recon exits go through the choke point | `targeted.py::_sweep_cancel_orphan` and `_sweep_block_orphan` call `task_interceptor.set_task_status` |
| Typed sub-model registry | `shared/src/shared/task_metadata.py::register_metadata_submodel`; self-registering `deploy_state.py`, `capability_manifest.py` |
| A sweep host that survives a halt | `orchestrator/src/orchestrator/background_service.py::BackgroundService`; `harness.py` registers `stranded-reconcile` and others; the paused branch says background services keep running |
| Pause predicate | `orchestrator/src/orchestrator/scheduler.py::Scheduler.is_paused` |
| Live-claimant predicate | `shared/src/shared/task_claimant.py::has_live_claimant(task, now, ttl)`, as used by the dispatch gate |
| Sentinel born-at-L2 escalation precedent | `harness.py::_DIRTY_TREE_ESCALATION_SENTINEL`; harness sentinel role prefix `orchestrator-` (`escalation/src/escalation/server.py::_HARNESS_SENTINEL_ROLE_PREFIXES`) |
| Naming a dead holder (best-effort) | `session_registry.py::resolve_session_slug_for_pid` (pid → slug via `~/.claude/fleet/sessions-by-pid`); `LEASE_HEARTBEAT_TTL` is the precedent the default `grace_secs` matches (not imported) |
| `$CLAUDE_PID` names the Claude CLI | measured 2026-10-07 in the authoring session: `/proc/$CLAUDE_PID/comm` = `claude`; `/proc/$CLAUDE_PID/stat` field 22 and `/proc/sys/kernel/random/boot_id` readable; `~/.claude/fleet/sessions-by-pid/$CLAUDE_PID` resolves to the session's slug; a `/team` subagent sees the same pid |
| The server and the sweep can read other processes' `/proc` | `systemctl --user show` on `fused-memory.service` and `orchestrator-dark-factory.service`: `ProtectProc=default`, `ProcSubset=all`; the fused-memory transport is local (`127.0.0.1:8002`) |
| The heartbeat is per turn, not continuous | the authoring session's `record.json` mtime was 21 min old mid-turn while it worked continuously, which is why a live holder must never lapse (decision 6) |
| Client rejection mapping | `scheduler.py::SetTaskStatusRejected` and its subclasses; `Scheduler.set_task_status` `error_code` branches; transient retry |
| Deferred-row read for the sweep and census | `tools.py::get_tasks(statuses=…)` |
| Script MCP transport | `scripts/legibility/census_trigger.py::post_mcp_tool_call` |
| Dashboard row shaping; dashboard depends on `shared`, not `orchestrator` | `dashboard/src/dashboard/data/active_tasks.py::_build_task_row`; `dashboard/pyproject.toml` |

New substrate built by this PRD's leaves:
- α: `task_deferral.py` and `process_identity.py`;
- β: the gate, the arguments and the floor;
- γ: the sweep and the client subclass.

There is no external prerequisite.

## 8. Cross-PRD relationship (G4)

| Other PRD / surface | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-status-authority-prd.md` (Table A, C1/D1/D5) | consumes | the choke point `_apply_status_transition`. Table A edges are unchanged. The deferral rule is a payload rule layered beside legality, not a new table, and it keys on data shape, not actor (D5) | **this PRD** | that PRD landed; enforce-mode live |
| `docs/prds/claimant-invariant-enforcement.md` (4618 pending; carrier 4866 landed `bfc61f6624` 2026-10-07) | consumes read-only | `shared.task_claimant.has_live_claimant` is the sweep's guard. Entering `deferred` leaves claimant columns untouched (its B2), and the lease never writes them | **this PRD** owns the guard call; that PRD owns claimant semantics | 4866 landed terminal-claimant clearing and the post-lock recurrence mint in `_apply_status_transition`; neither changes a premise here (`has_live_claimant` unchanged; `plan()` files no task under the lock). This PRD's wiring there is one `plan()` call, with no dependency edge |
| `orchestrator/src/orchestrator/session_registry.py` (Attention Rail, task 4193) | consumes read-only | best-effort pid → slug resolution, to name a dead holder in its notice; no heartbeat is read | **this PRD** (reader); that module is not edited | landed |
| `plans/scheduler-pause-halt-retirement-prd.md` (4686–4694, 5313, pending) | consumes | `Scheduler.is_paused` as "is dispatch halted"; the shape precedent for owner + expiry per hold class. Intentional divergence: an expired operator halt escalates and never resumes, while an expired session hold releases, because its owner is dead and leaving the work parked is the failure being fixed | that PRD owns the pause predicate. If it makes pauses class-scoped, `deferral_sweep` keeps asking the single question "is dispatch halted?" | pending |
| `plans/stranding-remediation-scheduler-ergonomics-prd.md` β (5311, pending) | none | the sweep never re-pends into a paused scheduler, so it emits no `repend_while_paused` and depends on nothing there | 5311 | pending |
| `docs/prds/milestone-tasks.md` | consumes convention | ISO-UTC wall-clock parsing for `expires_at`. Pure-time gates are steered to `pending` + `metadata.milestone {mode:'dated'}`. `_milestone_time_gated` is not touched | **this PRD** | landed |
| `deferred_watch`/`trigger_met` (task 2234) | sits beside | a human-trigger dispatch gate on *pending* tasks, unchanged. Task 2217's text seeds its `until_condition` record | n/a | landed |
| `docs/task-authoring.md` §8 task-metadata vocabulary | produces | documentation of the `deferral` key: a registered sub-model, not Tier-A blessed (Milestone precedent) | **this PRD** (β) | — |
| `plans/task-metadata-lookup-prd.md` (6385, pending) | optional consumer | once it lands, `find_tasks_by_metadata(key='deferral', …)` will find holds; the census does not wait for it | that PRD | pending |
| Dashboard status review 2026-10-06 (artifact `2YipixJ2gEUCMoKQxfa8p9`) | produces | the `deferral` field and `needs_human`, which a later lanes redesign will draw | **this PRD** draws the minimal "parked: …" rendering; the redesign is a separate follow-up | proposal |

There is no reciprocal ambiguity: every row names one owner, and no other PRD claims the
deferral record or the sweep.

## 9. Decomposition plan

There are six tasks:
- α, β and ε are intermediates that unblock others.
- γ and δ are leaves.
- ζ is the human gate that closes the batch.

Sizes follow the overlay's bands (`.claude/skills/prd/project.md`). No two tasks edit the
same file.

- **α — Deferral record and process identity (shared).** [medium; normal; ~500 LOC; 5 files]
  - Files:
    - `shared/src/shared/task_deferral.py` (new): `DeferralKind`, `CALLER_KINDS`, the request
      and stored models (RootModel-wrapped union), `Liveness`, `lapse_cause`, `needs_human`,
      constants, sub-model registration;
    - `shared/src/shared/process_identity.py` (new);
    - tests for both (`shared/tests/test_task_deferral.py`, `shared/tests/test_process_identity.py`),
      using real processes (spawn, capture, kill, re-check), including boundary row 12;
    - the registration import, wherever the registry needs it loaded.
  - **Unlocks** β, γ, δ and ε, which import it.

- **β — Server enforcement: gate, arguments, re-stamp, commit scope, clear-on-exit, floor,
  writers.** [high; normal; ~1,200–1,500 LOC; 13 files] depends on α. Intermediate: it
  unlocks γ, δ and ε, and its live check proves the producer.
  - Code:
    - `fused-memory/src/fused_memory/middleware/deferral_gate.py` (new: `plan()`);
    - `task_interceptor.py` (one `plan()` call and its application, the planning-birth stamp,
      the `submit_task` refusal, event fields);
    - `server/tools.py` (`deferral` and `expected_stamped_at` on `set_task_status`;
      `deferral` and `agent_id` on `commit_planning`; docstrings);
    - `backends/sqlite_task_backend.py` (the floor with echo passthrough, key removal on exit,
      the self-heal's drop);
    - `backends/task_backend_errors.py`;
    - `fused-memory/tests/test_deferral_gate.py` (boundary rows 1–11).
  - Every repo-tracked writer:
    - `docs/task-authoring.md`: the §2 table, the §8 key, and a §9 "Deferring a task" recipe
      with the three not-a-deferral shapes;
    - `skills/unblock/SKILL.md`: the hand hold becomes `held_by_session`, and "no sweep moves
      it back" is replaced by the lease rule;
    - `skills/do/SKILL.md`: the hand-carry guard, including the re-stamp to
      `carried_by(<carrier>)` when the merge moves to a carrier;
    - `skills/prd/references/decompose-mode.md`: births are stamped; a task held out of the
      commit gets a real record;
    - `skills/_shared/filing-the-trigger-chain.md`: gates left deferred are re-stamped
      `until_condition`;
    - the steward prompt in `orchestrator/src/orchestrator/agents/roles.py`: one sentence
      saying a steward deferral needs `until_condition`.
  - β re-greps `skills/` and `docs/` for any other instruction that sets `deferred` and lists
    them in its commit. If that would cross 15 files, it files the rest as a follow-up rather
    than widening.
  - **Live check:** fused-memory restarts onto β. Then an MCP `set_task_status` to `deferred`
    on a scratch task without `deferral` returns `deferral_required`, and `get_task` shows the
    task unchanged. The same call with `deferral={"kind":"until_condition","condition":"…"}`
    stores the record. A flip to `pending` removes it. The scratch task carries
    `metadata.milestone {mode:'dated', at:'2099-01-01T00:00:00Z'}` while pending, so the
    scheduler cannot dispatch it, and it is cancelled at the end. Boundary rows 1–11 are green.

- **γ — Expiry sweep, client mapping, de-flake scope (orchestrator).** [high; normal;
  ~1,000–1,300 LOC; 11 files] depends on α and β.
  - Files:
    - `orchestrator/src/orchestrator/deferral_sweep.py` (new: liveness read,
      `decide`, pass, sentinel notices, merge-queue check, streak);
    - `harness.py` (service registration and the notice filer);
    - `config.py` + `defaults.yaml` (three keys);
    - `scheduler.py` (`DeferralRejection`, `expected_stamped_at` forwarding);
    - `flake_ledger.py` (complete only `planning` owners);
    - `orchestrator/tests/test_deferral_sweep.py` (rows 13–21), and a flake-ledger test;
    - `docs/task-escalation-state-spec.md` (the `deferred` row: owner per kind, with the sweep
      as the only automatic exit);
    - `ARCHITECTURE.md` (state diagram and planning-mode prose);
    - `OPERATIONS.md` (the sweep, its config, its events, how the L2 watcher closes a holder
      notice).
  - **Signal (leaf):** on the live dark_factory orchestrator after restart:
    - A throwaway interactive session defers a scratch task `held_by_session` with
      `grace_secs=600`, then exits.
    - Within one pass, the L2 queue holds one INFO notice on a `deferral-holder:…` sentinel
      that names its pid and the task.
    - At most 12 minutes after exit, `get_task` shows the task `pending`, and the run's events
      hold `deferral_expired{cause:'holder_lost'}`.
    - A second scratch task, deferred `until_condition` with `expires_at` two minutes ahead, is
      `pending` within one pass of that time.
    - Both scratch tasks carry the 2099 dated milestone so they are not dispatched, and both
      are cancelled at the end.
    - Rows 13–21 are green.

- **ε — Census, migration script, check 6.** [medium; normal; ~700–1,000 LOC; 7 files] depends
  on α and β.
  - Files:
    - `scripts/deferral_census.py` (new);
    - `scripts/migrate_deferrals.py` (new; dry-run by default; repeatable `--project-root`);
    - `tests/scripts/test_deferral_census.py` and `tests/scripts/test_migrate_deferrals.py`
      (rows 22–23);
    - `skills/review-briefing/SKILL.md` (check 6 runs the census);
    - `review/briefing.yaml` (the invariant points at the mechanism instead of restating it,
      and the stale "1147/1853" sentence goes);
    - `scripts/sitting/ownership.py`: it reads `x_coalesced_into` as a live claim, so it reads
      `deferral.carrier_task_id` for deferred rows.
  - **Unlocks** δ (whose signal compares against the census) and ζ.
  - **Live check:** a dry-run of `scripts/migrate_deferrals.py` against every project with
    deferred rows prints a per-project class table. For each project, its `carried_by` and
    `until_condition` counts equal an independent read-only forensic count taken at the same
    moment: of `x_coalesced_into`, and of `x_armed_by`/`deferred_watch`, among deferred rows.
    `legacy_unknown` is the remainder. `deferral_census.py --check` exits non-zero and lists
    the unrecorded rows (pre-migration).

- **δ — Dashboard: draw the hold.** [medium; normal; ~400–700 LOC; ~6 files] depends on α, β
  and ε.
  - Files:
    - `dashboard/src/dashboard/data/active_tasks.py` (`_build_task_row` adds a `deferral`
      summary and the `needs_human` reason, with liveness from `process_identity`);
    - `dashboard/src/dashboard/static/redux/tab_tasks.jsx` (rendering and the header count);
    - `data.js` (row-shape comment);
    - `styles.css`;
    - dashboard tests (row 24).
  - **Signal (leaf):** on the live dashboard, after β is live, with one scratch record of each
    caller kind written through the MCP (dispatch-proofed as in β):
    - the tasks tab shows each row's "parked: …" text and age;
    - an unmigrated legacy row reads "no deferral recorded";
    - the header's needs-human count equals the count `deferral_census.py` reports for the same
      project at the same moment.

- **ζ — Human gate: apply the migration and triage.** [medium; `execution_class='operational'`
  pure gate] depends on γ, δ and ε.
  - The operator runs `scripts/migrate_deferrals.py --apply` for each project with deferred
    rows (dark_factory, reify, know_live and autopilot_video at authoring), then
    `scripts/deferral_census.py --check`.
  - They then:
    - close or re-pend the closable `carried_by` rows (45 in dark_factory at authoring);
    - re-pend or re-carry the orphaned ones;
    - triage `legacy_unknown` rows into a real kind, `pending` or `cancelled` as time allows
      (the census keeps listing the rest);
    - update the two file memories that teach the old hold,
      `procedural_defer_a_pinned_task_to_guard_a_hand_carry.md` and
      `feedback_hand_carry_procedure.md`, which no dispatched agent can reach.
  - **Signal (leaf):** `deferral_census.py --check` exits 0 for every migrated project, its
    `legacy_unknown` count per project is reported, and the dashboard's needs-human count
    matches the census.

Dependencies: α → β; α, β → γ; α, β → ε; α, β, ε → δ; γ, δ, ε → ζ. γ and ε run in parallel
once β lands, and δ follows ε.

### Capability bindings (draft for the decompose manifest)

| Task | Capability | Evidence |
|---|---|---|
| β | refusal before mutation at the choke point | `_apply_status_transition` gate order (the step-2b precedent) |
| β | atomic record + status | `set_status_and_stamp_audit` via `audit_fields` |
| β | `update_task` cannot bypass | the `done_provenance` floor in `SqliteTaskBackend.update_task`, to clone |
| β | births stamped | `_submit_task_planning_mode` is the only `deferred` birth path (critic grep 2026-10-07) |
| β | holder corroboration | `/proc/<pid>/comm`, `/proc/<pid>/stat` and `boot_id` readable by the fused-memory unit (§7) |
| γ | the sweep runs while halted | `BackgroundService` registration in `harness.py` |
| γ | dead holder named in its notice | `resolve_session_slug_for_pid` (best-effort; pid is always present) |
| γ | a notice reaches the human queue without pinning the task | the sentinel born-at-L2 precedent (`_DIRTY_TREE_ESCALATION_SENTINEL`) |
| γ | the rejection reaches the client typed | `Scheduler.set_task_status` `error_code` branches |
| δ | real records to draw (INV-13) | β's producer is live, and the signal requires one record per caller kind written through the MCP |
| ε | counts checkable | read-only forensic query per `CLAUDE.md` §"Forensic reads of tasks.db" |

G7 walk (advisory at author time, against `docs/legibility/design-invariants.md`):
- INV-1: the contract is a schema plus a server guard, with its envelope in the tool docstring.
- INV-2: refusals and events carry `reason_code`, `field`, `value` and the cleared record.
- INV-3: the sweep's write is a compare-and-set at the choke point, plus the claimant and
  merge-in-flight guards, and the holder is corroborated at write time.
- INV-4: a failure-streak cap, one notice per holder by sentinel, and pass events only when
  non-zero.
- INV-5: one kind vocabulary and one `needs_human`, both in `task_deferral.py`.
- INV-6: `held_by_session` implies a live owner, the sweep is its reconciler, and the
  dashboard shows liveness.
- INV-7: decision 2's owner/bound table. A live `held_by_session` hold is bounded by its owner's
  life and drawn on the dashboard with its age; Leo ruled (2026-10-08) that it raises no notice.
- INV-8: bounded per-pass work in a background service.
- INV-9: `metadata.deferral` is the only home, `x_coalesced_into` becomes history, and the
  briefing points to the mechanism rather than restating it.
- INV-10: tests drive the real store and real processes.
- INV-11: a missing or invalid record is surfaced and fails `--check`, and is never read as
  "no hold".
- INV-12: `planning` is owned by `commit_planning`, and `legacy_unknown` by gate ζ.
- INV-13: "no deferral recorded" and "no producer has written" are distinct states.

No waiver.

## 10. Out of scope

- `merge-deferred` and the atomic-train hold.
- What the recovery, stranded, orphan-cancel and orphan-block sweeps may do to `deferred` rows
  (`harness.py` exclusions; recon `_sweep_cancel_orphan`/`_sweep_block_orphan`). Their exits
  clear the record through the choke point and are otherwise unchanged.
- Inventing flip conditions for existing rows: the migration stamps only what is recorded.
- The dashboard lanes redesign from the 2026-10-06 status review.
- Auto-closing `carried_by` rows when their carrier lands: closure is a human judgment.
- A `SessionEnd` hook to release holds on clean exit: the grace window covers it.
- Unifying `deferred_watch`/`trigger_met` with `until_condition`.
- Holds whose holder is on another host. These are surfaced, never flipped.
- Re-validating the population against Table A or dependency shape in `--check`. The census
  reads records; the briefing's "dependency-shaped gate" judgment stays a reviewer's.
- Any notice, flip or `needs_human` entry for a live holder, however long it holds (Leo,
  2026-10-08).
- Persisting `dead_since` across orchestrator restarts. A restart only lengthens grace
  (decision 6).

## 11. Open questions (tactical)

1. **Key removal on exit.** `set_status_and_stamp_audit` merges `audit_fields` and has no
   key-deletion semantics. β adds a narrow removal (for example a `clear_keys` parameter)
   rather than writing `null`.
2. **Census transport.** `post_mcp_tool_call` is the suggested transport. If
   `get_tasks(statuses=['deferred'])` proves too heavy on the live store, ε pages it.
3. **Registration loading.** Self-registering modules must be imported before
   `parse_metadata` runs. α follows whatever mechanism loads `deploy_state.py` and
   `capability_manifest.py` today.
4. **Default priority.** β is suggested high, because it closes the stranding source; the
   others medium. Decide at decompose.
5. **Merge-queue lookup.** γ uses whichever read of its own harness's merge queue names
   queued and verifying entries by task id. It considers the task's own id only; the carrier
   case is covered by the re-stamp to `carried_by`.
