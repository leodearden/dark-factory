# Deferral records: every `deferred` task says how it ends

**Status:** authored 2026-10-07 (`/team` + `/prd` author mode); decomposed 2026-10-08 into
tasks 6524 (α), 6525 (β), 6526 (γ), 6527 (δ), 6528 (ε) and 6529 (ζ), plus reify 8368. The decompose re-walk corrected premises that had drifted or were false
on main `fc55c9c7c8`. The corrections are made in place, and §12 records each one and why.
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
- When a `held_by_session` holder process exits, a human-queue notice names the dead session
  and every task it held. The notice is informational: nothing waits on it. Each task returns to
  `pending` once the sweep has seen the holder dead for `grace_secs` (default 7,200 s), at most
  `grace_secs` plus three sweep intervals after the exit.
- An `until_condition` deferral with `expires_at` returns to `pending` within two sweep
  intervals after `expires_at`.
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
| `held_by_session` | `holder_pid` (the caller's `$CLAUDE_PID`); optional `grace_secs` | the holding session, which lands or releases it | holder death ⇒ one informational notice (decision 7); death observed for `grace_secs` ⇒ flip to `pending`; while the holder lives, nothing (decision 6) | any caller whose pid is a live `claude` process on the fused-memory host |
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
  `deferred` write without a `deferral` on a `deferred` row stays today's no-op. A re-stamp
  is not a status change. It emits only `deferral_restamped` and returns before the
  interceptor's targeted-reconciliation step (`TaskInterceptor.STATUS_TRIGGERS` includes
  `deferred`). Without that, the migration's ~575 re-stamps would launch ~575 reconciliations.
- **Guarded re-stamp** (`restamp_only=True`, single id). The write is refused with
  `deferral_changed` unless the row is in `deferred` now and its record's `stamped_at` equals
  `expected_stamped_at`. An omitted `expected_stamped_at` means "expects no record". The
  migration uses it for every write. Without it, a human could re-pend a row between the
  migration's read and its write, and the write would be a fresh entry that silently re-parks
  released work (INV-3). With a status other than `deferred` it is refused
  `field_not_allowed`, with CSV ids `cas_needs_single_id`, and without `deferral`
  `deferral_required`.
- **Every successful write echoes the stored record** (`deferral` in the response). Skill text
  goes live when β merges, and the server only after its restart. An old server silently drops
  an unknown argument (FastMCP's argument model ignores extras). So every instruction β edits
  tells the caller to confirm the echo. If the echo is absent, the server predates β: the
  caller treats the hold as unrecorded and re-stamps it once the server restarts.
- **`commit_planning`** gains `agent_id`.
  - `commit_planning(target_status='pending'|'cancelled')` flips only rows whose record is
    `planning`, or absent (the pre-migration window). Any other kind is refused
    (`reason_code='not_planning_hold'`), so a human hold or a live hand-carry can never be
    released by a batch commit. `commit_planning` reaches the choke point through the CSV
    `TaskInterceptor.set_task_status`, so it passes `plan()` an internal commit-scope marker.
    The marker is not a parameter of the MCP `set_task_status`. A plain
    `set_task_status(status='pending')` out of `deferred` stays legal for every kind.
  - `commit_planning(target_status='deferred')` requires `deferral` (a caller kind) and
    forwards it as a re-stamp.
  - `orchestrator/src/orchestrator/flake_ledger.py` completes its owner only when the owner's
    record is `planning` or absent, the same scope as `commit_planning`. Otherwise it treats
    the owner as held.
- **`submit_task(planning_mode=True)`** stamps `{kind: 'planning'}` itself, attributed to the
  submitting `agent_id`. Any `submit_task` carrying a caller-supplied `metadata.deferral` is
  refused in `TaskInterceptor.submit_task` before the ticket/planning split. The backend's
  `add_task` stays neutral, because it cannot tell the interceptor from a caller.
- **`update_task(metadata=…)`** may not add, alter or remove `deferral`, in any metadata
  mode. A byte-equal echo of the stored record passes, so read-modify-write writers that
  return the blob unchanged keep working. The floor follows the `done_provenance` write floor
  in `SqliteTaskBackend.update_task`, with a new `DeferralWriteAuthorityError` in
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`. It is not a clone: the
  `done_provenance` floor refuses the key outright in `merge` and `additive` modes and lets an
  echo through only in `replace`. The echo rule here compares against the stored row in all
  three modes, so it runs inside the transaction. The same rule binds
  `SqliteTaskBackend.rewrite_audit_trail` (task 5771), the privileged whole-blob writer behind
  the recon-stage audit-trail rotation: it carries `deferral` through unchanged.
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
    outlive a crashed `claude`, and would otherwise pin the hold forever. A `/proc` that cannot
    be read for any other reason is refused with `holder_unverifiable`, never read as "not
    live". The reads run off the event loop (`asyncio.to_thread`) before the write lock is
    taken (INV-8).
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
  ticks differ, or the boot id differs. A **reboot is death**. "Gone" requires positive
  evidence: `ENOENT` on `/proc/<pid>` together with a self-check that this process can see
  other processes' `/proc` entries. Any other read failure (`EACCES`, a `hidepid` or
  `ProtectProc=invisible` mount) raises `ProcessIdentityUnreadable`. Callers map that to
  "hold and surface", never to `DEAD`. Otherwise a hardened unit would read every live holder
  as dead, and the sweep would release them all after grace.
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
  Config keys, restart-only like the other sweep intervals (`stranded_reconcile_*` and
  `main_tip_sweep_*` are not in `config.py::RELOADABLE_FIELDS`, and
  `Harness._build_lifecycle_registry` reads the enabled flag and the interval once, at
  registration):
  - `deferral_sweep_enabled` (default true);
  - `deferral_sweep_interval_secs` (default `task_deferral.DEFAULT_SWEEP_INTERVAL_SECS`, 60;
    bounded `0 < interval ≤ LAPSE_OVERDUE_AFTER_SECS / 2`, so a configured interval can never
    outrun the census's overdue threshold);
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
  | holder liveness unreadable (`ProcessIdentityUnreadable`) | `Hold`, surfaced as `liveness_unknown`; never a flip |

  The last row fails safe toward holding, as the milestone gate fails toward withholding.
- **Notices.** A notice is one level-2 escalation per holder. It is filed against a
  stable sentinel task id derived from the holder identity
  (`deferral-holder:<host>:<boot_id8>:<pid>:<start_ticks>`), following the
  `_DIRTY_TREE_ESCALATION_SENTINEL` precedent, with `agent_role='orchestrator-deferral-sweep'`
  (a harness sentinel role). The escalation id comes from `make_id` over one fixed key,
  `deferral-holder`, so no per-holder sequence files accumulate.
  - **Shape (Leo, 2026-10-09).** The notice is filed in-process at `level=2` with
    `severity='info'` and `category='deferral_holder_lost'`. Its summary says it is
    informational: nothing waits on it, and the release happens without anyone acting. The
    escalation watcher selects records by level and pushes `info` at default priority, not
    the urgent push every `critical`/`urgent` record gets
    (`escalation/src/escalation/watcher.py::_send_ntfy`).
  - **A documented exception.** `escalation/src/escalation/models.py` states that records
    are born at L2 when their severity is in `BORN_AT_L2_SEVERITIES` (`critical`, `urgent`). γ
    adds the exception to that docstring: informational harness-sentinel notices filed
    in-process at level 2. Checked at decompose: no reader breaks.
    - `escalation/src/escalation/pins.py` classifies `info` as non-pinning.
    - The MCP server's born-at-L2 gates are not on the in-process path.
    - Nothing validates the level/severity pair.
  - It names the holder (pid, best-effort slug), every task it holds with its worktree path and
    branch where one exists (so a human can `/resume` or salvage within grace), and what will
    happen ("released to `pending` at <t>" / "released").
  - Dedup is by sentinel id in any status, so one dead holder is reported once, across
    restarts. That read (`EscalationQueue.get_by_task(status=None)`) parses the whole archive,
    about 1.2 s at 3,495 records (measured 2026-10-08). So the sweep runs it off the event
    loop, once, when the service starts, to seed an in-memory set of notified holders. After
    that it reads only pending records.
  - Filing it against a sentinel rather than the re-pended task leaves that task's dispatch and
    pins untouched.
  - The L2 watcher closes it as it does other sentinel notices (`close_only`).
    `skills/escalation-watcher/SKILL.md` gains the row for `deferral_holder_lost`.
  - The notice is filed **before** the flip, so a crash between the two cannot lose it; the
    next pass finds the escalation and completes the flip.
  - `expires_at` lapses file nothing, because the human asked for that outcome. They emit only
    `deferral_expired` events.
- **Guarded write.** A flip calls `set_task_status(status='pending',
  expected_stamped_at=<record.stamped_at>)` with:
  - `agent_id='orchestrator-deferral-sweep'`;
  - `client_op_id=f'deferral-flip:{project_id}:{task_id}:{stamped_at}'`, so a transient retry
    replays the recorded outcome instead of reading its own success as a conflict.

  The sweep goes through `Scheduler.set_task_status`, which today fixes
  `agent_id=ORCHESTRATOR_MCP_IDENTITY` and has no `client_op_id`. So γ widens it with
  optional `agent_id`, `client_op_id` and `expected_stamped_at`, fixed before the retry loop
  so every retry resends the same key. If a flip's response lacks β's `deferral_cleared`
  field, the server predates β: the sweep stops flipping for that pass and surfaces it.

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
  After `deferral_flip_failure_streak` consecutive failures, the sweep files one born-at-L2
  escalation (`urgent`, the sweep's sentinel role) for that task and stops retrying it until
  its record changes. A pass event
  `deferral_sweep_pass {evaluated, flipped, would_flip, notified, surfaced, failed}` is emitted
  only when `flipped`, `would_flip`, `notified` or `failed` is non-zero, or when the set of
  surfaced task ids changed since the last pass. Without that edge trigger, every pass between
  γ going live and the migration (~575 unrecorded rows across five projects) would emit an
  event.
- **Per-pass cost (INV-8).** Each pass makes one `get_tasks` call, two small `/proc` reads per
  held row (off the event loop), one merge-queue lookup per lapsing row, at most one
  pending-only escalation read per dead holder, and awaited MCP writes. That was 335 rows at
  authoring.
- **Merge in flight** includes a coalesce train's members, read from the harness's in-process
  train state through one read-only accessor (γ exposes it if none exists; CLAUDE.md:
  `get_merge_queue` does not show a train's verify, task 5245). If that read raises or times
  out in a pass, the row is held for that pass. "Cannot be read" never means "no accessor
  exists"; that would hold every lapse forever.

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

Every `--apply` write is a `restamp_only` guarded re-stamp (decision 3) and must see the
record echoed. A response without the echo means the server predates the gate: the run
aborts with exit 2 before its next write, never printing a "stamped" table it did not write.
If β refuses a mapped `carried_by` or `until_condition` record (`deferral_invalid`, e.g.
`carrier_not_found`), the row falls back to a `legacy_unknown` re-stamp. Its `reason` carries
the source key, the refused value (truncated) and the `reason_code`, and the report lists
every fallback. A `deferral_changed` refusal is reported as "moved since read", not as a
failure.

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
- `until_condition` past `expires_at` by more than `LAPSE_OVERDUE_AFTER_SECS` (two default
  sweep intervals: the sweep is not running);
- `held_by_session` whose holder is dead (release pending), on another host, or whose
  liveness could not be read (`liveness_unknown`);
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
    `expires_at` lapse left unflipped for more than `LAPSE_OVERDUE_AFTER_SECS`.
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
skill files β edits live in this repo and reach every project: `~/.claude/skills/prd` is a
symlink into it, and `/unblock`, `/do` and `/review-briefing` reach sessions through
`~/.claude/commands/*.md` symlinks into the main checkout. The hint corrects any other
project-local prose that *transitions* a task into `deferred`. It cannot correct prose that
leaves a planning *birth* deferred, because births are never refused. That prose gets a
follow-up in its own repo: reify's `.claude/skills/audit` leaves Medium findings deferred as
triage proposals, so they would render as stale `planning` batches after 24 h (§12).

The live effect needs a fused-memory restart onto β's code. The orchestrator side (γ) needs
an orchestrator restart, and the dashboard (δ) a dashboard restart. No task in the batch
delivers a restart. fused-memory redeploys on its own staleness clock (8 h), while orchestrator
fleet deploys are paused (task 5020) and the dashboard has no staleness redeploy. So the
post-restart live checks belong to the human gate ζ, which orders the restarts (§9).

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
DEFAULT_SWEEP_INTERVAL_SECS = 60           # γ's config default reads it
LAPSE_OVERDUE_AFTER_SECS = 2 * DEFAULT_SWEEP_INTERVAL_SECS

# Liveness is defined in process_identity.py (§5.2) and imported here: task_deferral's
# stored model already imports ProcessIdentity, so defining it here would make the two
# modules import each other.
def lapse_cause(record, now, dead_since: datetime | None) -> LapseCause | None  # expired | holder_lost
def needs_human(record, now, carrier_status, liveness: Liveness | None) -> NeedsHumanReason | None
    # liveness None = not read or unreadable: held_by_session → liveness_unknown; other kinds ignore it
```

The predicates take liveness and times as arguments and do no I/O (heuristic 7). The
registry entry is the **stored** `Deferral` shape (wrapped in a `RootModel`), not the request.
`deferral` must be a known key in every process that parses metadata, not only in the four
consumers that import `task_deferral`. Otherwise `parse_metadata` reports `unknown_key` on
every deferred row in any other process. So `task_deferral.py` calls
`register_metadata_submodel('deferral', …)` at module level, and
`shared/src/shared/task_metadata.py` loads it with one bare side-effect import
(`import shared.task_deferral`) as its last statement, after the registry is defined. Never a
`from … import`: that would fail on a partially initialised module.

### 5.2 `shared/src/shared/process_identity.py`

```
class Liveness(StrEnum): ALIVE, DEAD, OTHER_HOST
class ProcessIdentityUnreadable(Exception)        # /proc could not be read; never "dead"
@dataclass(frozen=True)
class ProcessIdentity: pid: int; start_ticks: int; boot_id: str; host: str
# each function below takes keyword proc_root: Path = Path('/proc'), the seam rows 12 and 26 test through
def capture(pid: int) -> ProcessIdentity | None   # None: no such process (ENOENT); raises on unreadable
def command_name(pid: int) -> str | None          # /proc/<pid>/comm
def liveness(identity: ProcessIdentity) -> Liveness
    # OTHER_HOST iff host differs; DEAD iff same host and (boot_id differs, pid absent
    # with positive evidence, or start_ticks differ); else ALIVE; raises
    # ProcessIdentityUnreadable when /proc cannot be read (decision 6)
```

### 5.3 MCP surface (fused-memory)

```
set_task_status(id, status, project_root, tag=None, done_provenance=None,
                reopen_reason=None, claimant_run_id=…, heartbeat_at=…,
                agent_id=None, client_op_id=None,
                deferral: dict | None = None,              # NEW
                expected_stamped_at: str | None = None,    # NEW, single id only
                restamp_only: bool = False)                # NEW, single id only (decision 3)
commit_planning(project_root, task_ids, target_status='pending',
                deferral: dict | None = None,              # NEW, required iff target_status='deferred'
                agent_id: str | None = None)               # NEW
```

Refusals are dicts: nothing is persisted, and each is journalled with `success=False`.
A successful write that stores or clears a record echoes it in the response (`deferral` or
`deferral_cleared`).

| `error` | When | Typed fields |
|---|---|---|
| `deferral_required` | entering `deferred`, or `commit_planning` to `deferred`, without `deferral` | `task_id`, `from_status`, `accepted_kinds`, `alternatives`, `hint` |
| `deferral_invalid` | any validation failure | `task_id`, `reason_code`, `field`, `value`, `hint` (`reason_code` values below the table) |
| `deferral_changed` | `expected_stamped_at` does not match the current record's `stamped_at`, or the row has left `deferred`; with `restamp_only`, the row is not `deferred` or its record is not the expected one (none, when `expected_stamped_at` is omitted) | `task_id`, `expected_stamped_at`, `current_stamped_at`, `current_status` |
| `deferral_write_authority` | `update_task` or `submit_task` metadata adds, alters or removes `deferral` | `task_id` (if any), `path`, `hint` |

`reason_code` for `deferral_invalid` is one of: `unknown_kind`, `missing_field`,
`field_not_allowed`, `server_only_kind`, `holder_not_live`, `holder_not_session`,
`holder_unverifiable`,
`carrier_not_found`, `carrier_is_self`, `expires_at_not_future`, `grace_out_of_range`,
`text_too_long`, `status_not_deferred`, `not_planning_hold`, `cas_needs_single_id`.

Events: `task_status_changed` gains `deferral` (on entry) and `deferral_cleared` (on exit),
and a new event `deferral_restamped {task_id, old, new}` covers re-stamps.

### 5.4 Orchestrator

- **`orchestrator/src/orchestrator/scheduler.py`.**
  - `DeferralRejection(SetTaskStatusRejected)` carries `reason_code`, and
    `Scheduler.set_task_status` raises it for the four `deferral_*` codes.
  - `Scheduler.set_task_status` gains optional `agent_id`, `client_op_id` and
    `expected_stamped_at` (today it fixes `agent_id=ORCHESTRATOR_MCP_IDENTITY` and has no
    `client_op_id`), set once before its transient-retry loop. It returns the successful
    response instead of `None` (its Protocol stub changes with it), so the sweep can read the
    `deferral_cleared` echo.
- **`deferral_sweep.py`.** It contains `decide(...)` (pure) and `run_deferral_sweep_pass(...)`
  (the `BackgroundService` pass function). Events: `deferral_sweep_pass`, and
  `deferral_expired {task_id, kind, cause, record}`.
- **`flake_ledger.py`.** The de-flake owner is completed only when its record is `planning`
  or absent (decision 3). The ledger's liveness read today is status-only, and its
  `get_statuses` fallback cannot see a record, so the owner is treated as held when its
  record cannot be read.

## 6. Boundary-test sketch

Rows 1–21 and 25–28 run against a real SQLite task store in a temp `project_root`. They
drive the public tool and interceptor functions with real processes. Rows 22–24 run over
fixture rows: the scripts read only over MCP, through the `scripts/tests` fake-transport seam,
and the dashboard over its own fixture dicts. No row patches private names (Tests stance,
`docs/code-quality.md`). The owning task is in brackets.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Entry without a record [β] | task `pending` | `set_task_status(deferred)` → `deferral_required`; `get_task` still `pending`; journal row `success=False` |
| 2 | Each caller kind accepted [β] | `pending`, `in-progress` and `blocked` tasks; a live `claude`-named helper process | `deferred` with a valid record for each kind; `metadata.deferral` round-trips through the shared model; `stamped_by` = caller agent_id |
| 3 | Malformed record [β] | `until_condition` with `condition: "  "` | `deferral_invalid`, `reason_code='missing_field'`, `field='condition'` |
| 4 | Holder corroboration [β] | a live `sleep` (comm ≠ `claude`); an unused pid | `holder_not_session`; `holder_not_live` |
| 5 | Exit clears [β] | a deferred row | `→ pending`, `→ done (found_on_main)`, `→ cancelled` and `→ blocked` each leave no `metadata.deferral`; the event carries `deferral_cleared` |
| 6 | Re-stamp [β] | `planning`-born row | `set_task_status(deferred, deferral=until_condition)` replaces the record; `deferred_at` kept; `deferral_restamped` emitted, and no `task_status_changed` and no targeted reconciliation; a `deferred` write without `deferral` is still a no-op |
| 7 | Planning birth and commit scope [β] | `submit_task(planning_mode=True)`; a second task deferred `until_condition` | birth row has `kind='planning'`; `commit_planning(pending)` clears it; `commit_planning(pending)` on the second → `not_planning_hold`, row unchanged; `commit_planning(deferred)` without `deferral` → `deferral_required` |
| 8 | Floor [β] | deferred row | `update_task(metadata={'deferral': <altered>})` in merge, additive and replace modes → `deferral_write_authority`; replace without the key → refused; an echo of the stored record → accepted, record unchanged; `submit_task(metadata={'deferral': …})` → refused |
| 9 | CAS and replay [β] | deferred row re-stamped after a read | `set_task_status(pending, expected_stamped_at=<old>)` → `deferral_changed`; with the current value → `pending`; the same call replayed with its `client_op_id` returns the recorded success |
| 10 | Legacy shape [β] | a row inserted `deferred` with no record | `legacy_unknown` re-stamp accepted with `deferred_at=null`; the same kind on a recorded row → `server_only_kind` |
| 11 | Raw-SQL exit [β] | deferred row hit by the candidate-key self-heal | cancelled, no `metadata.deferral` |
| 12 | Liveness [α] | a spawned process; its identity with a forged old `boot_id`; with another `host` | `ALIVE`; `DEAD` (reboot is death); `OTHER_HOST` |
| 13 | Holder lost [γ] | `claude`-named helper process holds a row (`grace_secs=600`), then is killed; fake clock | first pass after death → one `deferral_holder_lost` notice (level 2, severity `info`), row still `deferred`; pass 601 s after first-seen-dead → row `pending`, `deferral_expired{cause:'holder_lost'}`; no second notice |
| 14 | Live holder is silent [γ] | live holder; fake clock advanced 7 days | `Hold` on every pass; no escalation filed; not in `needs_human`; never flipped |
| 15 | `expires_at` lapses [γ] | `until_condition`, `expires_at` in the past | one pass → `pending`; no escalation; `deferral_expired{cause:'expired'}` |
| 16 | Paused [γ] | row as in 13, scheduler paused | notice filed ("released on resume"); no flip; `would_flip=1`; first unpaused pass flips |
| 17 | Race [γ] | row as in 15; re-stamp between the sweep's read and its write | `deferral_changed`; row keeps the new record; counted as a benign skip, not a failure |
| 18 | Guards [γ] | row as in 15 with a fresh claimant; a lapsing `held_by_session` row whose task id is in the merge queue, one in a coalesce train, and one in neither | the first three held, the merge ones surfaced `merge_in_flight`; the one in neither IS flipped |
| 19 | Failure streak [γ] | server refusing writes for the row | after 5 passes, one born-at-L2 escalation (severity `urgent`, agent_role `orchestrator-deferral-sweep`); no retries until the record changes |
| 20 | Restart [γ] | holder dead, notice filed, sweep restarted | no duplicate notice (sentinel dedup); grace restarts from the new first-seen-dead |
| 21 | Client mapping and de-flake [γ] | stub server returning each `deferral_*` code; a de-flake owner deferred `until_condition` | `DeferralRejection` with the right `reason_code`; the ledger leaves the owner alone |
| 22 | Census [ε] | fixture: each kind, one unrecorded deferred row, one stray record on a `pending` row | `--check` exits non-zero naming both; after the fixes it exits 0 and lists `legacy_unknown`, closable and orphaned rows |
| 23 | Migration [ε] | fixture with `x_coalesced_into`, `x_armed_by`, `deferred_watch` and bare rows across two project roots | dry-run prints the mapping per project; `--apply` stamps it; a second `--apply` writes nothing |
| 24 | Dashboard [δ] | a fixture of row dicts like row 22's (each kind, one unrecorded, one stray) | each row renders its "parked: …" text and age; the unrecorded row reads "no deferral recorded"; a row without the field renders as today; the header count equals the number of rows for which `task_deferral.needs_human` is not `None` over the same dicts (the live census comparison is ζ step 5) |
| 25 | Guarded re-stamp [β] | an unrecorded deferred row; the same row re-pended between a read and a write | `restamp_only` write on the unrecorded row → accepted; on the re-pended row → `deferral_changed`, row stays `pending` |
| 26 | Unreadable `/proc` [α, β] | a process-identity read pointed at an unreadable proc root | `liveness` and `capture` raise `ProcessIdentityUnreadable`, never `DEAD`/`None`; a `held_by_session` entry is refused `holder_unverifiable` |
| 27 | Echo [β] | any successful entry, re-stamp and exit | the response carries `deferral` (entry, re-stamp) or `deferral_cleared` (exit) |
| 28 | Quiet passes [γ] | five unrecorded deferred rows, three passes, nothing changing | one `deferral_sweep_pass` event, not three; a sixth unrecorded row → one more |

## 7. Pre-conditions (G3): substrate verified on `5b645d822a`, re-verified on `fc55c9c7c8` (decompose, 2026-10-08)

| Capability | Evidence |
|---|---|
| Single status choke point; validators run before mutation | `task_interceptor.py::TaskInterceptor._apply_status_transition` (gate chain under `_write_lock`; `_validate_done_provenance` at step 2b; same-status guard with the `done → done` repair seam `_repair_done_provenance_same_status`) |
| Atomic status + metadata write | `sqlite_task_backend.py::SqliteTaskBackend.set_status_and_stamp_audit` (one `_txn`, shallow merge, `_write_status_and_verify`). There are no key-deletion semantics yet, so β adds the narrow removal |
| Write-authority floor to follow | `SqliteTaskBackend.update_task` `done_provenance` floor, `_assert_done_provenance_passthrough`, `task_backend_errors.py::DoneProvenanceWriteAuthorityError`. That floor refuses the key outright in `merge`/`additive` and passes an echo only in `replace`; the all-modes echo rule is new (decision 3) |
| A second whole-blob metadata writer | `SqliteTaskBackend.rewrite_audit_trail` (task 5771, landed after authoring), reached from `TaskInterceptor._bound_audit_trail`; it passes `done_provenance` through and must pass `deferral` through too |
| Only one birth path | `TaskInterceptor._submit_task_planning_mode` is the only `tm.add_task(status='deferred')` call; the curator and `task_knowledge_sync` births default to `pending` |
| `commit_planning` goes through the choke point | `tools.py::commit_planning` → `TaskInterceptor.set_task_status` (CSV); it has no `agent_id` today |
| Recon exits go through the choke point | `targeted.py::_sweep_cancel_orphan` and `_sweep_block_orphan` call `task_interceptor.set_task_status` |
| Typed sub-model registry | `shared/src/shared/task_metadata.py::register_metadata_submodel`; self-registering `deploy_state.py`, `capability_manifest.py` |
| A sweep host that survives a halt | `orchestrator/src/orchestrator/background_service.py::BackgroundService` (sleep-first loop: period = interval + pass time); `Harness._build_lifecycle_registry` registers `stranded-reconcile` and others; the paused branch says background services keep running |
| Sweep config tier | `orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` holds no sweep interval; the service reads its interval once at registration. The new keys are restart-only |
| Born-at-L2 shape, and the notice's exception to it | `escalation/src/escalation/models.py::BORN_AT_L2_SEVERITIES` = {`critical`, `urgent`}; `escalation/src/escalation/pins.py` classifies `info` as non-pinning; the in-process `EscalationQueue.submit` writes what it is given; `escalation/src/escalation/watcher.py::_send_ntfy` pushes `critical`/`urgent`/`blocking` at urgent priority |
| Pause predicate | `orchestrator/src/orchestrator/scheduler.py::Scheduler.is_paused` |
| Live-claimant predicate | `shared/src/shared/task_claimant.py::has_live_claimant(task, now, ttl)`, as used by the dispatch gate |
| Sentinel born-at-L2 escalation precedent | `harness.py::_DIRTY_TREE_ESCALATION_SENTINEL`; harness sentinel role prefix `orchestrator-` (`escalation/src/escalation/server.py::_HARNESS_SENTINEL_ROLE_PREFIXES`) |
| Naming a dead holder (best-effort) | `session_registry.py::resolve_session_slug_for_pid` (pid → slug via `~/.claude/fleet/sessions-by-pid`); `LEASE_HEARTBEAT_TTL` is the precedent the default `grace_secs` matches (not imported) |
| `$CLAUDE_PID` names the Claude CLI | measured 2026-10-07 in the authoring session: `/proc/$CLAUDE_PID/comm` = `claude`; `/proc/$CLAUDE_PID/stat` field 22 and `/proc/sys/kernel/random/boot_id` readable; `~/.claude/fleet/sessions-by-pid/$CLAUDE_PID` resolves to the session's slug; a `/team` subagent sees the same pid |
| The server, the sweep and the dashboard can read other processes' `/proc` | `systemctl --user show` on `fused-memory.service`, `orchestrator-dark-factory.service` and `dark-factory-dashboard.service`: `ProtectProc=default`, `ProcSubset=all` (re-measured 2026-10-08); the fused-memory transport is local (`127.0.0.1:8002`). This is configuration, not a guarantee, hence `ProcessIdentityUnreadable` (decision 6) |
| The heartbeat is per turn, not continuous | the authoring session's `record.json` mtime was 21 min old mid-turn while it worked continuously, which is why a live holder must never lapse (decision 6) |
| Client rejection mapping | `scheduler.py::SetTaskStatusRejected` and its subclasses; `Scheduler.set_task_status` `error_code` branches; transient retry. Its `agent_id` is fixed and it takes no `client_op_id` today (γ widens it) |
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

There are six tasks (decomposed 2026-10-08; ids in brackets):
- α, β, γ, δ and ε are code tasks. Each one's completion signal is its boundary rows, which its
  own agent can make green in its worktree.
- ε also carries a live check its own agent can run before β is live (read-only).
- ζ is the human gate that closes the batch. It is also the integration gate: every live check
  that needs a service restart onto this batch's code is ζ's, because no task in the batch
  delivers a restart (decision 12).

Sizes follow the overlay's bands (`.claude/skills/prd/project.md`). No two tasks edit the
same file; the dependency edges serialize any re-grep overlap.

- **α — Deferral record and process identity (shared).** [6524; high; normal; ~500 LOC; 5 files]
  - Files:
    - `shared/src/shared/task_deferral.py` (new): `DeferralKind`, `CALLER_KINDS`, the request
      and stored models (the stored shape registered, RootModel-wrapped), `lapse_cause`,
      `needs_human`, constants including `DEFAULT_SWEEP_INTERVAL_SECS` and
      `LAPSE_OVERDUE_AFTER_SECS`, sub-model registration;
    - `shared/src/shared/process_identity.py` (new): `Liveness`, `ProcessIdentity`,
      `ProcessIdentityUnreadable`, `capture`, `command_name`, `liveness`;
    - `shared/src/shared/task_metadata.py`: load the `deferral` registration so the key is
      known in every process that parses metadata (§5.1);
    - tests for both new modules (`shared/tests/test_task_deferral.py`,
      `shared/tests/test_process_identity.py`), using real processes (spawn, capture, kill,
      re-check), including boundary rows 12 and 26 (α half).
  - **Unlocks** β, γ, δ and ε, which import it.

- **β — Server enforcement: gate, arguments, re-stamp, commit scope, clear-on-exit, floor,
  writers.** [6525; high; normal; ~1,300–1,600 LOC; 13 files] depends on α. Intermediate: it
  unlocks γ, δ, ε and ζ.
  - Code:
    - `fused-memory/src/fused_memory/middleware/deferral_gate.py` (new: `plan()`);
    - `task_interceptor.py` (one `plan()` call and its application, the internal commit-scope
      marker, the planning-birth stamp, the `submit_task` refusal, the re-stamp's early return
      before targeted reconciliation, event fields, the response echo);
    - `server/tools.py` (`deferral`, `expected_stamped_at` and `restamp_only` on
      `set_task_status`; `deferral` and `agent_id` on `commit_planning`; docstrings);
    - `backends/sqlite_task_backend.py` (the floor with echo passthrough in all three modes,
      `rewrite_audit_trail` passthrough, key removal on exit, the self-heal's drop);
    - `backends/task_backend_errors.py`;
    - `fused-memory/tests/test_deferral_gate.py` (boundary rows 1–11, 25, 26 (β half), 27).
  - Every repo-tracked writer:
    - `docs/task-authoring.md`: the §2 table, the §8 key, and a "Deferring a task" recipe inside
      the existing §9 "Practical recipes" (no renumbering).
      The recipe is the prose home of the hold vocabulary. It covers the three
      not-a-deferral shapes and the hand-carry hold: `held_by_session` while a session
      carries the work, re-stamped `carried_by(<carrier>)` when the merge moves to a carrier
      task. It also covers confirming the response echo.
    - `skills/unblock/SKILL.md`: the hand hold becomes `held_by_session`, and "no sweep moves
      it back" is replaced by the lease rule;
    - `skills/prd/references/decompose-mode.md`: births are stamped; a task held out of the
      commit gets a real record;
    - `skills/_shared/filing-the-trigger-chain.md`, `docs/quality-findings-contract.md` and
      `skills/hotspot-survey/SKILL.md`: a chain left deferred past its filing session is
      re-stamped `until_condition`. Its later release becomes `set_task_status(status='pending')`,
      because `commit_planning` releases only `planning` holds;
    - the steward prompt in `orchestrator/src/orchestrator/agents/roles.py`: one new sentence
      saying a steward deferral needs `until_condition`. It goes live only on an orchestrator
      restart, so the `deferral_required` hint carries a worked call and stands alone.
  - β re-greps `skills/` and `docs/` for any other instruction that sets `deferred` and lists
    them in its commit. If that would cross 15 files, it files the rest as a follow-up rather
    than widening. `skills/do/SKILL.md` holds no hold text (its 52 lines never mention
    `deferred`), so it is not edited.
  - **Signal:** boundary rows 1–11 and 25–27 are green in the fused-memory suite. The live
    check after fused-memory restarts onto β is ζ's step 1.

- **γ — Expiry sweep, client mapping, de-flake scope (orchestrator).** [6526; medium; normal;
  ~1,100–1,400 LOC; 13 files] depends on α and β. Intermediate: it unlocks ζ.
  - Files:
    - `orchestrator/src/orchestrator/deferral_sweep.py` (new: liveness read, `decide`, pass,
      sentinel notices, merge-queue and train check, streak, edge-triggered pass event);
    - `harness.py` (service registration and the notice filer);
    - `config.py` (three restart-only keys);
    - `scheduler.py` (`DeferralRejection`; optional `agent_id`, `client_op_id` and
      `expected_stamped_at` on `Scheduler.set_task_status`, which returns the successful
      response);
    - `shared/src/shared/task_transitions.py` (the sweep joins the `(DEFERRED, PENDING)`
      pair's call-site anchor comment);
    - `flake_ledger.py` (complete only `planning` or unrecorded owners);
    - `orchestrator/tests/test_deferral_sweep.py` (rows 13–21, 28) and
      `orchestrator/tests/test_flake_ledger.py`;
    - `skills/escalation-watcher/SKILL.md` (the `close_only` row for `deferral_holder_lost`);
    - `escalation/src/escalation/models.py` (the documented exception for the level-2 `info`
      notice);
    - `docs/task-escalation-state-spec.md` (the `deferred` row: owner per kind, the sweep as the
      only automatic exit; a pointer to the `docs/task-authoring.md` recipe, not a copy);
    - `ARCHITECTURE.md` (state diagram and planning-mode prose);
    - `OPERATIONS.md` (the sweep, its config and events, how the L2 watcher closes a holder
      notice).
  - **Signal:** rows 13–21 and 28 are green in the orchestrator suite. The live check after the
    orchestrator restarts onto γ is ζ's step 4.

- **ε — Census, migration script, check 6.** [6528; medium; normal; ~700–1,000 LOC; 8 files]
  depends on α and β.
  - Files:
    - `scripts/deferral_census.py` (new);
    - `scripts/migrate_deferrals.py` (new; dry-run by default; repeatable `--project-root`;
      every write is a `restamp_only` guarded re-stamp);
    - `scripts/tests/test_deferral_census.py` and `scripts/tests/test_migrate_deferrals.py`
      (rows 22–23; `scripts/tests/` is where tests of `scripts/*.py` live);
    - `skills/review-briefing/SKILL.md` (check 6 runs the census);
    - `review/briefing.yaml` (the invariant points at the mechanism instead of restating it,
      and the stale "1147/1853" sentence goes);
    - `scripts/sitting/ownership.py` and `scripts/tests/test_sitting_ownership.py`: it reads
      `x_coalesced_into` as a live claim, so it reads `deferral.carrier_task_id` for deferred
      rows.
  - Both scripts enumerate projects from the `orchestrator-*.service` unit files under
    `~/.config/systemd/user` whose `ExecStart` carries `--config` (no D-Bus needed), rather
    than a fixed list; repeatable `--project-root` adds or overrides. solar-challenge had 2
    deferred rows on 2026-10-08 and is not among the four §2 names. A root given by hand with
    no unit is reported "no sweep: surfaced only". A project whose read fails makes the run
    exit non-zero and names it.
  - **Unlocks** δ (whose header count uses the same `needs_human`) and ζ.
  - **Signal (live, read-only, before β is live):** a dry-run of `scripts/migrate_deferrals.py`
    against every enumerated project prints a per-project class table. For each project, its
    `carried_by` and `until_condition` task-id **sets** equal those of an independent
    read-only forensic read taken at the same moment:
    - `carried_by`: deferred rows whose `x_coalesced_into` is non-null;
    - `until_condition`: deferred rows with a non-null `x_armed_by`, or a non-null
      `deferred_watch` with a `trigger`. Key presence is not enough: 2217 carries
      `x_armed_by: null`.

    Both sides exclude deferred rows that already carry `deferral`, so the check holds whether
    or not β is live yet. A mismatch counts only if it survives one re-run within a minute
    (the store moves).
    `deferral_census.py --check` exits non-zero and lists the unrecorded rows (pre-migration).
    Rows 22–23 are green.

- **δ — Dashboard: draw the hold.** [6527; medium; normal; ~400–700 LOC; ~7 files] depends on
  α, β and ε. Intermediate: it unlocks ζ.
  - Files:
    - `dashboard/src/dashboard/data/active_tasks.py`: `_build_task_row` emits a ready-to-render
      deferral summary (label text, state, age, `needs_human` reason) computed with
      `task_deferral` and `process_identity`. Carrier status comes from the snapshot's
      `status_map`, never a per-row `get_task`;
    - `dashboard/src/dashboard/static/redux/tab_tasks.jsx`: renders those fields with no kind
      switch (INV-5); a missing field (old backend) renders nothing new, and a `null` summary
      renders "no deferral recorded"; header count per project;
    - `data.js` (row-shape comment), `styles.css`, `index.html` (the `?v=` cache-buster bump
      the freshness guard requires);
    - dashboard tests (row 24): `dashboard/tests/test_active_tasks.py` and
      `dashboard/tests/test_tab_tasks_deferral.py`, over δ's own fixture dicts.
  - Liveness reads for `held_by_session` rows run once per snapshot, off the event loop.
  - **Signal:** row 24 is green. The live check after β is live and the dashboard restarts is
    ζ's step 5.

- **ζ — Human gate: restarts, live checks, migration, triage.** [6529; medium;
  `execution_class='operational'` pure gate] depends on γ, δ and ε (β and α transitively).
  The operator works this checklist in order:
  0. Confirm fused-memory restarted after β's merge (`systemctl --user show fused-memory.service
     -p ActiveEnterTimestamp`). It normally redeploys within 8 h; otherwise restart it per
     `OPERATIONS.md`.
  1. **β live check.** On a scratch task (recipe below): `set_task_status(status="deferred")`
     without `deferral` returns `deferral_required` and `get_task` shows it unchanged; with
     `deferral={"kind":"until_condition","condition":"β live check"}` the response echoes the
     record and `get_task` shows `metadata.deferral`; a flip to `pending` removes it.
  2. Re-run ε's dry-run, then `scripts/migrate_deferrals.py --apply` for every enumerated
     project, then `scripts/deferral_census.py --check`. Doing the migration before the
     orchestrator restart is preferred, not load-bearing: a sweep that meets unrecorded rows
     holds them. Run it when no planning batch is mid-decompose. A pre-β uncommitted birth is
     stamped `legacy_unknown`, and its later `commit_planning(pending)` fails
     `not_planning_hold`; release it with `set_task_status(status='pending')`.
  3. A drained fleet restart (`OPERATIONS.md` §"Fleet redeploy & watchdog", after the
     merge-queue check in `CLAUDE.md`), so γ is live on every project's orchestrator.
  4. **γ live check.** A throwaway session (`/spawn`) defers a scratch task `held_by_session`
     with its own `$CLAUDE_PID` and `grace_secs=600`, then exits. Within one pass, the L2 queue
     holds one `deferral_holder_lost` notice on a `deferral-holder:…` sentinel naming its pid
     and the task. At most `grace_secs` plus three sweep intervals (13 min) after the exit,
     `get_task` shows the task `pending`, and the run's events hold
     `deferral_expired{cause:'holder_lost'}`. A second scratch task deferred `until_condition`
     with `expires_at` two minutes ahead is `pending` within two sweep intervals of that time.
  5. Restart the dashboard (`scripts/restart-dashboard.sh`). **δ live check:** with one
     scratch record of each caller kind, the tasks tab shows each row's "parked: …" text and
     age, an unmigrated row, if any is left, reads "no deferral recorded", and the header's
     needs-human count for dark_factory equals `deferral_census.py`'s for the same project,
     both read on this host within the same minute. A mismatch counts only if a re-read
     repeats it.
  6. Close or re-pend the closable `carried_by` rows (45 in dark_factory at authoring), and
     re-pend or re-carry the orphaned ones. Triage `legacy_unknown` rows, de-flake owners
     first, into a real kind, `pending` or `cancelled`, as time allows.
  7. **Successor owner (INV-7).** For each project where `legacy_unknown` or other needs-human
     rows remain, file one successor pure gate before closing ζ. It names
     `/home/leo/src/dark-factory/scripts/deferral_census.py --project-root <root>` as its check
     and carries a `metadata.milestone {mode:'delayed', after_secs: 2592000}`, so the remainder
     keeps an owner and a 30-day bound.
  8. Update the two file memories that teach the old hold,
     `procedural_defer_a_pinned_task_to_guard_a_hand_carry.md` and
     `feedback_hand_carry_procedure.md`, and their two `MEMORY.md` index lines, to point at the
     `docs/task-authoring.md` recipe.
  9. Confirm the step-4 notice was a level-2 `info` record, delivered as a default-priority
     push (§12 item 2).
  10. Cancel every scratch task.

  **Scratch-task recipe.** `submit_task(planning_mode=True, task_kind='normal', priority='low',
  title='SCRATCH deferral live check <step> <date>', metadata={'milestone': {'mode':'dated',
  'at':'2099-01-01T00:00:00Z'}, 'x_scratch_for':'deferral-flip-condition-prd'})`, then
  `commit_planning` to `pending`. The dated milestone withholds it from dispatch
  (`Scheduler._milestone_time_gated`). Never pin it, give each a distinct title, and cancel it
  at the end.

  **Signal (leaf):** `deferral_census.py --check` exits 0 for every migrated project, each
  project's `legacy_unknown` count is reported, the step 1, 4 and 5 live checks passed, and
  every project with needs-human rows left has a successor gate.

Dependencies: α → β; α, β → γ; α, β → ε; α, β, ε → δ; γ, δ, ε → ζ. γ and ε run in parallel
once β lands, and δ follows ε. One out-of-batch follow-up: reify's audit-skill re-stamp
(reify 8368), which depends on β through `metadata.external_deps`.

### Capability bindings

The committed manifest is `plans/deferral-flip-condition-prd.capability-manifest.md`, with
its machine-readable twin `plans/deferral-flip-condition-prd.capability-manifest.yaml`. It
binds every task's capabilities to evidence on `fc55c9c7c8`. Its mechanical delivered checks
gate each producer's dependents.

G7 walk (advisory at author time, against `docs/legibility/design-invariants.md`):
- INV-1: the contract is a schema plus a server guard, with its envelope in the tool docstring.
- INV-2: refusals and events carry `reason_code`, `field`, `value` and the cleared record.
- INV-3: the sweep's write is a compare-and-set at the choke point, plus the claimant and
  merge-in-flight guards, and the holder is corroborated at write time.
- INV-4: a failure-streak cap, one notice per holder by sentinel, and edge-triggered pass
  events.
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

G7 re-walk at decompose (2026-10-08). An adversarial seat walked every task against every
invariant and found ten hits. Each is resolved in the task text and the decisions above;
none is waived:
- α, `no-silent-fail-soft`: an unreadable `/proc` raises `ProcessIdentityUnreadable`, never
  reads as `DEAD` (decision 6).
- β, `loop-thread-occupancy-bounded`: the `/proc` reads run off the event loop, before the
  write lock (decision 5).
- β, `no-silent-fail-soft`: the response echo, plus instructions that verify it, because an
  old server silently drops the unknown argument (decision 3).
- β, `one-fact-one-home`: `docs/task-authoring.md` §9 is the prose home of the hold
  vocabulary; other docs point at it.
- γ, `storm-escape-required`: edge-triggered pass events (decision 7).
- γ, `loop-thread-occupancy-bounded`: the archive-wide dedup scan runs once, off the loop, at
  service start (decision 7).
- γ, `contracts-machine-checked`: the watcher's `close_only` row for the new category, and a
  fixed `make_id` key.
- δ, `no-lockstep-duplication`: the backend emits a ready-to-render summary; the JSX has no kind
  switch.
- ε, `corroborate-before-acting`: every migration write is a `restamp_only` guarded re-stamp
  (decision 3).
- ζ, `holds-owned-and-bounded`: remaining needs-human rows get a successor gate with a 30-day
  bound (§9 ζ step 7).

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
4. **Default priority.** Decided at decompose: α and β high (α is β's only prerequisite, so
   a medium α would invert the critical path), γ, δ, ε and ζ medium.
5. **Merge-queue lookup.** γ uses whichever read of its own harness's merge queue names
   queued and verifying entries by task id. It considers the task's own id only; the carrier
   case is covered by the re-stamp to `carried_by`.

## 12. Decompose record (2026-10-08)

The decompose ran as a `/team`: an anchor re-walk (sonnet), an adversarial G6/G7 seat (opus),
a manifest seat (sonnet) and a fresh critic (opus). Main had moved from `5b645d822a` to
`fc55c9c7c8`. What changed in this document, and why:

1. **Live checks moved to ζ.** No task in the batch restarts a service. fused-memory redeploys
   on its own 8 h staleness clock, orchestrator fleet deploys are paused (task 5020), and the
   dashboard has no staleness redeploy. So a post-merge, post-restart check cannot be run by
   the agent that owns the task. β, γ and δ complete on their boundary rows, γ and δ became
   intermediates, and ζ became the integration gate that orders the restarts (decision 12,
   §9). ε's live check stays in ε, because it reads only and needs no restart.
2. **Notice severity.** The decompose first filed the notice as `urgent`, because born at L2
   means `critical` or `urgent` in the escalation package. That made every abandoned hold an
   urgent phone push. **Leo ruled `info` (2026-10-09):** a level-2 `info` record with a
   default-priority push, under a documented exception to the models' contract (decision 7).
   A reader check found nothing that breaks. 6526's and 6529's texts were updated the same
   day. The failure-streak escalation stays `urgent`.
3. **Sweep interval ownership.** "Two sweep intervals" had no owner upstream of α and ε (the
   manifest's two `producer-downstream` FAILs). α now owns `DEFAULT_SWEEP_INTERVAL_SECS` and
   `LAPSE_OVERDUE_AFTER_SECS`, and γ's config default reads the former (§5.1).
4. **Config tier.** The sweep keys are restart-only like their siblings. "Green-tier" was
   false (decision 7).
5. **Contract additions:** `restamp_only` and the response echo (decision 3, §5.3);
   `holder_unverifiable` and `ProcessIdentityUnreadable` (decisions 5 and 6); the internal
   commit-scope marker; the re-stamp's early return before targeted reconciliation; and
   `Scheduler.set_task_status`'s optional `agent_id`, `client_op_id` and `expected_stamped_at`
   (§5.4).
6. **Writers.** `rewrite_audit_trail` (task 5771) is a second whole-blob writer, so the floor
   binds it. The `update_task` echo rule is new behaviour in `merge`/`additive`, not a clone.
   `skills/do/SKILL.md` holds no hold text; the hand-carry recipe is new content in
   `docs/task-authoring.md` §9. Trigger-chain prose in three files leaves chains deferred
   across sessions; it re-stamps them and releases them with `set_task_status`.
7. **Placement.** `Liveness` lives in `process_identity.py` (an import cycle otherwise), and
   `task_metadata.py` loads the `deferral` registration so every process knows the key (§5.1).
8. **Paths and projects.** Script tests live in `scripts/tests/`. Scripts enumerate projects
   from the running orchestrator units: solar-challenge had 2 deferred rows. ε's live check
   compares id sets and counts values, not keys.
9. **Other repos.** reify's audit skill leaves planning births deferred as triage proposals.
   Births are never refused, so the hint cannot correct it; a reify follow-up re-stamps them
   (reify 8368, external dependency on β).
10. **Critic pass.** A fresh critic seat's findings were folded in. They covered:
    - the registration's import shape (§5.1);
    - `needs_human` taking `Liveness | None` → `liveness_unknown`;
    - the `proc_root` test seam;
    - `restamp_only`'s edge refusals;
    - the migration's echo abort and `legacy_unknown` fallback (decision 9);
    - `Scheduler.set_task_status` returning the response;
    - an explicit train-membership read, with row 18's flip case;
    - rows 22–24 over fixtures;
    - rows 13 and 19 and decision 2 aligned to the notice shape;
    - the sweep interval bounded by the overdue threshold;
    - project enumeration from unit files.
11. **Population** at decompose (read-only, 2026-10-08): dark_factory 334 deferred, 158 with
    `x_coalesced_into` (carriers: 113 `pending`, 45 `done`); reify 227; know_live 10;
    autopilot_video 1; solar-challenge 2.
