# Claimant invariant: enforce, detect, and repair `claimant_run_id` liveness

**Status**: active · authored 2026-08-21 · approach **B+H** (contract + two-way boundary tests)
**Verified against**: main @ `7cb0ef2e0cf732b4a39a14c8bdd99e89f440d084`, re-verified at
`13ee71b406` — the intervening two commits touch this file only, so there is **zero code drift**
between the two and every anchor below holds on both.
**Origin contract**: `plans/task-status-authority-prd.md` C4/D4 — this PRD owns the *enforcement* of
C4's clearing half; C4 remains the origin contract. (The tasks that built C4 — 2182, 2188, 2408 —
are **not named in that document**; the attribution comes from code comments at
`sqlite_task_backend.py:64`, `harness.py:8858` and `scheduler.py:4921`. Cite those, not the PRD, if
you need the provenance.)

## Goal

A task's `claimant_run_id` / `heartbeat_at` columns mean "a live orchestrator run owns this task".
Today that meaning is maintained by per-caller convention, and it has decayed: **104 rows carry a
claimant that owns nothing** — dark-factory 36 `pending` + 17 `done`; reify 39 `pending` + 11 `done`
+ 1 `cancelled`. Every one is stale, with **zero live pids** (dark-factory 53/53 dead pids; zero live
across reify's non-`in-progress` rows).

**The count is a timestamped measurement, not a constant** (re-measured 2026-08-21). It moves while
you read it: reify's non-`in-progress` count was observed falling 51 → 50 inside one 8-minute window,
and task 3996's independent 2026-08-20 dark-factory census (37 `pending`, 16 `done`) differs from
today's by exactly one row moving `pending` → `done`. **ζ must not hard-code it**, and no signal in
this PRD may assert a specific row count.

**Scope of "the fleet" is 2 of 7.** The other five orchestrator units measured **zero**
non-`in-progress` claimant rows and zero terminal-with-claimant rows (`autopilot-video` 651 rows,
`know-live` 600, `solar-challenge-platform` 168, `solar-challenge` 101, `pump-web-ui` 19). Arming the
detector fleet-wide while ζ repairs only two projects therefore does **not** violate D7.

After this PRD:

- Entering a terminal status clears the claimant **atomically with the status write**, at the
  fused-memory choke point, for **every caller that reaches the fused-memory write path** — which
  includes the orchestrator itself (chain verified below), every MCP caller, and reconciliation. The
  one writer it does **not** cover is the raw-SQL self-heal at `sqlite_task_backend.py:705`, carried
  as stated residual and covered by detection rather than prevention.
- The one reconcile path that re-pends a task without clearing does so explicitly, where it has
  already established death.
- `TaskGroundTruth` stops reporting a stale claimant as live on non-`in-progress` rows.
- A violation of the invariant is **alarmed as a logical error**, not silently absorbed.
- The existing rows are repaired by a corroborating, staleness-gated operator script.

## Background

### The defect is a missing invariant, not two sloppy call sites

C4 contracts the orchestrator to "stamp at dispatch, refresh, **clear at release**". The gap is that
several exits **never go through release**, so the clearing half of C4 is unenforced:

| Path | Clears? | Rows carrying a dead claimant |
|---|---|---|
| `_run_slot` finally (`harness.py:8873`, `finally:` at `:8804`) | yes | — |
| blocked→pending sweep `_phase_redispatch_stranded_blocked` (`scheduler.py:6430-6431`) | yes (explicit pre-clear) | — |
| `consume_redispatch_requests.py::_apply_repend` (`:604-620`) | yes (pre-clear, ABORTs on rejection) | — |
| `Harness._revert_in_progress_if_no_live_claimant` (`harness.py:6174`) | **no** | 75 `pending` |
| `Scheduler.mark_done` / `_finalise_recovery_done` (`scheduler.py:2898` / `workflow.py:13421`) | **no** | 28 `done` |
| `scripts/consume_redispatch_requests.py::_apply_close` (`:599`) | **no** | 1 `cancelled` |
| `sqlite_task_backend.py:705` raw-SQL `status='cancelled'` self-heal | **no** — bypasses the choke point entirely | unmeasured |

**Three of six compliant, plus a seventh writer that never reaches the choke point at all.**

Two corrections to the earlier reading of this inventory, both established first-hand and both worth
recording because the mistakes were instructive:

- **`Scheduler.release` does not clear anything.** Only **three** `set_task_claimant` call sites exist
  in the whole orchestrator: `scheduler.py:6430`, `harness.py:8873`, `workflow.py:2506`. The
  "`Scheduler.release` clears at release" reading came from C4's prose, not from the code.
- **`_apply_repend` was missed and is compliant** — it clears *before* the `pending` flip and aborts
  on a rejected clear, which is stricter than the scheduler's best-effort. It lives in the same file
  as the non-compliant `_apply_close`.

A convention propagated by imitation is what decays as paths multiply, and **the citations decayed
first**: three in-source comments (`scheduler.py:6307`, `scheduler.py:6422`,
`task_ground_truth.py:552`) still cite the slot-release clear at `harness.py:5693-5696` / `:5810`,
which actually lives at `harness.py:8873`. This PRD inherited that rot by copying its anchor block
from task 3996, **every one of whose anchors is stale on this HEAD** (`mark_done` 2464→2898,
`_finalise_recovery_done` 13322→13421, `_run_slot` 5310→8873). β, γ and δ each edit a file carrying
one of those rotted comments, so fixing them is nearly free and stops the next PRD inheriting them
the same way.

An additional violation was demonstrated live during authoring: a plain
`set_task_status(4028, 'done', …)` through the sanctioned MCP choke point carried a 22h-stale
claimant straight from `pending` into `done`. Task 4028 remains a live specimen of the invariant
violation and is ζ's named positive control.

### Why the readers cannot compensate

Three predicates disagree about the same row, because each was written for a different question:

| Predicate | Gate | On a stale-claimant non-`in-progress` row |
|---|---|---|
| `has_live_claimant` | TTL only, status-agnostic | not live (correct) |
| `is_stranded` | TTL **and** `status == 'in-progress'` | **False** → caller reads "live" |
| `is_stranded_blocked` | TTL **and** `status == 'blocked'` | **False** |

`TaskGroundTruth._resolve_live_claimant` asks "is anyone alive holding this row?" — a question with
no status content — but implements it with `is_stranded`, whose `in-progress` gate exists to answer a
different question. Consequence, self-documented at `task_ground_truth.py:547-561` and in the
`_RECOVERY` row-(g) note at `:831-837`: a `blocked` row with a 12-hour-stale claimant resolves
`live_claimant=True`, `classify_recovery` returns LEAVE, and row (g) `RE_FILE_ESCALATION` is
unreachable for it.

### This is INV-6's converse

`docs/legibility/design-invariants.md` INV-6 `status-matches-liveness` rules the forward direction —
*a status implying ownership is legal only while a live claimant exists*. The rot here is the dual: a
**claimant persisting after the status stopped implying ownership**. INV-6's own rule also *mandates*
writing the successor status "before the claim is released", which is exactly why the naive fix is
wrong (below).

### Current impact is legibility, not stalled work — measured

Verified first-hand, so the PRD is not sold on a false urgency:

- **Scheduling is not blocked** — but this is an *inference*, not the measured zero an earlier draft
  claimed. `has_live_claimant(row, now, ttl=300s)` returns **False** on real leaked rows (run
  directly against four of them, with a synthetic-fresh positive control returning `True`). The
  earlier "zero `dispatch refused: live claimant` lines against 168,458 journal lines" was **wrong on
  both halves**: the real 7-day fleet corpus is **989,331** lines (`orchestrator-dark-factory`
  557,007 + `orchestrator-reify` 432,324), and it contains **one** hit, not zero —
  `Task 6218 dispatch refused: live claimant 'run-277928f28748/6218-39c75d2c/pid=1805896'`
  (2026-08-18). The grep string is byte-identical to the emitter's format string
  (`scheduler.py:4941-4945`), so the search was sound and the zero was simply false. Inspecting that
  hit rescues the conclusion: `pid=1805896` is the pid of the emitting process itself, i.e. a **fresh
  self-claim inside the C3.1 teardown window** — the gate working as designed, not a leaked row
  blocking dispatch. **No leaked row has been observed to block a dispatch; that is a reasoned
  inference from one inspected hit, not a measured absence.**
- **Completion is not blocked** — demonstrated by the 4028 write above.
- **Dependency release is not blocked.** `_deps_satisfied` (`scheduler.py:4319-4328`) does not read
  the claimant. (It *could*: alongside `status_map: dict[str, str]` it also receives
  `tasks_by_id: dict[str, dict]` carrying full rows. It never consults the claimant columns, so the
  behaviour is as stated — but "cannot" was too strong.)

The cost today is triage confusion and a latent trap; the case for fixing it is preventing future
breakage of a field whose meaning several components already trust.

## Sketch of approach

Four mechanisms, one repair.

1. **Clear at the choke point, on entry to `TERMINAL` only.** In
   `TaskInterceptor._apply_status_transition`, when the target status is terminal and the caller did
   not explicitly supply a claimant, persist `NULL` for both columns in the *same* UPDATE as the
   status.
2. **Clear caller-side at the one reconcile path that has established death**
   (`harness.py:6174`), mirroring the idiom `scheduler.py:6428-6431` already gets right.
3. **Fix the reader asymmetry at the root, and give the contract an executable home** — add
   `is_stranded_any_status` to `shared/src/shared/task_claimant.py` and repoint
   `_resolve_live_claimant` at it. Note the liveness *core* is **already factored out**:
   `_claimant_liveness_stranded` (`task_claimant.py:63`) is the shared body of `is_stranded`,
   `is_stranded_blocked` and (negated) `has_live_claimant`, and all three already delegate to it.
   The only duplication is the verbatim `metadata.infra_hold` block copied at `:137-139` and
   `:180-182`, which the new predicate absorbs. The same module also gains the two predicates the
   contract needs (`violates_terminal_claimant_invariant`, `is_stale_nonterminal_claimant`) and the
   shared `DEFAULT_CLAIMANT_HEARTBEAT_TTL`, so ε and ζ consume one definition instead of minting
   their own.
4. **Detect violations on a recurring census** — a terminal row observed carrying a claimant is a
   logical error: log a structured fact per row and file one born-at-L2 escalation per episode,
   never raising. The detector runs as a **recurring invocation of the ζ census**, not as a hook in
   the reconcile sweep — see D8 for why the sweep cannot host it.
5. **Repair the existing rows** with a corroborating, staleness-gated operator script.

## Resolved design decisions

### D1 — The clear-set is exactly `TERMINAL`; no new constant

Every non-terminal status can legitimately coexist with a live workflow, so the clear-set collapses to
the existing `shared.task_statuses.TERMINAL` (`{done, cancelled}`). **Reuse `TERMINAL` directly**; do
not introduce a parallel `CLAIMANT_CLEARED_ON_ENTRY` frozenset. A second hand-maintained list of
statuses would have to stay in lock-step with `TERMINAL` forever — precisely INV-5
`no-lockstep-duplication`. It also resolves `review` and `infra-hold` automatically (both non-terminal
⇒ keep) rather than requiring a per-status decision that a future tenth status could silently miss.

### D2 — `pending` is NOT chokepoint-cleared (the naive fix is unsafe)

Two independent findings kill it:

- **C3.1 status-precedes-kill.** `harness.py:13752-13762` documents the ordering: step 4 writes
  `target_status` (`pending`) at `:13858`, step 5 *then* kills the workflow at `:13884`, polling
  `terminal_status_hard_cancel_polls` before hard-cancel. The row is *expected* to sit at `pending`
  while the original workflow runs and heartbeats. During that window the task-2408 dispatch gate is
  the only cross-process guard against dispatching into the dying workflow's worktree — and there is
  **no compare-and-swap** on the claim write (`sqlite_task_backend.py:1969-1980` has no
  `WHERE claimant_run_id IS NULL`), so that gate *is* the enforcement, not a redundancy.
- **Heartbeat cannot reconstitute it.** `_claimant_heartbeat_loop` (`workflow.py:2461-2508`)
  refreshes **only** `heartbeat_at` — `claimant_run_id` is "intentionally NOT passed on each tick".
  So a cleared claimant on a live task never comes back, and a heartbeat-starved false revert would
  leave `(pending, NULL, fresh)` → dispatched into a live worktree.

Both are the task-2588 un-claim class. `deferred` is excluded for the same reason: it is a
`WORKFLOW_PRESERVE` status meaning "leave this alone, human will sort it", set by interactive sessions
*while the workflow is still running* (`workflow.py:3798` only notices at the next checkpoint) — an
unbounded version of the same window.

### D3 — Two-tier predicate: invariant vs hygiene

Enforced invariant (alarmable): **a terminal row carrying any claimant, stale or fresh.**
Hygiene metric (repairable, not alarmable): **a non-terminal row carrying a *stale* claimant.**

Keeping these separate is what stops the alarm firing on every legitimate C3.1 teardown, and it is why
the repair script's success predicate is "zero **stale** violations", not "zero claimant-bearing rows".

### D4 — Include the reader fix, and accept its behaviour change

`_resolve_live_claimant` is repointed at `is_stranded_any_status`. This makes `_RECOVERY` row (g)
reachable: a stale-claimant `blocked` task now classifies `RE_FILE_ESCALATION` instead of LEAVE. That
is the follow-up the code itself invites ("left to a follow-up if this edge case proves to matter in
practice"). The existing pin
`test_stale_db_claimant_on_blocked_task_is_treated_as_live_by_design`
(`orchestrator/tests/test_task_ground_truth.py:783`) is **inverted, not deleted** — it must carry the
rationale for the reversal so the next reader sees a decision, not a regression.

`is_stranded_any_status` **must retain the `metadata.infra_hold` carve-out**. Dropping it (i.e. using
bare `has_live_claimant`) would let a legacy `in-progress` + `infra_hold` row with a stale heartbeat
resolve `live_claimant=None` → `_RECOVERY` row (c) `REVERT_TO_PENDING`, silently losing a protection
that `harness.py:6113` only re-checks for the *first-class* status.

**Calibration: the carve-out is defensive, not load-bearing.** Truthy `metadata.infra_hold` is
present on **0 of 4,545** dark-factory rows and **0 of 6,397** reify rows (positive control: 7 and 4
rows merely *contain* the substring), and `shared/task_claimant.py`'s own module docstring says the
metadata check "becomes permanently dead but harmless" post-omega4. Retain it — it is two lines and
costs nothing — but boundary test B9 pins a shape with zero live instances, and this PRD should not
claim the carve-out is protecting live traffic.

### D5 — Alarm shape: loud, deduped, never raising

The detector runs one census pass over every row (D8). A synchronous `assert` anywhere on a shared
path would propagate and collateral-strand siblings — the failure class the isolation guard at
`scheduler.py:6432-6453` (task 2849) exists to stop — so the alarm never raises. It:

- logs `claimant_invariant_violation:` at **ERROR**, once per violating row, with the structured
  fields (task id, status, claimant, heartbeat) — never a formatted prose blob;
- files **one** born-at-L2 escalation per episode under a durable sentinel, entirely inside
  `try/except Exception`, with an `escalation_queue is None` early return;
- **logs the swallowed failure at WARNING with `exc_info=True`** on both fail-soft paths, so an
  unwritable escalation store degrades loudly rather than to silence (INV-4);
- populates the record's `evidence` list with one capped entry per violating row, so the L2 is
  self-describing without a log scrape (INV-2);
- **auto-resolves** the sentinel after N consecutive clean passes, re-arming itself (INV-7).

**Record shape** — `severity='critical'`, `level=2`, `agent_role='orchestrator-deterministic'`,
`category='infra_issue'`, with the verbatim discriminator `claimant invariant violation` in the
`summary`. Three deliberate choices:

- **`category='infra_issue'`, not a new `invariant_violation` value.** `Escalation.category` is a
  free-form `str` with no submit-time validation (`escalation/models.py:238`), so a novel value
  *would* persist — but `models.py:243-247` carries a standing rule that **the next category
  addition must promote that vocabulary to an enum or a submit-time lint** (task 3709), an
  obligation this PRD does not budget for. The closest in-repo analogue —
  `TaskWorkflow._escalate_scope_invariant_violation` (`workflow.py:14183`), literally an
  invariant-violation escalation — reuses an existing category and discriminates on a load-bearing
  `summary` substring. ε copies that. It also keeps the PRD's "no novel substrate" claim true.
- **`agent_role='orchestrator-deterministic'` is load-bearing, not decoration.** It is in
  `L2_AUTO_CLOSE_DENY_ROLES` (`escalation/authority.py:94`). Without it the record matches the
  `stale_task_scoped` auto-close class, which is **category- and role-agnostic**
  (`authority.py:210-213`) and keys on evidence text matching `status\s*[=:]\s*(done|cancelled)`
  plus a task citation — precisely what ε's own evidence says. Since `get_pending_escalations` is
  pending-only, an auto-close would make ε's signal silently read zero.
- **Dedup uses `get_by_task(<sentinel>, status='pending', level=2)`, never `has_open_l1`** — the
  latter is hardcoded `level=1` (`escalation/queue.py:683`; `:671` is the `def`) and would never
  match, a trap already documented at `merge_queue.py:1100-1104`. A per-task id would file one
  critical per violating row on the first pass; the sentinel bounds it to one, with per-row detail
  in the ERROR log and the `evidence` list. The precedent is live and exact at
  `merge_queue.py:1118-1163`.

### D6 — Repair is a corroborating operator script, not a migration

Not a `_migrate_v4_to_v5` step — but **not for the reason an earlier draft gave**. That draft argued
that because `_migrate_v3_to_v4` is self-gating and deliberately leaves `user_version` at 3 when
residual duplicate `candidate_key`s remain (`sqlite_task_backend.py:68-73` — literally true), a
chained v5 step "would silently never run on a DB parked at 3". **That inference is false as coded**:
`_migrate` reads `PRAGMA user_version` exactly once (`:328`) and the `if version < 4:` branch
(`:406-411`) never advances the local variable, so an appended `if version < 5:` step in the same
style would see `version == 3` and **run**. The argument is withdrawn.

The decision stands on the writer-semantics ground alone, which is sufficient: the repair must write
via `set_task_claimant` precisely because that writer does not bump `updated_at`, and a migration
step would have no such affordance.

The script writes via `set_task_claimant`, **never raw SQL** — the backend's claimant writer
(`sqlite_task_backend.py:2265-2268`) deliberately does not bump `updated_at`, and a hand-rolled UPDATE
would reset `updatedAt`-keyed staleness detectors (including
`consume_redispatch_requests.py:465`) for no benefit. It is `--dry-run` by default and gated on staleness **as well as** status. Four hardening
requirements, each from a first-hand finding:

1. **Per-row corroboration, not just aggregate.** The batch-level "affected rows ∩ current
   holders" pre-flight (INV-3 `corroborate-before-acting`) is necessary but **not sufficient**: a
   `done` row can be reopened to `in-progress` with a fresh claimant between census and apply, and
   ζ would then clear a *live* claimant — the task-2588 un-claim class D2 goes to lengths to avoid
   elsewhere in this same PRD. ζ **re-reads each row immediately before its own
   `set_task_claimant`** and skips any row whose status or claimant changed since the census.
   The holder set is read via `get_scheduler_state` / `read_scheduler_state`
   (`fused_memory.mcp_tools.scheduler_state`), whose on-disk field is `current_holders` in
   `data/orchestrator/scheduler_state.json` — note there is **no symbol named "lane-holder"** in the
   tree; that spelling appears only in this PRD's prose.
2. **Import the TTL, never mint one.** Five hand-maintained copies of the 10-minute staleness
   window already exist (`task_ground_truth.py:274`, `harness.py:248`,
   `dashboard/data/tasks.py:283`, `live_workflow_detector.py:270`, `artifacts.py:1339` as `600.0`),
   and this PRD's own authoring measurement used a sixth value (300 s). ζ imports α's exported
   `DEFAULT_CLAIMANT_HEARTBEAT_TTL` (INV-5 `no-lockstep-duplication`).
3. **Structured output.** `--json` emits a machine-readable census
   (`{status, tier, task_id, claimant_run_id, heartbeat_at, measured_at}` per row plus totals)
   alongside the human print, per the house pattern at
   `scripts/repair_wiped_metadata_files.py:1120` and `scripts/audit_combine_gate_marker_loss.py:1162`
   (INV-2 `structured-facts-at-failure`). Without it, "zero stale violations" is a number a consumer
   must recover by parsing prose.
4. **Handle the un-corroboratable row.** reify task **5225** carries
   `claimant_run_id='agent-esc-5053-2-docs-fix'` — freeform prose with **no `pid=`** — and
   `heartbeat_at=NULL`. It is this PRD's one `cancelled` row, and it has neither a heartbeat to age
   nor a pid to probe. ζ must not fail closed on it: `_claimant_liveness_stranded` already treats a
   missing heartbeat as stranded, so routing through the shared predicate resolves it — but ζ's spec
   must say so explicitly, or the alarm fires on that row forever and D7 is violated for exactly the
   row this PRD counts. (That row also falsifies `tools.py:938-946`, which asserts a claimant is
   "always machine-composed by `compose_claimant_run_id()` … never freeform text" — the same
   stale-comment class β already corrects one screen below at `:1406-1407`. β corrects both.)

### D7 — Repair must precede the alarm

`task_interceptor.py:1008` short-circuits a same-status write as a no-op (returning
`{'success': True, 'no_op': True, …}` at `:1018`), so **β's coercion never runs on a row that is
already terminal** — a `done` row carrying a claimant cannot be healed by re-writing `done`.

The stronger claim an earlier draft made — that *no code path can heal an existing row* — is **false**
and is withdrawn. `TaskInterceptor.set_task_claimant` (`:1341`) is a thin delegate to
`SqliteTaskBackend.set_task_claimant` (`:2210`) with **no status gate whatsoever**, and it is exposed
as an MCP tool (`tools.py:7710`); passing `claimant_run_id=None, heartbeat_at=None` heals a terminal
row today. ζ's whole design depends on that being true, and task 3996 records Stage 2 having already
used it once by hand.

The correct statement is the one the ordering actually needs: **nothing heals an existing row
_automatically_.** Healing requires the deliberate write ζ performs, so enabling detection before the
repair runs would alarm on legacy residue that no running code will ever clear. The DAG enforces the
ordering.

### D8 — Detection is a recurring census, not a hook in the reconcile sweep

The obvious site for the alarm — inside `_resolve_live_claimant` or `derive_truth`, where the row is
already fetched — **cannot observe a single violation.** Established first-hand:

- `_RECONCILE_SWEEP_STATUSES = frozenset({'in-progress', 'blocked'})` (`harness.py:238`), enforced by
  `if status not in _RECONCILE_SWEEP_STATUSES: continue` (`harness.py:4946-4947`). Its own comment
  reads *"Intentionally EXCLUDES: 'done' / 'cancelled' — terminal-by-decision; nothing to recover"*.
- `TaskGroundTruth.recovery_for` → `derive_truth` → `_resolve_live_claimant` has exactly **one**
  production call site: `harness.py:5540`, inside `_reconcile_one_stranded`, downstream of that
  filter.
- C4-E1 is defined over `TERMINAL = {done, cancelled}` — **precisely the set the sweep excludes.**

So both candidate placements were the same blind code path, and an alarm there would fire **zero**
times, not "immediately on legacy residue". Three consequences worth recording, because each was an
argument this PRD previously made on a false premise: D5's alarm-shape rationale described a site
that never sees the violation; D7's "would fire the L2 immediately" was inverted; and the
`loop-thread-occupancy-bounded` flag raised against that site is moot once the site changes.

**Rejected alternatives.**

- *Widen `_RECONCILE_SWEEP_STATUSES`.* Rejected: the set is load-bearing (it is what stops the sweep
  trying to "recover" completed work) and is pinned by `test_reconcile_stranded.py:4196` and
  `test_repend_state_machine.py:681`.
- *Re-home the alarm to the interceptor.* Rejected: post-β the only in-band way a terminal row
  acquires a claimant there is an explicit caller-supplied claimant, which C4-E2 deliberately
  honours — and it would be **blind to `sqlite_task_backend.py:705`**, the raw-SQL self-heal that
  bypasses the choke point entirely and is the one remaining minter after β.

**The census wins on three counts.** It is the only vantage point matching C4-E1's universal
quantifier (every row, regardless of status); it is the only one that catches the `:705` bypass; and
it moves the escalation `submit` — a flock plus a durable temp-fd fsync plus a directory fsync
(`escalation/queue.py:392-412`) — **off the orchestrator's asyncio event-loop thread**, discharging
INV-8 rather than arguing about it.

The cost, stated plainly: **detection is no longer real-time.** Latency equals the census cadence.
That is acceptable because the violation is a data-integrity defect with no live consequence (see
*Current impact*), not an outage — and because ε alarms on a condition that, post-β, only a rare
raw-SQL path can create.

## Contract (B+H)

**Invariant C4-E1.** For any task row: `status ∈ TERMINAL ⇒ claimant_run_id IS NULL AND heartbeat_at
IS NULL`.

It ships as an **executable predicate, not prose**: α exports
`violates_terminal_claimant_invariant(task)` from `shared/src/shared/task_claimant.py`, and β's
tests, ε's detector and ζ's census all call it rather than re-expressing it. This is INV-1
`contracts-machine-checked`, and it is load-bearing rather than tidy: without a single definition,
ε and ζ would each hand-maintain their own copy of the invariant *plus* D3's two-tier
invariant/hygiene split, and those copies must agree byte-for-byte or the alarm and the repair
disagree about what counts as a violation. D3's hygiene tier gets the same treatment
(`is_stale_nonterminal_claimant(task, now, ttl)`).

**C4-E2 (write rule).** `TaskInterceptor._apply_status_transition` is the sole enforcing choke point.
When `status ∈ TERMINAL` and the caller did not explicitly supply `claimant_run_id`, both columns are
written `NULL` in the same UPDATE as the status column. An explicitly supplied claimant is honoured
(the caller is asserting a deliberate exception and owns it).

**C4-E3 (non-terminal rule).** For `status ∉ TERMINAL`, the claimant columns are untouched by the
status write. Clearing on those transitions is caller-side and legal **only** where the caller has
already established that no live claimant exists.

**C4-E4 (fail-safe).** The rule inherits the task-2182 columns-absent path
(`sqlite_task_backend.py:1970`): on a pre-migration connection the status write succeeds, the claimant
write is skipped, and a warning is logged. Enforcement must never make a status write fail.

**C4-E5 (ordering).** The coercion sits *after* the no-op short-circuit
(`task_interceptor.py:1008`), so an idempotent same-status write stays a true no-op and does not
become a real UPDATE.

**C4-E6 (read rule).** "Is anyone alive holding this row?" is answered by `is_stranded_any_status` —
TTL-based, status-agnostic, `metadata.infra_hold`-respecting. `is_stranded` and
`is_stranded_blocked` are its status-gated specialisations and must delegate, not duplicate.

**C4-E7 (violation).** An observed violation of C4-E1 is a logical error: alarmed loudly and
structurally, never raised, never silently absorbed. Observation happens on the **recurring census**
(D8), which is the only vantage point that sees every row regardless of status — and therefore the
only one that can also catch the `sqlite_task_backend.py:705` raw-SQL writer that bypasses the choke
point. A filing failure is itself logged at WARNING with `exc_info=True`, so the fail-soft path
cannot degrade to silence (INV-4 `storm-escape-required`).

## Boundary-test sketch (B+H)

Facing both the producer (interceptor) and consumer (reader) sides of the seam.

| # | Side | Scenario | Preconditions | Postconditions |
|---|---|---|---|---|
| B1 | producer | terminal write clears | row `in-progress` with a claimant | `set_task_status(id,'done')` → `get_task` shows both columns `null` |
| B2 | producer | non-terminal write preserves | row `in-progress` with a claimant | `set_task_status(id,'blocked'\|'deferred'\|'review'\|'infra-hold'\|'merge-deferred')` → claimant unchanged |
| B3 | producer | explicit supply wins | any row | `set_task_status(id,'done',claimant_run_id='run/s/pid=1')` → the string persists, not `NULL` |
| B4 | producer | same-status write stays a no-op | row `done` with a leaked claimant | `set_task_status(id,'done')` → `{'no_op': True}`, **zero** backend writes, claimant unchanged (pins D7's necessity) |
| B5 | producer | fail-safe | pre-migration connection lacking the columns | terminal write succeeds, status persisted, warning logged, no error |
| B6 | producer | CSV write clears every id | three claimed rows | `set_task_status('a,b,c','cancelled')` → all three cleared |
| B7 | consumer | stale claimant on `blocked` is no longer live | `blocked`, claimant stale past TTL | `_resolve_live_claimant` → `None`; `classify_recovery` → `RE_FILE_ESCALATION` (row g) |
| B8 | consumer | fresh claimant on `blocked` is still live | `blocked`, claimant fresh | `_resolve_live_claimant` → `Claimant(DB)`; classify → LEAVE (the teardown window keeps working) |
| B9 | consumer | infra_hold carve-out survives | `in-progress`, `metadata.infra_hold`, stale heartbeat | `is_stranded_any_status` → `False`; **not** `REVERT_TO_PENDING` |
| B10 | consumer | C3.1 window is not broken | `pending` + fresh claimant (live workflow mid-teardown) | `_eligible_for_dispatch` still refuses dispatch |
| B11 | alarm | violation alarms once | two terminal rows carrying claimants in one census pass | one ERROR per row; exactly **one** L2 escalation filed |
| B12 | alarm | alarm never crashes the census | escalation queue raising on `submit` | the census still completes and still reports both rows; the filing failure is logged at WARNING with a traceback (mutation-tested) |
| B13 | producer | δ clears **then** flips | `in-progress` row whose claimant is dead | after the sweep, `get_task` shows `claimant_run_id: null` **and** `status: pending` |
| B14 | producer | δ's crash window is backstopped | fault injected between δ's clear and its flip | row is left `(in-progress, NULL)` — never `(pending, stale claimant)` — and the next sweep still reverts it |
| B15 | repair | ζ refuses a row that changed under it | census sees `done`+claimant; row is reopened to `in-progress` with a fresh claimant before apply | ζ's per-row re-read skips it; the live claimant survives |
| B16 | repair | ζ handles a claimant with no heartbeat | row `cancelled`, claimant freeform prose, `heartbeat_at IS NULL` (reify 5225's shape) | classified stale via the shared predicate and repaired — not skipped as un-corroboratable |

## Pre-conditions for activating

None external. Every substrate capability is present on main @ `7cb0ef2e0c` (G3, verified):

| Capability | Evidence |
|---|---|
| `_CLAIMANT_WIRE_UNSET` sentinel distinguishing unsupplied from explicit-null | `fused-memory/src/fused_memory/server/tools.py:946`, defaults at `:7603-7604`, `:7714-7715` |
| Single enforcing funnel for both status writers | `middleware/task_interceptor.py:871` `_apply_status_transition`, reached from `:833-848` and `:857-867` |
| Sole **status-write** SQL emitter for the columns | `backends/sqlite_task_backend.py:1922` `_write_status_and_verify`, tri-state block `:1969-1981`. (Not the *only* emitter: `set_task_claimant` at `:2210` writes them too — D6 depends on that — and the raw-SQL self-heal at `:705` writes `status` without them.) |
| Columns-absent fail-safe | `sqlite_task_backend.py:1970`, pinned by `tests/test_sqlite_task_backend.py:682` |
| `TERMINAL` frozenset | `shared/src/shared/task_statuses.py:61` |
| `set_task_claimant` writer that does not bump `updated_at` | `sqlite_task_backend.py:2265-2268` |
| Born-at-L2 escalation idiom from orchestrator code | `orchestrator/merge_queue.py:1140-1163`, `proc_supervision.py:205-226` |
| `_RECOVERY` row (g) `RE_FILE_ESCALATION` exists to become reachable | `task_ground_truth.py:838`; keyed on the 5-tuple `(BLOCKED, no-open-escalation, GONE_NO_MARKER, False, None)` at `:838-839` — γ's demo must satisfy all five, and `_RECOVERY` is a dict consumed by exact-key lookup (`:916`), so no row can shadow another |
| `release_workflow` refuses to park an infra-hold row (production site) | `escalation/server.py:2678` — cited alongside `escalation/tests/test_release_workflow.py:320`, since a test docstring is weak evidence for a load-bearing claim |
| `escalation_queue` handle already plumbed into `TaskGroundTruth` | `task_ground_truth.py:364`, `:372` — ε needs no new wiring for its `None` early-return |

No novel substrate is introduced.

### Status-producer audit (G3, 2026-08-21)

D1 reduces the clear-set to `TERMINAL`, so every other status is *keep* by construction and no
per-status ruling is load-bearing. The audit was run anyway, because an earlier writer-inventory had
already been caught wrong about `deferred`, and it found the same error twice more:

| Status | Verdict | Evidence |
|---|---|---|
| `infra-hold` | **PRODUCED** — the "no producer" claim is false | `workflow.py:7385` `_mark_blocked(..., block_status='infra-hold')` landing at `workflow.py:14575`; the literal never appears at a `set_task_status` call site because it travels as a parameter (default `'blocked'` at `workflow.py:14485`). 15 journal writes, all `project_id='reify'`, 2026-07-18 → 2026-08-04 |
| `review` | **PRODUCED-BUT-DORMANT** — no code writer, but MCP-reachable and used | Literal-arg histogram over all non-test src: 23 `blocked`, 10 `pending`, 7 `done`, 3 `in-progress`, 3 `cancelled`, 2 `merge-deferred`, **0 `review`** (control positive). One journal write ever: reify task 2958, 2026-05-07, `review` held **15 seconds**. Already documented as residue at `docs/task-escalation-state-spec.md:193` (§8-E14 retire-or-document) |

Corpus: **0** rows at either status in either project today, each zero carried against a positive
control (df `blocked` 22 / `deferred` 12 / `in-progress` 11; reify `in-progress` 48).

**Both are correctly KEEP, and `infra-hold` is keep *by design*, not by accident.** It means a hold on
a live, verify-complete branch whose worktree is preserved; clearing its claimant would be actively
wrong. Three mechanisms already depend on that: `release_workflow` refuses to park it
(`escalation/tests/test_release_workflow.py:320` — "the status IS the hold"), `is_stranded` hard-gates
on `in-progress`, and `_RECONCILE_SWEEP_STATUSES` (`harness.py:238`) excludes it.

**Consequence for C4-E6 — gate on the TTL, never on mere presence.** infra-hold holds run for weeks
(one reify task sat 18+ days), so a legitimately-kept claimant there will be arbitrarily stale. Any
consumer that treats *presence* as ownership will misread it. `is_stranded_any_status` is TTL-based and
therefore correct; the pre-existing raw-presence check at `task_interceptor.py:2211-2212` (the curator
combine guard, `pending` targets only) is **not**, and is noted as inherited, not introduced, by this
PRD.

**β additionally corrects the stale comment at `tools.py:1405-1407`**, which asserts "No current writer
emits 'infra-hold', so this is inert today" — falsified on this same HEAD by `workflow.py:7385` and by
the 15 journal writes. It sits in the file β edits, and it is the likeliest source of the earlier wrong
inventory, so leaving it would re-seed the same error.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-status-authority-prd.md` | extends | C4's "clears at release" half — `claimant_run_id`/`heartbeat_at` lifecycle | **this PRD** (enforcement); C4 remains the origin contract | wired |
| `docs/legibility/design-invariants.md` INV-6 `status-matches-liveness` | complements | this PRD enforces INV-6's converse | design-invariants (rule) / this PRD (this direction's enforcement) | wired |

**No reciprocal-ownership ambiguity exists** — INV-6's own *House pattern* already cites the
claimant columns as task-status-authority D4's mechanism and claims no ownership of the clearing
half, so nothing has to be taken away from another owner.

But the amendment to C4 is **not** the one-line pointer an earlier draft proposed. task-status-
authority does not merely gesture at the clearing half; it specifies a *mechanism*, in three places,
and after β that mechanism is an incomplete description rather than a wrong one:

- **D4** (`:158-159`): "refreshed on a lightweight heartbeat, **set NULL at release
  (scheduler.release / `_run_slot` finally)**" — and `Scheduler.release` does not in fact clear
  anything (see *Background*), so this sentence is doubly stale.
- **C4** (`:239`): "stamps at dispatch, refreshes, **clears at release**".
- **Acceptance test A5** (`:285`): "Claimant stamped at dispatch, cleared at release … **NULL after
  release**" — which reads as though release is the *only* clear.

C4-E2 makes `TaskInterceptor._apply_status_transition` the sole *enforcing* choke point, clearing on
terminal-entry rather than at release; C4-E3 preserves slot-release as a legal caller-side clear, so
there is no contradiction. A reader landing on D4 or A5, however, would not learn that. **η therefore
amends all three sites**, not just C4 — that is why η's signal greps for three hits, not one.

## Decomposition plan

Phase 1 — foundation (α). Phase 2 — enforcement slice (β, γ, δ). Phase 3 — repair (ζ).
Phase 4 — detection (ε). **Repair precedes detection by design** (D7): nothing heals an existing
row automatically, so arming the alarm first would fire it on residue no running code will clear.
η rides along with β.

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **α** | Add the shared claimant predicates: `is_stranded_any_status`, `violates_terminal_claimant_invariant`, `is_stale_nonterminal_claimant`, `DEFAULT_CLAIMANT_HEARTBEAT_TTL` | `shared` | intermediate | Unlocks **γ, ζ and ε**: the status-agnostic liveness predicate γ repoints onto, plus the C4-E1 and hygiene-tier predicates and the single TTL constant that ζ's census and ε's detector both consume instead of minting their own | — |
| **β** | Clear the claimant columns on entry to a terminal status at the fused-memory choke point; correct three falsified comments in the files it touches | `fused-memory` | intermediate (unlocks ζ, ε, η) | Through the product's own read path: `set_task_status(id,'done')` on a claimed row, then `get_task(id)` returns `claimant_run_id: null` / `heartbeat_at: null`; a `blocked` write on the same row leaves both intact | — |
| **γ** | Repoint `_resolve_live_claimant` at the status-agnostic predicate; invert the blocked-is-live pin | `orchestrator` | **leaf** | A `blocked` task whose branch is gone with no merge marker, carrying a claimant stale past the TTL, holding **no open escalation at any level**, and not `task_kind='deterministic'`, now has a `stranded_blocked` L1 filed for it by the reconcile sweep — visible via `get_task_escalations(<id>)` — where today the sweep classifies LEAVE and files nothing | α |
| **δ** | Clear the claimant before the reconcile revert-to-pending flip | `orchestrator` | intermediate (unlocks ζ) | After the stranded sweep reverts a task, `get_task` shows `claimant_run_id: null` (today it shows the dead run's id) | — |
| **ζ** | `scripts/clear_leaked_claimants.py` — corroborating, staleness-gated census + repair | `scripts` | intermediate (unlocks ε) | `--dry-run` prints, and `--json` emits, a per-status/per-tier violation census for each supplied project root — enumerating task **4028** among the terminal-tier violations; after `--apply`, an immediate re-run reports **zero terminal-tier violations among the rows the apply pass enumerated**, with any row that newly went stale during the window listed separately as `arrived_during_window` rather than counted as failure | α, β, δ |
| **ε** | Recurring census detector: alarm a terminal row carrying a claimant as a logical error (born-at-L2, sentinel-deduped) | `scripts`, `orchestrator` | **leaf** | A seeded terminal row carrying a claimant makes the next census pass file exactly one born-at-L2 record: `get_pending_escalations(level=2)` returns one row with `severity='critical'`, `category='infra_issue'` and a summary containing the verbatim discriminator `claimant invariant violation`, filed under the process sentinel rather than the violating task's id; a second seeded violating row in the same pass leaves that count at **one**, while each row gets its own ERROR log line | β, ζ |
| **η** | Amend the origin contract: C4 pointer, D4's mechanism sentence, and acceptance test A5 | `plans` | **leaf** (non-code) | `git grep -n 'claimant-invariant-enforcement' -- plans/task-status-authority-prd.md` returns hits in **all three** places — the `### C4 — claimant/heartbeat columns` section, D4's clearing-mechanism sentence, and an annotation on acceptance test A5 — where it returns **zero** hits today | β |

**η routing note.** η is a genuinely non-code leaf. `planning_mode` bypasses the curator-side routing
guards, so η must self-declare its execution path at filing: **`metadata.execution_class`
= `'operational'`** — one of the two non-`code_tdd` members of
`fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES` that
`routing_intent_guard` treats as an honest declaration rather than a mismatch
(`routing_intent_guard.py`, `_EXEMPT_EXECUTION_CLASSES`). Note `execution_class` is not in
`_BLESSED_METADATA_KEYS`, so filing it emits an `unknown_key` warning — harmless and expected (272
tasks already carry it). η's docs-only signal is acceptable *because* the change is documentation by
nature; it is not a code task closing via a docs commit (the shape G2 rejects), and the signal is a
`git grep` rather than a prose assertion so it is falsifiable by inspection.

**G2 note.** The batch has **four intermediates — α, β, δ and ζ** — and three true DAG sinks: γ, ε
and η. (An earlier draft called α "the only intermediate"; that was wrong, and it would have misled
whoever wired the dependency edges.) Each intermediate names the task(s) it unlocks in the table
above, satisfying G2 step 3.

Every task nonetheless carries a signal observable through a product read path (`get_task`,
`get_task_escalations`, `get_pending_escalations`), a CLI output difference (ζ), or a `git grep` (η)
— deliberately stronger than G2 requires, since step 2 obliges only the sinks. **None rests on "a
unit test passes against synthetic input."**

Two honest caveats on observability, recorded rather than papered over:

- **γ and ε need seeded rows.** There are currently **zero** `blocked` rows carrying a claimant
  fleet-wide, and after ζ runs there will be zero terminal violations either. Both signals are
  demonstrated against a deliberately seeded row. That is not the shape G2 rejects — the observation
  is still made through the product's own read path, on the real code path, with a real escalation
  record — but neither can be demonstrated against found traffic.
- **δ is not operator-invocable.** The reconcile sweep fires at startup and on cadence
  (`harness.py:2490`, `:2163`, `:2621`); demonstrating δ needs a restart or a wait.

**G6 note.** ζ's signal asserts a number (zero), so it needs an achievability basis — and the basis
an earlier draft gave was **contradicted by this PRD's own Residual section**. That draft claimed
that after β and δ land "the residual set is exactly the 105 pre-existing rows". It is not: β closes
only TERMINAL writes and δ closes only the `harness.py:6174` revert; **neither closes process
death**, which this PRD elsewhere credits with the bulk of the corpus. On a live fleet the number is
a moving target — reify ran 48 concurrent heartbeating workflows during authoring, and any row whose
heartbeat crosses the TTL between `--apply` and the re-run newly qualifies.

The predicate is therefore scoped to what is actually achievable, and is **closed over an enumerated
set rather than over time**: *zero terminal-tier violations **among the rows the apply pass
enumerated***, with newly-stale arrivals reported separately as `arrived_during_window`. It is also
deliberately the **stale** tier (D3), because a fresh claimant on a non-terminal row is legal during
the C3.1 window and must not count as failure.

**The zero also carries a named positive control.** A census that cannot read its corpus also prints
zero, so the post-apply zero is unfalsifiable on its own. Task **4028** (dark-factory, `status=done`,
carrying `run-a1d3b5dba75a/4028-ba4c3e3e/pid=1807449`, heartbeat `2026-08-19T21:57:20Z`) is a live
C4-E1 violation measured during authoring; ζ's `--dry-run` must enumerate it (or a named equivalent)
*before* `--apply`, or the zero afterwards proves nothing.

ε depends on ζ so detection is not armed against legacy residue (D7), and on β so the dominant mint
path is closed first.

**G7 walk.** Re-derived against `docs/legibility/design-invariants.md` (8 invariants, INV-1..INV-8,
no drift). Several claims in an earlier draft of this paragraph were false; the resolutions below
are the ones actually adopted.

- **`contracts-machine-checked`** — resolved in α. C4-E1 and D3's hygiene tier ship as exported
  predicates that β's tests, ε and ζ all call. *Previously false:* the draft claimed the invariant
  "ships as an executable predicate" while no task delivered one — α delivered only C4-E**6**, the
  read predicate.
- **`structured-facts-at-failure`** — ε emits task id / status / claimant / heartbeat as fields and
  populates the escalation record's `evidence` list (one capped entry per violating row), so a human
  reading the L2 need not scrape the ERROR log to learn *which* rows violated. ζ emits `--json`.
  *Previously incomplete:* the draft's ζ reported its census in prose only.
- **`corroborate-before-acting`** — ζ re-reads **each row** immediately before its own write, in
  addition to the batch-level holder intersection (D6.1). *Previously insufficient:* an aggregate
  pre-flight cannot catch a row reopened between census and apply.
- **`storm-escape-required`** — the durable sentinel dedup bounds the L2 to one per episode, and it
  genuinely survives restart (`get_by_task(<sentinel>, status='pending', level=2)` reads from disk,
  so a crash-loop files one record, not one per restart). The gap the draft missed is the
  **fail-soft path**: the mandated `try/except` and the `escalation_queue is None` early return
  would degrade to silence. Both now log at WARNING with `exc_info=True`, per
  `task_ground_truth.py:729-734` and `harness.py:8089`.
- **`no-lockstep-duplication`** — D1 reuses `TERMINAL` rather than minting a parallel set, and α's
  shared predicates stop ε and ζ each hand-maintaining the invariant. *Previously false:* the draft
  said "α deletes the duplicated liveness cores". There are none — `_claimant_liveness_stranded`
  (`task_claimant.py:63`) is already the single shared core and all three predicates already
  delegate to it. The only duplication is the two-line `metadata.infra_hold` block at `:137-139` /
  `:180-182`. ζ additionally imports the TTL rather than minting a sixth copy.
- **`status-matches-liveness`** — this PRD is INV-6's converse. **δ's clear-then-flip ordering is
  correct and does not violate INV-6's "successor status before the claim is released" clause**,
  and it is worth stating why rather than leaving a reader to trip over the apparent contradiction:
  INV-6's ordering clause governs an exit from a *claimed* state, whereas δ acts on a row whose
  claimant is already established dead, making δ the crash backstop INV-6 explicitly sanctions.
  Reversing it would be actively worse — a crash between the two writes would leave
  `(pending, stale claimant)`, re-minting this PRD's own defect, and a late-landing clear could NULL
  a fresh claimant stamped by a concurrent dispatcher. `scheduler.py:6421-6429` already reasons this
  out in-source. The second clause ("what test pins every exit?") was a real gap: δ had no boundary
  row, now B13/B14.
- **`holds-owned-and-bounded`** — ε's L2 is a human-owned hold with the L2 watcher as exit, **plus
  an auto-resolve**: after N consecutive clean census passes the sentinel record is resolved,
  re-arming the detector. Without it the hold suppresses its own detector — while the sentinel L2
  sits pending, `get_by_task` returns non-empty and ε goes quiet — which is INV-7's own cited
  failure. The idiom is `_file_watcher_outage_l2` / `_resolve_watcher_outage_l2`
  (`harness.py:8030`, `:8091`).
- **`loop-thread-occupancy-bounded`** — **discharged by D8, not waived.** The flag was real: the
  originally-proposed site is on the asyncio event-loop thread and `EscalationQueue.submit` is a
  flock plus two fsyncs. Moving detection to the census takes that write off the loop thread
  entirely. Recording the disposition here rather than deferring it to the implementer, since
  "confirm at implementation time" is not one of G7's two dispositions.

## Out of scope

- **Widening enforcement to `pending`/`deferred`** — refuted by D2. If the C3.1 window is ever given
  a compare-and-swap on the claim write, revisit.
- **Making `set_task_claimant` status-aware** (refusing a claimant stamp onto a terminal row). That
  would make the dispatch gate structurally dead and deletable, but costs a `get_task` on the
  lock-free heartbeat hot path. Deliberately not taken: the invariant is enforced at transition time
  and violations are alarmed, not assumed impossible.
- **Deleting the task-2408 dispatch gate.** It is reachable-only-on-violation, not dead, and it is
  the only backstop against dispatching into a live worktree — and it demonstrably fires (one hit in
  the 7-day journal; see *Current impact*). It is left exactly as it is. *An earlier draft promised
  it would be "promoted to an alarm site"; no task in this decomposition delivers that, and the
  promise is withdrawn rather than left unowned.* It already logs at INFO
  (`scheduler.py:4939-4943`), which is sufficient.
- **Preventing the raw-SQL bypass at `sqlite_task_backend.py:705`.** The v3→v4 duplicate-
  `candidate_key` self-heal writes `status='cancelled'` in raw SQL, bypassing
  `_apply_status_transition` entirely and leaving the claimant untouched. After β it is the only
  remaining minter of fresh C4-E1 violations. It is **covered by detection, not prevention**: ε's
  census sees it (D8), whereas a choke-point fix by construction cannot. Changing that write is a
  migration-path decision with its own blast radius and is deliberately not taken here.
- **Having the heartbeat loop re-stamp `claimant_run_id`.** Only relevant to a `pending` clear, which
  D2 rejects.
- **Backfilling other projects' corpora beyond dark-factory and reify.** ζ takes project roots as
  arguments; running it elsewhere is an operator action.
- **The `review` status's unbounded-hold hazard.** `review` is human-writable with no exit owner and
  no bound, and a `review` row is unreachable by *every* reaper (`is_stranded`'s `in-progress` gate,
  the `_RECONCILE_SWEEP_STATUSES` exclusion, and starvation ineligibility) — so a human setting
  `review` on a task with a stale claimant produces a permanently unreapable row. Inherited, not
  introduced: it belongs to `docs/task-escalation-state-spec.md` §8-E14 (retire-or-document), which
  already tracks it.
- **The infra-hold resume path's claimant-less `in-progress` write** (spec §8-E5) — the resume
  "manufactures the strand shape" and waits for the sweep, discarding the claimant on the very next
  transition. Keeping the claimant through the hold is still correct, but this PRD does not fix E5.
- **Adding a TTL to the curator combine guard** (`task_interceptor.py:2211-2212`). Its raw-presence
  check will misread a legitimately-stale infra-hold claimant, but that is pre-existing behaviour on a
  `pending`-target-only path, and changing it is a curator-semantics decision, not a claimant-lifecycle
  one.

## Residual after this PRD (accepted, stated)

Because D2 declines to clear `pending` at the choke point, **process death can still leak a
`pending` row**: the dying run never reaches slot release, which is the unconditional clear.

**Attribution, corrected.** An earlier draft said both that the Background table's
`harness.py:6174` row accounted for all 76 pending rows *and* that process death "is the exact
mechanism that produced the current 76 pending rows" — mutually exclusive claims, neither measured.
The actual heartbeat clustering: dark-factory's 36 = **32 at 2026-08-19T21** (the fleet-redeploy
batch) + **4 at 2026-08-17T22**; reify's 39 = **26 at 2026-08-21T00** + a 13-row tail spanning
2026-07-23 → 2026-08-18. Task 3996's "all from the 2026-08-19T21:57 death" is likewise wrong by four
rows. The "rows leaked" column attributes by *path shape*, not by a measured per-row provenance, and
should be read that way.

**The leak is actively re-accumulating.** 26 of reify's 39 pending-claimant rows were minted in a
single batch roughly 13 hours before this measurement — so δ's urgency is materially higher than a
"38-hour-old residue" framing suggests, and ζ's repair will need re-running unless δ lands with it.

δ covers the dominant case — the post-restart reconcile sweep re-pends stranded rows through
`harness.py:6174`. What remains uncovered is any *other* path that writes `pending` and is not
followed by a slot release. Those are detectable on demand via ζ's hygiene tier
but are deliberately **not** alarmed, because a fresh claimant on a `pending` row is legal during the
C3.1 window and an alarm cannot distinguish the two at write time.

Accepted because the leaked rows are provably inert for scheduling, completion, and dependency
release (measured — see *Current impact*), and because the alternative is the double-dispatch hazard
D2 rejects. If the hygiene tier is observed to re-accumulate materially after δ lands, the follow-up
is a periodic sweep clearing *stale* claimants on non-terminal rows — safe precisely because
staleness excludes the C3.1 window — not a widening of the write rule.

## Open questions (tactical)

1. **ε's sentinel spelling.** A single fleet-wide constant (e.g. `claimant-invariant-violation`) vs
   one per project id. **Suggested resolution:** fleet-wide single constant; per-project only if the
   single record proves hard to attribute. Note the sentinel must be a **durable string**, not a
   per-process value — dedup reads it back off disk via `get_by_task`, which is what makes a
   restart-loop file one record rather than one per restart. Decide during ε.
2. **Whether ζ should also report the non-terminal stale-claimant tier by default or only under a
   flag.** **Suggested resolution:** report both tiers, repair only what the flags select. ζ must
   in any case name its hygiene-tier status scope **explicitly** — `{pending, deferred, review,
   merge-deferred}` — rather than deriving it as "non-terminal": `in-progress` is non-terminal, and
   clearing an `in-progress` claimant is the task-2588 un-claim class D2 rejects *and* would blind
   `is_stranded`, the reaper's own detector. (reify held 48 claimant-bearing `in-progress` rows
   during authoring; they are excluded today only because they were fresh.) Decide during ζ.
3. **ε's census cadence and its N-clean-passes auto-resolve threshold.** Both are tactical numbers,
   not design choices. Decide during ε.

*(The former Open Question 3 — whether ε's check belongs inside `_resolve_live_claimant` or
`derive_truth` — is closed by D8: neither, because both sit inside a sweep that cannot see a
terminal row.)*
