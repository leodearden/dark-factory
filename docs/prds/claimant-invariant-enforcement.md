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
  earlier "zero `dispatch refused: live claimant` lines against a control of 168,458 journal lines"
  was wrong, and its first correction was wrong too. **State the journal claim carefully or not at
  all** — it decays fast:

  - The journal retains only what has not rotated. As of 2026-08-22 the earliest retained entry for
    **both** units is `2026-08-18T20:47` — a **~3.5-day** window, not 7 days — and
    `journalctl --since "7 days ago"` silently returns only the retained subset, so any
    "N lines over 7 days" denominator is mislabelled by construction.
  - Within the retained window the refusal string appears **0** times today. A refusal *was*
    observed on 2026-08-18 (`Task 6218 … pid=1805896` — the emitting process's own pid, i.e. a fresh
    self-claim inside the C3.1 teardown window, the gate working as designed). That entry has since
    **aged out and is no longer reproducible.**
  - The emitter is a **format string** (`scheduler.py:4939-4943`,
    `'Task %s dispatch refused: live claimant %r (task 2408 mechanism 1)'`), so grepping the literal
    *including* `%s`/`%r` returns a guaranteed zero. Grep the stable prefix
    `dispatch refused: live claimant`, always against a known-positive control.

  **Conclusion at its real strength:** no leaked row has been *observed* to block a dispatch, and the
  one refusal ever inspected was a live self-claim rather than a leaked row. That is an inference
  from a single expired observation — **not** a measured absence, and not re-verifiable from today's
  journal.
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
   contract needs (`violates_terminal_claimant_invariant`, `is_stale_hygiene_tier_claimant`) and the
   shared `DEFAULT_CLAIMANT_HEARTBEAT_TTL`, repointing the four existing hand-maintained TTL copies
   at it so ζ — and the detection PRD after it — consume one definition instead of minting more.
4. **Detection is deferred** — a terminal row carrying a claimant is a logical error, but the
   unattended alarm gets its own PRD (D8). What lands here is the *substrate* for it: α's executable
   predicate and ζ's on-demand census, so an operator can observe the invariant today.
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

### D5 — Alarm shape (carried forward to the detection PRD, not built here)

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
  `summary` substring. **But the transplant fails**: that precedent is born at **L0**
  (`severity='blocking'`, no `level` ⇒ default 0) and its discriminator is consumed by an L0-only
  reaper (`if esc.level != 0: continue`, `harness.py:11831`), so a born-at-L2 record changes level,
  role and severity and no consumer can see it. `infra_issue` is also the largest category in both
  corpora (829/2,693 on dark-factory, 13 already pending at L2). The detection PRD must pick a
  discriminator that actually discriminates.
- **`agent_role='orchestrator-deterministic'` is load-bearing, not decoration.** It is in
  `L2_AUTO_CLOSE_DENY_ROLES` (`escalation/authority.py:94`). Without it the record matches the
  `stale_task_scoped` auto-close class, which is **category- and role-agnostic**
  (`authority.py:210-213`) and keys on evidence text matching `status\s*[=:]\s*(done|cancelled)`
  alone (the evidence field is **one OR-group of three alternatives**, `authority.py:214-227` — not
  `status=done` AND a citation, as an earlier draft said). Note the regex reads the **`resolution`
  string supplied at close time** (`authority.py:264`), not the record's own `evidence`/`summary`.
  And the role denylist governs only `authority.py`: a **second closer**,
  `Harness._revalidate_open_deterministic_escalation` (`harness.py:13138-13187`), bypasses it and its
  predicate is exactly `category == 'infra_issue'` **and** `agent_role in
  _DETERMINISTIC_ESCALATION_SENTINEL_ROLES` — the same pair. Today only the sentinel failing to
  resolve to a real task saves it. The detection PRD owns this analysis.
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
(`:407-412`) never advances the local variable, so an appended `if version < 5:` step in the same
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
   tree (the phrase occurs elsewhere only as merge-lane prose).
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
   row this PRD counts. (That row also falsifies `tools.py:948-952`, which asserts a claimant is
   "always machine-composed by `compose_claimant_run_id()` … never freeform text" — the same
   stale-comment class β already corrects one screen below at `:1328-1329`. β corrects both.)

### D7 — Repair must precede the alarm

`task_interceptor.py:1008` short-circuits a same-status write as a no-op (returning
`{'success': True, 'no_op': True, …}` at `:1018`), so **β's coercion never runs on a row that is
already terminal** — a `done` row carrying a claimant cannot be healed by re-writing `done`.

The stronger claim an earlier draft made — that *no code path can heal an existing row* — is **false**
and is withdrawn. `TaskInterceptor.set_task_claimant` (`:1341`) is a thin delegate to
`SqliteTaskBackend.set_task_claimant` (`:2210`) with **no status gate whatsoever**, and it is exposed
as an MCP tool (`tools.py:7673`); passing `claimant_run_id=None, heartbeat_at=None` heals a terminal
row today. ζ's whole design depends on that being true, and task 3996 records Stage 2 having already
used it once by hand.

The correct statement is the one the ordering actually needs: **nothing heals an existing row
_automatically_.** Healing requires the deliberate write ζ performs, so enabling detection before the
repair runs would alarm on legacy residue that no running code will ever clear. The DAG enforces the
ordering.

### D8 — Detection is deferred to its own PRD (the sweep cannot host it, and the census needs a design)

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

**A recurring census was the chosen replacement, and it did not survive its own gate walk.** Three
justifications were offered for it and all three were falsified:

- *"It is the only vantage point that catches the `sqlite_task_backend.py:705` raw-SQL bypass."*
  **`:705` is unreachable.** Both entries to `_migrate_v3_to_v4` gate on `user_version < 4`
  (`_migrate` `:330-331` with `_SCHEMA_VERSION = 4`; `reaudit_candidate_key_index` `:2311-2313` and
  again under the lock at `:2322-2324`), and **all nine task DBs on this host read
  `user_version = 4`** with zero residual duplicate `candidate_key` groups. It can only fire on a DB
  parked at v3; none exists.
- *"`category='infra_issue'` keeps the record discriminable."* `infra_issue` is the **largest
  category in both corpora** — 829/2,693 records on dark-factory (30.8%), **13 already pending at
  L2** — and 665/2,052 on reify. The verbatim summary substring would be doing 100% of the
  discrimination.
- *"It copies the `_escalate_scope_invariant_violation` precedent."* That precedent is **born at L0**
  (`severity='blocking'`, `agent_role='orchestrator'`, no `level` ⇒ default 0) and its discriminator
  is consumed by an **L0-only** reaper (`_is_scope_divergence_orphan`, filtered by
  `if esc.level != 0: continue` at `harness.py:11831`). A born-at-L2 detector changes level, role
  *and* severity, so the consumer can never see it. The transplant copies the fragile half — a prose
  substring held in manual lockstep — and drops the half that made it work.

Four further gaps were found and none is tactical: the census has **no named invocation mechanism**
(process, timer, ladder slot, project scope, or target queue — and escalation queues are
**per-project**, so "one record fleet-wide" is undefined across 7 queues); its **auto-resolve can
silence it permanently**, because "N consecutive clean passes" is satisfied identically by health and
by a census that cannot read its corpus; a **second closer**
(`Harness._revalidate_open_deterministic_escalation`, `harness.py:13138-13187`) bypasses
`authority.py` entirely and its predicate is *exactly* `category == 'infra_issue'` **and**
`agent_role in _DETERMINISTIC_ESCALATION_SENTINEL_ROLES` — the very pair chosen to defeat the *first*
closer; and C4-E1's heartbeat race (above) would make it alarm on routine completions.

**Decision: detection is out of scope for this PRD and gets its own design pass.** The enforcement
and repair halves (α, β, γ, δ, ζ, η) are independently valuable, independently verifiable, and
urgent — the hygiene tier re-accumulated 26 rows in 13 hours during authoring. Holding them behind a
detector that has now failed two design passes serves nobody. The blindness finding above is the
durable result to carry forward: **whatever builds the detector must not site it in the reconcile
sweep**, and must answer the four gaps.

## Contract (B+H)

**Invariant C4-E1.** For any task row: `status ∈ TERMINAL ⇒ claimant_run_id IS NULL`.

**The assertion is on `claimant_run_id` alone, deliberately.** An earlier draft required
`heartbeat_at IS NULL` too, and that form is unachievable: `_claimant_heartbeat_loop`
(`workflow.py:2504-2508`) ticks `set_task_claimant(task_id, heartbeat_at=…)` every
`claimant_heartbeat_interval_secs` (default **60 s**, `config.py:2979`) with **no status gate**,
while `_stop_claimant_heartbeat` runs only from `_on_terminal_cleanups` (`workflow.py:3169`) —
*after* `mark_done` is called from inside the workflow body. So β clears both columns on the terminal
write and the next tick re-stamps `heartbeat_at`, leaving `(done, NULL, fresh heartbeat)` for up to a
minute after **every ordinary completion**. Under the two-column form that is a violation minted by a
hot path on routine traffic.

The existing safety comment on `_stop_claimant_heartbeat` — *"Called first from
`_on_terminal_cleanups` (before the harness clears the claimant at slot release) so the loop can
never race a post-clear re-stamp"* — is true only of the **slot-release** clear. β introduces a
second, earlier clear the guarantee never covered: the argument survives the words but not the
referent.

`claimant_run_id` is the column that means ownership, and the residue is **provably inert, not merely
harmless**: `_claimant_liveness_stranded` (`shared/src/shared/task_claimant.py:63`) reads
`claimant_run_id` **first** and returns "no live claimant" on a NULL/blank one *before it ever parses
`heartbeat_at`*. Every liveness predicate in the repo delegates to that core, and the dashboard only
carries the column through for display — so a `(NULL, timestamp)` row cannot influence any liveness
decision anywhere. Narrowing is therefore not a weakening; it states the invariant on the column that
carries the meaning. D3's alarmable tier is likewise "a terminal row carrying a **claimant**", which
this now matches exactly.

β still writes **both** columns NULL. The residual `(NULL, timestamp)` untidiness is accepted.

**The narrowing is load-bearing, not a stopgap — do not "fix" it later by moving the heartbeat stop
and re-widening C4-E1 to both columns.** Cancelling the heartbeat task does not un-send an
already-dispatched MCP write, so an in-flight tick can still land after the terminal write however
early the stop is moved. Moving it takes the residue from *systematic* (every completion, up to one
interval) to *sporadic* — and a sporadically-false invariant driving an alarm is worse than a
systematically-false one, because it yields a flaky, hard-to-reproduce signal instead of one you can
reason about. The narrow form is true by construction; the wide form cannot be made true.

Stopping the loop after a terminal write is still worthwhile **as hygiene of the heartbeat loop**
(it stops writing freshness for a claim that no longer exists) and is filed separately. Note the
placement: **after a confirmed-successful terminal write, never before it.** Stopping ahead of the
write means a *failed* write leaves a live workflow holding a claim nobody is refreshing — it ages
past the TTL, the reconcile sweep classifies the row stranded, and dispatch can re-enter a live
worktree. That is precisely the hazard D2 exists to prevent, traded for cosmetic tidiness. (An
earlier revision of this paragraph prescribed "ahead of the terminal write"; that prescription was
wrong and is withdrawn.)

It ships as an **executable predicate, not prose**: α exports
`violates_terminal_claimant_invariant(task)` from `shared/src/shared/task_claimant.py`, and β's
tests and ζ's census both call it rather than re-expressing it, and the detection PRD inherits it
instead of minting a third copy. This is INV-1
`contracts-machine-checked`, and it is load-bearing rather than tidy: without a single definition,
ζ and a future detector would each hand-maintain their own copy of the invariant *plus* D3's
two-tier split, and those copies must agree byte-for-byte or the repair and the alarm disagree about
what counts as a violation. D3's hygiene tier gets the same treatment
(`is_stale_hygiene_tier_claimant(task, now, ttl)` — named for the **tier**, not for
"non-terminal", because the hygiene scope is the explicit allowlist
`{pending, deferred, review, merge-deferred}` and NOT all seven non-terminal statuses: `in-progress`
is the task-2588 un-claim class D2 rejects, and `infra-hold` legitimately carries arbitrarily-stale
claimants for weeks per the status-producer audit).

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

**C4-E7 (violation).** An observed violation of C4-E1 is a logical error: it must be surfaced
loudly and structurally, never raised, never silently absorbed. **This PRD delivers the detection
*substrate* but not the detector** (D8): α exports `violates_terminal_claimant_invariant` and ζ
reports every violating row through `--dry-run` / `--json`, so an operator can observe the invariant
on demand today. The recurring, unattended alarm is deferred to the detection PRD, which owns the
invocation model, the discriminator, the closer analysis, and the re-arm control.

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
| B11 | producer | δ clears **then** flips | `in-progress` row whose claimant is dead | after the sweep, `get_task` shows `claimant_run_id: null` **and** `status: pending` |
| B12 | producer | δ's crash window is backstopped | fault injected between δ's clear and its flip | row is left `(in-progress, NULL)` — never `(pending, stale claimant)` — and the next sweep still reverts it |
| B13 | repair | ζ refuses a row that changed under it | census sees `done`+claimant; row is reopened to `in-progress` with a fresh claimant before apply | ζ's per-row re-read skips it; the live claimant survives |
| B14 | repair | ζ handles a claimant with no heartbeat | row `cancelled`, claimant freeform prose, `heartbeat_at IS NULL` (reify 5225's shape) | classified stale via the shared predicate and repaired — not skipped as un-corroboratable |
| B15 | producer | the heartbeat race is benign under C4-E1 | claimed `in-progress` row completed via `mark_done` while its heartbeat loop is live | immediately after the `done` write both columns are NULL; within one heartbeat interval `heartbeat_at` may be re-stamped while `claimant_run_id` stays NULL — **not** a C4-E1 violation |

## Pre-conditions for activating

None external. Every substrate capability is present on main @ `7cb0ef2e0c` (G3, verified):

| Capability | Evidence |
|---|---|
| `_CLAIMANT_WIRE_UNSET` sentinel distinguishing unsupplied from explicit-null | `fused-memory/src/fused_memory/server/tools.py:953`, defaults at `:7566-7567`, `:7677-7678` |
| Single enforcing funnel for both status writers | `middleware/task_interceptor.py:871` `_apply_status_transition`, reached from `:833-848` and `:857-867` |
| Sole **status-write** SQL emitter for the columns | `backends/sqlite_task_backend.py:1922` `_write_status_and_verify`, tri-state block `:1969-1981`. (Not the *only* emitter: `set_task_claimant` at `:2210` writes them too — D6 depends on that — and the raw-SQL self-heal at `:705` writes `status` without them.) |
| Columns-absent fail-safe | `sqlite_task_backend.py:1970`, pinned by `tests/test_sqlite_task_backend.py:682` |
| `TERMINAL` frozenset | `shared/src/shared/task_statuses.py:61` |
| `set_task_claimant` writer that does not bump `updated_at` | `sqlite_task_backend.py:2265-2268` |
| Born-at-L2 escalation idiom from orchestrator code | `orchestrator/merge_queue.py:1140-1163`, `proc_supervision.py:205-226` |
| `_RECOVERY` row (g) `RE_FILE_ESCALATION` exists to become reachable | `task_ground_truth.py:838`; keyed on the 5-tuple `(BLOCKED, no-open-escalation, GONE_NO_MARKER, False, None)` at `:838-839` — γ's demo must satisfy all five, and `_RECOVERY` is a dict consumed by exact-key lookup (`:916`), so no row can shadow another |

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
wrong. Two mechanisms already depend on that: `is_stranded` hard-gates on `in-progress`, and
`_RECONCILE_SWEEP_STATUSES` (`harness.py:238`) excludes it.

A third was claimed and does not hold up: `release_workflow` is said to "refuse to park" an
infra-hold row, citing `escalation/tests/test_release_workflow.py:320` ("the status IS the hold").
The production guard is `if cur == 'in-progress':` (`escalation/server.py:2678`), and the string
`infra-hold` does not appear anywhere in that module — so the non-parking is **incidental** (an
infra-hold row simply is not `in-progress`), not a deliberate refusal. The test docstring asserts an
intent the code does not separately encode. Keep the conclusion, drop the third leg.

**Consequence for C4-E6 — gate on the TTL, never on mere presence.** infra-hold holds run for weeks
(one reify task sat 18+ days), so a legitimately-kept claimant there will be arbitrarily stale. Any
consumer that treats *presence* as ownership will misread it. `is_stranded_any_status` is TTL-based and
therefore correct; the pre-existing raw-presence check at `task_interceptor.py:2211-2212` (the curator
combine guard, `pending` targets only) is **not**, and is noted as inherited, not introduced, by this
PRD.

**β additionally corrects the stale comment at `tools.py:1328-1329`**, which asserts "No current writer
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
η rides along with β. **Detection is deliberately absent** — see D8; it gets its own PRD, and D7's
repair-before-detection ordering is preserved for free because nothing here arms an alarm.

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **α** | Add the shared claimant predicates (`is_stranded_any_status`, `violates_terminal_claimant_invariant`, `is_stale_hygiene_tier_claimant`) and export `DEFAULT_CLAIMANT_HEARTBEAT_TTL`, **repointing the four existing hand-maintained TTL copies at it** | `shared`, `orchestrator`, `dashboard`, `fused-memory` | intermediate | Unlocks **γ and ζ**: the status-agnostic liveness predicate γ repoints onto, plus the C4-E1 predicate and single TTL constant ζ's census consumes instead of minting its own. Observable through the consumers: after α, `git grep -c 'timedelta(minutes=10)'` over the four repointed sites returns **0** where it returns 4 today | — |
| **β** | Clear the claimant columns on entry to a terminal status at the fused-memory choke point; correct the two falsified comments in the file it edits (`tools.py:1328-1329` infra-hold-is-inert, `tools.py:948-952` claimant-is-never-freeform) | `fused-memory` | intermediate (unlocks ζ, η) | Through the product's own read path: `set_task_status(id,'done')` on an **`in-progress`** claimed row, then `get_task(id)` returns `claimant_run_id: null` / `heartbeat_at: null`; a `blocked` write on the same row leaves both intact; and the same `done` write on an **already-`done`** claimed row (with no `done_provenance` supplied) returns `{'no_op': True}` and leaves the claimant intact, pinning D7's ordering necessity | — |
| **γ** | Repoint `_resolve_live_claimant` at the status-agnostic predicate; invert the blocked-is-live pin | `orchestrator` | **leaf** | A `blocked` task whose branch is gone with no merge marker, carrying a claimant stale past the TTL, holding **no open escalation at any level**, carrying **no `metadata.deploy_state`** and not `task_kind='deterministic'`, on an orchestrator with `stranded_blocked_escalate_enabled` (default) and an escalation queue wired, now has a `stranded_blocked` L1 (`agent_role='harness-stranded-blocked-reaper'`, `level=1`) filed for it by the next reconcile sweep — visible via `get_task_escalations(<id>)` — where today the sweep classifies LEAVE and files nothing | α |
| **δ** | Extract `Scheduler.clear_claim_then_set_status` and call it from both the stranded-blocked sweep and the reconcile revert-to-pending flip; fix the two rotted slot-release citations in `scheduler.py` | `orchestrator` | intermediate (unlocks ζ) | After the stranded sweep reverts a task, `get_task` shows `claimant_run_id: null` (today it shows the dead run's id) | — |
| **ζ** | `scripts/clear_leaked_claimants.py` — corroborating, staleness-gated census + repair | `scripts` | **leaf** | `--dry-run` prints, and `--json` emits, a per-status/per-tier violation census for each supplied project root — enumerating task **4028** (or a named equivalent, re-verified live at run time) among the terminal-tier violations; after `--apply`, an immediate re-run reports **zero terminal-tier violations among the rows the apply pass enumerated and did not skip**, with every remaining row accounted for in exactly one named bucket: `repaired`, `skipped_changed_under_us` (D6.1's per-row re-read declined it, reported with the observed delta), or `arrived_during_window` (newly qualified after the census snapshot) — none of which counts as failure | α, β, δ |
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

**G2 note.** The batch has **three intermediates — α, β and δ** — and three true DAG sinks: γ, ζ
and η. Each intermediate names the task(s) it unlocks in the table above, satisfying G2 step 3.
(Note α's row says it unlocks γ and ζ; there is no α→η or α→γ→ζ shortcut to wire — the edges are
exactly α→γ, α→ζ, β→ζ, β→η, δ→ζ.)

Every task carries a signal observable through a product read path (`get_task`,
`get_task_escalations`), a CLI output difference (ζ), or a `git grep` (α, η) — deliberately stronger
than G2 requires, since step 2 obliges only the sinks. **None rests on "a unit test passes against
synthetic input."**

Two honest caveats on observability, recorded rather than papered over:

- **γ needs a seeded row.** There are currently **zero** `blocked` rows carrying a claimant
  fleet-wide, so γ's signal is demonstrated against a deliberately seeded row. That is not the shape
  G2 rejects — the observation is still made through the product's own read path, on the real code
  path, producing a real escalation record — but it cannot be demonstrated against found traffic.
  The one short-circuit that could pre-empt it, `_maybe_submit_stranded_verified_green`, is
  **structurally unreachable** for γ's shape: it requires `resolve_branch_sha` to resolve the branch,
  and `GONE_NO_MARKER` is reached precisely *because* that same call already returned None.
- **δ is not operator-invocable.** The reconcile sweep fires at startup and on cadence
  (`harness.py:2490`, `:2163`, `:2621`); demonstrating δ needs a restart or a wait.

**G6 note.** ζ's signal asserts a number (zero), so it needs an achievability basis — and the basis
an earlier draft gave was contradicted by this PRD's own Residual section. β closes only TERMINAL
writes and δ only the `harness.py:6174` revert; **neither closes process death**, so on a live fleet
the total is a moving target.

**But the tier that matters is stable, and that is the real basis.** Re-measured across one day
(2026-08-21 → 08-22) the fleet total fell 104 → 84 — *entirely* in the hygiene tier (reify `pending`
39 → 20) — while the **terminal tier held at exactly 29** (28 `done` + 1 `cancelled`) on both days.
The invariant-tier corpus ζ actually repairs is not churning; only the hygiene tier is. ζ's predicate
is therefore stated against the terminal tier and closed over an **enumerated set** rather than over
time: *zero terminal-tier violations among the rows the apply pass enumerated and did not skip*, with
every other row landing in exactly one named bucket (`repaired`, `skipped_changed_under_us`,
`arrived_during_window`), none of which counts as failure.

**The zero carries a named positive control.** A census that cannot read its corpus also prints zero.
Task **4028** (dark-factory, `status=done`, `run-a1d3b5dba75a/4028-ba4c3e3e/pid=1807449`, heartbeat
`2026-08-19T21:57:20Z`) is a live C4-E1 violation re-verified through `get_task` on 2026-08-22; ζ's
`--dry-run` must enumerate it (or a named equivalent, re-verified at run time) *before* `--apply`, or
the zero afterwards proves nothing.

**G7 walk.** Re-derived against `docs/legibility/design-invariants.md` (8 invariants, INV-1..INV-8,
no drift), twice — the second walk was run against the amended plan and found four hits the first
could not see. Resolutions actually adopted:

- **`contracts-machine-checked`** — resolved in α. C4-E1 and D3's hygiene tier ship as exported
  predicates that β's tests and ζ both call. *Previously false:* an earlier draft claimed the
  invariant "ships as an executable predicate" while no task delivered one — α delivered only
  C4-E**6**, the read predicate.
- **`no-lockstep-duplication` / α (TTL)** — **redesigned, scope grown deliberately.** Exporting a
  constant while leaving the five hand-maintained copies standing would be a net *regression* (six
  sites, not one). α therefore repoints the four that are genuinely the same threshold
  (`task_ground_truth.py:274`, `harness.py:248`, `dashboard/data/tasks.py:283`,
  `live_workflow_detector.py:270`); `artifacts.py:1339`'s `600.0` is a `plan.lock` threshold on a
  different mechanism and is annotated as coincidentally-equal, not repointed. The objection that the
  dashboard cannot import shared is false — `dashboard/pyproject.toml:21` already declares
  `dark-factory-shared`, as do all four packages. Two of the existing copies already cite stale
  anchors, which is the decay this closes.
- **`no-lockstep-duplication` / δ** — **redesigned.** δ no longer adds a third hand-written
  clear-then-flip pair; it extracts `Scheduler.clear_claim_then_set_status(task_id, status)` and
  both `scheduler.py:6430-6431` and `harness.py:6174` call it. There is no harness/scheduler seam to
  cross — `Harness.scheduler` is a `Scheduler` and δ's site already calls
  `self.scheduler.set_task_status`. The extraction also collapses the two rotted slot-release
  citations (`scheduler.py:6307`, `:6422`, both pointing at `harness.py:5693-5696`; actual
  `harness.py:8873`) into one docstring. This matters because the PRD's own Background argues that
  convention-propagated-by-imitation is this defect's root cause; a waiver here would be
  self-refuting.
  > `G7 waiver: no-lockstep-duplication — the third clear-then-flip site, scripts/consume_redispatch_requests.py::_apply_repend, is an out-of-process consumer reaching fused-memory through an MCP client with a different call signature (client.set_task_claimant(task_id, project_root, ...)), so it cannot call the in-process Scheduler helper this batch extracts. Its ordering is additionally strictly stronger (a rejected clear ABORTs rather than proceeding best-effort), so rendering it from the shared site would be a behaviour regression. Mitigation: the extracted helper's docstring becomes the single normative statement of the ordering rule, and _apply_repend's docstring is repointed at it by symbol name rather than line anchor — its current citation (scheduler.py:5726-5736) has already rotted, which is the failure this waiver bounds rather than denies.`
- **`structured-facts-at-failure`** — ζ emits `--json` per the house pattern
  (`scripts/repair_wiped_metadata_files.py:1120`, `scripts/audit_combine_gate_marker_loss.py:1162`),
  so "zero" is machine-readable rather than recovered by parsing prose.
- **`corroborate-before-acting`** — ζ re-reads **each row** immediately before its own write, in
  addition to the batch-level holder intersection (D6.1). An aggregate pre-flight cannot catch a row
  reopened between census and apply; B13 pins it.
- **`status-matches-liveness`** — this PRD is INV-6's converse. **δ's clear-then-flip ordering is
  correct**, independently re-derived twice: INV-6's ordering clause governs an exit from a *claimed*
  state, whereas δ acts on a row whose claimant is already established dead, making δ the crash
  backstop INV-6 explicitly sanctions. Reversing it would leave `(pending, stale claimant)` on a
  crash — re-minting this PRD's own defect — and would let a late clear NULL a concurrent
  dispatcher's fresh stamp. `scheduler.py:6421-6429` reasons this out in-source. The second clause
  ("what test pins every exit?") was a real gap: δ had no boundary row, now B11/B12.
- **`storm-escape-required`**, **`holds-owned-and-bounded`**, **`loop-thread-occupancy-bounded`** —
  all three attached to the detector, which D8 removes from this batch. They are **carried forward as
  open requirements on the detection PRD**, not silently dropped: it must answer who runs the census,
  what bounds its fan-out, what notices when it stops, and how a re-arm distinguishes "clean" from
  "blind".

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
- **The unattended detector.** A recurring alarm on C4-E1 is deferred to its own PRD (D8). This
  batch ships the substrate — α's executable predicate and ζ's on-demand census — so the invariant is
  observable today; it does not ship the alarm. The detection PRD inherits four unmet requirements:
  an invocation model (process, cadence, project scope, and which of the **7 per-project** escalation
  queues receives the record), a discriminator that actually discriminates (`infra_issue` is 30.8% of
  the dark-factory corpus), a full closer analysis (`authority.py`'s denylist governs only one of at
  least three closers), and a re-arm that can tell "clean" from "blind".
- **Preventing the raw-SQL bypass at `sqlite_task_backend.py:705`.** The v3→v4 duplicate-
  `candidate_key` self-heal writes `status='cancelled'` in raw SQL, bypassing
  `_apply_status_transition` and leaving the claimant untouched. **It is currently unreachable** —
  both entries gate on `user_version < 4` and all nine task DBs on this host read 4, with zero
  residual duplicate groups — so it can only fire on a DB parked at v3. Recorded as a conditional
  residual, not an active hazard; an earlier draft called it "the only remaining minter after β",
  which was wrong twice over (it is unreachable, and `set_task_claimant` is MCP-exposed with no
  status gate).
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

1. **Whether ζ should report the hygiene tier by default or only under a flag.** **Suggested
   resolution:** report both tiers, repair only what the flags select. The hygiene scope is
   **already decided and is not tactical** — it is the explicit allowlist
   `{pending, deferred, review, merge-deferred}`, encoded in α's
   `is_stale_hygiene_tier_claimant` rather than derived as "non-terminal": `in-progress` is the
   task-2588 un-claim class D2 rejects (and clearing it would blind `is_stranded`, the reaper's own
   detector), and `infra-hold` legitimately carries weeks-stale claimants per the status-producer
   audit. Only the default-vs-flag presentation is left to ζ.
2. **ζ's `--apply` batch size / whether to require an explicit `--project-root` per run rather than
   defaulting to both.** Operator-ergonomics only. Decide during ζ.

*(Two former open questions are closed. "Where does ε's check belong — `_resolve_live_claimant` or
`derive_truth`?" is answered by D8: neither, because both sit inside a sweep that cannot see a
terminal row. "What is ε's sentinel spelling?" moves to the detection PRD, where it is **not**
tactical: any spelling `scheduler.get_task` can resolve to a real task hands the record to a second
closer the role denylist does not govern.)*
