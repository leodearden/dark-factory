# Claimant invariant: enforce, detect, and repair `claimant_run_id` liveness

**Status**: active · authored 2026-08-21 · approach **B+H** (contract + two-way boundary tests)
**Verified against**: main @ `7cb0ef2e0cf732b4a39a14c8bdd99e89f440d084`
**Origin contract**: `plans/task-status-authority-prd.md` C4/D4 (tasks 2182, 2188, 2408) — this PRD
owns the *enforcement* of C4's clearing half; C4 remains the origin contract.

## Goal

A task's `claimant_run_id` / `heartbeat_at` columns mean "a live orchestrator run owns this task".
Today that meaning is maintained by per-caller convention, and it has decayed: **105 rows across the
fleet carry a claimant that owns nothing** (dark-factory 36 `pending` + 17 `done`; reify 40 `pending`
+ 11 `done` + 1 `cancelled`), every one of them stale, with **zero live pids**.

After this PRD:

- Entering a terminal status clears the claimant **atomically with the status write**, at the
  fused-memory choke point, for every caller in every targeted project.
- The one reconcile path that re-pends a task without clearing does so explicitly, where it has
  already established death.
- `TaskGroundTruth` stops reporting a stale claimant as live on non-`in-progress` rows.
- A violation of the invariant is **alarmed as a logical error**, not silently absorbed.
- The existing 105 rows are repaired by a corroborating, staleness-gated operator script.

## Background

### The defect is a missing invariant, not two sloppy call sites

C4 contracts the orchestrator to "stamp at dispatch, refresh, **clear at release**". The gap is that
several exits **never go through release**, so the clearing half of C4 is unenforced:

| Path | Clears? | Rows leaked |
|---|---|---|
| `Scheduler.release` / `_run_slot` finally (`harness.py:8858-8875`) | yes | — |
| blocked→pending sweep (`scheduler.py:5709-5773`) | yes (explicit pre-clear) | — |
| `Harness._revert_in_progress_if_no_live_claimant` (`harness.py:6174`) | **no** | 76 `pending` |
| `Scheduler.mark_done` / `_finalise_recovery_done` (found_on_main recovery) | **no** | 28 `done` |
| `scripts/consume_redispatch_requests.py::_apply_close` (`:599`) | **no** | 1 `cancelled` |

Two of five compliant — and the two that comply carry copy-pasted comments citing each other
("mirroring the slot-release NULL-claimant convention at harness.py:5693-5696"). A convention
propagated by imitation is what decays as paths multiply. A sixth path was demonstrated live during
authoring: a plain `set_task_status(4028, 'done', …)` through the sanctioned MCP choke point carried
a 22h-stale claimant straight from `pending` into `done`.

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

- **Scheduling is not blocked.** `has_live_claimant(row, now, ttl=300s)` returns **False** on real
  leaked rows (run directly against four of them, with a synthetic-fresh positive control returning
  `True`). Zero `dispatch refused: live claimant` lines against a control of 168,458 journal lines.
- **Completion is not blocked** — demonstrated by the 4028 write above.
- **Dependency release is not blocked.** `_deps_satisfied` (`scheduler.py:4319`) takes a
  `status_map: dict[str, str]` projection and cannot read the claimant.

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
3. **Fix the reader asymmetry at the root** — add `is_stranded_any_status` to
   `shared/src/shared/task_claimant.py`, re-express `is_stranded` / `is_stranded_blocked` through it,
   and repoint `_resolve_live_claimant` at it.
4. **Alarm on violation** — a terminal row observed carrying a claimant is a logical error: log a
   structured fact and file one born-at-L2 escalation per episode, never raising.
5. **Repair the existing 105 rows** with a corroborating, staleness-gated operator script.

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
that `harness.py:6112` only re-checks for the *first-class* status.

### D5 — Alarm shape: loud, deduped, never raising

`_resolve_live_claimant` runs once per swept task inside the reconcile sweep. A synchronous `assert`
there would propagate out through `derive_truth` → `recovery_for` → the sweep and collateral-strand
every sibling — the failure class the isolation guard at `scheduler.py:6437-6455` (task 2849) exists
to stop. The alarm therefore: logs `claimant_invariant_violation:` at ERROR with the structured fields
(task id, status, claimant, heartbeat), then files **one** born-at-L2 escalation per episode under a
process-scoped sentinel, entirely inside `try/except Exception`, with an `escalation_queue is None`
early return.

Dedup must use `get_by_task(<sentinel>, status='pending', level=2)`, **not** `has_open_l1` — the
latter is hardcoded `level=1` (`escalation/queue.py:671`) and would never match, a trap already
documented at `merge_queue.py:1100-1104`. A per-task id would file 105 criticals on the first sweep;
the sentinel bounds it to one, with per-task detail in the ERROR log.

### D6 — Repair is a corroborating operator script, not a migration

Not a `_migrate_v4_to_v5` step: `_migrate_v3_to_v4` is self-gating and deliberately leaves
`user_version` at 3 when residual duplicate `candidate_key`s remain
(`sqlite_task_backend.py:68-73`), so a chained v5 step would silently never run on a DB parked at 3.

The script writes via `set_task_claimant`, **never raw SQL** — the backend's claimant writer
(`sqlite_task_backend.py:2265-2268`) deliberately does not bump `updated_at`, and a hand-rolled UPDATE
would reset `updatedAt`-keyed staleness detectors (including
`consume_redispatch_requests.py:465`) for no benefit. It is `--dry-run` by default, gated on
staleness **as well as** status, and re-runs the "affected rows ∩ current lane-holders" intersection
as a pre-flight assertion immediately before applying (INV-3 `corroborate-before-acting`: the empty
intersection measured during authoring is a measurement on a live fleet, not a theorem).

### D7 — Repair must precede the alarm

`task_interceptor.py:1008` short-circuits a same-status write as a no-op, so **no code path can heal
an existing row**. Enabling the alarm before the repair runs would fire the L2 immediately on legacy
residue. The DAG enforces this ordering.

## Contract (B+H)

**Invariant C4-E1.** For any task row: `status ∈ TERMINAL ⇒ claimant_run_id IS NULL AND heartbeat_at
IS NULL`.

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
structurally, never raised, never silently absorbed.

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
| B11 | alarm | violation alarms once | two terminal rows carrying claimants in one sweep | one ERROR per row; exactly **one** L2 escalation filed |
| B12 | alarm | alarm never crashes the sweep | escalation queue raising on `submit` | `derive_truth` still returns a well-formed `TruthReport` (mutation-tested) |

## Pre-conditions for activating

None external. Every substrate capability is present on main @ `7cb0ef2e0c` (G3, verified):

| Capability | Evidence |
|---|---|
| `_CLAIMANT_WIRE_UNSET` sentinel distinguishing unsupplied from explicit-null | `fused-memory/src/fused_memory/server/tools.py:946`, defaults at `:7603-7604`, `:7714-7715` |
| Single enforcing funnel for both status writers | `middleware/task_interceptor.py:871` `_apply_status_transition`, reached from `:833-848` and `:857-867` |
| Sole SQL emitter for the columns | `backends/sqlite_task_backend.py:1922`, tri-state block `:1969-1981` |
| Columns-absent fail-safe | `sqlite_task_backend.py:1970`, pinned by `tests/test_sqlite_task_backend.py:682` |
| `TERMINAL` frozenset | `shared/src/shared/task_statuses.py:61` |
| `set_task_claimant` writer that does not bump `updated_at` | `sqlite_task_backend.py:2265-2268` |
| Born-at-L2 escalation idiom from orchestrator code | `orchestrator/merge_queue.py:1140-1163`, `proc_supervision.py:205-226` |
| `_RECOVERY` row (g) `RE_FILE_ESCALATION` exists to become reachable | `task_ground_truth.py:838` |

No novel substrate is introduced.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-status-authority-prd.md` | extends | C4's "clears at release" half — `claimant_run_id`/`heartbeat_at` lifecycle | **this PRD** (enforcement); C4 remains the origin contract | wired |
| `docs/legibility/design-invariants.md` INV-6 `status-matches-liveness` | complements | this PRD enforces INV-6's converse | design-invariants (rule) / this PRD (this direction's enforcement) | wired |

C4 is amended by reference only: a one-line pointer is added to it so a reader of C4 alone does not
conclude "clears at release" is the whole story. No reciprocal-ownership ambiguity exists.

## Decomposition plan

Phase 1 — foundation. Phase 2 — enforcement slice. Phase 3 — detection. Phase 4 — repair.

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **α** | Add `is_stranded_any_status`; re-express `is_stranded`/`is_stranded_blocked` through it | `shared` | intermediate | Unlocks γ — the status-agnostic liveness predicate γ repoints onto | — |
| **β** | Clear the claimant columns on entry to a terminal status at the fused-memory choke point | `fused-memory` | **leaf** | Through the product's own read path: `set_task_status(id,'done')` on a claimed row, then `get_task(id)` returns `claimant_run_id: null` / `heartbeat_at: null`; a `blocked` write on the same row leaves both intact | — |
| **γ** | Repoint `_resolve_live_claimant` at the status-agnostic predicate; invert the blocked-is-live pin | `orchestrator` | **leaf** | A `blocked` task with a stale claimant now yields recovery action `RE_FILE_ESCALATION` instead of LEAVE, visible in the emitted recovery-disposition event | α |
| **δ** | Clear the claimant before the reconcile revert-to-pending flip | `orchestrator` | **leaf** | After the stranded sweep reverts a task, `get_task` shows `claimant_run_id: null` (today it shows the dead run's id) | — |
| **ε** | Alarm a terminal row carrying a claimant as a logical error (born-at-L2, sentinel-deduped) | `orchestrator` | **leaf** | A seeded violating row makes `get_pending_escalations` return one `level=2`, `severity=critical`, `category=invariant_violation` record; a second violating row in the same sweep adds none | β, ζ |
| **ζ** | `scripts/clear_leaked_claimants.py` — corroborating, staleness-gated repair | `scripts` | **leaf** | `--dry-run` prints the per-status violation census across both project roots; after `--apply`, a re-run prints **zero stale violations** | β, δ |
| **η** | Add the C4 cross-reference pointer | `plans` | **leaf** (non-code) | `plans/task-status-authority-prd.md` C4 names this PRD as the enforcement owner (docs-only) | β |

**η routing note.** η is a genuinely non-code leaf. `planning_mode` bypasses the curator-side routing
guards, so η must self-declare its execution path at filing — `metadata.execution_class` set to the
project's non-code value, so the routing-intent lint does not flag a `task_kind="normal"` leaf whose
own text describes a docs-only change. Its docs-only signal is acceptable *because* the change is
documentation by nature; it is not a code task closing via a docs commit (the shape G2 rejects).

**G2 note.** α is the only intermediate; it names γ as the prerequisite it unlocks. Every other task
carries a signal observable through a product read path (`get_task`, `get_pending_escalations`), a
CLI output difference (ζ), or an emitted event (γ) — none rests on "a unit test passes against
synthetic input".

**G6 note.** ζ's signal asserts a number (zero). Its achievability basis: after β and δ land, the two
paths that mint terminal-row and revert-path violations are closed, so the residual set is exactly
the 105 pre-existing rows the script targets — hence ζ `depends_on` β and δ. The predicate is
deliberately "zero **stale** violations" (D3), because a fresh claimant on a non-terminal row is
legal during the C3.1 window and must not count as failure. ε likewise depends on ζ so the alarm is
not armed against legacy residue (D7).

**G7 walk.** `contracts-machine-checked` — the invariant ships as an executable predicate plus the ε
alarm, not prose. `structured-facts-at-failure` — ε emits task id / status / claimant / heartbeat as
fields, not a log scrape. `corroborate-before-acting` — ζ re-runs the lane-holder intersection as a
pre-flight (D6). `storm-escape-required` — ε's sentinel dedup bounds one L2 per episode.
`no-lockstep-duplication` — D1 reuses `TERMINAL` rather than minting a parallel set; α deletes the
duplicated liveness cores. `status-matches-liveness` — this PRD is its converse.
`holds-owned-and-bounded` — ε's L2 is a human-owned hold with the standard L2 watcher as exit.
`loop-thread-occupancy-bounded` — **flagged**: ε performs a synchronous escalation `submit` (file
I/O) from inside a sweep. Bounded to one write per episode by the sentinel, but ε must confirm the
call site is not on the event-loop thread, or offload if it is. Recorded as ε's first implementation
step, not waived.

## Out of scope

- **Widening enforcement to `pending`/`deferred`** — refuted by D2. If the C3.1 window is ever given
  a compare-and-swap on the claim write, revisit.
- **Making `set_task_claimant` status-aware** (refusing a claimant stamp onto a terminal row). That
  would make the dispatch gate structurally dead and deletable, but costs a `get_task` on the
  lock-free heartbeat hot path. Deliberately not taken: the invariant is enforced at transition time
  and violations are alarmed, not assumed impossible.
- **Deleting the task-2408 dispatch gate.** It is reachable-only-on-violation, not dead, and it is the
  only backstop against dispatching into a live worktree. It is promoted to an alarm site, not removed.
- **Having the heartbeat loop re-stamp `claimant_run_id`.** Only relevant to a `pending` clear, which
  D2 rejects.
- **Backfilling other projects' corpora beyond dark-factory and reify.** ζ takes project roots as
  arguments; running it elsewhere is an operator action.

## Residual after this PRD (accepted, stated)

Because D2 declines to clear `pending` at the choke point, **process death can still leak a
`pending` row**: the dying run never reaches slot release, which is the unconditional clear. This is
the exact mechanism that produced the current 76 pending rows.

δ covers the dominant case — the post-restart reconcile sweep re-pends stranded rows through
`harness.py:6174`, which is measurably the path that minted the 2026-08-19 batch (~35 rows written at
22:15 after the 21:55 fleet redeploy). What remains uncovered is any *other* path that writes
`pending` and is not followed by a slot release. Those are detectable on demand via ζ's hygiene tier
but are deliberately **not** alarmed, because a fresh claimant on a `pending` row is legal during the
C3.1 window and an alarm cannot distinguish the two at write time.

Accepted because the leaked rows are provably inert for scheduling, completion, and dependency
release (measured — see *Current impact*), and because the alternative is the double-dispatch hazard
D2 rejects. If the hygiene tier is observed to re-accumulate materially after δ lands, the follow-up
is a periodic sweep clearing *stale* claimants on non-terminal rows — safe precisely because
staleness excludes the C3.1 window — not a widening of the write rule.

## Open questions (tactical)

1. **ε's sentinel spelling.** A process-scoped constant (e.g. `claimant-invariant-violation`) vs one
   per project id. **Suggested resolution:** process-scoped; per-project only if the fleet-wide single
   record proves hard to attribute. Decide during ε.
2. **Whether ζ should also report the non-terminal stale-claimant tier by default or only under a
   flag.** **Suggested resolution:** report both tiers, repair only what the flags select. Decide
   during ζ.
3. **Exact placement of ε's check** — inside `_resolve_live_claimant` versus in `derive_truth` where
   the row is already fetched. **Suggested resolution:** whichever avoids a second read; confirm
   against the loop-thread finding in the G7 walk. Decide during ε.
