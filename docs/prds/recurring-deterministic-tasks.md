# Recurring deterministic tasks: recurring jobs as first-class citizens of the task system

**Status**: active · authored 2026-08-24 · approach **B+H** (contract + two-way boundary tests)
**Code anchors** verified against main `f73a769923` (2026-08-24). Main moves fast — cite-by-symbol;
re-locate lines at implementation time.
**Origin**: the ε-detector investigation (2026-08-23/24) established that the factory has no
general mechanism for keeping recurring jobs alive, and Leo ratified building one on the
deterministic-task machinery (2026-08-24). Design refuted-and-hardened before authoring; the
substrate facts below were re-verified first-hand on this HEAD.

## Goal

Recurring jobs whose cadence is hours-to-days become **chains of `task_kind='deterministic'`
predicate tasks**: each link is a dated-milestone task; completing a link mints its successor.
Runs, failures, overdue-ness, and broken chains are visible in the task tree, the escalation
queue, and the dashboard — the surfaces with standing readers — instead of in systemd unit
state, which this host demonstrably cannot interpret (the orchestrator-watchdog timer is
disabled unnoticed; 2 of 5 documented nightly jobs are uninstalled unnoticed; the house
`.service` wrappers deliberately avoid `failed` state; nothing scans `--failed`).

A user observes: `get_task` on a completed link's successor shows an advanced fire time and
mint provenance; a failed link is a `blocked` task with a pending born-at-L2 in the deny-listed
`milestone_check_failed` category; the dashboard's invariants endpoint lists every chain with
its state (`scheduled | overdue | paused | broken | ended`); and the two currently-dead nightly
jobs (transcript-check, reify closure-staleness) run again, as the first chains.

## Background — what exists (verified) and what is missing

- **Milestone withholding exists and is validated at submit.** `Scheduler._milestone_time_gated`
  (from `_eligible_for_dispatch`) withholds a `metadata.milestone` `dated` task until its `at`;
  malformed specs fail-safe-withhold with `_note_milestone_malformed` (streak-deduped WARNING).
  Submit-time validation **already exists**: the `Milestone` pydantic submodel
  (`shared/src/shared/task_metadata.py::Milestone`, registered via
  `register_metadata_submodel('milestone', …, cardinality='dict')`) is enforced by
  `deterministic_task_guard.py::_validate_milestone` — malformed specs are rejected with a
  structured ValidationError, never persisted. This PRD **reuses** that pattern for
  `metadata.recurrence`; it does not invent one. (The residual `_note_milestone_malformed` path
  guards legacy/bypass shapes only, e.g. the task-4142 cardinality leak, now symmetric-rejected.)
- **Deterministic predicate machinery exists.** `DeterministicRunner`: predicate rc=0 → `done`
  with `done_provenance.kind='deterministic-milestone'`; rc≠0 →
  `_file_milestone_check_failed_and_block` files born-at-L2
  `(category='milestone_check_failed', agent_role='orchestrator-deterministic',
  severity='critical')` and the task goes `blocked`; **timeout → born-at-L2 `infra_issue`**
  (same role) — the one leg outside the deny-listed category (fixed by r3 below).
  `milestone_check_failed` is in `L2_AUTO_CLOSE_DENY_CATEGORIES` (`escalation/authority.py`),
  and `Harness._recover_stranded_deterministic_gate` (Source A) re-files a lost gate escalation
  byte-identical — the only self-healing alarm-record path in the system. `resume` re-runs the
  predicate (read-only contract). Predicate kind **forbids** `before_done.target_unit` and
  top-level `always_escalates` at submit.
- **The interceptor can mint tasks, but the write lock is NOT re-entrant.** The per-project
  `_write_lock` is a plain `asyncio.Lock`; `_apply_status_transition` holds it for its
  write steps, and the planning-mode submit path takes the same lock around
  `tm.add_task` — calling the interceptor-level submit from inside the transition **deadlocks**.
  The transition's **post-lock seam** (step 5 journal emit → step 6 targeted-reconciliation
  fire-and-forget, which already discriminates `STATUS_TRIGGERS` = done/cancelled) is the
  natural mint site: it re-acquires the lock sequentially and safely.
- **Chain state is queryable.** `_row_to_task` returns `metadata`; the dashboard carries it
  per-row (`data/tasks.py::_shape_task`). The ε-detection PRD
  (`docs/prds/claimant-invariant-detection.md`, task d1) lands the per-root direct-sqlite pool
  path + `/api/v2/dashboard/invariants` endpoint this PRD's chain panel extends.
- **Seed jobs are real.** `scripts/reify-closure-staleness-sweep.sh` exists, executable,
  stdlib-only — but **always exits 0** by design (wrapper-era rationale) and needs a
  predicate-shaped variant (its `$sweep_rc`/`$consumer_rc` are already captured).
  `scripts/legibility-transcript-check@.service` names a runnable command
  (`uv run --frozen --project shared python scripts/legibility/check_transcript_persistence.py
  --project-id %i`) whose unit was never installed. Both are currently delivering zero value.

## Resolved design decisions

### R-D1 — Links are dated-milestone tasks; no new scheduler machinery

A chain link is an ordinary `task_kind='deterministic'`, `before_done.kind='predicate'` task
carrying `metadata.milestone = {mode: 'dated', at: <fire time>}` plus `metadata.recurrence`.
Withholding-until-fire-time is the existing `_milestone_time_gated` gate — zero scheduler
changes (INV-5: reuse over parallel machinery). The one-shot dated milestone (the
confusion-reduction §51 shape) becomes the degenerate case: a chain of one.

### R-D2 — Successor mint-on-terminal, at the interceptor's post-lock seam

The mint fires where **every** writer's terminal transition passes: the fused-memory choke point
— the same all-writers argument that placed β there. A scheduler-side minter would miss
human-driven closes (`resolve_issue`, interactive cancels). Mechanically: a post-lock hook in
`_apply_status_transition`'s step-5/6 seam — after the status write commits, for a task whose
metadata carries `recurrence`: if the new status is `done`, mint exactly one successor via the
backend add path; if `cancelled`, mint nothing (the chain ends). The mint is sequential, not
atomic with the status write — the crash window between them leaves a chain whose latest link is
`done` with no successor, which is exactly the **broken** state R-D5's gauge derives and
surfaces. Mint failure is fail-soft: it never fails the status write, logs a structured ERROR,
and leaves the same gauge-visible broken state (INV-4: the standing gauge enumeration is the
escape, not a silent suppressor).

**Idempotence/dedup**: before minting, the hook checks for an existing non-terminal link with
the same `recurrence.key`; if one exists, it skips (protects against replayed transitions and
concurrent double-writes). Chain serialization falls out: a successor cannot exist before its
predecessor is terminal, so runs never overlap — no lock, no exclusion guard.

### R-D3 — Chain semantics

- `done` → successor minted: same title/description/`before_done`/`recurrence`, `milestone.at`
  advanced to `terminal_time + interval_secs`, `metadata.recurrence.minted_from = <predecessor
  id>`, status `pending` (machine-minted; no planning phase — it is a copy of an
  already-vetted spec, and its provenance says so).
- `cancelled` → chain **ends** (cancel is the operator's stop verb; visible in the tree).
- `blocked` (predicate failed / timed out) → chain **pauses**: no successor until the pending
  escalation is resolved and the link reaches a terminal status. The pending L2 is the standing
  alarm and the hold's named owner (INV-7); resolution → terminal → mint is the re-arm, on
  positive evidence by construction.
- **No catch-up**: a link dispatched late (orchestrator downtime, lock contention) runs once;
  the successor's `at` computes from the terminal time, never from missed slots.

### R-D4 — `metadata.recurrence` shape and validation

`{key: <stable chain id, kebab-case>, interval_secs: int > 0, minted_from: <task id | absent on
the seed link>}`. Delivered as a `Recurrence` pydantic submodel registered via
`register_metadata_submodel('recurrence', …, cardinality='dict')`, key added to
`_BLESSED_METADATA_KEYS`, vocabulary entry in `docs/task-authoring.md`, and a
`deterministic_task_guard` rule (the `_validate_milestone` precedent): malformed shape →
structured ValidationError + hint at submit; **`recurrence` on a non-predicate `before_done`
kind (or on `task_kind != 'deterministic'`) is rejected** — the deploy-kind combination is
deliberately unruled and therefore forbidden until ruled (Out of scope).

### R-D5 — Chain-state gauge: a pure function of task rows

Per `recurrence.key`, over the latest link: `pending` + `at` in the future → **scheduled**;
`pending` + `at` past by more than a grace window → **overdue**; `blocked` → **paused** (with
the pinning escalation id); `done` + no successor → **broken**; `cancelled` → **ended**. Served
on the ε-PRD's `/api/v2/dashboard/invariants` endpoint as a `chains` section (per root), and
read by the watcher rotations' existing once-per-cycle invariants step (extended with the chain
triage rule: `overdue`/`broken` ⇒ finding). The starvation watchdog explicitly does not cover
milestone-withheld tasks (verified on the 3619 specimen) — this gauge is the noticer, not a
hope that existing machinery catches it.

### R-D6 — Failure category unified for carriers

For recurrence-carrying tasks, the deterministic runner's **timeout** leg files
`category='milestone_check_failed'` (today `infra_issue`), matching the predicate-failure leg —
so every failure leg of a recurring job sits in the deny-listed, discriminable category with the
clean closer walk. Scoped to carriers only: deploy-shaped deterministic tasks keep `infra_issue`
timeouts, which `_revalidate_open_deterministic_escalation` (Source B) exists to auto-close —
an unscoped change would blind that closer's designed population. (Note for implementers:
Source A's re-file path builds fields via the `milestone_gate` builder; a lost
`milestone_check_failed` record re-files through the same seam — acceptable, noted, not
changed here.)

## Contract (B+H)

**C-1 (carrier shape).** A recurrence carrier is `task_kind='deterministic'` +
`before_done.kind='predicate'` + valid `metadata.milestone{mode:'dated'}` + valid
`metadata.recurrence{key, interval_secs}`. Anything else carrying `recurrence` is rejected at
submit with a structured error.

**C-2 (mint rule).** On the commit of a carrier's transition to `done`: exactly one successor
exists afterwards for that `key` (minted, or pre-existing non-terminal → skip). On `cancelled`:
none is minted. On any non-terminal status: none. The mint never fails the status write; a mint
failure logs structured ERROR and leaves the broken state visible per C-4.

**C-3 (successor shape).** Copies title/description/`before_done`/`recurrence.{key,
interval_secs}`; `milestone.at = terminal_time + interval_secs`; `recurrence.minted_from =
predecessor id`; `status='pending'`; carries `metadata.source='recurrence-mint'`.

**C-4 (gauge).** The chain-state function of R-D5 is total over every task carrying
`recurrence.key`: every chain resolves to exactly one of
`scheduled | overdue | paused | broken | ended`, and the endpoint enumerates all of them with
`measured_at`. `broken` and `overdue` are the alarm-shaped states.

**C-5 (category).** Every escalation a carrier's deterministic run files carries
`category='milestone_check_failed'`, on both the predicate-failure and timeout legs.

**C-6 (no catch-up).** Successor `at` derives from terminal time only. Missed windows collapse
to one late run.

## Boundary-test sketch (B+H)

| # | Side | Scenario | Preconditions | Postconditions |
|---|---|---|---|---|
| B1 | mint | done mints one successor | carrier link flips `done` via the interceptor | `get_task` finds exactly one new `pending` task with same `key`, `at = terminal_time + interval_secs`, `minted_from` set |
| B2 | mint | cancelled ends the chain | carrier link flips `cancelled` | no successor; gauge reads `ended` |
| B3 | mint | blocked pauses | carrier predicate fails (rc≠0) | task `blocked`, `milestone_check_failed` L2 pending, no successor; gauge reads `paused` naming the escalation |
| B4 | mint | replay-safe | `done` transition replayed / double-written while a non-terminal successor exists | still exactly one non-terminal link for the key |
| B5 | mint | fail-soft | mint path fault-injected after a `done` write | status write intact; structured ERROR; gauge reads `broken` |
| B6 | validation | malformed recurrence rejected | `submit_task` with `interval_secs: 0` / missing `key` / `recurrence` on a deploy-kind task | structured ValidationError + hint; nothing persisted |
| B7 | category | carrier timeout deny-listed | carrier predicate exceeds `timeout_secs` | pending L2 has `category='milestone_check_failed'` (today: `infra_issue`) |
| B8 | gauge | overdue visible | carrier link `pending` with `at` past the grace window (seeded) | endpoint reads `overdue` for that key |
| B9 | gauge | human close continues the chain | a human resolves B3's escalation and the link reaches `done` | successor minted — the mint site sees non-orchestrator writers |
| B10 | e2e | seed chains live | r6's two seeded chains dispatched once | each shows link-1 `done` with `done_provenance.kind='deterministic-milestone'` and a minted link-2; gauge lists both `scheduled` |

## Pre-conditions for activating

- **Task 4619 (claimant-enforcement β)** edits `_apply_status_transition`'s terminal path — r2
  serializes behind it (same-region edit ordering, not a semantic dependency).
- **ε-detection d1** (`docs/prds/claimant-invariant-detection.md`) delivers the invariants
  endpoint + per-root tasks.db pool path r4 extends, and d2 the checklist step r5 extends.
- All other substrate verified present on `f73a769923` (Background).

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/claimant-invariant-detection.md` | consumes (endpoint + pool path + checklist step) | `/api/v2/dashboard/invariants` · watcher main-loop step | detection PRD produces; **this PRD owns its chain extensions (r4, r5)** | wired (r4→d1, r5→d2 deps) |
| `docs/prds/claimant-invariant-enforcement.md` | edit-ordering only | `TaskInterceptor._apply_status_transition` region | enforcement β owns its edit; r2 serializes behind 4619 | wired (r2→4619) |
| `plans/capability-delivered-checks-prd.md` | consumes (the `before_done` machinery it landed) | `DeterministicRunner` | that PRD (landed) | wired |
| confusion-reduction PRD §51 | supersedes-as-generalization (one-shot delayed milestone = chain of one) | `metadata.milestone` | milestone machinery unchanged; no edit needed | none |

## Decomposition plan

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **r1** | `Recurrence` metadata submodel + blessing + guard rules (shape; predicate-kind-only; deploy-kind rejection) | `shared`, `fused-memory` | intermediate (unlocks r2, r3, r4) | Unlocks the mint, the category scoping, and the gauge; directly: `submit_task` with a malformed `recurrence` (B6 shapes) returns the structured ValidationError + hint where today the key merely warns `unknown_key` | — |
| **r2** | Interceptor mint-on-terminal at the post-lock seam (dedup, fail-soft ERROR, provenance, `at` advance) | `fused-memory` | intermediate (unlocks r6) | After flipping a seeded carrier `done` through `set_task_status`, `get_task` finds the minted successor per C-3 (B1); `cancelled` on a sibling seeded carrier mints nothing (B2) | r1, 4619 |
| **r3** | Carrier timeout-leg category → `milestone_check_failed` | `orchestrator` | **leaf** | A seeded carrier whose predicate sleeps past a small `timeout_secs`: `get_task_escalations` shows the pending L2 with `category='milestone_check_failed'`, where today's identical setup yields `infra_issue` (B7) | r1 |
| **r4** | Dashboard chain panel on the invariants endpoint (R-D5 state function) | `dashboard` | **leaf** | The endpoint's `chains` section lists a seeded chain and walks the states: future-dated `scheduled`, past-dated `overdue`, done-without-successor `broken` — each rendered distinctly, never absent (B8, B5's read side) | r1, **d1** (detection PRD) |
| **r5** | Watcher checklist chain-triage extension | `skills` | **leaf** (non-code) | `git grep -n 'overdue\|broken' -- skills/escalation-watcher/SKILL.md skills/recon-escalation-watcher/SKILL.md` shows the chain-triage rule added to the invariants step in both files | r4, **d2** (detection PRD) |
| **r6** | Seed the first two chains: predicate variant of the reify closure-staleness sweep + transcript-check; demonstrate link→successor end-to-end | `scripts`, task filing | **leaf** (integration gate) | B10 through the product read path: `get_task` shows link-1 `done` with `done_provenance.kind='deterministic-milestone'` and link-2 minted with advanced `at` + `minted_from`; the chain panel lists both chains | r2, r3 |

**Routing notes.** r5 is a genuinely non-code leaf: `metadata.execution_class='operational'`.
r6 is `task_kind='normal'` (it writes the predicate-variant script and files carriers); the
carriers it files are themselves `task_kind='deterministic'`.

**G1.** The mechanism's consumers are real and in-batch: r6 converts two currently-dead
documented jobs into the first chains (the integration gate), and the ε-detection PRD's ε.2
census is a named future consumer (explicitly not built until its trigger fires). A third
migration candidate (the 6-hourly `cleanup_test_collections` cron) is a follow-up, not queued.

**G2.** r1/r2 are intermediates naming their unlocks; every leaf signal goes through
`submit_task` errors, `get_task`/`get_task_escalations`, the endpoint, or `git grep` — none
rests on a synthetic-input unit test. Seeded carriers are the demonstration vehicle (the
enforcement PRD's γ precedent); r6 additionally demonstrates against real jobs.

**G6.** r3's signal asserts a category *difference* whose today-value (`infra_issue` on the
timeout leg) is verified on this HEAD; B7 names both sides. r1's signal asserts a
rejection whose mechanism the task builds and the boundary tests observe firing (G6 branch 4).
No signal asserts a corpus count.

**G7 walk.** INV-1: the recurrence contract ships as a registered submodel + submit-time guard
(machine-checked), not prose. INV-2: mint failures and the gauge carry structured facts
(`measured_at`, escalation ids). INV-3: the mint acts on the transition it itself just
committed and re-checks for an existing non-terminal link before writing (B4). INV-4: the
fail-soft mint path's escape is the standing broken/overdue enumeration on the gauge plus the
watcher triage rule — persistent visibility, not a suppressor. INV-5: reuses milestone
withholding, the milestone category, the submodel-registration pattern, and the ε-PRD endpoint;
successors copy *data*, not logic. INV-6: no claimed-status semantics touched. INV-7: every
held state names its owner and bound — scheduled (scheduler gate, bound = `at`), paused
(pending L2 + watcher, surfaced with age), overdue/broken (gauge + watcher). INV-8: the mint
is one bounded backend call per carrier terminal transition on the existing offloaded write
path; the gauge computation is per-root, ms-scale, in the dashboard.

## Out of scope

- **Sub-minute jobs** (dashboard-watchdog, load-sampler) — wrong tool; they stay timers.
- **The self-referential watchdogs** — a job whose subject is the factory core's own liveness
  cannot be a task inside it. This PRD makes the orchestrator-watchdog *more* load-bearing;
  re-arming it is an operator action this PRD motivates but does not own.
- **Catch-up/backfill semantics** — deliberately absent (C-6).
- **Calendar-aligned cadence** (`at`-anchored, cron-like) — v1 is interval-from-terminal;
  drift-accepting. Follow-up if drift bites a real chain.
- **Recurrence on deploy-kind deterministic tasks or on `code_tdd` tasks** — rejected at submit
  until separately ruled.
- **Retention policy for accumulated done links** (~365 rows/year per daily chain) — accepted
  for now; revisit if tree size bites.
- **Migrating the remaining timer jobs** (flag-marker sweep, reclaim-orphaned-worktrees, the
  cron cleanup) — follow-ups once r6's chains have soaked.

## Open questions (tactical)

1. **Overdue grace window size** (how far past `at` before `overdue`). Suggested: one full
   `interval_secs`, capped at 24h. Decide during r4.
2. **Whether the minted successor copies `metadata.files`** (lock-set inheritance). Suggested:
   yes, verbatim — same job, same locks. Decide during r2.
3. **Seed cadences for r6's two chains.** Suggested: 24h both, matching their former timer
   slots. Decide during r6.
