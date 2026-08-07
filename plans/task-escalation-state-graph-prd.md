# PRD — Task/escalation state-graph remediation: no unowned status, no unbounded hold

**Status:** active — authored 2026-08-02 (autonomous session, brief
`2026-08-02-task-escalation-state-graph-spec`; Leo AFK — safe defaults
recorded inline, tactical residue in §Open questions).
**Normative source:** `docs/task-escalation-state-spec.md` (the functional
spec this PRD implements; its §8 divergence register E1-E15 is the defect
list, referenced throughout). Invariants: INV-6 `status-matches-liveness`,
INV-7 `holds-owned-and-bounded` (`docs/legibility/design-invariants.md`).
**Approach:** **B + H** (blast radius 5 packages — orchestrator, shared,
escalation, fused-memory-adjacent surfaces, dashboard; load-bearing
dispatch/status semantics of a running fleet; ≥8 mechanisms). Contract +
boundary tests below.
**Extends/completes:** `plans/task-status-authority-prd.md` (W2 — goal 5/B2
ratified-but-unbuilt; D6/D8 deploy doctrine), `plans/workflow-state-machine-prd.md`
(W9 — SM-2), `plans/harness-supervision-prd.md` (W10 — `_RECOVERY`),
`plans/escalation-repend-state-machine-prd.md` (D4 gate semantics kept).
**Write-tag:** `agent_id="claude-state-graph-spec"`.

## Goal (user-observable behaviour)

1. **No task strands.** An orchestrator whose merge entry meets a gating
   escalation parks the task visibly (`blocked`, record attached) or
   resolves it in-slot through the bounded steward wait — never an
   `in-progress` row with a NULL claimant that nothing will ever
   redispatch. The 9 dark-factory + 3 reify live strands recover through
   the landed mechanisms (observed via normal read paths) without any
   manual status surgery.
2. **Every hold is explained.** A sweep that declines to recover a task
   says so — a structured event naming the pinning escalation ids and
   ages — and repeated identical holds escalate instead of repeating
   silently forever.
3. **Recovery discriminates.** A dead-steward L0 no longer pins its own
   task's recovery; queue-backed L1/L2 handoffs still hold, visibly,
   as `blocked`. `resolve_issue(resume)` on a stranded row re-pends it
   (W2 B2, finally). The L1-only `_already_landed_dispatch_gate` no
   longer phantom-dones a task past an open L0/L2.
4. **Counts stop lying.** Burndown/overview separate live from stranded
   in-progress; an `in_progress` census above `max_concurrent_tasks`
   alarms instead of silently rendering 33-of-24; dashboards mark
   stranded rows and escalations that pin recovery.
5. **The exit contract is exact.** `run()` cannot exit leaving a status
   whose implied owner is gone (INV-6); `_OUTCOME_ALLOWED` shrinks to
   the proven pairs; SM-2 violations escalate loudly instead of
   degrading to a synthetic mislabeled report.

## Background

The full mechanism analysis, live measurements, archaeology (Leo's
merge-throughput hypothesis REFUTED — the strand is emergent from four
locally-sound changes, none of which had a post-bail owner), and the
demand inventory are in the spec (`docs/task-escalation-state-spec.md`
§1-§3, §8) and are not restated here (INV-5). Key structural facts this
PRD builds on: dispatch is pending-only; the harness slot-`finally`
nulls the claimant on every outcome; the recovery veto is
`bool(open_escalations)` at any level in five hand-rolled copies;
resolution events wake only `blocked`/`infra-hold` rows; the L0 startup
amnesty plus the orphan-L0 reaper convert an aged strand into a
restart-proof strand.

## Substrate reality check (G3) — verified against working tree `c26d8dd6fa`

| Assumed capability | Status | Evidence |
|---|---|---|
| Claimant columns + `is_stranded` oracle | exists | `shared/src/shared/task_claimant.py:106-141`; columns per W2 D4 |
| `_RECOVERY` classification table + applier | exists | `orchestrator/src/orchestrator/task_ground_truth.py:533-641`; applier `harness.py:5087+` |
| Bounded steward-wait machinery (idle window, progress-refresh, L2 short-circuit) | exists | `workflow.py:12060-12343` (task 3170); `_StewardReescalated` at `:12176-12181` |
| Table B action-effects authority consumed by server + harness | exists | `escalation/action_effects.py:107-113`; `harness.py:11982-11988`, `:12889-12891` |
| Structured event-store emission pattern | exists | `harness.py:5503-5514` event usage; `orchestrator/event_store.py` |
| Escalation queue task-scoped reads with filters | exists | `escalation/queue.py` `get_by_task(status=, level=, agent_role=)` |
| `EscalationRef` carries severity | **absent — additive prerequisite (α)** | `task_ground_truth.py:134` has id/level/category only; `:478-481` drops severity at construction |
| Store-side transition enforcement live | exists (context for deploy risk) | `fused-memory/config/config.yaml:101-102` `enforce_transitions: true` (flipped 2026-07-14, esc-2216-1) |
| Dashboard data layer carries task rows / escalation analytics | exists | `dashboard/src/dashboard/data/{tasks,active_tasks,burndown,escalation_analytics}.py` |
| Orchestrator config green-tier hot-reload for sweep/watcher knobs | exists | `mcp__escalation__reload_config`; OPERATIONS.md §"Config reload vs restart" |

No other novel substrate. G3 passes with α as the one filed prerequisite.

## Resolved design decisions

**D1 — Path B unifies with Path A; the gate itself is untouched.** At
MERGE entry with gating escalations open, the workflow enters the same
ESCALATED machinery as every other phase: steward started, bounded
`_wait_for_resolution` (existing idle window; existing immediate
`_StewardReescalated` short-circuit for critical/urgent/L≥2 — a human
gate is never waited on in-slot), exit either resolution→merge-retry or
`_mark_blocked`→`blocked`. Rationale: 7/9 live strands are the task's own
just-filed L0 — in-slot consumption converts a full dispatch-burn loop
(3429's measured 15+ repeat escalations) into one bounded wait; and
uniformity is the point (one ESCALATED semantics, not two). Repend-PRD D4
stop-the-line semantics preserved exactly (nothing merges past an open
gating record).

**D2 — Strand conversion, not revert and not a new status.** The
stranded sweep's missing shapes get explicit `_RECOVERY` rows:
`(in-progress, no live claimant, open-escalation)` →
`CONVERT_TO_BLOCKED` for **all four** branch states (spec §7.2 —
including ON_MAIN/GONE_WITH_MERGE_MARKER: a landed-but-pinned row
converts, then completes via record-close → redispatch → the any-level
dispatch gate). Conversion mechanics (attack-hardened): **clear the
claimant before the status write** (every liveness oracle must agree);
**CAS re-read** of (status, claimant, record-open) immediately before
writing, falling to the matching other branch on mismatch; writes
`blocked` via the applier (a legal `in-progress→blocked` Table A edge),
stamps `metadata.strand_converted = {at, pinning_escalation_ids,
shape}`, files **no** new escalation (dedup — the open record is the
ownership token), preserves WIP (workflow-preserve status; lane policy
untouched). Anti-churn integration (spec §6 demand-12): the conversion
writer is **excluded from the park-and-stop window** (a backlog
conversion must not trip a false storm and pause the scheduler) and
carries its own INV-4 streak counter; the **sweep-driven**
blocked→pending flip charges the same signature-keyed reblock counter
the cascade flip charges (else strand→convert→L0-amnesty→flip→re-strand
cycles at the 8h fleet period with every cap reading zero). The
converted row rides the *existing* blocked machinery: resolution-event
wake, blocked-redispatch after close, `/unblock`, b3 (which correctly
aborts without a proposal — expected, and the escalation-watcher
playbook gains an explicit `strand_converted` row so the L1 rotation
has a disposition). Rejected alternatives: a `stranded` status
(needs its own consumer loop + joins the sweep-carve-out treadmill —
spec §9), bare revert-to-pending (second-guesses live handoffs, fights
the reblock guard, re-burns dispatch into the same gate).

**D3 — The pin predicate becomes one shared, severity-aware function.**
`escalation`-package helper (single home; importable by orchestrator)
classifying a task's open records into `{DEAD_L0, QUEUE_HANDOFF,
NON_PINNING}`: `info`-severity records never pin; an L0 with no live
workflow/steward for the task is DEAD (evidence: the claimant columns —
corroborated at use per INV-3); L1/L2 are QUEUE_HANDOFF. All five veto
copies (resolver `_shape`, harness `:5560`, scheduler `:5719`, harness
`:11759`+`:11373` — the last pair de-duplicated to one) consume it;
`EscalationRef` gains `severity` (α) so the resolver can carry it.
Done-flip vetoes stay maximally conservative: **any** non-info open
record still vetoes MARK_DONE (phantom-done protection is the half of
the veto that was always right). The deliberately-different predicates
stay separate and documented: the archive-inclusive deterministic gate
check (`:11773`) and `_is_gating_escalation` (producer-side).

**D4 — `_already_landed_dispatch_gate` goes any-level immediately.**
It is the drifted inverse hazard (phantom-done past an open L0/L2,
contradicting resolver row (f) policy on the same task). Small,
self-contained, lands first in the harness spine (η0).

**D5 — Emission before behavior.** Structured
`recovery_vetoed`/`recovery_left` events (task, shape, pinning ids,
ages) + sweep summaries that count holds land **before** any
disposition change, so the soak for D2 is observable and the live
strands generate evidence rows on the very next sweep — with zero
behavior change. Streak: N consecutive identical vetoes on one task →
one dedup-guarded L1 (INV-4; reuses the stall-detector pattern).

**D6 — Truthful exits at every producer** (spec §5): merge-phase
`_mark_blocked` writes its park status (after a dispatch-time
investigation of the carve-out's original rationale — Chesterton's
fence, recorded in the task); the merge-halt trio stops waiting
unbounded in-slot and instead escalates + `_mark_blocked`s — **under the
spec §7.9 halt-token constraints**: the rewrite must NOT run the wait's
existing cancellation cleanup (which unhalts when the waiter owns the
halt — a naive bound would unhalt the queue over a dirty tree); the
sole unhalt edge stays the durable record's resolution; halt
rehydration's `level == 1` filter widens to category-at-level-≥1 in the
same change (else a promoted halt-owner record silently un-halts at the
next restart); no sibling halt-category re-files. `WarmLaneRequeue` and
the soft-cancel spurious-wakeup fallback write `pending` before exit;
the two DONE-on-cancelled producers return CANCELLED; infra-hold resume
stops manufacturing the claimant-less strand shape — mechanism decided
at the task with the no-recompete property named as a hard requirement
(the in-progress hack existed to avoid the 3465 footprint
re-competition): either claim-then-status (resume stamps a claimant
before `infra-hold→in-progress`) or `pending` + a verify-only dispatch
fast-path exempt from implement-footprint locks. Cascade/Table-B status
writes join the loud-failure rule: a swallowed `SetTaskStatusRejected`
is a permanent silent hold, so rejection → retry-then-escalate, never
log-and-return. Each is a small, testable, per-site change on the
workflow.py / harness.py spines.

**D7 — Table tightening and loud SM-2 land dark, flip after soak** (W2
D6 precedent). `_OUTCOME_ALLOWED` shrinks to the proven pairs (spec §5)
and SM-2 violations escalate (structured record, not a synthetic
BLOCKED) — both behind a config gate defaulting to log-only
(`would-violate` WARNING + event). Flip is an operator gate after the
soak shows zero would-violations. Narrowing `escalated` →
`{BLOCKED}` is a Table A contract change owned here, with W9 test
updates in the same task (binding: W2 owns the table, W9 owns the
machine — this PRD edits the table *content* and the machine's tests,
not the machine).

**D8 — `resume` acts on claimant-liveness** (completes W2 goal 5/B2,
explicitly overturning the two codifications that pinned it:
`status == 'blocked'` at `harness.py:12873` and the `level >= 1` gate at
`harness.py:12014` for orphaned rows). Semantics: resume on a row with
no live claimant → `pending` (Table B target), regardless of
in-progress/blocked; resume with a live claimant → the existing live-
workflow wake. `granted_files` scope grants deliver on the re-pend
path. With D2 landed the stranded-in-progress class shrinks to the
race window, but the resolution path must still be correct for it.

**D9 — Observability is projection, not new measurement** (every datum
already exists in memory at its moment of need — research finding).
Burndown splits `in_progress` into live/stranded via claimant fields;
an `in_progress rows > max_concurrent_tasks` parity alarm; dashboard
task rows stop dropping claimant fields and render a `stranded` badge;
`get_pending_escalations` gains a computed `pins_recovery: [task_ids]`
annotation and the escalations tab a PINNING chip. No schema changes.

**D10 — Deploy per W2 D8.** Code lands dormant/log-mode; activation is
a deterministic pure operator gate (fleet restart runbook,
out-of-cgroup, never `--drain`); enforcement flips are a second
operator gate after the soak; the live strands are expected to convert
on the first post-flip sweep — the deploy gate's runbook says to verify
exactly that via `get_tasks` + the new events, not to hand-fix them.

**D11 — Adjacent tasks reconciled, not duplicated.** 3429 (tripwire
gating decision): this PRD resolves the contradiction as "the gate is
real and stays; the comments lied" — γ1 fixes the `workflow.py:3438`
and `_check_scope_invariant` docstrings in-spine — and 3429 is amended
in place (per house norm: update, don't cancel-refile) to its remaining
live question: option (c), why plan.files/metadata.files diverge at all,
now wired as a dependent of γ1. 3423: its strand-fix half is subsumed
(γ1 delivers the exact test its metadata promised); its rewritten
description (reify-5879 flap forensics) stands as its own
investigation — metadata.files updated to drop the strand-test files so
the lock footprint matches the real remaining ask.

## Contract (B+H §1)

- `escalation.pins.classify_pins(task_id, records, *, live_claimant: bool) ->
  PinReport(dead_l0: list, queue_handoff: list, non_pinning: list)` — pure;
  severity/level-aware; consumed by every recovery/redispatch veto site.
  Escalation references gain the **filing claimant/session identity**
  alongside severity: "live handoff" means the *filing* incarnation
  lives (spec S6 — a newer incarnation never keeps a prior
  incarnation's unconsumed L0 alive; the orphan-L0 reaper promotes on
  filing-incarnation death). Missing/out-of-vocabulary severity fails
  safe to PIN.
  **Store correctness (spec S6, esc-3163 lesson):** callers bind the read
  to the task's owning orchestrator's escalation store; queue-absent /
  read-failure is a distinguishable third state mapped to LEAVE +
  `recovery_left(reason=escalation_store_unavailable)` — never collapsed
  into "no records" (which would route a pinned strand into plain
  revert). The scheduler sweep's existing `escalation_queue is None ⇒
  never flip` guard is the model.
- `RecoveryAction.CONVERT_TO_BLOCKED` — new `_RECOVERY` action; applier
  writes `blocked` + `metadata.strand_converted`; never files an
  escalation; never touches `merge-deferred`/`infra-hold`/deterministic
  rows (their carve-outs precede the shape lookup, unchanged).
- Recovery emission: `EventType.recovery_vetoed` / `recovery_left` /
  `strand_converted` with `{task_id, shape, escalation_ids, ages_secs,
  measured_at}`; sweep summary gains `held(escalation)` / `left(shape)`
  counters and logs even when all-zero action was taken. The conversion
  write carries an attributable `agent_id` (D5 actor doctrine) so Table
  A's actor dimension and the structured fact both name the recovery
  actor.
- Exit contract: `run()` may exit only with the §5 (spec) outcome→status
  pairs; SM-2 violation → `escalate_info(category='workflow_exit_contract',
  severity='blocking', level=1)` + event, task status untouched (gated
  log-only until the flip).
- Ordering invariants: status write precedes claimant-null at every
  designed exit (S1); conversion is idempotent (re-running on an
  already-converted row is a no-op); conversion clears the claimant
  before the status write and CAS-re-reads (status, claimant,
  record-open) immediately before writing, falling to the matching
  other branch on mismatch (INV-3); startup ordering pinned: L0 amnesty
  → halt rehydration → stranded reconcile (spec §7.7).
- Orphaned-park invariant (spec §7.6): a `blocked`/`infra-hold` row with
  no open pinning record, no gate marker, and no live claimant is
  re-owned by the sweep (re-pend or re-file row) — the
  deterministic-gate archive-inclusive carve-out stays the documented
  operator-owned exception.
- §5 exit-contract relaxations (spec): observed-terminal reported as
  that terminal; status-write failure at exit reclassifies as
  crash-shaped + ONE structured store-unavailable record.

## Boundary-test sketch (B+H §2) — the ω integration-gate signal

| # | Scenario | Pre | Post |
|---|---|---|---|
| 1 | Path B bail enters bounded wait | gating L0 open at MERGE entry (tripwire shape) | steward consumes L0 → merge retries in-slot; task never exits `in-progress` without claimant; no new dispatch consumed |
| 2 | Path B with L2 open | born-at-L2 record open at MERGE entry | immediate `_StewardReescalated` → `_mark_blocked` → row `blocked`, L2 untouched, slot freed |
| 3 | Steward-wait expiry at merge entry | L0 never consumed within idle window | orphan L0s dismissed per existing machinery → `blocked` with L1; no strand |
| 4 | Conversion of a crash strand with open L1 | in-progress, NULL claimant, branch off-main, L1 open | sweep converts → `blocked`; `strand_converted` event names the L1; no new escalation; WIP branch untouched |
| 5 | Conversion race vs live claimant | claimant heartbeat fresh | applier aborts conversion (corroboration), `recovery_left` emitted |
| 6 | Converted row exits via existing machinery | row from #4; L1 resolved `resume` | cascade re-pends (claimant-liveness rule); redispatch adopts WIP branch |
| 7 | Dead-L0 pin does not veto | in-progress strand, only record = own L0, steward dead | classifier: DEAD_L0 → conversion proceeds (not LEAVE) |
| 8 | Info record never pins | strand with only `severity='info'` record | classified NON_PINNING; plain revert-to-pending path taken |
| 9 | Done-flip veto stays conservative | strand ON_MAIN with any non-info open record | MARK_DONE still vetoed; `recovery_vetoed` emitted with ids |
| 10 | Dispatch gate any-level | pending task + landing evidence + open L2 only | no auto-flip to done; veto logged with the L2 id |
| 11 | resume on stranded row | in-progress NULL-claimant row; its L1 resolved `resume` | row → `pending`; `granted_files` folded; no DEBUG-skip |
| 12 | REQUEUED writes pending | WarmLaneRequeue path | row `pending` at exit; no sweep dependency |
| 13 | Merge-halt escalate-and-block | stash-fail shape | L1 filed, halt asserted, row `blocked`, slot freed; restart re-asserts halt from the L1; resolution → unhalt + normal blocked exit |
| 14 | DONE-on-cancelled | soft-cancel lands on cancelled row | outcome CANCELLED, tallies correct |
| 15 | SM-2 loud | forced illegal exit pair in log-mode | `would-violate` WARNING + event, no synthetic BLOCKED mislabel; enforce-mode: escalation filed |
| 16 | Table tightened | after producer fixes | `outcome_allows_status('escalated', IN_PROGRESS)` False; W9 suite green |
| 17 | Streak escalation | same task vetoed N consecutive sweeps | one dedup-guarded L1 filed; no storm |
| 18 | Parity alarm | in_progress rows > cap in test fixture | burndown snapshot flags divergence; dashboard renders stranded badge |
| 19 | Mass conversion ≠ park-and-stop trip | 20 strand fixtures converted in one sweep | scheduler NOT paused (conversion writer excluded from the window); conversion streak counter incremented |
| 20 | Amnesty-flip loop is charged | strand→convert→L0-amnesty→sweep-flip cycle repeated | sweep flip charges the signature-keyed counter; at threshold: withhold + L2 — no silent 8h-period churn |
| 21 | Landed-but-pinned converts and completes | branch ON_MAIN, dead-L0 open, NULL claimant | converts → record closes → redispatch → any-level dispatch gate completes with provenance |
| 22 | Orphaned park re-owned | infra-hold row, no open record, no claimant (crashed cascade) | sweep re-owns (re-pend/re-file) + structured fact; deterministic archive-inclusive carve-out untouched |
| 23 | Conversion CAS race | pinning record resolved between sweep snapshot and write | falls to the record-free branch; no blocked-row-with-no-record is ever written |
| 24 | Halt token survives the rewrite | halt-owner record promoted L1→L2; orchestrator restarts | rehydration (category, level ≥ 1) re-asserts the halt; no unhalt fires on the wait's exit path |

## Cross-PRD relationship (G4)

| Other PRD / artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task-status-authority (W2) | extends | Table A content (`_OUTCOME_ALLOWED` narrowing); B2/goal-5 semantics; D6/D8 deploy doctrine | **this PRD** (W2 is landed; its own PRD stays historical record) | queued here (θ, ζ) |
| workflow-state-machine (W9) | edits consumer tests | SM-2 loudness + narrowed-table test updates | **this PRD** (machine structure untouched) | queued here (θ) |
| harness-supervision (W10) | extends | new `_RECOVERY` rows + shared pin classifier consumed by the sweeps (TG-2 preserved: one table, appliers dispatch on action) | **this PRD** | queued here (δ, η) |
| escalation-repend (D4 gate) | conforms | gate predicate untouched; post-bail disposition added | **this PRD** | γ1 |
| stranding-remediation-scheduler-ergonomics α | benefits | converted rows feed the existing verified-green fast path | no edit either side | n/a |
| b3-low-risk-auto-unblock | unaffected | converted rows carry no dry_run_proposal → b3 aborts by design | no edit | n/a |
| escalation-lifecycle-dashboard | extends | `pins_recovery` annotation + PINNING chip | **this PRD** (ι) | queued here |
| tasks 3429 / 3423 | reconciles | tripwire decision + strand-test ownership | **this PRD** per D11 | amended at queue time |

No reciprocal-ownership ambiguity: every seam lands in this PRD's batch.

## Decomposition plan

File-lock spines: `workflow.py` linear (γ1→γ2→γ3→θw); `harness.py` linear
(η0→β→δ→ζ→η); independent lanes: α (escalation+TG), ι (dashboard+server),
κ (docs). All code leaves `task_kind="normal"`, `force_full_path=true`
on god-file (workflow.py/harness.py) tasks; gates deterministic.

- **α — Widen `EscalationRef` with severity + shared pin classifier**
  (`escalation/src/escalation/pins.py` new, `task_ground_truth.py`).
  Intermediate; unlocks β/δ/η. Signal for its own tests: classifier
  verdicts on the three record classes incl. dead-L0 evidence rule.
- **η0 — `_already_landed_dispatch_gate` any-level** (`harness.py`).
  Leaf. Signal: boundary #10 — a pending task with only an open L0/L2 is
  not auto-flipped; veto logged with id. High priority (live
  phantom-done hazard).
- **β — Structured recovery emission + streak escalation** (`harness.py`,
  `scheduler.py`, `task_ground_truth.py`). Prereq α, η0. Signal:
  boundary #9/#17 in tests; operationally, the first post-deploy sweep
  emits `recovery_vetoed` rows naming each live strand's pinning ids
  (observed via event store / logs) with zero behavior change.
- **γ1 — Path B: merge-entry bail enters bounded escalated-wait; tripwire
  comments fixed** (`workflow.py`,
  `orchestrator/tests/test_workflow_merge_gating_strand.py` new).
  Prereq: none in-batch (Path A machinery exists). Signal: boundary
  #1/#2/#3; the test file 3423 promised finally exists and pins the
  no-strand exit property.
- **γ2 — Merge-phase `_mark_blocked` carve-out + merge-halt trio
  escalate-and-block under the §7.9 halt-token constraints**
  (`workflow.py`, `harness.py` rehydration filter). Prereq γ1 (spine).
  Signal: boundary #13/#24; every BLOCKED exit writes a status; the
  wait's exit path provably never unhalts (test pins it); rehydration
  matches category at level ≥ 1.
- **γ3 — Truthful REQUEUED/CANCELLED/infra exits** (`workflow.py`,
  `harness.py:12846` infra-resume write). Prereq γ2. Signal: boundary
  #12/#14; infra resume never leaves a claimant-less `in-progress` row
  (test pins no-manufactured-strand) while preserving the no-recompete
  property (D6 mechanism decision recorded in-task); cascade status-write
  rejection escalates instead of log-and-return.
- **δ — `CONVERT_TO_BLOCKED` rows + applier (log-mode)**
  (`task_ground_truth.py`, `harness.py`; watcher playbook row for
  `strand_converted`). Prereq α, β. All four branch states; claimant
  cleared pre-write; CAS; park-stop exclusion; sweep-flip counter
  charge; orphaned-park re-owning invariant (spec §7.6). Signal:
  boundary #4/#5/#7/#8/#19/#20/#21/#22/#23 in tests; operationally
  `would-convert` lines on the live strand fixtures in log-mode.
- **ζ — resume acts on claimant-liveness (B2)** (`harness.py`,
  `escalation/action_effects.py` docstring). Prereq δ (spine). Signal:
  boundary #6/#11.
- **η — Veto-site collapse onto the classifier + reaper
  filing-incarnation rule** (`harness.py`, `scheduler.py`). Prereq ζ
  (spine), α. Signal: one predicate consumed at all sites
  (grep-provable); scheduler/harness blocked-recovery relaxation
  unified; duplicate `:11759/:11373` check deleted; orphan-L0 reaper
  promotes on filing-incarnation death (a newer live workflow no longer
  defers a prior incarnation's unconsumed L0).
- **θ — Tighten `_OUTCOME_ALLOWED` + loud SM-2 (log-mode)**
  (`shared/src/shared/task_transitions.py`,
  `shared/tests/test_task_transitions.py`, `workflow.py`,
  `orchestrator/tests/test_workflow_state_machine_boundary.py` +
  `test_workflow_state_machine.py` — the W9 SM-2 property suite consults
  this map and MUST be updated in the same leaf). Prereq γ3 (producers
  fixed first), δ. Signal: boundary #15/#16 with the enforce flag off;
  zero `would-violate` under the full orchestrator suite.
- **ι — Observability projection** (`dashboard/src/…/{burndown,tasks,
  active_tasks,escalation_analytics}.py`, jsx tabs,
  `escalation/server.py` `pins_recovery`). Prereq α (classifier for the
  annotation). Signal: boundary #18; dashboard shows stranded badge +
  PINNING chip; burndown schema carries live/stranded split.
- **κ — Docs alignment** (`ARCHITECTURE.md`, `docs/task-authoring.md`).
  Prereq: γ1 (so docs describe the landed semantics). Signal: §3.6/3.7
  describe both paths + the conversion rule; transition diagram carries
  the missing edges; grep pins doc↔code anchors updated.
- **λ — Deploy gate (deterministic, operator)** — fleet restart runbook
  (out-of-cgroup, never `--drain`), verify emission rows appear for the
  live strands. Prereq: η0, β, γ1-γ3, δ, ζ, η, θ, ι.
- **μ — Soak + enforcement flip gate (deterministic, operator)** — after
  λ soak: zero `would-violate`, `would-convert` set matches expectation
  → flip conversion + SM-2 + table enforcement (config; green-tier
  reload where applicable), verify the live strands convert on the next
  sweep and drain through normal resolution. Prereq λ.

## Out of scope

- The scope-divergence root cause (3429 option (c)) — amended 3429 owns it.
- The reify-5879 flap forensics — 3423 (redirected) owns it.
- Warm-lane hygiene / lane-release policy changes (worktree-lane-lifecycle
  territory); `review` status retirement decision (E14 — filed as an open
  question, not worth a task until a writer or a retirement appears).
- Any change to `_is_gating_escalation`'s predicate, the L1/L2 ladder,
  dedup keys, or escalation record lifecycle.
- Escalation-queue SLA/aging for L2 (gate-backlog age reporting 3520-3522
  already owns the surfacing).

## Open questions (tactical; AFK-safe defaults taken)

1. **Merge-phase `_mark_blocked` carve-out rationale** — γ2 investigates
   before changing (default: the carve-out is removable once merge-halt
   paths escalate-and-block; if a real dependency surfaces, γ2 records it
   and routes the write through a merge-aware target instead).
2. **Streak threshold N for repeated vetoes** — default 3 consecutive
   sweeps (matches reblock-guard threshold); config-tunable.
3. **`review` status (E14)** — leave in vocabulary, document as
   human-only; retirement needs its own tiny PRD if ever.
4. **Whether conversion should also stamp `blocked_reason`** for
   dashboard parity with `_mark_blocked` rows — default yes via
   `metadata.strand_converted.reason`.
5. **Infra-resume no-recompete mechanism** (D6): claim-then-status vs
   `pending` + verify-only dispatch fast-path. Default claim-then-status
   (smaller blast radius; no scheduler changes); γ3's architect verifies
   the footprint mechanics before committing.
6. **Conversion rate-limit as belt-and-braces** alongside the park-stop
   exclusion — default: cap conversions per sweep at
   `max_concurrent_tasks/2`; excess carries to the next sweep with a
   structured fact.
