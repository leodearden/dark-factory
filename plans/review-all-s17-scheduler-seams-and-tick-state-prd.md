# S17 — Scheduler seams and tick state (PRD)

Status: authored 2026-11-11 (author pass, agent seat), folded 2026-11-11 (fold pass over
`critique/S17.critic.md`), ruling fold 2026-11-13 (the sitting of 2026-11-12 and Leo's residual answers of
2026-11-13, see `## Rulings (sitting of 2026-11-12)`), fold-verified 2026-11-13 (the lead's rulings on OQ-9,
the slow-phase severity, the `is_deterministic` delegate and INV-23 applied), aligned 2026-11-13 with the
lead's later rulings and the sibling drafts (`critique/S17.align.md`; not committed, no task filed). Program:
`plans/review-all-program-dark_factory-2026-10-10.md` (run `review-all-dark_factory-20261008`), ruling **R16**
(approved as proposed, no open choice), whose stated reason for the stranded-blocked shape the sitting amended
(see `## Rulings`). Size L, risk medium. Upstream seams: S14 (shared leaves), S8 (harness composition root);
wave-3 siblings S1 (escalation filing), S16 (config shape) and S20 (delivered-check kinds) share files and
seams (see `## Cross-PRD relationship`). No stream depends on this one. No R22 invariant is owned here; this
plan mints Leo's design invariant `one-decision-partition-per-area`, INV-23 (leaf ν, `## Design invariant`).

**Code anchors** verified against main `a85cb1ad78` (2026-11-13), re-checked at `fceb951e7e` (2026-11-13, 57
commits later) by the fold-verifier: of the declared paths only `escalation/src/escalation/server.py`,
`fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/services/live_workflow_detector.py`
and `skills/unblock/SKILL.md` changed (1 to 9 lines each), and every anchor this plan cites in them (the
importer line, the "section 3" pointer, the descriptions and comments ζ2 corrects) still reads as stated;
every sidecar check was re-run at `fceb951e7e` with the same polarity. Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Counts labelled `measured at a85cb1ad78` were re-measured at the
ruling fold. A count still labelled `measured at 1f240e6da2` or `measured at 97be1d71fa` was checked at the
fold by running the same measurement at both shas: it is unchanged unless its row says otherwise.
`delivered_checks.py`, `dependency_verdict.py`, `stranded_reconcile.py`, `mcp/markup_sink.py` and
`mcp/verdict_tools.py` have no commit since `97be1d71fa`; what changed on the other declared paths is listed
under `## Background`.

Quality vocabulary: `docs/code-quality.md` (heuristics cited by number and name), design invariants
`docs/legibility/design-invariants.md`, findings contract `docs/quality-findings-contract.md` §6–§8.

Edge notation everywhere in this document: **"X → Y" means X depends on Y (Y lands first).**

`PRIVATE_REACH_PATTERN` (defined once, in `orchestrator/tests/test_scheduler_private_reach.py`, and used
verbatim by ω's delivered check), a POSIX ERE applied to raw lines so comment and docstring mentions count:

```
(^|[^A-Za-z0-9_])((real|fresh)_)?(scheduler|sched)\._[a-z]|(^|[^A-Za-z0-9_])Scheduler\._[A-Za-z]
```

## Goal

The next change to the dispatch scheduler is made by an agent. Today that agent meets a 9,401-line module
whose tests reach its private names on 1,479 lines in 81 files (`PRIVATE_REACH_PATTERN`, measured at
`a85cb1ad78`), a dependency predicate whose meaning is chosen by which optional caches a caller passes, a
per-tick context whose unset fields look like real answers, a second recovery decider that contradicts the one
recovery table, a delivered-check evaluator copied three times with one copy holding a dependent silently
forever, and a deterministic runner whose `run` is one 1,236-line method re-reading raw metadata. This PRD
makes each of those a typed seam with one owner, and moves the scheduler's tests onto behaviour so the
internals can be changed without editing seventy test files.

What an operator or agent observes when this lands:

- A scheduler test is written against `orchestrator/tests/_scheduler_scenario.py::SchedulerScenario`:
  it seeds state through `acquire_next`, `release`, `record_requeue`, `OverrideStore.set_override` and an
  injected clock, and asserts on `Scheduler.get_state_snapshot()` and emitted events. A guard test fails
  when any file in `orchestrator/tests` matches `PRIVATE_REACH_PATTERN`; at the end of this PRD the allowed
  count is zero (from 1,479 lines).
- `scheduler_state.json` (and fused-memory's `get_scheduler_state`) carries a `tasks` section listing only
  tasks with live dispatch state — dispatched, requeued, in their merge phase or recently cancelled — with
  the lock modules and dispatch priority shown only for dispatched tasks. A scored task that is not
  dispatched adds nothing, so the file stays at today's size (68,002 bytes, read 2026-11-13 08:40Z).
- A dependency whose delivered check cannot be evaluated emits a `delivered_check_errored` event on the
  tick it starts erroring. After `delivered_checks.errored_grace_sweeps` (default 20) consecutive erroring
  sweeps, a cause the check's author can fix (malformed descriptor, git rejecting the pattern or path,
  missing or non-executable script) files the same born-at-L2 `dependency_capability` escalation a failing
  check files, naming the reason, and blocks the dependent. A transient cause (timeout, other runner error)
  is handled by the gate's own code: it records each transient grace window as an event, clears the count
  itself on the sweep where the check runs again, and on the third consecutive transient window files that
  same human-level escalation and blocks the dependent; transient windows 1 and 2 file no queue record at all
  (lead ruling OQ-9 = A). No LLM watcher step sits in this path. Today
  the dependent waits forever behind one WARNING.
- A tick whose `delivered_check_gate` phase takes more than 30 s files one escalation for a human naming
  the project, the phase time and the checks run, so that growth of the phase is seen before it matters
  (leaf λ). It carries severity `critical`, the lowest severity that reaches Leo's queue (D15). The checks
  are not moved off the tick.
- A `blocked` task with no live claimant is no longer flipped to `pending` by the scheduler about 15 s
  after it blocks. The one recovery table decides it (S8's `RECOVERY_RULES`, extended by task 4082): the
  harness stranded sweep returns a clean strand to `pending` within one `stranded_reconcile_interval_secs`
  (900 s) and emits a recovery event naming the rule; it leaves alone, each under a named reason, a blocked
  task whose merge is queued, one parked for a recheck after its parent was cancelled, and one blocked
  behind a lane halted for another task's heal; and a task it has re-pended twice in 24 hours that blocks
  again gets one level-1 escalation for diagnosis and no third re-pend. That escalation is filed at most
  once per task per window: a block after it inside the window is left alone under a named reason, so the
  cap cannot itself loop (D13). At today's rate this adds no
  escalations (26 scheduler flips over 26 distinct tasks from 2026-10-19 to 11-12, none repeated).
- Every row of the recovery table names who decides it: code, an LLM or a human. The design invariant
  `one-decision-partition-per-area` makes that a checked rule for every recovery or policy table (leaf ν).
- "Are this task's dependencies satisfied?" has one answer per tick: dispatch eligibility, the delayed
  milestone anchor and the park-hold predicate read the same `DepGates` value.
- Delivered checks are evaluated by one function, `delivered_checks.run`, at every call site (scheduler
  gate, done-write guard, landing-evidence differential); a hung grep in the landing differential is
  bounded by the same timeout as everywhere else.
- `DeterministicRunner.run` is a dispatch over three steps (gate, predicate, deploy) that read one typed
  `RunnerState`; a deploy resumes after a crash from `deploy_state.phase`, not from which stamps happen
  to be present.
- The plan-tools and verdict-tools MCP servers receive their markup channels as one `MarkupChannels`
  bundle; no test monkeypatches a private `plan_tools` markup name.

## Background

R16 (program doc, verbatim): a behaviour-level test builder over `acquire_next`/`release`/`record_requeue`/
`set_override` asserting on `get_state_snapshot` and events, so the private test references retire file by
file; one per-task `TaskRecord` replacing the task-keyed dicts; `TickContext` with required fields and
phases invoked by reference; `_deps_satisfied` takes a `DepGates` value; the stranded-blocked phase consults
`TaskGroundTruth.derive_truth` per R6 instead of performing the revert the recovery table forbids;
delivered-check evaluation delegates to `delivered_checks.run` as the one evaluator with an event on
ERRORED; shared `dependency_ids`/`train_id` helpers at the readers; `DeterministicRunner.run` split into
predicate/deploy/gate steps over `parse_metadata`; `plan_tools.create_server` takes a `MarkupChannels`
bundle; the `_SchedulerLike` shadow protocol and hand-maintained fakes replaced by the typed snapshot seam.
Seam ownership: `get_state_snapshot` as the public test seam (typed via S14's `SchedulerStateSnapshot`);
`delivered_checks.run` as the one evaluator (orchestrator owns; landing_evidence and scheduler call).

R6 (S8) is the recovery ruling this stream applies: `TaskGroundTruth` is the single classifier and the
sweeps apply its answer verbatim; S8 δ (task 9219) turns `_RECOVERY` into `RECOVERY_RULES` and S8 ε2a
(9223) makes the harness stranded sweep a `StrandedReconciler` over `in-progress` and `blocked`. The sitting
of 2026-11-12 kept that aim and changed one stated reason: a clean stranded blocked task is returned to
`pending` by a code row of that table, never silently (see `## Rulings`, D13).

What moved between the draft's tree `97be1d71fa` and `a85cb1ad78` (1,196 commits; the paths this plan
declares): task 6145 landed (`2b044c36ca`, done 2026-11-13): `escalation/pins.py::needs_human` is now the
one "only a human can dispose of this record" predicate, `recovery_pins.py::records_pin_blocked_recovery`
never relaxes a merge-remediable record set that holds such a record, the scheduler's flip log names the
records it went past (`scheduler.py::_describe_records_flipped_past`), and
`orchestrator/tests/test_merge_gate_implies_sweep_pin.py` pins "a record that gates merge entry also pins
the sweep". It did not remove the scheduler step; its follow-up task 9529 (pending, low) proposes a loop
breaker for that step. Task 3090 added `Scheduler.acquire_uncontested` and a seventeenth `SchedulerFacade`
method (with `orchestrator/tests/test_scheduler_acquire_uncontested.py`); the `Milestone.fired` repoint added
four dependency-region lines to `test_scheduler.py`; task 6390 (done) landed the deterministic runner's one
gate-fields builder, and `deterministic_runner.py` grew to 4,090 lines; four new test files construct
`FakeScheduler`; `test_agent_session_reaper.py` gained one `h.scheduler._dispatched` reach. Every other count
in this plan was re-run at `a85cb1ad78` and is unchanged.

What moved since the run's tree `6551b75aba` (scheduler.py: +1,593/−2,074 lines; 84 commits):
`ModuleLockTable` left for `orchestrator/lock_table.py` (task 6126, `0112c8fde0`); the dependency predicate
moved to `orchestrator/dependency_verdict.py::deps_verdict` as a tri-state verdict (task 6518) but kept its
optional caches; `get_state_snapshot` returns S14's typed `SchedulerStateSnapshot` (task 6717); the
starvation predicate moved to `starvation_watchdog.py` (task 3821, which also retired the
`_starvation_first_seen` alias); task 6488 extracted `before_done_script.py` and `predicate_output.py`
from the runner and recorded a heuristic-14 measurement, leaving `run` whole; task 6098 gave
`verdict_tools.create_server` a `resolve_project_root` keyword and left plan-tools for task 7627.

## Rulings (sitting of 2026-11-12)

Sources: `digests/wave3-rulings-applied.md` (v2, final), `digests/wave3-leo-answers-2026-11-12.md`,
`digests/wave3-leo-residual-answers-2026-11-13.md`, `digests/discussion/S17.md` (its "Effect on the plan if
adopted" text is part of each ruling it names), `digests/wave3-lead-rulings.md`.

- **S17-OQ-1** — the scheduler's state snapshot gains a per-task section; tests read dispatch facts there
  (option A). Ruled by the lead.
- **S17-OQ-2** (sitting key S17-1) — NEW option D: the scheduler's step is deleted and a code row of the
  recovery table returns a clean stranded blocked task to `pending` within 900 s with an event naming the
  rule; three code rows leave the parks alone; a task re-pended twice in 24 hours that blocks again gets one
  level-1 escalation for diagnosis and no third re-pend; task 4082 carries the rows; 9219 adds no scheduler
  hook; the recommendation was A (the watcher resumes). Ruled by Leo (question 2026-11-12, R-2 = A
  2026-11-13). Amends R16's stated reason ("instead of performing the revert the recovery table forbids")
  and R6's table rule ("no silent return to pending") for proven-unparked shapes only, never silently; the
  aim of both, one classifier the sweeps apply verbatim, stands (D13).
- **Decision-partition invariant** — Leo's principle becomes a design invariant carried by this plan (leaf
  ν), a four-site lockstep edit, text as recorded in the discussion, checkable as "every row of a recovery
  or policy table names who decides it", landing with or after task 4082's rows. Ruled by Leo 2026-11-13.
  Its number follows the lead's S2/OQ-10 rule, which amended R51's numbering clause: the six wave-3 minting
  tasks take INV-18 to INV-23 in landing order and this one, slug `one-decision-partition-per-area`, is
  minted last as **INV-23**; every check anchors on the slug (ruled by the lead 2026-11-13).
- **S17-OQ-4** — CHANGED from the recommendation: `shared/task_fields.py` is the one reader, fused-memory's
  four importers import it, `utils/task_dependency_ids.py` is deleted, no permanent re-export. Ruled by the
  lead.
- **S17-OQ-5** (sitting key S17-2) — option B as recommended: author-fixable causes reach a human after the
  grace period and block the dependent. Ruled by Leo. Plus the lead's reading under Leo's principle (shown to
  Leo, not objected to): the gate's code does the transient mechanics and the planned watcher playbook
  section is removed; how the gate records transient windows 1 and 2 is S17-OQ-9 (below). Plus Leo
  2026-11-13: no offloading; one low task escalates when the phase exceeds 30 s (leaf λ). The lead ruled
  2026-11-13 that this alarm reaches a human at the lowest severity that still reaches Leo's queue, one open
  record per project, threshold a hot-reloadable setting with default 30 s; and then confirmed the severity
  named `critical`, because it is the floor of the severities a record may be born with at level 2
  (`BORN_AT_L2_SEVERITIES`) and anything lower would need an LLM promotion (D15 says why).
- **S17-OQ-9** (new at the ruling fold) — option A: the delivered-check gate records transient grace
  windows 1 and 2 as its own `delivered_check_errored` events with no queue record (a `cleared` event marks
  the check running again), and the third consecutive window files the human-level record. No level-1
  record is filed, so no watcher rotation starts and no LLM is called. Ruled by the lead 2026-11-13; it
  corrects the lead's earlier wording ("files a level-1 and closes it itself").
- **`Scheduler.is_deterministic` kept as a delegate** (raised at the ruling fold: α keeps the method as one
  call to the shared reader instead of editing 27 more test files). Accepted by the lead 2026-11-13.
- **S17-OQ-6** (sitting key S17-3) — the delayed-milestone clock waits for delivered checks (option A, as
  recommended). Ruled by Leo.
- **S17-OQ-7** — only tasks with live dispatch state appear in the snapshot's task section (option A).
  Ruled by the lead.
- **S17-OQ-8** (sitting key S17-4) — option A plus a third rule row (behind a heal-halted lane), all three
  carried out by code rows, so the watcher re-check option A relied on is not built; the item-3 part of
  Leo's 2026-10-09 re-scope of task 4082 is preserved. Ruled by Leo (R-2 = A, discussed with S17-1).
- OQ-3 was retired at the 2026-11-11 fold (R16 rules the remedy).

Cross-stream consequences applied: the decision-partition invariant `one-decision-partition-per-area`,
INV-23 (`## Decision partition`); hermetic
evals (nothing here adds a store, sink or filer an eval cell can reach: `recovery_applied` and the gate's
events go to the orchestrator's event store, which S1's eval-hermeticity task keeps out of eval cells); the
test-file edge rule (ι2 → ι1, ι3 → ι2 and κ2 → 8778 are not wired); S16/OQ-11 (S16 waits for S17; S16 ζ2
carries 4082's two new settings). Caller identity (S1/S2), recon stage identity, the fused-memory shutdown
chain, restart settings, liveness and the operator sweeps do not touch this plan. Nothing is pending on a
residual question. The lead's ruling on S17's two new scheduler filers: they file through S1's one filing
entry point; this plan states each one's policy and sentinel (C5, C13), and because δ2 depends on S1 ζ2 and
λ follows δ2 on the chain, neither can land before the door, so S1 has nothing to convert. Under OQ-9 = A
the gate's second filer is its born-at-L2 record, not a self-resolving level-1.

## Findings disposition

Every key of the stream, exactly once. "measured" = at `1f240e6da2` unless it says `97be1d71fa` or
`a85cb1ad78`; every older count was re-run at `a85cb1ad78` and is unchanged unless the row gives the new value.

| Key | Sev | Disposition | Note |
|---|---|---|---|
| `fk-a571802ee4ea` tests pin Scheduler privates | high | leaf β, ι1, ι2, ι3, ω (closes at ω) | Still present: 1,479 lines in 81 files under `PRIVATE_REACH_PATTERN` (measured at `a85cb1ad78`; 1,473 in 79 at `97be1d71fa`; run: 1,483 references in 70 files); 117 distinct names through `scheduler._x` alone, 312 assignment lines (measured at `a85cb1ad78`, `\bscheduler\._name … =` over raw lines). The six new lines: `test_agent_session_reaper.py`'s `h.scheduler._dispatched` (ι3) and `test_merge_gate_implies_sweep_pin.py`'s docstring naming `Scheduler._phase_redispatch_stranded_blocked` (ζ), plus four in `test_scheduler.py` (γ). The pattern does not match the two "Mirrors test_harness_resume_scheduler._make_harness" docstrings and does match the `real_scheduler._x`/`fresh_scheduler._x` and `Scheduler._X` reaches. Aliases: 6 remain (`_starvation_first_seen` went with task 3821), 49 test references in 2 files, 1 production read outside `__init__` (`_phase_stale_sweep`'s `_starvation_escalated`). |
| `fk-4b35b1adea0d` dependency readiness has no single definition | high | leaf γ | Partly moved (task 6518: one predicate home, tri-state) but the defect stands: `deps_verdict` takes six optional parameters (`tasks_by_id`, `external_status_cache`, `external_resolver_failed`, `delivered_check_cache`, `terminal_dep_records`, `deferred_terminal_deps`) whose absence turns an arm off. Measured 4 call sites, 3 meanings: `_eligible_for_dispatch` and the park-hold predicate (all four arms); `_stamp_milestone_deps_satisfied`, whose docstring says it forwards the caches "so the anchor reflects the SAME local+external+scheduled-deploy dependency evaluation dispatch uses" — three arms, the delivered arm excluded; `_phase_redispatch_stranded_blocked` (local only). |
| `fk-cc01eb7b143f` stranded-blocked recovery has two deciders | high | leaf ζ (+ companion ζ2; the replacement rows are task 4082's) | Still present: the phase (253 lines, measured at `a85cb1ad78`) flips `blocked → pending` through `clear_claim_then_set_status` without consulting `task_ground_truth`; row (g) of `_RECOVERY` still says the revert is forbidden. Task 6145 (done, `2b044c36ca`) narrowed what the phase may flip past (a record set holding a record that needs a human now pins) and made its log line name those records; it left the second decider in place. Flips, all projects, `/var/log/syslog*`: 4,029 from 2026-10-13 to 11-12, 3,958 of them three solar-challenge tasks before task 5222 landed; 26 over 26 distinct tasks from 10-19 to 11-12, none flipped twice (measured 2026-11-12, `digests/discussion/S17.md`). |
| `fk-3e1e7c5129b5` TickContext defaults look like results | medium | leaf ε | Still present: `external_cache` and `delivered_check_cache` default `{}`; `None` means gate-off only on the kill switch; 22 phases dispatched by `getattr(self, f'_phase_{label}')` over `_TICK_PHASE_ORDER`; `_phase_backfill_dep_status` completes `status_map` in place, so a phase ordered before it reads an incomplete map that no declaration shows (measured by AST at `97be1d71fa`). |
| `fk-8af92edd6f38` three delivered-check runners | medium | leaf δ1 | Still present, `delivered_checks.py` unchanged since the run: the scheduler wraps `run_delivered_check` in its own `asyncio.wait_for` and collapse; `verify_delivered_checks_on_main` repeats both and still carries "mirror scheduler.py 3116-3129"; `merge_lane/landing_evidence.py::_delivered_checks_differential` calls `run_delivered_check` with no timeout. |
| `fk-6c103745c784` ERRORED check holds silently forever | high | leaf δ2 (λ adds the phase-time alarm Leo asked for under S17-2) | Still present (`delivered_checks.py` and the scheduler's gate unchanged since `97be1d71fa`): ERRORED leaves the dep uncached (now `DepState.UNKNOWN`), bumps `_delivered_check_error_streak`, logs WARNING only when the streak equals `_DELIVERED_CHECK_ERROR_LOG_THRESHOLD` (20), emits no event and files nothing. The `delivered_check_gate` phase never reached 60 s in the 7 days to 2026-11-12 (longest tick 12.8 s, reify; 12.2 s, dark-factory; `scheduler_tick` events in every `runs.db`), so no offload is built. |
| `fk-b2cd27267851` one task's lifecycle in 18 dicts | medium | leaf η | Still present and one larger: all 18 named containers remain plus `_anchor_pending_since` (19, measured at `a85cb1ad78`); `__init__` is 578 lines; `_phase_stale_sweep` GCs a subset. |
| `fk-09bf639213ca` scheduler.py purpose needs "and" | medium | leaf θ (contributions ζ, δ2, η) | Weakened, not gone: `ModuleLockTable` is `fixed:0112c8fde0` (task 6126); 9,401 lines, 2,982 docstring + 1,430 comment lines (46.9 % prose, measured at `a85cb1ad78` by AST and tokenizer); the remaining partitions are selection, the fused-memory task-store client with its rejection family, five watchdogs, the snapshot writer. The ruled remedy (backfill admission as a collaborator) is θ; the client partition is measured under Out of scope and returned to the program as an xref. |
| `fk-a2bc16e5812c` wire shapes re-parsed inline | low | leaves α0, α (ξ1 repoints the runner's copy) | Still present: dependency-id normalisation 8 copies (scheduler 5, `dependency_verdict` 1, `deterministic_runner` 1, `agents/briefing` 1 variant) beside fused-memory's own `utils/task_dependency_ids.py::task_dependency_ids` ("The single CSV-tolerant 'read a task's dependency ids' parser", task 3615; 4 importers, measured at `a85cb1ad78`), whose tolerance differs; train-id extraction 5 hand-written readers with three different empty-id rules (`train_order_key`, `tasks_by_train`, `dependency_verdict._local_verdict` ×2, `merge_deferred_recovery.DeclaredTrainRef.from_task`); inline deterministic-kind test 3 sites (scheduler 1, harness 2) beside `Scheduler.is_deterministic` "single source of truth". |
| `fk-4b3eb71db017` shadow protocol and hand fakes drift | medium | leaves κ1, κ2 (R16 rules the remedy) | Still present and grown: `scheduler.py::SchedulerFacade` declares 17 methods (task 3090 added `acquire_uncontested`); `orchestrator/tests/_workflow_helpers.py::FakeScheduler` is referenced in 42 files and constructed at 80 sites in 32 files; 8 test files name `SchedulerFacade` (all measured at `a85cb1ad78`). S10 β1 (9187) keeps them in parity and S9 μ2 (9279) gives MagicMock users an autospec double; neither removes the shadow. A third hand-kept double, `orchestrator/tests/_task_store_doubles.py::TaskStoreDouble`, is used by 2 test files (measured at `a85cb1ad78`). |
| `fk-5a5d08e775de` DeployState beside the legacy stamps | medium | leaf ξ2 | Still present: `DeterministicRunner.run` is 1,236 lines, 37 returns, 64 branch nodes (measured at `a85cb1ad78`); resume reads `before_done_ran_at`/`before_done_verified_at`/`before_done_scheduled_at`/`gate_escalated_at`; `deploy_state.phase` is consulted in two places (ESCALATED, RAN). |
| `fk-ebce5e23048a` runner dispatches on raw metadata | medium | leaves ξ1, ξ2 (closes at ξ2) | Still present: 36 `metadata.get`/`before_done.get` reads; `parse_metadata` is not called; `BeforeDone` is imported only for a type. Task 8356 (pending, low) proposes the per-branch split without the typed parse; ξ1 subsumes it. |
| `fk-6e7e9bb942d6` plan-tools markup wiring by monkeypatch | low | leaf μ | Plan-tools half still present (re-measured at `a85cb1ad78`, unchanged): 5 call-time lambdas in the two markup sink builders `create_server` wires (3 of them `resolve_root`/`open_channel` global lookups), 7 private markup names patched or read 22 times, 42 patch call sites in the three plan-tools test files. Verdict-tools half `fixed` by task 6098 (`resolve_project_root` keyword; 0 patch sites in `test_verdict_tools_markup_registration.py`); μ still moves verdict-tools onto the same bundle so both servers steer one axis one way (heuristic 3). |

## Pre-conditions for activating

- **S8 C3** (`SchedulerCallbacks` with required gates, `make_scheduler` test factory, `passthrough_gates()`):
  S8 β0 9217 and β 9218. The scenario builder constructs through that factory.
- **S8 C2** (`RECOVERY_RULES`, `RecoveryDecision`; `recovery_for`): S8 δ 9219; and S8 ε2a 9223
  (`StrandedReconciler` over `in-progress` and `blocked`). ζ deletes the scheduler's second decider only once
  the one applier handles blocked rows, and only once **task 4082** (re-scoped by the sitting, C12) has added
  the blocked rows that replace it: the three leave rows, the owned-record hold, the clean re-pend row, its loop cap and the once-per-window leave.
- **S1 C5** (`scheduler_escalations.py::SchedulerEscalations`, one `SchedulerCallbacks.escalations` port
  replacing `on_delivered_check_block`, `on_warm_base_warn`, `on_warm_base_promote_l2`, `on_warm_base_resolve`
  and the other scheduler hooks; S1 leaf ζ2 (after ζ1)) and **S1 C6** (`file_and_block`, the typed status-write
  outcome and the deterministic runner's tails; S1 leaf γ). δ2, λ and θ file through C5; ξ1 moves the tails C6
  reshapes. S1 is wave 3: the task ids are wired at decompose, whichever stream files second.
- **S14** shared leaves: `shared/scheduler_state.py::SchedulerStateSnapshot` (S14 η 6717, done, invariant
  I4); `shared/scheduler_overrides.py` (S14 θ 6718, pending; the builder seeds overrides through it);
  `shared/streak.py::KeyedStreaks` with the scheduler's `_delivered_check_error_streak` registered on it
  (S14 ο 6724, pending, behind 8528).
- **S10 C2** (`Scheduler.update_task` raises `MetadataWriteFailed`; S10 β1 9187, β2 9191) lands on
  `scheduler.py` and `deterministic_runner.py` first.
- **S5 C1** (`run_git` as the runner default in `delivered_checks.py`; S5 α 9255, ε 9263) and S6 μ 9201
  (`landing_evidence.py`) land on δ1's files first.
- **S1 γ's lower-module filer** (`status_write.py::file_and_block`, one born-at-L2 filer for the runner that
  imports nothing from `deterministic_runner.py`) lands before ξ1: it is what lets the steps leave
  `deterministic_runner.py` without a reach-back (heuristic 13). S1's decompose cancels task 8355 (the same
  filer, filed earlier) and task 3887 into γ.
- G3 substrate verified at `1f240e6da2` (re-checked at `97be1d71fa` and `a85cb1ad78`; for ν and λ also
  `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`, the `scheduler_tick` event's per-phase
  `phases` map written by `Scheduler._emit_tick_telemetry`, `config.py::OrchestratorConfig.stranded_reconcile_interval_secs`,
  `escalation/pins.py::needs_human`): `Scheduler.get_state_snapshot`,
  `Scheduler.flush_state_snapshot`, `Scheduler.is_dispatched`, `Scheduler.callbacks`,
  `Scheduler.record_requeue`, `Scheduler.release`, `Scheduler.dispatch_tool`, `OverrideStore.set_override`,
  `dependency_verdict.py::deps_verdict`/`DepsVerdict`, `task_ground_truth.py::TaskGroundTruth.recovery_for`,
  `delivered_checks.py::run_delivered_check`, `shared.task_metadata::parse_metadata`/`BeforeDone`,
  `shared/deploy_state.py::DeployState`/`DeployPhase` (re-exported by `orchestrator/deploy_state.py`),
  `shared/streak.py::KeyedStreaks`, `mcp/markup_sink.py::make_escalation_sink`/`resolve_project_root`/
  `open_escalation_channel`, `mcp/markup_journal.py::make_fact_journal`,
  `fused-memory/src/fused_memory/utils/task_dependency_ids.py::task_dependency_ids` (deleted by α0).

**Consequence of the per-file order (stated, not hidden).** Every wave-2 task on `scheduler.py` precedes
this stream's (the wave-2 chain is 9234 → 9231 → 9219 → 9218 → 9217 → 9283 → 9209 → 9255 → 9191 → 9187,
so 9187 lands first and 9234 last; all pending at 2026-11-13), and 9217 onward waits on task 5040 (pending)
through S8. So every leaf on this stream's scheduler chain (β, α, ζ, γ, δ1, δ2, λ, ε, η, θ, κ1) and every
leaf that waits on it (ι1, ι2, ι3, κ2, ω, ν through 4082 → 9219) waits on 5040 too. α0, ξ1, ξ2 and μ are off
that chain and not held by it (ξ1/ξ2 wait on 8528 through S10 β2; μ waited on S9 α 9170, now done).

## Sketch of approach

Order: **tests onto behaviour first, then the internals.** The scenario builder and the snapshot's per-task
section (β) give every test a public way to seed and observe dispatch state. Three migration leaves (ι1,
ι2, ι3) move the highest-volume reaches — the lifecycle dicts, constructor-injected collaborators and streak
aliases — and every reach outside the structural leaves' own files to a name those leaves delete, onto it,
with no production change. After that each structural leaf changes one internal and migrates the remaining
tests that reach exactly that internal, lowering a shrink-only per-file baseline
(`scheduler_private_reach_baseline.json`). ω migrates the long tail and deletes the baseline: zero becomes
the rule.

1. **Test seam (β; C1, C2).** `SchedulerScenario` builds a real `Scheduler` through S8's `make_scheduler`
   over an in-memory fused-memory task store reached through `McpSessionLike.call_tool` (so the client
   code, envelope parsing included, runs for real), a real `OverrideStore` on `tmp_path`, a recording
   `EventStore`, an injected monotonic and wall clock, and a scripted PSI reader. `SchedulerStateSnapshot`
   gains a `tasks` section projecting the per-task lifecycle of tasks with live dispatch state. Two
   constructor keywords the tests need and task 6143 asked for (`psi_reader=`, `hold_history=`) are added.
2. **Readers (α0, α).** `shared/task_fields.py` holds `dependency_ids` (the union of the scheduler's and
   fused-memory's tolerances), `train_id`, `is_deterministic`; fused-memory's `utils/task_dependency_ids.py`
   is deleted and its four importers import the shared reader (import-line edits only); every reader goes
   through them; `Scheduler.is_deterministic` stays the scheduler's public method but its body becomes a
   call to the shared reader (the harness's slot path calls it through its scheduler, and 27 test files mock
   or call it there, measured at `a85cb1ad78`); an AST guard keeps the copies from returning.
3. **One recovery decider (ζ, on task 4082's rows).** Task 4082, re-scoped by the sitting, adds the blocked
   rows to `RECOVERY_RULES` (S8 δ) that the `StrandedReconciler` (S8 ε2a) applies (C12): three code rows
   leave a task alone under a named reason (its merge is queued; it is parked for a recheck after a
   cancelled parent; it is blocked behind a lane halted for another task's heal), a code row returns a clean
   strand to `pending` with a `recovery_applied` event naming the rule, and a task that row has re-pended
   twice in 24 hours and that blocks again gets one level-1 escalation for diagnosis instead of a third
   re-pend; once that escalation is filed, a further block inside the window is left alone under a named
   reason, and a task with an owned open record is held, never re-pended. Every row names who decides it. Then ζ deletes `_phase_redispatch_stranded_blocked`, its kill
   switch `stranded_blocked_redispatch_enabled` and the scheduler-side recovery-disposition branch, so blocked
   rows have one decider and one applier. ζ2 corrects every document that says the scheduler re-pends a
   stranded blocked task and records `deferred` with a deferral record as the blessed park (task 4082's
   former item 4).
4. **One dependency predicate (γ; C3).** `deps_verdict(task, gates: DepGates)`; `DepGates` has no
   optional field; the tick builds it once after its producers ran; `_deps_satisfied` is deleted.
5. **One delivered-check evaluator (δ1; C4) and a gate that reports (δ2; C5).** `delivered_checks.run`
   owns the per-check timeout, the collapse rule and the reasons; `verify_delivered_checks_on_main` and the
   scheduler's private loop go; `landing_evidence` calls `run`. The scheduler's 406-line
   `_compute_delivered_check_cache` becomes `delivered_check_gate.py::DeliveredCheckGate`, which emits
   `delivered_check_errored` and escalates a persistent ERRORED through S1's scheduler escalation port, all in
   code: a cause the check's author can fix files born-at-L2 at the end of the first grace window and blocks
   the dependent; a transient cause is counted by the gate in grace windows, cleared by the gate on the sweep
   the check runs again, and filed as that same born-at-L2 escalation on the third consecutive window;
   windows 1 and 2 file no queue record (lead, OQ-9 = A). No watcher playbook takes part.
6. **Tick phases by reference (ε; C6).** `tick_phases.py` holds `TickContext` (inputs required, producer
   fields unset until written and loud when read early), `TickPhase(name, run, reads, writes)` and
   `validate_phase_order`; `Scheduler` declares `TICK_PHASES` as a tuple of method references validated at
   import, with each phase's read and write sets derived from its body by AST.
7. **One record per task (η; C7).** `dispatch_ledger.py::DispatchLedger` of frozen `TaskRecord`s replaces
   14 of the 19 containers; lockstep rules become ledger transitions; GC is one `retain`; the six streak
   aliases are deleted. The other five containers go to their owners: backfill grants to θ, the ERRORED
   streak to δ2, the two age-anchor maps to task 6821's `age_anchors.py` (η → 6821), the terminal-dep record
   cache stays with the backfill phase.
8. **Collaborators out (θ; C8).** `backfill_admission.py::BackfillAdmission` (grants, outcome window, scan
   cache, hold history, overstay escalation) and `warm_base_watchdog.py::WarmBaseHardDownWatchdog`, both
   filing through S1's scheduler escalation port.
9. **The shadow protocol and the hand fakes go (κ1, κ2; R16).** `SchedulerFacade` is deleted and its
   consumers annotate `Scheduler`; `FakeScheduler` and `TaskStoreDouble` are deleted and their users run
   the scenario builder's real scheduler over the in-memory store (or S9 μ2's autospec double where they
   only record calls; chosen per file in κ2).
10. **Runner steps (ξ1, ξ2; C10).** `RunnerState.parse(task)` is the one metadata parse; gate, predicate
    and deploy are step classes in their own modules; the deploy step resumes from `DeployState.phase`.
11. **Markup channels (μ; C9).** `MarkupChannels` in `markup_sink.py`; both MCP servers take it as a
    required keyword; `main()` passes `MarkupChannels.production()`.
12. **Phase-time alarm (λ; C13).** The tick hands the `delivered_check_gate` phase time it already records
    for the `scheduler_tick` event to `DeliveredCheckGate.observe_phase_time`; above
    `delivered_checks.phase_alarm_secs` (default 30) the gate files one born-at-L2 escalation per project, at
    severity `critical`, naming the phase time and the checks it ran, with the remedy "add a wall-clock budget
    for the phase". The checks stay on the tick.
13. **The decision-partition invariant (ν).** Leo's principle becomes `one-decision-partition-per-area` in
    `docs/legibility/design-invariants.md` through the four-site lockstep edit, with task 4082's
    `RECOVERY_RULES.decided_by` column as its house pattern and a violating fixture of each shape.

## Resolved design decisions

- **D1 — Tests first, internals second.** ι1/ι2/ι3 precede every structural leaf that deletes a name tests
  reach outside its own files (γ onward; the edge is γ → ι1, ι2, ι3 and the chain carries it to δ1, δ2, λ,
  ε, η, θ and κ1; α and ζ delete no such name), and each structural leaf migrates the tests that reach the
  internal it changes in the files it declares. ι3
  exists because a deletion leaf landing before ω would otherwise break test files outside its own list
  (measured at `a85cb1ad78`: 30 files, 70 code lines reach a name γ, δ2 or η deletes outside the deleting
  leaf's and ι2's files). Rejected: migrating inside each structural leaf only (the lifecycle dicts are
  reached by 444 references in 40 files; η alone would be far beyond the band). Heuristic 9 (deep modules,
  narrow interfaces) and the Tests stance.
- **D2 — The guard is a shrink-only per-file baseline, then zero.** `test_scheduler_private_reach.py`
  defines `PRIVATE_REACH_PATTERN` once and counts, per file, the raw lines of every tracked file under
  `orchestrator/tests/` (ω's pathspec exactly; the builder modules and the guard's own source included, no
  file exempt) that match it, so a comment or docstring naming a private reach counts and is migrated with
  the code. Invariant: `test_scheduler_private_reach.py` matches `PRIVATE_REACH_PATTERN` on zero lines at
  every commit; rows 4 and 4b build the planted receiver and attribute as separate string tokens joined at
  run time (for example `receiver + '.' + attr`, with `receiver = 'real_scheduler'` and
  `attr = '_dispatched'`) and write them under `tmp_path`, never as one literal (row 4c). It fails when a file exceeds its baseline or an unlisted file is non-zero, and
  reports files whose count fell below baseline as "lower the baseline". Measured: 1,479 lines in 81 files
  at `a85cb1ad78`; the pattern does not match the two "Mirrors test_harness_resume_scheduler._make_harness"
  docstrings. Known limit, stated in the test's docstring: a private reach through a receiver the pattern
  does not name (`s`, `restarted`, `h.scheduler`'s aliases, a mock named otherwise) is not counted. Each
  leaf that migrates reaches lowers the baseline in the same commit; ω deletes the baseline file and the
  test asserts zero everywhere. ω's delivered check greps the same pattern, so the guard and the check
  cannot disagree. Precedent: `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- **D3 — The test store sits at the MCP-session boundary**, not at `Scheduler` methods: the existing
  `orchestrator/tests/_task_store_doubles.py::TaskStoreDouble` patches public methods (`wire_into`), so it
  never exercises the client's parsing, and κ2 retires it. The in-memory store answers every tool the
  scheduler and its `dispatch_tool` callers send (C2's list) with fused-memory's envelope shapes, raises a
  typed `UnscriptedTool(name)` for any other tool, and applies S10 C1's write modes through
  `shared.task_metadata_wire` (the same rule S10's `FakeMetadataBackend` uses). Prior art:
  `orchestrator/evals/runner.py::_StubMcpSession` (eval mode; not reused, because it is production code
  with eval semantics).
- **D4 — The snapshot's task section is bounded and holds no monotonic float.** A task appears only while
  it has live dispatch state (C1's inclusion rule); `modules` and `dispatched_priority` are projected
  only for a dispatched task. Measured: `Scheduler._get_modules` fills `_module_cache` for every task it
  scores, not only dispatched ones, and `data/orchestrator/scheduler_state.json` is already 68,002 bytes with
  3,276 `effective_priorities` entries (2026-11-13 08:40Z; 60,747 bytes and 3,069 entries on 2026-11-11), while
  `fused_memory/server/tools.py::get_scheduler_state` returns the snapshot whole over MCP (documented-safe
  response envelope ~62 KB). Time-windowed facts stay in `requeue_cooldowns` (wall-clock, as armed) or are
  projected as booleans (`in_merge_phase`, `workflow_cancelled`), so the content-dedup in
  `_write_snapshot_best_effort` keeps working: a newly scored task changes nothing.
- **D5 — `DepGates` is built once per tick and read by every dependency consumer of that tick.** With ζ the
  local-only meaning disappears; dispatch, the milestone anchor and the park-hold predicate read the
  same value. Accepted behaviour change (S17-3, ruled as recommended): a delayed milestone's clock starts
  only once its dependencies' delivered checks pass on main; the live delayed milestone it affects is trigger
  gate 6698 (pending, `after_secs` 259,200), whose delay starts once its high leaves' checks pass. A consumer
  outside a tick does not exist at `a85cb1ad78`; if one appears it builds a `DepGates` explicitly (no field
  has a default).
- **D6 — The per-tick check budget stays the caller's decision.** `delivered_checks.run` evaluates a whole
  check list; `DeliveredCheckGate` decides, before calling it, whether a dep's whole check list fits the
  remaining `max_checks_per_tick` (the first uncached dep of a sweep always runs, today's forward-progress
  rule). Two deliberate behaviour changes, each pinned by a boundary row: (a) today the budget is consumed
  per check inside the loop, so a dep can be cut mid-list after some checks ran and then deferred
  (`_compute_delivered_check_cache`'s `used`/`over_budget`); after δ2 a dep is admitted whole or deferred
  whole and no check runs for a deferred dep (row 8b); (b) today a dep's checks run sequentially; C4 runs
  them concurrently, order of results preserved (row 8c).
- **D7 — The phase table is data with declared read and write sets, validated at import.** Each phase's
  `TickPhase.reads`/`writes` is derived from its body by AST: `ctx.<field>` loads are reads; `ctx.<field>`
  stores, subscript stores and in-place mutator calls (`.update`, `.add`, `.pop`, …) are writes; a `ctx`
  passed to a `Scheduler` helper contributes that helper's loads and stores, transitively (measured at
  `97be1d71fa`: six phases — `park_gc`, `drain_park_eviction`, `reserve_now`, `starvation`, `select_pins`,
  `select_scored` — pass `ctx` on, and `_park_hold_refusal` alone reads seven ctx fields). A test asserts
  each declared `reads` ⊇ the derived loads and each `writes` ⊇ the derived stores. Fields a phase completes
  are produced fields: `status_map` is written by `backfill_dep_status` from the tick input
  `fetched_status_map`, `terminal_dep_records` by `backfill_terminal_dep_records`, `overrides_for_diff` by
  `reserve_now`. `validate_phase_order` raises `PhaseOrderError` naming the phase, the field and its
  producer when a phase reads a field no earlier phase writes and the tick inputs do not supply (heuristic
  10, informatively and redundantly: at import and at read). Field names are those after γ (`external`,
  `delivered`, `dep_gates`).
- **D8 — `TaskRecord` is frozen; the ledger holds the only mutable map.** Transitions are ledger methods
  that store a new record (heuristic 8); the lockstep rules now written in prose (for example
  `clear_requeue_count`'s note about `_requeue_until`) become the body of one method each (INV-5).
- **D9 — Backfill admission and the warm-base hard-down watchdog leave as state-owning classes**, each
  constructed by `Scheduler.__init__` from injected collaborators (S1 C5's `SchedulerEscalations` among
  them) and read by the tick through a narrow method. Heuristic 13: neither module imports
  `orchestrator.scheduler`; the scheduler imports them.
- **D10 — Deploy resume reads `DeployState.phase`; the stamps stay as written evidence.** The stamps
  (`gate_escalated_at`, `before_done_ran_at`, `before_done_verified_at`, `before_done_scheduled_at`) are
  read by 12 modules in 4 packages (measured), so retiring them is out of scope. The deploy step decides
  from the phase; a legacy row with stamps but no `deploy_state` is mapped once, in `RunnerState.parse`,
  by the module-level function `orchestrator/src/orchestrator/deploy_state.py::from_legacy_stamps` (one
  function, tested over every stamp combination the runner distinguishes today). It is not a `DeployState`
  classmethod: `DeployState` lives in `shared/src/shared/deploy_state.py`, and shared must not import
  orchestrator's `GateStamps`.
- **D11 — Steps live in their own modules below the runner.** `deterministic_gate_step.py`,
  `deterministic_predicate_step.py`, `deterministic_deploy_step.py` import `deterministic_state.py` and
  S1 γ's lower-module filer (`status_write.py::file_and_block`), never `deterministic_runner.py`;
  `DeterministicRunner.run` parses, picks the step and returns its outcome. The `# ── 1./2./3.` section names
  survive as the steps' names in `docs/deterministic-runner.md`, which external prose
  (`live_workflow_detector.py`) cites.
- **D12 — `MarkupChannels` is required, lazy and lives in `markup_sink.py`.** No default at
  `create_server` (a forgotten argument is a type error, the lesson of R28); `main()` builds
  `MarkupChannels.production()`; the resolver is not called at `create_server` time (startup-latency
  constraint recorded in `markup_sink`, tasks 1775/1776/2942).
- **D13 — A clean stranded blocked task is re-pended by a code row, never silently (S17-1/S17-4, option
  D).** The table's blocked rule "a silent return to `pending` is forbidden" existed because a deliberate park
  looked like a crash. Task 4082's rows name every park code can see (a queued merge, a recheck marker, a
  block behind a heal-halted lane, an open record that pins, including a record set holding one that needs a
  human, task 6145), so the shape left over is proven unparked: the row returns it to `pending` and emits
  `recovery_applied` with its rule name. The rule is relaxed for that shape only. Its aim, one classifier the
  sweeps apply verbatim (R6, R16), stands; R16's stated reason ("instead of performing the revert the
  recovery table forbids") does not, for this shape. Accepted risk: a bare `blocked` write by a person is
  re-pended within 900 s (today: within 15 s by the scheduler); `deferred` with a deferral record is the
  blessed park (task 6525, done) and ζ2 says so where writers look.

  **The loop cap, and why it cannot itself loop** (re-derived at the fold-verify against
  `task_ground_truth.py::_RECOVERY`, `harness.py::Harness._reconcile_one_stranded` and
  `recovery_pins.py::records_pin_blocked_recovery` at `fceb951e7e`). Two facts at main shape it. First, a
  `stranded_blocked` record does **not** pin a blocked task: it is the one member of
  `recovery_pins.py::MERGE_REMEDIABLE_ESC_CATEGORIES`, so `records_pin_blocked_recovery` relaxes past it
  (only a record that `escalation/pins.py::needs_human`, such as a level-2 promotion, keeps pinning). Today
  what holds a blocked task under its own open `stranded_blocked` L1 is the re-file branch's duplicate
  guard (`task_ground_truth.py::report_would_duplicate_a_handoff`), not a pin. Second, a watcher `resume`
  re-pends the task through `Harness._on_escalation_resolved`, outside the table, so it adds no
  `recovery_applied` event to the cap's count. The fold's text ("its open record then pins the task") was
  therefore wrong at main, and with only `repend_cap_reached` the over-cap row would re-file after every
  watcher resume inside the window: block, level-1, resume, block, level-1, without end. So C12 carries
  three things: (1) the clean row's shape requires that no record with an owner sits on the task (the
  duplicate guard's question, unchanged; an info record never counts, as 4082's 2026-10-09 re-scope rules),
  so a task under any owned open record, the over-cap row's own included, is held by a code row that leaves
  it under `escalation_pinned`; (2) a second fact, `repend_cap_escalated`, true when this task already has a
  `blocked_over_repend_cap` filing that landed inside the trailing window, read from the same event store;
  (3) a code row `blocked_repend_cap_escalated` ahead of the over-cap row that leaves such a task under a
  named reason, `REPEND_CAP_ESCALATED`. Per task and trailing 24 h the table can then do at most two code
  re-pends and one level-1 filing; the only further exits are the watcher's own judgement (resume or
  promote) and a human. Both facts are counts over `recovery_applied` events, so the cap survives a restart
  and keeps no new state (heuristic 7, INV-9).

  **Replay of the October runaway** (two solar-challenge tasks, 636 and 644: 3,896 scheduler flips and
  5,019 block cycles costing $674 from 2026-10-11 to 10-18, `digests/discussion/S17.md`). Each block carried
  only an info record (the task-6145 class), which neither pins nor dedups, so each block is a clean strand.
  Under the rule as written: block 1 → `blocked_clean_repend` (event 1); block 2 → re-pend (event 2);
  block 3 → `blocked_over_repend_cap`, one level-1 `stranded_blocked` for diagnosis; any block after that
  inside the window → `blocked_repend_cap_escalated`, left with no write and no filing. The loop stops on its
  third cycle. A trailing window lets the two re-pend events age out after 24 h, so at worst each task runs
  three cycles a day: about 21 per task, 42 for the pair over that week instead of 5,019, at the measured
  $0.134 a cycle about $6 instead of $674, plus at most 14 level-1 pickups. The 26 flips since task 5222
  (none repeated within 24 h) would all have been single clean re-pends, with no escalation.
- **D14 — The transient arm of an unevaluable delivered check is code (S17-2 under Leo's principle).**
  Whether a check ran, how many grace windows a `(dependent, dep)` pair has spent erroring and whether a
  reason is author-fixable are facts the gate holds, so the gate decides every case and a human gets the
  only case that needs one. The lead's reading named a level-1 record that the gate opens and closes
  itself; at main any pending level-1 record starts a watcher rotation
  (`harness.py::Harness._watcher_has_actionable_l1`) and the watcher promotes a category it has no playbook
  for (`skills/escalation-watcher-auto/SKILL.md`, "When unsure, PROMOTE"). So windows 1 and 2 are recorded as
  the gate's own events and file no queue record, and the third consecutive window files the born-at-L2
  record, which is what Leo sees in either reading (ruled by the lead 2026-11-13, S17-OQ-9 = A).
- **D15 — The phase-time alarm reads the one existing measurement (S17-2, Leo 2026-11-13).** The tick
  already times every phase for `scheduler_tick` (`Scheduler._emit_tick_telemetry`'s `phases`); λ hands the
  `delivered_check_gate` value to the gate rather than timing the phase a second time (INV-9). Crossing the
  threshold is a fact for code; what to do about a growing phase is an engineering judgement no LLM watcher
  can act on, so the filing goes straight to a human (born-at-L2, `agent_role='orchestrator-scheduler'`, as
  the scheduler's other human-level filings are), deduped to one open record per project. Severity
  (lead 2026-11-13: the lowest severity that still reaches Leo's queue; the severity named `critical`
  confirmed by the lead): **`critical`**. Why: at main `escalation/src/escalation/models.py::KNOWN_SEVERITIES` is
  `info`, `blocking`, `critical`, `urgent`, and only `BORN_AT_L2_SEVERITIES` (`critical`, `urgent`) are
  created at level 2; an `info` or `blocking` filing enters at level 0 or 1, where the LLM watcher would
  hold it and, having no playbook for it, promote it ("When unsure, PROMOTE"): a decidable case handed to an
  LLM, which the principle forbids. `critical` ranks below `urgent` in `SEVERITY_RANK`, so it is the floor
  of the human queue; a level-2 `blocking` record exists only by a watcher's promotion
  (S1 `FilingPolicy.promoted_l2`), which a filer may not claim. Offloading is not
  built; the remedy text points at the cheaper first step, a wall-clock budget for the phase that defers
  whole dependencies to the next tick the way the count budget already does.
- **D16 — The principle is recorded as a checked invariant, minted after its first conforming example.** ν
  depends on task 4082, whose `RECOVERY_RULES.decided_by` column and import-time validator are the house
  pattern; ν's fixtures are violating examples (a plan row that hands a predicate to an LLM watcher; a table
  row with no decider), so the walk proves the negative. Slug `one-decision-partition-per-area`; number
  INV-23, minted last of the six wave-3 invariant tasks (lead 2026-11-13; see `## Design invariant`).

## Decision partition

Leo's principle of 2026-11-12, cited by name as `one-decision-partition-per-area` (leaf ν). Every decision
area this plan adds or changes, one table each, with who decides each case. "Code" means a named row or
branch applied in one place; an LLM case names its path to a human.

Stranded blocked task — one table, `RECOVERY_RULES` (S8 δ 9219 plus task 4082's rows, C12), applied by the
`StrandedReconciler` (S8 ε2a 9223) every `stranded_reconcile_interval_secs`; the scheduler decides nothing
after ζ.

| Case | Row | Decided by | Outcome |
|---|---|---|---|
| Branch on main, nothing vetoing a done-flip | 9219's blocked on-main row | code | mark done |
| An open record pins (a merge-remediable set holding a record that needs a human pins too, task 6145) | 9219's pinned rows and default | code | leave; the record's own owner disposes of it |
| Deterministic task | the deterministic sweep's partition | code | its own sweep |
| Verified-green work exists | 9219's verified-green row | code | submit the merge |
| Its merge is queued | `blocked_merge_queued` | code | leave, `MERGE_QUEUED` |
| Parked for a recheck after a cancelled parent | `blocked_recheck_pending` | code | leave, `RECHECK_PENDING` |
| Blocked behind a lane halted for another task's heal | `blocked_behind_heal_halt` | code | leave, `BEHIND_HEAL_HALT` |
| A record with an owner sits on the task but does not pin (a merge-remediable `stranded_blocked`, the over-cap row's own included; a dead-filer L0) | `blocked_owned_record` (the re-file branch's duplicate guard, as a row) | code | leave, `escalation_pinned`; the record's owner disposes of it |
| The over-cap level-1 already landed inside the window, blocked again | `blocked_repend_cap_escalated` | code | leave, `REPEND_CAP_ESCALATED`; no write, no filing |
| Re-pended twice in the window and blocked again | `blocked_over_repend_cap` | LLM: the L1 watcher works out why it keeps blocking; escape: promote to a human | one level-1 `stranded_blocked` escalation, at most once per task per window |
| Clean (no live claimant, no record that pins or has an owner, off main, not deterministic, no park), under the cap | `blocked_clean_repend` | code | `pending`, `recovery_applied` naming the rule |

The table's other rows (9219's, for `in-progress` tasks) name their deciders too: mark done, revert to
pending and convert to blocked are code; the crashed-mid-deploy re-file (row (h) at main) is a human's,
because its record is the deterministic runner's own born-at-L2 escalation (the harness applier returns
before filing for any task with a deploy phase, `Harness._reconcile_one_stranded`'s first belt-and-braces
guard). `validate_recovery_rules` checks that every row says so.

Delivered check that cannot be evaluated — `DeliveredCheckGate` (δ2), per `(dependent, dep)`.

| Case | Decided by | Outcome |
|---|---|---|
| A window ends with an author-fixable reason | code, then a human | born-at-L2 `dependency_capability`; dependent blocked |
| A window ends with only transient reasons, windows 1–2 | code | `window` event, no queue record (OQ-9 = A); dependent pending and held |
| The third consecutive transient window ends | code, then a human | the same born-at-L2; dependent blocked |
| The check runs again | code | streak and window count reset; `cleared` event |
| More checks than the tick's budget | code (D6) | whole dependencies deferred to the next tick |

Delivered-check phase time — `DeliveredCheckGate.observe_phase_time` (λ).

| Case | Decided by | Outcome |
|---|---|---|
| Phase at or under `phase_alarm_secs` | code | nothing |
| Phase above it, no open alarm | code, then a human | one born-at-L2 `infra_issue` at severity `critical` for the project |
| Phase above it, alarm open | code | nothing new |

Dependency readiness (γ) and the tick's phase order (ε) are code only and hand nothing to anyone. No row in
these tables gives an LLM a pure predicate over facts the code holds: the one LLM row is the recurring-state
case the principle names, and it has a promote path. The `stranded_blocked` playbook in
`skills/escalation-watcher-auto/SKILL.md` changes from "check the predicate and resume" to that diagnosis
(task 4082), and the plan adds no other watcher playbook.

## Design invariant

ν performs the four-site lockstep edit `CONTRIBUTING.md` §6 describes. Slug
**`one-decision-partition-per-area`** (the stable id; every check anchors on it). Number: the next free
one when ν lands (lead S2/OQ-10). At `fceb951e7e` the last minted is INV-14; R51 reserves INV-15/16/17 for
S6 9140, S10 9242 and S7 9154; the six wave-3 minting tasks (S1 ω, S1's eval-hermeticity task, S2 ι, S16 ι,
S19 ε and this ν) take INV-18 to INV-23 in landing order, and the lead ruled 2026-11-13 that ν is minted
last: **INV-23**. The lead wires the invariant-file chain at decompose and corrects the number there if the
order changes; only the number in the four sites would change, never the slug.

Sites (`scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`): `docs/legibility/design-invariants.md`
(the section, the header dating paragraph, the `## Fixtures` date list);
`docs/legibility/design-invariants-fixtures.md` (`INV-N-PRD`, `INV-N-CODE`, and a dated addendum walk with its
two rehearsal rows and the cumulative match count); `skills/prd/references/gates.md` (the family-inventory row
and the `inv-trigger-shapes` span). `CONTRIBUTING.md` gets nothing; the partial heuristic mapping in
`orchestrator/src/orchestrator/agents/code_quality.md` is not edited. Draft text for the normative section:

> **Rule** (Leo, 2026-11-12): **One decision partition per decision area.** Each decision area has exactly
> one partition between what code decides and what an LLM judges. Code decides every case that facts
> available to code settle, in one table or function, applied in one place. An LLM gets only the cases that
> table names as needing judgement: facts that contradict each other, a state that keeps recurring, or a
> situation too complex to enumerate. Each such case has a path to a human. An LLM playbook step that is a
> pure predicate over facts the code already holds is a defect: move it into the table.
>
> **Checkable design question(s)**: Does this change add or change a recovery, triage or policy table row,
> an alarm rule or a watcher playbook step? Does every row name who decides it (code, an LLM, or a human)?
> Is any case that facts available to code settle handed to an LLM? Is any playbook step a pure predicate
> over facts the code holds? Does every LLM-judged case have a path to a human? Is the same decision area
> also decided somewhere else?
>
> **Evidence**: in October 2026 the scheduler's own stranded-blocked step and the harness's recovery table
> both decided stranded blocked tasks, and the level-1 watcher's `stranded_blocked` playbook was a
> three-condition predicate over facts the code held; the step re-pended two solar-challenge tasks 3,896
> times in one week, for 5,019 block cycles costing $674. Run `review-all-dark_factory-20261008`, sitting
> S17-1, measured on 2026-11-12.
>
> **House pattern**: `orchestrator/src/orchestrator/task_ground_truth.py::RECOVERY_RULES`, whose every row
> carries `decided_by`, checked at import by `validate_recovery_rules` (task 4082).
>
> **Family boundary**: INV-9 `one-fact-one-home` governs where a fact lives; this governs who decides a
> case. INV-3 `corroborate-before-acting` governs what a decider reads before it acts; this governs which
> decider it is. INV-7 `holds-owned-and-bounded` governs a held state; this governs the decision that holds
> or releases it.

Fixtures (both violating, so the walk proves the negative): `INV-23-PRD`, a plan leaf in which a sweep files
a level-1 escalation and the watcher "resumes the task if it is still blocked, has no active workflow and no
other open escalation" (expected `flag: one-decision-partition-per-area`; the redesign that clears it is a
code row that re-pends with an event); `INV-23-CODE`, a recovery table in which one re-file row names no
decider at all (no `decided_by` field, not merely a wrong one) and whose playbook comment restates the
predicate (expected `invariant_findings` entry naming that row). The addendum walk applies the checkable
question "Does every row name who decides it?" to both fixtures and records the answer "no" with the
offending row, and to the house pattern (`RECOVERY_RULES` after 4082) with the answer "yes": the check is
shown to fail on a row with no decider and to pass on a conforming table. The code half is mechanical too:
boundary row 14f constructs a `RecoveryRule` with no `decided_by` and expects a `TypeError` naming the
missing field (the field is required and has no default), so the form cannot be satisfied by omission.

```python
# C1 — shared/src/shared/scheduler_state.py (β)
class TaskLifecycle(BaseModel):                     # model_config = _RECORD_CONFIG (frozen, extra='ignore')
    dispatched: bool = False
    dispatched_priority: str | None = None          # projected only when dispatched
    requeue_count: int = 0
    transient_requeue_count: int = 0
    modules: tuple[str, ...] = ()                   # the lock modules held; projected only when dispatched
    in_merge_phase: bool = False
    workflow_cancelled: bool = False

class SchedulerStateSnapshot(BaseModel):            # existing fields unchanged, plus:
    tasks: Mapping[str, TaskLifecycle] = Field(default_factory=dict)
    # a task appears iff dispatched or requeue_count > 0 or transient_requeue_count > 0
    #   or in_merge_phase or workflow_cancelled

# C2 — orchestrator/tests/_in_memory_task_store.py and orchestrator/tests/_scheduler_scenario.py (β)
class UnscriptedTool(Exception): name: str
class InMemoryTaskStore:                            # implements McpSessionLike
    def add(self, task_id: str, *, status: str = 'pending', dependencies=(), metadata=None, **fields) -> None
    async def call_tool(self, name: str, arguments: dict, *, timeout: float = ...) -> dict
    # answers: get_tasks, get_statuses, get_task, set_task_status, set_task_claimant, update_task,
    #   submit_task, get_external_statuses, add_dependency (workflow.py), commit_planning,
    #   find_tasks_by_metadata, search_tasks (chronic_flake.py), resolve_ticket (escalation_router.py);
    #   any other name raises UnscriptedTool(name)
    def record(self, task_id: str) -> dict          # deep copy of the stored row
    def status_history(self, task_id: str) -> tuple[str, ...]
    def writes(self, task_id: str) -> tuple[StoreWrite, ...]   # frozen (tool, arguments) per write
    def refuse(self, tool: str, *, error_type: str, message: str) -> None   # scripted rejection

class SchedulerScenario:
    @classmethod
    def build(cls, tmp_path: Path, *, config: OrchestratorConfig | None = None,
              callbacks: SchedulerCallbacks | None = None) -> SchedulerScenario
    scheduler: Scheduler; store: InMemoryTaskStore; events: RecordingEventStore
    overrides: OverrideStore; clock: ScenarioClock; psi: ScriptedPsi
    async def tick(self) -> TaskAssignment | None                     # acquire_next
    def release(self, task_id: str, *, requeued: bool = False) -> None
    def requeue(self, task_id: str, **record_requeue_kwargs) -> None  # record_requeue
    def set_override(self, task_id: str, **fields) -> None            # OverrideStore.set_override
    def advance(self, secs: float) -> None                            # monotonic and wall together
    def snapshot(self) -> SchedulerStateSnapshot                      # get_state_snapshot
    def events_of(self, event_type: EventType) -> list[dict]
# Scheduler (β): two constructor keywords, production defaults unchanged
#   psi_reader: Callable[[], PsiSample] | None = None
#   hold_history: HoldHistory | None = None

# C3 — orchestrator/src/orchestrator/dependency_verdict.py (γ)
@dataclass(frozen=True)
class ExternalDeps:                                 # resolved(statuses) | resolver_failed()
    statuses: Mapping[str, str] | None              # None only for resolver_failed
@dataclass(frozen=True)
class DeliveredChecks:                              # projection(verdicts) | gate_off()
    verdicts: Mapping[str, bool] | None             # None only for gate_off
@dataclass(frozen=True)
class DepGates:                                     # no field has a default
    status_map: Mapping[str, str]
    tasks_by_id: Mapping[str, dict]
    terminal_dep_records: Mapping[str, dict]
    deferred_terminal_deps: frozenset[str]
    external: ExternalDeps
    delivered: DeliveredChecks
def deps_verdict(task: dict, gates: DepGates) -> DepsVerdict   # arms and combine rule unchanged

# C4 — orchestrator/src/orchestrator/delivered_checks.py (δ1)
class ErroredReason(StrEnum):
    TIMEOUT = 'timeout'                             # exceeded check_timeout_secs
    MALFORMED = 'malformed'                         # descriptor fails DeliveredCheckMeta
    UNEVALUABLE = 'unevaluable'                     # git rejected the pattern or pathspec (rc >= 2 at a resolved ref)
    SCRIPT_UNRUNNABLE = 'script_unrunnable'         # OSError spawning the script: missing or not executable
    RUNNER_RAISED = 'runner_raised'                 # any other exception from the runner
    @property
    def author_fixable(self) -> bool: ...           # MALFORMED, UNEVALUABLE, SCRIPT_UNRUNNABLE
@dataclass(frozen=True)
class CheckEvaluation:
    check: Mapping[str, Any]; result: DeliveredCheckResult; reason: ErroredReason | None   # reason iff ERRORED
@dataclass(frozen=True)
class DeliveredChecksVerdict:                       # existing outcome/failed_check kept; main_sha renamed ref
    outcome: Literal['all_delivered', 'failed', 'errored']
    ref: str
    evaluations: tuple[CheckEvaluation, ...]
    failed_check: Mapping[str, Any] | None
async def run(checks: Sequence[Mapping[str, Any]], *, project_root: str | Path, ref: str,
              check_timeout_secs: float, runner: GitRunner = <S5 C1 default>) -> DeliveredChecksVerdict
# never raises; every check is bounded by check_timeout_secs; checks run concurrently, order preserved;
# ref is resolved by the caller (DeliveredCheckGate resolves shared.delivered_check_polarity.GATE_REF)

# C5 — orchestrator/src/orchestrator/delivered_check_gate.py (δ2)
class DeliveredCheckGate:
    def __init__(self, *, config: DeliveredChecksConfig, project_root, event_store,
                 escalations: SchedulerEscalations,          # S1 C5, from SchedulerCallbacks.escalations
                 errored_streaks: KeyedStreaks[str], run_checks=run): ...
    async def evaluate(self, *, deps_by_dependent: Mapping[str, tuple[str, ...]],
                       dep_records: Mapping[str, dict], status_map: Mapping[str, str]) -> DeliveredChecks
# config.py: DeliveredChecksConfig.errored_grace_sweeps: int = Field(default=20, ge=1)
#   beside grace_cycles; 20 is today's _DELIVERED_CHECK_ERROR_LOG_THRESHOLD, which is deleted
# a grace window = errored_grace_sweeps consecutive ERRORED sweeps of one dep; the streak (keyed by dep,
#   KeyedStreaks I7) resets on any non-ERRORED evaluation, so windows are consecutive while it is unbroken
# EventType.delivered_check_errored: data {dep_id, dependents, main_sha, checks: [{name, reason}], streak,
#   phase: 'started' | 'window' | 'cleared', window: int}
#   'started' on the sweep a dep's ERRORED run starts; 'window' on each sweep the streak reaches a multiple
#   of errored_grace_sweeps; 'cleared' on the first non-ERRORED evaluation after at least one window closed
# at each window end, once per (dependent, dep):
#   some errored check's reason is author_fixable, or this is the third consecutive window
#     -> escalations.file_and_block(dependent, draft, FilingPolicy.born_at_l2(...), expected_status='pending')
#        category dependency_capability, the FAILED path's filing; dependent blocked; the draft names each
#        errored check, its ErroredReason and the window count
#   otherwise (a transient reason in window 1 or 2) -> no queue record (lead, OQ-9 = A); the 'window' event
#        is the gate's own record; the dependent stays pending and held
# the filing's policy is FilingPolicy.born_at_l2() through S1 C5; subject = the dependent task id, category
#   dependency_capability, agent_role 'orchestrator-scheduler' (S1's door dedupes on subject and category)
# after a born-at-L2 filing the pair's window count restarts at zero: a dep still erroring files again after
#   another window (author-fixable) or three (transient), and S1's door dedupes against an open record of the
#   same subject and category, so only a closed record yields a new one (INV-7: the hold is re-armed;
#   INV-4: one event per window, never one per tick)
# config.py: errored_grace_sweeps is green-tier hot-reloadable like its siblings (a RELOADABLE_FIELDS
#   entry, or S16's Reload.GREEN marker if S16 β1 has landed); DeliveredCheckGate reads it at evaluation
#   time from the held submodel (config.py::_set_leaf preserves submodel identity)

# C6 — orchestrator/src/orchestrator/tick_phases.py (ε)
# `from __future__ import annotations`; Scheduler, PsiHold, TaskAssignment and OverrideRow are imported
# under TYPE_CHECKING only (TickContext's docstring names the circular import as the reason it lives in
# scheduler.py today); TaskAssignment and PsiHold stay in scheduler.py
UNSET: Final = _Unset()                             # sentinel; reading it raises TickFieldUnset(field, producer)
@dataclass(frozen=True)
class TickPhase:
    name: str
    run: Callable[[Scheduler, TickContext], Awaitable[object]]   # an unbound method reference, never a string
    reads: frozenset[str]
    writes: frozenset[str]
class PhaseOrderError(Exception): phase: str; field: str; producer: str | None
def validate_phase_order(phases: Sequence[TickPhase], inputs: frozenset[str]) -> None
# scheduler.py: TICK_PHASES: tuple[TickPhase, ...] declared after class Scheduler; validated at import
# inputs: tasks, fetched_status_map, tasks_by_id, max_id; status_map is produced by backfill_dep_status

# C7 — orchestrator/src/orchestrator/dispatch_ledger.py (η)
@dataclass(frozen=True)
class TaskRecord:
    task_id: str; dispatched: bool; dispatched_priority: str | None; last_dispatch_at: float | None
    requeue_until: float | None; requeue_cooldown: RequeueCooldown | None; requeue_count: int
    transient_requeue_count: int; requeue_history: tuple[RequeueRecord, ...]
    pending_transient_cooldown: int; skip_count: int; modules: tuple[str, ...] | None
    fallback_warned: bool; workflow_cancel_at: float | None; merge_phase_at: float | None
class DispatchLedger:
    def get(self, task_id: str) -> TaskRecord       # the all-default record when absent
    def note_dispatched(self, task_id: str, *, priority: str, modules: tuple[str, ...], at: float) -> TaskRecord
    def released(self, task_id: str, *, requeued: bool, cooldown: RequeueCooldown | None) -> TaskRecord
    def requeued(self, task_id: str, record: RequeueRecord, *, transient: bool) -> TaskRecord
    def retain(self, live_task_ids: Collection[str]) -> frozenset[str]   # the one GC; returns removed ids
    def lifecycles(self) -> Mapping[str, TaskLifecycle]                  # C1's projection and inclusion rule
    # plus one method per remaining transition, each replacing one prose-stated lockstep rule

# C8 — orchestrator/src/orchestrator/backfill_admission.py, warm_base_watchdog.py (θ)
class BackfillAdmission:      # grants, outcome window, scan cache, HoldHistory, overstay-rate escalation
    def admit(self, ...) -> BackfillGrant | None; def settle(self, task_id: str, ...) -> None
    def predicted_hold(self, ...) -> float | None; def predicted_remaining(self, holder: str) -> float | None
class WarmBaseHardDownWatchdog:                     # the singleton latch, window and L2 promotion
    async def observe(self, *, now: float) -> None
# both take escalations: SchedulerEscalations (S1 C5) at construction

# C9 — orchestrator/src/orchestrator/mcp/markup_sink.py (μ)
@dataclass(frozen=True)
class MarkupChannels:
    resolve_project_root: Callable[[Path], Path | None]
    open_escalation_channel: Callable[[Path], tuple[Any, Any] | None]
    fact_journal: Callable[..., MarkupFactSink]      # markup_journal.make_fact_journal in production
    @classmethod
    def production(cls) -> MarkupChannels
# plan_tools.create_server(artifacts, *, channels: MarkupChannels) -> FastMCP
# verdict_tools.create_server(artifacts, role, session_id='', *, channels: MarkupChannels) -> FastMCP

# C10 — orchestrator/src/orchestrator/deterministic_state.py and the step modules (ξ1, ξ2)
@dataclass(frozen=True)
class GateStamps:
    gate_escalated_at: str | None; before_done_ran_at: str | None
    before_done_verified_at: str | None; before_done_scheduled_at: str | None
@dataclass(frozen=True)
class RunnerState:
    task_id: str; metadata: TaskMetadata; before_done: BeforeDone | None
    deploy: DeployState | None; stamps: GateStamps; curator_gate: bool
    @classmethod
    def parse(cls, task: Mapping[str, Any]) -> RunnerState   # parse_metadata(direction='read'); never raises
class RunStep(Protocol):
    async def run(self, state: RunnerState) -> RunOutcome
# orchestrator/src/orchestrator/deploy_state.py (ξ2), module level:
def from_legacy_stamps(stamps: GateStamps) -> DeployState | None
#   GateStamps imported under TYPE_CHECKING only (deterministic_state.py imports deploy_state.py)

# C11 — shared/src/shared/task_fields.py (α0); stdlib only
def dependency_ids(task: Mapping[str, Any]) -> list[str]         # a fresh list per call (fused-memory's
#   callers pass it on as `ids: list[str]`, e.g. get_statuses_raw); the union of the scheduler's and
#   fused-memory's tolerances: `dependencies` as a list/tuple/set whose entries are {'id': x} | int | str
#   (falsy entries dropped), or a CSV string (pieces stripped, empty pieces dropped); any other non-empty
#   shape is stringified whole; order kept; never raises
def train_id(task: Mapping[str, Any]) -> str | None              # metadata.train.id when a non-empty value, else None
def is_deterministic(task: Mapping[str, Any]) -> bool            # metadata (dict or JSON string) task_kind == 'deterministic'
# fused-memory/src/fused_memory/utils/task_dependency_ids.py is deleted (no re-export, lead S17/OQ-4); each
#   of its four importers changes only its import line, to
#   `from shared.task_fields import dependency_ids as <the local name it uses today>`

# C12 — orchestrator/src/orchestrator/task_ground_truth.py, recovery_emission.py, stranded_reconcile.py,
#   event_store.py, config.py (task 4082, re-scoped by the sitting; extends S8 C2, which 9219 lands)
class Decider(StrEnum):
    CODE = 'code'; LLM = 'llm'; HUMAN = 'human'
@dataclass(frozen=True)
class RecoveryRule:                                 # S8 C2's fields, plus:
    decided_by: Decider                             # required; CODE iff action != RE_FILE_ESCALATION; a
                                                    #   RE_FILE row names the tier its filing reaches
def validate_recovery_rules(rules: Sequence[RecoveryRule]) -> None   # run at import; raises naming the row
@dataclass(frozen=True)
class TruthReport:                                  # S8 C2's fields (merge_in_flight included), plus:
    recheck_pending: bool                           # the row carries needs_recheck_against_main
    behind_heal_halt: bool                          # a structured marker written by the main-health block
                                                    #   AND that lane is still halted (read live, INV-3)
    repend_cap_reached: bool                        # this task's blocked_clean_repend recovery_applied events
                                                    #   in the trailing window >= the cap (read at call time)
    repend_cap_escalated: bool                      # a blocked_over_repend_cap filing for this task landed
                                                    #   (recovery_applied, outcome APPLIED, never DEDUPED) inside
                                                    #   the same trailing window (read at call time)
    owned_record_open: bool                         # report_would_duplicate_a_handoff(report): a record with an
                                                    #   owner sits on the task (info never counts; a dead L0 does)
# both counts come from one injected counter over the event store; the cap and window values reach it at
#   construction, so no row condition reads config (S8 C2's rule)
# LeaveReason (recovery_emission.py) gains MERGE_QUEUED, RECHECK_PENDING, BEHIND_HEAL_HALT, REPEND_CAP_ESCALATED
# blocked rows, after 9219's on-main and pinned rows, the deterministic hand-off and the verified-green submit:
#   name                          condition              action                        decided_by
#   blocked_merge_queued          merge_in_flight        LEAVE (MERGE_QUEUED)           CODE
#   blocked_recheck_pending       recheck_pending        LEAVE (RECHECK_PENDING)        CODE
#   blocked_behind_heal_halt      behind_heal_halt       LEAVE (BEHIND_HEAL_HALT)       CODE
#   blocked_owned_record          owned_record_open      LEAVE (escalation_pinned)      CODE
#   blocked_repend_cap_escalated  repend_cap_escalated   LEAVE (REPEND_CAP_ESCALATED)   CODE
#   blocked_over_repend_cap       repend_cap_reached     RE_FILE_ESCALATION (level 1)   LLM (diagnose; promote = human)
#   blocked_clean_repend          (shape only)           REVERT_TO_PENDING              CODE
# the clean row covers both blocked branch shapes off main (exists off main; gone without marker); it
#   replaces the two blocked re-file rows, whose re-file becomes the over-cap row. blocked_owned_record
#   carries today's re-file duplicate guard into the table: a stranded_blocked record is merge-remediable and
#   so does not pin (recovery_pins.py::records_pin_blocked_recovery relaxes it), and without this row the
#   clean row would re-pend a task under its own open level-1
# loop bound: per task and trailing window, at most stranded_blocked_repend_cap code re-pends and one
#   level-1 filing; a watcher resume (Harness._on_escalation_resolved) is the LLM's decision and adds no
#   blocked_clean_repend event; no row can fire twice in a window without a counted event (D13)
# EventType.recovery_applied: data {rule, action, from_status, to_status, outcome}; one per applied action
# config.py, green tier, beside stranded_reconcile_interval_secs:
#   stranded_blocked_repend_cap: int = Field(default=2, ge=1)
#   stranded_blocked_repend_window_secs: float = Field(default=86400.0, gt=0)

# C13 — orchestrator/src/orchestrator/delivered_check_gate.py, scheduler.py, config.py (λ)
class DeliveredCheckGate:                           # C5's class, plus:
    async def observe_phase_time(self, phase_ms: int) -> None
# called by Scheduler._emit_tick_telemetry (after ε, by the phase table's telemetry) with the same
#   phases['delivered_check_gate'] value the scheduler_tick event records — one measurement, two readers
# config.py: DeliveredChecksConfig.phase_alarm_secs: float = Field(default=30.0, gt=0), green tier
# above phase_alarm_secs: escalations.file(draft, FilingPolicy.born_at_l2(...)), severity 'critical' (the
#   lowest of BORN_AT_L2_SEVERITIES, so the floor of the human queue; D15), category infra_issue,
#   agent_role 'orchestrator-scheduler', subject the project sentinel '__delivered_check_phase_slow__'
#   (a Sentinel member of S1's sentinel-id registry in escalation/models.py, S2/OQ-6);
#   S1's door dedupes against the open record of that subject and category (one open alarm per project)
#   summary: the project and the phase time; detail: the checks run that tick (dep, check name, outcome,
#   elapsed) slowest first, and the remedy "add a wall-clock budget for the phase that defers whole
#   dependencies to the next tick, as max_checks_per_tick does for the count"
```

Invariants and error semantics.

- C1: `from_json(s.to_json()) == s` (S14 I4) with `tasks` populated; an older reader ignores `tasks`
  (`extra='ignore'`); a task is emitted iff C1's inclusion rule holds; a scored but never-dispatched task is
  never emitted; after η the section is `DispatchLedger.lifecycles()` and nothing else.
- C2: the builder and the store touch no `_`-prefixed attribute of `Scheduler` (the D2 guard scans them);
  the store's envelopes are the ones fused-memory sends (an `isError` reply and a structured rejection are
  scriptable through `refuse`); a tool outside C2's list raises `UnscriptedTool(name)`; `advance` moves the
  monotonic and the wall clock by the same amount; `test_scheduler_private_reach.py` matches
  `PRIVATE_REACH_PATTERN` on zero lines at every commit (rows 4 and 4b assemble their planted reach from
  separate tokens under `tmp_path`; row 4c).
- C3: constructing `DepGates` without a field is a `TypeError`; `ExternalDeps.resolver_failed()` makes every
  external dep UNKNOWN; `DeliveredChecks.gate_off()` makes the delivered arm SATISFIED, as the kill switch
  does today; the verdict for a given task and gates equals today's `deps_verdict` with the same caches
  passed (golden test generated before γ over the four arms).
- C4: `run([])` returns `errored` (today's empty-list rule); a timeout is `ERRORED` with reason
  `timeout`; an `OSError` spawning a script is `script_unrunnable`; `failed_check` is the first FAILED check
  in input order; the module holds the one collapse rule and no caller re-derives it;
  `verify_delivered_checks_on_main` no longer exists.
- C5: a dep's ERRORED run emits one `delivered_check_errored` event when it starts, one at each window end
  and one when it clears, never one per tick (INV-4); at a window end it files one born-at-L2
  `dependency_capability` per `(dependent, dep)`, with the dependent blocked, when some errored check's
  reason is author-fixable or the window is the third consecutive one; a transient window 1 or 2 files no
  queue record at any level and leaves the dependent pending and held (OQ-9 = A); the streak, keyed by dep, resets on any
  non-ERRORED evaluation (I7 semantics of `KeyedStreaks`) and with it the window count; after a filing the
  pair's window count restarts, so a dep still erroring files again, and S1's door dedupes against an open
  record of the same subject and category, so only a closed record yields a new one; every escalation names
  each errored check, its `ErroredReason` and the window count; no LLM watcher step exists for this category;
  INV-11, INV-7.
- C6: reading a producer field before its producer ran raises `TickFieldUnset` naming the field and the
  producer; `validate_phase_order(TICK_PHASES, INPUTS)` passes at import; a reordered table raises
  `PhaseOrderError`; each `TickPhase.reads`/`writes` is a superset of the AST-derived set; no `getattr`
  builds a phase name.
- C7: `retain` is the only removal path; a record is never mutated in place; `lifecycles()` is a pure
  projection; no other scheduler attribute is keyed by task id except the five named in Sketch step 7.
- C8: neither module imports `orchestrator.scheduler`; tests construct both directly.
- C9: `create_server` without `channels` is a type error; the resolver is never called during
  `create_server`; both servers take the same type.
- C10: `RunnerState.parse` never raises (a malformed row parses to a state the gate step escalates, as
  today); exactly one step runs per `run`; every escalation summary, detail and log line pinned by
  `test_deterministic_runner.py` is byte-identical (ξ1 is behaviour-neutral); ξ2's only behaviour change is
  the deploy step's resume authority (D10), pinned by the golden table of boundary row 19.
- C11: `dependency_ids` never raises and returns today's scheduler normalisation on every shape it handles
  (`{'id': 1}`, `1`, `'1'`, `{'id': '1'}`, a dict without `id`) and fused-memory's on every shape
  `fused-memory/tests/test_task_dependency_ids.py` pins (CSV strings, falsy entries, a bare int, a dict
  as the whole value); `train_id` coerces `''`, `None` and a non-dict `train` to `None` (today's
  `dependency_verdict` rule; `tasks_by_train` and `DeclaredTrainRef.from_task` adopt it);
  `fused-memory/src/fused_memory/utils/task_dependency_ids.py` does not exist and no module re-exports the
  reader under its old path.
- C12: the first matching row wins and the default stays LEAVE; every row names its decider and
  `validate_recovery_rules` refuses, at import, a row whose `decided_by` disagrees with its action (a
  `RE_FILE_ESCALATION` row named CODE, or any other row named LLM or HUMAN), naming the row; a row's
  condition reads only `TruthReport` facts, never config (S8 C2's rule), so the cap reaches the table as
  `repend_cap_reached`; a blocked task carrying a structured park marker is never re-pended while its
  park holds; `blocked_clean_repend` always emits `recovery_applied` naming itself and never fires for a task
  with an owned open record; `blocked_over_repend_cap` files at most once per task per window, enforced by
  `repend_cap_escalated` and the `blocked_repend_cap_escalated` row, not by a pin (a `stranded_blocked`
  record does not pin a blocked task at main; while it is open `blocked_owned_record` holds the task); a
  DEDUPED re-file is not counted as a filing; `validate_recovery_rules` refuses a row whose `decided_by`
  disagrees with its action, and a row with no `decided_by` cannot be constructed; the watcher's
  `stranded_blocked` playbook asks why the task keeps blocking and ends in a resume after the cause is fixed
  or a promote, never a bare close.
- C13: at most one open phase-time record per project, at severity `critical`; a tick at or under `phase_alarm_secs` files nothing;
  the alarm never moves a check off the tick or skips one; a failure to file is logged and never fails the
  tick (the telemetry path's PROPERTY-1 fail-open rule).

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | builder ↔ scheduler | dispatch then release with requeue | scenario with tasks A (pending, module m) | after `tick()`: `snapshot().tasks['A'].dispatched`, `current_holders['m'] == 'A'`; after `release('A', requeued=True)`: not dispatched, `requeue_count == 1`, `requeue_cooldowns['A']` present |
| 2 | builder ↔ store | a structured `set_task_status` rejection | `store.refuse('set_task_status', …)` | the scheduler raises the typed rejection the client maps today; `store.status_history` unchanged |
| 2b | builder ↔ store | an unscripted tool | the scheduler or a `dispatch_tool` caller sends a tool outside C2's list | `UnscriptedTool(name)` raised naming the tool |
| 3 | snapshot wire | `tasks` round trip and old reader | snapshot with two lifecycles | `from_json(to_json())` equal; a reader model without `tasks` parses the same JSON |
| 3b | snapshot size | many scored, undispatched tasks | 500 pending tasks scored over several ticks, none dispatched | `snapshot().tasks == {}`; the serialized snapshot grows by less than 1 KB against the same scenario with 0 tasks |
| 4 | guard | private reach planted | a file under `tmp_path` whose line is assembled at run time from `receiver = 'scheduler'`, `'.'` and `attr = '_dispatched'`, beyond its baseline | `test_scheduler_private_reach.py` fails naming the file and count; at ω any reach fails |
| 4b | guard | prefixed receiver planted | as 4, with `receiver = 'real_scheduler'` | the guard fails naming the file (the pattern covers `real_`/`fresh_` receivers) |
| 4c | guard | the guard's own source | the guard run over `orchestrator/tests/test_scheduler_private_reach.py` | zero matching lines; no file is exempt from the count |
| 5 | predicate | equivalence of `DepGates` (γ) | golden table built before γ over local/scheduled-deploy/external/delivered arms | `deps_verdict(task, gates)` equals the golden verdict row for row |
| 6 | predicate | milestone anchor waits for a failing delivered check (S17-3) | delayed milestone M depends on done D whose check FAILS | no `milestone_deps_satisfied_at` write; after the check passes the anchor is written that tick |
| 7 | predicate | missing field | `DepGates(...)` without `delivered` | `TypeError` at construction |
| 8 | evaluator | one check hangs (δ1) | fake runner that never returns for check 2 | `run` returns within `check_timeout_secs`; evaluation 2 is ERRORED/`timeout`; outcome per collapse |
| 8b | gate budget (δ2; D6 a) | a dep whose check list exceeds the remaining budget | `max_checks_per_tick=3`; dep D1 (1 check) evaluated first, dep D2 with 3 checks second | D2 is deferred whole: no check of D2 runs this sweep and D2 stays uncached; next sweep D2 runs first and whole |
| 8c | evaluator (δ1; D6 b) | a dep's checks run concurrently | two checks whose fake runner each sleeps `t` | `run` returns in under `2t`; evaluations in input order |
| 9 | evaluator ↔ landing evidence | differential through `run` | tmp git repo: check absent at parent, present at citation and main | leg verdict `confirmed`; a hung check yields `no_signal` within the timeout, not a hang |
| 10 | gate | an author-fixable ERRORED for the grace period (δ2) | dep D's script missing; dependent T pending; `errored_grace_sweeps=20` | sweep 1: one `delivered_check_errored` event (`started`); sweeps 2–19: none; sweep 20: a `window` event and one born-at-L2 `dependency_capability` for `(T, D)` naming the check, `script_unrunnable` and window 1; T `blocked` |
| 10b | gate | one transient window (δ2; OQ-9 = A) | dep D's grep check times out every sweep; dependent T pending | sweep 20: a `window` event with `window: 1` naming `timeout`; no escalation record of any level; T stays `pending` and is not dispatched |
| 10c | gate | three consecutive transient windows (δ2; OQ-9 = A) | as 10b for 60 sweeps | sweeps 20 and 40 file nothing; sweep 60: one born-at-L2 `dependency_capability` for `(T, D)` naming `timeout` and window 3; T `blocked`; a recording `SchedulerEscalations` saw exactly one filing and no level-1 filing |
| 10d | gate | the check runs again after a transient window (δ2) | as 10b; the check returns DELIVERED at sweep 25 | one `cleared` event at sweep 25; streak and window count reset; the next ERRORED run starts at window 0, so a later timeout needs three fresh windows |
| 11 | gate | ERRORED then DELIVERED before a window ends | as 10 but the script appears at sweep 5 | no escalation and no `cleared` event; streak reset; next ERRORED run starts at 1 |
| 11b | phase alarm (λ) | a slow `delivered_check_gate` phase | `phase_alarm_secs=30`; a scripted runner makes the phase take 31 s on two consecutive ticks | tick 1: one born-at-L2 `infra_issue` at severity `critical` under the project sentinel naming the project, 31 s and each check run slowest first, with the wall-clock-budget remedy; tick 2: no second record while the first is open; a 29 s phase files nothing; the `scheduler_tick` event's `phases` value equals the value the gate saw |
| 12 | tick | phase reads an unset field | a phase table with `stamp_milestone` moved before `external_dep_policy` | `validate_phase_order` raises `PhaseOrderError(phase='stamp_milestone', field='dep_gates', producer='dep_gates')` (after γ the phase reads `dep_gates`, which replaces today's `external_cache` load) |
| 12b | tick | declared sets cover the body | every phase in `TICK_PHASES` | each `reads` ⊇ its AST-derived loads (helpers receiving `ctx` included) and each `writes` ⊇ its AST-derived stores and in-place mutations |
| 13 | tick | early read at runtime | a test phase reading `ctx.dep_gates` before it is written | `TickFieldUnset('dep_gates', 'dep_gates')` |
| 14 | recovery | a clean strand (ζ; task 4082) | task B `blocked`, no claimant, no record that pins, branch off main, not deterministic, no park marker; scheduler ticks for 60 s, then one `StrandedReconciler` pass | no scheduler write during the 60 s; the pass writes `pending` and emits `recovery_applied` with `rule='blocked_clean_repend'`; no escalation is filed |
| 14b | recovery | merge queued (task 4082) | B `blocked` by the architect-desync park with its merge in flight (`TruthReport.merge_in_flight`) | LEAVE with `MERGE_QUEUED` and `recovery_left` naming `blocked_merge_queued`; no `pending` write, no filing, while the merge is in flight |
| 14c | recovery | recheck park (task 4082) | B carries `needs_recheck_against_main` from `targeted.py::_sweep_block_orphan` | LEAVE with `RECHECK_PENDING`; no `pending` write |
| 14d | recovery | behind a heal-halted lane (task 4082) | B blocked by `workflow.py::TaskWorkflow._auto_heal_main_health`'s lane-already-halted branch; the normal lane halted | LEAVE with `BEHIND_HEAL_HALT`; after the lane unhalts, the next pass applies row 14 |
| 14e | recovery | the loop cap (task 4082) | B re-pended by `blocked_clean_repend` twice in the trailing 24 h, blocked again | the pass files one `stranded_blocked` level-1 escalation naming `blocked_over_repend_cap` and writes no `pending`; a later pass while it is open leaves B under `blocked_owned_record` (`escalation_pinned`), although `records_pin_blocked_recovery` relaxes that record; after a restart the count is the same (read from the event store); with `stranded_blocked_repend_cap` hot-reloaded to 3 the pass re-pends once more |
| 14f | recovery | every row names its decider (task 4082; the invariant's code fixture) | `validate_recovery_rules` over `RECOVERY_RULES`, over a copy whose `RE_FILE_ESCALATION` row says CODE, over a copy whose `REVERT_TO_PENDING` row says LLM; and a `RecoveryRule(...)` constructed with no `decided_by` | the real table passes; each altered copy raises naming the row; the construction without a decider raises `TypeError` naming `decided_by` |
| 14g | recovery | the cap cannot itself loop (task 4082) | as 14e, then the watcher resolves the level-1 with `resume` (`Harness._on_escalation_resolved` re-pends B), and B blocks again twice inside the window; then the window passes | after the resume no pass writes `pending` and none files: each leaves B under `blocked_repend_cap_escalated` (`REPEND_CAP_ESCALATED`); across the whole window the store shows exactly two `blocked_clean_repend` writes and one level-1; a DEDUPED re-file is not counted; once both re-pend events and the filing have aged out of the window, the next clean block is re-pended once by `blocked_clean_repend` |
| 15 | recovery | info-only L2 (task 6145's loop shape) | B `blocked` under an open info L2 only | no scheduler write; the sweep treats the info record as neither pinning nor owned (by design, 4082's 2026-10-09 re-scope) and applies row 14; if B keeps re-blocking, row 14e stops the loop on its third block and row 14g keeps it stopped for the window (the October 636/644 replay, D13) |
| 16 | ledger | GC (η) | records for A, B; tick whose active fetch holds only A | `retain` removes B; snapshot has no `B`; `requeue_history('B') == []` |
| 17 | ledger | the six aliases are gone | — | `test_scheduler_private_reach.py` and an attribute probe find none of `_external_unresolved_counts`, `_local_backfill_unresolved_counts`, `_external_hold_streak`, `_external_hold_cause`, `_external_resolver_degraded_counts`, `_starvation_escalated` |
| 18 | runner | step dispatch (ξ1) | pure gate, predicate, deploy rows | exactly one step runs; outputs byte-identical to the pre-ξ1 golden capture of `test_deterministic_runner.py`, captured on ξ1's rebased base |
| 19 | runner | crash-window resume from the phase (ξ2) | rows with `deploy_state.phase` ∈ {SCHEDULED, RAN, VERIFIED, FAILED, ESCALATED, DONE} × stamp sets, plus legacy rows with stamps and no `deploy_state` | the deploy step's decision equals the golden table generated before ξ2 from today's stamp logic for every row today handles; rows where stamps and phase disagree follow the phase and are listed in the task record |
| 20 | markup | channels injected (μ) | `create_server(artifacts, channels=fake)` where `fake.resolve_project_root` records calls | no call during `create_server`; the first leaked-markup record resolves once and lands in the fake channel; no `monkeypatch` of a `plan_tools._` name in the four test files |
| 21 | facade | one implementation (κ1, κ2) | `TaskWorkflow` constructed with `SchedulerScenario.build(...).scheduler` | a workflow test drives a status write and reads it from `store.status_history`; `SchedulerFacade`, `FakeScheduler` and `TaskStoreDouble` are importable from nowhere |
| 22 | readers | one normalisation (α0, α) | AST scan of `orchestrator/src`, `shared/src` and `fused-memory/src` | no dependency-id normalisation (`d.get('id', d)` under an `isinstance(d, dict)` test, or a split of `dependencies` on `','`) and no `task_kind` comparison to `'deterministic'` outside `shared/task_fields.py`; no module in `fused-memory/src` defines a dependency-id parser and `utils/task_dependency_ids.py` does not exist; no `metadata['train']` id read in `scheduler.py`, `dependency_verdict.py` or `merge_deferred_recovery.py` |
| 23 | invariant (ν) | the family stays consistent | `scripts/tests/test_design_invariants_consistency.py` after ν | passes with `one-decision-partition-per-area` in every pinned site; its fixtures `INV-23-PRD` and `INV-23-CODE` are violating examples and the addendum walk flags both |

## Cross-PRD relationship

| Other PRD / stream / task | Direction | Seam mechanism | Owner | Edges (X → Y: X depends on Y) |
|---|---|---|---|---|
| S8 harness composition root (`plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md`) | consumes / extended by task 4082 | C3 (`make_scheduler`, `passthrough_gates`, required gates; β0 9217, β 9218); C2 (`RECOVERY_RULES`, `recovery_for`; δ 9219), which task 4082 extends with C12 (a `decided_by` column, four leave reasons, five `TruthReport` facts, the seven blocked rows, `outcome` on `recovery_applied`); `StrandedReconciler` (ε2a 9223), the one applier of those rows | S8 | β → 9234 (implies 9217, 9218 through the `scheduler.py` chain); ζ → 9219, 9223, 4082. S8's C3 adds `SchedulerCallbacks.recovery_for` "for S17's phase"; ζ deletes that phase, so the lead amends 9219 to add no scheduler recovery hook (ruled 2026-11-12), or ζ deletes the field if 9219 already landed it. Task 6143's landed-outbox item is S8 β's; its PSI-reader and hold-history items are β's here. |
| S8 per-file order | same files | `harness.py` tail ω 9236; `config.py` tail θ 9229; `workflow.py` tail κ1 9232; `event_store.py` tail ι2 9231; `agents/briefing.py` tail μ2 9235; `mcp_lifecycle.py` tail μ1 9234 | — | α → 9236; ζ → 9236, 9229, 9232; δ2 → 9231 (implied by 9234); κ1 → 9232 |
| S1 escalation filing, C5 `SchedulerEscalations` (S1 leaf ζ2, after ζ1; `plans/review-all-s1-one-escalation-filing-entry-point-prd.md`, wave 3) | consumes | one `SchedulerCallbacks.escalations` port replacing `on_delivered_check_block` and the three warm-base hooks; `file_and_block` for blocking filings; R22 invariant 1 (every escalation through one entry point) is S1's | S1 | δ2 → S1 ζ2; θ → S1 ζ2 (S1 leaf; ζ2 → ζ1 covers S1's "θ → ζ1 and → ζ2"; wire the task id at decompose, whichever stream files second); λ reaches S1 ζ2 through δ2. C5's `escalations` parameter is S1 C5's type; δ2 now uses only `file_and_block` (born-at-L2) and no longer `file(…, FilingPolicy.open_l1())`; λ uses `file(…, FilingPolicy.born_at_l2())` at severity `critical` with a `Sentinel` member `__delivered_check_phase_slow__` that λ adds to S1's registry in `escalation/src/escalation/models.py`. Lead 2026-11-13: S17's two scheduler filers go through S1's entry point; δ2 → S1 ζ2 and λ → δ2 mean neither lands before the door, so S1 ζ1 has nothing to convert. Under OQ-9 = A the gate files no self-resolving level-1: its two filings are both born-at-L2 (`dependency_capability` per `(dependent, dep)`; the phase alarm per project) |
| S1 C6 / S1 leaf γ (the deterministic runner's tails: 8 non-reraising `set_task_status` arms) and task 8355 (six born-at-L2 filers in a lower module) | same files / consumes | both reshape the filers ξ1 moves out of `deterministic_runner.py` | S1 | ξ1 → S1 γ (S1's decompose cancels 8355 and 3887 into γ; γ's `status_write.py::file_and_block` is the lower-module filer; S1 leaf, wire at decompose, whichever stream files second) |
| S1 leaf κ2 (`mcp/markup_sink.py` files through the door; deletes its `EscalationQueue(...)` construction) | same file | μ's C9 bundles `open_escalation_channel` from that file | S1 | μ → S1 κ2 (S1 leaf; wire at decompose, whichever stream files second) |
| S16 config shape, leaves ζ2 (recovery sweeps as one submodel each, `recovery_sweeps.stranded`) and ζ3 (a `scheduler` submodel for `backfill_*`, `requeue_*`, claimant TTL; `delivered_checks.max_record_fetches_per_tick` split from `max_checks_per_tick`) (`plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md`, wave 3) | same files | both declare `scheduler.py` and `config.py`; ζ deletes `stranded_blocked_redispatch_enabled`, which S16 ζ2 moves; η, θ and δ2 read fields S16 ζ3 moves | S16 | S16 ζ2 → S17 ζ; S16 ζ3 → S17 κ1 (S16's draft wires both; lead S16/OQ-11: S16 waits for S17). S16 ζ2 lands after task 4082 (ζ → 4082), so its `recovery_sweeps.stranded` move carries 4082's `stranded_blocked_repend_cap` and `stranded_blocked_repend_window_secs`. S16 ζ3's reader scan, re-run at its decompose, covers S17's new `backfill_admission.py` and `delivered_check_gate.py`, and `delivered_checks.errored_grace_sweeps` and `delivered_checks.phase_alarm_secs`. S17 adds no edge onto S16 |
| S20 CLI invocation, leaf μ (delivered-check kinds as a discriminated union; `plans/review-all-s20-cli-invocation-and-usage-gate-prd.md`, wave 3) | same file | `delivered_checks.py`'s kind dispatch (S20 μ) and its one-evaluator move (δ1) | — | δ1 ↔ S20 μ: whichever stream files second depends on the other (S20's draft states the same rule) |
| S14 shared leaves (`plans/review-all-s14-shared-leaves-prd.md`) | consumes | `SchedulerStateSnapshot` (η 6717, done; I4); `shared/scheduler_overrides.py` (θ 6718); `KeyedStreaks` with the scheduler's error streak registered (ο 6724; I7) | S14 | β → 6718; δ2 → 6724. C1 adds one field to S14's model, in the one place S14's module docstring names for a new field. |
| S10 task-write contract | consumes / same files | C2 `MetadataWriteFailed` (β2 9191); C1 write modes (the in-memory store applies them); `SchedulerFacade`/`FakeScheduler` parity (β1 9187), which κ1/κ2 then retire; ο 9285 (`targeted.py`, `live_workflow_detector.py`) and λ 9254 (`live_workflow_detector.py`); μ2 9280 (`server/tools.py`) | S10 | β → 9234 (implies 9187, 9191, 9209, 9283); ξ1 → 9191 (`deterministic_runner.py`), 9254, 9285; ζ2 → 9285, 9280. α0's import-line edits in `targeted.py` and `live_workflow_detector.py` take no edge (rebase-only, S19/Q-C6) |
| S9 workflow spine | same files / consumes | `plan_tools.py` (α 9170, done); C8 spec'd scheduler double and `DryRunScheduler` port (μ1 9253, μ2 9279): S9 D15 states "`DryRunScheduler` is a narrower Protocol that `SchedulerFacade` satisfies", and μ1 adds `orchestrator/tests/test_port_conformance.py`; `Scheduler` satisfies `DryRunScheduler` once the facade goes | S9 | μ → 9170 (satisfied); κ1 → 9253, 9279 (both precede 9232); κ1 declares `test_port_conformance.py` and `dry_run_unblock.py` |
| S5 GitOps / git runner | same files | C1 `run_git` default in `delivered_checks.py` (α 9255, ε 9263) | S5 | δ1 → 9263 |
| S6 merge lane | same files | `landing_evidence.py` (μ 9201) | S6 | δ1 → 9201 |
| S3 escalation server | same files (prose) | `release_workflow`'s comments about the scheduler flip (`escalation/src/escalation/server.py` tail ζ 9200); `ARCHITECTURE.md`/`OPERATIONS.md` tail μ 9210 | S3 | ζ2 → 9200, 9210 |
| S10 κ1 (`shared/task_transitions.py`, 9248) | same file (prose) | the actor note on `blocked → pending` | S10 | ζ2 → 9248 |
| S7 (`test_plan_tools_server.py` in γ 9145) | same test file | — | S7 | none (test files are left to the scheduler's locks, R54) |
| S22 dashboard readers | follow-up | the dashboard may show C1's `tasks` section | S22 | xref only |
| Live `deterministic_runner.py` tasks that change the branches ξ1/ξ2 move (pending at 2026-11-13): 7827 (high, predicate gates answer "not enough data yet"), 3638 (medium, pure gates never self-complete after resume), 3887 (low, pure-gate trailing `set_task_status('blocked')`), 5312 (medium, detached-restart cwd fallback, deploy path). Task 6390 (review_due gate-fields builder) is done (`6d9a69fef3`, `3a607efadd`); ξ1 moves its builder with the gate branch | same file, moving code | file locks stop serializing them once the code has left `deterministic_runner.py`; ξ1's golden capture is invalidated by any of them landing between capture and merge | each task | ξ1 → 7827, 3638 (3887 is cancelled into S1 γ, which ξ1 depends on); ξ2 → 5312 |
| Task 8356 (runner per-branch split, low) | subsumed | — | ξ1/ξ2 | lead cancels 8356 with `x_acceptance_reason` naming ξ1 and ξ2 |
| Task 7627 (plan-tools create_server seams, low) | subsumed | — | μ | lead cancels 7627 naming μ |
| Task 4082 (stranded sweep park invariant, high, pending, no dependencies at 2026-11-13) | re-scoped by the sitting (S17-1, S17-4; R-2 = A): it carries C12 | the blocked rows that replace the scheduler step: three code leave rows (merge queued through 9219's `TruthReport.merge_in_flight`; `needs_recheck_against_main` from `targeted.py::_sweep_block_orphan`; blocked behind a lane halted for a heal, `workflow.py::TaskWorkflow._auto_heal_main_health`'s lane-already-halted branch), the clean re-pend row with its event, the owned-record hold (today's re-file duplicate guard as a row), the loop cap, the over-cap re-file and the once-per-window leave that keeps the cap from looping (D13), the `decided_by` column and its validator, and the watcher playbook rewrite. Its items 1 (a blessed manual park; `deferred` since task 6525) and 5 (pin the scheduler step's query) are dropped; item 4's documentation moves to ζ2 | 4082 | lead amends 4082 (lead action); 4082 → 9219, 9223 (the applier), 9229 (`config.py` tail), 9231 (`event_store.py` tail), 9232 (`workflow.py` tail); ζ → 4082; ν → 4082 |
| Task 6145 (merge-entry gate and stranded sweep disagree on info escalations, medium) | done 2026-11-13 (`2b044c36ca`) | it made `records_pin_blocked_recovery` refuse to relax a record set holding a record that needs a human (`escalation/pins.py::needs_human`) and made the flip log name the records it went past; ζ deletes the scheduler code it touched (`_describe_records_flipped_past` and the comment block) with the phase, and re-points its test's docstring | — | none (the planned cancel is moot) |
| Task 9529 (loop breaker for the scheduler's stranded-blocked step; a test in that step's test file that passes through the isolation guard; low, pending, filed 2026-11-12 from 6145) | subsumed | item 1 is the loop cap of task 4082's rows, on the one applier, because ζ deletes the step it would guard; item 2's test file `test_scheduler_stranded_blocked_redispatch.py` is deleted by ζ, and 4082's rows are tested with real `Escalation` records (row 14e) | ζ, 4082 | lead cancels 9529 naming ζ and 4082 |
| Task 6143 (Scheduler constructor seams, low) | subsumed | PSI reader and hold history (β); landed-outbox gate (S8 β 9218) | β | lead cancels 6143 naming β and 9218 |
| Task 6035 (delivered-check lint bypasses, medium) | overlap | its item 4 (`_resolve_main_sha` uses `GATE_REF`) is δ2's: `DeliveredCheckGate` resolves `GATE_REF` and passes it to `run` as `ref` | δ2 | lead amends 6035 (drop item 4; drop `scheduler.py` from its `metadata.files`); items 1–3, 5 stay |
| Task 4700 (premise recheck in the scheduler-side evaluator, medium) | same region | the evaluator it warns from moves in δ2 | — | δ2 → 4700 (pending at 2026-11-13) |
| Task 6821 (age-anchor extraction, low) | boundary | `_pending_anchor`/`_anchor_pending_since` are its, not `TaskRecord`'s | 6821 | η → 6821 (pending, low, no dependencies), so R16's "replacing the task-keyed dicts" does not wait on an unordered task |
| Task 3825 (passed-over counting, high) | same files | adds `_passed_over_count` and a snapshot field | 3825 | none (lands long before η); η folds the counter into `TaskRecord` if present and lists the moved literal in the decompose `moves` |
| Task 8778 (harness escalation-queue test seams, low, pending) | same test files | edits `_task_store_doubles.py`, `test_restart_recut_dispatch.py`, `test_restart_recut_e2e.py`, which κ2 re-points and deletes; its item 4 migrates a helper onto `TaskStoreDouble`, which κ2 deletes | 8778 | none: the edge rested only on test files (lead test-file rule, restating R54), so the scheduler's file locks serialize them; lead amends 8778's item 4 to migrate onto `orchestrator/tests/_in_memory_task_store.py::InMemoryTaskStore` once β has landed |
| Wave-3 invariant leaves (S1 ω `escalation-filing-one-door`, S1's eval-hermeticity task, S2 ι, S16 ι `config-default-one-home`, S19 ε `fused-memory-subpackage-dag`) and R51's wave-2 chain (9140 INV-15, 9242 INV-16, 9154 INV-17) | same files | ν edits the same three invariant files (`docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`) | the lead (S2/OQ-10) | the lead chains ν among the wave-3 invariant leaves at decompose; ruled 2026-11-13: the six wave-3 minting tasks take INV-18 to INV-23 in landing order and ν is minted last as INV-23 (`## Design invariant`); every check anchors on the slug `one-decision-partition-per-area`, never the number |
| Task 7082 (plan-tools heuristic-14 partition, low) | owns the remainder | the plan-tools git-probe patch sites μ leaves (Out of scope) | 7082 | none; lead note on 7082 naming them |
| Live `scheduler.py` tasks without a semantic link (78 live at `1f240e6da2`: e.g. 3824, 4213, 4523, 5467, 6593, 6700, 6942, 7109, 9029) | same file | — | — | none; serialized by the scheduler's file locks |

## Decomposition plan

Overlay bands: ~300–1,500 changed LOC, ≤ 10–12 declared files; the four mechanical test migrations (ι2,
ι3, κ2, ω) exceed the 15-file review trigger by construction (one rewrite per reach) and say so; ζ, γ and κ1
exceed 12 files because each also declares the test files its deletion would otherwise break.
Every leaf carries `metadata.x_finding_key` (the keys listed), `x_finding_run = "review-all-dark_factory-20261008"`,
`source = "review-all"`, file-level `metadata.files`, priority = the highest severity of its keys (λ and ν,
which carry no key, take the priority the sitting gave them: λ low, ν medium). High leaves (β, ι1, ι2, ι3, ζ,
γ, δ2, ω) become dependencies of trigger-chain gate 6698 at decompose. Edges onto a wave-3 sibling's leaf
(S1 ζ2/γ/κ2, S20 μ) are wired by task id at decompose by whichever stream files second; ν's place among the
wave-3 invariant leaves is the lead's (S2/OQ-10). S16 ζ2/ζ3 depend on S17 ζ/κ1 as S16's draft states; S17
adds no edge onto S16. Task 4082 is not a leaf of this plan; the lead amends it (`## Mechanical tasks`) and
ζ and ν depend on it.

Chain on `scheduler.py`: κ1 → θ → η → ε → λ → δ2 → δ1 → γ → ζ → α → β, so β lands first and κ1 last. The
migration leaves are off that chain (they edit no production file) and wait only on β: ι1, ι2, ι3 → β. D1's
order is kept by a real edge: γ, the first leaf that deletes a name tests reach outside its own files, depends
on ι1, ι2 and ι3, and every later structural leaf inherits that through the chain. κ2 → κ1 and ω → κ2 (ω is the
integration gate). Off the chain: α0; ξ2 → ξ1; μ; ζ2 → ζ; ν → 4082. Edges that rested only on a shared test
file (ι2 → ι1, ι3 → ι2 on the baseline json; κ2 → 8778) are not wired, per the lead's test-file rule; the
scheduler's file locks serialize those tasks and the later one rebases.

**α0 — One shared reader for a task's dependency ids, train id and deterministic kind; fused-memory's own copy is deleted**
(`task_kind=normal`, `complexity=simple`, priority low, ~250 LOC)
- Files: `shared/src/shared/task_fields.py` (new), `shared/tests/test_task_fields.py` (new),
  `fused-memory/src/fused_memory/utils/task_dependency_ids.py` (deleted),
  `fused-memory/tests/test_task_dependency_ids.py` (deleted; its cases move to `shared/tests/test_task_fields.py`),
  `fused-memory/src/fused_memory/middleware/recon_write_policy.py`, `fused-memory/src/fused_memory/middleware/task_curator.py`,
  `fused-memory/src/fused_memory/reconciliation/targeted.py`, `fused-memory/src/fused_memory/services/live_workflow_detector.py`
  (import line only, each) — 8 files.
- Deps: none. The four importer edits rewrite only an import line, so they are rebase-only under file locks
  (S19/Q-C6) and take no edge onto the wave-2 tasks (9254, 9285) or S19's moves that hold those files.
- Keys: `fk-a2bc16e5812c` (contributes). Contract: C11 (lead S17/OQ-4, as changed: no re-export).
- Work: C11's `dependency_ids` is the union of both tolerances and returns a fresh `list[str]`; each of the
  four importers changes its import line to `from shared.task_fields import dependency_ids as <its local
  name>` (`task_dependency_ids` in three, `_task_dependencies` in `task_curator.py`); no call site changes;
  `utils/task_dependency_ids.py` and its test are deleted.
- Signal: `shared/tests/test_task_fields.py` pins each wire shape of C11 (dict with id, int, str, dict
  without id, CSV string, falsy entries, a bare int and a dict as the whole value; train id
  empty/None/non-dict/whitespace) against today's two normalisations, including every case
  `fused-memory/tests/test_task_dependency_ids.py` pins today; `git grep -n 'utils.task_dependency_ids'`
  over `fused-memory` finds nothing.

**β — Scheduler test builder over public calls, an in-memory task store, a per-task snapshot section, and the private-reach guard**
(priority **high**, ~1,300 LOC)
- Files: `orchestrator/tests/_scheduler_scenario.py` (new), `orchestrator/tests/_in_memory_task_store.py`
  (new), `orchestrator/tests/test_scheduler_scenario.py` (new), `orchestrator/tests/test_scheduler_private_reach.py`
  (new), `orchestrator/tests/scheduler_private_reach_baseline.json` (new), `shared/src/shared/scheduler_state.py`,
  `shared/tests/test_scheduler_state.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/tests/test_scheduler_hold_history.py`, `orchestrator/tests/test_scheduler_pin_reservations.py`
- Deps: 9234 (S8 μ1, the tail of the wave-2 `scheduler.py` chain, which brings S8 β0 9217's
  `make_scheduler` and β 9218's C3); 6718 (S14 θ, the shared override store).
- Keys: `fk-a571802ee4ea` (contributes). Contracts: C1, C2; D2, D3, D4.
- Deletes: the post-construction `_read_psi_sample` and `_hold_history` seams' test reaches (36 lines in the
  two named files, 33 + 3, measured at `a85cb1ad78` under `PRIVATE_REACH_PATTERN`); 6143's remaining items.
- Records: the measured 68,002-byte `scheduler_state.json` baseline (2026-11-13), against which row 3b's
  bound is stated.
- Signal: boundary rows 1–4b; `scheduler_state.json` written by a running orchestrator carries `tasks`
  for each dispatched task and stays within 1 KB of its pre-β size on an idle queue, and fused-memory's
  `get_scheduler_state` returns it once both run this code.
- G7: INV-13 (the new wire field has its producer proven by row 1's real `get_state_snapshot`); INV-9.

**ι1 — Move test_scheduler.py's lifecycle, constructor and streak reaches onto the scenario builder**
(priority **high**, ~1,500 LOC, tests only)
- Files: `orchestrator/tests/test_scheduler.py`, `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: β.
- Keys: `fk-a571802ee4ea` (contributes).
- Work: the 318 lines of `test_scheduler.py` that reach the lifecycle containers, constructor-held
  collaborators and streak aliases (175 + 84 + 59 by group, overlapping; measured at `97be1d71fa` under
  `PRIVATE_REACH_PATTERN`; unchanged at `a85cb1ad78`) move to C1/C2 and public methods; its 122 dependency
  and 38 phase lines (measured at `a85cb1ad78`) are left for γ and ε. A test whose only purpose was to pin a container's shape is deleted when an equivalent
  behaviour test exists, with the deletion listed in the task record.
  Mapping for the most-reached names: `_dispatched` → `snapshot().tasks[t].dispatched` /
  `is_dispatched`; `_callbacks` → `callbacks`; `_project_root` → `config.project_root`; `_requeue_until`,
  `_requeue_counts` → `requeue_cooldowns`, `tasks[t].requeue_count`; `_skip_count` → `skip_counts`;
  `_module_cache`, `_get_modules` → `tasks[t].modules`, `seed_modules`; the streak aliases → the events
  their streaks gate (`external_dep_gate_held`, the external-dep escalation callback).
- Signal: the baseline row for `test_scheduler.py` falls by 318; the file's suite is green with no
  production diff.

**ι2 — Move the other lifecycle-heavy scheduler test files onto the scenario builder**
(priority **high**, ~1,300 LOC, tests only; 16 files, above the review trigger, one mechanical rewrite per reach)
- Files: `orchestrator/tests/test_retry_cap.py`, `orchestrator/tests/test_scheduler_dispatch_admission.py`,
  `orchestrator/tests/test_transient_requeue_backoff.py`, `orchestrator/tests/test_scheduler_state.py`,
  `orchestrator/tests/test_scheduler_landed_dispatch_gate.py`, `orchestrator/tests/test_harness.py`,
  `orchestrator/tests/test_harness_warm_lane_wiring.py`, `orchestrator/tests/test_reconcile_stranded.py`,
  `orchestrator/tests/test_interactive_warm_worktree_integration_gate.py`, `orchestrator/tests/test_scheduler_callbacks.py`,
  `orchestrator/tests/test_harness_scheduler_seam.py`, `orchestrator/tests/test_harness_starvation_watchdog.py`,
  `orchestrator/tests/test_harness_terminal_lane_record_reclaim.py`, `orchestrator/tests/test_harness_module_tagging.py`,
  `orchestrator/tests/test_scheduler_deterministic.py`, `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: β (no edge onto ι1: they share only the baseline json).
- Keys: `fk-a571802ee4ea` (contributes).
- Work: 314 lines (measured at `97be1d71fa` under `PRIVATE_REACH_PATTERN`; unchanged at `a85cb1ad78`), including
  `test_scheduler_state.py`'s 81 lines (the snapshot-writer reaches move onto `flush_state_snapshot` and
  the `state_snapshot_path=` keyword), `test_transient_requeue_backoff.py`'s dependency reaches onto
  dispatch behaviour, and `test_scheduler_deterministic.py`'s 10 `real_scheduler._get_modules` /
  `_deps_satisfied` / `_already_landed_gate` / `_consult_already_landed` lines (a file with no bare
  `scheduler._` reach, which only `PRIVATE_REACH_PATTERN` counts).
- Signal: each listed file's baseline row reaches 0; no production diff.

**ι3 — Move the remaining lifecycle, alias and dependency reaches outside the scheduler's own test files onto the scenario builder**
(priority **high**, ~500 LOC, tests only; 32 files, above the review trigger, one mechanical rewrite per reach)
- Files: `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/test_agent_session_reaper.py`, `orchestrator/tests/test_atomic_train_merge.py`,
  `orchestrator/tests/test_crash_recovery.py`, `orchestrator/tests/test_cross_project_dispatch_integration.py`,
  `orchestrator/tests/test_cross_repo_gate.py`, `orchestrator/tests/test_delivered_check_gate_e2e.py`,
  `orchestrator/tests/test_fleet_staleness_composition.py`, `orchestrator/tests/test_harness_fleet_drain.py`,
  `orchestrator/tests/test_harness_merge_heartbeat.py`, `orchestrator/tests/test_harness_merge_registry_wiring.py`,
  `orchestrator/tests/test_harness_offline_lane.py`, `orchestrator/tests/test_harness_offline_lane_trigger.py`,
  `orchestrator/tests/test_harness_pipeline_tripwire.py`, `orchestrator/tests/test_harness_reblock_guard.py`,
  `orchestrator/tests/test_harness_service_restart.py`, `orchestrator/tests/test_offline_lane_infra_integration.py`,
  `orchestrator/tests/test_offline_lane_integration.py`, `orchestrator/tests/test_orphan_l0_reaper.py`,
  `orchestrator/tests/test_orphan_reaper.py`, `orchestrator/tests/test_recut_ruling_dispatch.py`,
  `orchestrator/tests/test_repend_state_machine.py`, `orchestrator/tests/test_resume_claimant_liveness.py`,
  `orchestrator/tests/test_scheduler_hermetic_psi.py`, `orchestrator/tests/test_scheduler_tick_phases.py`,
  `orchestrator/tests/test_session_resume_integration_gate.py`, `orchestrator/tests/test_stranded_blocked_sweep.py`,
  `orchestrator/tests/test_substrate_gate.py`, `orchestrator/tests/test_transcript_archive_startup_sweeper.py`,
  `orchestrator/tests/test_warm_lane_structural_exhaustion.py`, `orchestrator/tests/test_harness_deterministic_dispatch.py`,
  `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: β (no edge onto ι2: they share only the baseline json).
- Keys: `fk-a571802ee4ea` (contributes).
- Enumeration rule (re-run at decompose): every `orchestrator/tests` file whose code (comment lines
  excluded) references a name γ, δ1, δ2, ε, η or θ deletes, through a scheduler receiver
  (`sched*`, `Scheduler`, `scheduler_*`, `real_scheduler`, `fresh_scheduler`) or as a quoted patch target,
  and that neither the deleting leaf nor ι2 declares; plus `test_harness_deterministic_dispatch.py` (2
  `scheduler._deps_satisfied` calls). Measured at `a85cb1ad78`: 30 files and 70 code lines (η names 64
  lines in 29 files, among them 7 alias reaches through `harness.scheduler.` in
  `test_cross_project_dispatch_integration.py`, 8 `scheduler_mock._dispatched` lines in `_orch_helpers.py` and
  the `h.scheduler._dispatched` line `test_agent_session_reaper.py` gained since `97be1d71fa`;
  δ2 names 6 lines in `test_delivered_check_gate_e2e.py` and `test_scheduler_tick_phases.py`), plus the 2.
  The count is an upper bound: some receivers are mocks, which ι3 rewrites or deletes. Excluded as a
  false positive: `test_plan_tools_server.py`'s `'_resolve_main_sha'` patch targets `plan_tools`, not the
  scheduler.
- Scope: in its files ι3 migrates every code reference the rule finds and every raw line that matches
  `PRIVATE_REACH_PATTERN` and names a name γ, δ1, δ2, ε, η or θ deletes, comments and docstrings included
  (62 such lines in 28 of its files at `97be1d71fa`, plus `test_agent_session_reaper.py`'s one at
  `a85cb1ad78`; e.g. `test_stranded_blocked_sweep.py`'s two
  `#` lines and two docstring lines naming `scheduler._dispatched`, and `test_orphan_l0_reaper.py`'s
  docstring ``Scheduler._merge_phase_at``), so ω's list need not cover them.
- Signal: after ι3, a grep of every name γ, δ1, δ2, ε, η and θ delete over `orchestrator/tests` finds code
  references, comments and docstrings only in files those leaves or ω declare; no production diff.

**α — Every scheduler-side reader of dependency ids, train ids and the deterministic kind uses the shared readers**
(priority low, ~350 LOC)
- Files: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/dependency_verdict.py`,
  `orchestrator/src/orchestrator/agents/briefing.py`, `orchestrator/src/orchestrator/merge_deferred_recovery.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_task_field_readers.py` (new)
- Deps: β (chain on `scheduler.py`); α0; ξ1 (the runner's copy is repointed there, so the guard is green
  over the whole tree); 9236 (S8 ω, tail of `harness.py`; implies 9235, tail of `agents/briefing.py`, and
  9283, tail of `dependency_verdict.py`).
- Keys: `fk-a2bc16e5812c` (closes). Contract: C11 consumed.
- Deletes: 7 dependency-id normalisations (scheduler 5, `dependency_verdict` 1, `briefing` 1), 5 train-id
  readers, 3 inline deterministic-kind comparisons (`scheduler.py` 1, `harness.py` 2) and the rule's
  own body inside `Scheduler.is_deterministic`, which becomes one call to `shared.task_fields.is_deterministic`.
  The method is kept, not deleted (corrected at the ruling fold): the harness's slot path calls
  `self.scheduler.is_deterministic` and 27 test files mock or call it on a scheduler (measured at
  `a85cb1ad78`), so deleting it would put 27 more files in this leaf; the hand fakes' copy goes with κ2.
- Signal: boundary row 22 in CI.
- G7: INV-5, INV-9.
- Priority note: low by its key, but ζ (high) follows it on the chain; the scheduler's priority
  inheritance raises it, no override (contract §4).

**ζ — The scheduler stops re-pending stranded blocked tasks; the recovery table's rows decide them**
(priority **high**, ~950 LOC)
- Files: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/config.py`,
  `orchestrator/src/orchestrator/recovery_pins.py`, `orchestrator/src/orchestrator/recovery_emission.py`,
  `orchestrator/src/orchestrator/stranded_reconcile.py` (prose), `orchestrator/src/orchestrator/harness.py` (prose),
  `orchestrator/src/orchestrator/workflow.py` (prose), `orchestrator/tests/test_scheduler_stranded_blocked_redispatch.py`
  (deleted), `orchestrator/tests/test_stranded_blocked_sweep.py`, `orchestrator/tests/test_recovery_pins.py`,
  `orchestrator/tests/test_clear_claim_then_set_status.py`, `orchestrator/tests/test_resume_claimant_liveness.py`,
  `orchestrator/tests/test_stranded_blocked_one_decider.py` (new), `orchestrator/tests/test_scheduler_tick_phases.py`,
  `orchestrator/tests/test_recovery_emission.py`, `orchestrator/tests/test_recovery_emission_wiring.py`,
  `orchestrator/tests/test_merge_gate_implies_sweep_pin.py` (docstring: its consumer becomes the
  `StrandedReconciler`'s blocked partition), `orchestrator/tests/scheduler_private_reach_baseline.json`,
  `tests/scripts/test_reify_closure_staleness_sweep_retired.py` — 19 files: one deletion, its tests, the
  tests that name the deleted phase or recovery site, and the comments that would otherwise state a removed
  behaviour.
- Prior constraint re-pointed, not dropped: task 5247's retirement test
  (`tests/scripts/test_reify_closure_staleness_sweep_retired.py::test_hard_constraint_scheduler_redispatch_phase_preserved`)
  pins this phase as a "SURVIVING owner of the stranded-blocked population". Its intent (a surviving owner
  must exist) holds after ζ; ζ re-points the assertion at the `StrandedReconciler`'s `blocked` partition
  (S8 ε2a) and says so in the commit message.
- Deps: α (chain); 9219 (S8 δ, C2); 9223 (S8 ε2a, `StrandedReconciler` over `blocked`); 9229 (S8 θ, tail
  of `config.py`); 9232 (tail of `workflow.py`); 9236 (tail of `harness.py`); 4082 (re-scoped by the sitting:
  the blocked rows that replace this phase, C12).
- Keys: `fk-cc01eb7b143f` (closes).
- Deletes: `Scheduler._phase_redispatch_stranded_blocked` (253 lines, measured at `a85cb1ad78`) and its
  `_TICK_PHASE_ORDER` entry (and `test_scheduler_tick_phases.py`'s assertions that `redispatch_stranded_blocked`
  is in the order and sits between `cooldown_gc` and `build_candidates`); the module-level helper
  `scheduler.py::_describe_records_flipped_past` that task 6145 added for the phase's log line, with its
  `TYPE_CHECKING` import of `Escalation` if nothing else uses it; `OrchestratorConfig.stranded_blocked_redispatch_enabled`
  (set in no project yaml, measured);
  `recovery_emission.py::RecoverySite.scheduler_blocked_redispatch` (the scheduler arm of
  `recovery_emission`) and its references in `test_recovery_emission.py` and
  `test_recovery_emission_wiring.py`; `SchedulerCallbacks.recovery_for` if 9219 added it; the scheduler's
  inline deterministic check (already repointed by α, deleted with the phase); the 55 lines of
  `test_scheduler_stranded_blocked_redispatch.py` (measured at `a85cb1ad78` under `PRIVATE_REACH_PATTERN`)
  and that file's baseline row.
- Signal: boundary rows 14, 14b–14e, 14g and 15, asserted across scheduler ticks and a `StrandedReconciler`
  pass in `test_stranded_blocked_one_decider.py`; operator-visible: the WARNING
  "crash-stranded-blocked-redispatch — flipping blocked -> pending" never appears again in an orchestrator
  journal, and a clean stranded blocked task returns to `pending` within one
  `stranded_reconcile_interval_secs` (900 s) with a `recovery_applied` event naming `blocked_clean_repend`,
  and no escalation.
- G7: INV-9 (one decider), INV-3, INV-6. Stated consequence (ruled): no extra escalations at today's rate
  (26 flips over 26 distinct tasks from 2026-10-19 to 11-12, none repeated within 24 h); recovery takes up to
  900 s instead of about 15 s; no LLM call for routine recovery.
- Task 6145 is done; its follow-up 9529 is cancelled naming ζ and 4082 (lead action). 4082 is re-scoped, not
  cancelled.

**ζ2 — Correct every document that says the scheduler re-pends stranded blocked tasks, and name `deferred` as the park**
(`complexity=simple`, priority low, ~200 LOC, prose and comments only)
- Files: `ARCHITECTURE.md`, `OPERATIONS.md`, `skills/unblock/SKILL.md`, `docs/task-escalation-state-spec.md`,
  `docs/task-authoring.md`, `escalation/src/escalation/server.py`, `fused-memory/src/fused_memory/reconciliation/targeted.py`,
  `fused-memory/src/fused_memory/server/tools.py` (the `set_task_status` and `update_task` descriptions),
  `shared/src/shared/task_claimant.py`, `shared/src/shared/task_transitions.py`, `shared/tests/test_task_transitions.py`
  (it pins the actor note's text), `escalation/tests/test_release_workflow.py`, `fused-memory/tests/test_targeted.py`,
  `orchestrator/tests/test_harness_already_landed_gate_wiring.py` (comments) — 14 files, prose and comments.
  If S3 has moved `release_workflow` out of `escalation/src/escalation/server.py`, or S15 ε2 has moved
  `set_task_status`/`update_task` into `server/tool_families/task_write.py`, by decompose, the decompose pass
  names the new module in place of the old.
- Deps: ζ; 9200 (S3 ζ, tail of `escalation/server.py`); 9210 (S3 μ, tail of `ARCHITECTURE.md` and
  `OPERATIONS.md`); 9248 (S10 κ1, `task_transitions.py`); 9285 (S10 ο, `targeted.py`); 9280 (S10 μ2, tail of
  `server/tools.py`); S15 ε2 and ι if filed first (same descriptions; wire at decompose, whichever stream
  files second).
- Keys: none (companion correction; ζ closes the key).
- Also (task 4082's former item 4, moved here by the ruling): `ARCHITECTURE.md`'s scheduler and recovery
  sections state the blocked partition once: the stranded sweep re-pends a proven-unparked blocked task by a
  code row, leaves the three parks alone, an info record never pins (so a row covered only by info records
  is re-pended by design), and `deferred` with a deferral record is the blessed park; `docs/task-authoring.md`
  §9 "Deferring a task" gains one sentence that a bare `blocked` write is re-pended within one
  `stranded_reconcile_interval_secs`; the `set_task_status` and `update_task` descriptions point at that
  recipe instead of restating it (INV-9; task 6525's writer-instruction text is checked first and cited if
  it already says so).
- Signal: `git grep -n 'redispatch_stranded_blocked' -- '*.py' '*.md' ':!plans' ':!review' ':!docs/prds'`
  returns nothing (the pattern also matches `ARCHITECTURE.md`'s tick-order line, which names the phase
  without its `_phase_` prefix: `cooldown_gc → redispatch_stranded_blocked → external_dep_policy →`).
  Dated PRDs and the fused-memory calibration corpora (`fused-memory/calibration/*.jsonl`,
  `fused-memory/tests/fixtures/*.jsonl`) keep the name: they are records, not claims about current code.

**γ — Dependency readiness is one predicate over a required `DepGates` value**
(priority **high**, ~1,050 LOC; 14 files, two of them re-checked only and one renamed only)
- Files: `orchestrator/src/orchestrator/dependency_verdict.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/config.py`, `orchestrator/tests/test_dependency_verdict.py`,
  `orchestrator/tests/test_dep_gates.py` (new), `orchestrator/tests/test_scheduler.py`,
  `orchestrator/tests/test_delivered_check_gate.py`, `orchestrator/tests/test_deterministic_task.py`,
  `orchestrator/tests/test_atomic_train_merge.py`, `orchestrator/tests/test_harness_reblock_guard.py`,
  `orchestrator/tests/test_harness_deterministic_dispatch.py`, `orchestrator/tests/test_scheduler_deterministic.py`,
  `orchestrator/tests/test_scheduler_tick_phases.py` (rename only: it names `external_cache`,
  `external_resolver_failed` or `delivered_check_cache` on 9 lines), `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: ζ (chain); ι1, ι2, ι3 (D1: the migrations precede the first leaf that deletes a name tests reach
  outside its own files; every later structural leaf inherits this through the chain).
- Keys: `fk-4b35b1adea0d` (closes). Contract: C3; D5 (S17-3: the milestone clock waits for delivered checks).
- Deletes: `Scheduler._deps_satisfied`; the six optional parameters of `deps_verdict`; the per-call cache
  forwarding in `_eligible_for_dispatch`, `_stamp_milestone_deps_satisfied` and the park-hold predicate;
  the dependency-region test reaches in the listed files (`_deps_satisfied`, `_eligible_for_dispatch`,
  `_stamp_milestone_deps_satisfied`, `_build_reverse_index`, `_milestone_time_gated`,
  `_apply_external_dep_policy`, `_terminal_dep_record_cache`: 182 lines across the suite, 175 in these
  files, measured at `a85cb1ad78` under `PRIVATE_REACH_PATTERN`; ι2 and ι3 have already moved the rest,
  including the 2 `scheduler._deps_satisfied` calls in `test_harness_deterministic_dispatch.py` and the 2
  `real_scheduler._deps_satisfied` calls in `test_scheduler_deterministic.py`, which γ re-checks).
- Also: rewrites `config.py::DeliveredChecksConfig.enabled`'s description, which says the kill switch makes
  "`_deps_satisfied` take its legacy arm-off path" (it now builds `DeliveredChecks.gate_off()`); TickContext's
  `external_cache`/`external_resolver_failed` become `external: ExternalDeps` and `delivered_check_cache`
  becomes `delivered: DeliveredChecks`, with `dep_gates` written once per tick.
- Signal: boundary rows 5–7; operator-visible: trigger gate 6698's delay starts only once its high leaves'
  delivered checks pass on main (S17-3).
- G7: INV-1 (the contract is a constructor, machine-checked), INV-9.

**δ1 — Every delivered-check evaluation goes through `delivered_checks.run`**
(priority medium, ~750 LOC)
- Files: `orchestrator/src/orchestrator/delivered_checks.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/merge_lane/landing_evidence.py`, `orchestrator/tests/test_delivered_checks_run.py`
  (new), `orchestrator/tests/test_landing_evidence.py`, `orchestrator/tests/_landing_evidence_helpers.py`,
  `orchestrator/tests/test_delivered_check_gate_e2e.py`, `orchestrator/tests/test_harness_already_landed_gate_wiring.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`, `orchestrator/tests/test_delivered_check_gate.py`,
  `orchestrator/tests/test_reconcile_stranded.py`, `orchestrator/tests/test_merge_queue_landed_reconciler.py` (docstring)
- Deps: γ (chain); 9263 (S5 ε, tail of `delivered_checks.py`); 9201 (S6 μ, tail of `landing_evidence.py`);
  S20 μ ↔ δ1 (same file; whichever stream files second depends on the other).
- Keys: `fk-8af92edd6f38` (closes). Contract: C4; D6 (b). Owns the G4 seam "`delivered_checks.run` as the one
  evaluator".
- Deletes: `verify_delivered_checks_on_main` (its one caller, `gate_mark_done_on_delivered_checks`,
  calls `run`) and the 17 lines of `test_delivered_check_gate.py` that name it, the string patch target
  `_VERIFY_TARGET = 'orchestrator.delivered_checks.verify_delivered_checks_on_main'` in
  `test_reconcile_stranded.py` (re-pointed at `run`), and the docstring mention in
  `test_merge_queue_landed_reconciler.py` (measured at `97be1d71fa`); the scheduler's own `asyncio.wait_for`
  and collapse; the "mirror scheduler.py 3116-3129" comment; `landing_evidence`'s direct
  `run_delivered_check` call.
- Signal: boundary rows 8, 8c, 9; `git grep -n 'run_delivered_check(' -- orchestrator/src` lists only
  `delivered_checks.py`.
- Note on wording: the decompose return lists `verify_delivered_checks_on_main` and
  `DeliveredChecksVerdict.main_sha` under `moves`, so the clerk pass re-points any delivered check that
  names them.

**δ2 — A delivered check that cannot be evaluated is reported and escalated, never held silently**
(priority **high**, ~1,450 LOC)
- Files: `orchestrator/src/orchestrator/delivered_check_gate.py` (new), `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/event_store.py`, `orchestrator/src/orchestrator/config.py`,
  `orchestrator/tests/test_delivered_check_gate.py`, `orchestrator/tests/_delivered_check_fixtures.py`,
  `orchestrator/tests/test_delivered_check_errored.py` (new), `orchestrator/tests/scheduler_private_reach_baseline.json`
  — 8 files (no watcher skill: the ruling removed the playbook section).
- Deps: δ1 (chain); 6724 (S14 ο, the error streak on `KeyedStreaks`); 9231 (S8 ι2, tail of
  `event_store.py`; implied by the chain); 4700 (pending at `a85cb1ad78`); S1 ζ2 (C5 `SchedulerEscalations`;
  S1 leaf, wire at decompose, whichever stream files second).
- Keys: `fk-6c103745c784` (closes); contributes to `fk-09bf639213ca`. Contract: C5; D6 (a); D14 (OQ-9 = A).
- Deletes: `_compute_delivered_check_cache` (406 lines), `_note_delivered_hold`,
  `_fire_delivered_check_escalation`, `_resolve_main_sha`, the `_delivered_check_cache`/`_delivered_check_sha`/
  `_delivered_check_error_streak` attributes and `_DELIVERED_CHECK_ERROR_LOG_THRESHOLD` from `scheduler.py`
  (they become `DeliveredCheckGate` state and the `errored_grace_sweeps` config field); the remaining lines
  of `test_delivered_check_gate.py` that match `PRIVATE_REACH_PATTERN` (203 measured at `97be1d71fa`, of
  which γ takes the 36 dependency lines first).
- Also: adds `DeliveredChecksConfig.errored_grace_sweeps` beside `grace_cycles`, registered in the
  green (hot-reload) tier like its siblings `grace_cycles`, `check_timeout_secs`, `enabled` and
  `max_checks_per_tick` — a `config.py::RELOADABLE_FIELDS` entry, or S16's `Reload.GREEN` marker if S16 β1
  has landed — and read by `DeliveredCheckGate` at evaluation time from the held submodel
  (`config.py::_set_leaf` preserves submodel identity); builds the transient arm in the gate's own code
  (ruled; D14): the window count per `(dependent, dep)`, the `cleared` event on the sweep the check runs
  again, and the born-at-L2 filing on the third consecutive transient window, with no watcher playbook
  section and no queue record for windows 1 and 2 (OQ-9 = A); rewrites `check_timeout_secs`'s description, which says a timeout means "no streak bump,
  dep left uncached, retried next tick" and names `run_delivered_check` (now: ERRORED with reason `timeout`,
  a streak bump, escalated per C5); `DeliveredCheckGate` resolves `shared.delivered_check_polarity.GATE_REF`
  and passes it to `run` as `ref` (task 6035 item 4).
- Signal: boundary rows 8b, 10, 10b, 10c, 10d, 11; operator-visible: a dependency whose check script is
  missing produces a `delivered_check_errored` row in `runs.db` within one tick and a `dependency_capability`
  L2 naming the script after `errored_grace_sweeps` sweeps; a check that keeps timing out produces one
  `window` event per grace window and the same L2 at the third consecutive window (at least 2 hours at the
  120 s timeout), and nothing reaches the LLM watcher.
- G7: INV-11, INV-7 (the hold is owned and bounded, and re-armed after its record closes), INV-4 (one event
  per window, not one per tick), INV-2; `one-decision-partition-per-area` (INV-23): every case is a code row
  (§Decision partition).

**λ — Escalate to a human when a tick's delivered-check phase takes more than 30 seconds**
(priority low, ~250 LOC; ruled by Leo 2026-11-13 under S17-2)
- Files: `orchestrator/src/orchestrator/delivered_check_gate.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/config.py`, `orchestrator/tests/test_delivered_check_phase_alarm.py` (new),
  `escalation/src/escalation/models.py` (one `Sentinel` member in S1's sentinel-id registry) — 5 files.
- Deps: δ2 (chain; the gate exists and files through S1 C5, so λ always lands after S1's door and its
  sentinel registry, which δ2 → S1 ζ2 implies); its place on the `scheduler.py` chain is between δ2 and ε,
  so ε carries the one call into the phase table's telemetry.
- Keys: none (a follow-up Leo added to `fk-6c103745c784`'s ruling; it does not close the key).
  `x_finding_run` and `source` as every leaf.
- Contract: C13; D15.
- Work: `Scheduler._emit_tick_telemetry` passes `phases['delivered_check_gate']` (the value the
  `scheduler_tick` event already carries) to `DeliveredCheckGate.observe_phase_time`; the gate keeps its last
  sweep's per-check run list (dep, check name, outcome, elapsed) and, above `phase_alarm_secs`, files the
  born-at-L2 `infra_issue` at severity `critical` through `FilingPolicy.born_at_l2()` under the project
  sentinel `__delivered_check_phase_slow__`, deduped by S1's door to one open record per project.
  `DeliveredChecksConfig.phase_alarm_secs` (default 30.0) is green-tier like its siblings. Severity ruled by
  the lead 2026-11-13: the lowest that reaches Leo's queue, which is `critical` at main, the floor of
  `BORN_AT_L2_SEVERITIES` (confirmed by the lead; D15). Offloading is not built.
- Signal: boundary row 11b; operator-visible: on a project whose phase exceeds 30 s, one human-level
  escalation names the project, the phase time and the slowest checks, with the wall-clock-budget remedy;
  no second record while it is open.
- Priority note: low as ruled; ε (medium) follows it on the chain, so the scheduler's priority inheritance
  raises it, no override (contract §4).
- G7: INV-4 (one open record, never one per tick), INV-9 (one phase measurement, two readers), INV-2 (the
  record carries the phase time and the checks as structured detail).

**ε — The tick's phases are a validated table of method references over a context with required fields**
(priority medium, ~1,200 LOC)
- Files: `orchestrator/src/orchestrator/tick_phases.py` (new), `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/tests/test_tick_phase_order.py` (new), `orchestrator/tests/test_scheduler_tick_phases.py`,
  `orchestrator/tests/test_scheduler.py`, `orchestrator/tests/test_scheduler_scoring.py`,
  `orchestrator/tests/test_delivered_check_gate.py`, `orchestrator/tests/test_clear_claim_then_set_status.py`,
  `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: λ (chain; ε keeps λ's `observe_phase_time` call in the phase table's telemetry).
- Keys: `fk-3e1e7c5129b5` (closes). Contract: C6; D7.
- Deletes: `TickContext`'s `{}` defaults for producer fields; `_TICK_PHASE_ORDER` and the
  `getattr(self, f'_phase_{label}')` dispatch, with its 6 test lines (5 in `test_scheduler_tick_phases.py`,
  1 in `test_delivered_check_gate.py`); the docstring read/write sets (now data derived by AST); every line
  of `test_scheduler_tick_phases.py` (99) and `test_scheduler_scoring.py` (11) and the 38 phase lines of
  `test_scheduler.py` that match `PRIVATE_REACH_PATTERN` (measured at `97be1d71fa`); the direct
  `TickContext(...)` constructions that set only some fields (29 in `test_scheduler_tick_phases.py`, 28 in
  `test_delivered_check_gate.py`, 9 in `test_scheduler.py`, 1 each in `test_scheduler_scoring.py` and
  `test_clear_claim_then_set_status.py`, measured at `97be1d71fa`), which move to a test factory that
  supplies the inputs and leaves producer fields `UNSET`.
- Signal: boundary rows 12, 12b, 13.
- G7: INV-1, INV-2 (`PhaseOrderError` names phase, field, producer).

**η — One frozen record per task in a dispatch ledger; the streak aliases go**
(priority medium, ~1,200 LOC)
- Files: `orchestrator/src/orchestrator/dispatch_ledger.py` (new), `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/tests/test_dispatch_ledger.py` (new), `orchestrator/tests/test_scheduler.py`,
  `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: ε (chain); 6821 (age-anchor extraction; η leaves `_pending_anchor`/`_anchor_pending_since` where
  6821 put them).
- Keys: `fk-b2cd27267851` (closes); contributes to `fk-09bf639213ca`. Contract: C7; D8.
- Deletes: 14 task-keyed containers (`_dispatched`, `_dispatched_priority`, `_last_dispatch_at`,
  `_requeue_until`, `_requeue_cooldown_meta`, `_requeue_counts`, `_transient_requeue_counts`,
  `_requeue_history`, `_pending_transient_cooldown`, `_skip_count`, `_module_cache`, `_workflow_cancel_at`,
  `_merge_phase_at`, `_fallback_warned`); the six alias attributes; `_phase_stale_sweep`'s per-container GC.
- Signal: boundary rows 16, 17; `get_state_snapshot().tasks` equals `DispatchLedger.lifecycles()`.
- G7: INV-5, INV-9.

**θ — Backfill admission and the warm-base hard-down watchdog become collaborators the scheduler constructs**
(priority medium, ~1,300 LOC)
- Files: `orchestrator/src/orchestrator/backfill_admission.py` (new), `orchestrator/src/orchestrator/warm_base_watchdog.py`
  (new), `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/tests/test_backfill_admission.py` (new),
  `orchestrator/tests/test_scheduler_park_backfill.py`, `orchestrator/tests/test_warm_lane_pool_hard_down.py`,
  `orchestrator/tests/scheduler_private_reach_baseline.json`
- Deps: η (chain); S1 ζ2 (the warm-base L2 promotion and the backfill overstay escalation file through C5
  `SchedulerEscalations`; S1 leaf, wire at decompose, whichever stream files second).
- Keys: `fk-09bf639213ca` (closes). Contract: C8; D9.
- Deletes: `_backfill_admission`, `_settle_backfill_grant`, `_record_backfill_grant`,
  `_maybe_escalate_backfill_overstay_rate`, `_BackfillGrant`, `_BackfillScanCache`, `_backfill_grants`,
  `_backfill_outcomes`, `_apply_warm_base_hard_down_watchdog` and its latch fields from `scheduler.py`;
  every line of the two named test files that matches `PRIVATE_REACH_PATTERN` (88 + 44, measured at
  `97be1d71fa`).
- Signal: the two modules' tests construct them directly; the heuristic-14 record (lines, prose share,
  remaining partitions, each with its heuristic-13 symptom or none) goes in the task record, with a
  one-line pointer in `scheduler.py`'s docstring.

**κ1 — The scheduler's shadow protocol goes: TaskWorkflow and its peers type the scheduler they use**
(priority medium, ~850 LOC; 15 files)
- Files: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/chronic_flake.py`, `orchestrator/src/orchestrator/flake_ledger.py`,
  `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/tests/_in_memory_task_store.py`,
  `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/test_scheduler_callbacks.py`,
  `orchestrator/tests/test_workflow_dry_run_hook.py`, `orchestrator/tests/test_workflow_e2e.py`,
  `orchestrator/tests/test_pyright_gate_for_workflow_e2e.py`, `orchestrator/tests/test_chronic_flake.py`,
  `orchestrator/tests/test_escalation_router.py`, `orchestrator/tests/test_scheduler_acquire_uncontested.py`
  (asserts its fake is a `SchedulerFacade`; task 3090), `orchestrator/tests/test_port_conformance.py`
- Deps: θ (chain); 9232 (S8 κ1, tail of `workflow.py`); 9253, 9279 (S9 μ1, μ2).
- Keys: `fk-4b3eb71db017` (contributes).
- Deletes: `scheduler.py::SchedulerFacade` (17 methods, measured at `a85cb1ad78`) and every conformance block
  naming it (8 test files, plus S9 μ1's `test_port_conformance.py` case that `SchedulerFacade` satisfies
  `DryRunScheduler`, re-pointed at `Scheduler`); `TaskWorkflow`'s `TYPE_CHECKING` import of it.
- Signal: `git grep -n 'SchedulerFacade' -- '*/src/*' '*/tests/*'` is empty; pyright green.

**κ2 — Workflow tests use the real scheduler over the in-memory task store; the hand-written FakeScheduler and TaskStoreDouble go**
(priority medium, ~1,700 LOC, tests only; 45 files, above the review trigger, one mechanical rewrite per construction)
- Files: `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/_task_store_doubles.py` (deleted),
  `orchestrator/tests/test_restart_recut_dispatch.py`, `orchestrator/tests/test_restart_recut_e2e.py`,
  and the 41 other files that reference `FakeScheduler` (`git grep -l FakeScheduler main -- orchestrator/tests`,
  42 files with `_workflow_helpers.py`, measured at `a85cb1ad78`):
  `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_info_l0_router_boundary.py`,
  `orchestrator/tests/test_scheduler_acquire_uncontested.py`, `orchestrator/tests/test_workflow_scope_requests.py`
  (the four added since `97be1d71fa`),
  `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/test_agent_capability_wiring.py`,
  `orchestrator/tests/test_coalesce_integration_gate.py`,
  `orchestrator/tests/test_config_reload_integration_gate.py`, `orchestrator/tests/test_deterministic_task.py`,
  `orchestrator/tests/test_done_phase_wedge.py`, `orchestrator/tests/test_eval_mcp_isolation.py`,
  `orchestrator/tests/test_harness_train_callbacks.py`, `orchestrator/tests/test_harness_warm_lane_wiring.py`,
  `orchestrator/tests/test_info_l0_router_workflow.py`,
  `orchestrator/tests/test_invoke_role_config_resolution.py`,
  `orchestrator/tests/test_mark_blocked_outcome_scoping.py`,
  `orchestrator/tests/test_merge_deferred_recovery.py`, `orchestrator/tests/test_merge_lane_parked_custody.py`,
  `orchestrator/tests/test_merge_queue_coalesce.py`, `orchestrator/tests/test_one_step_variant_boundary.py`,
  `orchestrator/tests/test_pyright_gate_for_workflow_e2e.py`,
  `orchestrator/tests/test_routing_integration_gate.py`, `orchestrator/tests/test_scheduler_callbacks.py`,
  `orchestrator/tests/test_session_resume_integration_gate.py`,
  `orchestrator/tests/test_shadow_invocation_sandbox.py`,
  `orchestrator/tests/test_workflow_agent_session_preserve.py`,
  `orchestrator/tests/test_workflow_dry_run_hook.py`, `orchestrator/tests/test_workflow_e2e.py`,
  `orchestrator/tests/test_workflow_escalated_steward_stall.py`,
  `orchestrator/tests/test_workflow_halt_owner.py`, `orchestrator/tests/test_workflow_helpers.py`,
  `orchestrator/tests/test_workflow_mark_blocked_l0_coverage.py`,
  `orchestrator/tests/test_workflow_merge_gating_strand.py`,
  `orchestrator/tests/test_workflow_merge_phase_block_status.py`,
  `orchestrator/tests/test_workflow_metadata_files_reconcile.py`,
  `orchestrator/tests/test_workflow_sandbox_events.py`,
  `orchestrator/tests/test_workflow_scope_wipe_regression.py`,
  `orchestrator/tests/test_workflow_state_machine_boundary.py`,
  `orchestrator/tests/test_workflow_status_on_resume.py`,
  `orchestrator/tests/test_workflow_train_halt_owner.py`, `orchestrator/tests/test_workflow_truthful_exits.py` (the decompose pass re-runs the `git grep -l` and replaces this list with its output if it differs)
- Deps: κ1. No edge onto task 8778, which edits `_task_store_doubles.py` and both restart-recut test files:
  the edge would rest only on test files (lead test-file rule); file locks serialize them and the lead amends
  8778's item 4 onto `InMemoryTaskStore`.
- Keys: `fk-4b3eb71db017` (closes).
- Work: per file, the real scheduler over the in-memory store where the test drives scheduler behaviour,
  S9 μ2's `spec_scheduler` where it only records calls (the tactical choice R16 leaves; listed in the task
  record). The 2 `TaskStoreDouble` users move onto `InMemoryTaskStore`.
- Deletes: `_workflow_helpers.py::FakeScheduler` (80 constructions in 32 files, measured at `a85cb1ad78`);
  `_task_store_doubles.py::TaskStoreDouble` and its module.
- Signal: boundary row 21.

**ω — No test reaches a private Scheduler name: the long tail moves and the baseline is deleted (integration gate)**
(priority **high**, ~600 LOC, tests only; 34 files, above the review trigger, mechanical)
- Files: `orchestrator/tests/test_scheduler.py`, `orchestrator/tests/test_module_charter.py`,
  `orchestrator/tests/test_scheduler_hold_history.py`, `orchestrator/tests/test_harness_already_landed_gate_wiring.py`,
  `orchestrator/tests/test_harness_landed_dispatch_gate_wiring.py`, `orchestrator/tests/test_harness_park_eviction_wiring.py`,
  `orchestrator/tests/test_harness_park_stop.py`, `orchestrator/tests/test_cross_project_dispatch_integration.py`,
  `orchestrator/tests/test_delivered_check_gate_e2e.py`, `orchestrator/tests/test_deterministic_task.py`,
  `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_scheduler_hermetic_psi.py`,
  `orchestrator/tests/test_scheduler_starvation.py`, `orchestrator/tests/_orch_helpers.py`,
  `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/conftest.py`,
  `orchestrator/tests/test_eval_stub_mcp_session.py`, `orchestrator/tests/test_harness_action_dispatch.py`,
  `orchestrator/tests/test_harness_infra_resume_truthful.py`, `orchestrator/tests/test_harness_override_wiring.py`,
  `orchestrator/tests/test_restart_durable_guards.py`, `orchestrator/tests/test_scheduler_claimant.py`,
  `orchestrator/tests/test_scheduler_psi_saturation_transition.py`, `orchestrator/tests/test_briefing.py`,
  `orchestrator/tests/test_harness_cancel_workflow.py`, `orchestrator/tests/test_harness_lane_record_divergence.py`,
  `orchestrator/tests/test_lock_table.py`, `orchestrator/tests/test_milestone_integration_gate.py`,
  `orchestrator/tests/test_scheduler_pin_reservations.py`, `orchestrator/tests/test_streaks.py`,
  `orchestrator/tests/test_substrate_gate.py`, `orchestrator/tests/test_workflow_metadata_files_reconcile.py`,
  `orchestrator/tests/test_scheduler_private_reach.py`, `orchestrator/tests/scheduler_private_reach_baseline.json` (deleted)
- Enumeration rule (re-run at decompose): the files matching `PRIVATE_REACH_PATTERN` at main that keep a
  matching line after every earlier leaf migrates the names it is scoped to (ι2, θ, ε and δ2 zero their
  whole files; ι1, β, γ, ι3 and η migrate named groups only). Measured at `a85cb1ad78` (unchanged from
  `97be1d71fa`): 32 files and 82 lines (e.g. `test_scheduler.py` 9 lines naming `_normalize_task_metadata`/`_deferred_watch_gated`,
  `test_scheduler_hold_history.py` 5). A file that an earlier leaf declares stays in this list when that
  leaf is not scoped to its remaining lines.
- Deps: κ2 (the integration gate: every structural leaf precedes it through κ2 → κ1 and the `scheduler.py`
  chain, and ι1–ι3 through γ).
- Keys: `fk-a571802ee4ea` (closes).
- The guard file `orchestrator/tests/test_scheduler_private_reach.py` stays listed and clean: it matches
  `PRIVATE_REACH_PATTERN` on zero lines (row 4c), and no file is exempt from the zero assertion.
- Signal (stream integration gate): boundary rows 1–22 (with 10b–10d, 11b, 14b–14g) green in one run of
  the orchestrator suite; the guard asserts zero private reaches; `git grep -nE "$PRIVATE_REACH_PATTERN" --
  orchestrator/tests` is empty (the same pattern as the guard and as ω's delivered check). Row 23 is ν's and
  is not part of this gate.

**ξ1 — The deterministic runner parses a task once into RunnerState and dispatches to gate and predicate steps**
(priority medium, ~1,400 LOC, behaviour-neutral)
- Files: `orchestrator/src/orchestrator/deterministic_state.py` (new), `orchestrator/src/orchestrator/deterministic_gate_step.py`
  (new), `orchestrator/src/orchestrator/deterministic_predicate_step.py` (new),
  `orchestrator/src/orchestrator/deterministic_runner.py`, `orchestrator/tests/test_deterministic_runner_state.py`
  (new), `orchestrator/tests/test_deterministic_runner.py`, `docs/deterministic-runner.md`,
  `fused-memory/src/fused_memory/services/live_workflow_detector.py` (its "section 3" pointer)
- Deps: α0; S1 γ (S1's decompose cancels 8355 and 3887 into γ; γ's `status_write.py::file_and_block` is the
  lower-module filer; C6 and the runner's tails; S1 leaf, wire at decompose, whichever stream files
  second); 9191 (S10 β2, tail of `deterministic_runner.py`); 9254 (S10 λ) and 9285 (S10 ο), both on
  `live_workflow_detector.py`; 7827, 3638 (live tasks that change the gate and predicate branches ξ1 moves).
- Keys: `fk-ebce5e23048a` (contributes). Contract: C10; D11.
- Deletes: the gate and predicate branches of `run`, the `gate_escalated_at = metadata.get(...)`-style
  raw reads they use, the runner's dependency-id normalisation (now `shared.task_fields.dependency_ids`).
- Golden capture: row 18's capture of `test_deterministic_runner.py` output is regenerated on ξ1's
  rebased base immediately before merge, so a runner task landing between capture and merge cannot
  invalidate it.
- Signal: boundary row 18. Subsumes 8356 together with ξ2.

**ξ2 — A deterministic deploy resumes after a crash from its recorded deploy phase**
(priority medium, ~1,100 LOC)
- Files: `orchestrator/src/orchestrator/deterministic_deploy_step.py` (new), `orchestrator/src/orchestrator/deterministic_runner.py`,
  `orchestrator/src/orchestrator/deterministic_state.py`, `orchestrator/src/orchestrator/deploy_state.py`,
  `orchestrator/tests/test_deterministic_deploy_resume.py` (new), `orchestrator/tests/test_deterministic_runner.py`,
  `docs/deterministic-runner.md`
- Deps: ξ1; 5312 (live task changing the detached-restart deploy path ξ2 moves).
- Keys: `fk-5a5d08e775de` (closes), `fk-ebce5e23048a` (closes). Contract: C10; D10.
- Deletes: the deploy branches of `run` (sub-cases (a), (b-self), (b-self-stale), (b), (c), the ε
  self-restart, the target_unit-less and cross-unit paths move into the deploy step); every
  `metadata.get('before_done_*')` read in the runner; the nested `_RunFnProcShim` class (module scope in the
  deploy step).
- Signal: boundary row 19; `DeterministicRunner.run` is under 100 lines.

**μ — Plan-tools and verdict-tools receive their markup channels as one bundle**
(priority low, ~600 LOC)
- Files: `orchestrator/src/orchestrator/mcp/markup_sink.py`, `orchestrator/src/orchestrator/mcp/plan_tools.py`,
  `orchestrator/src/orchestrator/mcp/verdict_tools.py`, `orchestrator/tests/test_plan_tools_markup_guard.py`,
  `orchestrator/tests/test_plan_tools_markup_repair.py`, `orchestrator/tests/test_plan_tools_server.py`,
  `orchestrator/tests/test_verdict_tools_markup_registration.py`, `orchestrator/tests/_plan_markup_fixtures.py`
- Deps: 9170 (S9 α, `plan_tools.py`; done, so satisfied); S1 κ2 (`markup_sink.py` files through the door; S1 leaf, wire at
  decompose, whichever stream files second).
- Keys: `fk-6e7e9bb942d6` (closes). Contract: C9; D12.
- Deletes: `plan_tools._markup_project_root`, `plan_tools._escalation_channel`, the 5 call-time lambdas
  in `plan_tools.create_server`, the private constant re-export aliases the tests read
  (`_MARKUP_RESIDUE_ANCHOR_TASK_ID` and five siblings, per task 7627), `verdict_tools.create_server`'s
  `resolve_project_root` keyword (replaced by `channels`); the 22 markup-name reaches (measured).
- Signal: boundary row 20. Subsumes 7627. The git-probe seams the same test files patch are out of scope
  (see Out of scope) and noted on task 7082.

**ν — Record Leo's principle as the design invariant one-decision-partition-per-area (four-site lockstep edit)**
(`task_kind=normal`, priority medium, ~250 LOC of docs; ruled by Leo 2026-11-13)
- Files: `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`,
  `skills/prd/references/gates.md` — 3 files. `CONTRIBUTING.md` gets nothing (pinned as an absence) and
  `orchestrator/src/orchestrator/agents/code_quality.md`'s partial mapping is not edited.
- Deps: task 4082 (its `RECOVERY_RULES.decided_by` column and `validate_recovery_rules` are the house pattern
  and the first conforming example; the ruling says ν lands with or after them); the lead's chain of the
  wave-3 invariant leaves on the three invariant files (S2/OQ-10): ν is minted last of the six, INV-23 (lead
  2026-11-13); the lead wires that chain at decompose.
- Keys: none (a ruling, not a finding). `x_finding_run` and `source` as every leaf.
- Work: `## Design invariant`: the rule text Leo's ruling names (recorded in `digests/discussion/S17.md`
  §"The principle, recorded"), and the checkable question, evidence, house pattern, family boundary and
  fixtures drafted in that section; the fixtures are violating examples of each shape, and the dated
  addendum walk flags both. `INV-23-CODE` carries a row with no `decided_by` at all, and the walk records
  the checkable question "Does every row name who decides it?" answering no on it, naming the row, and yes
  on the house pattern (Leo's note of 2026-11-13: check that the form really fails on a row with no
  decider).
- Deletes: nothing.
- Signal: boundary row 23: `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py`
  passes with the new slug in every pinned site; the addendum walk's rehearsal table shows the
  `INV-23-CODE` row with no decider flagged and the house pattern clear (the negative proved in the docs,
  and in code by task 4082's boundary row 14f, a `RecoveryRule` without `decided_by` raising `TypeError`); a `/prd` G7 walk or `/review` phase 2 cites
  `one-decision-partition-per-area` by name.
- Lead action on landing: amend the memory that records the principle
  (`feedback_deterministic_vs_llm_single_partition`, via the memory-amendment procedure) to cite the invariant
  by number.
- G7: INV-10 (the fixtures prove the negative), INV-5 (the principle is stated once, here, and cited by name
  everywhere else).

### Deletable accounting (synthesis `deletable`)

| Synthesis item | Re-measured at `1f240e6da2` | Deleted by |
|---|---|---|
| 7 test-only alias attributes (134 refs) | 6 aliases, 49 refs (`_starvation_first_seen` gone with task 3821) | ι1 and ι3 port the refs, η deletes the aliases |
| 7 dep-id normalisation copies | 8 sites (plus fused-memory's own reader, which α0 deletes) | α (7), ξ1 (1), α0 (fused-memory's module) |
| the mirror comment in `delivered_checks.py` | present | δ1 |
| 4 optional gate parameters | 6 optional parameters on `deps_verdict` | γ |
| 11 call-time lambdas and 42 patch sites | plan-tools 5 lambdas, 42 patch call sites; verdict-tools' lambdas fixed by 6098 | μ (markup ones); git-probe patches remain (Out of scope; noted on 7082) |
| 37 `metadata.get` reads in the runner | 36 | ξ1, ξ2 |

## Mechanical tasks and live-task dispositions (contract §8)

Step 1 (`find_tasks_by_metadata(key="x_finding_key")`) returned no task for any of the 13 keys
(re-run 2026-11-13). Step 2 (`search_tasks`, confirmed with `get_task`) found the live tasks below,
including two new to this fold (9529, from task 6145; nothing for the phase-time alarm or the invariant at a
similarity of 0.5 or above); statuses re-read 2026-11-13. No Phase-5 mechanical task carries a key of this
stream.

| Task | Status (2026-11-13) | Relation | Disposition |
|---|---|---|---|
| 8356 Decompose `DeterministicRunner.run()` into per-branch steps | pending, low | subsumed by ξ1 + ξ2 (they add the typed parse and the phase-authority resume) | cancel with `x_acceptance_reason` naming ξ1, ξ2 |
| 8355 One born-at-L2 filer for the runner in a lower module | pending, low | delivered by S1 γ (`status_write.py::file_and_block`) | none here: S1's decompose cancels 8355 into γ; ξ1 → S1 γ |
| 7627 Plan-tools create_server seams | pending, low | subsumed by μ | cancel naming μ |
| 4082 Stranded-blocked sweep park invariant | pending, high | re-scoped by the sitting to carry C12 (see Cross-PRD) | amend (ruled; text in the return's lead actions); 4082 → 9219, 9223, 9229, 9231, 9232; ζ → 4082; ν → 4082 |
| 6145 Merge-entry gate vs stranded sweep on info escalations | done (`2b044c36ca`) | hardened the step ζ deletes | none (the planned cancel is moot) |
| 9529 Loop breaker for the scheduler's stranded-blocked step | pending, low | its loop breaker is 4082's loop cap; its test file is deleted by ζ | cancel naming ζ and 4082 |
| 6143 Scheduler constructor seams | pending, low | PSI and hold-history items in β; landed-outbox item in S8 β 9218 | cancel naming β and 9218 |
| 6035 Delivered-check lint bypasses | pending, medium | item 4 delivered by δ2 (`DeliveredCheckGate` resolves `GATE_REF`) | amend: drop item 4 and `scheduler.py` from its files; item 2's authoring-time `unevaluable_descriptor` complements δ2's runtime escalation |
| 4700 Delivered-check premise recheck | pending, medium | same region as δ2 | δ2 → 4700 |
| 6821 Age-anchor extraction | pending, low | owns the two age-anchor maps | η → 6821 |
| 3825 Passed-over counting | pending, high | same files | none; η folds its counter |
| 7827, 3638, 3887, 5312 runner branch tasks | pending | change branches ξ1/ξ2 move | ξ1 → 7827, 3638; ξ2 → 5312; 3887 is cancelled into S1 γ by S1's decompose (ξ1 → S1 γ) |
| 6390 review_due gate-fields builder | done | landed in the gate branch ξ1 moves | none (the planned note is moot) |
| 8778 Harness escalation-queue test seams | pending, low | same test files as κ2; its item 4 targets the double κ2 deletes | no edge (test-file rule); amend item 4 onto `InMemoryTaskStore` |
| 7082 Plan-tools heuristic-14 partition | pending, low | owns the plan-tools git-probe seams μ leaves | note naming them |

## G7 walk (advisory at author time; re-walked at decompose)

- INV-1 `contracts-machine-checked`: C3 (constructor), C6 (import-time validator), C9 (required keyword).
- INV-2 `structured-facts-at-failure`: `PhaseOrderError`, `TickFieldUnset`, the ERRORED escalation's reasons.
- INV-3 `corroborate-before-acting`: ζ removes an un-corroborated write (the flip acted on a null claimant
  and a queue read alone); 4082's heal-halt row reads the lane's live halt state, not only the marker.
- INV-4 `storm-escape-required`: C5 emits one event per grace window, never one per tick; λ keeps one open
  record per project; 4082's loop cap bounds the re-pend loop.
- INV-5 `no-lockstep-duplication`: C7 (lockstep rules as ledger methods), C11, C4.
- INV-6 `status-matches-liveness`: ζ — a blocked row is only moved by the applier of the recovery table.
- INV-7 `holds-owned-and-bounded`: δ2 bounds the ERRORED hold at `errored_grace_sweeps` and names its owner.
- INV-8 `loop-thread-occupancy-bounded`: `delivered_checks.run` keeps its git calls on the async runner;
  no new synchronous read on the tick.
- INV-9 `one-fact-one-home`: one decider (ζ), one predicate (γ), one evaluator (δ1), one record (η), one
  dependency-id reader across orchestrator, shared and fused-memory (α0, α).
- INV-11 `no-silent-fail-soft`: δ2.
- INV-13 `readers-prove-their-producer`: C1's `tasks` is produced by a real `get_state_snapshot` in row 1.
- No R22 invariant is promoted by this stream; the private-reach guard is a test ratchet, not a design
  invariant. This stream mints Leo's `one-decision-partition-per-area` (ν), and every table it adds conforms
  (`## Decision partition`). S1's invariant `escalation-filing-one-door` (R22 1) holds for δ2, λ and θ
  because they file through S1 C5.

## Out of scope

- **The fused-memory task-store client partition of `scheduler.py`** (`set_task_status` 215 lines,
  `update_task`, `get_tasks`, `get_statuses`, `get_external_statuses`, `submit_task`, `read_task`, the
  rejection family `SetTaskStatusRejected` … `ExternalResolverError` and its helpers), measured at
  `1f240e6da2` as the largest remaining partition after θ. Heuristic-13 symptom: none found by reading
  (the methods use `_mcp_session`, `_memory_url`, `config` and the event store; no tick state). Not ruled
  by R16 (the synthesis remedy names backfill admission); θ's task record carries the measurement and the
  return's program xref asks the lead to route a follow-up or accept it.
- **Retiring the deterministic stamps** (D10): 12 readers in 4 packages.
- **`workflow.py`'s own train-membership reads** (6 sites, measured): the workflow spine is S9's region;
  they can import `shared.task_fields.train_id` when S9 or a later change touches them.
- **The plan-tools git-probe seams** patched in the same test files (`_sha_exists_on_branch` 9,
  `_sha_branch_reachability` 5, `_resolve_main_sha` 2, `_probe_writer`, `_probe_flavoured`,
  `plan_tools.subprocess.run` 5; about 21 sites): a git seam, not the markup wiring the finding names; task
  7082 (plan-tools heuristic-14 partition) is their owner, and a lead note on 7082 names them.
- The dashboard rendering C1's `tasks` (S22).
- Other modules' private-reach in tests (the guard is scoped to `Scheduler`).
- **Offloading delivered checks off the tick** (S17-2, Leo: "OK for now"): not built; λ escalates if the
  phase grows past 30 s, and its remedy names the cheaper first step (a wall-clock budget for the phase).
- **Deleting `Scheduler.is_deterministic`**: after α the method is one call to the shared reader; the
  harness's slot path and 27 test files reach it through a scheduler (measured at `a85cb1ad78`), so removing
  the name is left to a later change that moves those tests (keeping the delegate accepted by the lead
  2026-11-13).
- **The blocked rows themselves** are task 4082's (re-scoped by the sitting), not a leaf of this plan; this
  plan states their contract (C12) and depends on them.

## Open questions

None. Every question of the sitting is ruled, and S17-OQ-9, raised at the ruling fold, was ruled by the lead
on 2026-11-13 as option A (`## Rulings`): transient windows 1 and 2 are the gate's own events with no queue
record, and the third consecutive window files the human-level record.

Tactical (decide in the leaf): which `test_scheduler.py` tests are deleted rather than rewritten (ι1,
listed in the task record); the `StoreWrite` record's fields (β); per file, real scheduler or
`spec_scheduler` (κ2); the structured marker the main-health block writes for 4082's heal-halt row.

## META

If this PRD is decomposed and queued without further oversight: the test seam lands before any internal
moves, and ι3 moves every reach a later deletion would break outside that leaf's own files, so each
structural leaf changes one internal with its own tests already on the public seam and the suite green at
every merge; every seam consumed from S8, S10, S14, S5, S6 and S9 is an edge onto a filed task, and every
wave-3 seam (S1, S16, S20) is an edge wired at decompose; the guard and ω's delivered check use one pattern,
so the trigger gate cannot be held by a check that measures something else; the behaviour change with
operator impact (ζ on task 4082's rows: a clean strand is re-pended by the sweep within 900 s instead of by
the scheduler within 15 s, the three parks are left alone, a looping task reaches the watcher for diagnosis
once per window and the cap cannot itself loop) is measured, replayed against October's runaway and ruled; the ERRORED escalation and the phase-time alarm file through S1's port; every table
names who decides each case, and ν makes that a checked invariant after its first conforming example lands.
The answer is yes under the rulings, OQ-9 included.
