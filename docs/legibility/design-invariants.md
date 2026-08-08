# Design invariants

A gate checklist, not an essay. INV-1..INV-5 encode
the agent-legibility survey's cross-cutting root causes
(`plans/agent-legibility-survey-2026-07-13.md` §3) as named,
checkable design-time questions; INV-6..INV-7 were added 2026-08-02
from the task/escalation strand investigation
(`docs/task-escalation-state-spec.md`); INV-8 was added 2026-08-06 from
the reconciliation loop-blocking incident (task 3778). They gate `/prd`
decompose (G7, `skills/prd/references/gates.md`) and `/review` phase 2's
cross-module audit — both consumers Read this doc at run time;
it is the single normative copy (no restatement, per INV-5). Stable slug
ids are load-bearing: G7 waivers, `/review`'s `invariant_findings`, and
the confusion census's optional `invariant_violated` field all reference
them. Numeric aliases INV-1..INV-8 are prose convenience only.

## INV-1 `contracts-machine-checked`

**Rule**: Any eligibility/routing/capability contract lives where it's
consumed, machine-checked — an enforced schema field or a submit-time lint,
never description prose or dispatcher-internal heuristics.

**Checkable design question(s)**: Does this feature introduce a contract
(eligibility, routing, capability envelope, tool filter/result-envelope
convention) that lives only in prose or a dispatcher's internals? Does a new
tool/agent surface declare its envelope where callers see it, or is it
discovered by failure?

**Survey evidence**: Simple-task fast path dead ~7,950 tasks behind an
unadvertised title regex; `prose-routing-intent` (12);
`watcher-capability-envelope` (18).

**House pattern**: ValidationError+hint guard at the submit boundary
(execution_class_guard 2225; routing lint 2563); server-stamped identity +
level gate (2041-2044).

## INV-2 `structured-facts-at-failure`

**Rule**: Emit structured facts at the failure point; never re-derive
stories by log-scraping facts the emitter already had in a variable.

**Checkable design question(s)**: Must any consumer of this feature's output
parse logs/prose to recover a fact the emitter knew (exit code, step
identity, SHA)? Do its escalations/reports separate raw observation (with
`measured_at`) from hypothesis?

**Survey evidence**: `block-report-misattribution` (16);
`guards-assert-unverified-diagnoses` (14).

**House pattern**: Table-driven FailureCategory ladder (2131); structured
`evidence` field + observation/hypothesis split (2558); FAIL-anchored
excerpting.

## INV-3 `corroborate-before-acting`

**Rule**: State read from a snapshot/cache/metadata is re-corroborated
against ground truth (git, DB, live process) before an agent or sweep acts
on it.

**Checkable design question(s)**: Does this feature act (dispatch, delete,
requeue, merge, rewind) on state that could have changed since read? Where
exactly is the re-check?

**Survey evidence**: `merge-state-not-git-corroborated` (13);
`unverified-task-premises` (15); phantom-done family.

**House pattern**: Merge Tier-3.5 git-authority corroboration (2037);
`already_merged` genuine-check (5026); `premise_lint()`.

## INV-4 `storm-escape-required`

**Rule**: Every fail-soft path (suppression, fallback, degradation,
retry-absorb) carries a rate/streak-threshold escalation — loud-over-silent
applied at design time.

**Checkable design question(s)**: If this feature's fallback fires 100× in
an hour, who hears about it, and via what counter?

**Survey evidence**: Judge fallback verdicts hid a total subsystem outage
(`one-shot-subagent-contract`, 17); curator degrade-to-create; 1755
storm-counter precedent.

**House pattern**: Consecutive-streak gate (`merge_liveness.py`, generalized
by 2558); storm counter (1755); LLM-adjudicated guard failing safe to
strict.

## INV-5 `no-lockstep-duplication`

**Rule**: No duplicated lock-step logic: two sites that must agree
byte-for-byte are one site plus a call (or render-from-source) —
extraction over documentation.

**Checkable design question(s)**: Does this feature copy logic, constants,
or prompt text that must stay in agreement with another site? What is the
shared-helper / render-from-code alternative?

**Survey evidence**: `canonical_queued_branch_name` un-normalized site
(`server.py:1000`); already-merged guard duplicated until 5026;
sibling-tool envelope divergence; hand-transcribed prompt text drifted
twice in one file.

**House pattern**: Extract helper (`canonical_queued_branch_name`); render
prompts/examples from live schemas (2559) with drift/pinning tests.

## INV-6 `status-matches-liveness`

**Rule**: A status implying active ownership (`in-progress`) is legal only
while a live claimant exists. Every exit from a claimed state writes its
successor status through the transition's designated choke point *before*
the claim is released, and any operator-facing aggregate of "active work"
reconciles against the enforcing ground truth (live slots / claimants),
alarming on divergence. Sweeps are crash backstops — never a path's
designed exit.

**Checkable design question(s)**: Does any exit/bail/park/requeue path in
this feature leave a task in a status whose implied owner is gone
(claimant released, no steward)? Which choke point writes the successor
status, and what test pins every exit? If the feature counts or surfaces
"active" tasks, can a stranded row be counted as active without an alarm?

**Evidence**: Path B merge-entry bail — 9 tasks stranded 13-52h, 7 pinned
by their own dead-steward L0 (`docs/task-escalation-state-spec.md`
§8-E1); `_OUTCOME_ALLOWED`'s IN_PROGRESS-bearing entries as documented
strand windows (E10); burndown `in_progress` exceeded
`max_concurrent_tasks` in 859/7871 snapshots (peak 33 vs 24) with no
parity alarm (E12).

**House pattern**: claimant columns + `is_stranded` oracle
(task-status-authority D4); `_mark_blocked` sole-writer choke point; W9
SM-2 run()-exit consistency check, made loud per spec §5; spec §5
outcome contract.

## INV-7 `holds-owned-and-bounded`

**Rule**: Every hold on a non-terminal task — a parked status, an open
escalation pinning recovery, a wait loop — names a machine-readable owner
(the actor that will exit it unprompted) and carries a bound: a deadline
or progress-refreshed idle window for automation waits, a streak/cap
escalation for repeating suppressions, or a supervised consumer plus age
surfacing for queue-backed handoffs. `park` is the only sanctioned
unbounded hold. A hold that vetoes recovery emits a structured fact
naming the pinning record (INV-2 applied to holds).

**Checkable design question(s)**: For each held state this feature
introduces or touches: who exits it, what bounds it, and where does an
operator see the hold with its reason? If the exit owner dies (process
exit, the 8h fleet redeploy), which mechanism notices — and does the
hold's record survive or expire coherently across restart?

**Evidence**: the any-level recovery veto held tasks 13-52h in silence —
211 `reverted` log lines, zero non-revert explanations
(`docs/task-escalation-state-spec.md` §8-E7/E12); the merge-halt
unbounded `_escalation_event.wait()` (E3); an L2 queue at 70 pending and
not draining turns "held pending human" into "held indefinitely" with no
age alarm.

**House pattern**: steward-wait idle window (task 3170) —
progress-refreshed deadline + escalation-of-escalation; orphan-L0 reaper
(`orphan_l0_timeout_secs`); watcher-outage L2 tripwire; `park` →
`blocked` + open L2 as the explicit unbounded marker (task 1792);
gate-backlog age reporting (tasks 3520-3522).

## INV-8 `loop-thread-occupancy-bounded`

**Rule**: In a long-lived async process, no coroutine occupies the
event-loop thread for an unbounded time. Any call that can block
(subprocess, network, filesystem, lock, sleep) is either non-blocking or
offloaded (`asyncio.to_thread`/executor); AND any per-item work that can
block or whose per-item cost is non-trivial, fanned out over a collection
whose size is not already bounded by an upstream contract (pagination, a
config cap, a fixed enum), has its loop-invariant part hoisted out of the
body and its item count explicitly bounded. Neither limb alone bounds
occupancy — offloading an unbounded fan-out still burns unbounded wall
clock, and capping a still-blocking fan-out still stalls the loop — so a
fan-out tripping both needs both fixes. A cheap, fully non-blocking loop
over an upstream-bounded collection trips neither.

**Checkable design question(s)**: For each coroutine this feature adds or
touches: what is the worst-case wall time it holds the loop thread, and
what makes that case worst rather than typical? If it iterates a
collection doing work that can block or is non-trivial per item, who
bounds that collection's size — "already bounded upstream, by pagination
or a config cap or a fixed enum" is a complete answer — and which work
inside the body is invariant across iterations? If it shells out, does
the process spawn itself (fork/exec) run on the loop thread?

**Evidence**: `_render_live_workflow_section`
(`fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`
— a sync `def` called from `async def assemble_payload`) fanned three
blocking `subprocess.run(['git', ...])` calls over the uncapped
`filtered.active_tasks`: 56.7 ms/task × 514 (dark_factory) and
82.7 ms/task × 525 (reify) => 29-43 s per render on the loop thread.
`/health` went 12-35 ms idle -> 31-43 s under recon load; 184 freezes
>=12 s over Aug 02-05. `asyncio.timeout(5)` could not fire because the
loop enforcing it *was* the blocked thread. 726 of the sampled
subprocesses were a byte-identical, render-invariant
`git worktree list --porcelain`. (task 3778)

**House pattern**: `asyncio.to_thread` at the boundary
(`fused-memory/src/fused_memory/middleware/task_interceptor.py::_apply_status_transition`;
`middleware/task_curator.py::curate_batch_prepared`); the async
subprocess runner `orchestrator/src/orchestrator/git_ops.py::_run`;
hoist the loop-invariant probe out of the body and bound the fan-out with
an explicit cap that logs what it dropped (no silent truncation); loop-lag
heartbeat firing above a threshold (INV-4 applied to scheduling).

## Census seam

Incident records MAY carry an optional `invariant_violated: <slug>` field.
The slug vocabulary is *this* doc — the eight ids above. The coding pipeline
that populates the field is owned by `plans/confusion-reduction-prd.md`,
which ships the field in its γ task and names this doc reciprocally in its
§10 (Cross-PRD relationship). A slug violated repeatedly across census
batches is an enforcement gap: file a guard task.

## Fixtures

Calibration fixtures — two seeded violations per invariant plus a rehearsal
verdict table exercising the as-landed G7 and `/review` phase-2 text — live
at `docs/legibility/design-invariants-fixtures.md` (landed 2026-07-14;
INV-6/INV-7 fixtures added 2026-08-02; INV-8 fixtures added 2026-08-06).
