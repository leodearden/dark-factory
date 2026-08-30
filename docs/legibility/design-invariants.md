# Design invariants

A gate checklist, not an essay. INV-1..INV-5 encode
the agent-legibility survey's cross-cutting root causes
(`plans/agent-legibility-survey-2026-07-13.md` §3) as named,
checkable design-time questions; INV-6..INV-7 were added 2026-08-02
from the task/escalation strand investigation
(`docs/task-escalation-state-spec.md`); INV-8 was added 2026-08-06 from
the reconciliation loop-blocking incident (task 3778); INV-9 was added
2026-08-24 from the answered-but-unrecorded escalation investigation
(esc-6107-7, plus five further instances measured 2026-08-22→24);
INV-10 was added 2026-08-30 from the doc-guard reconciliation
(task 4666). They gate `/prd`
decompose (G7, `skills/prd/references/gates.md`) and `/review` phase 2's
cross-module audit — both consumers Read this doc at run time;
it is the single normative copy (no restatement, per INV-5). Stable slug
ids are load-bearing: G7 waivers, `/review`'s `invariant_findings`, and
the confusion census's optional `invariant_violated` field all reference
them. Numeric aliases are prose convenience only.

Adding or removing an invariant? Append/remove its trigger shape in
`skills/prd/references/gates.md` §G7 **in the same commit** — that list is
hand-maintained and has drifted twice (2026-08-02, 2026-08-06); task 3802
mechanizes the pairing check.

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
(`escalation/src/escalation/server.py::merge_request`); already-merged
guard duplicated until 5026; sibling-tool envelope divergence;
hand-transcribed prompt text drifted twice in one file.

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
bounds that collection's size — "already bounded upstream" is complete
on its own only for small bounds (a fixed enum, a single-digit config
cap); for anything page-sized or configurable, state the numeric bound
and the worst-case per-item cost, because the product, not the existence
of a bound, answers the question (this repo's own pagination contract
permits page_size 2000; 2000 × 30 ms of fully non-blocking per-item CPU
still holds the loop for ~60 s) — and which work
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

## INV-9 `one-fact-one-home`

**Rule**: A fact about the world — a design ruling, a premise, a state
claim — has exactly one authoritative home; every other surface that
mentions it carries a dated pointer (id + commit/date anchor) or is
rendered from the home, never an independent copy. Where two surfaces
must both assert it and cannot be collapsed, an explicit reconciliation
mechanism (a cascade, a sweep, render-from-source) names which side wins
and when it runs. INV-5 is this rule scoped to code/prompt text; INV-9
is the same rule for world-facts across records and stores.

**Checkable design question(s)**: Does this feature write the same
world-fact into more than one record/store (task metadata, escalation
record, PRD, manifest, memory, a session note)? Which surface is the
home, and do the others point at it rather than restate it? If a copy is
machine-read (a `delivered_check`, a gate predicate), what re-checks its
premise against the home, and on what cadence? When the fact changes at
the home, which mechanism updates or invalidates each copy — and if
none, what is the accepted staleness window and who measures it?

**Evidence**: esc-6107-7 — a ruled, implemented, measured design
decision propagated into four sibling task descriptions but never into
the escalation record, its cockpit DecisionRecord, or the PRD; the
record sat answered-but-unrecorded 183h and a fresh session re-derived
the settled answer. The 2026-08-22→24 watcher sweep measured five such
records (category-agnostic), 30.2 answered-yet-open days, including a
Leo-released, fully-verified fix (task 3875) not shipping for 6.8 days.
A manifest `delivered_check` (`expect: absent`) actively enforced a
premise that had become false. ~14 memory-corpus entries record the same
decay family.

**House pattern**: escalation record + git as the two homes for a ruling
(ratified 2026-08-24); correction blocks citing `esc-id + commit +
date`; ruling-time amendment (unblock SKILL.md Step 4) bumping
`updated_at` to re-arm the watcher's re-verify; the ruled-elsewhere
check + `scripts/member-chain-sweep.py` as the drift detector;
`reap-decisions` closure-sync.

## INV-10 `guards-exercise-behaviour`

**Rule**: A guard exercises the behaviour it protects; it does not match
text that describes it. Where the protected claim is an instruction or a
recipe, the guard RUNS it against a fixture and asserts the outcome.
Where nothing is runnable, it mirrors a marker-delimited span against the
live artifact that span must agree with — a config value, a model, a
family derived from its own source of truth — so that rewording moves
nothing. A regex or substring over prose is not a guard: it pins wording
rather than behaviour, goes red on edits that broke nothing, and carries
an untested matcher whose characteristic failure is a silent green. This
is INV-1 (`contracts-machine-checked`) applied to the checks themselves.

**Checkable design question(s)**: Does this feature add a test or check
whose assertion target is TEXT rather than behaviour? If what needs
protecting is an instruction, a documented command, or a payload example,
can it be EXECUTED against a fixture instead of matched? If it genuinely
cannot, which live artifact does it mirror, and what delimits the span?
If a matcher is unavoidable, what proves the matcher itself correct, and
what goes red when it silently stops matching? Would fully rewording the
surrounding sentence, with the mechanism unchanged, turn this check red —
and if so, why is that the right behaviour?

**Evidence**: eleven prose-pinning test modules or blocks were deleted on
review between 2026-04-24 and 2026-08-25 — `4ed37e9367`, `d53cd62b68`,
`3e8d369b24`, `10978d1ddc`, `9427896b8c`, `fabba102c7`, `d733c1bdc7`,
`9c73deb78d`, `cb6d74359e`, `c3f8fa0b35`, `ba7fcffdbc` — the largest a
769-line marker-anchored module whose revert message records that it
"exercised no runtime behaviour". The predicted matcher bug is not
hypothetical: task 4095's own guard scopes SHA derivations with a
character class excluding the hyphen, so `skills/orchestrate/SKILL.md`'s
`task/<task-id>` placeholder makes it report that runbook's CORRECT
instructions as violations. The same failure reaches non-test checks —
capability `qdrant-vector-access-for-ann` (task 3210) was gated by a
file-scoped `grep` that would have reported DELIVERED had the parameter
landed on the wrong function (`scripts/check_method_param_wiring.py`),
and `docs/task-authoring.md`'s `delivered_checks` guidance states the
same rule for that mechanism.

**House pattern**: a three-tier ladder, in preference order. (1) EXECUTE
the documented thing — `tests/scripts/test_package_source_lookup_convention.py`
(task 3959) runs the recipe CLAUDE.md hands agents and asserts it resolves
a real package. (2) MIRROR a marker-delimited span against the live
artifact — `tests/scripts/test_contributing_lint_command_drift.py` (task
3558) against `dark-factory-orchestrator.yaml`'s `lint_command`, and
`scripts/tests/test_design_invariants_consistency.py` (task 3802) against
this doc's own headings. (3) Nothing else: a substring/regex over prose is
a finding, not a guard, and "harden the regex" is not a remedy. The
reviewer's standing discriminator
(`orchestrator/src/orchestrator/agents/roles.py::REVIEWER_COMPREHENSIVE`)
is the same test stated from the other side — if fully rewording the
surrounding sentence while keeping the mechanism leaves the check green,
it is referential integrity rather than a wording pin.

## Census seam

Incident records MAY carry an optional `invariant_violated: <slug>` field.
The slug vocabulary is *this* doc — the ids above. The coding pipeline
that populates the field is owned by `plans/confusion-reduction-prd.md`,
which ships the field in its γ task and names this doc reciprocally in its
§10 (Cross-PRD relationship). A slug violated repeatedly across census
batches is an enforcement gap: file a guard task.

## Fixtures

Calibration fixtures — two seeded violations per invariant plus a rehearsal
verdict table exercising the as-landed G7 and `/review` phase-2 text — live
at `docs/legibility/design-invariants-fixtures.md` (landed 2026-07-14;
INV-6/INV-7 fixtures added 2026-08-02; INV-8 fixtures added 2026-08-06;
INV-9 fixtures added 2026-08-24;
INV-10 fixtures added 2026-08-30).
