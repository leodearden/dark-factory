# DeterministicRunner dispatch: state machine and rationale

This is the design rationale for
`orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner.run`,
the orchestrator-side runner for deterministic gate tasks
(`metadata.task_kind == 'deterministic'`). Sections 1-3 below match the
`# ── 1.`, `# ── 2.` and `# ── 3.` heading comments in `run()`. A bare symbol
name is a member of `deterministic_runner.py`, except `ScriptTimeout`
(`orchestrator/src/orchestrator/before_done_script.py`) and
`summarize_predicate_output` (`orchestrator/src/orchestrator/predicate_output.py`).

Phase β delivers the **pure-gate** pattern
(`before_done=None`, `always_escalates=True`).

Phase γ adds the **before_done blocking cross-unit deploy** path
(`before_done` is a dict, `always_escalates=False`):

## 1. Idempotency / quiescence (checked first)

If `metadata.gate_escalated_at` is already set:
- If a pending escalation still exists for the task → return BLOCKED (B3:
  no second escalation on quiescence).
- Else (escalation resolved) → drive the task to `done` and return DONE
  (I2/B4/B11: resume path) — except a pure gate that now waits on more
  than it did when it escalated, which re-files (§3).

If `metadata.before_done_ran_at` is already set (γ):
- If a pending infra_issue escalation exists → return BLOCKED (B7: reaper
  no-rerun, I1 once-only).
- Else (no pending escalation) → require POSITIVE proof of a terminal
  outcome before completing (phantom-done guard for the crash window
  between stamping `before_done_ran_at` and recording a terminal result):
  * `before_done_verified_at` set → verify passed (crash before the done
    write) → drive to `done`.
  * an escalation was filed and resolved → human acted (act-then-ask) →
    drive to `done`, no re-run.
  * neither → crash mid-deploy before any terminal decision.  Task 2618:
    if `deploy_state.phase` is `RAN` (the genuine crash-window strand)
    AND a PERSISTED `deploy_state.verify_baseline` is available, RE-RUN
    the read-only verify inspect against `target_unit` and re-classify
    via `_deterministic_deploy_health_verdict` (the same classifier the
    harness recon-sweep uses) — a positive `'healthy'` verdict means the
    deploy actually succeeded but crashed inside
    `_writeback_deploy_success` before `before_done_verified_at` could
    be stamped (task 2584's shape), so it drives straight to `done` via
    `_writeback_deploy_success` instead of re-escalating.  Any other
    verdict (including no persisted baseline at all) falls through to
    re-escalate (infra_issue, blocked); never phantom-done, never re-run
    (I1).  Auto-recovery is best-effort across an ORCHESTRATOR restart
    only — `verify_baseline`'s monotonic clock resets on a MACHINE
    reboot, so a post-reboot re-verify conservatively yields
    `'unconfirmed'` and still escalates to a human.  This is a
    deliberate, narrowly-scoped exception to
    `_deterministic_deploy_health_verdict`'s documented "callers only
    re-file/resolve an escalation, never flip status directly" contract
    — see the sub-case (c) inline comment in
    `orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner.run`
    for the accepted-risk rationale.

## 2. before_done execution (γ: `before_done` is not None)

**Stop-instruction guard** (task 2509, reconciliation finding 0aac21b4):
on the FIRST dispatch of a non-predicate `before_done` task — after the
predicate dispatch below and after the `before_done_ran_at` idempotency
block above, but BEFORE the `before_done_ran_at` stamp is written —
`orchestrator.stop_instruction.detect_stop_instruction` scans the task
description for an explicit stop instruction (e.g. "do not apply").  If
found: file a born-at-L2 `stop_instruction` escalation and block WITHOUT
running the deploy and WITHOUT stamping `before_done_ran_at`.  Unlike
the other escalations in this section, `gate_escalated_at` is
deliberately NOT stamped — this is a re-checkable HALT (mirrors task
2273's SIGTERM-kill on its human-rehearsal mandate as a self-halt rather
than an external kill), not a resolve-to-done gate: it guarantees the
NEXT dispatch re-evaluates this guard from scratch instead of taking
section 1's `if gate_escalated_at:` resume-to-done branch (which would
either raise `NotImplementedError` — no `before_done_ran_at` proof —
or wrongly drive to done without the action ever running).  That
re-evaluation only happens once the task is back in `pending`, though:
a `blocked` task holding this open pending escalation is NOT
re-dispatched automatically, so removing the stop instruction from the
description is necessary but not sufficient — full recovery needs BOTH
(1) editing the description (else the guard re-fires and re-escalates on
the very next dispatch) AND (2) resolving the `stop_instruction`
escalation itself (e.g. `resolve_issue` action `resume`/`restart`,
which is what actually flips the task's status back to `pending` — see
`escalation.server.resolve_issue` Table B), or cancelling the task.
Excludes predicates (read-only, apply nothing — see the predicate
sub-path below) and excludes every resume/idempotency branch above (those
already have `before_done_ran_at` or `gate_escalated_at` set, so they
never reach this first-dispatch check).

The `before_done_ran_at` stamp is written FIRST (crash-safe I1) and is
SHARED between the self-target (ε) and cross-unit sub-paths below.

Phase ε **self-restart** sub-path (`before_done.target_unit` == own unit):
- Detects self-target by comparing `before_done.target_unit` to the
  orchestrator's own systemd unit name, resolved from the `ORCH_UNIT`
  environment variable via `_default_resolve_own_unit()`.  An empty or
  unresolved `ORCH_UNIT` fails-open to the cross-unit path so existing
  CI runs (where `ORCH_UNIT` is unset) are unaffected.
- **Operator requirement**: set `ORCH_UNIT=<unit-name>` in the
  `[Service] Environment` of the orchestrator's own systemd unit.
- Instead of running the blocking cross-unit deploy (which would kill this
  runner mid-execution), schedules a detached `systemd-run` transient
  unit that fires *after* `run()` returns.
- The transient unit's payload is a `/bin/sh -c` wrapper that runs the
  restart and, *only if it exits non-zero*, fires δ's `escalation submit`
  CLI (file-backed, no MCP server needed in the detached unit) to file a
  born-at-L2 `infra_issue` escalation.  Because the whole unit is deferred
  via `--on-active`, nothing runs at scheduling time — the escalation is
  reached only through the shell failure branch at fire time, never on the
  success path.
- If scheduling succeeds (`rc == 0`): stamp `before_done_scheduled_at`
  (a dict carrying the transient unit name and fire delay) as a crash-safe
  marker.  If the runner crashes between this stamp and the done write or
  gate filing, the resume path (b-self) in `DeterministicRunner.run` handles recovery.
- If scheduling succeeds (`rc == 0`) and `always_escalates=False`:
  set task `done` with
  `done_provenance.kind='deterministic-deploy-scheduled'` carrying the
  transient unit name and fire delay; return DONE (done = *scheduled*, not
  *verified*).  A crash between the stamp and done write is recovered by
  (b-self) driving to done with scheduled provenance.
- A dependent of such a task is held (verdict kind `scheduled_deploy`, park
  reason `deps_unsatisfied:scheduled_deploy`) until its `deploy_state.phase`
  reaches `verified` or `done`
  (`orchestrator/src/orchestrator/deploy_state.py::awaits_scheduled_restart`).
  `orchestrator/src/orchestrator/scheduled_deploy_settle.py::ScheduledDeploySettler`
  advances every phase that holds a dependent, one hop per pass, at the next
  orchestrator startup and on the deterministic-recon cadence:
  - `scheduled` → `verified` once this process started after the transient
    fired and the transient unit has finished
    (`scripts/restart-all-orchestrators.sh` restarts the dispatching unit
    last); `scheduled` → `failed` when the unit reads `failed` or an own-role
    `infra_issue` escalation was filed since scheduling.  That is the
    transient's RP-4 fire-time filing, or the runner's own completion-failure
    filing, which shares its role and category; the settler cannot tell them
    apart, so it errs toward holding.
  - `failed` → `escalated`.  The settler files the born-at-L2 itself
    (`orchestrator/src/orchestrator/deterministic_runner.py::file_deterministic_escalation`)
    only when no such own-role escalation exists.
  - `escalated` → `done` once no own-role escalation for the task is pending
    and the restart has landed.  This also releases a deploy that went done
    after a completion-failure escalation and sits at phase `escalated` (the
    task 2912 shape).
  - A held phase no writer advances (e.g. `ran`, or an unreadable slice) is
    reported at WARNING on every settle pass and must be repaired by hand.
    A transient whose state cannot be read, and a restart still unobserved
    six hours after its fire time, are reported at WARNING on every pass too.

  The dispatch tick keeps serving a held record from its cache when the
  re-fetch fails or is deferred past the per-tick fetch budget
  (`orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_backfill_terminal_dep_records`).
  A dep never yet cached that the budget defers makes its dependents wait
  (verdict `unknown`) until a later tick fetches it: after a restart the
  cold cache takes ceil(terminal deps / `delivered_checks.max_checks_per_tick`)
  ticks to warm.  Only a never-cached dep whose fetch fails outright fails
  open, for that tick.  Accepted residual: an RP-4 submit that itself failed
  reads as `verified`.
- If scheduling succeeds (`rc == 0`) and `always_escalates=True`
  (act-then-ask): fall through directly to section 3 (the gate) WITHOUT
  running the blocking cross-unit deploy.  The detached restart is already
  scheduled; the gate is filed as a `milestone_gate` escalation; the task
  is set to `blocked`.  Neither `unit_inspector` nor `script_runner`
  are invoked on this path (self-kill and double-deploy prevention).  A
  crash between the stamp and the gate filing is recovered by (b-self)
  re-filing the gate — the human-approval gate is NEVER bypassed.
- If scheduling fails (`rc != 0`): file born-at-L2 infra_issue, block
  (parallel to γ's rc≠0/verify-fail handling); `before_done_ran_at`
  already stamped (not re-run, I1).

Phase γ **cross-unit** sub-path (`target_unit` ≠ own unit or `ORCH_UNIT` unset):
- Capture baseline unit state (`unit_inspector`).
- Run the deploy script to completion (`script_runner`, blocking).
- If `rc != 0`: file born-at-L2 `infra_issue` escalation, set blocked
  (B7a).
- If the script overran its OWN `before_done['timeout_secs']` under the
  default runner (`ScriptTimeout`): file born-at-L2 `infra_issue`, set
  blocked — reporting the three facts the runner MEASURED and no others
  (task 4252): the budget it overran, the script's own exit code when the
  teardown found it already exited (else that none was produced), and
  WHICH signal that teardown dispatched — which for the common
  already-exited shape is none at all, leaving the script's children
  running and the operator told to go find them.  Never a synthetic rc,
  and never an assumed kill.
- Re-inspect and verify freshness (B7b), delegated to
  `proc_supervision.RestartPlan.execute()`'s `FreshPidVerify` check
  (task 2238/δ): when the pre-deploy baseline had a persistent MainPID
  (`baseline_main_pid > 0`), a fresh non-sentinel `MainPID` (>0,
  different from the baseline) and a strictly-later
  `ActiveEnterTimestampMonotonic` are both required. When the baseline
  was EMPTY (`baseline_main_pid == 0` — a `.timer` unit or a
  `Type=oneshot` service, neither of which ever reports a live MainPID,
  even once genuinely active — task 2611/esc-2584-1), the `MainPID>0`
  requirement is dropped: freshness instead defers to the shared
  `systemd_inspect._empty_baseline_fresh` predicate, requiring the
  `ActiveEnterTimestampMonotonic` to have strictly advanced past the
  baseline AND `ActiveState` to have settled into `'active'` or
  `'inactive'` (an allowlist — a transient/mid-transition state, a
  wedged/missing `ActiveState`, and `'failed'` are all rejected).
- If `always_escalates=False`: hand off to `_writeback_deploy_success`
  (task 2066), which stamps `before_done_verified_at` (the positive proof
  the resume path requires) and then sets the task to `done` with
  `done_provenance.kind='deterministic-deploy'` carrying the fresh PID
  and timestamp (B6).  Both writes are patiently retried — bounded and
  paced — because the deploy script may itself have severed the
  orchestrator's own fused-memory/MCP connection (e.g. it restarted the
  service backing that connection); a plain retry recovers once the
  connection auto-reconnects.  If the writeback budget is exhausted
  first, files a durable local `infra_issue` escalation and returns
  BLOCKED instead of silently stranding the task with an empty queue.
- If `always_escalates=True`: fall through to gate (act-then-ask).

Phase γ **cross-unit, target_unit-less** sub-path (`target_unit` is
falsy — an explicit `None`, or the key omitted entirely so
`before_done.get('target_unit', '')` is `''`; the documented
"`target_unit=None` → cross-unit, no named unit" configuration): there
is no specific systemd unit to baseline-inspect or fresh-PID-verify
against — inspecting the empty unit name returns a degenerate
`{'MainPID': 0, 'ActiveEnterTimestampMonotonic': 0}` dict with no
`ActiveState`, which the baseline gate above misreads as a wedge (task
2632 / esc-2585-1's exact escalation — the 2-key baseline repr
byte-matches an inspect of `''`, not a real wedge). Both the baseline
gate and the FreshPidVerify/RestartPlan leg are skipped entirely; the
deploy is driven on the script's exit code alone:
- `rc == 0`: hand off to `_writeback_deploy_success` (empty
  `new_state`, `pid=0` — the same helper the named-target path uses)
  unless `always_escalates=True`, in which case fall through to the
  gate (act-then-ask) instead, since the script already ran.
- `rc != 0`, the default runner's own per-script timeout
  (`ScriptTimeout` — reported as which signal the teardown actually
  dispatched, carrying an exit code only when the script really produced
  one; never a synthetic rc and never an assumed kill; task 4252), an
  outer wall-clock guard timeout,
  or an unexpected `run_fn` error: file born-at-L2 `infra_issue`,
  return BLOCKED (parallel to B7a); `before_done_ran_at` is already
  stamped (I1), so the deploy is NOT re-run.

Named-target genuine-wedge detection (the baseline/verify logic above)
is entirely unchanged — this sub-path is reached only when
`target_unit` itself is falsy.

Phase γ **predicate** sub-path (`before_done['kind'] == 'predicate'`):
- A read-only exit-code VERDICT check — NOT a deploy.  Dispatched at the
  very top of section 2, above the deploy-only `target_unit`/I1/baseline
  machinery, so a predicate NEVER touches systemd (no `unit_inspector`,
  no fresh-PID verify) and NEVER stamps `before_done_ran_at` (re-running
  a read-only check on crash-resume is harmless — there is no I1 side
  effect to double-apply).  This dispatch also happens BEFORE
  `always_escalates` is ever consulted, so a predicate task's
  `always_escalates` value is simply ignored — it is meaningful only on
  the deploy/gate paths' act-then-ask semantics.  A `kind='predicate'`
  task with `always_escalates=True` set still goes straight to
  `done`/`milestone_check_failed` from the check's exit code alone,
  with no gate ever filed.  Rejecting (or normalizing) that combination,
  if ever desired, belongs in fused-memory's `submit_task`/
  `deterministic_task_guard` validation, not in the runner (out of this
  runner's scope).
- Runs the check script under the same outer wall-clock guard pattern as
  the deploy path (`script_runner`/`_default_run_script` +
  `asyncio.wait_for(timeout_secs + run_timeout_grace_secs)`).
- `rc == 0` (invariant holds): set task `done` with
  `done_provenance.kind='deterministic-milestone'` (never
  `'deterministic-deploy'` — no deploy happened).  The `note` carries
  a BOUNDED STRUCTURED VERDICT from
  `orchestrator/src/orchestrator/predicate_output.py::summarize_predicate_output`,
  not the raw stdout tail (task 3286): the note is appended to a Mem0
  completion summary downstream, so raw subprocess output landing there is
  ingested into memory (task 2902's specimen).  The raw output is logged
  at INFO first, so nothing is silently discarded.
- `rc != 0` (invariant violated — a VERDICT, not an infra fault): file a
  born-at-L2 `milestone_check_failed` escalation, stamp
  `gate_escalated_at` (reusing the gate resume machinery so a human
  resolving the escalation on the next dispatch routes through section 1's
  quiescence/resolve-to-done fork), and block.  A non-zero rc reaching this
  branch is always a verdict — the default runner's own timeout no longer
  arrives as one (next bullet).
- `rc == 75` on a gate declaring `before_done.recheck_secs` with a dated
  milestone is "not yet": the gate re-arms instead of failing
  (`DeterministicRunner._run_predicate` carries the table, PRD
  `plans/completion-driven-triggers-prd.md` C1).
- `ScriptTimeout` — the DEFAULT runner's INNER per-script timeout (task
  4065) — is an infra fault, NOT a verdict: `infra_issue` + blocked with
  NO `gate_escalated_at` stamp.  See the
  `orchestrator/src/orchestrator/before_done_script.py::ScriptTimeout` class
  docstring for the full rationale.
- Outer-guard timeout, an unresolvable factory script root
  (`FactoryRootUnresolved`, task 6386) or an unexpected `run_fn` error
  (likewise no verdict — an infra fault, not a check failure): file
  born-at-L2 `infra_issue` and block — again NO `gate_escalated_at`
  stamp, so the check is simply re-attempted on the next dispatch.  The
  outer `asyncio.wait_for` guard is the backstop for a seam that never
  returns AT ALL (a detached/unkillable child, or a hanging custom runner).
  Every no-verdict arm files the same category as the others, so their
  escalation wording is deliberately distinct.
- RECURRENCE CARRIER (task 4678 / r3, PRD
  `docs/prds/recurring-deterministic-tasks.md` R-D6 + contract C-5): when
  the task carries `metadata.recurrence` — one link of a recurring chain
  — those no-verdict arms file `milestone_check_failed` instead of
  `infra_issue`, so every failure leg of a recurring job sits in one
  deny-listed, discriminable category
  (`escalation.authority.L2_AUTO_CLOSE_DENY_CATEGORIES`) rather than
  vanishing into the fleet's largest bucket.  Three things are deliberately
  NOT changed by that swap: the `gate_escalated_at` stamp is still never
  written on a no-verdict leg (task 4065 — the check is re-attempted, not
  latched into resolve-to-done); the SUMMARIES stay byte-identical, so
  wording remains the discriminator telling a human which guard fired; and a
  NON-carrier predicate keeps `infra_issue` throughout.  Every DEPLOY path
  is untouched too — the `category` override is passed only from
  `_run_predicate`, which no deploy branch reaches, so the
  `infra_issue`-keyed deploy-stranded population that
  `Harness._revalidate_open_deterministic_escalation` auto-closes is
  unaffected.  That population is structurally out of reach here anyway: it
  additionally requires `before_done_ran_at` (or `deploy_state.phase ==
  RAN`), and a read-only predicate never writes either.
- Section-1 resume: when `gate_escalated_at` is set and the
  `milestone_check_failed` escalation is resolved, RE-RUNS the predicate
  check (delegating back to `_run_predicate` — read-only, so repeating it
  is harmless) rather than trusting the resolution blindly: resolving an
  escalation is not proof the invariant now holds (a human may resolve it
  prematurely or in error).  `rc == 0` drives to `done` with
  `deterministic-milestone` provenance; `rc != 0` re-files
  `milestone_check_failed` (the dedup guard sees no pending escalation,
  since it was just resolved) and stays blocked; a timeout/error routes to
  `infra_issue`, identical to the first-dispatch path.  This
  short-circuits BEFORE the `before_done_ran_at` proof-check (a
  predicate never stamps it).

### How the arms handle `ScriptTimeout`

`orchestrator/src/orchestrator/before_done_script.py::ScriptTimeout` states
the caller contract and why a timeout is an infra fault rather than a
verdict.  The runner meets it like this (tasks 4065 / 4252):

- Only the default runner raises it.  An injected `script_runner` keeps the
  plain `(rc, tail)` contract, so its non-zero rc is still an honest
  verdict on the predicate path.
- Three arms catch it, each BEFORE its `except Exception` catch-all: the
  target_unit-less deploy in `_run_deploy_script_guarded`, the cross-unit
  deploy in `run()` section 2, and `_run_predicate`.
- Both `run_fn` seam wrappers, `_invoke_run_fn_translating_timeout` and
  `_run_predicate`'s local `_invoke_run_fn`, translate a seam-internal
  `TimeoutError` into `RuntimeError`, so the outer guard's
  `except TimeoutError` means only that the outer guard fired.
  `ScriptTimeout` passes through them untouched because it is not a
  `TimeoutError`.
- The two deploy arms print the same fact lines
  (`_deploy_timeout_fact_lines`), so they cannot describe one event two
  ways.  The predicate arm shares only `ScriptTimeout.budget_line`.

## 3. Pure gate (`before_done=None`, `always_escalates=True`)

- File one born-at-L2 escalation (I3: in-process submit, sentinel role
  `orchestrator-deterministic` keeps level=2 past the server downgrade gate).
  Dedup: if a pending escalation already exists for the task (e.g. from a prior
  crash-safe re-dispatch), filing is skipped to avoid duplicate L2 escalations.
- Stamp `gate_escalated_at` and `gate_escalated_preconditions` (the local
  dependency ids, `external_deps` and `milestone` the gate escalated under)
  in one write (crash-safe: file-before-stamp means a stamp failure re-files
  rather than silently skipping the gate).
- Set task status to `blocked`.
- Return BLOCKED (B2).

**Resuming a pure gate** — the proof ladder has TWO rungs, not one:

(i) An archive-inclusive, role-scoped escalation record must exist (task
2954). This proves a gate was established and is no longer pending. An
empty PENDING queue alone is not proof: a LOST escalation would otherwise
let an operator's re-pend of a stuck task silently bypass the gate.

Between the rungs runs a re-hold check (task 9520). A resolution that held
the gate behind something new is "not yet", not approval: a watcher that
finds a gate fired prematurely wires its real precondition and resumes it.
On 2026-11-12 gates 9424 and 8939 were held behind a new dependency that
way and closed as `pure gate resolved` the moment it landed, with no human
step. So on resume
`orchestrator/src/orchestrator/gate_preconditions.py::gate_held_since_escalation`
compares what the task waits on now with the `gate_escalated_preconditions`
record its filing stamped, and the runner re-files a fresh `milestone_gate`
and stays BLOCKED when a local dependency or an `external_deps` entry was
added, or a present `milestone` differs. The re-file restamps the record,
so the next resolution with nothing new closes normally. Removing a
dependency or a milestone never re-fires, because it never extends the
wait. A gate stamped before task 9520 carries no record and closes as
before; a record that is present but unreadable fails CLOSED and re-files.

(ii) If `metadata.human_curator_gate` is truthy, then
`metadata.human_curator_adjudicated_at` must carry a non-empty string (task
3341). This proves the human-judgment CONTENT work actually happened.

Rung (i) passing is NOT evidence rung (ii) holds. Task 3181 was a
`human_curator_gate` pure gate; `esc-3181-1` was resolved by
`escalation-watcher` with `action='resume'` on 2026-07-30T19:41Z, and
that resolution's own text says the ~13-entry Mem0 corpus edit was
"curator action, deliberately NOT executed here". The resume path
nonetheless drove the task to `done` with the generic
`note='pure gate resolved'`, contradicting the task's own
`consolidation_note`. Closing an escalation record and performing the
curator's content review are different propositions.

Rung (ii) is a PURE-gate rung only: `human_curator_gate` and a
`before_done` action are contradictory (one says only a human closes
this, the other is a machine step that does), so a task carrying both
takes the act-then-ask path with the marker unread. That combination is a
task-authoring defect and is logged as a WARNING on every dispatch rather
than being silently ignored. It is ALSO rejected at write time by
`shared.task_metadata`'s cross-field validator (task 3369), so it can no
longer land through the `submit_task`/`update_task` boundary; this
WARNING is retained as the defence-in-depth backstop for records that did
not pass through it.

Both rung-(ii) checks fail CLOSED: a truthy-but-not-`True` marker still
trips the guard, and a non-`str` or blank stamp is not proof. An
unproven curator gate files a born-at-L2 `curator_adjudication_missing`
escalation — a DISTINCT category, so the re-ask is legible in the census
and does not present to a resolver as "the same gate again" — and stays
BLOCKED. The block → resolve → re-dispatch → block loop is bounded by the
harness `reblock_guard` (`Harness._check_reblock_guard`), which keys
on `category:summary`, so the distinct category gets its own counter
(INV-4's storm-escape, with no second counter built here — INV-5).

Ordering matters and is pinned by test: rung (i) runs FIRST. Zero records
anywhere means the gate was never established, so re-filing the ORIGINAL
`milestone_gate` is the correct recovery; asking for an adjudication
stamp on a gate nobody has yet seen would be incoherent. The re-hold check
runs after rung (i) and before rung (ii), so a re-held curator gate is
re-asked as the gate, not as an adjudication request.

A curator gate that DOES close carries a specific `done_provenance.note`
naming the adjudication stamp (plus the gate record via
`escalation_id`), never the generic string — that genericness is what
made 3181's phantom closure indistinguishable from a genuine one. The
interpolated stamp is length-capped because this note is memory-ingested
downstream (`_format_outcome_echo`, task 3286) — the same constraint
section 2's predicate subsection documents, so both note-writing sites in
`deterministic_runner.py` read consistently.

BOTH gate closes cite the resolving gate record through
`done_provenance.escalation_id` (task 5241, PRD C5): `milestone_gate`, or
`review_due` for a `trigger_chain` human gate
(plans/completion-driven-triggers-prd.md C4). The curator close has cited
it since task 3341 and the PLAIN pure-gate close since 5241, both via the
one shared rule in `_cite_gate_escalation_id`. The two notes still differ — that
discriminator is deliberate and load-bearing — but neither close is
uncitable any more, and fused-memory now REFUSES a
`kind='deterministic-gate'` blob that names no record. Ordering note for
a fleet redeploy: a unit still carrying pre-5241 code has its pure-gate
closes refused until it restarts. The failure is self-healing — the gate
task simply stays blocked and re-resolves on the next dispatch — and no
state is lost, which is why the PRD ratifies landing the two halves
together.

## Human-curator gate contract (task 3341)

The marker key plus the stamp key that section 3's rung (ii) reads.

`human_curator_gate` marks a pure deterministic gate whose resolution requires
human CONTENT adjudication, not merely a closed escalation record.  It is
written today by reconciliation Stage 2 (LLM-authored; no code produces it).
`human_curator_adjudicated_at` is the ISO-8601 stamp that PROVES the per-entry
review actually happened — stamped on the task via `update_task` by whoever
performed the review.  Both are blessed Tier-A keys
(shared.task_metadata._BLESSED_METADATA_KEYS) because
`deterministic_runner.py` reads them.

The two are deliberately separate: the marker says "a human must judge the
CONTENT", the stamp says "a human did".  Task 3181 is the incident where the
runner had the first and inferred the second from a closed escalation record.

SINGLE SOURCE (task 3369, extended to the stamp by task 3420): both the
MARKER's and the STAMP's spellings are imported from `shared.task_metadata`
(see the import block in `deterministic_runner.py`) rather than restated as
local literals.  That module's `_deterministic_invariants` validator reads
the marker — to reject it alongside a non-null `before_done` at the write
boundary — so a second
definition would put two spellings of one key in the two modules that both
act on it, which is exactly the drift a named constant exists to prevent.
No validator reads the stamp today, but it gets the same single-definition
treatment for the same reason: `deterministic_runner.py`'s own
dispatch-time guard (`_curator_adjudication_confirmed`) is the one place
that decides whether a curator gate may close, so a drifted stamp spelling there would silently
defeat it.

A fork is caught BEHAVIOURALLY, by the tests that stamp the bare wire literal
and drive real code: `TestHumanCuratorGateAdjudicationGuard` in
orchestrator/tests/test_deterministic_runner.py, and shared's `parse_metadata`
blessed-key tests in shared/tests/test_task_metadata.py.  Do NOT add an
in-process `is`/`id()` identity check against the shared constant — CPython
interns identifier-shaped string literals, so a forked local literal would be
the SAME object and such a check passes either way (a vacuous one was removed
by task 3420 for exactly this reason).

NAMING (reviewer amendment): the stamp is `human_curator_adjudicated_at`, NOT
`curator_adjudicated_at`.  The bare `curator_*` metadata namespace is already
owned by a different subsystem — `curator_action` / `curator_justification` /
`combined_at` are written by the AUTOMATED task curator's combine flow
(fused_memory.task_interceptor).  Squatting that prefix for the HUMAN content
curator would invite a reader of the Tier-A list, or a census consumer
grouping by prefix, to conflate two unrelated actors.  The `human_curator_`
prefix pairs the stamp unambiguously with its marker.
