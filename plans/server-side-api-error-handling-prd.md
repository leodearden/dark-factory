# PRD: Server-side API error handling (529/5xx) — classify, requeue, back off, breaker

Status: **active** · 2026-07-30 · milestone: none (infrastructure hardening, three phases)
Origin: design session design-df-3663597 (report: `~/.claude/fleet/sessions/design-df-3663597/report.md`),
forensic investigation of the 2026-07-29 reify park-stop incident (scheduler paused 14.5h after a
~90-minute Anthropic 529/500 burst blocked 15 tasks whose work had all passed verify).

## Goal

A server-side API degradation (HTTP 529/500 burst) must not block tasks, file per-task
escalations, or halt the scheduler. Operators observe: affected tasks cycle through
`pending` with growing jittered cooldowns and self-heal when the provider recovers; at most **one**
fleet-level `api_degraded` escalation narrates the outage and auto-resolves; a genuine local wedge
(reify-4429 shape: no result JSON, no HTTP status) still fast-fails to `blocked` exactly as today.

## Background

The orchestrator already has a transient-5xx requeue lane — `is_transient_api_requeue`
(scheduler.py:407-428), `transient_requeue_cap: 10` (config.py:2730-2742) — but the 2026-07-29
incident bypassed it end-to-end:

1. The pre-turn-1 watchdog (`startup_grace_secs=120`, config.py:240-251; kill at
   cli_invoke.py:2116-2129) kills the CLI mid-way through the CLI's **own** internal 529 retry
   cycle (~10 attempts with increasing backoff — multi-minute; Leo-confirmed 2026-07-30),
   manufacturing the timeout and resetting the CLI's backoff ladder every ~2 min (retry storm:
   67 starts in one half-hour bucket).
2. The SIGTERM-flush kill path harvests the CLI's result JSON, so `api_error_status=529` **is
   stamped** on the killed `AgentResult` (cli_invoke.py:2196-2220 → :1754/:1833) — but
   `classify_agent_failure` ranks `timed_out` (:660) above `api_error_status` (:698), so the
   529 evidence is discarded and the marker `agent API error: HTTP N` is never produced.
3. Only the planning phase emits that marker (workflow.py:3902-3905, :3985), and converting it to
   a scheduler-visible REQUEUED requires a **steward** (an LLM call into the same degraded API) to
   succeed. Execute never classifies failures (workflow.py:6248-6252); review folds every failure
   into `verdict=='ERROR'` (workflow.py:7924-7934) and blocks marker-free (:5836-5841);
   simple_task's REQUEUED is an internal sentinel (workflow.py:2455-2464).
4. The zero-output breaker (`max_consecutive_zero_output_timeouts=2`, config.py:2585-2592;
   workflow.py:6259-6282) and park-stop (`_record_blocked_transition(task_id)` — category-blind,
   scheduler.py:1966, :2279-2281; 15 distinct blocked tasks/1h, defaults.yaml:692-705) key on
   shape, not cause.
5. No backoff (flat 30s `requeue_cooldown_secs`, config.py:2693; flat 2s reviewer stagger,
   workflow.py:7839-7840), no jitter, no host/account-level 5xx state (`InvokeSlot.report` drops
   `Failure` outcomes, usage_gate.py:443-444), stewards/watchers dispatched into the outage
   ungated, and the only operator signal is one silent L1.

## Sketch of approach — three phases

- **Phase 1 (incident-shape vertical slice):** preserve the 5xx cause across the watchdog kill
  (classifier), route the implementer zero-output-5xx shape to REQUEUED (no blocked write, no
  steward, no park-stop count), add jittered exponential backoff to the transient requeue lane,
  and land the observability fixes (evidence JSON, steward prompt, stale comments).
- **Phase 2 (coverage):** the same requeue-not-block treatment for review, planning, and
  simple_task paths; transient-cap-exhaust escalation names the provider outage.
- **Phase 3 (breaker + watchdog):** a process-wide `ApiHealthGate` (rate-windowed 5xx breaker
  with probe-driven close), dispatch throttling + steward/watcher suppression while open, a
  single auto-resolving `api_degraded` fleet escalation, evidence-gated park-stop auto-resume,
  and a two-regime startup grace so the CLI's internal 529 recovery is not preempted.

## Mechanisms and consumers (G1)

| # | Mechanism | Consumer |
|---|---|---|
| M1 | `is_server_error_status()` helper + `ServerError` `InvocationOutcome` variant + `classify_agent_failure` 5xx rule above `timed_out` | existing transient lane (scheduler.py:407-428); `invoke_with_cap_retry` cap-net guard; escalation summaries read by operators/stewards |
| M2 | `TerminalReport.api_error_status` structured field → `record_requeue` field-first routing | scheduler transient-bucket routing (replaces regex-over-prose as primary; INV-1) |
| M3 | REQUEUED-from-execute/review/planning/simple_task for 5xx-attributed failures (via existing `_requeue` path) | scheduler requeue lane (existing, harness.py:7487→8276→scheduler.py:7226); park-stop protection is a consequence (no `blocked` write) |
| M4 | Jittered exponential transient cooldown | existing `_requeue_until` dispatch-eligibility reader (scheduler.py:4478-4480) |
| M5 | Evidence/forensics: `api_error_status` + output tail in `zero_output_evidence-*.json`; steward-prompt API_ERROR guidance fix; stale-comment fixes; OPERATIONS.md row update | operator forensics; steward behaviour; maintainers |
| M6 | `ApiHealthGate` (Phase 3) + `api_error` account-event row | scheduler dispatch throttle; workflow steward-spawn gate; harness watcher-rotation gate; `api_degraded` escalation lifecycle; park-stop auto-resolve; dashboard provider-health strip |
| M7 | Two-regime startup grace (Phase 3) | watchdog kill decision in `_run_subprocess` |

No orphan mechanisms; every consumer exists today or is introduced with its producer in the same
phase and wired by a named task below.

## Resolved design decisions

1. **Classifier scope**: only 5xx jumps above `timed_out`; kind becomes `API_ERROR` with the
   marker plus timeout/progress context in the summary. 4xx placement, `AuthFailed` (401/403)
   precedence, and the deliberate 429-is-not-transient carve-out (scheduler.py:416-421,
   invocation_outcome.py:348-352) are all unchanged.
2. **`is_zero_output_timeout` stays shape-based.** Its resume-wedge guard consumer
   (cli_invoke.py:1233-1245) is correct for 529s too (never re-resume an orphaned session).
   Cause-awareness lives in the classifier and the workflow, not in the shape predicate.
3. **Exit-on-first, pace-in-scheduler**: the execute loop returns REQUEUED on the *first*
   5xx-attributed zero-output result rather than retrying in-loop — in-loop retries have no
   pacing; the scheduler lane has cooldown, counters, and the cap. `consecutive_zero_output`
   is not incremented for this class (a genuine wedge signature carries no HTTP status).
4. **No cross-account failover on 5xx.** Server-side errors are not account-scoped (incident
   data: the freshest account had the *highest* failure rate). `invoke_with_cap_retry` returns
   the failed result to the caller; the heuristic zero-cost cap net (cli_invoke.py:1303-1308)
   gains a `ServerError` guard so a fast 529 can no longer mark a healthy account CAPPED.
5. **Structured field over regex** (INV-1): `is_transient_api_requeue` becomes field-first with
   the regex retained as legacy fallback in that one site only (INV-5).
6. **Blocked is reachable for 5xx only via transient-cap exhaust** (existing enforcement,
   harness.py:8285-8305). Park-stop is **not** made category-aware: its blindness is deliberate
   (defaults.yaml:699-700) and today's categories are misassigned anyway (reviewer "infrastructure
   errors" file as `task_failure`, workflow.py:5836-5841 + :12077-12088). Protection = fewer
   wrongful `blocked` writes.
7. **Backoff shape**: `min(base·2^(n-1), cap)` with equal jitter (`d/2 + U(0, d/2)`), base 30s,
   cap 900s, `n` = the task's transient count at arming. Genuine requeues keep flat 30s.
8. **Review-phase conservatism**: only when **all** errored reviewers are 5xx-attributed does the
   phase REQUEUE; any genuine verdict-quality error (successful run, invalid/missing verdict —
   workflow.py:7911-7921) keeps today's blocking path. 5xx-attributed reviewers skip the
   in-phase retry burn.
9. **simple_task 5xx → real scheduler REQUEUED**, not the in-dispatch architect fall-through —
   escalating to planner mode is a wrong response to a provider outage and immediately re-invokes
   into it.
10. **Breaker trips on rate + breadth, closes on probe** (Phase 3): failures ≥ `min_failures` AND
    distinct tasks ≥ `min_distinct_tasks` AND failure-rate ≥ threshold within `window_secs`;
    close on consecutive probe successes (hysteresis 2). While open: dispatch throttled to a
    probe trickle (not halted — partial degradation must not strand the healthy fraction),
    steward spawns for 5xx-attributed L0s suppressed, watcher rotation deferred unless
    non-API-attributed L1s are pending.
11. **Park-stop auto-resume is evidence-gated and config-gated (default ON)**: only when the
    scheduler pause is park-stop-originated AND ≥50% of the window's blocked transitions were
    5xx-attributed AND the gate has been closed ≥ `park_stop_resume_settle_secs`. Resolution goes
    through the existing sentinel-L1 resolve → auto-resume path (harness.py:11437-11448) with a
    resolution note naming the evidence — loud trail, no silent state flip (INV-2/INV-3).
    AFK-operator preference (correctness + self-healing) drives the ON default; the off-switch is
    the escape for operators who want manual resume.
12. **Storm escapes** (INV-4): transient requeues → cap → `blocked` + `retry_cap_exhausted` L1
    (existing, with n_transient/n_genuine breakdown, scheduler.py:7291-7306); breaker open >
    `max_open_before_l2_hours` → promote `api_degraded` to L2; steward suppression ends at
    gate-close or that same L2.
13. **Rollout reality**: reify's config is load-once (reify yaml:353-355); defaults land in
    dark-factory `defaults.yaml`, take effect per-project on the next orchestrator restart
    (existing fleet-redeploy machinery). No per-project config edits in this PRD.

## Pre-conditions for activating

- None external. Phase 3's watchdog task ν is gated on substrate-validation task λ (below).
- The REQUEUED-with-reason → transient-count chain is verified substrate on main
  (steward-resolved path: workflow.py:12379 → harness.py:7487/8276 → scheduler.py:7226). G3 ✓.
- The SIGTERM-flush path stamping `api_error_status` on timed-out results is verified substrate
  (cli_invoke.py:2196-2220, :1754, :1833; incident `turns=2` signature). G3 ✓.
- **Unverified substrate (G3, scoped to ν)**: which config-dir/transcript artifacts reliably
  distinguish "startup complete, awaiting first token" from a from-source-build/uv/MCP wedge at
  t<120s. Task λ validates empirically before ν builds on it.

## Cross-PRD relationship (G4)

No cross-PRD seams: no other active PRD owns classifier precedence, the requeue lane, the
watchdog, or account state. Related-but-disjoint: `plans/capability-delivered-checks-prd.md`
(decompose mechanics only); the sibling host-overload investigation (journal-volume anomaly)
is explicitly out of scope here. The escalation vocabulary gains one category (`api_degraded`)
owned by this PRD.

## Contract section (G5: B+H)

### C1 — Failure classification (shared)

- `is_server_error_status(status: int | None) -> bool` ≡ `status is not None and 500 <= status <= 599`.
  Single source; classifier, workflow, and scheduler all call it (INV-5).
- `InvocationOutcome` gains `ServerError(status: int)`. Precedence: AuthFailed > ModelNotFound(404)
  > OK > ModelNotFound(marker) > CliLocalError > CapHit/NearCap > **ServerError** > ZeroOutputWedge
  > Failure. (Below cap detection: a 5xx body never carries cap prefixes, and 429-body semantics
  must not move. Above wedge: a timed-out 529 flush is a ServerError, not a wedge.)
- `classify_agent_failure`: new rule between `ended_awaiting_background` and `timed_out`:
  `is_server_error_status(result.api_error_status)` → kind `API_ERROR`, summary
  `agent API error: HTTP <status>` + appended kill context when `timed_out`
  (e.g. `(killed at <N>s pre-first-token; transcript_turns=0)`). Rules for 4xx unchanged.
- `invoke_with_cap_retry`: on ServerError — no failover, no account mutation, heuristic cap net
  suppressed, result returned to caller. (Phase 3 adds: report to ApiHealthGate.)

### C2 — Terminal report / requeue routing (orchestrator)

- `TerminalReport.api_error_status: int | None = None`, threaded → `TaskReport` →
  `record_requeue(api_error_status=...)`. Routing: field-first via `is_server_error_status`,
  regex `agent API error: HTTP (\d{3})` retained as fallback for legacy reasons only.
- **Requeue-not-block invariant**: a phase attributing a failure to ServerError returns REQUEUED
  via the existing `_requeue` machinery with the marker in reason AND the structured field set;
  it MUST NOT write `blocked`, file an L0, or spawn a steward. `blocked` for this class is
  reachable only via transient-cap exhaust.

### C3 — Transient backoff (scheduler)

- `cooldown(n) = min(transient_requeue_backoff_base_secs · 2^(n-1), transient_requeue_backoff_cap_secs)`;
  armed value = `cooldown/2 + U(0, cooldown/2)` (equal jitter). Defaults 30.0 / 900.0.
  Applies only to transient-classified requeues; genuine requeues keep flat
  `requeue_cooldown_secs`. Counter basis: the task's transient count at arming.

### C4 — ApiHealthGate (shared; Phase 3)

- `report(outcome, *, task_id, account, role)` from the invocation choke point;
  `state() -> Closed | Open(since, stats) | Probing`.
- Trip: within `window_secs`: 5xx-count ≥ `min_failures` ∧ distinct task_ids ≥
  `min_distinct_tasks` ∧ 5xx/(completed invocations) ≥ `failure_rate_threshold`.
- Close: 2 consecutive successful invocations (probe or natural). Open-state consumers:
  scheduler (throttle new dispatches to ≤1 concurrent probe), workflow (suppress steward spawn
  for 5xx-attributed L0s), harness (defer watcher rotation unless non-API L1s pending),
  escalation lifecycle (file `api_degraded` on open; auto-resolve on close; promote to L2 after
  `max_open_before_l2_hours`), park-stop auto-resolve per decision 11.
- Persistence: in-memory state + `api_error` rows in `account_events` for forensics/dashboard.
  Config block `api_health.*`: `enabled=true, window_secs=600, min_failures=8,
  min_distinct_tasks=3, failure_rate_threshold=0.5, max_open_before_l2_hours=2.0,
  park_stop_auto_resume=true, park_stop_resume_settle_secs=300`.

### C5 — Two-regime startup grace (shared; Phase 3)

- Startup regime kill at `startup_grace_secs` fires only when the startup-completion predicate
  (artifact basis from λ) is FALSE. When TRUE and the process is alive awaiting first token, the
  pre-turn-1 bound extends to `server_error_startup_grace_secs` (default 900.0; never above the
  per-role ceiling). Predicate unreadable → conservative degrade to today's 120s behaviour
  (mirrors the existing B7 pattern).

## Boundary-test sketch (G5: B+H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Timed-out 529 with SIGTERM-flushed JSON (incident shape) | AgentResult: timed_out=True, transcript_turns=0, api_error_status=529 | kind=API_ERROR; marker+field present; execute returns REQUEUED; no `blocked` write; no L0; zero-output counter unchanged |
| 2 | SIGKILL sub-path (no JSON flushed) | timed_out=True, api_error_status=None, transcript_turns=0 | classifies wedge exactly as today; counter increments; blocks at 2 with `infra_issue` |
| 3 | Genuine local wedge ×2 (reify-4429 shape) | two consecutive no-status zero-output timeouts | BLOCKED `infra_issue` at threshold — unchanged |
| 4 | Transient requeues back off | task requeues transiently n=1..5 | armed cooldowns grow ~30→480s (jittered, within envelope); genuine requeue stays flat 30s |
| 5 | Burst across M tasks (park-stop protection) | ≥15 tasks hit shape #1 within 1h | park-stop deque stays empty; scheduler never pauses; 0 per-task L1s |
| 6 | Transient cap exhaust | 10 transient requeues, provider still down | 11th blocks; `retry_cap_exhausted` L1 names HTTP status + n_transient |
| 7 | Review: all-5xx reviewers | both reviewers return api_error_status=529 | phase REQUEUEs with marker/field; in-phase retry burn skipped |
| 8 | Review: quality error | reviewer succeeds but emits invalid verdict | blocks exactly as today (no misrouting to transient) |
| 9 | Fast 529 (<5s, zero cost) | non-timed-out 529 result in cap-retry loop | classified ServerError; account NOT marked capped; no failover |
| 10 | 429 / 401 regression | cap-body 429; auth 403 | 429 → cap machinery unchanged; 401/403 → auth failover unchanged |
| 11 | Breaker trip + recovery (Phase 3) | injected 5xx storm ≥ thresholds, then recovery | exactly 1 `api_degraded` escalation filed then auto-resolved; dispatch throttled to probe trickle while open; steward spawns suppressed; full dispatch resumes on close |
| 12 | Park-stop auto-resume (Phase 3) | park-stop tripped with ≥50% 5xx-attributed blocks; gate closes | sentinel L1 resolved with evidence note; scheduler resumes within settle window; non-5xx park-stop pauses never auto-resume |
| 13 | Watchdog two-regime (Phase 3) | (a) startup-complete then silent stall; (b) pre-startup wedge | (a) survives 120s, killed at extended bound, classified per C1; (b) killed at 120s as today |

## Decomposition plan (G2 signals; Greek labels → task ids at decompose)

**Phase 1 — incident-shape vertical slice**

- **α — shared: 5xx classification** (`shared`): `is_server_error_status`, `ServerError` variant,
  `classify_agent_failure` 5xx-above-timed_out rule, cap-net guard + no-failover in
  `invoke_with_cap_retry`, drift-guard/test updates. *Signal*: boundary rows 1 (classification
  half), 9, 10 green in shared test suite. *Prereqs*: none.
- **β — orchestrator: structured transient routing** (`orchestrator`):
  `TerminalReport.api_error_status` threaded to `record_requeue`; field-first
  `is_transient_api_requeue` with regex fallback; stale scheduler.py:398-403 comment corrected
  here (it documents this exact seam). *Signal*: a REQUEUED report with the field set lands in
  `_transient_requeue_counts` with the marker-regex deleted from the assertion path of the new
  test (field alone suffices). *Prereqs*: none.
- **γ — execute loop: requeue-not-block** (`orchestrator`): first 5xx-attributed zero-output
  result → REQUEUED via `_requeue` with marker+field; zero-output counter untouched; no L0/steward.
  *Signal*: boundary rows 1, 2, 3 green in an orchestrator integration test driving
  `_execute_iterations` with injected results. *Prereqs*: α, β.
- **δ — scheduler: jittered transient backoff** (`orchestrator`): C3 formula + two config knobs +
  defaults.yaml entries. *Signal*: boundary row 4 green; `get_scheduler_state` (product read
  path) shows growing `_requeue_until` deltas across simulated transient requeues. *Prereqs*: β.
- **ε — observability + prompt/doc corrections** (`orchestrator`): `api_error_status` + output
  tail in `_capture_zero_output_evidence`; roles.py:1167-1168 steward guidance corrected (5xx:
  retry-later, failover does NOT help; distinguish from 4xx); evidence-path docstring fixes
  (workflow.py:6272/:6114 `.task/` → `.task-meta/`); OPERATIONS.md:809 row updated to name the
  evidence field. *Signal*: evidence JSON from a simulated 529 zero-output run contains
  `api_error_status: 529` and the CLI error tail; OPERATIONS.md row tells the operator where to
  look. *Prereqs*: none.
- **ζ — Phase-1 integration gate** (leaf, `orchestrator`): end-to-end harness simulation of the
  incident burst. *Signal*: boundary rows 1-5 green in one harness-level test: N simulated 529
  kills across M tasks → all requeue transiently with growing cooldowns, park-stop never trips,
  zero L0/L1 filed, tasks recover to `done` when injected results turn healthy. *Prereqs*: γ, δ, ε.

**Phase 2 — coverage**

- **η — review phase** (`orchestrator`): per decision 8 — 5xx-attributed reviewer detection,
  retry-burn skip, all-5xx → REQUEUED with marker/field; mixed/quality errors keep blocking.
  *Signal*: boundary rows 7, 8 green in review-phase integration tests. *Prereqs*: α, β.
- **θ — planning + simple_task + exhaust text** (`orchestrator`): planning 5xx → direct
  `_requeue` (no L0/steward); simple_task 5xx → real scheduler REQUEUED (not architect
  fall-through); transient-cap-exhaust report/L1 text names HTTP status distribution + last error
  tail. *Signal*: simulated 529 planning invocation → task `pending` with transient count 1, no
  steward invocation recorded, no L0 in the escalation store (product read paths: `get_task`,
  escalation list); boundary row 6 text assertion green. *Prereqs*: α, β.
- **κ — Phase-2 integration gate** (leaf, `orchestrator`): cross-phase e2e — 529s injected at
  planning/review/simple paths all requeue transiently; cap exhaust yields exactly one
  `retry_cap_exhausted` L1 per task naming the outage. *Signal*: boundary rows 6-8 green at
  harness level. *Prereqs*: η, θ, ζ.

**Phase 3 — breaker + watchdog**

- **λ — substrate validation: startup-completion artifacts** (`shared`, investigation-output
  task): empirically characterize what exists (config-dir files, transcript JSONL, /proc state)
  at t<1s for a healthy `-p` invocation vs during a from-source-build/uv/MCP wedge; commit the
  matrix + chosen predicate + fixtures. *Signal*: committed doc + machine-readable fixture pair
  under `shared/tests/fixtures/` that ν's tests consume; the doc names the predicate and its
  failure-mode table. *Prereqs*: none.
- **μ — ApiHealthGate core** (`shared`): C4 state machine, `api_error` account-event rows,
  unit tests incl. partial-degradation rate cases. *Signal*: gate trips/closes per C4 in unit
  suite; `account_events` rows queryable via the existing cost-store read path. *Prereqs*: α.
- **ν — watchdog two-regime grace** (`shared`): C5 using λ's predicate;
  `server_error_startup_grace_secs` knob. *Signal*: boundary row 13 green (both halves).
  *Prereqs*: λ, α.
- **ξ — gate integration: throttle + suppression** (`orchestrator`): `invoke_with_cap_retry`
  reports to the gate; scheduler dispatch throttle (probe trickle) while open; steward-spawn and
  watcher-rotation gating per C4. *Signal*: injected 5xx storm at harness level trips the gate,
  dispatch drops to ≤1 concurrent, steward spawns suppressed for 5xx-attributed L0s, all restored
  on close. *Prereqs*: μ.
- **ο — operator surface** (`orchestrator` + `escalation` + `dashboard`): `api_degraded`
  escalation lifecycle (file/auto-resolve/L2-promote per C4), park-stop auto-resume per decision
  11, dashboard provider-health strip from `api_error` events. *Signal*: boundary rows 11
  (escalation half), 12 green; dashboard page renders the 5xx-rate strip from a seeded
  `account_events` fixture. *Prereqs*: μ, ξ.
- **π — Phase-3 integration gate** (leaf, `orchestrator`): full-storm simulation across roles +
  steward + watcher invocations. *Signal*: boundary rows 11-13 green in one harness run: ≤1 fleet
  escalation, no per-task L1 storm, throttled dispatch, auto-recovery, park-stop either never
  trips or auto-resumes with evidence note. *Prereqs*: ν, ξ, ο.

G7 walk (advisory, this session): INV-1 satisfied by M2 (field over regex); INV-2 by ε/θ/ο
(structured status at every failure surface); INV-3 by decision 11's live-gate corroboration;
INV-4 by decision 12's escapes; INV-5 by `is_server_error_status` single-sourcing. No waivers
anticipated; decompose re-walks per gate order.

## Out of scope

- The journal-volume anomaly while paused (99k-163k lines/h) — sibling host-overload
  investigation owns it.
- Why the sentinel-L1 auto-resume never engaged during the 2026-07-29 incident — cheap one-off
  journal check, not a mechanism; noted for the operator.
- Eval-metrics 5xx attribution (`evals/metrics.py:286` is 429-only) — eval-infra nicety, file
  separately if wanted.
- Per-project (reify) config edits and fleet restart orchestration — existing redeploy machinery.
- Codex/Gemini backend 5xx vocabularies — `ServerError` keys on the structured
  `api_error_status` only; backend-specific error-body parsing stays as-is.
- Making park-stop category-aware — explicitly rejected (decision 6).

## Open questions (tactical)

1. **Ordering of `record_requeue` vs `release()` cooldown-arming** — the backoff (δ) needs the
   post-increment transient count at arming; verify call order in `_run_slot`'s finally path and
   thread the count if release runs first. Decide in δ.
2. **Hot-reload tier assignment** for the new knobs (`transient_requeue_backoff_*`,
   `api_health.*`, `server_error_startup_grace_secs`) — green tier preferred (scheduler/watcher
   tuning family); confirm against reload plumbing in each task.
3. **Probe implementation for gate-close** (μ/ξ): dedicated cheap invocation vs piggyback on the
   natural probe trickle. Suggested: natural trickle only (no new invocation type). Decide in ξ.
4. **`api_degraded` escalation level at filing** (ο): L1 (auto-watcher visible) vs born-at-L2.
   Suggested: L1 with the `max_open_before_l2_hours` promotion. Decide in ο.
5. **Exact kill-context format** appended to the API_ERROR summary when timed_out (α) — keep the
   marker regex-compatible prefix `agent API error: HTTP <N>` verbatim; context suffix free-form.
