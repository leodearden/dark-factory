# Capability manifest — server-side-api-error-handling-prd

Mechanizes G3+G6 for `plans/server-side-api-error-handling-prd.md`. Evidence
bindings verified against main at decompose time (2026-07-30, HEAD
`515b02b82b`). Line numbers are as-verified-today; the YAML sidecar twin
(`server-side-api-error-handling-prd.capability-manifest.yaml`) carries the
pattern-anchored `delivered_check`s that survive drift.

Substrate note: the PRD's forensic line anchors drifted slightly since the
design session (e.g. `classify_agent_failure` timed_out rule now :657,
api_error rule :690; `record_requeue` now scheduler.py:7178) but **every
claimed symbol/mechanism was re-verified present** — including the exact
wrong steward guidance at `agents/roles.py:1167-1168` and the stale
`.task/zero_output_evidence` docstrings (workflow.py:6423/:6581) that ε
corrects. `cli_invoke.py` lives in `shared/src/shared/` (the PRD's bare
`cli_invoke.py:N` refs), not the orchestrator package.

## α — shared: 5xx classification

- `AgentResult.api_error_status` field exists → grep:shared/src/shared/cli_invoke.py:291 — PASS
- `classify_agent_failure` rule table with timed_out (:657) ranked above api_error_status (:690) — the inversion α corrects → grep:shared/src/shared/cli_invoke.py:579-693 — PASS
- `InvocationOutcome` tier set + documented precedence to insert `ServerError` into → grep:shared/src/shared/invocation_outcome.py:36-47,372-373 — PASS
- Heuristic zero-cost cap net to guard (`not success ∧ cost==0 ∧ turns≤1 ∧ <5s`) → grep:shared/src/shared/cli_invoke.py:1303-1315 — PASS
- SIGTERM-flush harvests result JSON so a killed 529 still stamps `api_error_status` (incident signature) → grep:shared/src/shared/cli_invoke.py:2190-2225 (harvest), :1759/:1837 (stamp) — PASS

## β — orchestrator: structured transient routing

- `TerminalReport` → producer:workflow_types.py:92; `TaskReport` → harness.py:870 — PASS
- `record_requeue` recording site + `_transient_requeue_counts` → grep:orchestrator/src/orchestrator/scheduler.py:7178,:1496,:7227-7247 — PASS
- `is_transient_api_requeue` regex site (becomes fallback-only; INV-1/INV-5) → grep:orchestrator/src/orchestrator/scheduler.py:404-428 — PASS
- Stale comment block documenting this seam (scheduler.py:398-403 in PRD; adjacent to the regex today) — corrected here — PASS

## γ — execute loop: requeue-not-block

- Existing `_requeue`/REQUEUED machinery (workflow REQUEUED returns, harness `record_requeue` call :8410, scheduler :7178) — PASS
- Zero-output counter + breaker to leave untouched for 5xx → grep:orchestrator/src/orchestrator/workflow.py:6573; config.py:2659 — PASS
- `is_server_error_status` single source → producer:α upstream — PASS (DAG-direction: α is a prereq)
- Structured field on the report path → producer:β upstream — PASS

## δ — scheduler: jittered transient backoff

- `_requeue_until` arming site + dispatch-eligibility reader → grep:orchestrator/src/orchestrator/scheduler.py:7046,:4478 — PASS
- Flat `requeue_cooldown_secs` default 30.0 to preserve for genuine requeues → grep:orchestrator/src/orchestrator/config.py:2760 — PASS
- Transient count basis at arming → grep:orchestrator/src/orchestrator/scheduler.py:7227-7247 — PASS (open question 1 — record/release ordering — decided in-task)
- Product read path `get_scheduler_state` for the signal → grep:fused-memory/src/fused_memory/server/tools.py:5138; δ extends the snapshot if cooldown deadlines are not yet exposed (producer:δ itself) — PASS
- Backoff envelope achievable: min(30·2^(n-1), 900), equal jitter d/2+U(0,d/2); n=1..5 → 30→480s (numeric-premise check: monotone, capped, jitter floor 15s > 0) — PASS

## ε — observability + prompt/doc corrections

- `_capture_zero_output_evidence` → grep:orchestrator/src/orchestrator/workflow.py:6762 — PASS
- Wrong steward guidance to correct ("API_ERROR … account failover often helps. Retry is reasonable.") → grep:orchestrator/src/orchestrator/agents/roles.py:1167-1168 — PASS (rejection-style delivered_check: text must become ABSENT)
- Stale `.task/` evidence-path docstrings → grep:orchestrator/src/orchestrator/workflow.py:6423,:6581 — PASS (delivered_check: ABSENT)
- OPERATIONS.md 529-burst troubleshooting row to extend → grep:OPERATIONS.md:~810 ("A burst of zero-output invocations…") — PASS

## ζ — Phase-1 integration gate (leaf)

- All legs produced upstream (α classifier, β routing, γ execute path, δ backoff, ε evidence) — DAG-direction PASS
- Harness-level simulated-result test substrate: orchestrator/tests/ carries existing injected-result integration suites (e.g. test_retry_cap.py, test_upstream_stall_classification.py) — PASS

## η — review phase

- Review ERROR fold + marker-free blocking to make 5xx-aware → grep:orchestrator/src/orchestrator/workflow.py:8140-8240 — PASS
- Reviewer stagger/in-phase retry burn site (flat 2s; PRD :7839-7840 region) — PASS
- Classifier + field → producer:α,β upstream — PASS

## θ — planning + simple_task + exhaust text

- Planning-phase `classify_agent_failure` call sites → grep:orchestrator/src/orchestrator/workflow.py:3906,:4023 — PASS
- simple_task architect fall-through sentinel to replace with real REQUEUED → grep:orchestrator/src/orchestrator/workflow.py:2455-2464 — PASS
- Transient-cap-exhaust report with n_transient/n_genuine breakdown → grep:orchestrator/src/orchestrator/scheduler.py:7291-7306 — PASS
- Classifier + field → producer:α,β upstream — PASS

## κ — Phase-2 integration gate (leaf)

- All legs upstream (η, θ, ζ) — DAG-direction PASS

## λ — substrate validation: startup-completion artifacts

- Deliberately-unverified substrate (which artifacts distinguish "startup complete, awaiting first token" from a build/uv/MCP wedge at t<120s) — this task IS the G3 resolution (b): empirical characterization, committed matrix + fixtures under shared/tests/ that ν consumes — PASS by construction (investigation-output)

## μ — ApiHealthGate core

- `account_events` store to extend with `api_error` rows → grep:shared/src/shared/cost_store.py (account_events) — PASS
- `InvokeSlot.report` choke point that today drops Failure outcomes → grep:shared/src/shared/usage_gate.py:389,:443-444 — PASS
- `ServerError` outcome → producer:α upstream — PASS

## ν — watchdog two-regime grace

- `startup_grace_secs=120.0` watchdog + `_run_subprocess` kill decision → grep:shared/src/shared/cli_invoke.py:797,:1613,:1929 — PASS
- Startup-completion predicate basis → producer:λ upstream (hard dependency; PRD pre-conditions name this) — PASS
- Classifier for the killed-at-extended-bound result → producer:α upstream — PASS

## ξ — gate integration: throttle + suppression

- `invoke_with_cap_retry` reporting seam → grep:shared/src/shared/cli_invoke.py (cap-retry loop, :1189 region) — PASS
- Scheduler dispatch-eligibility seam for the throttle → grep:orchestrator/src/orchestrator/scheduler.py:4478 region — PASS
- Steward-spawn and watcher-rotation sites to gate → orchestrator workflow/harness (steward spawn; watcher rotation) — PASS
- `ApiHealthGate` → producer:μ upstream — PASS
- `api_health.*` config block lands here (orchestrator config + defaults.yaml) — producer:ξ itself — PASS

## ο — operator surface

- Escalation filing + `scheduler_paused` sentinel-L1 + resolve→`force_resume_scheduler` auto-resume path → grep:orchestrator/src/orchestrator/harness.py:5983,:6538,:6655 — PASS (decision 11's evidence-gated resume builds on this)
- Park-stop trip wiring (`on_park_stop_trip` → `pause_scheduler`) → grep:orchestrator/src/orchestrator/scheduler.py:1880-1944,:2280; harness.py:1270 — PASS
- `api_error` account-event rows for the dashboard strip → producer:μ upstream — PASS
- Gate open/close signal → producer:μ,ξ upstream — PASS
- `api_degraded` escalation category: NEW vocabulary owned by this PRD (G4 note) — producer:ο itself — PASS

## π — Phase-3 integration gate (leaf)

- All legs upstream (ν watchdog, ξ throttle/suppression, ο lifecycle) — DAG-direction PASS

## FAIL bindings

None. No `declared-only`, `test-only`, `producer-downstream`, `producer-absent`,
`fixture-ERROR`, `bound≤floor`, or `rejection-absent` verdicts — the batch is
clear to queue.
