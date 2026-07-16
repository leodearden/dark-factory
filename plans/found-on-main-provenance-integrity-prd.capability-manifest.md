# Capability manifest — found-on-main-provenance-integrity-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Substrate verified on main tip 3659966237, 2026-07-16. Line refs are authoring-time evidence; delivered_checks in the YAML twin are pattern-anchored.

## Leaf γ — enforce flip + end-to-end reopen-sticks test

- `interceptor-freshness-gate+typed-error` → **producer: task-α (upstream)** — chokepoint substrate exists today: `grep:fused-memory/src/fused_memory/middleware/task_interceptor.py:988,4287` (`_validate_done_provenance` invoked from the `status=='done'` block; shells git; holds pre-write metadata incl. `reopen_at`). PASS
- `gate-terminal-this-tick+conflict-L2` → **producer: task-β (upstream)** — dedupe substrate exists: `grep:escalation/src/escalation/dedupe.py:117-174` (`dedupe_fingerprint`). PASS
- `config-knob reject_stale_done_evidence` → **producer: task-α (upstream)**; config surface exists (`fused-memory/src/fused_memory/config/schema.py`). PASS
- DAG-direction: α, β upstream of γ. PASS

## Leaf ε — call-site tightening (shared helper; fallbacks removed; escalate-instead-of-stamp)

- `silent-fallbacks-exist-today (to be removed)` → `grep:orchestrator/src/orchestrator/harness.py:7479` (`anchor = citation or await self.git_ops.get_main_sha()`) and `grep:orchestrator/src/orchestrator/merge_queue.py:8980` (`resolve_branch_sha(branch) or main_sha`). PASS (delivered_check: patterns ABSENT post-ε)
- `FIX1′-second-parent-effect + subject-anchored-citation-regex` → **producer: task-δ (upstream)** — current primitives exist: `grep:git_ops.py:6116` (`commit_effect_present_in_main`), `grep:git_ops.py:219` (`DEFAULT_COMMIT_CITATION_PATTERN`). PASS
- `escalation-infra-for-no-citation-case` → existing `escalate_blocker`/`escalate_info` MCP surface (wired, exercised fleet-wide). PASS
- `single-shared-helper (INV-5)` → design obligation on ε itself; judged by review, not grep. PASS (manual)

## Leaf ζ — LandedOutbox consume-on-happy-path

- `consume()-exists` → `grep:orchestrator/src/orchestrator/landed_outbox.py:117` (`def consume(self, task_id)`) — declared and already wired on crash-recovery paths; ζ wires the happy path (the task-2155 KNOWN LIMITATION at `merge_queue.py:3865-3877`). PASS
- `happy-path-mark-done-sites-exist` → `workflow.py` mark_done call sites (RCA substrate map). PASS
- Wiring judgment is ζ's own deliverable. PASS (manual)

## Leaf η — operational-verified provenance kind + submit-time suggestion

- `provenance-kind-validator-extensible` → `grep:task_interceptor.py:4276` (kinds handled inside `_validate_done_provenance`; commitless `deterministic-*` kinds precedent — task 2648 closed with `kind='deterministic-gate'`). PASS
- `caller-class-distinction (non-recon override/close rule)` → `grep:fused-memory/src/fused_memory/middleware/recon_write_policy.py` (recon-stage caller classification exists — Gate 1 scoping). PASS
- `NO cross-service escalation lookup` → verified ABSENT (fused-memory's only escalation client is its own 8103 recon queue, `reconciliation/harness.py:1412-1428`) — design deliberately does NOT assume it; escalation_id is recorded-for-audit only. PASS (constraint, not capability)
- `submit-time-lint-surface` → existing submit-boundary guard pattern (`execution_class_guard`, routing-intent lint). PASS

## Leaf κ — deploy gate (fused-memory restart)

- `deterministic-pure-gate-preset` → live precedent: task 2648 itself ran the pure-gate cycle to `done` (`deterministic-gate` provenance, 2026-07-16). PASS
- DAG-direction: γ, η upstream. PASS

## Leaf θ — +7d soak (delayed-milestone pure gate)

- `milestone-delayed-mode` → `grep:shared/src/shared/task_metadata.py:229-239` (`Milestone`, `after_secs`). PASS
- `audit-script-exists` → `fused-memory/scripts/audit_found_on_main_provenance.py` (30KB, exercised live by task 2648's audit run). PASS
- `predicate-wrapper` → **producer: task-ι (upstream)**. PASS
- `spurious-rate-premise (G6 numeric)` — assertion is **zero new** flagged stamps with stamp-time after κ's deploy, not a rate bound; achievability basis: ε/γ close every RCA-identified stamping path, and the audit already distinguishes flag classes. No floor issue (count, not accuracy). PASS
- DAG-direction: ε, ι, κ upstream. PASS

No FAIL bindings. Batch clear to queue.
