# Capability manifest — task-status-authority (W2)

Beside `plans/task-status-authority-prd.md`. Binds each leaf's asserted capabilities to evidence,
mechanizing G3 (substrate exists) + G6 (premise valid). **Any FAIL blocks the batch.** All bindings
below are PASS at decompose time (2026-07-06); anchors re-verified against the working tree.

Batch (16 tasks): τ1=2163, τ2=2168, ρ1a=2171, ρ1b=2175, ρ2=2182, ω1=2188, ω2=2191, ε1=2193,
ω3=2196, ω4=2200, ε2=2204, ε3=2209, ε4=2211, ζ=2214, Δ1=2215, Γ=2216.

## G3 — substrate bindings (assumed capability → evidence it exists)

| Task | Assumed capability | Binding | Verdict |
|---|---|---|---|
| τ1/τ2 | `shared/` importable by orchestrator + fused-memory; `StrEnum` (3.11) | `grep:dark-factory-shared` in both pyprojects (workspace dep); requires-python `>=3.11,<4` all three | **PASS** |
| ρ1a/ρ1b | single durable status write chokepoint `_apply_status_transition` | `task_interceptor.py:619-857`; backend `update_task` rejects status, MCP rejects it, `commit_planning` routes through interceptor | **PASS** |
| ρ1b | caller identity resolvable at the interceptor | `_resolve_identity` exists (tools.py:409-441, wired to memory tools); ρ1b threads it to task tools — **additive plumbing, not assumed present** | **PASS (built by ρ1b)** |
| ρ2 | tasks.db can gain columns | `_SCHEMA_VERSION`/`_migrate`/`PRAGMA user_version` (sqlite_task_backend.py:39/165-238); full-rebuild idiom in use | **PASS** |
| ε1/ε2 | `Escalation` has `resolution_action`/`level`/`category`; `resolve_issue` reads `X-Escalation-Identity` | models.py:53/55/66/86; server.py:565-598 | **PASS** |
| ω3/ω4 | harness action dispatch + infra_hold sites exist | `_on_escalation_resolved`:8189-8316, `_ACTION_TARGETS`:8300, `_cascade_unblock_member`:8659-8756, infra_hold write:8704 / skip:3462 / stamp:workflow.py:4842 | **PASS** |
| ε3 | `escalation_id_lock` sidecar primitive for a fsync'd counter | queue.py:24-69 (stable-sidecar flock, defeats tmp+rename churn) | **PASS** |
| ε4 | code already on `datetime.UTC`; ruff UP enforces it | server.py:10 `from datetime import UTC`; pyproject `UP` selected, `requires-python>=3.11` | **PASS (no runtime change)** |

## G6 — premise bindings (asserted number/exactness/capability/rejection → validation)

| Task | Premise class | Premise | Evidence / de-risking | Verdict |
|---|---|---|---|---|
| τ2 | **completeness/numeric-floor** | the encoded transition set covers every transition that occurs in production (a miss bricks dispatch) | (a) derive the set by enumerating all live `set_task_status` call sites; (b) **empirical gate**: enforcement stays log-only until the Γ soak shows zero `illegal_transition would-reject` WARNINGs. Premise is never asserted, it is measured. | **PASS** |
| ε1 | **anti-inversion / rejection-backed** | the `(action,level,category)` combos marked illegal are unused by live callers (loud rejection must not strand a real workflow) | ε1's own step cross-checks the `data/escalations` archive: no combo marked illegal appears as an actually-used resolution. Rejection mechanism is built + observed to fire in ε1's own test (G6 branch-4). | **PASS** |
| ρ1a | **field-population / exactness** | the enum is a superset of every legitimate status (else a valid write is rejected) | τ1 derives `TaskStatus` as the exact union of the four current copies; ρ1a's parity test asserts it; rejection fires only for statuses ∉ that union. | **PASS** |
| ε2 | **rejection-backed / anti-lockout** | header-less callers keep full authority (a default-deny would repeat the esc-2087-2 lockout) | test C1: a header-less `resolve_issue(park\|close_only\|resume)` on an L2 succeeds; the identity→ceiling map fires only for mapped identities; the deployed auto-watcher is asserted a no-op. | **PASS** |
| ρ2/ω1 | **rejection-backed** | claimant writes fail safe under version skew (a partial restart must not error) | guarded/feature-detected write; test asserts a backend without the columns does not raise. | **PASS** |

## Anti-orphan / wired — every producer has a queued consumer

| Producer (task) | Consumer (task or named surface) |
|---|---|
| τ1 shared vocab | ρ1a, ρ1b, ρ2, ω2, ε1 (queued); W3/W9/W10 (downstream, wave 2) |
| τ2 transition table + `outcome_allows_status` | ρ1b (enforcement); W9 `WorkflowStateMachine` (downstream) |
| ρ2 claimant columns + `is_stranded` | ω1 (stamping, queued); W10 `TaskGroundTruth` (downstream) |
| ε1 `ACTION_EFFECTS` table | resolve_issue (in ε1); ω3 harness (queued) — **same table both sides** |
| ω4 `infra-hold` status + `is_infra_held` | reconcile sweep + resume path (both in ω4) |
| ε2 role ceilings / promote gate | resolve_issue + promote_to_l2 (in ε2); auto-watcher connection (unchanged) |
| ε3 ID counter | queue.make_id/get_by_task (in ε3) |
| every code leaf | ζ integration gate (2214) exercises it end-to-end |
| ζ green | Δ1 fleet deploy (2215) → Γ enforcement flip (2216) |

No orphan producer: every mechanism introduced is consumed by a queued sibling or a named downstream
stream (W3/W9/W10 declare their deps on this batch in wave 2). No fake-done risk: each leaf's signal is a
test/grep/live-observation through the product's own read paths, and the high-stakes seam has the ζ
two-way boundary gate before deploy.

## Deploy note (surfaced to the orchestrator)

Δ1/Γ are **deterministic pure gates** (`task_kind=deterministic`, `always_escalates=true`, no
`before_done`) — a born-at-L2 escalation with an operator runbook, not an automated restart script
(PRD D8). fused-memory restarts use out-of-cgroup `systemctl --user restart fused-memory.service`
(program decision #6), never `--drain`. The `user_observable_signal`/`consumer_ref`/`substrate_confirmed`
metadata fields are substrate for a future tracking-infra session — the orchestrator does not read them today.
