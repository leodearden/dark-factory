# Capability manifest — usage-gate-model-scoped-caps-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), authored at decompose
2026-07-20. Line refs are as-of main `935e185d11` and drift; the YAML sidecar
twin carries the pattern-anchored delivered_checks.

## α — Scope substrate: config + state + derivation
- `usage-cap-config-extensible` → **PASS** — `class UsageCapConfig` (pydantic,
  `shared/src/shared/config_models.py:25`); additive `scoped_cap_models` field.
- `account-state-extensible` → **PASS** — `@dataclass AccountState`
  (`shared/src/shared/usage_gate.py:133`); additive `scope_caps` dict.

## β — Scope-aware attribution
- `model-in-scope-at-slot-acquisition` → **PASS (wired)** —
  `invoke_kwargs.get('model', ...)` at `shared/src/shared/cli_invoke.py:968`
  precedes `usage_gate.invoke_slot()` at `:1104` on the production retry loop.
- `caphit-report-seam` → **PASS (wired)** — `InvokeSlot.report` dispatches
  `CapHit → _handle_cap_detected` (`usage_gate.py:367-368`); scope rides the
  same call.
- `resets-at-parse-reuse` → **PASS (wired)** — `_parse_resets_at`
  (`usage_gate.py:1693`) already generic over "resets …" phrases.

## γ — Scope-aware selection, waiting, uncap + read API
- `before-invoke-selection-loop` → **PASS (wired)** — the account-iteration
  predicate (`usage_gate.py:680-715`); scope skip is an additional condition.
- `timer-sweep-precedent` → **PASS (wired)** — `_refresh_capped_accounts`
  (`usage_gate.py:1211`).
- `retry-bound-storm-escape` → **PASS (wired)** — `consecutive_cap_hits` /
  `max_cap_retries` / `cap_wait_sanity_secs` → `AllAccountsCappedException`
  (`cli_invoke.py:995-1023`); γ must feed scoped re-caps through the same
  counters (INV-4).

## δ — Resolver capacity fail-safe
- `resolver-fail-safe-layer-skip` → **PASS (wired)** — the per-model-ceiling
  layer-skip + `RoutingDecision.rejected` mechanics
  (`orchestrator/src/orchestrator/routing.py`, task 2535 landed;
  `model-ceiling-exhausted` precedent). The capacity check is a third check
  with identical semantics.
- `invoke-chokepoint-threading` → **PASS (wired)** — `TaskWorkflow._invoke`
  adopts `resolve_route` (task 2535); the gate snapshot threads at the same
  call site.

## ε — Scope integration gate (leaf; B1–B8)
- Every capability the suite exercises is produced upstream in this batch:
  scope state (α), attribution (β), selection/wait/snapshot (γ), resolver
  check (δ) — **DAG-direction PASS** (producers all upstream). The fake
  `invoke_fn` injection seam exists (`invoke_with_cap_retry(invoke_fn=...)`,
  wired; harness-backend T1 boundary test precedent).

## ζ — Docs + operator surface (leaf)
- `claude-md-routing-section` → **PASS** — CLAUDE.md "Model Routing" section
  exists to extend; `skills/orchestrate/SKILL.md` reload section exists.

No FAIL bindings; batch clear to queue.
