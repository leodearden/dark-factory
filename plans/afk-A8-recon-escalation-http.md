# AFK A8: Wire fused-memory recon to orchestrator escalation MCP via HTTP (paused — gated on A7)

## Status

**Paused.** A8 is the HTTP transport for A7's closure wiring. A7 was paused on 2026-05-15 pending investigation into the actual problem (5,315 actionable findings sit dormant in `stage_reports` without escalations; see afk-A7-recon-closure.md). With no A7 consumer, A8 has no caller. Revisit when A7 scope is settled.

## Pre-resumption corrections

Three issues from the original plan that must be addressed before resuming:

1. **`resolve_issue` handler does not accept `project_id`** today (`escalation/src/escalation/server.py:119-139` takes only `escalation_id, resolution, terminate, resolved_by, resolution_turns`). The plan's "orch validates and rejects mismatch with 400" is not enforceable as written; either add the parameter or drop the validation claim.
2. **Streamable-HTTP MCP is a 4-step protocol** (initialize → notify → call → close). The 5s timeout + 1 retry budget needs to cover 4 round-trips per close call. Pre-existing `procedural_streamable_http_mcp_session.md` snippet should be reused.
3. **HTTP library unspecified.** fused-memory has no httpx/aiohttp; `urllib.request` is blocking and unfit for the async harness. Choose a library before implementation.

## Original plan below

## Problem

Reconciliation stage-2 emits findings as info-escalations and runs a remediation pass on actionable ones, but it cannot close those escalations because escalation MCP tools are not surfaced to fused-memory's reconciliation harness. Result: findings remain `status=pending` forever even after remediation. Backlog: ~2,347 actionable pending escalations.

## Solution

Add an HTTP client in fused-memory that calls the orchestrator's escalation MCP. Project-routed:
- `dark_factory` → `127.0.0.1:8102`
- `reify` → `127.0.0.1:8100`

Mapping in fused-memory config under `escalation.project_ports`. ~50-line client supporting at minimum `resolve_issue(escalation_id, resolution, resolved_by)` (other escalation MCP methods can be added on demand).

Behavior:
- 5s connect/read timeout
- 1 retry on 5xx or connection-refused
- On final failure: log structured event, return typed `EscalationClosureFailed` exception or `False`; the recon cycle MUST NOT abort

## Files to touch

- `fused-memory/src/fused_memory/reconciliation/escalation_client.py` (new)
- `fused-memory/config/config.yaml` (add `escalation.project_ports` mapping)
- `fused-memory/src/fused_memory/reconciliation/harness.py` (consumer wiring; the actual close calls land here as part of A7)
- `fused-memory/tests/reconciliation/test_escalation_client.py` (new)

## Acceptance criteria

- Stage-2 can call `escalation_client.resolve_issue(...)` and successfully close an escalation in either project (verified end-to-end against a live orch escalation MCP)
- Orch-down case: HTTP fails after one retry; caller observes the typed failure; recon cycle continues; pending finding stays in queue for next cycle
- Audit: every closure attempt (success or failure) recorded with timestamp, escalation_id, project, outcome, latency_ms
- Unit tests cover: 200 success, 5xx + retry-success, 5xx + retry-fail, connection-refused, timeout

## Risks

- **Orchestrator restart causes connection failure** — safe by design; finding stays pending, retried next cycle.
- **Wrong port mapping silently writes to wrong project's queue** — mitigation: include `project_id` in the resolve call body; orch validates and rejects mismatch with 400.
- **Firewall / port binding change** — config-driven port mapping makes it a one-line fix.

## Out of scope

Filing new escalations from recon — recon already does this via direct queue write. This brief is closure-only.
