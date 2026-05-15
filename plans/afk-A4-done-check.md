# AFK A4: Auto-resolve escalations for terminal tasks; route review_suggestions direct to curator

## Problem

The orchestrator submits new escalations even when the target task is already `done` or `cancelled`. The steward then fetches the escalation, looks up the task, sees it's terminal, and dismisses with reason "task already DONE". Wasteful round-trip; during AFK each one ties up steward budget with no benefit.

Separately, `review_suggestions` escalations route through the steward today despite the steward adding no unique value over the curator (per the A4.6 deep-dive). The path is fragile under steward timeout/budget limits and inefficient (steward and curator both deduplicate).

## Solution — two changes

### 1. Terminal-task chokepoint in escalation MCP handlers

In the orch's escalation MCP server's `escalate_blocker` and `escalate_info` handlers, before queueing an incoming escalation:

- Look up the task's status via the existing fused-memory client
- If status ∈ `{done, cancelled}`: file the escalation but immediately mark it `resolved` with `resolution=f"auto-resolved: task already terminal (status={s})"` and `resolved_by="escalation-mcp-pre-submit-check"`. Caller receives a successfully-filed escalation that is already resolved.
- Bypass parameter: handlers accept `terminal_state_is_the_bug: bool = False`. When True, skip the chokepoint entirely. The four `gate_failure` call sites at `orchestrator/src/orchestrator/workflow.py:378`, `:394`, `:999`, `:4034` pass `True` — their entire point is "the row says terminal but my workflow disagrees, that's the bug."
- Exception: `category=review_suggestions` skips the chokepoint (the routing change in §2 takes over).

### 2. Direct-to-curator routing for review_suggestions

At the workflow.py site that today emits review_suggestions escalations (~line 4720, per A4.6 agent), replace the escalation submission with a direct curator call:

- Parse the suggestion JSON array
- Convert each suggestion to a `CandidateTask` (existing schema in `fused-memory/src/fused_memory/middleware/task_curator.py:168`) with `priority='low'`, `spawned_from=<task_id>`, `spawn_context='review_suggestions'`, `details=<original suggestion JSON>`, `title=f"[{sugg['category']}] {sugg['location']}: {sugg['description'][:60]}"`
- Call `task_curator.curate_batch(candidates)` directly
- No L0 escalation filed. Steward never sees suggestions.

Justification (from A4.6 agent): we now run a single generalist reviewer (not five specialists), so in-batch suggestion duplication is rare; the curator's payload-hash dedup absorbs what remains; the steward's value-add was pre-triage filtering for batches ≥10 (configured threshold), and the curator's batch API natively supports filtering.

## Files to touch

- Orch escalation MCP handlers (find via `grep -rn "escalate_blocker\|escalate_info" orchestrator/src/orchestrator/mcp/`)
- `orchestrator/src/orchestrator/workflow.py:378`, `:394`, `:999`, `:4034` — add `terminal_state_is_the_bug=True` to these four `escalate_to_human` callers
- `orchestrator/src/orchestrator/workflow.py:~4720` — replace `escalation_queue.submit(esc)` with the direct curator call when category=review_suggestions
- Orchestrator unit tests for each path

## Acceptance criteria

- A new escalation against a `done` task is auto-resolved at submission with the audit reason; same for `cancelled`
- `deferred` and `blocked` task escalations remain open (not auto-resolved)
- The four gate_failure sites still produce open escalations even when the task is terminal (bypass flag works)
- review_suggestions never reach the escalation queue; suggestions arrive as `CandidateTask` rows in the curator
- Unit tests cover: done auto-resolve, cancelled auto-resolve, deferred kept-open, bypass flag, review_suggestions route

## Risks

- **Race**: task transitions to done between escalation decision and submission — auto-resolve fires; correct outcome (escalation is moot).
- **Curator throughput surge from raw suggestions** — generalist reviewer reduces dup rate; curator's batch dedup absorbs the rest. If load spikes, tune `CuratorConfig.batch_max` downward.
- **Lost branch-context dedup** — steward had worktree access; curator doesn't. In practice suggestions are about the diff being reviewed, so they rarely duplicate code already on the branch. Accept the trade-off.

## No dependency on A8

This brief's chokepoint runs inside the orch's escalation MCP handler; the auto-resolve `resolve_issue` call is in-process (not HTTP). A8 is only required for the recon-side closure (A7).
