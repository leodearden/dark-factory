# WP-D: Reconciliation backlog escalation policy

## Mission
Bound the reconciliation event backlog and fail loudly when the bound is exceeded. No more silent growth for days.

## Defect context
On 2026-04-17, reify's event_buffer reached **803 buffered events** while the project was halted by the judge circuit-breaker. No agent, dashboard, or operator saw the growth until fused-memory finally deadlocked on SQLite writes. Project memory confirms: `buffer_size_threshold = 250` exists in config but isn't used as an enforcement gate.

## Design decision (agreed)
Fundamental infra: **work stops when it's broken.** Two branches:

1. **Orchestrator is running** for the affected project → push an **L1 escalation** into that project's escalation queue so the orchestrator halts and surfaces the problem. Don't silently mutate.
2. **No orchestrator** → **reject all WRITE MCP tools** for the affected project with a structured error; reads still work.

Also: the judge halt condition must emit an escalation / prominent signal — it was silent on 2026-04-17.

## Scope

### 1. Backlog threshold check
- Config key: use `buffer_size_threshold` from existing config (or pick a new key if semantics diverge — document choice). Default suggestion: 500 (above the existing 250, so routine ops have slack; still well below the 803 we saw).
- Implement a helper `backlog_size_for(project_id) -> int` that queries the event_buffer table. Cheap (COUNT with index on project_id+status).
- Call it:
  - At each mutating MCP tool entry, OR
  - Periodically (e.g. every 5s in the harness loop) with the result cached in-memory for sub-second lookups.
- Prefer periodic + cached to avoid COUNT-per-write overhead.

### 2. Orchestrator detection
An orchestrator is "running for project P" if there's an active orchestrator process with project P as its target.

Detection strategies (pick one or combine):
- **Lock file**: project memory mentions `data/orchestrator/orchestrator.lock` — check its presence and freshness. Read its content; it probably has the target project path.
- **Process scan**: `pgrep -af 'orchestrator run.*<project_root>'` — works but dirtier.
- **Escalation-MCP registry**: if there's a live registry of active orchestrators, use that.

Document your choice in the plan. Cache the detection result briefly.

### 3. L1 escalation write
When an orchestrator is running and backlog exceeds threshold:
- Push a new escalation to the project's escalation queue.
- Directory: `<project_root>/data/escalations/`.
- File: `esc-reconciliation-backlog-<timestamp>.json`.
- Schema: match existing escalation JSON shape (open an existing file like `/home/leo/src/reify/data/escalations/esc-1959-52.json` for reference — fields `id`, `task_id` (synthetic or null), `agent_role`, `severity: "blocking"`, `category: "infra_issue"`, `summary`, `detail`, `level: 1`, etc.).
- **Idempotency**: don't write a new escalation every 5 seconds. Track last-issued-ts in memory and rate-limit (once per project per N minutes, default 15).

### 4. Write rejection path
When NO orchestrator is running and backlog exceeds threshold:
- Every mutating MCP tool at the server boundary (`fused-memory/src/fused_memory/server/tools.py`) checks the cached backlog status for the project and returns:
  ```json
  {
    "error": "reconciliation backlog exceeded (N > M) for project P; writes halted until drained. See plans/wp-d-briefing.md or run scripts/drain-reconciliation.sh (if exists) to recover.",
    "error_type": "ReconciliationBacklogExceeded",
    "backlog": N,
    "threshold": M,
    "project_id": "P"
  }
  ```
- Reads (`get_tasks`, `get_task`, `search`, `get_entity`, `get_episodes`) proceed normally.
- The rejection message must be **actionable** — operator should know what to do.

### 5. Halt-condition escalation
When the judge halts a project (look at `fused_memory.reconciliation.judge._check_error_trends` / harness halt path):
- If orchestrator running → emit L1 escalation describing the halt reason.
- Otherwise → prominent ERROR log + a local halt-state file so operators can see it.

No auto-unhalt — per user: fundamental infra, human decides.

### 6. Watchdog integration
WP-C adds a watchdog for stuck-writer conditions. When that watchdog fires, route through the same escalation/rejection policy (same thresholds don't apply — use your best judgment; document in plan).

## Files
- `fused-memory/src/fused_memory/reconciliation/event_buffer.py` — add `backlog_size_for(project_id)`.
- `fused-memory/src/fused_memory/reconciliation/harness.py` — emit escalation on judge halt; periodic backlog poll.
- `fused-memory/src/fused_memory/middleware/task_interceptor.py` or `server/tools.py` — gate mutating tools.
- New helper: orchestrator detection (put in `shared/` if cross-cutting, or `fused-memory/src/fused_memory/services/orchestrator_detector.py`).
- Config: `fused-memory/config/config.yaml` + `fused-memory/src/fused_memory/config/schema.py` — surface the threshold keys.
- Tests.

## Tests to add
1. **Threshold check correctness**: seed the event_buffer with N buffered events, call `backlog_size_for`, expect N.
2. **Rejection path**: threshold exceeded + no orchestrator → mutating MCP tool returns `ReconciliationBacklogExceeded` error. Reads still work.
3. **Escalation path**: threshold exceeded + orchestrator detected → escalation file written with correct schema.
4. **Escalation rate-limiting**: trigger repeatedly in a short window → only one file written.
5. **Judge halt → escalation**: simulate halt → escalation written.
6. **Post-drain recovery**: seed high backlog → trigger policy → drain events → mutating tools accept again.

## Out of scope
- Building a drain tool (operator runs ad-hoc recovery).
- Auto-unhalt.
- Dashboards for backlog visibility (can follow).
- Watchdog implementation itself (WP-C).
- Cost-based prioritization.

## Acceptance
- [ ] Backlog exceeded + orchestrator running → one L1 escalation per rate-limit window.
- [ ] Backlog exceeded + no orchestrator → mutating MCP tools return structured error with actionable message.
- [ ] Judge halt → escalation (not just log line).
- [ ] All existing tests pass.
- [ ] New tests cover all branches.
- [ ] Operator-facing error messages are clear.

## Dependencies
- **WP-B merged** — backlog is meaningful only with reliable journalling.
- **WP-C merged** — watchdog integration relies on the watchdog surface.

## Workflow for this session

1. Read this briefing.
2. Search memory: "reconciliation halt", "backlog", "judge halt", "buffer_size_threshold", "orchestrator.lock" — capture prior context.
3. Read:
   - `fused-memory/src/fused_memory/reconciliation/harness.py`
   - `fused-memory/src/fused_memory/reconciliation/judge.py` (or wherever `_check_error_trends` lives)
   - `fused-memory/src/fused_memory/reconciliation/event_buffer.py`
   - `fused-memory/src/fused_memory/server/tools.py`
   - One sample escalation JSON from `/home/leo/src/reify/data/escalations/esc-1959-52.json` to learn the schema.
   - Orchestrator-lock implementation (grep `orchestrator.lock` in `orchestrator/` and `shared/`).
4. Write `plans/wp-d-plan.md`: threshold values, detection strategy, escalation schema alignment, rejection message text, rate-limit policy, test list.
5. Branch: `wp-d/backlog-escalation` off main.
6. Implement. Suggested chunking:
   - Commit 1: `backlog_size_for` + config keys
   - Commit 2: orchestrator detection helper
   - Commit 3: escalation writer (rate-limited)
   - Commit 4: rejection path at MCP boundary
   - Commit 5: judge halt → escalation
   - Commit 6: tests
7. Run `cd fused-memory && uv run pytest tests/ -x`.
8. Manual check: start the server locally against a test DB, seed high backlog, confirm rejection error appears.
9. `/reflect`: note the messaging you chose for the rejection error, any rate-limit tuning you made, whether the orchestrator detection feels robust, what should follow (e.g., a drain tool or dashboard).
10. Emit JSON summary:
    ```json
    {
      "wp": "D",
      "branch": "wp-d/backlog-escalation",
      "commits": ["<sha>", ...],
      "backlog_threshold": N,
      "rate_limit_mins": M,
      "detection_strategy": "lock_file|pgrep|registry",
      "tests_added": [...],
      "tests_passed": N,
      "reflection_saved": true,
      "notes": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Do NOT push, modify git config, merge to main, or restart the fused-memory systemd unit.
