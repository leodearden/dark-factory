# WP-D (tight): Backlog escalation policy

Previous WP-D session hit "Prompt is too long" during exploratory research. This is a tightened restart: specific files, specific patterns, write sooner.

## Goal
Bound the reconciliation backlog. When exceeded, escalate or reject. No silent growth.

## Policy (already decided — do not redesign)
- Compute `db_backlog[project_id] = COUNT(event_buffer WHERE status='buffered' AND project_id=P)` + `queue_depth + retry_in_flight` (from EventQueue.stats()).
- Threshold: new config key `reconciliation.backlog_hard_limit` default **500**.
- When `db_backlog > limit`:
  - If orchestrator is live for that project → write L1 escalation to `<project_root>/data/escalations/esc-reconciliation-backlog-<timestamp>.json` (schema below). Rate-limit: once per project per 15 min.
  - Else → mutating MCP tools (`add_task`, `set_task_status`, `update_task`, `add_subtask`, `remove_task`, `add_dependency`, `remove_dependency`, `add_memory`, `add_episode`) return `{"error": "ReconciliationBacklogExceeded: backlog N > limit M for project P; drain before retrying.", "error_type": "ReconciliationBacklogExceeded", "backlog": N, "threshold": M, "project_id": "P"}`. Reads unaffected.
- When judge halts a project → same escalation-or-log policy.
- When SqliteWatchdog fires wedge (via its `wedge_callback`) → same policy, different error shape (`"error_type": "SqliteDrainerWedged"`).

## Files to touch (exact list — don't explore broadly)

Write:
1. `fused-memory/src/fused_memory/reconciliation/backlog_policy.py` — NEW. Contains:
   - `class BacklogPolicy` holding: `event_buffer`, `event_queue`, `orchestrator_detector`, `thresholds`, and a last-escalation-per-project timestamp dict for rate limiting.
   - `async def check(project_id) -> BacklogVerdict` — returns one of `ok`, `rejection(reason, backlog, threshold)`, `escalated(path)`.
   - `async def on_watchdog_wedge(payload: dict)` — called from the SqliteWatchdog's wedge_callback; routes to rejection or escalation.
   - `async def on_judge_halt(project_id: str, reason: str)` — called from the harness when the judge halts.
2. `fused-memory/src/fused_memory/services/orchestrator_detector.py` — NEW. `def is_orchestrator_live_for(project_root: str) -> bool`. Reads `<project_root>/data/orchestrator/orchestrator.lock` if present, parses PID, checks `os.kill(pid, 0)` for liveness. Stale lock → not live.
3. `fused-memory/tests/test_backlog_policy.py` — NEW. Tests listed at bottom.

Edit (small changes only):
4. `fused-memory/src/fused_memory/config/schema.py` — add to `ReconciliationConfig`:
   - `backlog_hard_limit: int = Field(default=500)`
   - `backlog_escalation_rate_limit_seconds: float = Field(default=900.0)`  # 15 min
5. `fused-memory/src/fused_memory/server/main.py` — instantiate `BacklogPolicy`, pass it to TaskInterceptor and to the memory_service, wire as SqliteWatchdog's `wedge_callback`.
6. `fused-memory/src/fused_memory/middleware/task_interceptor.py` — each mutating public method calls `await self._backlog_policy.check(project_id)` before the existing lock acquisition; if it returns a rejection verdict, return the structured error dict without mutating.
7. `fused-memory/src/fused_memory/server/tools.py` — for `add_memory` and `add_episode` (non-task mutations), do the same `check()` + early-return-error pattern. Leave reads untouched.
8. `fused-memory/src/fused_memory/reconciliation/harness.py` — when the judge halts a project, call `backlog_policy.on_judge_halt(project_id, reason)`.

## Escalation JSON shape
```json
{
  "id": "esc-reconciliation-backlog-<iso8601>",
  "task_id": null,
  "agent_role": "fused-memory",
  "severity": "blocking",
  "category": "infra_issue",
  "summary": "Reconciliation backlog exceeded for <project>: <N>/<limit>",
  "detail": "<human-readable description + suggested action>",
  "suggested_action": "drain_reconciliation",
  "timestamp": "<iso8601 with tz>",
  "status": "pending",
  "level": 1,
  "workflow_state": "infra",
  "backlog": N,
  "threshold": M,
  "project_id": "P"
}
```
Match shape by reading one existing escalation for field ordering: `/home/leo/src/reify/data/escalations/esc-1959-52.json` (opened only if needed; the fields above are enough).

## Tests for test_backlog_policy.py (cover these)
1. `test_ok_verdict_when_under_threshold` — policy returns ok.
2. `test_rejection_verdict_when_over_threshold_and_no_orchestrator` — returns structured rejection.
3. `test_escalation_when_over_threshold_and_orchestrator_live` — writes escalation JSON; detector returns True.
4. `test_rate_limit_prevents_spam` — trigger twice in rate-limit window → only one file.
5. `test_rate_limit_allows_after_window` — trigger, advance clock, trigger again → two files.
6. `test_on_judge_halt_writes_escalation` — simulated halt writes escalation.
7. `test_on_watchdog_wedge_writes_escalation_with_wedge_error_type` — payload routed, error_type correct.
8. `test_orchestrator_detector_stale_lock_pid_dead` — returns False.
9. `test_orchestrator_detector_live_pid` — returns True.
10. `test_orchestrator_detector_no_lock_file` — returns False.
11. `test_task_interceptor_add_task_rejects_when_over_limit` — wire the policy into interceptor, fire add_task → returns error dict, no tasks.json mutation, no event enqueued.
12. `test_task_interceptor_add_task_ok_when_under_limit` — normal path.

Use `fused-memory/tests/test_task_interceptor.py` as a template for interceptor-level test structure. Use `fused-memory/tests/test_sqlite_watchdog.py` as a template for policy-level test structure with mock stats.

## Notes on integration
- `EventQueue.stats()` signature already provides `queue_depth`, `retry_in_flight`, `last_commit_ts`, `events_committed`, `overflow_drops`. See `fused-memory/src/fused_memory/reconciliation/event_queue.py`.
- `SqliteWatchdog` already accepts a `wedge_callback` parameter. See its `__init__` in `fused-memory/src/fused_memory/reconciliation/sqlite_watchdog.py`.
- 149 reify events currently buffered in the real DB — use a test DB or mock EventBuffer in tests.
- orchestrator.lock is at `<project_root>/data/orchestrator/orchestrator.lock` — first line is usually `PID N started <ts>`. A stale PID from Apr 13 (349526) was observed.

## Workflow
1. Create branch `wp-d/backlog-escalation` off main.
2. Write backlog_policy.py + orchestrator_detector.py with their tests FIRST (tests visible to drive design).
3. Run `cd fused-memory && uv run pytest tests/test_backlog_policy.py -x -q`. Fix until green.
4. Wire into server/main.py + task_interceptor.py + server/tools.py + harness.py.
5. Run `cd fused-memory && uv run pytest tests/test_backlog_policy.py tests/test_task_interceptor.py tests/test_sqlite_watchdog.py tests/test_event_queue.py tests/test_server_shutdown.py -x -q`. All green.
6. Commit in 2-3 logical chunks.
7. `/reflect` at end.
8. Emit JSON summary:
```json
{"wp": "D", "branch": "wp-d/backlog-escalation", "commits": [...], "tests_added": [...], "tests_passed": N, "reflection_saved": true, "notes": "..."}
```

## Constraints
- Don't explore beyond the files listed above. Grep only if you genuinely need to find a call site.
- Do NOT push, merge, restart fused-memory systemd, or modify taskmaster-ai / memory_consolidator.py / usage-accounts.yaml.
- `--dangerously-skip-permissions` enabled.
