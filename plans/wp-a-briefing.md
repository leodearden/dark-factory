# WP-A: Incident Recovery

## Mission
Restore reify's `tasks.json` to a consistent state, drain the stuck reconciliation backlog, and get fused-memory healthy again — so subsequent WPs (B..G) have a sound baseline to fix against.

## Defect context (2026-04-17)
- `/home/leo/src/reify/.taskmaster/tasks/tasks.json` has three identical duplicate tasks at ids **1953, 1954, 1955**: title *"Tighten handleSave errorSpy assertion to match sibling-test pattern"*, no `updatedAt`.
- Task **1959** is missing. A steward reported its original content (a DesignTree App.tsx wiring task) was silently overwritten by a different `add_task` payload and subsequently disappeared entirely. Original content is preserved in the escalation record — see `Files` below.
- `metadata.taskCount = 1947` but the actual task count is 1950 — counter desync.
- `/home/leo/src/reify/.worktrees/1959/.taskmaster/tasks/tasks.json` is a stale, divergent copy (string task IDs, max numeric id 1952). Do NOT use it as source of truth.
- **803 reify events are buffered** in `/home/leo/src/dark-factory/data/reconciliation/reconciliation.db` (`SELECT COUNT(*) FROM event_buffer WHERE status='buffered' AND project_id='reify'`).
- fused-memory has held the SQLite writer lock for 16h+. Every `add_task` / `set_task_status` / `update_task` returns `{"error": "database is locked", "error_type": "OperationalError"}`.
- Reify reconciliation was **halted by the judge circuit-breaker** earlier in the week and never unhalted.
- The orchestrator that caused the corruption (PID 1480578, pointed at reify) is already stopped.

This is the first of seven work packages (WP-A..G) to sweep and fix the defects that led to this corruption. **Do not attempt any fix beyond recovery** — the code-change WPs (B..G) will follow in separate sessions.

## Approach

### Step 1 — Snapshot forensic state
Copy everything potentially relevant to a timestamped backup directory before touching anything:
- `/home/leo/src/dark-factory/data/reconciliation/reconciliation.db` (and `-wal`, `-shm`)
- `/home/leo/src/reify/.taskmaster/tasks/tasks.json`
- All `/home/leo/src/reify/data/escalations/esc-1959-*.json` and `esc-872-216.json`

Target: `/home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/`.

### Step 2 — Export buffered reconciliation events
Before restart, export the 803 buffered reify events from `reconciliation.db` to `data/backups/2026-04-18-wp-a/reify-buffered-events.jsonl`. Use an external `sqlite3` client or a Python script with sqlite3 stdlib — the fused-memory process holds the writer lock, but external readers work fine.

Query: `SELECT * FROM event_buffer WHERE project_id='reify' AND status='buffered'`.

### Step 3 — Restart fused-memory
Use the project's documented procedure: `scripts/restart-fused-memory.sh --drain`.

Verify the service comes up healthy:
- `systemctl --user status fused-memory.service` → active (running)
- A quick MCP healthcheck. You can use the `fused-memory` MCP tools directly in your session — they should succeed within seconds.
- Inspect the last few log lines via `journalctl --user -u fused-memory.service -n 30` — no "database is locked" errors in the new instance.

### Step 4 — Restore task 1959
Read the original task content from `/home/leo/src/reify/data/escalations/esc-1959-52.json` — the `detail` field contains a section titled *"Original content of task 1959 (to restore)"* with the exact title, description, and details.

Create the restored task via `mcp__fused-memory__add_task` with `project_root=/home/leo/src/reify`. The new task will get a fresh ID (probably 1956 or 1960 depending on Taskmaster's behavior post-dedupe) — **do not force it to be id=1959**. Record the assigned ID for the summary.

Include a metadata field marking this as a recovery: `metadata={"recovery": "wp-a-2026-04-18", "original_id_was": "1959"}`.

### Step 5 — Dedupe 1953/1954/1955
1. Fetch current tasks via `mcp__fused-memory__get_tasks`.
2. Check whether any other task has 1953, 1954, or 1955 in its dependencies. Build a dependents map.
3. Keep task 1953 as the survivor (lowest id). For 1954 and 1955: if any task depends on them, rewrite those dependencies to point at 1953 first (`add_dependency` + `remove_dependency`).
4. Remove 1954 and 1955 via `mcp__fused-memory__remove_task`.

### Step 6 — Fix metadata counters
Try this path first: call a benign taskmaster metadata-touching op (e.g., `set_task_status` on any task to the same status — the interceptor may short-circuit; then to a different status and back). Check whether `metadata.taskCount` now matches actual count.

If metadata does not self-correct, perform a one-off recompute script:
- Read tasks.json.
- Set `metadata.taskCount = len(master.tasks)` and `metadata.completedCount = sum(1 for t in master.tasks if t.status == 'done')`.
- Update `metadata.updated` to current ISO timestamp.
- Write back atomically (temp file + rename).
- **Only do this with fused-memory idle** (no active MCP write calls in-flight) to avoid racing with Taskmaster's writer.
- Document this script in `plans/wp-a-plan.md` and note it as one-off recovery, not a recurring pattern.

### Step 7 — Unhalt reify reconciliation
Call `mcp__fused-memory__unhalt_reconciliation` with the reify project identifier.

Watch the drain begin:
- `reconciliation.db-wal` should not grow indefinitely
- `SELECT COUNT(*) FROM event_buffer WHERE project_id='reify' AND status='buffered'` should decrease over a few minutes
- journalctl should show drain activity, no new lock errors

If drain stalls (count doesn't move for 5+ minutes), that's a WP-C concern — note it in the summary but don't try to fix it.

### Step 8 — Validate final state
- `get_tasks` for reify → count matches `metadata.taskCount` and actual master.tasks length.
- Spot-check tasks 1940-1960: all present, no content duplicates, statuses plausible.
- Task 1959's original content is now at the new ID you created.
- fused-memory `get_status` returns healthy for Graphiti, Mem0, Taskmaster.
- No `database is locked` entries in the last 5 minutes of fused-memory logs.

## Files / entry points
- Escalation with original 1959 content: `/home/leo/src/reify/data/escalations/esc-1959-52.json`
- Restart script: `/home/leo/src/dark-factory/scripts/restart-fused-memory.sh` (use `--drain` flag)
- Databases: `/home/leo/src/dark-factory/data/reconciliation/reconciliation.db` (+ wal/shm)
- Target tasks.json: `/home/leo/src/reify/.taskmaster/tasks/tasks.json`
- Backup dir to create: `/home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/`
- Project memory: search for "SQLite lock requires process restart", "restart-fused-memory", "reify orphan" via `mcp__fused-memory__search` if you want more context.

## Constraints
- **All tasks.json mutations go through fused-memory MCP**, never direct text edits. The one exception is the one-off metadata-recompute script in Step 6, and only with fused-memory idle.
- **Do not delete reconciliation.db or any live state** without first snapshotting to the backup dir.
- **Do not modify fused-memory or taskmaster-ai source code** — that is WP-B through WP-G's job.
- Do not `git push`, do not modify git config, do not `git reset --hard`, do not `git clean -f`.
- Do not start any orchestrator — reify orchestrator stays stopped until WP-G lands.

## Acceptance
- [ ] `tasks.json` valid: `metadata.taskCount` matches actual count; no duplicate-content tasks in 1940-1960; task 1959's original content present at some ID.
- [ ] fused-memory service running; no `database is locked` errors in the last 5 min of logs.
- [ ] Reify `event_buffer` buffered count ≈ 0, or clear drain progress (log it either way).
- [ ] All forensic backups saved to `data/backups/2026-04-18-wp-a/`.

## Dependencies
None. First WP.

## Workflow for this session

1. Read this briefing in full.
2. Write a detailed step-by-step recovery plan at `plans/wp-a-plan.md`: exact shell commands, MCP tool calls with their argument shapes, and checkpoint-verify criteria between major steps.
3. If you need to add scripts to the repo, create branch `wp-a/recovery` off main. If your work is purely operational (no new source files), no branch is needed.
4. Execute the plan. At each checkpoint, verify state before proceeding.
5. If a step fails in a way you don't understand, **stop**. Write `plans/wp-a-incident.md` describing what you saw and your hypothesis. Do not improvise.
6. At the end, run `/reflect` to save session learnings to fused-memory. If `/reflect` errors (e.g., fused-memory still sick), save to `plans/wp-a-reflection.md` as fallback.
7. Emit a final JSON summary to stdout in this exact shape:
   ```json
   {
     "wp": "A",
     "status": "success|partial|failed",
     "new_1959_task_id": "<id or null>",
     "tasks_removed": [1954, 1955],
     "service_healthy": true,
     "buffered_events_remaining": 0,
     "backups_at": "data/backups/2026-04-18-wp-a/",
     "notes": "...",
     "reflection_saved": true
   }
   ```

## Permission mode
This session runs with `--dangerously-skip-permissions`. You may freely read, edit, and run commands. You may restart the fused-memory systemd user service (this WP explicitly requires it). Do NOT: push to remote, modify git config, run `git reset --hard` / `git clean -f` / `git push --force`, start orchestrators.
