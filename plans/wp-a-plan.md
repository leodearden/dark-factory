# WP-A Recovery Plan (2026-04-18)

## Current forensic state (verified before plan)

- `fused-memory.service` active but wedged: every recon loop (autopilot_video, dark_factory, reify) returns `database is locked`.
- Subprocess under fused-memory cgroup: PID 2258925 `git show HEAD:.taskmaster/tasks/tasks.json` **has been running for 4d06h** — almost certainly the reason the SQLite writer lock has been held. Restart will kill it.
- `/home/leo/src/reify/.taskmaster/tasks/tasks.json`: 1950 tasks, max id 1955; metadata.taskCount=1947 (off by 3); 1953/1954/1955 identical ("Tighten handleSave errorSpy assertion…"), no deps. 1959 absent.
- `reconciliation.db.event_buffer`: reify=803 buffered, dark_factory=31, autopilot_video=5.
- No other task depends on 1953/1954/1955/1959 (scanned all tasks). Dedupe and 1959 restore need no dependency rewrites.
- Task 1959 original content preserved in `/home/leo/src/reify/data/escalations/esc-1959-52.json`.

## Step-by-step procedure

### Step 1 — Snapshot forensic state

Shell:
```
mkdir -p /home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/
# Reconciliation DB + wal/shm
cp -av /home/leo/src/dark-factory/data/reconciliation/reconciliation.db{,-wal,-shm} \
    /home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/
# write_journal for good measure
cp -av /home/leo/src/dark-factory/data/reconciliation/write_journal.db{,-wal,-shm} \
    /home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/
# reify tasks.json
cp -av /home/leo/src/reify/.taskmaster/tasks/tasks.json \
    /home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/reify-tasks-pre.json
# relevant escalations
cp -av /home/leo/src/reify/data/escalations/esc-1959-*.json \
       /home/leo/src/reify/data/escalations/esc-872-216.json \
       /home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/
```

**Checkpoint:** `ls -la data/backups/2026-04-18-wp-a/` shows all expected files.

### Step 2 — Export buffered reify events

Use external sqlite3 reader (lock holder only affects writers):

```python
# data/backups/2026-04-18-wp-a/reify-buffered-events.jsonl
import sqlite3, json
conn = sqlite3.connect('file:/home/leo/src/dark-factory/data/reconciliation/reconciliation.db?mode=ro', uri=True, timeout=10)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM event_buffer WHERE project_id='reify' AND status='buffered' ORDER BY id").fetchall()
with open('/home/leo/src/dark-factory/data/backups/2026-04-18-wp-a/reify-buffered-events.jsonl','w') as f:
    for r in rows:
        f.write(json.dumps({k: r[k] for k in r.keys()}, default=str) + "\n")
print(len(rows))
```

**Checkpoint:** file line-count equals 803.

### Step 3 — Restart fused-memory (drain)

```
bash /home/leo/src/dark-factory/scripts/restart-fused-memory.sh --drain
```

Notes:
- The stuck `git show` subprocess (PID 2258925, 4+ days old) is a child of the fused-memory python process. Systemd cgroup restart should cascade-kill it. If it does not, we kill that specific PID only.
- If drain times out (120s), the script proceeds with restart anyway — that is acceptable here because we have the backup.

**Checkpoint:**
- `systemctl --user status fused-memory.service` → active (running), fresh start time.
- No child `git show` subprocess present (`ps --ppid <new python pid>`).
- `journalctl --user -u fused-memory -n 50 --no-pager` → startup log, no `database is locked` in the last 30s.
- `mcp__fused-memory__get_status` returns healthy shape (graphiti/mem0/taskmaster fields populated, no exception).

### Step 4 — Restore task 1959 content as a NEW task

Call:
```
mcp__fused-memory__add_task(
  project_root="/home/leo/src/reify",
  title="Wire DesignTree panel into App.tsx so multi-selection is user-reachable",
  description=<full description from esc-1959-52.json>,
  details=<full details from esc-1959-52.json — includes prerequisite spec>,
  priority="high",
  metadata={"recovery": "wp-a-2026-04-18", "original_id_was": "1959", "source_escalation": "esc-1959-52"}
)
```

Record the new id. The prerequisite-task spec is included inside the `details` field as documentation; we are not creating a separate prerequisite task in this WP — recovery scope is restore-the-content, not re-plan the work.

**Checkpoint:** `mcp__fused-memory__get_task(id=<new_id>)` returns the restored content. Record the id.

### Step 5 — Dedupe 1953 / 1954 / 1955

Pre-check confirmed no other task depends on these ids. So:
```
mcp__fused-memory__remove_task(id="1954", project_root="/home/leo/src/reify")
mcp__fused-memory__remove_task(id="1955", project_root="/home/leo/src/reify")
```

Keep 1953 as survivor.

**Checkpoint:** `get_tasks` shows only id 1953 for the duplicate title; ids 1954 and 1955 absent.

### Step 6 — Fix metadata counters (taskCount desync)

After Step 5 the task count should be 1950 − 2 (deleted 1954, 1955) + 1 (new 1959 replacement) = **1949**. metadata.taskCount currently = 1947. Target = 1949.

Attempt a self-correction first: the MCP writer should have rewritten metadata during add_task/remove_task. Re-read tasks.json and compare.

If still desynced, run this one-off recompute script **with fused-memory idle** (no orchestrator running; no MCP writes in-flight):

```python
# recompute-reify-metadata.py (one-off recovery — NOT to be committed as a recurring tool)
import json, os, shutil, datetime, tempfile
path = '/home/leo/src/reify/.taskmaster/tasks/tasks.json'
with open(path) as f:
    data = json.load(f)
master = data['master']
tasks = master['tasks']
master['metadata']['taskCount'] = len(tasks)
master['metadata']['completedCount'] = sum(1 for t in tasks if t.get('status') == 'done')
master['metadata']['updated'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
fd, tmp = tempfile.mkstemp(prefix='tasks-', suffix='.json', dir=os.path.dirname(path))
with os.fdopen(fd, 'w') as f:
    json.dump(data, f, indent=2)
os.replace(tmp, path)
print('taskCount', master['metadata']['taskCount'], 'completedCount', master['metadata']['completedCount'])
```

**Checkpoint:** `metadata.taskCount == len(tasks) == metadata.completedCount + remaining`.

### Step 7 — Unhalt reify reconciliation

```
mcp__fused-memory__unhalt_reconciliation(project_id="reify")
```

Observe drain over ~5 minutes:
- External sqlite reader: `SELECT COUNT(*) FROM event_buffer WHERE project_id='reify' AND status='buffered'` trends down.
- `reconciliation.db-wal` size stays bounded.
- `journalctl --user -u fused-memory -f` shows reify recon runs completing (no `database is locked`).

If drain stalls >5 min: note in summary, do not try to fix (WP-C).

### Step 8 — Validate final state

- `get_tasks(project_root="/home/leo/src/reify")` count == metadata.taskCount == actual len(master.tasks).
- Spot-check ids 1940..1955 and new id: no duplicate titles, statuses plausible.
- New-id task has restored 1959 content; metadata carries `recovery: wp-a-2026-04-18`.
- `get_status` healthy for graphiti, mem0, taskmaster.
- Last 5 min of fused-memory journal shows no `database is locked`.

### Step 9 — Session close

1. `/reflect` to write learnings to fused-memory. If it errors, fallback to `plans/wp-a-reflection.md`.
2. Emit final JSON summary per briefing schema.

## Risk register

| Risk | Mitigation |
|------|------------|
| Restart kills an in-flight write, corrupts tasks.json | Backup taken in Step 1; writer is already wedged for 16h+, no productive writes in progress. |
| Stuck `git show` subprocess survives restart | Systemd restart kills the whole cgroup. If orphaned, kill single PID by number. |
| `add_task` again corrupts into a reused slot (repro of esc-1959-52 bug) | Bug only manifested under DB-lock storm. Post-restart, writer is healthy. Verify new id is distinct and no existing task is mutated afterwards. |
| `unhalt_reconciliation` stage drain conflicts with live writes | Only one orchestrator affects reify, and it is stopped per briefing. |
| Metadata recompute races Taskmaster writer | Only run script if self-correction fails; verify no recent mtime on tasks.json and no active MCP write calls in-flight. |

## Non-goals (out of scope for WP-A)

- Fixing the add_task-during-lock corruption bug (WP-B+).
- Creating the entity-tree-bridge prerequisite task (waits for 1959 owner).
- Clearing dark_factory/autopilot_video buffered events (not blocking; smaller backlogs).
- Restarting reify orchestrator.
