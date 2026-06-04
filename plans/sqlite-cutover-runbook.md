# SQLite Task Backend Cutover Runbook

Phase 2 of `plans/do-1-on-a-happy-pony.md` is on main behind
`taskmaster.backend_mode='taskmaster'` (default). This runbook is the
sitting that flips it to `'sqlite'`. Plan on ~4–6 hours of attention
split across two soak windows.

---

## Pre-flight (5 min)

1. **Confirm orchestrators are stopped.** No running orchestrator process
   should be active for any project — the migration mutates
   `.taskmaster/tasks/tasks.json`'s sibling DB and a live orchestrator
   would race the swap.

   ```bash
   pgrep -af 'orchestrator' | head    # expect empty
   ```

   If anything is running: SIGTERM it and **wait** for graceful drain
   (memory: `feedback_graceful_shutdown_patience.md`). Do not kill the
   process group.

2. **Confirm fused-memory is up.** It still serves memory reads even with
   no orchestrator.

   ```bash
   systemctl --user status fused-memory.service | head
   ```

3. **Note current backend.** Should be `taskmaster` (the default).

   ```bash
   grep -A2 '^taskmaster:' /home/leo/src/dark-factory/fused-memory/config/config.yaml
   ```

   If `backend_mode` is not present, it defaults to `taskmaster` — that's
   fine.

---

## Step 1 — Migrate per-project tasks.json (5 min)

The migration script is idempotent + validating. It refuses to overwrite
without `--replace`, so first runs are safe.

Project roots come from the canonical list (memory:
`project_dashboard_known_project_roots_canonical.md`). As of
2026-05-01: `/home/leo/src/dark-factory`,
`/home/leo/src/autotrade`, `/home/leo/src/know-live`, plus any others in
the dashboard env list.

```bash
cd /home/leo/src/dark-factory/fused-memory
PYTHONPATH=$PWD/src:$PWD/../shared/src \
  /home/leo/src/dark-factory/.venv/bin/python \
  -m scripts.migrate_tasks_json_to_sqlite \
  /home/leo/src/dark-factory \
  /home/leo/src/autotrade \
  /home/leo/src/know-live \
  --verbose
```

Expected output (per project):

```
INFO OK <project_root> — tasks=N deps=K → <path>/tasks.db
INFO VALIDATE OK <project_root>
```

If any project reports `VALIDATE FAILED` — STOP. The DB doesn't match the
JSON. Inspect the logged first-mismatch entry and fix before flipping
anything.

If any project reports `REFUSE … tasks.db already exists` — that's a
prior cutover attempt. Either delete the stale `.db` (`.db-wal`, `.db-shm`
too) or pass `--replace` to overwrite.

---

## Step 2 — Flip fused-memory to dual_compare (taskmaster primary) (5 min)

Edit `fused-memory/config/config.yaml`. Add or change in the
`taskmaster:` section:

```yaml
taskmaster:
  backend_mode: dual_compare
  dual_compare_primary: taskmaster
  # … existing fields stay
```

Then restart fused-memory. The shared script handles drain + restart
cleanly (memory: `project_fused_memory_restart_procedure.md`).

```bash
/home/leo/src/dark-factory/scripts/restart-fused-memory.sh
```

Verify the backend wired up:

```bash
journalctl --user -u fused-memory.service -n 50 | grep -E 'Task backend|backend_mode'
```

Expect a line like `Task backend: DualCompareBackend (mode=dual_compare)`.

---

## Step 3 — Soak with orchestrator running (2–4 hours of active traffic)

Resume orchestrators on their normal cadence. Then watch for divergences.

**Live divergence monitor (run in a separate terminal):**

```bash
journalctl --user -u fused-memory.service -f | grep --line-buffered \
  -E 'dual_compare\.divergence|ERROR|TASKMASTER_UNAVAILABLE'
```

Each divergence line carries: method name, args, kwargs, and the two
sides' (truncated) responses. The wrapper deliberately strips volatile
fields (`updatedAt`, `metadata` JSON encoding, asymmetric id types)
before comparing — anything that fires is a real semantic disagreement.

**What to do on a divergence:**

1. Read the line — it tells you which method and which arguments.
2. Decide: is this a wire-shape gap in `SqliteTaskBackend` (fix the
   backend), or in the comparator's normalisers (fix
   `DualCompareBackend`)?
3. **Do not flip onward.** Quiesce, fix, redeploy fused-memory, restart
   the soak window from zero divergences.

**What "enough" looks like:**

- Zero `dual_compare.divergence` lines after ~2 hours of active
  orchestrator traffic, OR
- Counting principle: ~hundreds of write operations across every
  method (`set_task_status`, `add_task`, etc.). With the
  orchestrator's normal cadence this is ~1–2 hours.

A passive soak (orchestrator idle) doesn't count — there's nothing to
diverge on.

**Performance sanity:**

```bash
journalctl --user -u fused-memory.service --since '30 min ago' | \
  grep -E 'thread_monitor|memory_service' | tail
```

If thread count grows monotonically or memory balloons, that's a sqlite
backend leak (probably connection lifecycle). Bail.

---

## Step 4 — Cut to sqlite-only (5 min)

Once step 3 is clean, you can either:

- **(Conservative)** Flip `dual_compare_primary: sqlite`, restart, soak
  another 2 hours. This proves sqlite as the served-side response is
  what callers see (the comparator is now driven by sqlite and observes
  taskmaster's mirror).
- **(Aggressive — recommended)** Skip the inverse soak. The
  taskmaster-primary phase already proved sqlite agrees on every wire
  shape and operation; flipping primary doesn't add observability,
  it just changes which side is in the response path.

For the aggressive path, edit config:

```yaml
taskmaster:
  backend_mode: sqlite
  # dual_compare_primary becomes irrelevant
```

Restart fused-memory:

```bash
/home/leo/src/dark-factory/scripts/restart-fused-memory.sh
```

Verify:

```bash
journalctl --user -u fused-memory.service -n 30 | grep 'Task backend'
```

Expect `Task backend: SqliteTaskBackend (mode=sqlite)`.

The Taskmaster Node subprocess is no longer spawned. Confirm:

```bash
pgrep -af 'task-master\|mcp-server' | head    # expect empty
```

Resume normal orchestrator operations.

---

## Rollback

At any point before step 4, set `backend_mode: taskmaster` in config and
restart. The on-disk `tasks.json` was never touched by the SQLite path;
it is still authoritative. The SQLite DB just becomes a stale mirror you
can ignore (or `rm -f .taskmaster/tasks/tasks.db .db-wal .db-shm`).

After step 4: same rollback works as long as `tasks.json` hasn't drifted
too far. Note: `tasks.json` is no longer written under `sqlite` mode, so
its timestamp tells you when you cut over. If you need to roll back days
later, you'd need a "sqlite → tasks.json" reverse migration (not
written; would be another half-day if it ever comes up).

---

## Post-cutover follow-up (separate PR, not in this sitting)

Once `sqlite` has been live for a week with no escalations, retire the
Taskmaster code:

- Delete `fused-memory/src/fused_memory/backends/taskmaster_client.py`
  and `taskmaster_types.py`.
- Drop the `taskmaster-ai` Node submodule + `package.json` entry.
- Delete the supervisor-related tests.
- Drop `backend_mode` and `dual_compare_primary` from the config schema
  (they become single-valued).

---

## Reference

- Plan: `plans/do-1-on-a-happy-pony.md` §Cycle 2
- Backend: `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`
- Comparator: `fused-memory/src/fused_memory/backends/dual_compare_backend.py`
- Migration: `fused-memory/scripts/migrate_tasks_json_to_sqlite.py`
- Memory:
  - `project_fused_memory_restart_procedure.md` — restart script details
  - `feedback_graceful_shutdown_patience.md` — orchestrator drain
  - `project_dashboard_known_project_roots_canonical.md` — project list
