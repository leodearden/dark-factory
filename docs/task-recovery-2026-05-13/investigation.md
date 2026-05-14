# Task DB loss — investigation (2026-05-13)

Forensic write-up of the silent loss of ~150 reify task rows during the
fused-memory watchdog SIGABRT + restart on 2026-05-13 14:25 BST. Shared
baseline for the hardening, audit, and recovery work-streams.

## TL;DR

- Prior `fused-memory.service` instance (PID 1009821) was watchdog-killed
  with SIGABRT at 2026-05-13 14:25:28 BST after 71h 45min uptime.
- New instance opened `/home/leo/src/reify/.taskmaster/tasks/tasks.db` at
  14:25:46 BST. ~3 days of writes that lived only in the WAL did not survive
  the restart. Effective DB state rolled back to its 2026-05-11 01:45:16
  snapshot.
- Lost rows include the ComputeNode contract DAG (3491-3502), multi-kernel-
  phase-3 DAG (3526-3542), GR-024 buckling-eigensolver DAG (3576-3588), and
  ~6 single-shot curator tasks (3575, 3578, 3579, 3580, 3581, 3584). Plus
  ~80 more in the gaps. Plus reverted status transitions for 3379, 3383,
  3384 and many "done" flips.
- The data is unrecoverable from `tasks.db` and `tasks.db-wal` — the bytes
  for the lost task titles are physically absent or salt-abandoned.
- Two failures combined: (1) auto-checkpoint stalled for 60+ hours so the
  main DB file was never advanced from the 2026-05-11 01:45:16 snapshot,
  (2) WAL recovery on the new instance discarded the un-checkpointed frames
  rather than replaying them.

## Timeline

| Time (BST) | Event |
|---|---|
| 2026-05-10 14:40:25 | `fused-memory.service` started — prior instance, PID 1009821 |
| 2026-05-11 01:45:16 | `tasks.db` and `tasks.db-wal` Birth time. Last main-DB file write before the loss event. |
| 2026-05-12 ~21:15 UTC | Curator creates tasks 3575, 3578, 3579, 3580, 3581, 3584 (tickets DB confirms `status=created` with those `task_id`s) |
| 2026-05-12 22:27 BST | Commit `63bf59fe24` lands `phase-3-eight-dag-filing-log.md` (tasks 3491-3502) |
| 2026-05-13 ~10:01 UTC | Orchestrator escalation `esc-3491-165` filed against task 3491 (HTTP 429 during planning) — task 3491 demonstrably exists |
| 2026-05-13 11:44 BST | `gr024-buckling-eigensolver-filing-log.md` written (tasks 3576-3588) |
| 2026-05-13 14:20:11 | Orchestrator log: `Task 3491 acquired locks`, `Starting workflow for task 3491: ComputeNode α: CacheStore::pending_cause admit NodeId::Compute(_) as chain root`. Same line for 3582 at 14:20:34. |
| 2026-05-13 14:24:50 | Last fused-memory heartbeat (`thread_monitor: threads=44 delta=-1`) |
| 2026-05-13 14:25:28 | `systemd[5733]: fused-memory.service: Watchdog timeout (limit 30s)!` → SIGABRT to PIDs 1009584 (uv) and 1009821 (python3) |
| 2026-05-13 14:25:30 | `Main process exited, code=dumped, status=6/ABRT` |
| 2026-05-13 14:25:46 | New instance started, PID 3941020. `SqliteTaskBackend opened /home/leo/src/reify/.taskmaster/tasks/tasks.db` |
| 2026-05-13 16:11:49 | Orchestrator: `set_task_status('3582') rejected: TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 3582` — first proof of post-restart loss |
| 2026-05-13 16:16:52 | Same error for task 3491 |
| 2026-05-13 18:32:44 UTC | Curator creates ticket `tkt_0RNWSTBGSNTD4KF3HE3FNSM399` → resolves as `created task_id=3415`. The MAX(id)+1 allocator has rolled back from 3502+ to 3414. |
| 2026-05-13 ~20:50 BST | User notices "tasks 3491-3502 don't exist" and starts /deb |

## Evidence

### Live DB state

```
$ python3 -c "
import sqlite3
con = sqlite3.connect('/home/leo/src/reify/.taskmaster/tasks/tasks.db')
cur = con.execute('SELECT count(*), max(CAST(id AS INTEGER)) FROM tasks')
print(cur.fetchone())"
(3389, 3419)
```

Spot-checks for the supposedly-filed tasks all return zero rows:
```
ids = [3491,3492,...,3502, 3505, 3525,3526,3527, 3546,3547,3549, 3564,3565,
       3574, 3576,3577,...,3588]
# Of 35 expected: 0 present, 35 absent
```

Tasks 3417, 3418, 3419 are the curator's first post-restart allocations
(spawned from 3452 / 3454 follow-ups — note 3452/3454 themselves are also
gone). The ID allocator (`SELECT MAX(id)+1`) returned 3415 first, confirming
`MAX(id)=3414` at restart.

### File mtimes — the smoking gun

```
$ stat /home/leo/src/reify/.taskmaster/tasks/tasks.db
  Birth:  2026-05-11 01:45:16
  Modify: 2026-05-11 01:45:16   (unchanged for 60+ hours, until /deb session)
  Size:   7421952
```

Main DB was never written to between 2026-05-11 01:45:16 and the user's /deb
session at 2026-05-13 20:52 (which triggered a passive checkpoint). With
`wal_autocheckpoint=1000` (SQLite default — never overridden), normal
operation should hit checkpoints multiple times per day. **Auto-checkpoint
made zero progress for 60+ hours.**

DB header `change_counter` is 15 — the file has only ever been rewritten
~15 times in its lifetime.

### WAL forensics

`tasks.db-wal` post-restart held 1002 frames across three salt generations:

| Salt | Frame range | Count | Status |
|---|---|---|---|
| `0xa441c666 / 0x6be780d5` | 0..24 | 25 | Current valid (post-restart writes only) |
| `0xa441c665 / 0x91f3d645` | 25..624 | 600 | Abandoned (pre-restart, salt-mismatched) |
| `0xa441c664 / 0xe4bc5b4b` | 625..1001 | 377 | Abandoned (older pre-restart) |

WAL header `checkpoint_seq=2`, salt matches generation c666. Frames in c664
and c665 have valid checksum chains within their generation but are
discarded by SQLite because their salt no longer matches the header.

Comparison of WAL frame 627 (page 1789, salt c664) vs main DB page 1789:
**byte-identical**. So salt-abandoned frames *do* contain pre-restart data
that was at one point flushed to main DB. That means the c664→c665 and
c665→c666 salt bumps each represent a successful WAL TRUNCATE/RESTART
checkpoint where pages reached main DB.

But: a binary search of all 1002 WAL frames for unique titles from the
filing logs (`pending_cause admit NodeId`, `shift-invert Lanczos`,
`MultiCaseBucklingResult`, `solve_buckling_kernel`, `BucklingPanel`,
`@optimized lowering wire`, `cost_per_byte LRU comparator`, etc.) returned
**zero hits**. Same titles in the main DB file: also zero hits. The data
for the lost rows is not in either file in any form.

So the data path was: created in-process → WAL-committed → ??? → gone. The
"???" step is where the loss occurred.

### Pertinent code

`SqliteTaskBackend._get_connection` — the only place WAL pragmas are set:

```python
# fused-memory/src/fused_memory/backends/sqlite_task_backend.py:316-321
await conn.execute('PRAGMA journal_mode=WAL')
await conn.execute('PRAGMA busy_timeout=5000')
await conn.execute('PRAGMA synchronous=NORMAL')
await conn.execute('PRAGMA foreign_keys=OFF')
await conn.executescript(_SCHEMA_SQL)
await conn.commit()
```

Notable absences: `wal_autocheckpoint`, `journal_size_limit`, any explicit
`wal_checkpoint(...)` call. The schema is all `CREATE … IF NOT EXISTS`, so
the executescript is non-destructive. Connections are kept open for the
entire process lifetime (cached in `self._connections[project_root]`).

`_txn` wrapper at `:329-353` shields commit/rollback against asyncio
cancellation; this is fine and not implicated.

`add_task` (`:494-557`) allocates IDs as `SELECT COALESCE(MAX(id), 0) + 1`
inside a per-project write lock — explains the rollback to 3415 cleanly.

### Systemd watchdog config

`scripts/fused-memory.service.template`:
```
Type=notify
NotifyAccess=all
WatchdogSec=30
TimeoutStopSec=60
TimeoutStartSec=300
```

Heartbeat death at 14:24:50 → SIGABRT at 14:25:28 = exactly the 30s
WatchdogSec window. The reconciliation pipeline had paused ~14:24:11 with
"All accounts capped" — heartbeat may have been blocked by the same
asyncio event the cap-wait was sleeping on.

### Cross-references that prove the lost rows existed

- `journalctl --user -u fused-memory --since "2026-05-13 14:20"` shows the
  scheduler interacting with task IDs that are now absent.
- `/home/leo/src/reify/logs/orch-2026-05-13.log` line at 14:20:11 names
  task 3491 by full title; the 16:11:49 line shows `set_task_status('3582')
  rejected: No tasks found for ID(s): 3582` after restart.
- `/home/leo/src/dark-factory/data/reconciliation/tickets.db`:
  ```sql
  SELECT ticket_id, status, task_id, created_at, resolved_at FROM tickets
  WHERE project_id='reify' AND created_at >= '2026-05-12T00:00';
  ```
  Six tickets resolved as `created` with `task_id` in {3575, 3578, 3579,
  3580, 3581, 3584}. Two tickets in the same window resolved as `combined`
  into earlier tasks. None of the `created` task_ids exist in the live DB.
- `/home/leo/src/reify/data/escalations/archive/2026-05-13/esc-3491-165.json`
  records orchestrator working in `/home/leo/src/reify/.worktrees/3491` at
  10:11 BST — the worktree, the branch, and the task all existed then.
- Worktrees with task IDs the live DB doesn't have (created 2026-05-13
  14:20-21 BST, just before the kill):
  ```
  /home/leo/src/reify/.worktrees/{3491,3505,3525,3526,3527,3546,3547,3549,
                                  3564,3565,3574,3582,3593,3598}
  ```
- Tasks the filing log marked cancelled on May 12 (`3379`, `3383`, `3384`)
  are now back to pending/in-progress — those status transitions also lost.

### Surviving filing-log artifacts (recovery inputs)

Tracked:
- `/home/leo/src/reify/docs/architecture-audit/phase-3-eight-dag-filing-log.md`
  — full title table, prereqs, intra-DAG dep edges, supersession provenance
  for 3491-3502. Cross-PRD edge κ=3500→2945, 2946 retroactive.
- `/home/leo/src/reify/docs/architecture-audit/gr020-multi-kernel-phase-3-filing-log.md`
  — for 3526-3542.

Untracked but on disk:
- `/home/leo/src/reify/docs/architecture-audit/gr024-buckling-eigensolver-filing-log.md`
  — for 3576-3588.

Curator candidate JSON for the 6 single-shot rows is intact in
`tickets.db.candidate_json` and can be replayed verbatim via `submit_task`
(per `[[procedural_recover_expired_tickets]]`).

The gap-register at
`/home/leo/src/reify/docs/architecture-audit/gap-register.md` references
the lost IDs in disposition prose — needs an old→new ID rewrite after
re-filing.

## Root cause analysis

### Proximate cause
Systemd watchdog SIGABRT-killed `fused-memory.service` at 14:25:28 May 13.
The new instance opened the WAL but did not recover ~3 days of writes that
had been WAL-committed but never durably checkpointed to the main DB file.

### Root causes (in order of confidence)

**1. Auto-checkpoint silently stalled for ~60 hours.** *(Very high.)* The
main DB file mtime stayed pinned at the WAL Birth time despite continuous
write activity. Default `wal_autocheckpoint=1000` should have flushed pages
to disk many times per day; it did not. Most likely a long-held reader
snapshot (concurrent `get_tasks` from orchestrator + steward + reconciliation
sharing one persistent connection per project) was permanently blocking
PASSIVE checkpoints from advancing the readmark. Without checkpoint
progress, the WAL grew but the main DB file never advanced.

**2. WAL recovery on restart did not apply the in-WAL committed frames.**
*(High.)* Empirically: at restart the WAL had committed frames for ~150
task creations across multiple salt generations. After restart, the new
instance's `MAX(id)` returned 3414 (the May 11 01:45 snapshot value). The
new instance's `add_task` then started allocating from 3415 forward, and
the salt generations containing the lost data are gone from `tasks.db-wal`
without their content reaching `tasks.db`. Most likely path: SHM index
state corrupted by SIGABRT mid-write → SQLite recovery reset the WAL salt
rather than replaying the prior frames, and the abandoned frames were
overwritten as the new instance reused the WAL file.

**3. `synchronous=NORMAL` magnifies blast radius without per-commit fsync.**
*(Contributing factor, not root.)* In WAL mode with `synchronous=NORMAL`,
SQLite skips fsync on each commit, syncing only at checkpoint time. With
checkpoints not progressing (cause #1), nothing is fsynced. A SIGABRT alone
should not lose page-cache pages (kernel survives), but combined with #2
the result is total loss of the un-checkpointed window.

### User-raised hypotheses — verdicts

1. **planning_mode tasks stranded as deferred.** Rejected. `planning_mode=
   True` (tools.py:1731 → task_interceptor:_submit_task_planning_mode) is
   synchronous; it returns the real task_id only after `tm.add_task`
   returns. The rows aren't in the DB at all (not visible as deferred).
2. **DB rollback / migration.** Effectively yes, but no migration code ran.
   The DB file was at its 2026-05-11 01:45:16 state; the new instance
   presented that as ground truth. Functionally indistinguishable from a
   rollback.
3. **Filing-log writes happened but submit_task calls never completed
   (curator timeouts ate the tickets).** Rejected. The tickets DB shows the
   curator did create the rows (status=created with the missing task_ids).
   The orchestrator log proves tasks 3491 and 3582 were live at 14:20 May
   13. The losses happened later, at restart.

The 115s `resolve_ticket` timeouts the user hit during the /deb session are
likely a separate symptom of the same disk/checkpoint pathology under load
(SQLite ops slow as the WAL grows unbounded), worth tracking but not the
primary cause of the row loss.

## Affected scope

### What's gone (lower bound)
- 12 ComputeNode contract §8 DAG tasks (3491-3502)
- 17 multi-kernel-phase-3 §8 DAG tasks (3526-3542)
- 13 GR-024 buckling-eigensolver §13 DAG tasks (3576-3588)
- 6 curator-created singletons (3575, 3578, 3579, 3580, 3581, 3584)
- ~14 more known via worktrees: 3505, 3546-3549, 3564-3565, 3574, 3593, 3598
- Probably ~80 more in gaps 3417-3490 / 3503-3524 / 3528-3545 etc.
- Status flips on 3379, 3383, 3384 (cancelled → reverted to
  pending/in-progress) and many "done" flips between 2026-05-11 and
  2026-05-13 14:25
- All metadata updates (memory_hints, files, dependencies) in the same
  window

### What survived (recovery inputs)
- The three filing logs (titles, deps, supersession)
- 6 ticket `candidate_json` blobs in `tickets.db`
- 14 git branches `task/<id>` and worktrees (mostly empty — architects
  hit caps before producing output)
- `gap-register.md` prose (needs ID rewrite)

### Other projects — TBD
The same `SqliteTaskBackend` serves dark_factory, autopilot_video, autotrade,
know_live with the same connection-cached/per-process WAL pattern. Whether
they suffered comparable losses depends on their per-project DB mtimes
relative to 2026-05-13 14:25. The audit work-stream owns this question.

## Open questions

1. What specifically blocks PASSIVE checkpoints? Is it really long-held read
   snapshots, or is the per-project asyncio.Lock pinning something? Worth
   instrumenting before the hardening session commits to a fix.
2. Why didn't the SHM file survive SIGABRT cleanly? The kernel page cache
   should have preserved it. Possibly aiosqlite's connection-worker thread
   was mid-write when killed and the partial state corrupted the index.
3. Is `wal_checkpoint(TRUNCATE)` on a clean shutdown enough, or do we need
   a periodic background TRUNCATE? The hardening session should answer this
   by reproducing the loss in a controlled SIGKILL test.

## Forensic helper snippets

Reading WAL frames (read-only, safe to repeat):
```python
import struct
with open('/path/to/.taskmaster/tasks/tasks.db-wal', 'rb') as f:
    data = f.read()
hdr_pgsz = 4096
frame_sz = 24 + hdr_pgsz
hdr_salt1, hdr_salt2 = struct.unpack('>II', data[16:24])
nframes = (len(data) - 32) // frame_sz
for i in range(nframes):
    off = 32 + i * frame_sz
    page_num, db_size, s1, s2, c1, c2 = struct.unpack('>IIIIII', data[off:off+24])
    page = data[off+24:off+24+hdr_pgsz]
    # … grep for unique title strings, count salt generations, etc.
```

Querying the DB without sqlite3 CLI (which mismatches the on-host SQLite
version):
```python
import sqlite3
con = sqlite3.connect('/home/leo/src/<project>/.taskmaster/tasks/tasks.db')
con.execute('PRAGMA wal_checkpoint(PASSIVE)')   # safe; do NOT use TRUNCATE
print(con.execute('SELECT count(*), max(CAST(id AS INTEGER)) FROM tasks').fetchone())
```

Cross-referencing tickets that claim to have created now-missing rows:
```python
import sqlite3
con = sqlite3.connect('/home/leo/src/dark-factory/data/reconciliation/tickets.db')
for r in con.execute("""
    SELECT ticket_id, project_id, task_id, created_at
    FROM tickets WHERE status='created' AND created_at >= '2026-05-11'
    ORDER BY created_at DESC""").fetchall():
    # cross-check task_id exists in <project>/.taskmaster/tasks/tasks.db
    pass
```
