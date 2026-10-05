# Trial E1 readout for task 5408 — merge-gate parallelism, 7 days before vs after

Task 5469. Task 5408 ("Merge-gate parallelism config") added `-n auto --dist loadgroup` to the
pytest legs of four merge-gate modules: scripts, dashboard, escalation and cockpit. It also added
`pyright --threads 8` to the shared and sampler type checks. Its WORK item 4 defines this trial
and its stop rule, verbatim from the task record:

> 4. Trial E1: per-module suite wall and CPU-seconds from the junit/summary files, 7 days before
> vs after (keyed by whether the tree contains the edit); stop rule: verify timeouts above the 4%
> baseline or gate cgroup peak RSS +25% → revert the addopts.

The same record states the user-observable signal: "merge_verify per-module timings:
scripts/dashboard/escalation/cockpit each under half their serial archive median".

Sources. All of them are read-only and outside the worktree:

- `/home/leo/src/dark-factory/data/orchestrator/runs.db`, opened as `sqlite3 -readonly …` or
  `sqlite3.connect('file:…/runs.db?mode=ro', uri=True)`. Only the `events` table is used.
- `/home/leo/src/dark-factory/data/verify-logs/<task_id>/`, the verify archive. Merge-gate junit
  reports are `attempt-N.<module>.junit-<UTC stamp>Z.xml.gz`; failure summaries are
  `attempt-N.<module>.summary-<UTC stamp>Z.json`.
- `git log` on `main`, through the shared object store.

## 1. Landing, windows, keying, arms, and what the archive can and cannot say

### (a) Landing

```
$ git log -1 --format='%H %cI %s%n parents: %P' b39a4f57ca
b39a4f57caa4818967fbf9ccb9c1556891a8fed5 2026-09-17T05:22:50+01:00 Merge task/5408 into main
 parents: c954faa62bade0892601705e4a8d172d9b7790d1 b685979a9349124655c143965ece945c42b819c8
```

```sql
SELECT timestamp, event_type, json_extract(data,'$.state'), json_extract(data,'$.passed'),
       json_extract(data,'$.duration_ms'), json_extract(data,'$.speculative'), json_extract(data,'$.merge_sha')
FROM events
WHERE task_id = '5408' AND event_type IN ('merge_queued', 'merge_verify', 'merge_finalized')
ORDER BY timestamp;
-- 2026-09-17T02:53:10.737016+00:00 | merge_queued    |      |   |         |   |
-- 2026-09-17T06:27:06.679644+00:00 | merge_verify    |      | 1 | 3544481 | 1 | b39a4f57caa4818967fbf9ccb9c1556891a8fed5
-- 2026-09-17T06:30:01.881486+00:00 | merge_finalized | done |   |         |   | b39a4f57caa4818967fbf9ccb9c1556891a8fed5
```

**T = 2026-09-17T06:30:01.881486Z**, the `merge_finalized` event with state `done`. The merge
commit's committer time, 04:22:50Z, is two hours earlier because the merge commit is built before
its verify runs. 5408's own verify ran on runner `local`, speculatively, from 05:28:02Z to
06:27:06Z (3,544 s), and passed. T is the moment main advanced, which is when the edit began
gating other merges. The 6021 report anchors on the same kind of event.

### (b) Windows

- **BEFORE** = [T − 7 d, T) = [2026-09-10T06:30:01.881486Z, T).
- **AFTER** = [T, T + 7 d) = [T, 2026-09-24T06:30:01.881486Z).

A `merge_verify` row belongs to the window that contains its timestamp, which is the moment the
verify ended. Event timestamps are ISO-8601 UTC strings that all end in `+00:00`, so string
comparison is time comparison. The windows are not widened anywhere in this report.

### (c) Keying rule: does the verified tree contain the edit?

`merge_verify.data.merge_sha` is the exact tree the gate verified. A row is **edit-present** iff

```
git merge-base --is-ancestor b39a4f57caa4818967fbf9ccb9c1556891a8fed5 <merge_sha>
```

exits 0. Exit 1 means **edit-absent**. Any other exit is **unresolved**; §2 counts and lists those
rows, and finds none.

Keying by tree rather than by date puts each verify where it belongs. 5408's own verify is the
clearest case: it ended at 06:27:06Z, before T, so by date it is in BEFORE. Its merge sha is
b39a4f57ca itself, so it is edit-present:

```
$ git merge-base --is-ancestor b39a4f57caa4818967fbf9ccb9c1556891a8fed5 b39a4f57caa4818967fbf9ccb9c1556891a8fed5; echo $?
0
```

### (d) Arm rule: the xdist worker count in force when the verify started

Each module's `-n auto` resolves its worker count through the environment variable
`PYTEST_XDIST_AUTO_NUM_WORKERS`. The orchestrator sets it from `verify_env` in
`dark-factory-orchestrator.yaml`. After 5408 that covers the four changed modules, and it already
covered orchestrator and fused-memory through their own addopts. So the pin decides both how many
workers 5408's new `-n auto` gets and how many the gate's longest leg, orchestrator, gets.

The pin changed three times around the windows, and each change is a hot `config_reload`:

```sql
SELECT timestamp, run_id, json_extract(data,'$.applied.verify_env')
FROM events
WHERE event_type = 'config_reload' AND json_extract(data,'$.applied.verify_env') IS NOT NULL
ORDER BY timestamp;
-- 2026-09-08T10:18:07.313447+00:00 | run-4a4313b1e5e7 | {"old":{},"new":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"16"}}
-- 2026-09-10T19:06:50.398743+00:00 | run-4a4313b1e5e7 | {"old":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"16"},"new":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"8"}}
-- 2026-09-17T11:39:26.361661+00:00 | run-9b3c5d8df8ae | {"old":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"8","DF_REQUIRE_SANDBOX_TESTS":"1"},"new":{"PYTEST_XDIST_AUTO_NUM_WORKERS":"16","DF_REQUIRE_SANDBOX_TESTS":"1"}}
```

A verify's **arm** is the value of the last of these events at or before its start, where start
= event timestamp − `data.duration_ms`. Arms are keyed at the start because a gate runs for about
an hour. The 8 → 16 switch at 09-17T11:39Z falls five hours after T, inside AFTER. Keying by the
end would credit some 8-worker verifies to 16. Ignoring arms altogether would credit 5408 with
5204's worker-count effect, which is the esc-5408-1 confound.

The git side agrees. The two arm commits are 5204's A/B switches:

```
$ git log --format='%h %cI %s' bd4825403e -1; git log --format='%h %cI %s' 16c25d49f8 -1
bd4825403e 2026-09-10T13:42:14+01:00 config(ab): merge test leg PYTEST_XDIST_AUTO_NUM_WORKERS=8 (pytest -n A/B arm switch, 2026-09-10T12:42:14Z)
16c25d49f8 2026-09-17T12:39:20+01:00 config(ab): merge test leg PYTEST_XDIST_AUTO_NUM_WORKERS=16 (pytest -n A/B arm switch, 2026-09-17T11:39:20Z)
```

bd4825403e reached the file at 12:42Z, but the running process kept 16 until the reload at
19:06:50Z. The reload event, not the commit, is the authority. A restart re-reads the file, so
each restart inside the windows must also read the value the reload rule assigns. All six do:

```sql
SELECT run_id, min(timestamp), max(timestamp) FROM events
WHERE timestamp >= '2026-09-10T06:30:01' AND timestamp < '2026-09-24T06:30:01'
GROUP BY run_id ORDER BY min(timestamp);
```

```
$ for ts in 2026-09-10T19:09:58Z 2026-09-11T16:06:37Z 2026-09-12T09:39:17Z 2026-09-14T12:31:48Z 2026-09-15T23:00:46Z 2026-09-21T03:44:54Z; do
    c=$(git rev-list -1 --first-parent --before=$ts main)
    echo "$ts $(git rev-parse --short=10 $c) $(git show $c:dark-factory-orchestrator.yaml | grep -E '^\s+PYTEST_XDIST_AUTO_NUM_WORKERS')"; done
```

The first query's earliest timestamp for run-4a4313b1e5e7 is clipped to the window; that run was
already running on 09-07 and made the 09-08 reload.

| run | first event (UTC) | main at that instant | file value | reload rule says |
|---|---|---|---|---|
| run-4a4313b1e5e7 | (running since before 09-08) | — | — | 16, then 8 from 09-10T19:06:50 |
| run-8e746d406e3b | 09-10T19:09:58 | 105168bace | 8 | 8 |
| run-b756e05f8ff0 | 09-11T16:06:37 | 6cfdb75960 | 8 | 8 |
| run-063e9795fcf6 | 09-12T09:39:17 | e9d1055ed8 | 8 | 8 |
| run-5352d42ce5fa | 09-14T12:31:48 | 99ab62335a | 8 | 8 |
| run-9b3c5d8df8ae | 09-15T23:00:46 | 016f3547a8 | 8 | 8, then 16 from 09-17T11:39:26 |
| run-ab649438402a | 09-21T03:44:54 | 4ff6f032f8 | 16 | 16 |

### (e) Scope: merge-role legs only

- At the task and background roles, `orchestrator/src/orchestrator/verify.py::_with_pytest_numprocesses_str`
  injects an explicit `-n`, so 5408's addopts do not change those legs' worker shape. They are
  outside the trial.
- At the merge role nothing is injected, and the module's own addopts decide. That is the shape
  5408 changed.
- The merge role is also the only per-module timing corpus that exists for the windows. In them,
  junit reports were written only for merge verifies at full breadth
  (`verify.py::_JunitReportPlan`, kind `attributing`). Task 5671 added task-role junit on
  2026-10-01, after both windows closed.

### (f) Confound census

Everything inside [T − 7 d, T + 7 d) that can move gate wall or a timeout budget. The git half
comes from

```
$ git log --first-parent main --format='%h %cI %s' --since=2026-09-10T06:30:01Z --until=2026-09-24T06:30:01Z \
    -- '*/orchestrator.yaml' '*/pyproject.toml' dark-factory-orchestrator.yaml \
       orchestrator/src/orchestrator/verify.py orchestrator/src/orchestrator/verify_cmd.py
```

which lists 37 first-parent commits. Each was read with `git diff <c>^1 <c> -- <same paths>`.
Commits that change only comments, dependency pins, model routing, agent budgets, alarms or lint
commands are left out. So are dbfe85adbf (4071: an admission slot is released when a verify is
cancelled mid-acquire, a correctness fix) and fa95988c8e (merge-queue drift handling, in
ed18089894), which is process code merged on 09-23 with no restart before the window closed.

A change takes effect in one of three ways:

- **tree**: module `orchestrator.yaml` and `pyproject.toml` files are re-read from the verified
  tree on every verify, so the change reaches exactly the trees that contain it, as 5408 does;
- **reload**: a hot `config_reload` event;
- **restart**: process code (`verify.py`, `verify_cmd.py`) and restart-only keys take effect at the
  next orchestrator restart. The restarts are listed in (d). There was none between
  09-15T23:00:46Z and 09-21T03:44:54Z.

| effective (UTC) | how | change | source | moves | window touched |
|---|---|---|---|---|---|
| before both windows | — | `merge_verify_max_concurrent_modules: 1`: the gate runs its modules one at a time. Unchanged throughout | f7504a2f91, 2026-07-19 | — (holds gate shape fixed) | both |
| 09-10T19:06:50 | reload | arm 16 → 8 | config_reload; bd4825403e | gate wall | BEFORE |
| 09-11T16:06:37 | restart | `max_concurrent_tasks` 48 → 24 | 6cfdb75960 | host load | BEFORE (the first 34 h ran at 48) |
| 09-12T07:40:47 | tree | per-test pytest timeout 60 → 300 in every package | 64e24b547f | per-test budget | both, almost entirely |
| 09-12T09:39:17 | restart | fleet verify budget 3600 → 7200 s warm, 5400 → 10800 s cold | 36c4c71eb4 | leg timeout budget | BEFORE from 09-12, all of AFTER |
| 09-12T09:39:17 | restart | `verify_env` gains `DF_REQUIRE_SANDBOX_TESTS: "1"` (4635) | 12d9b91bf2 | sandbox tests run instead of skipping | BEFORE from 09-12, all of AFTER |
| 09-13T09:46:37 | tree | orchestrator module `verify_command_timeout_secs: 7200` (5422) | bea48785d0 | orchestrator leg budget, equal to the fleet value | both |
| 09-14T12:31:48 | restart | merge admission no longer queues ungated roles behind the executor (5424) | bd4574c8d1 | merge-verify queueing | BEFORE from 09-14, all of AFTER |
| **09-17T06:30:01** | **tree** | **the treatment: 5408's `-n auto --dist loadgroup` in four modules, and `pyright --threads 8` in shared and sampler** | **b39a4f57ca** | **gate wall** | **AFTER, plus 5408's own verify** |
| 09-17T11:39:26 | reload | arm 8 → 16 | config_reload; 16c25d49f8 | gate wall | AFTER |
| 09-19T14:28:18 | tree | per-test pytest timeout 300 → 540 in dashboard, escalation, fused-memory, orchestrator and shared; cockpit and sampler capped at 540 for the first time (5442) | 3c7926ce10, 99cf99d1ef, ccc6d83e7f | per-test budget | AFTER |
| 09-19T21:03:40 | tree | orchestrator module `verify_cold_command_timeout_secs: 10800` (3353) | b27049d02d | orchestrator cold budget | AFTER |
| 09-21T03:44:54 | restart | five days of process code at once: 5408's serial-recovery flag stripping in `verify_cmd.py`; the xdist-crash allow-list retirement (b5bf73106e); 3353's worker-count and load stamping; 5670's green+red junit retention (a726d0b83e); 5580's re-run command parsing | see the left | junit corpus starts here; gate wall not expected to move | AFTER |
| 09-23T11:53:28 | reload | `verify_admission_task_slots` 1 → 2 | config_reload; c4359d1f2e | host load | AFTER, last 19 h |

### (g) Data coverage

`merge_verify` rows are complete in both windows:

```sql
SELECT CASE WHEN timestamp < '2026-09-17T06:30:01.881486+00:00' THEN 'BEFORE' ELSE 'AFTER' END,
       count(*), sum(json_extract(data,'$.passed')), count(DISTINCT task_id),
       group_concat(DISTINCT json_extract(data,'$.runner'))
FROM events
WHERE event_type = 'merge_verify'
  AND timestamp >= '2026-09-10T06:30:01.881486+00:00' AND timestamp < '2026-09-24T06:30:01.881486+00:00'
GROUP BY 1 ORDER BY 1 DESC;
-- BEFORE | 112 |  83 |  85 | local
-- AFTER  | 146 | 126 | 128 | local

SELECT min(timestamp) FROM events WHERE event_type = 'merge_verify' AND json_extract(data,'$.runner') != 'local';
-- 2026-10-01T05:14:36.301881+00:00
```

Every one of the 258 ran on runner `local`. The first non-local merge verify, on the laptop, came
on 2026-10-01.

Merge-role junit starts on 2026-09-21. The earliest archived junit per module, and the count
with a filename stamp in each window:

```python
import glob, os, re
from collections import Counter
from datetime import datetime, timedelta, timezone
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
pat = re.compile(r'attempt-\d+\.(?P<module>[^.]+)\.junit-(?P<stamp>\d{8}T\d{6})_\d+Z\.xml\.gz$')
first, n = {}, Counter()
for p in glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/attempt-*.junit-*Z.xml.gz'):
    m = pat.search(os.path.basename(p))
    stamp = datetime.strptime(m['stamp'], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    first[m['module']] = min(first.get(m['module'], stamp), stamp)
    n[m['module'], 'BEFORE' if stamp < T else 'AFTER' if stamp < T + timedelta(days=7) else 'later'] += 1
for module in sorted(first, key=first.get):
    print(module, first[module].isoformat(), n[module, 'BEFORE'], n[module, 'AFTER'])
```

| module | earliest junit (UTC) | BEFORE | AFTER |
|---|---|---|---|
| cockpit | 2026-09-21T03:45:23 | 0 | 67 |
| dashboard | 2026-09-21T03:46:33 | 0 | 67 |
| escalation | 2026-09-21T03:47:16 | 0 | 67 |
| fused-memory | 2026-09-21T03:51:49 | 0 | 67 |
| orchestrator | 2026-09-21T04:21:25 | 0 | 65 |
| sampler | 2026-09-21T04:22:04 | 0 | 65 |
| scripts | 2026-09-21T04:23:23 | 0 | 65 |
| shared | 2026-09-21T04:26:09 | 0 | 65 |
| tests_scripts | 2026-09-21T04:30:19 | 0 | 65 |

All nine modules start within 45 minutes of the 09-21T03:44:54Z restart that deployed 5670's
green+red junit retention. So BEFORE has no per-module junit at all, and AFTER has it for
09-21T03:45Z → 09-24T06:30Z only: about 3.1 of its 7 days.

Archived summaries are failure-only. Before task 5671 they carry no role:

```python
import glob, json, sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, 'scripts')
from verify_budget_census import parse_record_path
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
W = {'BEFORE': (T - timedelta(days=7), T), 'AFTER': (T, T + timedelta(days=7))}
tally = {w: Counter() for w in W}; rcs = {w: Counter() for w in W}; roles = {w: Counter() for w in W}
for p in glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/*.summary-*.json'):
    rec = parse_record_path(Path(p))
    if rec is None:
        continue
    at = datetime.strptime(rec.archived_at.split('_')[0].rstrip('Z'), '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    for w, (lo, hi) in W.items():
        if lo <= at < hi:
            d = json.load(open(p))
            tally[w][d.get('category')] += 1; rcs[w][d.get('rc') == 0] += 1; roles[w][d.get('role')] += 1
for w in W:
    print(w, sum(tally[w].values()), dict(tally[w]), 'rc==0:', dict(rcs[w]), 'role:', dict(roles[w]))
```

```
BEFORE 40 {'test_failure': 31, 'unknown_test_failure': 9} rc==0: {False: 40} role: {None: 40}
AFTER 35 {'test_failure': 30, 'disk_full': 1, 'unknown_test_failure': 3, 'infra_kill': 1} rc==0: {False: 35} role: {None: 35}
```

Run from the worktree root; `parse_record_path` is
`scripts/verify_budget_census.py::parse_record_path`. None of the 75 summaries has rc 0. These are
task-role and merge-role legs mixed together, and §3(b) separates them.

**Retention.** `verify.py::_prune_archive` deletes archive files older than 30 days by mtime
(`_DEFAULT_ARCHIVE_MAX_AGE_DAYS = 30`), then oldest-first above 500 MB. The oldest surviving
summary is from 2026-09-05, and the archive held 455,651,126 bytes on 2026-10-05. So nothing
inside either window has been evicted yet. But BEFORE's summaries start ageing out on
2026-10-10. §3(b)'s per-record table is the durable copy of them.

| quantity | BEFORE | AFTER |
|---|---|---|
| `merge_verify` rows (gate wall) | complete: 112 | complete: 146 |
| merge-role junit (per-module wall, Σ testcase time) | none | 09-21T03:45Z onward only, 65-67 per module |
| archived summaries | 40, all failures, role not recorded | 35, all failures, role not recorded |
| CPU-seconds | no field in any event, summary or junit | the same |
| cgroup peak RSS | no field anywhere: DF verifies run unscoped, because `dark-factory-orchestrator.yaml` does not set `verify_use_cgroup_scope` and its default in `orchestrator/src/orchestrator/config.py` is `False`. Tasks 5206 and 5415 own it | the same |

CPU-seconds and peak memory therefore cannot be read from production for either window. §4
measures them in a controlled run instead.
