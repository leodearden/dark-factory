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

## 2. Gate wall (`merge_verify.duration_ms`), keyed by tree and arm

`merge_verify.duration_ms` is the whole gate: nine modules run one after another, and each
module's test, lint and type-check legs run side by side. One Python block keys every
`merge_verify` row in [T − 7 d, T + 7 d) by §1(c) and §1(d), then reports each
(tree × arm) cell. Run it from the worktree root; it calls `git merge-base` once per row.

```python
import json, sqlite3, statistics, subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
EDIT = 'b39a4f57caa4818967fbf9ccb9c1556891a8fed5'
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
ARMS = [(datetime(2026, 9, 8, 10, 18, 7, 313447, tzinfo=timezone.utc), 16),   # config_reload instants, §1(d)
        (datetime(2026, 9, 10, 19, 6, 50, 398743, tzinfo=timezone.utc), 8),
        (datetime(2026, 9, 17, 11, 39, 26, 361661, tzinfo=timezone.utc), 16)]

def arm_at(t):
    return [value for boundary, value in ARMS if boundary <= t][-1]

def pct(xs, p):  # nearest rank, the definition scripts/merge-pytest-n-ab-analysis.py::pct uses
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, int(round(p * (len(ys) - 1)))))]

rows = []
for ts, task, data in sqlite3.connect(DB, uri=True).execute(
        "SELECT timestamp, task_id, data FROM events WHERE event_type = 'merge_verify' "
        "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
        ((T - timedelta(days=7)).isoformat(), (T + timedelta(days=7)).isoformat())):
    d = json.loads(data)
    end = datetime.fromisoformat(ts)
    rc = subprocess.run(['git', 'merge-base', '--is-ancestor', EDIT, d['merge_sha']], capture_output=True).returncode
    rows.append(dict(ts=ts, task=task, secs=d['duration_ms'] / 1000, passed=d['passed'], spec=bool(d.get('speculative')),
                     runner=d.get('runner'), window='BEFORE' if end < T else 'AFTER',
                     edit={0: 'edit', 1: 'no-edit'}.get(rc, f'unresolved rc={rc}'),
                     arm=arm_at(end - timedelta(milliseconds=d['duration_ms']))))

def clean(sel):  # passed, non-speculative, local
    return [r['secs'] for r in sel if r['passed'] and not r['spec'] and r['runner'] == 'local']

print('unresolved:', [(r['ts'], r['task'], r['edit']) for r in rows if r['edit'].startswith('unresolved')])
cells = defaultdict(list)
for r in rows:
    cells[r['edit'], r['arm']].append(r)
med = {}
for key in sorted(cells):
    sel = cells[key]; xs = clean(sel); med[key] = statistics.median(xs) if xs else None
    print(key, 'n', len(sel), 'failed', sum(not r['passed'] for r in sel), 'spec', sum(r['spec'] for r in sel),
          'clean', len(xs), 'median', round(med[key]), 'p90', round(pct(xs, 0.9)), 'max', round(max(xs)),
          'first', sel[0]['ts'][:19], 'last', sel[-1]['ts'][:19],
          'windows', sorted({r['window'] for r in sel}), 'tasks' if len(sel) < 6 else '', [r['task'] for r in sel] if len(sel) < 6 else '')
for w in ('BEFORE', 'AFTER'):
    xs = clean([r for r in rows if r['window'] == w]); med[w] = statistics.median(xs)
    print(w, 'clean n', len(xs), 'median', round(med[w]))
print('(i) naive AFTER/BEFORE', round(med['AFTER'] / med['BEFORE'], 3))
print('(ii) arm 16 edit/no-edit', round(med['edit', 16] / med['no-edit', 16], 3))
print('(iii) arm 8 edit/no-edit', round(med['edit', 8] / med['no-edit', 8], 3))
predicted = med['no-edit', 8] / 1.296
print('arm switch alone predicts', round(predicted), 'observed (edit,16)', round(med['edit', 16]),
      'residual s', round(med['edit', 16] - predicted), 'residual ratio', round(med['edit', 16] / predicted, 3))
```

```
unresolved: []
('edit', 8) n 5 failed 0 spec 2 clean 3 median 4196 p90 4561 max 4561 first 2026-09-17T06:27:06 last 2026-09-17T11:35:19 windows ['AFTER', 'BEFORE'] tasks ['5408', '5450', '4984', '5359', '4252']
('edit', 16) n 142 failed 20 spec 66 clean 62 median 3136 p90 4048 max 4665 first 2026-09-17T13:05:54 last 2026-09-24T06:11:07 windows ['AFTER']
('no-edit', 8) n 98 failed 27 spec 30 clean 49 median 4224 p90 5329 max 6857 first 2026-09-10T20:13:45 last 2026-09-17T05:24:41 windows ['BEFORE']
('no-edit', 16) n 13 failed 2 spec 7 clean 5 median 3294 p90 4289 max 4289 first 2026-09-10T06:33:09 last 2026-09-10T18:16:27 windows ['BEFORE']
BEFORE clean n 54 median 4188
AFTER clean n 65 median 3169
(i) naive AFTER/BEFORE 0.757
(ii) arm 16 edit/no-edit 0.952
(iii) arm 8 edit/no-edit 0.994
arm switch alone predicts 3259 observed (edit,16) 3136 residual s -123 residual ratio 0.962
```

**All 258 rows resolved: 0 unresolved.** Every edit-absent row ended in BEFORE. Every row that
ended in AFTER is edit-present, and so is 5408's own verify in BEFORE.

"Clean" means passed, non-speculative and runner `local`: the population 5204's A/B used.
Speculative and failed rows overlap, so the three counts need not sum to n. Percentiles are
nearest-rank, as in `scripts/merge-pytest-n-ab-analysis.py::pct`. First and last are verify end
times.

| tree | arm | n (all) | failed | speculative | n clean | median s | p90 s | max s | first → last (UTC) |
|---|---|---|---|---|---|---|---|---|---|
| edit-absent | 16 | 13 | 2 | 7 | **5** | 3294 | 4289 | 4289 | 09-10T06:33:09 → 09-10T18:16:27 |
| edit-absent | 8 | 98 | 27 | 30 | 49 | 4224 | 5329 | 6857 | 09-10T20:13:45 → 09-17T05:24:41 |
| edit-present | 8 | 5 | 0 | 2 | **3** | 4196 | 4561 | 4561 | 09-17T06:27:06 → 09-17T11:35:19 |
| edit-present | 16 | 142 | 20 | 66 | 62 | 3136 | 4048 | 4665 | 09-17T13:05:54 → 09-24T06:11:07 |
| *supplementary, quoted from 5204's report, not recomputed:* edit-absent | 16 | 52 | 27 | 12 | 14 | 3212 | 4270 | 4289 | 09-08T10:38 → 09-10T18:16 |

No verify straddles an arm switch. The last 16-arm verify before the 09-10 switch ended at
18:16:27, and the first 8-arm one ended at 20:13:45. Around the 09-17 switch the gap runs from
11:35:19 to 13:05:54. Keying arms by start or by end therefore gives the same cells here.

The edit-present 8-arm cell is 5408's own verify plus four verifies that ended in the five hours
between T and the 09-17T11:39Z switch. The edit-absent 16-arm cell is the first 12 hours of
BEFORE, before the 09-10T19:06Z switch. Both are tiny, and **neither the windows nor the cells
are widened to fill them.** The supplementary row is 5204's committed 16-arm figure from
`plans/merge-pytest-n-ab-report-2026-09-16.md`. Its window starts on 09-08, outside BEFORE, and
overlaps this report's edit-absent 16-arm cell only on 09-10.

### Contrasts

| contrast | numerator (n) | denominator (n) | ratio | reading |
|---|---|---|---|---|
| (i) naive 7 d / 7 d | AFTER 3169 s (65) | BEFORE 4188 s (54) | **0.757** | −24.3%. Mixes 5408 with the arm switch; see below |
| (ii) arm-matched at 16 | edit-present 3136 s (62) | edit-absent 3294 s (**5**) | 0.952 | −4.8%. **Indicative only**: n = 5 < 10 |
| (iii) arm-matched at 8 | edit-present 4196 s (**3**) | edit-absent 4224 s (49) | 0.994 | −0.6%. **Indicative only**: n = 3 < 10 |
| supplementary, at 16 | edit-present 3136 s (62) | 5204's 16 arm 3212 s (14) | 0.976 | −2.4%. Different window; quoted, not recomputed |

**What the arm switch alone predicts.** 5204 measured median(8) / median(16) = 1.296 on the
whole gate. Applied to this report's edit-absent 8-arm median, that predicts a 16-arm gate of
4224 / 1.296 = **3259 s** with no edit at all. The observed edit-present 16-arm median is
3136 s. The **residual is −123 s, a ratio of 0.962**.

So the arm switch alone predicts a 965 s drop (4224 → 3259 s), against the naive drop of
1,019 s (4188 → 3169 s). Roughly 4% of gate wall is left for 5408 and every other confound in
§1(f). The three arm-aware estimates
of 5408's effect (−4.8%, −0.6% and −3.8%) and the supplementary row (−2.4%) agree in sign and
size. None of them rests on two cells that both have n ≥ 10.

The 1.296 ratio was measured on edit-absent trees. On edit-present trees the four changed
modules also scale with the pin, so the ratio may differ somewhat there. §5(e) compares the
residual with the per-module savings from §3.

### Landings per window

A landing is a `merge_finalized` event with state `done`, or a `train_merged` event, deduped by
merge sha. That is the 6021 report's §2 definition.

```sql
WITH landings AS (
  SELECT min(timestamp) AS ts,
         COALESCE(json_extract(data,'$.merge_sha'), json_extract(data,'$.merge_commit_sha')) AS sha
  FROM events
  WHERE timestamp >= '2026-09-10T06:30:01.881486+00:00' AND timestamp < '2026-09-24T06:30:01.881486+00:00'
    AND ((event_type = 'merge_finalized' AND json_extract(data,'$.state') = 'done') OR event_type = 'train_merged')
  GROUP BY sha
)
SELECT CASE WHEN ts < '2026-09-17T06:30:01.881486+00:00' THEN 'BEFORE' ELSE 'AFTER' END AS w,
       substr(ts, 1, 10) AS day, count(*)
FROM landings GROUP BY w, day ORDER BY day;
```

| UTC day | 09-10* | 09-11 | 09-12 | 09-13 | 09-14 | 09-15 | 09-16 | 09-17* | total | per hour |
|---|---|---|---|---|---|---|---|---|---|---|
| BEFORE | 11 | 5 | 12 | 13 | 10 | 13 | 12 | 5 | **81** | 0.48 |

| UTC day | 09-17* | 09-18 | 09-19 | 09-20 | 09-21 | 09-22 | 09-23 | 09-24* | total | per hour |
|---|---|---|---|---|---|---|---|---|---|---|
| AFTER | 14 | 18 | 17 | 17 | 15 | 25 | 18 | 2 | **126** | 0.75 |

\* partial day: each window starts and ends at 06:30:01Z, and 09-17 is split at T. The 207 rows
(189 `merge_finalized`, 18 `train_merged`) carry 207 distinct shas.

**The landing rate is arrival-bounded,** as the 6021 report's caveat says. Arrivals rose too:

```sql
SELECT CASE WHEN timestamp < '2026-09-17T06:30:01.881486+00:00' THEN 'BEFORE' ELSE 'AFTER' END AS w,
       count(*), count(DISTINCT task_id)
FROM events
WHERE event_type = 'merge_queued'
  AND timestamp >= '2026-09-10T06:30:01.881486+00:00' AND timestamp < '2026-09-24T06:30:01.881486+00:00'
GROUP BY w ORDER BY w DESC;
-- BEFORE | 186 | 110
-- AFTER  | 208 | 151
```

Distinct tasks enqueued rose 37%, and landings rose 56%. The +56% therefore does not measure
5408, nor the gate on its own.
