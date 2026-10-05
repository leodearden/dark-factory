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
- At the merge role nothing is injected, and the module's own addopts (or, for scripts, its test
  command) decide. That is the shape 5408 changed.
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

- **tree**: a module's `pyproject.toml` (its pytest `addopts` and `timeout`) is read by pytest
  from the verified tree, so the change reaches exactly the trees that contain it;
- **reload**: a hot `config_reload` event;
- **restart**: process code (`verify.py`, `verify_cmd.py`), restart-only keys, and every module
  `<module>/orchestrator.yaml` take effect at the next orchestrator restart. The restarts are
  listed in (d). There was none between 09-15T23:00:46Z and 09-21T03:44:54Z.
  - Module `orchestrator.yaml` files belong here, not under "tree". At full breadth the local merge
    gate takes each module's test, lint and type commands from `config.module_configs_or_empty`
    (`orchestrator/src/orchestrator/verify_plan.py::effective_merge_module_configs`).
    `config.py::load_config` fills that once at startup, from the main checkout, via
    `_discover_module_configs`. §3(f) shows the consequence for 5408, which edited
    `scripts/orchestrator.yaml`.

| effective (UTC) | how | change | source | moves | window touched |
|---|---|---|---|---|---|
| before both windows | — | `merge_verify_max_concurrent_modules: 1`: the gate runs its modules one at a time. Unchanged throughout | f7504a2f91, 2026-07-19 | — (holds gate shape fixed) | both |
| 09-10T19:06:50 | reload | arm 16 → 8 | config_reload; bd4825403e | gate wall | BEFORE |
| 09-11T16:06:37 | restart | `max_concurrent_tasks` 48 → 24 | 6cfdb75960 | host load | BEFORE (the first 34 h ran at 48) |
| 09-12T07:40:47 | tree | per-test pytest timeout 60 → 300 in every package | 64e24b547f | per-test budget | both, almost entirely |
| 09-12T09:39:17 | restart | fleet verify budget 3600 → 7200 s warm, 5400 → 10800 s cold | 36c4c71eb4 | leg timeout budget | BEFORE from 09-12, all of AFTER |
| 09-12T09:39:17 | restart | `verify_env` gains `DF_REQUIRE_SANDBOX_TESTS: "1"` (4635) | 12d9b91bf2 | sandbox tests run instead of skipping | BEFORE from 09-12, all of AFTER |
| 09-14T12:31:48 | restart | orchestrator module `verify_command_timeout_secs: 7200` (5422, merged 09-13T09:46:37) | bea48785d0 | orchestrator leg budget, equal to the fleet value | BEFORE from 09-14, all of AFTER |
| 09-14T12:31:48 | restart | merge admission no longer queues ungated roles behind the executor (5424) | bd4574c8d1 | merge-verify queueing | BEFORE from 09-14, all of AFTER |
| **09-17T06:30:01** | **tree** | **the treatment, part 1: 5408's `-n auto --dist loadgroup` in the dashboard, escalation and cockpit `pyproject.toml` addopts** | **b39a4f57ca** | **gate wall** | **AFTER, plus 5408's own verify** |
| 09-17T11:39:26 | reload | arm 8 → 16 | config_reload; 16c25d49f8 | gate wall | AFTER |
| 09-19T14:28:18 | tree | per-test pytest timeout 300 → 540 in dashboard, escalation, fused-memory, orchestrator and shared; cockpit and sampler capped at 540 for the first time (5442) | 3c7926ce10, 99cf99d1ef, ccc6d83e7f | per-test budget | AFTER |
| **09-21T03:44:54** | **restart** | **the treatment, part 2: 5408's `-n auto --dist loadgroup` in `scripts/orchestrator.yaml`'s test command, and `pyright --threads 8` in the shared and sampler type commands. See §3(f)** | **b39a4f57ca** | **gate wall** | **AFTER, last 3.1 days** |
| 09-21T03:44:54 | restart | orchestrator module `verify_cold_command_timeout_secs: 10800` (3353, merged 09-19T21:03:40) | b27049d02d | orchestrator cold budget | AFTER, last 3.1 days |
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

## 3. Per-module suite wall and worker-seconds

### (a) AFTER, from the merge-role junit archive

The 6021 report's §4 parser, extended in three ways: to all nine modules, to Σ testcase `time`,
and to a match against `merge_verify`. Each junit whose filename stamp falls in [T, T + 7 d) is
matched to a `merge_verify` row by two conditions:

- the same task id as its directory;
- a filename stamp inside [row start, row end + 300 s].

The matched row supplies the junit's tree key and arm. A junit that matches no row or more than
one is counted and left out.

- **suite wall** = Σ `<testsuite time>`, the pytest session's own wall;
- **worker-s** = Σ `<testcase time>`, the seconds each test held a worker;
- **parallelism** = worker-s ÷ suite wall, per leg, then the median over legs.

```python
import glob, gzip, json, os, re, sqlite3, statistics, subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
EDIT = 'b39a4f57caa4818967fbf9ccb9c1556891a8fed5'
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
ARMS = [(datetime(2026, 9, 8, 10, 18, 7, 313447, tzinfo=timezone.utc), 16),
        (datetime(2026, 9, 10, 19, 6, 50, 398743, tzinfo=timezone.utc), 8),
        (datetime(2026, 9, 17, 11, 39, 26, 361661, tzinfo=timezone.utc), 16)]
MODULES = ['scripts', 'dashboard', 'escalation', 'cockpit',
           'orchestrator', 'fused-memory', 'shared', 'sampler', 'tests_scripts']
pat = re.compile(r'attempt-\d+\.(?P<module>[^.]+)\.junit-(?P<stamp>\d{8}T\d{6})_\d+Z\.xml\.gz$')

def pct(xs, p):
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, int(round(p * (len(ys) - 1)))))]

verifies = defaultdict(list)
for ts, task, data in sqlite3.connect(DB, uri=True).execute(
        "SELECT timestamp, task_id, data FROM events WHERE event_type = 'merge_verify' "
        "AND timestamp >= ? AND timestamp < ?", ((T - timedelta(days=7)).isoformat(), (T + timedelta(days=8)).isoformat())):
    d = json.loads(data); end = datetime.fromisoformat(ts); start = end - timedelta(milliseconds=d['duration_ms'])
    verifies[task].append(dict(start=start, end=end, sha=d['merge_sha'], arm=[v for b, v in ARMS if b <= start][-1]))

legs, unmatched, ambiguous = defaultdict(list), [], []
for p in sorted(glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/attempt-*.junit-*Z.xml.gz')):
    m = pat.search(os.path.basename(p)); task = p.split('/')[-2]
    stamp = datetime.strptime(m['stamp'], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    if not T <= stamp < T + timedelta(days=7):
        continue
    hits = [v for v in verifies[task] if v['start'] <= stamp <= v['end'] + timedelta(seconds=300)]
    if len(hits) != 1:
        (unmatched if not hits else ambiguous).append((task, m['module'], stamp.isoformat()))
        continue
    v = hits[0]
    root = ET.parse(gzip.open(p)).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    attr = {k: sum(float(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped', 'time')}
    worker_s = sum(float(tc.get('time', 0)) for tc in root.iter('testcase'))
    edit = subprocess.run(['git', 'merge-base', '--is-ancestor', EDIT, v['sha']], capture_output=True).returncode
    legs[m['module'], edit, v['arm']].append(dict(attr, worker_s=worker_s, red=attr['failures'] + attr['errors'] > 0))

print('unmatched', len(unmatched), unmatched)
print('ambiguous', len(ambiguous), ambiguous)
for module in MODULES:
    for (mod, edit, arm), sel in sorted(legs.items()):
        if mod != module:
            continue
        wall = [x['time'] for x in sel]; ws = [x['worker_s'] for x in sel]
        print(f"{module:13} edit_rc={edit} arm={arm} n={len(sel)} red={sum(x['red'] for x in sel)} "
              f"tests={statistics.median(x['tests'] for x in sel):.0f} "
              f"wall med/p90/max={statistics.median(wall):.1f}/{pct(wall, 0.9):.1f}/{max(wall):.1f} "
              f"worker_s med={statistics.median(ws):.1f} "
              f"par med={statistics.median(w / t for w, t in zip(ws, wall)):.2f}")
```

```
unmatched 4 [('4978', 'cockpit', '2026-09-23T02:47:13+00:00'), ('4978', 'dashboard', '2026-09-23T02:48:14+00:00'), ('4978', 'escalation', '2026-09-23T02:48:49+00:00'), ('4978', 'fused-memory', '2026-09-23T02:52:41+00:00')]
ambiguous 0 []
scripts       edit_rc=0 arm=16 n=65 red=1 tests=6104 wall med/p90/max=97.2/123.3/154.9 worker_s med=814.1 par med=8.39
dashboard     edit_rc=0 arm=16 n=66 red=1 tests=2627 wall med/p90/max=64.2/78.2/99.5 worker_s med=608.1 par med=9.51
escalation    edit_rc=0 arm=16 n=66 red=0 tests=2069 wall med/p90/max=36.4/48.0/62.7 worker_s med=211.4 par med=5.77
cockpit       edit_rc=0 arm=16 n=66 red=1 tests=482 wall med/p90/max=16.1/24.0/29.6 worker_s med=85.2 par med=5.31
orchestrator  edit_rc=0 arm=16 n=65 red=13 tests=22283 wall med/p90/max=1811.9/2322.8/2951.9 worker_s med=26581.5 par med=14.75
fused-memory  edit_rc=0 arm=16 n=66 red=1 tests=22271 wall med/p90/max=319.9/397.3/474.8 worker_s med=2582.9 par med=8.09
shared        edit_rc=0 arm=16 n=65 red=2 tests=5952 wall med/p90/max=133.5/154.3/185.1 worker_s med=107.1 par med=0.81
sampler       edit_rc=0 arm=16 n=65 red=0 tests=142 wall med/p90/max=6.1/9.1/11.8 worker_s med=5.4 par med=0.91
tests_scripts edit_rc=0 arm=16 n=65 red=0 tests=2280 wall med/p90/max=234.9/303.2/346.5 worker_s med=228.0 par med=0.98
```

The four unmatched junits are task 4978's merge verify that began at 2026-09-23T02:46:49Z. It was
superseded by train `coalesce-4978-b25c5003` (`merge_finalized` state `superseded`, 04:53:33Z)
and never emitted a `merge_verify` row. Every matched leg is edit-present and arm 16. AFTER's
junit coverage begins on 09-21, after the 09-17 arm switch, so there is only one arm to report.

| module | changed by 5408 | n legs | legs with failures | median tests | wall median / p90 / max s | worker-s median | parallelism median |
|---|---|---|---|---|---|---|---|
| **scripts** | yes, test command | 65 | 1 | 6104 | **97.2** / 123.3 / 154.9 | 814.1 | 8.39 |
| **dashboard** | yes, addopts | 66 | 1 | 2627 | **64.2** / 78.2 / 99.5 | 608.1 | 9.51 |
| **escalation** | yes, addopts | 66 | 0 | 2069 | **36.4** / 48.0 / 62.7 | 211.4 | 5.77 |
| **cockpit** | yes, addopts | 66 | 1 | 482 | **16.1** / 24.0 / 29.6 | 85.2 | 5.31 |
| orchestrator | no; `-n auto` in its own addopts | 65 | 13 | 22283 | 1811.9 / 2322.8 / 2951.9 | 26581.5 | 14.75 |
| fused-memory | no; `-n auto` in its own addopts | 66 | 1 | 22271 | 319.9 / 397.3 / 474.8 | 2582.9 | 8.09 |
| shared | no; serial | 65 | 2 | 5952 | 133.5 / 154.3 / 185.1 | 107.1 | 0.81 |
| sampler | no; serial | 65 | 0 | 142 | 6.1 / 9.1 / 11.8 | 5.4 | 0.91 |
| tests_scripts | no; serial | 65 | 0 | 2280 | 234.9 / 303.2 / 346.5 | 228.0 | 0.98 |

The four 5408 modules sum to 214 s of median wall. The post-restart gate median from §3(f) is
3,108 s, so they are about 7% of it. orchestrator alone is 1,812 s, about 58%. The serial
modules show a parallelism just under 1, because the suite wall also counts collection.

### (b) BEFORE, from the failure-selected summaries

There is no BEFORE junit (§1(g)). The only per-module walls for BEFORE are the archived failure
summaries. This takes the summaries of the four 5408 modules, stamped in [T − 7 d, 2026-09-21),
and keeps the ones that match a `merge_verify` row by the same rule as (a). Running the same
range past T also catches the AFTER days that have no junit. Path parsing reuses
`scripts/verify_budget_census.py::parse_record_path`. Run from the worktree root:

```python
import glob, json, sqlite3, subprocess, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, 'scripts')
from verify_budget_census import parse_record_path
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
EDIT = 'b39a4f57caa4818967fbf9ccb9c1556891a8fed5'
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
JUNIT_START = datetime(2026, 9, 21, tzinfo=timezone.utc)
ARMS = [(datetime(2026, 9, 8, 10, 18, 7, 313447, tzinfo=timezone.utc), 16),
        (datetime(2026, 9, 10, 19, 6, 50, 398743, tzinfo=timezone.utc), 8),
        (datetime(2026, 9, 17, 11, 39, 26, 361661, tzinfo=timezone.utc), 16)]
MODULES = ('scripts', 'dashboard', 'escalation', 'cockpit')

verifies = defaultdict(list)
for ts, task, data in sqlite3.connect(DB, uri=True).execute(
        "SELECT timestamp, task_id, data FROM events WHERE event_type = 'merge_verify' "
        "AND timestamp >= ? AND timestamp < ?", ((T - timedelta(days=7)).isoformat(), (T + timedelta(days=8)).isoformat())):
    d = json.loads(data); end = datetime.fromisoformat(ts); start = end - timedelta(milliseconds=d['duration_ms'])
    verifies[task].append(dict(start=start, end=end, sha=d['merge_sha'], arm=[v for b, v in ARMS if b <= start][-1]))

seen = 0
for p in sorted(glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/*.summary-*.json')):
    rec = parse_record_path(Path(p))
    if rec is None or rec.module_prefix not in MODULES:
        continue
    stamp = datetime.strptime(rec.archived_at.split('_')[0].rstrip('Z'), '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    if not T - timedelta(days=7) <= stamp < JUNIT_START:
        continue
    seen += 1
    hits = [v for v in verifies[rec.task_id] if v['start'] <= stamp <= v['end'] + timedelta(seconds=300)]
    if len(hits) != 1:
        print('not merge-role' if not hits else 'ambiguous', rec.task_id, rec.module_prefix, stamp.isoformat())
        continue
    v = hits[0]; d = json.load(open(p))
    test = [c for c in d.get('commands') or [] if c.get('label') == 'test']
    edit = subprocess.run(['git', 'merge-base', '--is-ancestor', EDIT, v['sha']], capture_output=True).returncode
    print('MERGE', rec.module_prefix, rec.task_id, stamp.isoformat(), 'window', 'BEFORE' if stamp < T else 'AFTER',
          'edit_rc', edit, 'arm', v['arm'], 'category', d.get('category'),
          'test_secs', round(test[0]['duration_secs'], 1) if test else None,
          'test_rc', test[0].get('rc') if test else None, 'timed_out', test[0].get('timed_out') if test else None,
          'has_-n', (' -n ' in f" {test[0]['cmd']} ") if test else None,
          'failed_labels', [c.get('label') for c in d.get('commands') or [] if c.get('rc')])
print('summaries seen', seen)
```

```
MERGE scripts 4917 2026-09-14T13:43:19+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 493.4 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE scripts 5021 2026-09-10T23:03:48+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 424.4 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE dashboard 5024 2026-09-15T01:21:02+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 256.1 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE dashboard 5025 2026-09-13T19:06:34+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 299.5 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE dashboard 5032 2026-09-14T13:57:26+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 280.1 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE cockpit 5112 2026-09-17T17:50:11+00:00 window AFTER edit_rc 0 arm 16 category test_failure test_secs 21.5 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE dashboard 5379 2026-09-13T02:19:04+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 152.5 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE scripts 5379 2026-09-14T04:11:42+00:00 window BEFORE edit_rc 1 arm 8 category test_failure test_secs 428.8 test_rc 1 timed_out False has_-n False failed_labels ['test']
MERGE scripts 5557 2026-09-19T01:25:26+00:00 window AFTER edit_rc 0 arm 16 category test_failure test_secs 569.0 test_rc 1 timed_out False has_-n False failed_labels ['test']
summaries seen 9
```

All 9 four-module summaries in the range match a merge verify, so all 9 are merge-role.

| window | module | task | summary stamp (UTC) | tree | arm | test leg s | test rc | category |
|---|---|---|---|---|---|---|---|---|
| BEFORE | scripts | 5021 | 09-10T23:03:48 | edit-absent | 8 | 424.4 | 1 | test_failure |
| BEFORE | scripts | 5379 | 09-14T04:11:42 | edit-absent | 8 | 428.8 | 1 | test_failure |
| BEFORE | scripts | 4917 | 09-14T13:43:19 | edit-absent | 8 | 493.4 | 1 | test_failure |
| BEFORE | dashboard | 5379 | 09-13T02:19:04 | edit-absent | 8 | 152.5 | 1 | test_failure |
| BEFORE | dashboard | 5025 | 09-13T19:06:34 | edit-absent | 8 | 299.5 | 1 | test_failure |
| BEFORE | dashboard | 5032 | 09-14T13:57:26 | edit-absent | 8 | 280.1 | 1 | test_failure |
| BEFORE | dashboard | 5024 | 09-15T01:21:02 | edit-absent | 8 | 256.1 | 1 | test_failure |
| AFTER, before junit | cockpit | 5112 | 09-17T17:50:11 | edit-present | 16 | 21.5 | 1 | test_failure |
| AFTER, before junit | scripts | 5557 | 09-19T01:25:26 | edit-present | 16 | **569.0** | 1 | test_failure |

None of these commands has `-n` in its command line. For dashboard, escalation and cockpit
that is expected, because 5408 put their `-n` in `pyproject.toml` addopts. For scripts,
whose `-n` 5408 put in the command itself, it is not: see (f).

**BEFORE per-module wall, failure-selected:**

- scripts: n = 3, median 428.8 s (424.4-493.4).
- dashboard: n = 4, median 268.1 s (152.5-299.5).
- escalation and cockpit: no merge-role record in BEFORE at all.

These walls are usable as module walls. No production test command or addopts carries `-x`,
`--exitfirst` or `--maxfail`, so a failed leg still ran its whole suite: 5024's log ends
`1 failed, 2389 passed, 1 xfailed in 250.03s`. Each is a single failed leg, though, and n is
3 and 4, selected by failing. Treat them as corroboration of the serial baseline in (c), not
as a baseline themselves. These summaries also age out of the archive from 2026-10-10 (§1(g)),
so this table is their durable copy.

### (c) Serial baseline, quoted

Two sources:

- Task 5408's description gives the serial merge-gate walls: **scripts 394-490 s,
  dashboard 172-238 s, escalation 85-89 s, cockpit 29-47 s**.
- The verify-speed study gives the matching serial worker-seconds. The source is
  `plans/verify-speed-study-df-2026-09-10/A-baseline.md`: §5a is arm A, 2026-08-19; §5a-bis is
  arm C, 2026-09-10; §4b is a hand-timed scripts run.

That study directory is **untracked and exists only at the main checkout**, so its figures are
quoted here rather than linked:

| module | serial suite wall s (§5a / §5a-bis) | serial worker-s (§5a / §5a-bis) | tests then (§5a-bis) |
|---|---|---|---|
| scripts | 489.75 pytest s, 499.8 s wall (§4b, `-p no:xdist`, 5180 passed) | — (no junit) | 5182 (§5a) |
| dashboard | 171.6 / 238.2 | 159.2 / 222.5 | 2229 |
| escalation | 89.3 / 84.7 | 78.7 / 74.2 | 1605 |
| cockpit | 29.0 / 47.3 | 26.0 / 43.5 | 346 |

§1(d) resolves a puzzle the study flagged in §5a-bis. That gate started at 2026-09-10T18:16:27Z
and was treated as arm C, `-n auto` = 8. But orchestrator's measured parallelism, 14.7, was
"consistent with ~16, not 8". The `config_reload` that applied 8 came at 19:06:50Z, 50 minutes
after that gate started. It ran at 16.

### (d) Signal check: "each under half their serial archive median"

The serial archive median is not recorded anywhere for these modules. The check uses half the
midpoint of 5408's serial range, and, more strictly, half its low end:

| module | serial range s | ½ × midpoint s | ½ × low end s | AFTER junit median s | vs ½ midpoint | vs ½ low end |
|---|---|---|---|---|---|---|
| scripts | 394-490 | 221.0 | 197.0 | 97.2 | **PASS** | **PASS** |
| dashboard | 172-238 | 102.5 | 86.0 | 64.2 | **PASS** | **PASS** |
| escalation | 85-89 | 43.5 | 42.5 | 36.4 | **PASS** | **PASS** |
| cockpit | 29-47 | 19.0 | 14.5 | 16.1 | **PASS** | FAIL, by 1.6 s |

Against the in-window BEFORE medians from (b), the halves are 214.4 s for scripts and 134.1 s
for dashboard. Both pass by a wide margin.

**Signal: met for scripts, dashboard and escalation; met for cockpit against the midpoint but
not against the low end.** Cockpit's test count has also grown 39% since the 09-10 serial figure,
from 346 to 482.

### (e) Worker-seconds are not CPU-seconds

Σ testcase `time` measures how long each test held a worker, in wall-clock time. On a loaded
host that time stretches with contention, and per-worker fixture setup adds to it. The study's
§5a-bis offered "not a clean worker-wall measure" as one possible explanation for its puzzle.
(c) shows the puzzle had a different cause, but the point stands: this is wall time per worker,
not CPU time. With that caveat, the AFTER/serial ratios are:

| module | AFTER worker-s (tests) | serial worker-s (tests), §5a-bis | ratio | ratio per test |
|---|---|---|---|---|
| dashboard | 608.1 (2627) | 222.5 (2229) | 2.73 | 2.32 |
| escalation | 211.4 (2069) | 74.2 (1605) | 2.85 | 2.21 |
| cockpit | 85.2 (482) | 43.5 (346) | 1.96 | 1.41 |
| scripts | 814.1 (6104) | ≈ 490 s serial wall (5180) | ≈ 1.7 | ≈ 1.4 |

Worker-occupied time rose about 1.4-2.3× per test under 16 workers. This is a proxy only. It
cannot tell more CPU apart from the same CPU spread thinner across contending workers. §4
measures real cgroup CPU-seconds.

### (f) The scripts leg went parallel at the 09-21 restart, not at T

The two halves of 5408's edit reached the merge gate at different times:

- **dashboard, escalation and cockpit** carry `-n auto` in `pyproject.toml` addopts. pytest reads
  that from the verified tree, so it took effect at T. 5112's cockpit merge leg on 09-17T17:50Z
  ran under xdist, and its archived log shows `bringing up nodes...` and a `[gw12]` worker.
- **scripts** carries `-n auto --dist loadgroup` in `scripts/orchestrator.yaml`'s
  `test_command`, and so do the shared and sampler `pyright --threads 8` type commands. The
  local merge gate does not read those from the verified tree. It reads the orchestrator's
  startup snapshot (§1(f), "restart"). No restart happened between 09-15T23:00:46Z and
  09-21T03:44:54Z, so between T and 09-21T03:44Z the scripts merge leg ran **serially** on
  edit-present trees:
  - 5557's merge verify (merge sha 43b02aa5f8, edit-present) has
    `git show 43b02aa5f8:scripts/orchestrator.yaml` ending `… --import-mode=importlib -n auto --dist loadgroup`.
  - Its archived summary records the command the gate actually ran, which has no `-n`:
    `uv run --project shared pytest tests/scripts/ scripts/tests/ --tb=short -q --timeout=300 --import-mode=importlib`.
  - Its log shows no worker lines and ends `1 failed, 5557 passed, 2 skipped, 10 deselected in 562.10s (0:09:22)`.

This is the startup-snapshot behaviour already recorded for task 1818 and in task 4536's
analysis. 5408's own description assumed the opposite ("module configs are re-discovered from
the worktree per verify … no reload, no restart"). That premise holds for its `pyproject.toml`
edits and not for its `orchestrator.yaml` edits.

The edit-present 16-arm gate cell from §2 therefore mixes two treatments. Append this to §2's
block and run it:

```python
R = datetime(2026, 9, 21, 3, 44, 54, 753602, tzinfo=timezone.utc)  # first event of run-ab649438402a
for label, keep in (('started before R', lambda s: s < R), ('started at or after R', lambda s: s >= R)):
    sel = [r for r in cells['edit', 16] if keep(datetime.fromisoformat(r['ts']) - timedelta(seconds=r['secs']))]
    xs = clean(sel)
    print(label, 'n', len(sel), 'clean', len(xs), 'median', round(statistics.median(xs)), 'p90', round(pct(xs, 0.9)),
          'first', sel[0]['ts'][:19], 'last', sel[-1]['ts'][:19])
```

```
started before R n 74 clean 34 median 3159 p90 4244 first 2026-09-17T13:05:54 last 2026-09-21T03:08:54
started at or after R n 68 clean 28 median 3108 p90 3816 first 2026-09-21T04:33:24 last 2026-09-24T06:11:07
```

No verify straddles R. Treatment part 1 alone (three small modules parallel, scripts serial) had
a clean median of 3,159 s (n = 34). The full treatment had 3,108 s (n = 28). That is −51 s, or
−1.6%. The scripts saving that (a) and (c) predict is about 490 − 97 ≈ 390 s, and it does not
show up at the gate at this n. The second period also carries the other 09-21 restart changes
and the 09-23 admission-slot increase (§1(f)). §5(e) takes this up.
