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

## 4. Controlled CPU-seconds and cgroup peak memory, serial vs parallel

Production records neither CPU-seconds nor peak memory (§1(g)), and the stop rule needs peak
memory. This section measures both directly, on this host, for each module 5408 changed.

**Host.** 32 cores and 125 GiB RAM, measured on 2026-10-05: `nproc`, `free -g`.

- The agent shell runs in `orchestrator-dark-factory.service`'s cgroup, whose peak covers the
  whole fleet. So each run gets its own `systemd-run --user --scope`, and memory and CPU are read
  from inside that scope.
- `systemd-run --user --wait` reports a bogus "Memory peak: 256.0K" here. The in-scope readout
  was probed first: a 150 MB child read `memory.peak=161898496`.
- Runs went from 01:45Z to 02:36Z. The 1-minute load average ran from 54 to 446 across them, and
  CPU PSI `some avg10` from 8% to 68%. The host was shared and loaded throughout.

**Tree.** The runs before 02:15Z (cockpit, escalation, dashboard, scripts serial) are at base
1498265b10. The orchestrator restarted at about 02:15Z and the branch was rebased, so later runs
are at base 54a84fba86. `git diff --stat 1498265b10 54a84fba86 -- tests/scripts scripts cockpit
dashboard escalation uv.lock pyproject.toml shared/pyproject.toml orchestrator/orchestrator.yaml
orchestrator/pyproject.toml` is empty. The ten files that differ are under `fused-memory/` and
`orchestrator/`, one of them a new orchestrator test file. So every A/B pair below ran on
identical module trees.

**Arms.** Both arms use each module's CURRENT `test_command`, so suite growth since 09-17 cancels
out and only 5408's edit differs.

| module | PARALLEL: the verbatim command, `PYTEST_XDIST_AUTO_NUM_WORKERS=16` | SERIAL: `verify_cmd.py::serial_pytest`, applied by hand |
|---|---|---|
| cockpit | `uv run --directory cockpit pytest tests/ --tb=short -q` | `… -q -p no:xdist -o addopts= -m 'not smoke'`. Re-adds the smoke deselect that `-o addopts=` drops |
| escalation | `uv run --directory escalation pytest tests/ --tb=short -q --timeout=300` | `… --timeout=300 -p no:xdist -o addopts=` |
| dashboard | `uv run --directory dashboard pytest tests/ --tb=short -q --timeout=300` | `… --timeout=300 -p no:xdist -o addopts=` |
| scripts | `uv run --project shared pytest tests/scripts/ scripts/tests/ --tb=short -q --timeout=300 --import-mode=importlib -n auto --dist loadgroup` | the same without `-n auto --dist loadgroup`, plus `-p no:xdist`. `--dist` without workers is a usage error |

**Wrapper.** Each run is one foreground call, `bash /tmp/5469-e1-run.sh <module> <arm> '<command>'`.
It does three things around the run:

- It strips the inherited venv the way `verify.py::_target_subprocess_env` does: no
  `VIRTUAL_ENV`, and no `.venv/bin` on `PATH`.
- It records `/proc/loadavg` and `/proc/pressure/cpu` before and after.
- It counts xdist workers from OUTSIDE the scope, so the counter's own CPU stays out of the
  scope's `cpu.stat`. pytest's `-q`, non-tty output never prints the worker count: xdist's
  `TerminalDistReporter.getstatus` prints only "bringing up nodes..." at verbosity < 0. So the
  sampler counts the scope's processes whose command line is execnet's worker bootstrap,
  `exec(eval(sys.stdin.readline()))`.

```bash
#!/bin/bash
# usage: bash /tmp/5469-e1-run.sh <module> <arm> '<verbatim command>'
# One (module, arm) run in its own systemd --user scope; prints the scope's own
# wall, memory.peak, pids.peak and cpu.stat, plus the peak xdist worker count.
cd /home/leo/src/dark-factory/.worktrees/5469 || exit 99
module=$1; arm=$2; cmd=$3
unit=e1-5469-$module-$arm
cg=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/$unit.scope
out=/tmp/5469-$module-$arm
clean_path=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.venv/bin$' | paste -sd:)
echo 0 > "$out.workers"
cat /proc/loadavg /proc/pressure/cpu > "$out.load-before"
# Sampler runs OUTSIDE the scope, so its own CPU is not in the scope's cpu.stat.
( peak=0; while sleep 2; do
    [ -d "$cg" ] || continue
    n=$(sed 's|.*|/proc/&/cmdline|' "$cg/cgroup.procs" 2>/dev/null | xargs -r grep -zlsF 'exec(eval(sys.stdin.readline()))' | wc -l)
    if [ "$n" -gt "$peak" ]; then peak=$n; echo "$peak" > "$out.workers"; fi
  done ) &
sampler=$!
env -u VIRTUAL_ENV PATH="$clean_path" PYTEST_XDIST_AUTO_NUM_WORKERS=16 \
  systemd-run --user --scope --quiet --unit="$unit" -p MemoryAccounting=yes \
  bash -c 's=$(date +%s.%N); eval "$1"; rc=$?; e=$(date +%s.%N); cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); echo "E1 rc=$rc wall=$(echo "$e - $s" | bc) memory.peak=$(cat $cg/memory.peak) pids.peak=$(cat $cg/pids.peak)"; grep -E "^(usage|user|system)_usec" $cg/cpu.stat; exit $rc' e1 "$cmd" \
  > "$out.log" 2>&1
rc=$?
kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
cat /proc/loadavg /proc/pressure/cpu > "$out.load-after"
echo "rc=$rc xdist_workers_max=$(cat "$out.workers")"
tail -6 "$out.log"
```

The first cockpit parallel run, `parallel-run1`, used an earlier sampler that read
`PYTEST_XDIST_WORKER` from `/proc/<pid>/environ`. That file shows only a process's initial
environment, and xdist sets the variable after the worker starts, so that sampler under-counted
(5). Cockpit parallel was re-run with the cmdline sampler and read 16. Both cockpit parallel runs
are reported.

**Comparator, the unchanged leg.** orchestrator's verbatim `test_command` at
`PYTEST_XDIST_AUTO_NUM_WORKERS=16`, run in the same wrapper behind `timeout 300`:
`bash /tmp/5469-e1-run.sh orchestrator bounded300 'timeout 300 uv run --directory orchestrator pytest tests/ --tb=short -q --timeout=300'`.
It stopped at 40% of the suite. Its `memory.peak` is therefore a **lower bound** on the
orchestrator leg's full peak.

**Other legs.** The plan's peak argument assumed 5408 changed only the four test legs. It also
added `pyright --threads 8` to the shared and sampler type legs. And within a module, the gate
runs the test, lint and type legs side by side. So two more kinds of run were made with
`/tmp/5469-e1-legs.sh`:

- for the four test-leg modules, their unchanged lint and type legs together (`typelint`);
- for shared and sampler, all three legs together, exactly as the gate runs that module
  (`module`).

```bash
#!/bin/bash
# usage: bash /tmp/5469-e1-legs.sh <module> <arm> '<lint>' '<type>' ['<test>']
# Runs a module's legs CONCURRENTLY in one scope, the way the gate runs them.
lint=$3; type=$4; test=${5:-}
if [ -n "$test" ]; then
  cmd="($lint) > /dev/null 2>&1 & lp=\$!; ($type) > /tmp/5469-$1-$2.type.log 2>&1 & tp=\$!; ($test); xr=\$?; wait \$lp; lr=\$?; wait \$tp; tr=\$?; echo legs test_rc=\$xr lint_rc=\$lr type_rc=\$tr; [ \$xr -eq 0 ] && [ \$lr -eq 0 ] && [ \$tr -eq 0 ]"
else
  cmd="($lint) > /dev/null 2>&1 & lp=\$!; ($type) > /tmp/5469-$1-$2.type.log 2>&1; tr=\$?; wait \$lp; lr=\$?; echo legs lint_rc=\$lr type_rc=\$tr; [ \$lr -eq 0 ] && [ \$tr -eq 0 ]"
fi
exec bash /tmp/5469-e1-run.sh "$1" "$2" "$cmd"
```

The commands are each module's verbatim `lint_command` and `type_check_command` from
`<module>/orchestrator.yaml` (and `test_command`, for `module`). For example:

`bash /tmp/5469-e1-legs.sh shared module "uv run --directory shared ruff check src/ tests/ && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests" "uv run --directory shared pyright --threads 8 src/ tests/" "uv run --directory shared pytest tests/ --tb=short -q --timeout=300"`

**Readout.** The table is parsed from the logs by:

```python
import re
from pathlib import Path
RUNS = [('cockpit', 'serial'), ('cockpit', 'parallel'), ('cockpit', 'parallel-run1'),
        ('escalation', 'serial'), ('escalation', 'parallel'),
        ('dashboard', 'serial'), ('dashboard', 'parallel'),
        ('scripts', 'serial'), ('scripts', 'parallel'), ('orchestrator', 'bounded300'),
        ('scripts', 'typelint'), ('dashboard', 'typelint'), ('escalation', 'typelint'),
        ('cockpit', 'typelint'), ('shared', 'module'), ('sampler', 'module')]
OUTCOMES = ('passed', 'failed', 'skipped', 'xfailed', 'xpassed', 'error', 'errors')
for module, arm in RUNS:
    base = Path(f'/tmp/5469-{module}-{arm}')
    log = base.with_suffix('.log').read_text() if base.with_suffix('.log').exists() else Path(f'{base}.log').read_text()
    e1 = dict(re.findall(r'(rc|wall|memory\.peak|pids\.peak)=(\S+)', log.split('E1 ', 1)[1].splitlines()[0]))
    cpu = dict(re.findall(r'^(usage|user|system)_usec (\d+)$', log, re.M))
    summary = [l for l in log.splitlines() if re.search(r' in [0-9.]+s', l) and re.search(r'\d+ (passed|failed)', l)]
    total = sum(int(n) for n, k in re.findall(r'(\d+) (\w+)', summary[-1]) if k in OUTCOMES) if summary else None
    loads = [Path(f'{base}.load-{w}').read_text().split()[0] for w in ('before', 'after')]
    workers = Path(f'{base}.workers').read_text().strip()
    print(f"{module:12} {arm:13} rc={e1['rc']:>3} wall={float(e1['wall']):7.1f} "
          f"cpu={int(cpu['usage'])/1e6:7.1f} (u {int(cpu['user'])/1e6:6.1f} s {int(cpu['system'])/1e6:6.1f}) "
          f"peak_MiB={int(e1['memory.peak'])/2**20:7.0f} pids={e1['pids.peak']:>3} workers={workers:>2} "
          f"outcomes={total} load={loads[0]}->{loads[1]}")
```

```
cockpit      serial        rc=  0 wall=   59.7 cpu=   32.3 (u   28.1 s    4.2) peak_MiB=    327 pids= 10 workers= 0 outcomes=503 load=111.41->132.30
cockpit      parallel      rc=  0 wall=   14.6 cpu=   63.6 (u   59.6 s    3.9) peak_MiB=   1102 pids= 70 workers=16 outcomes=503 load=98.79->97.90
cockpit      parallel-run1 rc=  0 wall=   16.8 cpu=   64.5 (u   59.2 s    5.3) peak_MiB=   1117 pids= 70 workers= 5 outcomes=503 load=146.56->138.89
escalation   serial        rc=  0 wall=  142.5 cpu=  104.5 (u   96.1 s    8.4) peak_MiB=    373 pids= 38 workers= 0 outcomes=2159 load=104.13->71.45
escalation   parallel      rc=  0 wall=   40.4 cpu=  213.9 (u  196.7 s   17.2) peak_MiB=   2661 pids= 91 workers=16 outcomes=2159 load=68.72->78.25
dashboard    serial        rc=  0 wall=  294.3 cpu=  211.0 (u  150.2 s   60.8) peak_MiB=    634 pids=145 workers= 0 outcomes=3129 load=80.03->397.35
dashboard    parallel      rc=  1 wall=   72.7 cpu=  429.8 (u  319.2 s  110.5) peak_MiB=   2553 pids=225 workers=16 outcomes=3129 load=445.81->266.73
scripts      serial        rc=  0 wall=  789.8 cpu=  474.5 (u  340.6 s  133.9) peak_MiB=   1106 pids=140 workers= 2 outcomes=7697 load=85.93->127.32
scripts      parallel      rc=  0 wall=   99.1 cpu=  685.1 (u  555.6 s  129.4) peak_MiB=   4604 pids=418 workers=16 outcomes=7697 load=73.80->75.89
orchestrator bounded300    rc=124 wall=  300.3 cpu= 2038.3 (u 1691.2 s  347.1) peak_MiB=  10429 pids=709 workers=17 outcomes=None load=62.61->86.57
scripts      typelint      rc=  0 wall=   56.3 cpu=   65.9 (u   63.2 s    2.7) peak_MiB=    990 pids= 43 workers= 0 outcomes=None load=53.80->97.82
dashboard    typelint      rc=  0 wall=   38.7 cpu=   48.3 (u   46.4 s    1.9) peak_MiB=    764 pids= 52 workers= 0 outcomes=None load=135.49->144.90
escalation   typelint      rc=  0 wall=   30.3 cpu=   33.7 (u   31.9 s    1.8) peak_MiB=    723 pids= 42 workers= 0 outcomes=None load=144.90->150.31
cockpit      typelint      rc=  0 wall=   14.2 cpu=   17.0 (u   16.1 s    0.9) peak_MiB=    377 pids= 50 workers= 0 outcomes=None load=147.42->147.67
shared       module        rc=  0 wall=  231.6 cpu=  319.7 (u  287.2 s   32.6) peak_MiB=   2998 pids= 75 workers= 0 outcomes=6766 load=149.36->42.46
sampler      module        rc=  0 wall=   11.6 cpu=   26.5 (u   22.6 s    3.9) peak_MiB=    912 pids= 74 workers= 0 outcomes=142 load=147.67->150.54
```

`wall` is the scope's own clock around the command. `cpu` is `cpu.stat` `usage_usec` / 1e6,
with user and system. `peak_MiB` is `memory.peak` / 2^20. `outcomes` sums passed, failed,
skipped, xfailed, xpassed and errors from pytest's last summary line.

### Validity checks

- **Same collected outcome total in both arms, for all four modules.** cockpit 503 / 503,
  escalation 2,159 / 2,159, dashboard 3,129 / 3,129, scripts 7,697 / 7,697. The serial lines also
  report deselected items (cockpit 4, scripts 10); the xdist controller does not report
  deselection, and deselected items are not outcomes.
- **Parallel arms ran 16 xdist workers.** The sampler read 16 for cockpit (re-run), escalation,
  dashboard and scripts. Every parallel log contains xdist's "bringing up nodes..." and no serial
  log does.
  - The sampler counts every process in the scope that carries execnet's bootstrap command
    line. Some scripts tests spawn their own xdist subprocesses, which is why scripts SERIAL reads
    2 with no xdist in the run itself. The orchestrator comparator's 17 is the same effect.
- **One arm failed, and it is reported.** dashboard PARALLEL, rc 1:
  `1 failed, 3127 passed, 1 xfailed`.
  - The failing test is `tests/test_scheduler_page.py::test_evict_park_endpoint_rejects_invalid_body[missing-project_root]`,
    on worker gw1. Its assertion reads "Unexpected MCP tool call(s) on 400 path:
    {'get_curator_state'}", and the call list includes a background metrics sampler's
    `get_queue_stats` / `get_curator_state` calls.
  - The serial arm, run just before, passed the same test.
  - `flake_occurrence` has no row for this test id, so it has not yet shown up at the merge gate.
  - This is the order- and timing-dependence risk 5408 named. It is filed as a low-priority
    follow-up, ticket `tkt_0RVE0WBY3YQM9CEX9DFZ1PWAPR`.
  - The failed arm still ran the whole suite, so its wall, CPU and peak stand.

### Per-module result, parallel ÷ serial

| module | wall s, serial → parallel | ratio | CPU-s, serial → parallel | ratio | peak MiB, serial → parallel | ratio |
|---|---|---|---|---|---|---|
| cockpit | 59.7 → 14.6 (run1 16.8) | 0.24 | 32.3 → 63.6 (run1 64.5) | 1.97 | 327 → 1,102 (run1 1,117) | 3.37 |
| escalation | 142.5 → 40.4 | 0.28 | 104.5 → 213.9 | 2.05 | 373 → 2,661 | 7.13 |
| dashboard | 294.3 → 72.7 | 0.25 | 211.0 → 429.8 | 2.04 | 634 → 2,553 | 4.03 |
| scripts | 789.8 → 99.1 | 0.13 | 474.5 → 685.1 | 1.44 | 1,106 → 4,604 | 4.16 |
| **Σ four modules** | **1,286.3 → 226.8 (−1,059.5 s)** | **0.18** | **822.3 → 1,392.4 (+570.1 CPU-s)** | **1.69** | — | — |
| comparator: orchestrator, first 300 s | — | — | 2,038.3 in 300 s | — | **≥ 10,429** (lower bound) | — |

**CPU-seconds.** The four suites take 1.4-2.1× the CPU when run at 16 workers. Each 16-worker
process imports and sets up its fixtures separately. That is real CPU spent per gate, which
3353's ruling makes a binding objective. On this host and load:

- each edit-present merge gate runs the four suites about 1,060 s faster;
- it spends about 570 more CPU-seconds doing so;
- 570 CPU-s is 28% of what the orchestrator suite burns in its first 300 s alone (2,038 CPU-s).

These ratios agree with §3(e)'s worker-seconds proxy, 1.4-2.3× per test.

### Gate-peak argument

`merge_verify_max_concurrent_modules: 1`, unchanged since 2026-07-19, means the gate runs one
module at a time. The gate's peak is therefore the largest module peak. A module's peak is at
most the sum of its concurrent legs' peaks. The orchestrator module is untouched by 5408, and its
test leg alone reached **≥ 10,429 MiB** in its first 300 s, in both arms. The changed modules,
after the edit, at most:

| module | after-edit legs | module peak upper bound, MiB | vs the orchestrator lower bound |
|---|---|---|---|
| scripts | test 4,604 (parallel) + type+lint 990 | ≤ 5,594 | below |
| escalation | test 2,661 + type+lint 723 | ≤ 3,384 | below |
| dashboard | test 2,553 + type+lint 764 | ≤ 3,317 | below |
| cockpit | test 1,102 + type+lint 377 | ≤ 1,479 | below |
| shared | all three legs measured together, with `pyright --threads 8` | 2,998 | below |
| sampler | all three legs measured together, with `pyright --threads 8` | 912 | below |

**Every changed module's after-edit upper bound is below the unchanged orchestrator module's
lower bound.** The largest module peak in a gate is orchestrator's in both arms, so 5408 moved the
gate's cgroup peak by **0%**, far inside the stop rule's +25%. This comes from two numbers measured
on the same host on the same day. The 0% is a structural result. It rests on the one-module-at-a-time
setting, and it would no longer hold if `merge_verify_max_concurrent_modules` were raised above 1.

### Caveats

- `memory.peak` includes page cache charged to the scope, so it over-states anonymous memory,
  more so for the serial arms. The comparison runs in the safe direction: the upper bounds are
  inflated, not the orchestrator lower bound.
- There is one repetition per arm (two for cockpit parallel), on a shared host whose load swung
  from 54 to 446. Serial and parallel ran back to back, which limits drift between the arms of a
  pair but does not remove it. Absolute walls are not comparable with other days. Scripts serial
  took 790 s here for 7,697 outcomes, against 490 s for 5,182 tests in the 09-10 study, which
  reflects both suite growth and load. CPU-seconds are much less affected by load than wall is.
- The wrapper sets only `PYTEST_XDIST_AUTO_NUM_WORKERS` from production's `verify_env`. It does
  not set `DF_REQUIRE_SANDBOX_TESTS=1`, and the agent shell runs inside the agent sandbox. A
  controlled run is not production: production gates also run with other verifies and agents
  beside them.

## 5. Stop rule, verdict, caveats, follow-ups

The stop rule, from 5408 WORK item 4: "verify timeouts above the 4% baseline or gate cgroup peak
RSS +25% → revert the addopts".

### (a) Timeout half, from production

A merge-role verify timeout can show up in two places:

- the `[category: …]` tag in a `merge_finalized` or `merge_blocked` reason. The category is a tag
  inside the reason string; it is not a separate field;
- an archived summary whose top-level or per-command `timed_out` is true, or whose category is
  `infra_timeout`. The summary is kept only if it matches a `merge_verify` row by §3's rule.

Each source is keyed to the tree of the verify it belongs to. A reason event is matched to the
verify of the same task that ended at most 600 s earlier. The same block also counts the
peak-memory proxies for (b). Run from the worktree root:

```python
import glob, json, re, sqlite3, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, 'scripts')
from verify_budget_census import parse_record_path
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
EDIT = 'b39a4f57caa4818967fbf9ccb9c1556891a8fed5'
T = datetime(2026, 9, 17, 6, 30, 1, 881486, tzinfo=timezone.utc)
LO, HI = T - timedelta(days=7), T + timedelta(days=7)
con = sqlite3.connect(DB, uri=True)

verifies, cells = defaultdict(list), Counter()
for ts, task, data in con.execute("SELECT timestamp, task_id, data FROM events WHERE event_type = 'merge_verify' "
                                  "AND timestamp >= ? AND timestamp < ?", (LO.isoformat(), HI.isoformat())):
    d = json.loads(data); end = datetime.fromisoformat(ts)
    edit = {0: 'edit', 1: 'no-edit'}[subprocess.run(['git', 'merge-base', '--is-ancestor', EDIT, d['merge_sha']]).returncode]
    cell = ('BEFORE' if end < T else 'AFTER', edit)
    verifies[task].append(dict(start=end - timedelta(milliseconds=d['duration_ms']), end=end, cell=cell))
    cells[cell] += 1

def verify_for(task, at, slack):
    hits = [v for v in verifies[task] if v['start'] <= at <= v['end'] + slack]
    return hits[0] if len(hits) == 1 else None

# Source 1: the category tag in merge failure reasons, matched to the verify that just ended.
reason_cats, worded = defaultdict(Counter), Counter()
OOM = re.compile(r'out of memory|OOM|oom_kill|MemoryError|SIGKILL|signal 9|rc[=: ]+-?(137|9)\b|exit (code|status) -?(137|9)\b|Killed')
for ts, task, data in con.execute("SELECT timestamp, task_id, data FROM events WHERE event_type IN ('merge_finalized', 'merge_blocked') "
                                  "AND timestamp >= ? AND timestamp < ?", (LO.isoformat(), HI.isoformat())):
    reason = json.loads(data).get('reason') or ''
    m = re.search(r'\[category: (\w+)\]', reason)
    worded['reasons'] += bool(reason)
    worded['oom/kill wording'] += bool(OOM.search(reason))
    if re.search(r'(?i)timed out|timeout', reason):
        worded[f"timeout wording, category {m[1] if m else None}"] += 1
    if m:
        v = verify_for(task, datetime.fromisoformat(ts), timedelta(seconds=600))
        reason_cats[v['cell'] if v else 'unmatched'][m[1]] += 1
print('reason categories:', {k: dict(c) for k, c in reason_cats.items()})
print('reason wording:', dict(worded))

# Source 2: archived summaries of merge-role legs (matched as in §3), every module.
flags, kept, skipped = defaultdict(list), Counter(), 0
for p in sorted(glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/*.summary-*.json')):
    rec = parse_record_path(Path(p))
    if rec is None:
        continue
    stamp = datetime.strptime(rec.archived_at.split('_')[0].rstrip('Z'), '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    if not LO <= stamp < HI:
        continue
    v = verify_for(rec.task_id, stamp, timedelta(seconds=300))
    if v is None:
        skipped += 1
        continue
    d = json.load(open(p)); cmds = d.get('commands') or []
    kept[v['cell']] += 1
    timed_out = bool(d.get('timed_out')) or any(c.get('timed_out') for c in cmds) or d.get('category') == 'infra_timeout'
    killed = d.get('category') == 'infra_kill' or any(c.get('rc') in (-9, 137) for c in cmds) or d.get('rc') in (-9, 137)
    if timed_out or killed:
        flags[v['cell']].append((rec.task_id, rec.module_prefix, stamp.isoformat(), d.get('category'), d.get('rc'),
                                 'timed_out' if timed_out else 'killed'))
print('merge-role summaries kept per cell:', dict(kept), 'not merge-role:', skipped)
print('timeout/kill summaries:', dict(flags))
for cell in sorted(cells):
    timeouts = reason_cats[cell]['infra_timeout'] + sum(1 for f in flags[cell] if f[-1] == 'timed_out')
    print(cell, 'merge_verify rows', cells[cell], 'timeouts', timeouts, f'rate {timeouts / cells[cell]:.1%}')
```

```
reason categories: {('BEFORE', 'no-edit'): {'unknown_test_failure': 6, 'test_failure': 27}, ('AFTER', 'edit'): {'test_failure': 26, 'unknown_test_failure': 2}}
reason wording: {'reasons': 154, 'oom/kill wording': 0, 'timeout wording, category test_failure': 3}
merge-role summaries kept per cell: {('AFTER', 'edit'): 34, ('BEFORE', 'no-edit'): 35} not merge-role: 6
timeout/kill summaries: {('AFTER', 'edit'): [('4427', 'orchestrator', '2026-09-21T00:01:43+00:00', 'infra_kill', -9, 'killed')]}
('AFTER', 'edit') merge_verify rows 146 timeouts 0 rate 0.0%
('BEFORE', 'edit') merge_verify rows 1 timeouts 0 rate 0.0%
('BEFORE', 'no-edit') merge_verify rows 111 timeouts 0 rate 0.0%
```

- **No `infra_timeout` tag** appears in any of the 154 merge failure reasons in either window.
  The tags present are `test_failure` and `unknown_test_failure`. One failed verify can tag both
  its `merge_finalized` and its `merge_blocked` event, so the tag counts are not verify counts.
- Three reasons contain timeout wording, and all three are tagged `test_failure`. The wording is
  in test names or assertion text, not a verify clock:
  `TestRow11TimeoutMargin::…`, a `CHAIN_BUILD_TIMEOUT_SECS` set member, and
  "(crash, not a timeout)".
- **No merge-role summary** has `timed_out` true or category `infra_timeout`. That covers 35 in
  the edit-absent cell and 34 in the edit-present cell, across all modules. So the four 5408
  modules had **0 timeouts each** in both cells.

| tree | `merge_verify` rows | verify timeouts | rate | vs 4% baseline |
|---|---|---|---|---|
| edit-absent (all in BEFORE) | 111 | 0 | 0.0% | below |
| edit-present (146 AFTER + 5408's own) | 147 | 0 | **0.0%** | below. With 0 of 147, the one-sided 95% upper bound ("rule of three") is 3 / 147 = 2.0%, also below 4% |

The 4% baseline comes from the verify-speed study's `A-baseline.md` §3, quoted because that
directory is untracked: "Categories over the same corpus: passed 392, test_failure 48,
unknown_test_failure 37, **infra_timeout 19**, infra_kill 1. So ~4 % of task-leg verifies die on
the clock". That is 19 / 497 = 3.8%, over the 30 days to 2026-09-10. It is weaker as a
comparator than it looks, for two reasons:

- it is a **task-leg** rate, and this trial is merge-role;
- it predates the 2026-09-12 fleet budget doubling, 3600 → 7200 s warm (§1(f)), which applies to
  both of this trial's arms almost entirely.

A 0-against-0 reading under a doubled budget says little about headroom. It does say the half of
the stop rule that the data can test did not trip.

### (b) Peak-RSS half

**From the controlled run (§4): 0%.** The gate runs one module at a time. The unchanged
orchestrator test leg reached at least 10,429 MiB. Every after-edit upper bound for a changed
module is below that, the largest being scripts at ≤ 5,594 MiB. So the gate's peak, which is
orchestrator's, did not move.

That is a bound, not a production measurement. One production data point corroborates it: task
5415's record (2026-09-30) quotes a single full DF merge gate on the laptop, with cgroup scopes
on, `merge_verify_max_concurrent_modules: 1` and 16 workers. It peaked at "13.3 GiB, in the
orchestrator pytest leg; fused-memory was second at 12.6 GiB". Neither module is one 5408
changed.

**Production proxies, per tree, from the same block:**

- **No merge failure reason** mentions OOM, `MemoryError`, `SIGKILL`, signal 9, rc 137 / −9, or
  `Killed`, in either window.
- **One merge-role summary is a kill: 4427, orchestrator leg, edit-present.**
  - Summary: `attempt-1.orchestrator.summary-20260921T000143_884680Z.json`, category
    `infra_kill`, test rc −9 after 1,674.8 s. Its lint and type legs passed. The log ends in an
    xdist `OSError: cannot send (already closed?)` at `pytest_sessionfinish`.
  - `journalctl -k --since '2026-09-20 23:55:00 UTC' --until '2026-09-21 00:05:00 UTC'` is
    readable for that window and contains no OOM-killer line.
  - The verify's attempt 1 then failed on a real branch test (`fails_in_isolation`).
  - The killed leg belongs to the unchanged orchestrator module, not to one of 5408's four.
  - Hypothesis: the SIGKILL came from somewhere other than the kernel OOM killer. This was not
    investigated further.
- The edit-absent cell has no kill.

### (c) Verdict

Rule (5408 WORK item 4): **REVERT iff** the edit-present timeout rate exceeds 4%, **or** the gate
cgroup peak rises by more than 25%.

| half | measured how | result | trips? |
|---|---|---|---|
| timeouts | **from production**: every merge verify in both windows, keyed by tree | 0 / 147 edit-present (≤ 2.0% at 95%), against 0 / 111 edit-absent | no |
| gate peak | **bounded by the controlled run (§4)**, corroborated by 5415's laptop gate | +0%: the gate peak stays orchestrator's | no |

**Verdict: KEEP the addopts.** Neither half trips, so no revert escalation is filed.

### (d) Expected signal

From §3(d): each of scripts, dashboard and escalation runs under half its serial median, with
AFTER medians of 97.2, 64.2 and 36.4 s. Cockpit (16.1 s) is under half the serial range's
midpoint (19.0 s), but not under half its low end (14.5 s).

### (e) Attribution, against 5204's caveat

| quantity | seconds | source |
|---|---|---|
| naive gate delta, AFTER − BEFORE clean medians | −1,019 (4188 → 3169) | §2 (i) |
| what the arm switch alone predicts | −965 (4224 → 3259, ÷ 1.296) | §2, 5204 |
| residual left for 5408 and the other confounds | **−123** (3136 against 3259) | §2 |
| residual from the restart split: full treatment against part 1 only | −51 (3159 → 3108) | §3(f) |
| sum of per-module savings, serial midpoint − AFTER junit median | **−558** (scripts 345, dashboard 141, escalation 51, cockpit 22) | §3(a), (c) |
| the same four suites in the controlled run, serial − parallel | −1,060 (at that day's load) | §4 |

- **The naive −24% is not 5408's.** The arm switch alone accounts for about 965 of its 1,019 s.
- 5408's effect on each module's wall is large and directly measured. Its effect on the whole
  gate is small: about −123 s, or −4%, against −558 s expected from the per-module savings.
- Three hypotheses for the gap, none tested here:
  - gate-wall noise. In every cell with at least 28 clean rows, the p90 sits 700-1,100 s above
    the median, and the arm-matched comparison cells have n = 5 and n = 3;
  - host contention. The parallel legs add about 570 CPU-seconds per gate (§4) on a host shared
    with task verifies and agents;
  - the 1.296 arm ratio may not carry over unchanged from edit-absent trees, measured 09-08 to
    09-16, to edit-present trees.
- The scripts leg, the largest expected saving at 345 s, only went parallel on 09-21 (§3(f)).
  Comparing the periods before and after that restart showed −51 s.

### (f) Caveats

- **Junit coverage gap.** There is no per-module junit for all of BEFORE or for AFTER's first
  four days (09-17 to 09-21). AFTER's per-module walls come from 09-21 to 09-24 only.
- **BEFORE's per-module walls rest on failure-selected samples**: n = 3 for scripts and n = 4 for
  dashboard, with none for escalation or cockpit. The serial baseline is quoted from the 09-10
  study, not measured in-window.
- **Worker-seconds are not CPU-seconds** (§3(e)). Real CPU-seconds come only from §4's
  controlled run.
- **§4 is a single controlled repetition** per arm (two for cockpit parallel), on a loaded,
  shared host. Its peaks include page cache.
- **Confounds** (§1(f)), inside the AFTER window:
  - the arm switch;
  - the 09-21 restart, which deployed five days of process code together with 5408's own
    `orchestrator.yaml` half;
  - 5442's per-test timeout raise;
  - the 09-23 admission-slot increase.
- **The CPUWeight / cgroup cpu.weight mechanism does not apply** to these windows. DF verifies ran
  unscoped inside `orchestrator-dark-factory.service`'s cgroup, because
  `verify_use_cgroup_scope` is off, so no per-role CPU weighting separated the gate from agents
  or task verifies.
- **The gate-peak result is structural.** It holds while `merge_verify_max_concurrent_modules`
  is 1, and it would need re-measuring if that is raised.

### (g) Follow-ups

- **Production peak RSS** is owned by task **5206** (enable DF verify cgroup scopes) and task
  **5415** (measure `memory.peak` of merge-verify scopes over 7 days). Both are pending. Nothing
  new is filed for RSS.
- **Per-verify CPU-seconds.** `search_tasks` found no existing owner: 3353 stamps load and
  worker count, 5671 role and slot wait, 5651 plan hash and outcomes. So one follow-up was filed:
  stamp `cpu.stat` usage, user and system CPU-seconds (and `memory.peak`, once 5206 lands) on
  each verify command's summary. Ticket **`tkt_0RVE1CW1SX7JWX6JXAJPNZYFK9`**, low priority,
  `suggestion_hash` `e1-cpu-seconds-per-verify-leg`.
- **The dashboard parallel failure** in §4: ticket **`tkt_0RVE0WBY3YQM9CEX9DFZ1PWAPR`**, low
  priority. `test_evict_park_endpoint_rejects_invalid_body[missing-project_root]` sees a
  background metrics sampler's MCP calls under 16 workers.
- **The startup-snapshot behaviour** in §3(f), where a module `orchestrator.yaml` edit reaches the
  local merge gate only at the next restart, is already recorded (task 1818's incident, task
  4536's analysis). It is a known property, not a new defect, so nothing is filed. A future
  before/after trial of an `orchestrator.yaml` edit must key by restart, not by git ancestry.
