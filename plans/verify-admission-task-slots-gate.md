# verify_admission_task_slots — gate history

This file holds the gate history for one operator lever,
`verify_admission_task_slots` in `dark-factory-orchestrator.yaml` (the
TASK-ROLE VERIFY ADMISSION WIDTH block directly above the key). The code
default is 1 (`orchestrator/src/orchestrator/config.py::OrchestratorConfig`).
The knob is green-tier (`config.py::RELOADABLE_FIELDS`), so it applies through
`reload_config` without a restart.

Value history: 1 -> 3 on 2026-09-03, -> 1 on 2026-09-08, -> 2 on 2026-09-23,
-> 1 on 2026-09-27 (task 5797, §2026-09-27 below).
Each later re-measurement appends a dated section below and reuses the
Reproduction commands.

---

## 2026-09-27 — task 5797: 48h gate on the 1 -> 2 raise

### 1. What is being judged

Leo's ruling of 2026-09-23: *"Try B, both monitor the situation directly and
file a 48h delayed milestone to check the impact and revert if it appears to be
negative."* The yaml clause he approved reads: *"if median task duration or
timeouts/day worsen, put this back to 1."*

The change under test is commit `c4359d1f2e` (`config: verify_admission_task_slots
1 -> 2`), amended by `ca06d49340`. It went live at the runs.db `config_reload`
event **id 420916, 2026-09-23T11:53:28.101Z**, which applied
`verify_admission_task_slots {old: 1, new: 2}` with `restart_required {}`. No
later `config_reload` touches the key (checked through id 429223).

The raise rested on four premises: 24/24 concurrency slots occupied with 20
tasks parked at `phase_enter verify`; merges/day rising from 9.71 to 16.29;
merge verify falling from 64.3 to 55.2 min mean (~49 min on the day); and a
backlog growing by +16.6 pending/day.

### 2. Windows

| Name | Window (UTC, half-open) | Length | Slots |
|---|---|---|---|
| Pivot P | 2026-09-23T11:53:28Z (config_reload 420916) | — | — |
| Measurement instant T | 2026-09-27T19:56:51Z; main at `3cae0d9d94` | — | — |
| **PRE** (primary control) | 2026-09-19T03:50:05Z..2026-09-23T11:53:28Z | 4.336 d | 1 |
| **POST** | 2026-09-23T11:53:28Z..2026-09-27T19:56:51Z | 4.336 d | 2 |
| SENS (sensitivity pre-arm) | 2026-09-12T07:24:38Z..2026-09-23T11:53:28Z | 11.187 d | 1 |
| EP1 (calibration) | 2026-08-27T00:00:00Z..2026-09-03T15:58:07Z | 7.67 d | 1 |
| EP3 (calibration) | 2026-09-03T15:58:07Z..2026-09-08T10:18:07Z | 4.76 d | 3 |

PRE has the same length as POST and ends at the pivot, so every day of it ran
at 1 slot. SENS starts at commit `36c4c71eb4`, which raised the warm verify
wall 3600 -> 7200s, so SENS is the whole 1-slot period under today's wall.
EP1/EP3 are the 09-03 episode's arms, cut at config_reload 383043 (1 -> 3) and
395705 (3 -> 1). They show whether an instrument registers a regression that is
already known.

A leg or event is placed in an arm by its start timestamp (census) or its
emission timestamp (runs.db).

### 3. Before / after

Clean p50 is the census's figure: timed-out and failed legs are excluded,
because their duration is not the suite's. The ">= 3600s share" is taken over
ALL selected legs (clean + failed + timed out), which is like-for-like with the
09-03 "fraction at the wall" framing.

| Measure | Source | PRE (1 slot) | POST (2 slots) | Δ | Regressed? |
|---|---|---|---|---|---|
| M1a task-leg clean p50 / p90, orchestrator suite | verify-summary census | 3233s / 4995s (n=14 clean, 19 failed, 0 t/o) | 4995s / 6366s (n=29 clean, 20 failed, 0 t/o) | p50 ×1.55 | **yes — R1** |
| M1a share of legs >= 3600s | census corpus, inline snippet | 27.3% (9/33) | 63.3% (31/49) | +36.0 pp | **yes — R2** |
| M1a share >= 7200s / timed out | census corpus | 0% / 0% | 2.0% (1 leg, clean, cold) / 0% | — | no |
| M1b rebase-verify p50 / p90 (all modules) | runs.db `rebase_verify_cost` | 222s / 3796s (n=100) | 633s / 5856s (n=128) | p50 ×2.85 | **yes — R1** |
| M1b share >= 3600s / >= 7200s | runs.db `rebase_verify_cost` | 11.0% / 0% | 28.1% / 0% | +17.1 pp | **yes — R2** |
| M2 merge-verify p50 (by runner, local) | `merge_lane_throughput.py` | 51.5 min (n=91) | 54.7 min (n=86) | ×1.06 | no (R4 needs ×1.15) |
| M2 merge-verify mean | runs.db `merge_verify` `$.duration_ms` | 50.8 min (n=91) | 56.2 min (n=86) | ×1.11 | context |
| M3 timeouts/day (primary) | census corpus, every `commands[]` entry, all modules | 0.00 (0 of 288 entries) | 0.00 (0 of 479 entries) | 0 | no |
| M3 timeouts/day (secondary) | journal, `Command timed out after` | 0.00 (09-19..09-23: 0,0,0,0,0) | 0.46 (09-23..09-27: 0,2,0,0,0) | +2 | **yes — R3, secondary only** |
| M4 backlog direction | burndown.db measured rows | +19.7 pending/day, +18.9 done/day (5 rows, 30.5 h of 104 h) | −24.9 pending/day, +36.0 done/day (53 rows, 8.7 h of 104 h) | — | NOT COMPARABLE (see below) |
| M5 in-verify depth, hourly p50 / max | runs.db phase events | 17 / 22 (excl. >24h-stale: 14 / 19) | 16 / 23 (excl. >24h-stale: 13 / 23) | −1 | no |
| M5 verify-phase dwell p50 / p90 | runs.db phase events | 5.04h / 19.73h (n=201) | 4.43h / 16.01h (n=194) | ×0.88 | no (improved) |
| M5 verify-phase exits/day | runs.db phase events | 46.1 | 48.4 | +5.0% | no |
| Throughput: workflow_verify/day | runs.db | 41.7 (181) | 43.8 (190) | +5.0% | context |
| Throughput: landings/day (`merge_finalized` state=done) | runs.db | 17.1 (74) | 14.1 (61) | −17.6% | context |
| Throughput: first-parent merges on main/day | git | 15.2 (66) | 12.9 (56) | −15.2% | context |
| Merge-lane queue wait p50 / lead-time p50 | `merge_lane_throughput.py` | 14.4 / 96.2 min | 57.7 / 126.2 min | — | context |
| Merge-lane host occupancy (LOCF) | `merge_lane_throughput.py` | 74.4% | 78.1% | — | context |

Sensitivity arm (SENS, 1 slot, 11.2 d): M1a clean p50 3274s / p90 4594s (n=34
clean, 31 failed, 0 t/o), >= 3600s share 23.1% (15/65); M1b p50 230s, >= 3600s
share 7.3% (n=259); M2 mean 59.8 min, p50 57.1 min (n=209); workflow_verify
42.8/day; landings 14.0/day; first-parent merges 13.3/day. POST's regression
holds against SENS as well as against PRE.

Calibration (the 09-03 episode, same instruments): M1b p50 192s -> 439s (×2.29),
>= 3600s share 0.0% -> 11.6%. M1a clean p50 1509s (n=12) -> 3297s (n=8),
>= 3600s share 4.0% -> 15.1%, timed out 1 -> 7. Both instruments register the
known 3-slot regression. POST's M1b move (×2.85) is larger than the 3-slot
episode's (×2.29).

Literal anchors from the task text, all from the 09-03 era under a 3600s wall:
median ~2300s and ~10% at the wall. PRE already sits above both (3233s,
27.3%), so the anchors cannot serve as the baseline by themselves. POST is
further above both (4995s, 63.3%).

M1b within cohort, as a check that the rise is not a cohort-mix artifact:
continuous 145s -> 601s, big-jump 349s -> 816s. Both rise by 2.3–4.1x.

M4 in detail. There are no measured dark-factory snapshots between
2026-09-21T14:50:00Z (pending 1080, done 4002) and 2026-09-27T11:10:00Z
(pending 1193, done 4123); one `gap`-state row lies inside. Over the same
interval every other project has 432–460 measured rows. PRE's figures come from
the last 30.5 h before that hole, and POST's from the 8.7 h after it, so neither
arm is covered well enough to compare. Across the hole the backlog moved +19.3
pending/day and +20.7 done/day (5.85 d); that interval straddles the pivot and
cannot be split. The burndown gap is filed separately (see Status below).

### 4. Decision rule and verdict

The rule was pre-registered in the task plan before this measurement was
taken:

> REVERT to 1 if ANY of these holds (equal-length pre-arm vs post-arm).
> (R1) The M1a or M1b task-leg p50 in the post-arm is at least 1.15x the
> pre-arm p50 AND above the task's ~2300s anchor for M1a.
> (R2) The share of M1 legs at or above 3600s rises by at least 5 percentage
> points over the pre-arm AND exceeds the task's ~10% anchor.
> (R3) Timeouts/day (M3) rises over the pre-arm.
> (R4) The M2 merge-verify p50 is at least 1.15x the pre-arm AND neither
> landings/day nor workflow_verify/day rose by 15% or more.
> Otherwise KEEP 2.
> M4 (backlog) and M5 (queue depth) are reported as context. They cannot
> trigger a revert by themselves ... They are also not allowed to veto a
> triggered revert.

The 1.15 materiality threshold is `TOL_WORSE` in
`scripts/merge-pytest-n-ab-analysis.py`.

**VERDICT: REVERT to 1.**

- **R1 fires on both M1 sources.** M1a p50 3233s -> 4995s (×1.55, above 2300s).
  M1b p50 222s -> 633s (×2.85).
- **R2 fires on both M1 sources.** M1a >= 3600s share 27.3% -> 63.3% (+36.0 pp,
  above 10%). M1b 11.0% -> 28.1% (+17.1 pp, above 10%).
- R3 fires on the journal secondary only: 0 -> 2 timeouts. Both were
  fused-memory task-lane legs that hit that module's 1200s wall on 2026-09-24
  (20:12:36Z and 20:28:23Z; the second belongs to task 5800). The primary
  corpus shows 0 -> 0. With n=2 this is weak, and the verdict does not depend
  on it.
- R4 does not fire: merge-verify p50 ×1.06.

The throughput tension clause does not apply. workflow_verify/day rose 5.0%,
below the 15% bar, and landings/day fell 17.6%.

The one measure that moved in the raise's favour is M5 verify-phase dwell (p50
5.04h -> 4.43h, ×0.88), with verify exits/day up 5.0%. Under the pre-registered
rule M5 is context and cannot veto. It is recorded here because it is the only
counter-evidence.
Hypothesis: the second slot turned slot-wait time into run time. Each leg ran
1.5–2.9x longer on a host that is already CPU-bound (the census puts 22 of 29
POST legs in the `heavy` PSI band, against 3 of 14 in PRE; end-of-run cpu-some
avg60 p50 is 45 -> 52). The wait + run total per task stayed roughly flat, and
landings fell. The raise's own premise, "20 parked at phase_enter verify", did
not clear either: the in-verify depth p50 is 17 at 1 slot and 16 at 2.

### 5. Caveats and confounds

These are listed, not adjusted for.

- **M1a is survivorship-biased.** The corpus is the worktree summaries plus the
  archive, and worktrees are reclaimed after a task finishes. Older windows
  therefore lose records, and M1a's legs/day (PRE 7.61, POST 11.30) is NOT a
  throughput figure. The two fused-memory timeouts the journal records on 09-24
  are missing from the corpus for the same reason: 5800's worktree is gone, and
  its archive holds only 09-26 junit files. The corpus shifted during this
  measurement: a rerun at 2026-09-27T20:25Z found POST at 48 legs, with one
  failed leg reclaimed (failed 20 -> 19), which moves the >= 3600s share to
  64.6% (31/48). The clean-leg figures did not change. The table reports the
  values as first measured.
- **M1b covers only task-lane verifies that followed a real rebase**
  (`workflow.py::_emit_rebase_verify_cost`). It spans all modules, and its
  duration EXCLUDES slot wait, because the leg clock starts inside
  `verify.py::_admission_slot`. It is the longest check's wall time, not the
  sum of all checks.
- Neither source can see slot WAIT directly. M5 dwell (wait + run) is a
  phase-event proxy. A task that never emits `phase_exit verify` (a restart or
  a hung waiter) stays "in verify", so the depth is also reported with legs
  more than 24h stale excluded. Task 5798 (pending) makes both wait and
  duration first-class.
- **Clean-leg selection.** Failed legs (19 PRE, 20 POST) are excluded from the
  p50 but included in the >= 3600s share. The census's failed/clean partition
  is `verify_budget_census.py::summarise_legs`.
- **The walls differ by module and by warm/cold.** The orchestrator suite is
  7200s warm and 10800s cold (the one POST leg >= 7200s was clean). fused-memory
  is 1200s warm. The 09-03 anchors were set under 3600s warm.
- **The census flags its derived budget floor** (1.5 × max) as exceeding the
  7200s refusal ceiling in both arms: 9500s PRE and 11600s POST. That finding
  predates this raise (commit `36c4c71eb4`) and is not this gate's to act on.
  It is noted only because POST's 7713s max widens it.
- Legs are bucketed by start time. A leg that started shortly before P and
  overlapped the 2-slot period (for example 3316's 6276s leg at 11:10Z) counts
  as PRE.
- Other config changes inside POST: effort max -> xhigh for architect,
  implementer, debugger and deep_reviewer (config_reload 422273,
  2026-09-24T06:47Z); merger effort max -> high (422565, 2026-09-24T09:28Z);
  B3 low-risk auto-unblock enabled (`78c1da55d9`, 09-25); steward-retry-fable
  routing rule removed (`373f5ac3ec`, 09-26). Verify-code changes: task 5337's
  worker-death truncation detection landed on 09-26 (`32b550b7ce`,
  `2f609f6393`). Inside PRE: steward_lifetime_budget 12 -> 20 (config_reload
  417574, 09-21) and task 5580's re-run non-verdict (`c6a92c762e`, 09-21).
  None of these changes verify concurrency. The effort reductions would, if
  anything, be expected to speed tasks up, not slow verify legs down.
- The arms are 4.3 days each, and the PRE arm's M1a sample is small (n=14
  clean). The direction agrees across both M1 sources, across SENS, and within
  each M1b cohort.

### 6. Reproduction

Every command runs from a checkout of this repo. The data lives only in the
MAIN checkout, so `--root` / `--project-root` are REQUIRED: without them the
tools default to the checkout they run from, which in a task worktree has no
`data/` and silently reports n=0. Every window is in the dated
`<iso>..<iso>` form, never `Nd`.

```bash
ROOT=/home/leo/src/dark-factory
PRE=2026-09-19T03:50:05Z..2026-09-23T11:53:28Z
POST=2026-09-23T11:53:28Z..2026-09-27T19:56:51Z
SENS=2026-09-12T07:24:38Z..2026-09-23T11:53:28Z
EP1=2026-08-27T00:00:00Z..2026-09-03T15:58:07Z
EP3=2026-09-03T15:58:07Z..2026-09-08T10:18:07Z

# Premises
git -C $ROOT show main:dark-factory-orchestrator.yaml | grep -n '^verify_admission_task_slots'
sqlite3 "file:$ROOT/data/orchestrator/runs.db?mode=ro" \
  "select id, timestamp, json_extract(data,'$.applied') from events
   where event_type='config_reload' and data like '%verify_admission_task_slots%'"

# M1a (and the census's by-day timed_out)
for W in $PRE $POST $SENS $EP1 $EP3; do
  python3 scripts/verify_budget_census.py --root $ROOT --module orchestrator --window $W
done

# M2 and landings/day
for W in $PRE $POST; do
  python3 scripts/merge_lane_throughput.py --project-root $ROOT --window $W
done
```

M1a wall shares and M3 primary. Run from the repo root. The snippet uses only
the census's public functions, so the leg-selection rule is the census's own:

```python
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'scripts')
import verify_budget_census as c

ROOT = Path('/home/leo/src/dark-factory')
ARMS = {
    'pre':  '2026-09-19T03:50:05Z..2026-09-23T11:53:28Z',
    'post': '2026-09-23T11:53:28Z..2026-09-27T19:56:51Z',
    'sens': '2026-09-12T07:24:38Z..2026-09-23T11:53:28Z',
    'ep_1slot': '2026-08-27T00:00:00Z..2026-09-03T15:58:07Z',
    'ep_3slot': '2026-09-03T15:58:07Z..2026-09-08T10:18:07Z',
}
corpus = c.load_records([ROOT])
expected = c.read_module_test_command(ROOT, 'orchestrator')
orch = c.select_full_suite_legs(corpus, expected=expected, prefix='orchestrator').legs

def window(spec):
    lo, hi = spec.split('..')
    return c.parse_instant(lo), c.parse_instant(hi)

def days(w):
    return (w[1] - w[0]).total_seconds() / 86400

for arm, spec in ARMS.items():
    w = window(spec)
    legs = c.within_window(orch, w)
    n = len(legs)
    ge3600 = sum(1 for l in legs if l.duration_secs >= 3600)
    ge7200 = sum(1 for l in legs if l.duration_secs >= 7200)
    tout = sum(1 for l in legs if l.timed_out)
    print(f'M1a {arm:9s} {spec}  all_legs={n} legs/day={n/days(w):.2f} '
          f'>=3600s={ge3600} ({100*ge3600/max(n,1):.1f}%) >=7200s={ge7200} '
          f'timed_out={tout} ({100*tout/max(n,1):.1f}%)')

# M3 primary: every timed-out commands[] entry, any module, any label, per UTC day.
entries = []
for r in corpus.records:
    for e in r.payload.get('commands') or []:
        if isinstance(e, dict) and e.get('started_at'):
            entries.append((r.where.module_prefix, e))
for arm in ('pre', 'post'):
    w = window(ARMS[arm])
    inwin = [(m, e) for m, e in entries
             if (t := c.parse_instant(e['started_at'])) and w[0] <= t < w[1]]
    tos = [(m, e) for m, e in inwin if e.get('timed_out')]
    per_day = Counter(c.day_bucket(e['started_at']).isoformat() for _, e in tos)
    print(f'M3  {arm:4s} entries={len(inwin)} timed_out={len(tos)} '
          f'per_day={len(tos)/days(w):.2f} by_day={dict(sorted(per_day.items()))} '
          f'by_module_label={dict(Counter((m, e.get("label")) for m, e in tos))}')
```

M1b (`rebase_verify_cost`, top-level `duration_ms`):

```python
import sqlite3, statistics
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
ARMS = {
    'pre':      ('2026-09-19T03:50:05', '2026-09-23T11:53:28'),
    'post':     ('2026-09-23T11:53:28', '2026-09-27T19:56:51'),
    'sens':     ('2026-09-12T07:24:38', '2026-09-23T11:53:28'),
    'ep_1slot': ('2026-08-27T00:00:00', '2026-09-03T15:58:07'),
    'ep_3slot': ('2026-09-03T15:58:07', '2026-09-08T10:18:07'),
}
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)
con = sqlite3.connect(DB, uri=True)
for arm, (lo, hi) in ARMS.items():
    rows = [r[0] / 1000 for r in con.execute(
        "select duration_ms from events where event_type='rebase_verify_cost' "
        "and duration_ms is not null and timestamp >= ? and timestamp < ?", (lo, hi))]
    n = len(rows)
    print(f'M1b {arm:9s} {lo}Z..{hi}Z n={n} p50={pct(rows,.5):.0f}s p90={pct(rows,.9):.0f}s '
          f'mean={statistics.mean(rows):.0f}s >=3600s={100*sum(x>=3600 for x in rows)/n:.1f}% '
          f'>=7200s={100*sum(x>=7200 for x in rows)/n:.1f}%')
```

M2 mean and throughput context:

```python
import sqlite3, statistics, subprocess
from datetime import datetime, timezone
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
ARMS = {
    'pre':  ('2026-09-19T03:50:05', '2026-09-23T11:53:28'),
    'post': ('2026-09-23T11:53:28', '2026-09-27T19:56:51'),
    'sens': ('2026-09-12T07:24:38', '2026-09-23T11:53:28'),
}
utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
con = sqlite3.connect(DB, uri=True)
merges = [datetime.fromisoformat(s).astimezone(timezone.utc) for s in subprocess.run(
    ['git', '-C', '/home/leo/src/dark-factory', 'log', '--first-parent', 'main', '--merges',
     '--format=%cI', '--since=2026-09-10'], capture_output=True, text=True, check=True).stdout.split()]
for arm, (lo, hi) in ARMS.items():
    d = (utc(hi) - utc(lo)).total_seconds() / 86400
    mv = [r[0] / 60000 for r in con.execute(
        "select json_extract(data,'$.duration_ms') from events where event_type='merge_verify' "
        "and json_extract(data,'$.duration_ms') is not null and timestamp >= ? and timestamp < ?", (lo, hi))]
    wv = con.execute("select count(*) from events where event_type='workflow_verify' "
                     "and timestamp >= ? and timestamp < ?", (lo, hi)).fetchone()[0]
    done = con.execute("select count(*) from events where event_type='merge_finalized' "
                       "and json_extract(data,'$.state')='done' and timestamp >= ? and timestamp < ?",
                       (lo, hi)).fetchone()[0]
    fp = sum(1 for m in merges if utc(lo) <= m < utc(hi))
    print(f'{arm:4s} merge_verify n={len(mv)} mean={statistics.mean(mv):.1f}min '
          f'p50={statistics.median(mv):.1f}min | workflow_verify {wv/d:.1f}/day '
          f'| landings {done/d:.1f}/day | first-parent merges {fp/d:.1f}/day')
```

M3 secondary (journal). This is slow, about 100s or more per day, so run the
days in parallel and give it a long timeout or background it. Journal retention
is short: every day from 09-18 through T had coverage here, but a day that
prints `lines=0` has none.

```bash
for d in 18 19 20 21 22 23 24 25 26 27; do
  nd=$(date -u -d "2026-09-$d +1 day" +%Y-%m-%d)
  ( journalctl --user -u orchestrator-dark-factory \
      --since "2026-09-$d 00:00:00 UTC" --until "$nd 00:00:00 UTC" --no-pager -o short-iso \
    | awk -v day=2026-09-$d '/^[0-9]{4}-/{if(!f)f=$1; l=$1; n++} /Command timed out after/{c++}
        END{printf "%s timed_out=%d lines=%d first=%s last=%s\n", day, c+0, n+0, f, l}' ) &
done; wait
```

M4 (burndown). Measured rows only, using the predicate
`dashboard/src/dashboard/data/burndown.py::_measured_rows`:

```python
import sqlite3
from datetime import datetime
DB = 'file:/home/leo/src/dark-factory/data/burndown/burndown.db?mode=ro'
PROJ = '/home/leo/src/dark-factory'
MEASURED = "(state IS NULL OR state = 'value')"
ARMS = {'pre': ('2026-09-19T03:50:05', '2026-09-23T11:53:28'),
        'post': ('2026-09-23T11:53:28', '2026-09-27T19:56:51')}
con = sqlite3.connect(DB, uri=True)
for arm, (lo, hi) in ARMS.items():
    rows = con.execute(f"select ts, pending, done from snapshots where project_id=? and {MEASURED} "
                       "and ts >= ? and ts < ? order by ts", (PROJ, lo, hi)).fetchall()
    if len(rows) < 2:
        print(arm, 'NOT MEASURABLE from burndown.db', rows); continue
    (t0, p0, d0), (t1, p1, d1) = rows[0], rows[-1]
    h = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 3600
    print(arm, len(rows), t0, p0, d0, t1, p1, d1, f'{h:.1f}h',
          f'pending/day={(p1-p0)*24/h:+.1f} done/day={(d1-d0)*24/h:+.1f}')
lo_gap = con.execute(f"select max(ts) from snapshots where project_id=? and {MEASURED} "
                     "and ts < '2026-09-23T11:53:28'", (PROJ,)).fetchone()[0]
hi_gap = con.execute(f"select min(ts) from snapshots where project_id=? and {MEASURED} and ts > ?",
                     (PROJ, lo_gap)).fetchone()[0]
print('gap', lo_gap, '->', hi_gap)
for r in con.execute(f"select project_id, count(*) from snapshots where {MEASURED} "
                     "and ts > ? and ts < ? group by project_id", (lo_gap, hi_gap)):
    print('  measured rows inside gap', r)
```

M5 (phase-event occupancy and dwell). A task is IN VERIFY at t when its latest
`phase_enter`/`phase_exit` event (any phase) before t is `phase_enter` with
phase `verify`:

```python
import sqlite3, statistics
from datetime import datetime, timedelta, timezone
DB = 'file:/home/leo/src/dark-factory/data/orchestrator/runs.db?mode=ro'
ARMS = {'pre': ('2026-09-19T03:50:05', '2026-09-23T11:53:28'),
        'post': ('2026-09-23T11:53:28', '2026-09-27T19:56:51')}
T = datetime(2026, 9, 27, 19, 56, 51, tzinfo=timezone.utc)
STALE = timedelta(hours=24)
con = sqlite3.connect(DB, uri=True)
ev = [(datetime.fromisoformat(ts), task, et, ph) for ts, task, et, ph in con.execute(
    "select timestamp, task_id, event_type, phase from events where event_type in "
    "('phase_enter','phase_exit') and task_id is not null and timestamp >= '2026-09-10' "
    "order by timestamp, id")]

def depth_at(t, cap):
    last = {}
    for ts, task, et, ph in ev:
        if ts >= t:
            break
        last[task] = (ts, et, ph)
    return sum(1 for ts, et, ph in last.values()
               if et == 'phase_enter' and ph == 'verify' and (cap is None or t - ts <= cap))

for arm, (lo, hi) in ARMS.items():
    lo_t, hi_t = (datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in (lo, hi))
    samples = [lo_t + timedelta(hours=i) for i in range(int((hi_t - lo_t) / timedelta(hours=1)) + 1)]
    raw = [depth_at(s, None) for s in samples]
    capped = [depth_at(s, STALE) for s in samples]
    open_, dwell = {}, []
    for ts, task, et, ph in ev:
        if ph != 'verify':
            continue
        if et == 'phase_enter':
            open_[task] = ts
        elif task in open_:
            start = open_.pop(task)
            if lo_t <= start < hi_t:
                dwell.append((ts - start).total_seconds() / 3600)
    exits = sum(1 for ts, _, et, ph in ev if et == 'phase_exit' and ph == 'verify' and lo_t <= ts < hi_t)
    d = (hi_t - lo_t).total_seconds() / 86400
    print(arm, 'depth p50/max', statistics.median(raw), max(raw),
          '| >24h-stale excluded', statistics.median(capped), max(capped),
          f'| dwell n={len(dwell)} p50={statistics.median(dwell):.2f}h '
          f'p90={sorted(dwell)[int(.9 * len(dwell))]:.2f}h | exits/day={exits / d:.1f}')
print('at T', depth_at(T, None), depth_at(T, STALE))
```

### 7. Status and hot-apply

- The revert is the commit on branch `task/5797` titled `config(5797):
  verify_admission_task_slots 2 -> 1` (`verify_admission_task_slots: 1`, plus
  a dated paragraph in the yaml block pointing here). Its SHA changes when the
  branch is rebased or merged, so find it by that title on main. It was
  validated with `orchestrator check-config` (no unknown keys) and
  `load_config` (resolves to 1), and with
  `tests/scripts/test_orchestrator_config_duplicate_keys.py`,
  `tests/scripts/test_orchestrator_restart_config_drift.py` and
  `orchestrator/tests/test_config_verify_admission_reload.py` (14 + 16
  passed).
- **The revert is NOT live when this branch is committed.** It goes live only
  after (a) `task/5797` merges to main AND (b) `reload_config` runs against
  the running dark-factory orchestrator, which re-reads
  `/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` (the
  `config_path` recorded in config_reload 420916). Nothing performs (b)
  automatically: the yaml is not in `orchestrator_restart_watch_prefixes`,
  `orchestrator_restart_on_merge_enabled` is false, and nothing auto-reloads
  config.
- **Success condition:** the returned `applied.verify_admission_task_slots ==
  {old: 2, new: 1}` with `restart_required` empty. Judge by those
  dispositions, not by the top-level `reloaded` flag. It is confirmed
  afterwards by a new runs.db `config_reload` event carrying that `applied`
  entry.
- **No restart is needed or wanted.** The knob is green-tier, and a restart
  destroys the in-flight work it exists to protect.
- At measurement time no operator revert existed: main still read 2, and no
  config_reload after 420916 touched the key. The hot-apply ask is
  **esc-5797-2**. esc-5797-1 (filed at planning time) offered Leo the earlier
  direct-to-main route.
- Out-of-scope follow-ups: fused-memory was unreachable (ECONNREFUSED) when
  `submit_task` was called, so both were filed through `escalate_info` as the
  plan's fallback, and no ticket ids exist:
  - **esc-5797-3** (cleanup_needed): codify this A/B cut as a runnable script
    once task 5798 lands (suggestion_hash `f0ba647a8e9f59c4`). The script
    must import the measured-rows predicate from
    `dashboard/src/dashboard/data/burndown.py::_measured_rows`, not copy the
    SQL literal that §6's M4 snippet carries; the literal drifts silently
    when that predicate changes (addendum esc-5797-5).
  - **esc-5797-4** (infra_issue): the burndown.db dark-factory snapshot hole,
    2026-09-21T14:50Z..2026-09-27T11:10Z (suggestion_hash `4dc5cea5d2cb7646`).

### 8. Re-measure when

Take this measurement again when any of the following lands. Each one changes
either what the right slot count is or how it can be measured:

- **5677**: re-land the AsyncMock drain tracking. The orchestrator full suite
  goes from ~3000s to ~350–900s, which changes the per-leg cost of a slot.
- **5097**: the `verify_host_policy` knob, preferring the remote verify host. A
  second host changes the CPU contention this gate measured.
- **5349**: the remote-verify integrity layer, which has never emitted an
  event.
- **5416**: the remote-verify parity window procedure.
- **5291**, **5295**, **4196**: further task/merge verify-duration and
  remote-verify work named in Leo's 2026-09-23 ruling.
- **5798**: makes the admission gate trackable. It changes the SOURCES, not the
  answer: `workflow_verify` gains `duration_ms`, and slot-wait events appear.
  Once it lands, M1 should come from runs.db `workflow_verify` rather than the
  survivorship-biased summary corpus and the rebase-only `rebase_verify_cost`
  proxy, and M5 should come from the slot-wait events rather than the
  phase-event occupancy proxy.

The next measurement appends a dated section to this file. It reuses §6's
Reproduction commands with new arms, cut at the relevant config_reload
instant, or it uses the runnable script from esc-5797-3 if that exists by
then. Use the same pre-registered rule shape: an equal-length pre-arm, the
×1.15 / +5 pp materiality bars, and M4/M5 as context only.
