# Post-landing merge-throughput check for task 5677 — window 2026-09-29T22:38:57Z → cut 2026-10-01T05:48Z

Task 6021. Question: after task 5677 re-landed the creation-time AsyncMock drain tracking,
did the merge gate stall (revert first), or do merges keep landing at least as often as before,
with orchestrator merge-gate legs of about 21.8k tests in under 1,000 s?

Sources. All of them are read-only and outside the worktree:

- `/home/leo/src/dark-factory/data/orchestrator/runs.db`, opened as `sqlite3 -readonly …` or
  `sqlite3.connect('file:…/runs.db?mode=ro', uri=True)`. The tables are `events` and `flake_occurrence`.
- `/home/leo/src/dark-factory/data/verify-logs/<task_id>/attempt-N.<module>.junit-<UTC ts>Z.xml.gz`.
  These are the merge gate's per-leg junit archives.
- `git log` on `main`.

Fixed window: **T = 2026-09-29T22:38:57.018609Z**, the `train_merged` event of
`coalesce-5813-6452a2e1`. **CUT = 2026-10-01T05:48Z**. That is 31.15 h. Every query below is
parameterised by `:t` = `2026-09-29T22:38:57` and `:cut` = `2026-10-01T05:48`. Event timestamps
are ISO-8601 UTC strings, so string comparison is time comparison.

## 1. Landing

The prescribed premise query returns nothing:

```
$ git log main --oneline --grep='task/5677'
(no output)
```

That is because 5677 never landed under its own branch name. Its `done_provenance` is kind
`found_on_main`, commit `448290544c`. Merge-carrier task 6065 hand-carried it, rebased
linearly. It landed as the anchor of train `coalesce-5813-6452a2e1`, whose merge commit
subject is "Merge task/5813 into main". The 5677 commits it carries are titled `test(5677): …`,
and none of those subjects contains the string `task/5677`.

```
$ git log -1 --format='%H %cI %s%n parents: %P' 10907428b1
10907428b1d2272d347f317274250e33d34a0542 2026-09-29T23:04:55+01:00 Merge task/5813 into main
 parents: 493782d6c03aa3f81fd2c0edee1628dbc4d5dcb3 6b86810b35b955c1fff931f01385d8dbc4b75eaf

$ git log --format='%h %cI %s' 10907428b1^1..10907428b1^2 | grep 5677
448290544c 2026-09-29T23:04:52+01:00 test(5677): soak — 10 consecutive green full orchestrator suites on the re-landed drain tracking
077ec5c1d4 2026-09-29T23:04:52+01:00 test(5677): re-land 5668's creation-time AsyncMock drain tracking (reverses 3e7d55ce47)
71622efd33 2026-09-29T23:04:52+01:00 test(5677): RED — restore 5668's creation-time drain-tracking tests
```

```sql
SELECT timestamp, task_id, data FROM events
WHERE event_type = 'train_merged' AND json_extract(data,'$.train_id') = 'coalesce-5813-6452a2e1';
-- 2026-09-29T22:38:57.018609+00:00 | 5813 | {"train_id": "coalesce-5813-6452a2e1",
--   "member_task_ids": ["6065", "5813"],
--   "merge_commit_sha": "10907428b1d2272d347f317274250e33d34a0542",
--   "base_sha": "493782d6c03aa3f81fd2c0edee1628dbc4d5dcb3"}
```

The merge commit's committer time is 2026-09-29T22:04:55Z. `train_merged`, the moment main
advanced, is 34 minutes later at 22:38:57Z, because the merge commit is built before its verify
runs. T is anchored on `train_merged`: that is when the re-landed drain tracking began gating
other merges.

## 2. Throughput

A landing is a `merge_finalized` event with state `done`, or a `train_merged` event. Landings
are deduped by merge sha.

```sql
SELECT timestamp, task_id, event_type,
       COALESCE(json_extract(data,'$.merge_sha'), json_extract(data,'$.merge_commit_sha')) AS sha
FROM events
WHERE timestamp >= :t AND timestamp < :cut
  AND ((event_type = 'merge_finalized' AND json_extract(data,'$.state') = 'done')
       OR event_type = 'train_merged')
ORDER BY timestamp;
```

**42 rows and 42 distinct shas. That is 1.35 landings per hour over 31.15 h.** Leaving out
5677's own train at T, it is 41, or 1.32 per hour. The 42 landings, in UTC with the short sha:

| # | landed | task | via | sha |
|---|---|---|---|---|
| 1 | 09-29 22:38:57 | 5813 (+6065 carrying 5677) | train | 10907428b1 |
| 2 | 09-29 23:32:49 | 5593 | finalize | 4c2ab77e6f |
| 3 | 09-30 00:54:09 | 6029 | finalize | 28069aca06 |
| 4 | 09-30 01:23:30 | 4449 | finalize | ff18e8b8d8 |
| 5 | 09-30 02:41:31 | 4780 | finalize | d3ca04a02f |
| 6 | 09-30 03:36:52 | 4507 | finalize | fa4ac0894a |
| 7 | 09-30 04:33:56 | 5873 | finalize (linear) | 1bb51da994 |
| 8 | 09-30 05:29:23 | 6060 | finalize | 6c11fddc81 |
| 9 | 09-30 06:17:50 | 4465 | finalize | acb35a765b |
| 10 | 09-30 07:09:24 | 4207 | finalize | 6571992a28 |
| 11 | 09-30 07:48:17 | 6036 | finalize | 5c9a0fb0f1 |
| 12 | 09-30 08:29:35 | 5376 | finalize | ecdfbff075 |
| 13 | 09-30 09:09:09 | 5972 | finalize | b4e1349e1c |
| 14 | 09-30 10:17:46 | 5100 | finalize | 5e9efc6eba |
| 15 | 09-30 11:00:49 | 4172 | finalize | a387d5c2fa |
| 16 | 09-30 11:41:57 | 3731 | finalize | 43b44834de |
| 17 | 09-30 12:26:32 | socket-activation-ops-doc | finalize | c110a2d43e |
| 18 | 09-30 13:32:55 | 5460 | finalize | 1afa88e9e2 |
| 19 | 09-30 14:10:50 | 5590 | finalize | 9216ed6d04 |
| 20 | 09-30 14:56:16 | 4731 | finalize | aaf0b2d07d |
| 21 | 09-30 16:26:01 | 5068 | finalize | bc3a5b2e68 |
| 22 | 09-30 16:58:09 | 4880 | finalize | d5f015e641 |
| 23 | 09-30 17:30:51 | 5084 | finalize | d7b7e1e5cb |
| 24 | 09-30 18:18:31 | 4023 | finalize | 5308c6cead |
| 25 | 09-30 19:27:23 | 5974 | finalize | 8d55436e64 |
| 26 | 09-30 20:01:57 | 5099 | finalize | 60990a942c |
| 27 | 09-30 20:33:15 | 5122 | finalize | a6f95d1852 |
| 28 | 09-30 21:05:09 | 6073 | finalize | a884a5f118 |
| 29 | 09-30 21:41:43 | 4856 | finalize | 76b758338d |
| 30 | 09-30 22:21:31 | 6104 | finalize | bf8b9a4d52 |
| 31 | 09-30 23:01:48 | 5883 | finalize | 72f35a450a |
| 32 | 09-30 23:50:42 | 5106 (+4216, 4935) | train | e1655733f2 |
| 33 | 10-01 00:25:25 | 5308 (+5114, 5119) | train | 10466776b5 |
| 34 | 10-01 01:09:02 | 5105 | finalize | 0c461eae5c |
| 35 | 10-01 01:39:34 | 4865 | finalize | 04355914f0 |
| 36 | 10-01 02:10:04 | 6110 | finalize | d31a5317d1 |
| 37 | 10-01 02:42:48 | 5199 (+5086) | train | 7afd7329c3 |
| 38 | 10-01 03:18:24 | 4586 | finalize | 3a3e0ffa41 |
| 39 | 10-01 03:49:37 | 5594 | finalize | f58bf8ad6d |
| 40 | 10-01 04:42:59 | 5884 | finalize (linear) | bab7cb4c78 |
| 41 | 10-01 05:44:25 | 4715 | finalize | d61aac0088 |
| 42 | 10-01 05:46:26 | 4855 | finalize | a1cbb84427 |

**Baseline.** Task 5677 states the pre-landing rate as "about 1 per hour". Four consecutive
landings on 2026-09-28 show that cadence: 8c2caab7d8 at 11:38Z, 2929f723d4 at 12:39Z,
b78ab08464 at 13:39Z and 2c73ffdfd8 at 14:40Z, each confirmed by its `merge_finalized` or
`train_merged` event. The same landing query, run per UTC day over the seven days before T,
gives a lower baseline:

| day | 09-22 | 09-23 | 09-24 | 09-25 | 09-26 | 09-27 | 09-28 |
|---|---|---|---|---|---|---|---|
| landings | 25 | 18 | 12 | 11 | 17 | 19 | 18 |
| per hour | 1.04 | 0.75 | 0.50 | 0.46 | 0.71 | 0.79 | 0.75 |

The median is 0.75 per hour. The window's 1.35 per hour is above both "about 1 per hour" and
every one of those seven days.

**Git cross-check.**

```
$ git -C /home/leo/src/dark-factory log main --first-parent \
    --since=2026-09-29T22:38:57Z --until=2026-10-01T05:48Z --format='%h %cI %s'
```

That lists 54 first-parent commits, and 41 of the 42 landing shas are among them. All 42 are
on main's first-parent chain. The one missing from the date-filtered list is 10907428b1:
`--since` filters on committer time, and that time (22:04:55Z) comes before T. The other 13
first-parent commits are the inner commits of the two linear landings (5873: 6, 5884: 4) and
three direct-to-main commits: 620b3ef82a and e033932845 (nightly census) and d3859667bb
(usage-accounts config).

The task's prescribed `git log --merges` undercounts. It returns 39 for the same range, because
two landings are linear advances with no merge commit: 5884 (bab7cb4c78, `merge_finalized` at
2026-10-01T04:42:59Z) and 5873 (1bb51da994, at 2026-09-30T04:33:56Z). A run of linear landings
would read as a stall under `--merges`.

## 3. Gaps over 60 minutes

Gaps run between consecutive landings in §2's list. The first gap is measured from T. There
are **7 gaps over 60 min**, which matches the architect's count.

Per-gap queries:

```sql
-- queue depth seen by the merge heartbeat
SELECT min(json_extract(data,'$.depth')), max(json_extract(data,'$.depth')), count(*),
       sum(json_extract(data,'$.verify_in_progress') IS NOT NULL)
FROM events WHERE event_type = 'merge_heartbeat' AND timestamp > :gap_start AND timestamp < :gap_end;

-- verifies that closed inside the gap
SELECT timestamp, task_id, json_extract(data,'$.passed'), json_extract(data,'$.duration_ms')/1000.0,
       json_extract(data,'$.attempt'), json_extract(data,'$.runner'), json_extract(data,'$.speculative')
FROM events WHERE event_type = 'merge_verify' AND timestamp > :gap_start AND timestamp <= :gap_end;

-- blocks, and the worker restart
SELECT timestamp, task_id, event_type, json_extract(data,'$.state'), json_extract(data,'$.reason')
FROM events
WHERE (event_type = 'merge_blocked'
       OR (event_type = 'merge_finalized' AND (json_extract(data,'$.state') != 'done'
                                               OR json_extract(data,'$.reason') LIKE '%shutting down%')))
  AND timestamp > :gap_start AND timestamp < :gap_end;

-- arrivals (queue_depth is the worker intake queue at enqueue time)
SELECT timestamp, task_id, json_extract(data,'$.queue_depth')
FROM events WHERE event_type = 'merge_queued' AND timestamp > :gap_start AND timestamp < :gap_end;
```

In the table, **verify-busy** is the union, clipped to the gap, of the intervals
[`merge_verify` timestamp − `data.duration_ms`, `merge_verify` timestamp]. **Lane idle** is the
gap minus verify-busy. **Longest heartbeat silence** is the largest interval between
consecutive heartbeats, with the gap bounds counting as heartbeats. The heartbeat is emitted
about every 5 minutes, and **suppressed entirely when the pipeline depth is 0**: see
`orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._maybe_log_queue_heartbeat`,
which returns early on `snap['depth'] == 0`. So a silence much longer than 5 minutes means the queue was empty.

| gap (UTC) | min | heartbeat depth min/max (n; with verify_in_progress) | longest heartbeat silence | first arrival in gap (`queue_depth`) | `merge_verify` in gap | verify-busy / idle min | other |
|---|---|---|---|---|---|---|---|
| 09-29 23:32 → 09-30 00:54 | 81 | 16/18 (16; 6) | 5.5 min | 00:26:50 6029 (1) | 6029 passed, 1,580 s, attempt 0, local | 26 / 55 | — |
| 09-30 01:23 → 02:41 | 78 | 16/17 (15; 6) | 5.0 min | 02:08:57 4780 (1) | 4780 passed, 1,761 s, attempt 0, local | 29 / 49 | — |
| 09-30 09:09 → 10:17 | 69 | 2/2 (8; 8) | **33.4 min** (09:09:09 → 09:42:32) | 09:42:03 5100 (1) | 5100 passed, 2,133 s, attempt 0, local | 36 / 33 | 3731 `conflict` at 09:42:15 |
| 09-30 12:26 → 13:32 | 66 | 1/4 (8; 7) | **30.4 min** (12:26:32 → 12:56:54) | 12:56:45 5590 (1) | 5460 passed, 1,885 s, attempt 0, local | 31 / 35 | 5590 `merge_blocked` 12:56:47 (real rebase conflict). 5460's orchestrator leg flake absorbed, see §4 |
| 09-30 14:56 → 16:26 | 90 | 2/5 (18; 6) | 5.0 min | 15:52:27 5068 (1) | 5068 passed, 1,948 s, attempt 0, local | 32 / 57 | — |
| 09-30 18:18 → 19:27 | 69 | 2/4 (13; 6) | 5.0 min | 18:56:17 5974 (1) | 5974 passed, 1,796 s, attempt 0, local | 30 / 39 | — |
| 10-01 04:42 → 05:44 | 61 | 3/11 (13; 13) | 5.0 min | 04:50:56 (5 arrivals) | 6015 **failed**, 1,085 s, attempt 0, local, speculative. 4715 passed, 767 s, laptop. 4855 passed, 1,626 s, local, speculative | 45 / 16 | 6015 blocked `unknown_test_failure` 05:01:11. `Merge worker shutting down` for 4855 and 5125 (05:01:12) and 4715 (05:01:16). 5944 and 4584 superseded 04:43:00. `verdict_parity_ok` 4715 at 05:42:06 |

Measured corrections to the plan's expectations:

- **The queue did not "hold work in every gap".** The plan's expectation is that
  `merge_heartbeat` depth stayed at 1 or more throughout. Gaps 3 and 4 contain 33.4- and
  30.4-minute heartbeat silences, which means depth 0, an empty pipeline.
- **In gaps 1, 2, 5 and 6 the depth was non-zero but the lane was idle.** For 39-57 minutes,
  `verify_in_progress` was null and `occupancy.inflight_total` was 0. The head-of-line entry
  was 4792, state `queued`, age about 110 h (gaps 1 and 2), or 5590, state `queued`, age 2-6 h
  (gaps 5 and 6). 5590 had already landed at 14:10:50 through a second request. Every
  `merge_queued` arrival in gaps 1-6 reports `queue_depth = 1`, so the worker's intake queue was
  empty when each one arrived.
  - Hypothesis: the heartbeat's `depth` counts non-runnable registry entries, such as a
    blocked request that was never retired, so here it overstates the runnable work.
- **So gaps 1-6 are arrival-limited, not verify-limited.** In each, the lane waited 33-57
  minutes with nothing runnable. Then the first arrival verified in a single passing attempt-0
  verify (1,580-2,133 s) and landed. No merge verify failed in any of the six.
- **Gap 7** opens with 6015's speculative verify already in flight (start ≈ 04:43:03). That
  verify failed at 05:01:08. Run `run-357d1ceb138e` began cancelling tasks at 05:00:55, and its
  merge worker blocked the in-flight 4855, 5125 and 4715 requests as `Merge worker shutting down`.
  The next run, `run-cf328df0190f`, starts at 05:01:38 and re-queued all three at 05:01:41.
  4715 then ran a laptop verify (passed 05:14:36), followed by a local parity verify
  (`verdict_parity_ok` 05:42:06), and landed at 05:44:25. 4855 landed at 05:46:26.
  - 6015's failing leg was **`tests_scripts`, not the orchestrator leg**.
    `data/verify-logs/6015/attempt-1.tests_scripts.summary-20261001T050049_206278Z.json` records
    `rc: 143`, `timed_out: false`, `duration_secs: 28.7` from `started_at` 05:00:20Z, so it ended
    about 05:00:49Z, and `cause_hint: opaque test exit (no failure markers in output)`.
  - 6015's orchestrator leg, `attempt-1.orchestrator.junit-20261001T045622_088358Z.xml.gz`,
    shows 23,295 tests, 0 failures, 362.0 s.
  - Hypothesis: the `tests_scripts` leg was SIGTERMed by that orchestrator shutdown. rc 143 is
    128+15, and the leg ended about 6 s before the first `cancelled` event.

## 4. Orchestrator merge-gate leg times

There is one row per `data/verify-logs/*/attempt-*.orchestrator.junit-YYYYmmddTHHMMSS_*Z.xml.gz`
whose filename timestamp falls in [T, CUT). Each archive was parsed with `gzip` and
`xml.etree.ElementTree`, and the testsuite attributes `tests`, `failures`, `errors`, `skipped`
and `time` were read (summed over `<testsuite>` children when the root is `<testsuites>`).

```python
import glob, gzip, os, re, xml.etree.ElementTree as ET
from datetime import datetime, timezone
pat = re.compile(r'attempt-(\d+)\.orchestrator\.junit-(\d{8}T\d{6})_\d+Z\.xml\.gz$')
t0 = datetime(2026, 9, 29, 22, 38, 57, tzinfo=timezone.utc); t1 = datetime(2026, 10, 1, 5, 48, tzinfo=timezone.utc)
for p in glob.glob('/home/leo/src/dark-factory/data/verify-logs/*/attempt-*.orchestrator.junit-*Z.xml.gz'):
    m = pat.search(os.path.basename(p)); ts = datetime.strptime(m.group(2), '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    if t0 <= ts < t1:
        root = ET.parse(gzip.open(p)).getroot()
        suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
        print(p.split('/')[-2], ts, {k: sum(float(s.get(k, 0)) for s in suites)
                                     for k in ('tests', 'failures', 'errors', 'skipped', 'time')})
```

| task | attempt | junit UTC | tests | failures | errors | skipped | time s |
|---|---|---|---|---|---|---|---|
| 5593 | 1 | 2026-09-29T23:18:07Z | 23129 | 0 | 0 | 0 | 409.9 |
| 6029 | 1 | 2026-09-30T00:41:39Z | 23129 | 0 | 0 | 0 | 479.2 |
| 4449 | 1 | 2026-09-30T01:08:07Z | 23129 | 0 | 0 | 0 | 329.7 |
| 4780 | 1 | 2026-09-30T02:25:58Z | 23129 | 0 | 0 | 0 | 467.3 |
| 4507 | 1 | 2026-09-30T03:20:32Z | 23134 | 0 | 0 | 0 | 432.6 |
| 5873 | 1 | 2026-09-30T03:54:24Z | 23134 | 0 | 0 | 0 | 442.1 |
| 5873 | 1 | 2026-09-30T04:19:46Z | 23134 | 0 | 0 | 0 | 356.5 |
| 3056 | 1 | 2026-09-30T04:48:32Z | 23134 | 0 | 0 | 0 | 416.6 |
| 6060 | 1 | 2026-09-30T05:17:09Z | 23134 | 0 | 0 | 0 | 496.8 |
| 4465 | 1 | 2026-09-30T06:06:08Z | 23134 | 0 | 0 | 0 | 486.0 |
| 4207 | 1 | 2026-09-30T06:54:47Z | 23134 | 0 | 0 | 0 | 442.8 |
| 6036 | 1 | 2026-09-30T07:31:24Z | 23134 | 0 | 0 | 0 | 675.1 |
| 5376 | 1 | 2026-09-30T08:10:35Z | 23182 | 0 | 0 | 0 | 617.4 |
| 5972 | 1 | 2026-09-30T08:48:53Z | 23182 | 0 | 0 | 0 | 499.7 |
| 3731 | 1 | 2026-09-30T11:23:44Z | 23182 | 0 | 0 | 0 | 757.5 |
| socket-activation-ops-doc | 1 | 2026-09-30T12:08:04Z | 23182 | 0 | 0 | 0 | 701.5 |
| 5460 | 1 | 2026-09-30T13:15:43Z | 23203 | **1** | 0 | 0 | 465.7 |
| 5590 | 1 | 2026-09-30T13:55:22Z | 23203 | 0 | 0 | 0 | 738.1 |
| 4731 | 1 | 2026-09-30T14:39:33Z | 23203 | 0 | 0 | 0 | 462.4 |
| 5068 | 1 | 2026-09-30T16:09:13Z | 23203 | 0 | 0 | 0 | 512.1 |
| 4880 | 1 | 2026-09-30T16:43:32Z | 23203 | **1** | 0 | 0 | 510.3 |
| 5084 | 1 | 2026-09-30T17:15:14Z | 23203 | 0 | 0 | 0 | 508.7 |
| 4023 | 1 | 2026-09-30T18:01:33Z | 23236 | 0 | 0 | 0 | 595.8 |
| 5974 | 1 | 2026-09-30T19:13:59Z | 23236 | 0 | 0 | 0 | 503.7 |
| 5099 | 1 | 2026-09-30T19:47:24Z | 23236 | 0 | 0 | 0 | 466.7 |
| 5122 | 1 | 2026-09-30T20:17:12Z | 23236 | 0 | 0 | 0 | 484.6 |
| 6073 | 1 | 2026-09-30T20:51:19Z | 23236 | 0 | 0 | 0 | 459.7 |
| 4856 | 1 | 2026-09-30T21:24:04Z | 23236 | 0 | 0 | 0 | 511.3 |
| 6104 | 1 | 2026-09-30T22:01:37Z | 23260 | 0 | 0 | 0 | 692.3 |
| 5106 | 1 | 2026-09-30T23:30:03Z | 23260 | 0 | 0 | 0 | 725.7 |
| 5308 | 1 | 2026-10-01T00:10:49Z | 23286 | 0 | 0 | 0 | 630.6 |
| 5105 | 1 | 2026-10-01T00:54:35Z | 23286 | 0 | 0 | 0 | 601.8 |
| 4865 | 1 | 2026-10-01T01:24:00Z | 23286 | 0 | 0 | 0 | 366.9 |
| 6110 | 1 | 2026-10-01T01:54:42Z | 23286 | 0 | 0 | 0 | 460.6 |
| 5199 | 1 | 2026-10-01T02:26:21Z | 23293 | 0 | 0 | 0 | 544.1 |
| 4586 | 1 | 2026-10-01T03:01:18Z | 23293 | 0 | 0 | 0 | 473.1 |
| 5594 | 1 | 2026-10-01T03:35:01Z | 23293 | 0 | 0 | 0 | 401.3 |
| 6015 | 1 | 2026-10-01T04:56:22Z | 23295 | 0 | 0 | 0 | 362.0 |
| 4855 | 1 | 2026-10-01T05:14:20Z | 23293 | 0 | 0 | 0 | 268.6 |
| 4715 | 1 | 2026-10-01T05:28:19Z | 23293 | 0 | 0 | 0 | 334.0 |

**40 legs. 23,129-23,295 tests each. `time` min/median/max = 268.6 / 481.9 / 757.5 s, and 0
legs exceed 1,000 s.** Per the plan's analysis, the 5677 soak ran 354-855 s. Every leg is inside
that range or below it, none above. All 40 legs are well under task 5677's "< 1,000 s"
criterion, at about 23.1-23.3k tests, against the criterion's "~21.8k", which reflects suite
growth since 2026-09-19.

The two legs with a failure were both absorbed by the merge gate's isolated re-run:

- **5460**, 2026-09-30T13:15:43Z: 1 failure,
  `tests.test_laptop_warm_verify_boundary::test_heartbeat_starved_hard_partition_tree_killed_via_timeout`.
  `flake_occurrence` verdict `passes_in_isolation` at 13:26:11, then `merge_flake_suppressed`
  at 13:29:33. The verify passed on attempt 0 (1,885 s, inside the window's merge-verify range
  of 767-2,748 s, median 1,836 s), and the task landed at 13:32:55.
- **4880**, 2026-09-30T16:43:32Z: 1 failure, `tests.test_cli::test_cancel_verify_real_impl_dead_pgid`.
  `passes_in_isolation` at 16:54:52, then `merge_flake_suppressed` at 16:57:14. Landed at 16:58:09.

**Merge verifies in the window, and the coverage of junit archives against them.**

```sql
SELECT timestamp, task_id, json_extract(data,'$.passed'), json_extract(data,'$.duration_ms'),
       json_extract(data,'$.attempt'), json_extract(data,'$.runner'), json_extract(data,'$.speculative')
FROM events WHERE event_type = 'merge_verify' AND timestamp >= :t AND timestamp < :cut ORDER BY timestamp;
```

There are 43 `merge_verify` rows covering 43 distinct tasks: 41 passed, 2 failed. By runner:
42 local and 1 laptop (4715).

- **3056**, 2026-09-30T04:58:44Z, failed. The failing test was in the **`shared`** leg:
  `shared/tests/test_loop_blocking_gate.py::TestRatchet::test_no_unblessed_findings`, verdict
  `fails_in_isolation`. That is a real ratchet failure in the branch, not a flake. 3056's
  orchestrator leg (04:48:32Z) shows 0 failures.
- **6015**, 2026-10-01T05:01:08Z, failed in the `tests_scripts` leg, rc 143; see §3, gap 7.

The 40 legs and the 43 verifies do not map one to one:

- **No archive at all for four verifies.** No `data/verify-logs/<task>/` directory exists for
  5100 (09-30 10:17Z), 4172 (09-30 11:00Z), 5883 (09-30 23:01Z) or 5884 (10-01 04:15Z). The
  cause was not determined.
  - All four of those merge verifies passed. A passed merge verify means every leg passed,
    so the missing archives cannot hide an orchestrator red. They only leave four leg times
    unmeasured.
- **4715's laptop verify** wrote a `remote-laptop.pass-summary` rather than a junit. Its
  orchestrator junit (05:28:19Z) comes from the local parity verify.
- **5873 has two orchestrator junits** (03:54:24Z and 04:19:46Z) but only one `merge_verify`
  row.

## 5. Flakes

```sql
SELECT observed_at, test_id, verdict, call_site, runner, task_id, merge_sha
FROM flake_occurrence WHERE observed_at >= :t AND observed_at < :cut ORDER BY observed_at;
```

| observed_at (UTC) | test_id | verdict | call_site | task | merge_sha |
|---|---|---|---|---|---|
| 2026-09-30T04:58:44 | tests/test_loop_blocking_gate.py::TestRatchet::test_no_unblessed_findings (shared) | fails_in_isolation | merge_gate | 3056 | 1d788a5b31 |
| 2026-09-30T07:43:08 | tests/test_escalations_data.py::TestFetchPinsRecovery::test_timeout_maps_to_none (dashboard) | passes_in_isolation | merge_gate | 6036 | 5c9a0fb0f1 |
| 2026-09-30T13:26:11 | tests/test_laptop_warm_verify_boundary.py::test_heartbeat_starved_hard_partition_tree_killed_via_timeout (orchestrator) | passes_in_isolation | merge_gate | 5460 | 1afa88e9e2 |
| 2026-09-30T16:54:52 | tests/test_cli.py::test_cancel_verify_real_impl_dead_pgid (orchestrator) | passes_in_isolation | merge_gate | 4880 | d5f015e641 |
| 2026-10-01T05:01:08 | `<unknown>` | unconfirmable | merge_gate | 6015 | 6ad3e468fa |

That is 5 rows, as the plan expected. Two are orchestrator-suite tests, and both passed in
isolation and were absorbed.

```sql
SELECT test_id, count(*) FROM flake_occurrence
WHERE observed_at >= :t AND observed_at < :cut GROUP BY test_id HAVING count(*) > 1;
-- (no rows): no test id recurs in the window
```

```sql
SELECT observed_at, test_id, verdict, call_site, task_id FROM flake_occurrence
WHERE test_id LIKE '%test_session_registry%test_main_lease_show%' ORDER BY observed_at DESC LIMIT 5;
-- 2026-09-05T08:29:58 | tests/test_session_registry.py::test_main_lease_show_on_a_corrupt_body_is_fail_soft | fails_in_isolation | merge_gate | 4234
SELECT count(*) FROM flake_occurrence
WHERE test_id LIKE '%test_session_registry%test_main_lease_show%' AND observed_at >= :t;
-- 0
```

The 2026-09-19 latent `test_session_registry` `test_main_lease_show_*` flake has not appeared
since T. Its most recent row is from 2026-09-05.

```sql
SELECT timestamp, task_id, data FROM events
WHERE event_type = 'merge_flake_suppressed' AND timestamp >= :t AND timestamp < :cut ORDER BY timestamp;
```

| timestamp (UTC) | task | node_ids |
|---|---|---|
| 2026-09-30T07:47:12 | 6036 | tests/test_escalations_data.py::TestFetchPinsRecovery::test_timeout_maps_to_none |
| 2026-09-30T13:29:33 | 5460 | tests/test_laptop_warm_verify_boundary.py::test_heartbeat_starved_hard_partition_tree_killed_via_timeout |
| 2026-09-30T16:57:14 | 4880 | tests/test_cli.py::test_cancel_verify_real_impl_dead_pgid |

Each of the three `passes_in_isolation` rows has a matching suppression. Each of those merges
then passed on attempt 0 and landed: 6036 at 07:48:17, 5460 at 13:32:55, 4880 at 16:58:09.

## 6. Tail check since the cut: [2026-10-01T05:48Z, 06:46:53Z]

The same landing, gap, `merge_verify`, junit and flake queries were re-run over [CUT, now], with
the gap anchored on the last landing before the cut (4855 at 05:46:26). These numbers stand
apart from §§2-5, which stay as measured for the fixed window.

- **Landings: 1.** 5125 landed at 06:38:41 (`merge_finalized` done, 1e6a7074c0). Its laptop
  verify passed at 05:57:12 (764 s, attempt 0). The local parity verify followed, with
  `verdict_parity_ok` at 06:36:45.
- **Gaps: none over 60 min.** 05:46:26 → 06:38:41 is 52 min. Heartbeat depth was 1 and
  `verify_in_progress` was 5125 on all 11 heartbeats in that gap. As of 06:46:53 no request has
  been queued since 06:38:41, and no heartbeat has been emitted, which means depth 0: the lane is
  empty, not stalled.
- **Merge verifies: 1**, passed.
- **Orchestrator legs: 1.** 5125's `attempt-1.orchestrator.junit-20261001T061929_601697Z.xml.gz`
  shows 23,293 tests, 0 failures, 0 errors, 650.7 s.
- **Flakes: 0** `flake_occurrence` rows and 0 `merge_flake_suppressed` events since the cut.
- The run is unchanged: all events since the cut come from `run-cf328df0190f`.

No new qualifying stall, so no `escalate_blocker` was filed.

## Verdict

Rule (task 6021, WORK 5). **REVERT** only if some gap of 60 minutes or more after T meets all
three conditions:

- **(a)** merge-heartbeat depth was at least 1 throughout the gap;
- **(b)** the gap contains an orchestrator-leg red, meaning a junit with failures + errors > 0
  that was not absorbed as a flake, or an orchestrator-leg timeout;
- **(c)** that red or timeout is what kept the gap from closing.

| gap (UTC) | (a) depth ≥ 1 throughout | (b) unabsorbed orchestrator red or timeout | (c) | qualifies |
|---|---|---|---|---|
| 09-29 23:32 → 00:54 | nominally (16-18), but no runnable work for 55 min | no: 6029 leg 0 failures | — | no |
| 09-30 01:23 → 02:41 | nominally (16-17), but no runnable work for 49 min | no: 4780 leg 0 failures | — | no |
| 09-30 09:09 → 10:17 | **no**: 33.4 min heartbeat silence, so depth 0 | no: 5100 merge verify passed; no junit archived | — | no |
| 09-30 12:26 → 13:32 | **no**: 30.4 min heartbeat silence, so depth 0 | no: 5460 leg's 1 failure absorbed (`passes_in_isolation`, `merge_flake_suppressed`) | — | no |
| 09-30 14:56 → 16:26 | nominally (2-5), but no runnable work for 57 min | no: 5068 leg 0 failures | — | no |
| 09-30 18:18 → 19:27 | nominally (2-4), but no runnable work for 39 min | no: 5974 leg 0 failures | — | no |
| 10-01 04:42 → 05:44 | yes (3-11) | no: the 6015, 4855 and 4715 orchestrator legs all had 0 failures. The red was 6015's `tests_scripts` leg (rc 143) | — | no |

There are no orchestrator-leg timeouts in the window or the tail. The only two failed merge
verifies are 3056, a `shared` leg, real failure, and 6015, a `tests_scripts` leg.

**Verdict: NO REVERT.** The merge gate did not stall after 5677 landed:

- landings ran at 1.35 per hour over the 31.15 h window, above the task's "about 1 per hour"
  and above every one of the seven prior days (0.46-1.04 per hour);
- all 40 orchestrator merge-gate legs ran 23,129-23,295 tests in 268.6-757.5 s, with none over
  1,000 s;
- the two orchestrator-leg flakes were absorbed by the isolated re-run without failing a merge;
- the tail check since the cut adds one landing, one 650.7 s leg and no new stall.

`git revert -m 1 10907428b1` is not needed.

**Recorded, not reverted** (WORK 5):

- **Arrival-limited gaps, 1-6.** Each one is an idle lane: 33-57 min with nothing runnable. In
  gaps 3 and 4 the pipeline was empty. In gaps 1, 2, 5 and 6 the heartbeat's depth counted only
  stale, non-runnable registry entries (4792 at about 110 h, and 5590's blocked first request
  after 5590 had landed). Each gap closed on the first arrival's single passing attempt-0 verify,
  1,580-2,133 s.
  - The stale entries are reported as recurrence evidence for task 3860 (phantom `queued`
    head-of-line) in `esc-6021-1`.
- **Gap 7, 04:42 → 05:44: infra.** The orchestrator restarted between 05:00:55 and 05:01:38
  (run `run-357d1ceb138e` → `run-cf328df0190f`). 6015's `tests_scripts` leg ended at about
  05:00:49 with rc 143, and the merge worker blocked 4855, 5125 and 4715 as `Merge worker
  shutting down`. All three re-queued at 05:01:41; 4715 and 4855 then landed in the gap
  (05:44:25, 05:46:26) and 5125 after the cut (06:38:41).
  - Hypothesis: the restart SIGTERMed 6015's leg; rc 143 = 128 + 15.
  - Separately, 4715 landed 30 min after its laptop verify passed, because the local parity
    verify runs first. 5125 shows the same pattern in the tail: a 764 s laptop pass, then a
    41-min wait to landing.

## Caveats

- **The rate is set by arrivals as well as by the gate.** In gaps 1-6 the lane waited for work,
  so 1.35 per hour shows the gate is not the bottleneck. It does not by itself measure a gate
  speed-up, and the per-day baselines are not controlled for arrivals or host load.
- **The window includes 5677's own train at T.** Without it, the window has 41 landings at
  1.32 per hour.
- **The junit archive covers only legs the merge gate archived.** 4 of the 43 `merge_verify`
  rows (5100, 4172, 5883, 5884, all passed) have no `data/verify-logs/<task>/` directory, and the
  cause was not determined. Because every leg of a passed verify passed, the missing archives
  hide no red. But four leg times are unmeasured. Follow-up filed as ticket
  `tkt_0RV9AEB6J2MV4ZY7VJH6C41HH9`.
- **Heartbeat depth overstates runnable work** (task 3860). Condition (a) was therefore judged
  from heartbeat silences, `verify_in_progress`, `occupancy.inflight_total` and `merge_queued`
  `queue_depth`, not from `depth` alone.
- **`merge_verify.duration_ms` is the whole multi-module gate**, 767-2,748 s in the window with a
  median of 1,836 s. It is not the orchestrator leg; per-leg times come from the junit `time`
  attribute.
