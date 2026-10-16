# `roll_up_journal_reads.py` — Operator Guide

Once-only move of `write_journal.db`'s per-read history into the hourly
`read_op_counts` rollup, then a VACUUM to give the space back (task 5405).

Since task 5405, `WriteJournal.log_write_op` COUNTS every successful read with
no per-row consumer into `read_op_counts`
(`fused_memory/services/read_rollup.py`) and stores no `write_ops` row for it.
Search reads, failed reads and every write keep their raw row. The history
journalled before that build is still in `write_ops`, roughly 97% of the
table. This script moves it.

Every read is in exactly ONE of the two tables, before, during and after the
move. The dashboard's Memory panel adds them in one statement, so its numbers
do not change at any point.

**Leo ratified deleting rolled-up READ rows on 2026-09-11.** `kind='write'`
rows, search reads and failed reads are never touched.

---

## Preconditions

1. **fused-memory AND the dashboard are both running a build that contains
   task 5405.** The dashboard restarts on merge; fused-memory may run the old
   build for up to ~8 h, until its staleness backstop restarts it. Check both
   with `systemctl --user status fused-memory.service dark-factory-dashboard.service`.
   - An OLD fused-memory still journals every read per row. `migrate` refuses
     while it sees a count-only raw read newer than the deploy grace
     (15 min), naming that row's `created_at`.
   - An OLD dashboard against a NEW fused-memory UNDER-COUNTS reads, because it
     does not read `read_op_counts`. That is cosmetic and ends when the
     dashboard restarts. Restart it before migrating, because the old
     dashboard cannot see moved rows either.
2. All commands below run from the repo root, through the project venv:

```bash
RJ="uv run --frozen --project fused-memory python fused-memory/scripts/roll_up_journal_reads.py"
```

`--journal` defaults to the live file,
`/home/leo/src/dark-factory/data/reconciliation/write_journal.db`.
Every command prints one JSON report on stdout, and refusals on stderr with
exit 1.

---

## Sequence

### 1. `status` (read-only)

```bash
$RJ status
```

`count_only_span` is the rowid span still holding count-only raw reads: the
migration's work queue. `rollup_rows` is the size of `read_op_counts`.

### 2. `snapshot` (read-only)

```bash
$RJ snapshot --out /tmp/task5405-panel-before.json
```

This records the Memory panel's numbers for 1 h, 24 h and 7 d windows, keyed
`(hour, kind, operation)`. The windows end at the hour before the current one,
so the in-flight hour is excluded. Override that with `--window-hours` /
`--window-end`. Windows reaching past the 30-day raw read horizon are refused,
because a fused-memory restart's prune may delete raw reads that old between
snapshot and verify.

### 3. Pilot

```bash
$RJ migrate                          # dry run: the first chunk's preview only
$RJ migrate --apply --max-chunks 25
$RJ verify --snapshot /tmp/task5405-panel-before.json
```

Each chunk moves one rowid window in ONE transaction. It aggregates the
window's count-only rows, upserts them through the live path's own
`COUNT_READ_SQL`, deletes them, then re-reads every key. It commits only if
the rows deleted equal the counts added, key by key; otherwise it rolls back
and exits naming the window, the key and both values. Compare the pilot's
`rows_per_second` with the measured figure below before running the bulk, and
check `max_chunk_seconds`. Each chunk holds SQLite's write lock for about its
transaction time, and fused-memory's `busy_timeout` is 5000 ms: a write that
waits longer is DROPPED (`log_write_op` swallows the error). Keep the maximum
well under ~2 s. If it is higher, lower `--chunk-rowids` or raise
`--pause-seconds`.

`verify` exits 1 and prints every `(window, hour, kind, operation)` whose
count changed.

### 4. Bulk migrate (online, resumable, abortable at any moment)

```bash
systemd-run --user --unit=task5405-rollup --same-dir \
  uv run --frozen --project fused-memory python \
  fused-memory/scripts/roll_up_journal_reads.py migrate --apply
journalctl --user -u task5405-rollup -f
```

`nohup … &` works as well. Stopping it (`systemctl --user stop
task5405-rollup`, Ctrl-C, a crash) loses nothing. The interrupted chunk rolls
back whole, and the remaining count-only rows ARE the work queue, so a rerun
resumes where it stopped and never double-counts. `--max-seconds T` bounds a
single run. A rerun after completion moves 0 rows.

### 5. Verify

```bash
$RJ verify --snapshot /tmp/task5405-panel-before.json
$RJ status        # count_only_span must be null
```

---

## Reclaim (a separate maintenance window)

Moving the rows frees pages, but `auto_vacuum` is NONE, so the file does not
shrink until a VACUUM. Run it only after `status` reports
`count_only_span: null`. `reclaim` refuses otherwise.

```bash
$RJ reclaim       # dry run: page figures, reclaimable bytes, blockers
systemctl --user stop fused-memory.socket
systemctl --user disable --now fused-memory.service
$RJ reclaim --apply --temp-dir /home/leo/src/dark-factory/data/reconciliation
systemctl --user start fused-memory.socket
systemctl --user enable --now fused-memory.service
```

- **Stop the socket.** `fused-memory.socket` holds the listening port across
  service restarts and starts the service on the next connection, whether or
  not the service is enabled. Start it again BEFORE the service, so the
  service inherits the socket's file descriptor.
- **DISABLE the service, not just stop it.** The watchdog's
  `fused_memory_liveness_pass` skips a DISABLED unit, but revives a merely
  stopped one after ~3 port-down ticks (~180 s).
- `reclaim` refuses while `systemctl --user is-active` reports either unit
  active, or when their state cannot be confirmed.
- **Downtime:** the VACUUM measured 103 s on a scratch of the post-migration
  keep-set (2.33 GB), so budget ~2-5 min with the stop/start around it.
  fused-memory is down for the whole window.
- **Disk:** VACUUM writes a temporary copy, about the post-reclaim size, into
  `--temp-dir`. Without `--temp-dir` it uses SQLite's own default:
  `$SQLITE_TMPDIR`, then `$TMPDIR`, `/var/tmp`, `/usr/tmp`, `/tmp`. It then
  writes the compacted pages back through the WAL, which peaks at about the
  same size again. Budget roughly 2 × the post-reclaim size free on those
  filesystems.
- VACUUM runs in place and keeps the inode, so the dashboard's long-lived
  read-only connection stays valid and the dashboard needs no restart.

---

## Expected task-3311 growth-alarm behaviour

The `write_journal_growth_alarm` ceilings in `fused-memory/config/config.yaml`
were re-anchored on the post-5405 volume:

- **Insert rate** (`max_rows_inserted_per_day`): clears by itself once
  fused-memory runs the 5405 build, because counted reads insert no
  `write_ops` row.
- **File size** (`max_file_bytes`): stays BREACHED from deploy until the
  reclaim above runs, because the file only shrinks under VACUUM. That one
  folded escalation is expected; this reclaim clears it.

One week after the reclaim, a follow-up task re-measures bytes/day and
inserts/day and re-anchors both ceilings if the regime differs from the
projection below.

---

## Measured

2026-10-16, task 5405's implementer. The live journal was opened read-only
(`?mode=ro`, ATTACHed). Every write went to scratch copies under
`/tmp/task5405-*`, built with `WriteJournal(...).initialize()` so schema and
indexes match. The live fused-memory was still the pre-5405 build, journalling
every read per row. Runs used `SQLITE_TMPDIR=/tmp/task5405-tmp`, because the
agent sandbox denies writes to `/var/tmp`.

**Live journal at measurement:** 19,250,776,880 B (4,699,587 pages of 4096 B,
freelist 0, `auto_vacuum` NONE), MAX(rowid) 60,127,078.

**(a) Post-reclaim size and VACUUM downtime.** The keep-set is everything
`migrate` leaves behind:

- 1,144,309 `write_ops` rows: writes, search reads, failed reads;
- 461,257 `backend_ops`, 13,344 `mem0_intents`, 61,064 `idempotent_ops` and
  223 `referent_findings` rows, copied whole;
- plus the rollup of (b).

Copying it took 376 s for `write_ops` alone; the scan over the live table
dominates that time. The VACUUM itself:

- `VACUUM` on that 3,415,425,512 B scratch took **103.3 s**. This is the
  expected fused-memory downtime for `reclaim --apply`; budget ~2-5 min with
  the stop/start around it.
- Post-reclaim on-disk size: **2,325,856,256 B** (2.17 GiB; 566,724 pages,
  freelist 0).
- The WAL peaked at 2,334,898,792 B during the VACUUM, about the post-reclaim
  size, on top of the temporary copy.
- Largest objects (`dbstat`): `idempotent_ops` 798 MB (bounded by its 7-day
  prune), `write_ops` 702 MB plus 309 MB across its 7 indexes, and
  `backend_ops` 406 MB plus 71 MB.

**(b) Whole-history reduction and 7-day latency.**

- The rollup of every count-only raw read took 472 s, one full scan of the
  live table.
- It turned 35,100,727 source rows into **28,839** `read_op_counts` rows
  (1,217x), about 575 rollup rows per day.
- The dashboard's `get_memory_ops(hours=168)` on the vacuumed scratch, run
  under `uv run --project dashboard`, took **0.53 s, then 1.03 s** (7,328,280
  reads, 82,897 writes, 21 operations). The task's pre-change baseline was
  168.03 s. `hours=24` took 0.38-0.67 s.

**Rows still journalled per row after the change** (keep-set `write_ops` per
UTC day, writes + search):

| Day | Rows/day |
|---|---|
| 2026-10-08 | 6,249 |
| 2026-10-09 | 5,420 |
| 2026-10-10 | 6,104 |
| 2026-10-11 | 15,486 |
| 2026-10-12 | 19,963 |
| 2026-10-13 | 24,667 |
| 2026-10-14 | 22,606 |
| 2026-10-15 | **38,759** |

That is a rising week. `backend_ops` ran 3.6k-15.9k/day over the same days.
Recent rows are larger than the historical average: 850 B vs 493 B of column
content for `write_ops`, and 778 vs 713 for `backend_ops`. With index overhead
that is about 1,240 and 1,100 B/row. **Normal growth (`write_ops` plus
`backend_ops`) is therefore ~35 MB/day** on the trailing 7-day mean, and
~66 MB/day on the peak day. Count-only reads, which a per-read-journalling
regression would bring back, ran 0.71-1.20M/day, with a trailing 7-day mean of
1,012,738/day. That is ~477 MB/day at the 2026-09-10 measurement of
471 B/row.

**(c) Migration rate.**

- **Setup:** one day of live `write_ops` (2026-10-08, rowids 51,290,176 to
  52,307,029, 1,016,853 rows) was copied into a scratch journal. Copies of it
  ran `migrate_chunk` in a loop for up to 120-150 s per setting.
- **Lock hold vs. checkpoint:** the second pass turned auto-checkpoint off and
  ran `PRAGMA wal_checkpoint(PASSIVE)` after each COMMIT. That separates the
  transaction (the write-lock hold) from the checkpoint, which auto-checkpoint
  runs after the lock is released.

| `--chunk-rowids` / `--pause-seconds` | rows/s | chunk transaction mean / p95 / max (s) | checkpoint mean (s) |
|---|---|---|---|
| 1000 / 0.25 | 1,651 | 0.32 / 0.55 / 1.04 | 0.03 |
| 2000 / 0.5 | 2,092 | 0.43 / 0.93 / 1.93 | 0.02 |
| 2000 / 0.5 (repeat) | 2,067 | 0.40 / 0.60 / 1.08 | 0.05 |
| **5000 / 0.5** (default) | **5,293** | **0.37 / 0.63 / 1.06** | 0.07 |
| 10000 / 0.5 (first pass, finished the day in 99.6 s) | 10,144 | 0.48 / 0.91 / 1.44 | included |

- **Fixed cost per transaction:** a transaction costs ~0.3 s almost
  regardless of size, which is the `synchronous=FULL` commit on this shared,
  loaded disk. Small chunks therefore buy no shorter lock and cost throughput.
- **Variance:** a first-pass 2000 / 0.5 run on a cold copy measured
  0.80 / 2.41 / 3.64 s. Treat the maxima as noisy, which is why the pilot
  reports `max_chunk_seconds`.
- **Projection:** the backlog is 35,100,727 count-only rows, plus ~1M a day
  until fused-memory restarts onto 5405. At ~5,000 rows/s that is **~2 h** on
  scratch. The live table holds ~35x the scratch's rows and its b-trees are
  colder, so expect slower; the pilot's `rows_per_second` is the real figure.
