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

`--journal` defaults to the file fused-memory's config names
(`reconciliation.data_dir`, read as `snapshot` reads the retention horizon;
a relative `data_dir` is anchored at this checkout, the service's working
directory), the live `data/reconciliation/write_journal.db`. Run it from the
main checkout `/home/leo/src/dark-factory`, as task 7026's invocation does; from
a worktree that default names the worktree's own (absent) `data/` and the
command fails closed, so pass `--journal` explicitly there.
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
`--window-end`. Windows reaching past the raw read horizon are refused,
because fused-memory's retention prune, which runs at startup and every
`prune_interval_seconds` (30 min), deletes raw reads that old between
snapshot and verify. The snapshot file records the horizon it was taken
under, and `verify` refuses, rather than reporting mismatches, once a window
has crossed it. With the default windows that leaves about three weeks
between snapshot and the last verify.

The horizon is fused-memory's `write_journal.read_retention_days`, 30 days
unless `WRITE_JOURNAL_READ_RETENTION_DAYS` says otherwise. The script loads
it the way the service does: from `$CONFIG_PATH`, else
`fused-memory/config/config.yaml`, with `${VAR:default}` expanded from YOUR
shell's environment. The report's `read_retention_days` shows the value it
used. If the service's environment differs from yours, pass the service's
value with `--read-retention-days`. The service's environment is the unit's
(`systemctl --user show fused-memory.service -p Environment`) plus
`fused-memory/.env`.

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
well under ~2 s (the full-scale pilot measured mean ~0.45 s, max ~0.6 s at the
default 2,000 rowids). If it is higher, lower `--chunk-rowids` or raise
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

**The retention prune keeps running. Leave it on.** fused-memory runs
`WriteJournal.prune_write_ops` at startup and every `prune_interval_seconds`
(task 6655), each pass in its own short transactions, while `migrate` runs.
They cannot lose or double-count a read. A chunk counts exactly the rows it
deletes in one `BEGIN IMMEDIATE` transaction, the prune deletes only raw rows
that still exist, and the prune never touches `read_op_counts`. So every
count-only read ends up either counted by `migrate` or deleted by the prune,
once. The prune takes only reads older than `read_retention_days`, the
horizon it applied before task 5405, so the `snapshot` windows never see it.
Such reads are dropped uncounted, as they always were. If the prune has drained
the pre-horizon backlog first, `read_op_counts` history starts roughly one
retention horizon before the migration. If it has not (live: oldest non-search
read 2026-08-23 at 08:58Z on 10-19), the bulk is ~35M rows / ~5 h, history
starts wherever the prune-vs-migrate race lands, and `status`'s
`count_only_span` is the real figure.
The two take the write lock in turn. A prune batch (500 rows, measured
0.17-0.55 s) can wait behind one chunk, and a journalled tool call can wait
behind both, about 2.5 s at the measured maxima. That is still inside
fused-memory's 5000 ms `busy_timeout`. A prune batch caught behind a chunk
longer than 5 s fails its pass with `Failed to prune write_ops: database is
locked` at ERROR, and the next pass retries. Treat that line the way the
pilot treats `max_chunk_seconds` above 2 s: lower `--chunk-rowids`.
Measured 2026-10-19 on a 400,000-row scratch with 40 days of history, a
prune pass every 0.2 s and live journal writes during `migrate --apply`:
for every key, rollup plus pruned equalled the original count exactly.
There were 0 journal drops and `verify` showed 0 mismatches.

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

# Hold: make both units unstartable for the window. The lock lives in
# $XDG_RUNTIME_DIR, so a reboot or logout mid-window releases it.
HOLD="$XDG_RUNTIME_DIR/task5405-reclaim.lock"
for unit in fused-memory.service fused-memory.socket; do
  mkdir -p ~/.config/systemd/user/$unit.d
  printf '[Unit]\nConditionPathExists=!%s\n' "$HOLD" \
    > ~/.config/systemd/user/$unit.d/task5405-hold.conf
done
touch "$HOLD" && systemctl --user daemon-reload
systemctl --user stop fused-memory.socket
systemctl --user disable --now fused-memory.service

$RJ reclaim --apply --temp-dir /home/leo/src/dark-factory/data/reconciliation

rm "$HOLD" ~/.config/systemd/user/fused-memory.{service,socket}.d/task5405-hold.conf
systemctl --user daemon-reload
systemctl --user start fused-memory.socket
systemctl --user enable --now fused-memory.service
```

- **Why the hold.** Stopping the service is not enough, because several
  things start it again:
  - `fused-memory.socket` starts it on the next connection, enabled or not;
  - the watchdog's `fused_memory_liveness_pass` revives a stopped, enabled
    unit after ~3 port-down ticks (~180 s);
  - the post-merge coordinator runs `scripts/restart-fused-memory.sh`;
  - the watchdog's staleness backstop;
  - `Wants=fused-memory.service` in the orchestrator units.

  A start whose `ConditionPathExists` fails is skipped, whoever asks. This is
  the same mechanism `plans/local-embedder-cutover-prd.md` designs a permanent
  hold around. Task 5405 did not run this recipe against the live units, and
  whether a restarter alarms on a skipped start is unverified. Check
  `journalctl --user -u fused-memory.service --since -10min` after the window.
- **Start the socket BEFORE the service**, so the service inherits the
  socket's file descriptor.
- `reclaim` itself refuses while `systemctl --user is-active` reports either
  unit active, or when their state cannot be confirmed.
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
  `write_ops` row. **Expect ONE fresh insert-rate escalation first, for up to
  ~24 h after fused-memory first runs the 5405 build.** The sample is a rowid
  difference over the trailing 24 h, so the ~0.8-1.1M per-read rows the old
  build inserted inside that window count even after `migrate` deletes them,
  and migrating does not shorten it. **Do NOT re-anchor the 75,000 ceiling on
  that transient.** It ends on its own when the window holds only 5405-build
  inserts. The alarm's suggested `GROUP BY operation, agent_id` over
  `write_ops` shows nothing anomalous once rows are migrated, which is the
  tell. A crossing that persists beyond ~24 h after the 5405 restart is a
  real regression.
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

**(c) Migration rate, at full scale (2026-10-18).** The 2026-10-16 figures
here came from a 1M-row scratch journal and did not transfer: on the live
journal's 35M `kind='read'` index entries the chunk statements planned a walk
of `idx_wo_kind_time` and held the write lock for minutes per chunk. Both chunk
statements and the first-rowid lookup are now `NOT INDEXED`, and tests pin their
plans (`fused-memory/tests/test_read_rollup.py::TestMigrationQueryPlans`).
Re-measured on 2026-10-18 against a full-size COPY of the live journal
(19,943,129,088 B, MAX(rowid) 62,188,633, taken with the sqlite3 backup API from
a `mode=ro` source in one step, 138 s; on the same NVMe volume as the live
file; deleted afterwards), with the script run exactly as the runbook says,
apart from creating the empty `read_op_counts` table the live pre-5405
build did not have yet:

- **Plans on the live file (read-only):** both chunk statements
  `SEARCH write_ops USING INTEGER PRIMARY KEY (rowid>? AND rowid<?)`; the
  first-rowid lookup `SCAN write_ops` (rowid order, no sort). One 5,000-rowid
  `RAW_READS_BY_ROLLUP_KEY_SQL` chunk took 0.018 s (35 keys, 4,946 rows); the
  old SQL took 73.7 s and 92.7 s for the same chunk. The first-rowid lookup
  took 6.0 s (old: 237-302 s).
- **`status`:** 12.9 s (old: ~5 min). `snapshot`: 83 s. `verify`: 68-102 s,
  empty `mismatches` after the pilot and again after 2.06M rows (5.9% of the
  backlog).
- **Setup is reported apart from the rate.** `setup_seconds` is the first-rowid
  lookup (0.3-9 s); `seconds` and `rows_per_second` cover the chunk loop only.

| `--chunk-rowids` / `--pause-seconds` | chunks | rows/s | chunk transaction mean / max (s) |
|---|---|---|---|
| 5000 / 0.5 (pilot, `--max-chunks 25`) | 25 | 2,554 | 1.44 / 2.09 |
| **2000 / 0.5** (default) | 25 | 2,096 | 0.44 / 0.61 |
| 1000 / 0.5 | 25 | 1,237 | 0.30 / 0.52 |
| **2000 / 0.5, sustained 600 s** | 593 | **1,960** | **0.51 / 5.69** |

- **The cost is per row, not per transaction.** A chunk deletes ~5,000 rows
  and updates every index of cold pages, so a 5,000-rowid chunk costs ~1.4 s
  and a 2,000-rowid chunk ~0.45 s. That is why the default is now 2,000 and
  not 5,000: the transaction alone then measured p95 0.58 s and max 1.94 s
  against fused-memory's 5000 ms `busy_timeout` (see the next bullet). A
  maximum is not guaranteed: the sustained run's 5.69 s chunk includes the
  checkpoint after COMMIT, so its lock hold is unknown. The pilot rule (step 3 and
  the Variance bullet) applies, with `--chunk-rowids 1000` (max 0.52 s over 25 chunks) as the
  fallback.
- **Lock hold vs. the reported time.** `*_chunk_seconds` include the
  auto-checkpoint that runs after COMMIT, outside the write lock. A separate run
  with checkpoints taken apart (129 chunks of 2,000 rowids) measured the
  transaction alone at mean 0.37 s, p95 0.58 s, max 1.94 s; 87 chunks of 5,000
  measured 0.78 / 1.36 / 2.71 s.
- **Variance:** the 600 s run had one 5.69 s chunk in 593, on a workstation that
  is ~97% CPU-busy. A maximum above 5 s is a dropped write in the live service,
  so watch `max_chunk_seconds` in the pilot; if the 25-chunk pilot shows more
  than ~2 s, lower `--chunk-rowids` or raise `--pause-seconds`.
- **Projection:** the backlog was ~35.1M count-only rows on 2026-10-16, plus
  ~1M a day until fused-memory restarts onto 5405. At the sustained
  1,960 rows/s that is about **5 h** of online migration (extrapolated from a
  10-minute run; the pilot's `rows_per_second` on the live file is the real
  figure). Since task 6655 the periodic prune deletes count-only reads older
  than `read_retention_days` before `migrate` reaches them, so once it has
  drained, the backlog is at most ~30 days of reads, ~30M rows and ~4 h.
  `status`'s `count_only_span` shows what is actually left.
