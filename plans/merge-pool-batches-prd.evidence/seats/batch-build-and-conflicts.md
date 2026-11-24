## Gate 10069: how often the chain build cuts a batch short, and how long it takes

**Main sources.** Every chain build writes one log line to syslog: `build_chain: depth=<links requested> built=<links built> truncated_at=<task> reason=<reason> elapsed_ms… [acquire_ms merge_ms]`. No event records the truncation reason, so this line is the only place it exists. I took those lines from `/var/log/syslog` and `syslog.1`, which cover 2026-11-08 to 11-20 (n=357). Each process ID was matched to a project by comparing its tip SHAs with that project's `merge_verify.merge_sha`. That gave 9 dark-factory PIDs (289 builds) and 7 reify PIDs (68 builds), with no overlap. I did not read the gzipped syslogs from before 11-08; they hold only 12 dark-factory chain verifies. The other source is `runs.db`, opened read-only. Scripts are in `/tmp/gate-10069/conflicts/`.

### 1. Planned vs built
Counting rule: the `depth` in the log line is the number of links requested, after the queue bound, `chain_cap-1` (5 links before 11-19) and the halving bisector have all been applied. `built` is the number of links actually built. A build is short when built < depth. That can only be a truncation, because depth already includes the queue bound. Script: `item1.py`.

| | builds | short | reasons for the short builds | shortfall in links (1/2/3/4/5) |
|---|---|---|---|---|
| dark-factory | 289 | 47 (16%) | already_merged 26, conflict 20, merge_error 1 | 9/9/11/2/16 |
| reify | 68 | 4 (6%) | conflict 4 | 1/1/2/0/0 |

- 22 of the 26 `already_merged` truncations are a single stale request (task 10492), rebuilt over and over on 11-18. Without it, dark-factory is short 25 times in 267 builds (9.4%).
- Grouping by (item, reason), there are only 20 distinct truncation events in 12 days. There were no `train_request`, `stacked_on_unlanded_base` or `lane_base_mismatch` truncations at all.
- Where the requested depth is recorded in `merge_verify.chain_depth_reason` (from 11-19 05:43, n=25, script `item1db.py`), the rows were (requested items → built items): 4→4 eleven times, 3→3 twelve times, 3→2 once, 3→1 once. So 2 of 25 fell short. `queue_depth` was 29 to 73 throughout, so the queue never bounded the depth.

### 2. Per-link failure rate
Counting rule: link k is attempted when depth ≥ k and built ≥ k−1. It fails when built = k−1 and a reason is recorded. Data is both projects, 11-08 to 11-20. "Dedup" counts a truncation by the same item only the first time. Script: `item2.py` and `item2b.py`.

| link (chain item n) | dedup fails/attempts | rate [95% Wilson] | raw rate, all reasons | raw rate, conflict + merge_error |
|---|---|---|---|---|
| 1 (n=2) | 10/325 | 0.031 [0.017, 0.056] | 0.067 | 0.032 |
| 2 (n=3) | 2/266 | 0.008 [0.002, 0.027] | 0.018 | 0.014 |
| 3 (n=4) | 1/158 | 0.006 [0.001, 0.035] | 0.069 | 0.012 |
| 4 (n=5) | 3/133 | 0.023 [0.008, 0.064] | 0.029 | 0.029 |
| 5 (n=6) | 3/117 | 0.026 [0.009, 0.073] | 0.050 | 0.034 |
| pooled | 19/999 | **0.019 [0.012, 0.030]** | 0.048 | |

The rate does not rise with position. Link 1 is the highest, and every interval overlaps. A constant hazard of about 2% per link (about 5% if repeats are counted) fits the data. It is not measured beyond link 5.

### 3. Pairwise conflicts in the waiting set
Counting rule: for each `merge_dequeued` row, take the requests with `in_graph` set (the picked request plus the waiting ones). Count all pairs among them, then the `textual_conflict_pairs` and the `footprint_pair_count`. n=237 rows; the field exists only from 11-15. Script: `item34.py` and `item3b.py`.

| day | rows | pairs | textual | textual share | footprint share |
|---|---|---|---|---|---|
| 11-15 | 28 | 3,454 | 0 | 0 | 0.44 |
| 11-16 | 52 | 16,034 | 0 | 0 | 0.99 |
| 11-17 | 64 | 59,864 | 131 | 0.0022 | 0.97 |
| 11-18 | 37 | 53,940 | 136 | 0.0025 | 0.98 |
| 11-19 | 34 | 74,687 | 309 | 0.0041 | 0.98 |
| 11-20 | 22 | 24,312 | 81 | 0.0033 | 1.00 |
| pooled | 237 | 232,291 | 657 | **0.0028** | 0.975 |

Points that limit how far this can be trusted:
- Footprint overlap is nearly every pair. The footprint is computed as the diff from current main to each branch head, so main's drift lands in every footprint and the measure says almost nothing.
- The zero textual pairs on 11-15 and 11-16 look like pairs that were never assessed, not a clean pool.
- The rows are heavily autocorrelated. From 11-17 there are only 29 distinct conflicting pairs, touching 42 requests, and the worst request (task 10720) has 5 of them.

**If conflicts were independent** (p=0.0028), the chance that every pair in a batch of N is clean, (1−p)^C(N,2), would be 0.83 for 12, 0.71 for 16 and 0.46 for 24.

**Measured directly instead:** I took rows with at least 24 requests that are `in_graph` and not reserved (n=179), took the first N in recorded order (high lane first, then FIFO, the same order as `chain_snapshot`), and checked them against the recorded textual conflicts.

| N | all pairs clean | longest clean prefix p10 / p50 / mean / min |
|---|---|---|
| 12 | 0.97 (173/179) | 12 / 12 / 11.9 / 5 |
| 16 | 0.80 | 12 / 16 / 15.3 / 5 |
| 24 | 0.69 | 12 / 24 / 21.1 / 5 |

From 11-17 only (n=144) the figures are 0.96, 0.75 and 0.61. Conflicts cluster on a few requests, so the measured figures beat independence at 24.

This pairwise view **underestimates** what a real build hits. It does not see conflicts with the frozen prefix or main, conflicts that need three or more requests, `already_merged`, or refusals. At a 0.28% pairwise rate, link 1 should fail about 0.3% of the time; the observed link-1 conflict rate is about 3%.

### 4. If the build skipped a bad link instead of stopping
Counting rule: on the same rows, walk the first N in order and add each request that has no recorded textual conflict with anything already added (a greedy subset). Script: `item34.py`.

| N | included p10 / p50 / min / mean |
|---|---|
| 12 | 12 / 12 / 11 / 12.0 |
| 16 | 15 / 16 / 15 / 15.8 |
| 24 | 22 / 24 / 21 / 23.5 |

The same caveat applies: this is pairwise only.

### 5. Build time
Script: `item5.py` and `item5b.py`.

`chain_build_ms` from `runs.db`, which is the caller's wall clock and includes acquiring the lane. Dark-factory only, n=209:

| chain_items | n | p50 s | p90 s | max s |
|---|---|---|---|---|
| 2 | 23 | 4.7 | 8.1 | 31 |
| 3 | 70 | 3.7 | 12.5 | 57 |
| 4 | 16 | 8.8 | 50.5 | 126 |
| 5 | 13 | 4.1 | 9.2 | 16 |
| 6 | 87 | 4.6 | 14.1 | 37 |

- **Total build time has no measurable slope:** −0.04 s/item [−1.06, +0.98]. The cost is dominated by acquiring the lane.
- The log lines split acquisition from merging from 11-19 03:25 (n=51):
  - Acquisition: dark-factory p50 5.5 s, p90 29 s, max 109 s. Reify p50 2.7 s, max 634 s.
  - Merge phase per merge attempted: dark-factory 1.27 s [−0.47, 3.01] (n=32); reify 0.64 s [−0.03, 1.30] (n=19).
- **Extrapolation, beyond the measured range:**
  - Merge phase alone: 12 items ≈ 9–16 s, 24 items ≈ 16–31 s.
  - Total, by a median fit: dark-factory ≈ 8 s at 12 items and ≈ 12 s at 24.
  - Both sit far under `merge_deep.chain_build_timeout_secs: 300` in `dark-factory-orchestrator.yaml` and the default of 1800 in `config.py::MergeDeepConfig`. Only an acquisition outlier like reify's 634 s comes near either limit.

### 6. Verify duration vs batch size
Counting rule: `merge_verify` from 11-11, full scope only (`retry_scope` null), n=538. Script: `item6.py`.

| chain_items | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| n | 346 | 19 | 68 | 16 | 11 | 78 |
| p50 min | 28.7 | 23.2 | 38.8 | 55.1 | 22.4 | 35.0 |

- Among chains (2 to 6 items, n=192), the slope is **−0.66 min/item [−2.58, +1.26]**. On the laptop runner it is +0.03 [−1.84, +1.89] (n=106). There is no sign of growth.
- If single-item verifies are included, the slope is +1.18 [0.35, 2.00]. That jump comes from 1 item to ≥2 items, which probably reflects the module scope a verify covers, not chain depth.

### 7. How a batch lands
Counting rule: green chain verifies that have `chain_request_ids` (from 11-15, n=48). I matched each member to its `merge_finalized` row with `landed_via_chain=1` within 6 h of the verdict. Script: `item7.py`.

| chain size | n | head landing after the verdict (p50 s) | last member landing after the verdict (p50 / p90 s) |
|---|---|---|---|
| 3 | 22 | 319 | 914 / 2918 |
| 4 | 6 | 292 | 1173 / 1596 |
| 6 | 16 | 178 | 1071 / 5421 |

- **Each member after the head takes 246 s at p50 and 438 s at p90** (n=38).
- **36 of 48 walks landed every member.** 9 stopped after 1, 2 after 3, and 1 landed nothing.
- In syslog, dark-factory's `deep chain walk` summaries show 24 partial walks out of 83. The logged stop reasons are "no longer buffered" 15 times and a failed advance 2 times; the rest are presumably the silent `_request_abandoned` break, which writes no log line.
- Reify: 1 partial walk out of 39.

### What this implies for batches of 12 and 24
1. Build truncation is roughly constant at about 2% per link (deduplicated; about 5% counting repeats). That gives about 81% (raw 58%) of 12-item builds and about 64% (raw 32%) of 24-item builds completing.
2. When a build truncates, the expected built length is about 11 of 12, or about 19 of 24 items, at 2%. Pairwise data puts the p10 clean prefix at about 12 for both sizes.
3. Skipping a bad link instead of stopping would recover almost everything pairwise (p10 of 22 out of 24), but pairwise misses conflicts with main and with three or more requests, so the build's own rate is the safer bound.
4. Build time is not the constraint. It is about 10–30 s extrapolated, against a 300 s timeout, and it is dominated by acquiring the lane.
5. Landing is the cost today, at about 250 s per member: roughly 45 min for 12 and 95 min for 24 (extrapolated). About 25% of walks also stop early. Task 10998 is what removes this.

### Could not measure
- A chain's requested depth before 11-19 at the policy level, since the halving state is not recorded. The `depth` in the log line stands in for it.
- Any failure rate beyond link 5, or for builds of more than 6 items.
- Conflicts with the frozen prefix or main (`conflicts_with_main`), and higher-order conflicts, in the waiting-set analysis.
- Pairwise data before 11-15, and 11-15/16 are probably unassessed.
- Builds before 11-08, because the gzipped syslogs were not read.
- The reason a walk stopped when it hit the silent `_request_abandoned` break.

**Independent re-run:** I recomputed two figures with a separate `sqlite3` CLI query (`/tmp/gate-10069/conflicts/recheck.sql`) instead of the Python scripts. The pooled textual-pair share came out 237 rows / 232,291 pairs / 657 textual = 0.0028, and the chain_items=6 build p50 came out 4.576 s. Both match.
