# Why per-role verify-scope CPUWeight (#5205) did not restore reify's local merge-gate latency

2026-11-19. Task 7112. A read-only study: no weight, slice property, config value or service was changed. Code read:
Dark Factory `main` at `14d11c7ca6`.

Sources are tagged: **[live]** read on this host 2026-11-19 07:11–07:27Z (`census/`), **[record]** from a store the era
itself wrote (reify `runs.db`, the systemd user journal, `data/load-samples.db`) via `history.tsv`, **[code]**,
**[doc]**, and **[inference]**. Causes not established by a measurement are on lines marked *Hypothesis:*.

The question has two eras, and neither is used to refute the other:
- *Flat era* (up to "Merge task/3394 into main", `3a8f8019ea`, 2026-11-19 00:23Z): every verify scope was a direct
  child of `app.slice` beside the orchestrator units. This is the layout the hypothesis is about. It can no longer be
  measured live; it is read from the records it produced.
- *Slice era* (since then): verify scopes sit in `df.slice/df-<project>.slice`; the daemons and their agents are still
  in `app.slice` until task 9985 reinstalls the units. The live census measures only this layout.

## 1. Answer in brief

- **Flat era: partly.** The structure the hypothesis describes is confirmed. #5205 could not lift a merge scope above
  its peers. The 11-12 live read found every verify scope a flat sibling of nine weight-100 orchestrator units
  [doc: weights study `mechanisms.md` F1, M1]. Before #5205 every scope ran at systemd's default weight 100 [code:
  `3d0eed5554` first sets `-p CPUWeight`]. #5205 left the merge at 100 and only lowered task and background scopes
  to 33 and 10. So the merge's share against the units, sccache and the desktop did not change [inference].
  The record does not support the second half, that this weight split is what held reify's merge gate back. Over
  the 144 flat-era local merges since 10-28 that could be joined to their merge-role test leg, a merge got 4.4–5.9
  cores on average (era p50s). Across all 147 joined merges the cores barely respond to host
  load: about 5.5 below a runqueue ratio of 3, and 4.3–4.8 above it [record]. Latency instead moved with the merge's
  own CPU work and something besides the weight. In the same flat layout with the same weights, p50 fell from 81.5
  to 50.8 minutes between the 10-13→11-02 era and 11-16→11-19 [record].
- **Slice era: refuted for reify's merge scope in the six windows measured.** The merge scope used 0.89–1.00 of its
  weight-fair share. Its threads waited on a runqueue for at most 0.12 cores' worth, so the scope was limited by its
  own demand, not by its weight [live]. What did not get its entitlement was the factory as a whole. In the three
  windows where both `df.slice` and `app.slice` wanted more than half the host, `df.slice` received 0.51–0.77 of its
  100:100 share [live]. The cause is not established (§3).
- **After task 9985 (analytic):** if no weights change, moving `orchestrator-reify.service` (daemon and agents, weight
  100) into `df-reify.slice` puts it beside the merge scope at equal weight. At the measured 07:20Z demand, a merge
  wanting 8 or more cores would then be entitled to 1.50 cores, down from 3.09 today and 2.07 in the flat layout
  [inference, `projection.py.txt`]. The intra-slice weights of task 9984 need to land with 9985, not after it.

## 2. The layouts, and what a weight entitles a reify merge scope to in each

A cgroup-v2 weight divides only what the parent receives, and only among siblings that are runnable. Each level
multiplies. The semantics, and the per-CPU caveat, are in `plans/cross-project-weights-study-2026-11-12/mechanisms.md`
M2, which this report does not restate.

**Flat era.** The tree is `mechanisms.md` M1 (live read 2026-11-12). It holds `app.slice` w100 with
`orchestrator-<9 projects>.service` w100 each, every `df-verify-*` scope (merge 100 / task 33 / background 10),
`sccache.service` w100, the dashboard, fused-memory, and the desktop applications. A merge scope is entitled to
C_app × 100 / Σ(weights of runnable app.slice children). Here C_app is what `app.slice` gets, which was nearly the
whole host.
- M2(a) works a quiet case: two projects, each running one merge and two task verifies, nothing else runnable. Each
  merge then gets 9.64 of 32 threads.
- With the 07:20Z demand put back into the flat layout, a merge wanting 8 or more cores would be entitled to
  **2.07** cores [inference, `projection.py.txt`].

**Slice era, before task 9985** (`docs/cgroup-slice-topology.md` §Topology, §Rollout). The tree is user@1000 →
{`app.slice` 100, `df.slice` 100, `session.slice` 100, `background.slice` 30}. Below `df.slice` sit four project
slices at 100 each, and inside `df-reify.slice` the merge scope (100) and the task scopes (33 each) [live:
`census/20261119T071122Z-context.txt`].
- Fully contended, with 4 project slices and 2 task scopes, a merge gets 32 × ½ × ¼ × 100/166 = **2.41** cores.
- At the measured 07:20Z demand it gets **3.09** cores [inference, `projection.py.txt`].

**After task 9985 (analytic).** Each daemon moves into its own project slice, and with it every agent and every
implement-phase build and test those agents run. With today's weights the merge then shares `df-reify.slice` with a
weight-100 daemon unit: entitlement 100/(100 + 100 + 33·k) of the slice. At 07:20Z demand that is **1.50** cores.
`df.slice` now spans more project slices, and `app.slice` (desktop, sccache, shared services) still wanted more than
half the host [inference]. M2(b) states the same dilution analytically (24 × 100/266).

## 3. Measured share

**Live census, slice era** [live: `census/20261119T071516Z-census.txt`, `census/20261119T071937Z-census.txt`].
Each window is 60 s, and every window was saturated (procs_running/nproc ≥ 1). Roles are read from cpu.weight: no
yaml overrides `verify_cgroup_cpu_weight_*`, so 100 means merge.

| Window (Z) | runqueue | reify merge scope | used | wait | used / fair | `df-reify.slice` used / fair | `df.slice` used / fair | host share |
|---|---|---|---|---|---|---|---|---|
| 07:16:18 | 2.22 | lint leg `40b7911f84ac` | 0.51 | 0.00 | 1.00 | 1.76 / 1.76 | 8.89 / 10.39 (0.86) | 0.017 |
| 07:18:20 | 1.50 | test leg `521e658e98c5` | 1.77 | 0.00 | 1.00 | 5.84 / 5.44 | 11.26 / 13.38 (0.84) | – |
| 07:20:40 | 8.16 | test leg | 0.87 | 0.02 | 0.98 | 2.29 / 2.35 | 9.00 / 13.91 (0.65) | 0.028 |
| 07:21:45 | 5.47 | test leg | 1.12 | 0.00 | 1.00 | 2.44 / 2.49 | 10.90 / 14.11 (0.77) | 0.035 |
| 07:22:48 | 2.78 | test leg | 1.04 | 0.12 | 0.89 | 2.89 / 2.50 | 7.62 / 14.83 (0.51) | 0.033 |

The 07:17:19 window (runqueue 1.66) has no merge scope: the lint leg had ended and the test leg had not yet started.
In it `df.slice` got 8.19 / 10.39 (0.79).

- **The kernel honoured the merge scope's weight, and the entitlement did not bind.** The scope used what it wanted.
  At 07:23Z its 84 processes were sleeping `bash`/`timeout` processes plus three `python3` in uninterruptible I/O
  wait (state D). The scope's cpu.pressure some avg10 was 5.2 and its io.pressure some avg10 8.0 [live:
  `census/20261119T072307Z-reify-merge-scope-state.txt`]. A reify merge in this layout gets 2–3% of the host,
  because that is all it asks for.
- **How to read used / fair.** When a cgroup wants less than its proportional share, its fair share is its own
  "wanted", so used / fair is just used / (used + wait). Then it measures only how long the cgroup's own threads sat
  on a runqueue, not a broken weight. That is the case for the merge scope in every row: it barely waited. The weight
  split binds only where siblings want more than their proportional shares.
- **One level up, the weight split did not hold.** From 07:20Z to 07:22Z both `df.slice` (wanted 17–24 cores) and
  `app.slice` (47–58) wanted more than half of the roughly 29 cores the user manager got. So the entitlement was an
  even split, whatever `app.slice`'s inflated "wanted" (§7). `df.slice` got 0.65, 0.77 and 0.51 of it, while
  `app.slice` got 1.43, 1.30 and 1.53 [live]. In the earlier windows `df.slice` wanted less than half, so its
  0.79–0.86 only says it queued.
  *Hypothesis:* M2 notes that a group's weight is spread over CPUs in proportion to its load on each. `df.slice`'s
  task verifies run at nice 15, so its weight may land where it does not win. This was not tested.

**History, flat era** [record: `history.tsv`; journal retention starts 10-28, so earlier eras have no cores data].

| Era (end boundary) | merge latency p50 / p90 min (n) | merges joined | mean cores p50 | test leg cores while running p50 / p90 | CPU-min per merge p50 | runqueue p50 |
|---|---|---|---|---|---|---|
| 48 (to 09-14) | 32.1 / 62.4 (1968) | – | – | – | – | – |
| 24, before #5205 (to 10-04) | 62.7 / 111.9 (273) | – | – | – | – | – |
| 24, #5205, governance on (to 10-13) | 80.1 / 131.0 (58) | – | – | – | – | – |
| 24, #5205, governance off (to 11-02) | 81.5 / 124.2 (85) | 32 | 4.45 | 4.28 / 5.44 | 373 | 3.21 |
| 16 (to the 11-08 pause) | 66.8 / 83.2 (39) | 36 | 4.89 | 4.61 / 5.84 | 369 | 3.58 |
| reify paused (11-08 16:00Z to the resume; the drain) | – | 12 | 5.93 | 5.62 / 6.90 | 375 | 2.94 |
| 16, resumed (to 11-16) | 64.1 / 98.3 (17) | 16 | 4.67 | 4.21 / 6.01 | 314 | 3.01 |
| 24, flat, November (to 11-19 00:23Z) | 50.8 / 67.3 (49) | 48 | 5.84 | 5.75 / 7.19 | 304 | 2.56 |
| 24, slice era (to 07:30Z) | 71.0 / 88.2 (3) | 3 | 3.96 | 4.17 / 4.67 | 247 | 3.09 |

- **Like-for-like windows** [record]. The 14 days before the 48→24 cut: p50 50.5 minutes; the task quotes 52.5. The
  14 days before #5205: 69.8; the task quotes 69.8. From #5205 to the 10-17 measurement: 80.1 (n=63); the task quotes
  97.8 on n=32. The 10-17 measurement's own windows cannot be recovered: `esc-7533-7.json` is not in reify's
  escalation archive (§7).
- **Mean cores against host load**, all joined merges together [record]: runqueue below 2, 5.59 (n=6); 2–3, 5.49
  (65); 3–4, 4.78 (69); 4 and above, 4.34 (7). Contention costs a merge about a fifth of its cores. Even on the least
  loaded host it got about 5.5 cores of 32.

## 4. Why #5205 did not restore latency

Each alternative is weighed against the data above.

1. **The merge's own demand and parallelism.**
   - The test leg ran at a p50 of 4.2–5.8 cores in every flat-era sub-era, including the pause, and its p90 never
     passed 7.2 [record]. Cores fall only about 20% from the least to the most loaded host [record].
   - Live, the test leg was idle-bound and I/O-bound [live].
   - *Hypothesis:* the reify merge gate's wall time is set mainly by the gate's own serial, I/O and harness phases
     and its test parallelism, not by its CPU share. This is the alternative the data supports best.
2. **CPU work per merge.** The median merge cost 373 CPU-minutes up to 11-02 and 304 by 11-16→11-19 [record]. With
   cores 31% higher (4.45 → 5.84), that accounts for the p50 dropping from 81.5 to 50.8 minutes in the same layout
   with the same weights [inference]. *Hypothesis:* the drop came from changes to reify's test suite or merge chains.
   This study did not check that.
3. **Host verify demand: 287 verifies a day against 126 in the 48 era.** Quoted from the 10-17 measurement in the
   task text, not re-measured here. It is consistent with the runqueue ratio of 3–3.6 seen during merges in the
   24/16 eras [record]. Through the weight it costs a merge about a fifth of its cores (§3).
4. **Agents' implement-phase builds and tests in the service cgroups.** `orchestrator-reify.service` used 10.4–14.2
   cores in three of the six windows, against 2.3–2.9 for all of `df-reify.slice` [live]. The weights study measured
   reify's daemon and agents at about a quarter of reify's CPU before the pause, and its verify scopes at about 70%
   [doc: weights `REPORT.md` / `mechanisms.md` F2]. This load competes with the merge at the top level, and per-role
   scope weights never reached it.
5. **Compiles charged to `sccache.service` in `app.slice`.** 3.9–4.3 cores in two windows and about 0.5 in the rest
   [live]. No reify cgroup is charged for this, and no verify weight reaches it.

The per-process view (`cgroup-stall-ratio.py`, three runs in the same period, `census/*-stall-ratio.txt`) shows Dark
Factory's task workers 72–77% stalled while the host was loaded. Its reify buckets read about 0 run cores even while
the census showed reify scopes using 0.5–5.8 cores [live]. It samples one thread per process and finds reify's
processes by name, so it does not see this layout's reify verify work. It adds nothing on reify here.

So #5205 changed the one term (the split among verify scopes) that could not raise a merge above its peers. The
records show the merge gate's latency moving with other things, chiefly the merge's own work and parallelism
[inference].

## 5. The lever: proposed, not applied

Restated against the current layout. Each option is owned elsewhere, and none is applied here: the task says a config
or code change is a separate decision, and the bounding rulings are Leo's.

| Option | Owner | Projected effect on a reify merge under contention | Rulings that bound it |
|---|---|---|---|
| Raise `df.slice` against `app.slice` | topology contract §Consequences (b) | Lifts the factory's floor against the desktop and, until 9985, against every daemon and agent. Where that split bound (07:20–07:22Z), `df.slice` got 0.51–0.77 of its 100:100 share. | Interactive work is privileged (08-03) |
| Per-project slice weights | DF task 9984 (values), 9985 (reinstall) | Splits `df.slice` between projects. For reify's merge it is worth only what reify's slice share is worth: 2–3% of the host today, all it asked for | No weights derived from live load |
| Merge-role sub-slice, or a lower weight on the daemon unit inside each project slice | DF task 9984 | After 9985 it decides whether a merge gets 1.50 or about 3 cores at 07:20Z demand (§2); with the daemon at 33, water-filling the same 07:20Z claims gives 2.10 [inference] | Merge verify is never throttled (09-07): this raises the merge and throttles nothing |
| Weights on task verifies, background runs and services (Layout A) | reify task 8827 | 2–5 minutes on a merge verify, by the weights study's model [doc: weights `REPORT.md` §5] | Leo answered 8827 "not now" on 10-13 |

**Recommendation.** 7112 is answered. Reify's merge was not weight-starved live, and in the flat era its cores were
only mildly sensitive to contention. So no weight lever is projected to restore the 48-era latency. The remaining
weight questions belong to reify 8827 (fold 7112's findings in, as weights `REPORT.md` §5 suggests) and to DF
9984/9985. For 9984 the one finding with a number attached: set the daemon unit's weight below the merge scope's
inside each project slice when 9985 lands. Otherwise the reinstall roughly halves a merge's entitlement (§2).

## 6. Revisit triggers for reify `max_concurrent_tasks` (record only)

| Trigger | Today's value | Source |
|---|---|---|
| A co-tenant winds down | All nine `orchestrator-*.service` units active at 07:27Z. CPU in the census windows: Dark Factory's service 4.0–8.9 cores, my-solar-challenge ≤ 1.24, solar-challenge-platform's verify scopes about 2.5, the other five about 0.01 each | [live] |
| Local gate p50 ≤ 55 min | Last 7 days: p50 54.4 min (passed 54.7), p90 76.0, n=69. Met, narrowly | [record] `history.tsv` `window last_7d` |
| Task-verify admission wait p50 < ~15 min | reify `runs.db` `verify_admission_plan` legs, role task, `waited_secs`. Last 7 days p50 286.4 min, p90 447.8, n=198. Not met | [record] `history.tsv` `admission last_7d` |

## 7. Limits

- **Short windows.** Six 60-second windows in one morning, with one reify merge in progress. The live verdict covers
  that merge's lint and test legs only.
- **"Wanted" is not a lower bound.** It is used plus the summed per-thread runqueue wait of the subtree. For
  CPU-bound work it is a lower bound, because threads shorter than a window are missed. For bursty or mostly sleeping
  threads it is an overestimate: element-desktop showed 1.24 cores of wait against 0.01 used [live]. That inflates the
  fair shares of desktop and agent siblings. It does not affect the merge-scope verdict, which rests on that scope's
  own wait being near zero.
- **Journal join ambiguity.** The test leg's role is read from the merge nice tier in the journal's Started command
  (`shared/src/shared/verify_admission.py::_NICE_TIERS`). Lint legs carry no tier. One is counted as the merge's only
  when a merge test leg starts within 2 s of it ending. Across the 152 merges, 254 lint scopes in merge windows stayed
  unpaired and uncounted. 5 merges overlapped another local merge, and 5 had no merge-role scope. Mean cores are
  therefore a lower bound for a merge's own CPU.
- **Flat-era limits.** It cannot be re-measured live. Its records hold no runqueue wait per scope, so used vs fair
  cannot be computed for it. The journal starts 10-28, so the 48 and pre-#5205 eras have latency but no cores.
- **Era boundaries.** These are commit times: the earliest a change could take effect, not the restart that applied
  it.
- **Lost record.** `reify data/escalations/archive/2026-10-17/esc-7533-7.json` does not exist. The archive starts
  10-21, and only `esc-7533.seq` remains. The task text is the only copy of the 10-17 figures.
- **`load-samples.db`.** Read with `immutable=1`, because read-only WAL access needs to write a `-shm` file outside
  this worktree. Frames not yet checkpointed are unseen; they matter only for the last hours.
- **A second cpu.stat reader.** The census reads `usage_usec` through its own one-line reader. The single parser for
  cgroup files is `shared/src/shared/cgroup_stats.py` (per-consumer-resource-recording PRD D7), and it belongs to
  leaf 1 (task 10456). The switch to it is filed as follow-up ticket `tkt_0RX5H1P7BAE70D75KCYHEFNE2W`, after 10456.
  Leaf 1 (continuous per-consumer CPU and stall) and leaf 8 (task 10463, CPU per verify role and leg) are the
  continuous replacement for this hand census.

## 8. How to re-run

From the dark-factory checkout root. Everything here is read-only.

```
# live census (step 7): pick a merge-role scope (cpu.weight 100) under df.slice/df-reify.slice
/home/leo/src/dark-factory/.venv/bin/python scripts/cgroup_weight_share.py --window 60 --repeat 3 \
  --chain user.slice/user-1000.slice/user@1000.service/df.slice/df-reify.slice/<merge scope> \
  [--chain <a task scope in the same slice>] > plans/7112-verify-scope-weight-share/census/<UTC>-census.txt 2>&1; echo rc=$?
python3 scripts/cgroup-stall-ratio.py --window 60 --json > plans/7112-verify-scope-weight-share/census/<UTC>-stall-ratio.txt 2>&1

# history (step 8): the journal read takes ~12 minutes under load
journalctl --user SYSLOG_IDENTIFIER=systemd -o short-iso-precise --no-pager --grep=df-verify-reify- > /tmp/journal.txt
python3 plans/7112-verify-scope-weight-share/history.py.txt /tmp/journal.txt > plans/7112-verify-scope-weight-share/history.tsv

# layout projection (section 2), from the committed census window
/home/leo/src/dark-factory/.venv/bin/python plans/7112-verify-scope-weight-share/projection.py.txt
```

The projection's output, as cited in §1 and §2:

```
# capacity user@1000.service used 29.00 cores; source 20261119T071937Z-census.txt window 1
merge_wants	layout	level_grants (outermost first)	merge_fair_cores
2	flat	27.84 -> 2.00	2.00
2	slice	13.92 -> 3.46 -> 2.00	2.00
2	9985	13.92 -> 3.94 -> 1.50	1.50
8	flat	27.84 -> 2.07	2.07
8	slice	13.92 -> 4.55 -> 3.09	3.09
8	9985	13.92 -> 3.94 -> 1.50	1.50
16	flat	27.84 -> 2.07	2.07
16	slice	13.92 -> 4.55 -> 3.09	3.09
16	9985	13.92 -> 3.94 -> 1.50	1.50
```
