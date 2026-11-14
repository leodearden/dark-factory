# esc-3190-1 / gate 3190: why the deep merge-ahead canary saw only 2 deep verifies

2026-10-29, /team investigation (session discuss-dark_factory-2180178). Read-only:
nothing was written to runs.db or tasks.db, no config was changed, nothing was
restarted, and esc-3190-1 is still pending.

## Answer in one paragraph

The canary measured is **reify's**. Gate 3190's `before_done` runs against
`/home/leo/src/reify/data/orchestrator/runs.db`. Reify's flip was a hot reload at
2026-10-20T18:08Z (task 3189). Dark Factory's flip (5057/7619) only went live at
10-27T22:59Z, so it is not in this gate's sample.

Reify ran only 2 deep verifies in the window. Three filters multiply:

1. **Few opportunities.** A chain needs at least one other item waiting in the
   lane buffers at the moment a *speculative* item dispatches. Reify's queue in
   October is shallow: 111 of 130 enqueues were at depth 1, and there were 35
   slot-2 verifies in the week. Only 5 of those 35 tried to build a chain.
2. **The chain build times out.** 3 of the 5 builds hit the 120 s
   `CHAIN_BUILD_TIMEOUT_SECS` ceiling and fell back to the adjacent verify. Even
   the builds that succeeded on reify took 38 s and 53 s; the PRD assumed 1–2 s.
3. **The predicate is not the cause.** It counts correctly.

The finding that matters more than the count:

- **When a deep tip passes, its chain walk has landed zero extra items, on both
  projects.** That is 4 of 4 walk-summary log lines, every one "landed 0 of N",
  and DF's second passing tip (6175) logged no walk at all.
- The reason: the tip verify takes 30–150 minutes. During it, the merger's
  look-ahead pops the chained followers out of the buffers and verifies them on
  their own.
- So `items_per_verify ≥ 1.3` **cannot be reached as built**. A longer window
  would produce more HOLDs, not a verdict.

## Evidence chain

### 1. Which canary and which window

- 3190 `metadata.before_done.args` = `--runs-db /home/leo/src/reify/data/orchestrator/runs.db --window-days 7 …`.
  `milestone_deps_satisfied_at` = 2026-10-20T18:16Z, and the gate fired at
  2026-10-27T19:04:17Z. That gives the window
  **W = (10-20T19:04:17Z, 10-27T19:04:17Z]**.
- Reify's runs.db has a `config_reload` at 10-20T18:08:54Z with
  `merge_deep.chain_cap 0 → 6`. The reify yaml still carries
  `merge_deep: chain_cap: 6`, and every restart since has re-read it, so the
  canary was live for all of W.
- DF's runs.db has a `config_reload` at **2026-10-27T22:59:35Z** with
  `applied: merge_deep.chain_cap 0→6`. The 3190 note expected it to go live only
  at the next restart; the reload happened anyway. DF has no η1-style gate of
  its own.
- The brief's throughput candidates mostly belong to other periods or another
  project:
  - The all-accounts usage cap started 10-28 10:00Z, after W.
  - The 10-26 EWA scheduler pause was Dark Factory's. Reify's runs.db has no
    `scheduler_paused` event in W.

### 2. Throughput in W was normal

Reify's per-window totals. Windows run 19:04Z to 19:04Z. W excludes the 10-26
restart storm.

| window | merge verifies (spec/head/null) | done landings | distinct tasks submitted | deep |
|---|---|---|---|---|
| W-3 | 120 (54/59/7) | 104 | 111 | 0 |
| W-2 | 115 (33/77/5) | 102 | 104 | 0 |
| W-1 | 81 (12/69/0) | 66 | 75 | 0 |
| **W** | **120 (35/79/6)** | **94** | **96** | **2** |

There were throughput hits inside W:

- **10-26 05:30–10:46Z:** a restart storm of 194 runs, with 579 `merge_finalized`
  rows reading "Merge worker shutting down" and no landings for 5.7 h.
- **10-21/22:** a 10 h gap with no landings.
- **Laptop outages:** three, which pushed verifies onto the 3–4× slower local
  runner.

Even so, W landed 94 items, the same level as W-2 and W-3. Throughput collapse
is **not** the main reason.

### 3. The opportunity funnel in W (code: `worker.py::SpeculativeMergeWorker._deep_chain_placement`)

These are the gates, in the order the code applies them. The other guards (no
merge commit, a lane-acquisition failure, `_chain_error_suppressed`) did not
fire in W.

| gate | code | count in W | source |
|---|---|---|---|
| item is speculative (slot-2 look-ahead) | `item.speculative` | 35 merge verifies | runs.db `merge_verify.data.speculative=true` |
| ≥1 item waiting in the lane buffers (`queue_len = 1 + len(chain_snapshot()) ≥ 2`) | `worker.py::select_chain_depth` | ~5–6 of 35 (reconstructed); **5 logged build attempts** | occupancy reconstruction from runs.db; syslog |
| build finishes in under 120 s | `_deep_chain_placement` timeout arm | **3 of 5 timed out**: 8182 (target depth 4), 9169 (d=2), 8558 (d=2), "falling back to the adjacent verify" | syslog |
| chain built and verified (`chain_items ≥ 2`) | `_run_inflight_verify` chain arm | **2** (7098, 7190) | runs.db and syslog `build_chain: … built=1` |

Notes on these gates:

- **Train coalescing** was a hypothesis from the code seat. It is refuted for W:
  there were 0 `train_coalesced` events. After W, two DF builds did truncate
  with `reason=train_request`.
- **Queue backlog:** in the W `merge_heartbeat` samples (300 s cadence; depth
  counts in-flight + verifier queue + lane buffers), depth ≥3 appeared in 331 of
  997 samples and ≥4 in 124 of 997. The pipeline already holds 2 items in
  flight, so time with a real backlog behind the speculative slot is a minority
  of the time. A chain also has to coincide with a speculative dispatch, which
  happens about 5 times a day.
- **The PRD's premise has drifted (G6).** The PRD's premise as of 07-29 was "the
  queue is verify-bound, mean arrival depth 4.1, 33–52 arrivals/day". Reify in
  October takes about 14 distinct submissions a day, and 85% of enqueues are at
  depth 1.

### 4. The two samples, read individually

- **Deep #1 (7098, 10-22T12:39Z): FAIL.** Built on `_spec-0` in 52.6 s; the tip
  verify ran 2751 s. The failure belongs to item 7098: it failed alone before
  the chain (11:24Z) and twice after it (12:55Z and 14:24Z, the same tests,
  esc-7098-17). So this is **zero chain-attributed failures**, which is the
  PRD's actual safety question.
- **Deep #2 (7190, 10-26T22:43Z → 10-27T01:07Z): PASS, 8643.64 s.**
  - The chain was built in a cold ephemeral lane (`_merge-31303890`, not a
    `_spec` lane) and ran on the local runner.
  - Local speculative verifies in W have a p50 of 6373 s and a p90 of 8504 s
    (n=17). All 36 merge verifies over 7200 s since 09-01 ran on local.
  - So 8644 s is ordinary local-runner speed, not chain overhead.
  - Reify's merge wall is `merge_verify_cold_command_timeout_secs: 10800`, not
    7200.
  - The walk then logged `deep chain walk: task 7266 is no longer buffered —
    stopping after 0 link(s)` and `landed 0 of 1 chained link(s)`. Item 7266 had
    already been verified alone on the laptop, which finished 1 h 25 m earlier,
    and it landed 9 s after 7190 through its own entry.

What the samples support: chains caused no interference (consistent with the
replay study's 57/59), and the chain added no landings. The three rates
themselves are noise at n=2.

### 5. Zero links landed by any walk, on either project (the main finding)

Every passing deep tip since the canaries went live:

| project | dispatching task | items in tip | walk result (syslog) |
|---|---|---|---|
| reify | 7190 (10-27 01:08Z) | 2 | landed 0 of 1: 7266 no longer buffered |
| reify | 8648 (10-28 07:21Z) | 2 | landed 0 of 2: 8648 no longer buffered; a second walk at 07:22:10 landed 0 of 1 (7486) |
| DF | 6102 (10-29 01:17Z) | 3 | landed 0 of 2: 6195 no longer buffered |
| DF | 6175 (10-29 07:19Z) | 2 | no walk line seen (a fleet drain/restart came at 07:28Z) |

The mechanism is described in
`orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._take_chain_link_request`,
whose docstring says: "`_run_inflight_verify` awaits a whole verify between the
build and this walk, and the merger loop is free to `_pop_next_pickable` a
follower during it. The caller MUST stop the walk on None."

The code handles this state, but nothing reserves chained links during the tip
verify. With merge verifies running 0.5–2.5 h, the follower is always popped
first. The walk in `worker.py::SpeculativeMergeWorker._land_chain_prefix` then
stops at link 1.

This breaks the PRD's first boundary row, "Tip pass lands full prefix … one
verify, `landed_via_chain=4`", in production. The ι tests presumably pass
because their merger does not pop followers mid-verify. **Net effect:** a deep
round lands exactly its dispatching item, the same as the adjacent verify it
replaced, after paying for a chain build and often a cold lane. That is the
"neutral week" the DF yaml comment reports.

### 6. Is the predicate sound?

**It does not under-count (ruled out):**

- All 120 `merge_verify` payloads in W carry an int `chain_items`, with values
  {1: 118, 2: 2}.
- Every event since 10-20 has the same ISO 'T' / microsecond / +00:00
  timestamp layout as the script's bounds, so the window edges are exact.
- The script reads the live reify store.
- Re-running with `--now 1793127857` reproduces the gate's line exactly. Run
  now, it gives `n=3, fail 0.33, items 1.0, p90 8643.64 → HOLD`.

It has three biases that matter once n is large enough to read:

- **items_per_verify.**
  - `landed_via_chain` is stamped 1 per landed item: on the dispatching item
    (`_land_advanced_candidate`) and on each walked link (`_land_chain_prefix`).
    The adopted slot-1 head is not stamped.
  - The denominator counts every deep `merge_verify` event, including failed
    tips and retry attempts.
  - So the metric folds in the pass rate. Given §5 it can never exceed 1.0.
- **verify_p90_secs.**
  - With n ≤ 9, `pct()` returns the maximum.
  - The 5400 s bound is "75% of 7200 s", but reify's merge wall is 10800 s, and
    reify's *ordinary* local speculative p50 is already 6373 s.
  - So any deep verify that runs on local fails the gate, whatever the chain
    does.
- **deep_fail_rate.**
  - The 0.35 ceiling comes from a deterministic replay (2/59).
  - The live per-attempt fail rate is about 0.2–0.33: W had 17 head fails out of
    79, and the shipped yaml comment cites about 0.33 flakes.
  - A k-item tip fails if *any* item fails, so a tip should run at or above that
    rate.
  - The ceiling is probably too tight, and it does not separate item-attributed
    failures (like 7098's) from chain-attributed ones.

There is **no minimum-n rule**: n=2 reports three "FAIL"s instead of
"insufficient signal". (The script does treat n=0 as a FAIL, deliberately.)

## Options

Option A, hold as-is (esc pending, 3190 blocked, 3191 gated):
- Nothing changes.
- cap 6 stays live on reify and DF, where it is roughly neutral: it costs some
  chain-build latency (including 120 s timeout stalls) and cold lanes, and adds
  no landings.
- Nobody re-runs the gate, so the HOLD sits indefinitely.

Option B, extend the window or re-anchor the milestone (e.g. 28 days):
- At about 2 deep verifies a week, reify needs 5+ weeks for n≈10.
- `items_per_verify` stays ≤1.0 by construction (§5), so this **only buys a
  later HOLD**.
- Not recommended unless C lands first.

Option C, fix first, then re-anchor:
- File a fix for the walk: reserve a chain's links against `_pop_next_pickable`
  for the duration of the tip verify, or let the walk land a link the pipeline
  holds but has not yet landed. Acceptance test: the "Tip pass lands full
  prefix" row with a live merger that pops followers mid-verify.
- File a fix for the chain build time: reify builds take 38–53 s against the
  PRD's 1–2 s, and 3 of 5 hit 120 s. Root-cause it before raising the timeout
  (cold `_merge-*` lane fallback? CoW seed?).
- Repair the predicate:
  - add a min-n rule;
  - make items-per-verify count per *passed* deep round (or per distinct tip);
  - anchor the p90 bound to the project's merge wall, or compare against the
    same runner's non-chain p90;
  - attribute failures (did the item fail alone?).
- Then re-pend 3190 with a fresh 7-day anchor.
- Cost: PRD-scale work (δ walk plus β build), about 3 tasks. The rollout waits on it.

Option D, question the premise and park the promotion:
- The η2 promotion (cap 6→32) is moot now: the largest chain ever built was 3
  items and reify's backlog rarely tops 4.
- Even a fixed walk would see only about 5–6 eligible rounds a week at
  October's arrival rate.
- Defer 3190 and 3191 on a measurable trigger. One example: "reify
  `merge_heartbeat.depth ≥ 4` in at least 30% of a week's samples, and the walk
  fix (C) landed".
- **Defer, don't cancel.** A cancelled local dependency counts as satisfied
  (`dependency_verdict.py`), so cancelling 3190 would *release* 3191's promotion.
- Leave cap 6 or flip it to 0 (green-tier `reload_config`, no restart).
- Re-scope θ to "is deep merge-ahead worth finishing at current arrival rates?"

**Recommendation: D now, with C's walk fix filed as the gating task.**

- The HOLD is the correct verdict, but for a reason the predicate can't see. The
  question to rule on is no longer "is the canary healthy?" but "is the
  mechanism worth completing?".
- Defer 3190 and 3191 behind (a) the walk fix and (b) an arrival-depth trigger,
  so neither re-fires a meaningless HOLD.
- Keep cap 6 live. It is safe (zero chain-attributed failures in 5 deep rounds
  across both projects) and keeps producing evidence that the walk is broken.
  Flip to 0 only if you'd rather not pay the build stalls.
- Fold the predicate repairs into the same fix batch.

## Open / unverified

- **Slow chain builds on reify (38–53 s, and 120 s timeouts):** not root-caused.
  Two of the built lanes were cold `_merge-<hex>` ephemerals, but the 7098 build
  on `_spec-0` also took 53 s.
- **Unlanded post-W build:** reify built a tip at 10-28 06:29:58 (`depth=2
  built=2 tip=633e563f`) with no matching `chain_items=3` verify. Not
  investigated.
- **Log coverage:** the syslog grep covers /var/log/syslog.1 and /var/log/syslog
  (10-18 → now). Any chain attempts on 10-20 or 10-21 would show there, and none
  did.
- **Reconstruction:** occupancy at dispatch is reconstructed from timestamps, not
  recorded. Chain declines for `queue_len<2`, a non-speculative item, or the
  halving floor are silent: no event, no log.
- **Stated from code, not tested:** a possible ordering hole in the snapshot
  when the in-flight head is non-speculative: S+1 is already popped, so the
  chain is built as S, S+2… That is still sound, but out of submission order.

## Queries (all read-only)

- Predicate:
  `bash scripts/merge-deep-canary-predicate.sh --runs-db /home/leo/src/reify/data/orchestrator/runs.db --window-days 7 --deep-fail-rate-max 0.35 --items-per-verify-min 1.3 --verify-p90-max-secs 5400 --now 1793127857`
  prints `n_deep_verifies=2 … HOLD`, exit 1.
- Deep count:
  `select count(*), sum(json_extract(data,'$.chain_items')>=2) from events where event_type='merge_verify' and timestamp>'2026-10-20T19:04:17+00:00' and timestamp<='2026-10-27T19:04:17+00:00'`
  returns `120|2` (opened with `file:…runs.db?mode=ro`).
- Landed by chain:
  `select count(*) from events where event_type='merge_finalized' and json_extract(data,'$.landed_via_chain') is not null and timestamp>… and timestamp<=…`
  returns `1` (task 7190).
- Enqueue depth:
  `select json_extract(data,'$.queue_depth'),count(*) from events where event_type='merge_queued' and <W> and not (timestamp between '2026-10-26T05:30:00' and '2026-10-26T10:46:00') group by 1`
  returns `{1:111, 2:7, 3:6, 4:3, 5:2, 6:1}`.
- Heartbeat depth in W: `merge_heartbeat.data.depth` gives
  `{1:456, 2:210, 3:207, 4:37, 5:31, 6:39, 7:17}`; `train_coalesced` in W: 0.
- DF reload: `select … from events where event_type='config_reload' and timestamp>'2026-10-27T20:00'`
  on DF's runs.db shows 10-27T22:59:35Z, `merge_deep.chain_cap 0→6`.
- Syslog:
  `grep -a -hE 'build_chain: |deep chain|deep tip verify|acquire_chain_build_lane' /var/log/syslog.1 /var/log/syslog`
  gave 27 lines. Reify and DF are told apart by pid and task id.
- The seat scripts (`/tmp/ro_q/q*.py`, `/tmp/xc.py`) hold the full daily tables
  and the occupancy reconstruction.

---

## Round 2 (2026-10-29 afternoon): what "build" means, warm lanes, the 120 s, and DF's data

### Corrections to round 1
- **The walk *can* land links.** On DF, 6175's walk landed link 3125: main advanced to `f8aa8579` at 10-29 07:28:34. A fleet-drain SIGTERM 1 s later cut off the bookkeeping, so there is no `merge_finalized` or `landed_via_chain` for 3125. Startup recovery reconciled it, and 3125 is `done`. Across both projects, walks landed **1 of 7** links, not 0.
- **There are two removal mechanisms, not one.**
  - On reify, the merger's look-ahead pops the link. Reify had zero coalesce events. 7266 shows it: runs.db 851906/851908.
  - On DF, train coalescing absorbs the link. 6195 was superseded into `coalesce-6163` 4 s after the build snapshot. The same train stacked 6018's branch, which produced the `stacked_on_unlanded_base` truncation.
- **The unexplained `633e563f` build.** It was not a late-finishing timed-out build. It was the next dispatch: 7482's 3-item tip, on a cold local lane. It was **cancelled after 3062 s** when 8648's laptop tip passed and adopted 7482. 8648 was itself one of 7482's links, which the merger had re-dispatched: one chain overtaking another.

### "Build" is lane provisioning plus git merges, never cargo
Everything inside `asyncio.timeout(CHAIN_BUILD_TIMEOUT_SECS)` (`worker.py::_deep_chain_placement` → `build_chain`):
1. Lane acquisition, `GitOps.acquire_spec_lane`:
   - warm `_spec-N` lane: `git reset --hard`, then `git clean -xfd -e target`, then **an unconditional re-seed** (`_seed_warm_lane('--fresh-checkout')` → reify `scripts/seed-warm-lane.sh`). The re-seed is a `cp -a --reflink=always` of the base `target/` plus find/sed/mtime sweeps, under `flock -x -w 30`. Its cost scales with the file count of `target/`. It runs on a loopback XFS image.
   - cold `_merge-<hex>` fallback: `git worktree add` with **no `target/`**.
2. One `git merge --no-ff` per link, about 0.5 s each.

No compile happens in either step. What makes reify slow is the **re-seed of `target/`**:
- GC-sweep re-seeds of `_spec` lanes since 10-20: p50 174 s, p90 450 s. 72% exceed 120 s.
- Today's seed stamps: 219 s and 1029 s.

Every reify timeout was a warm-lane acquisition still seeding when the 120 s expired. 7190's "38 s build" broke down as 4.4 s reset, then a **30 s seed-lock timeout** (rc=124), then a 2.9 s cold worktree, then a 0.7 s merge.

### The 120 s bound was assumed, not measured
- Introduced in `de7b545873` (2026-08-22). Its rationale: "a wedged lane is the failure this catches, not a merely slow one". No measurement backed it.
- The PRD's only cost figure, "~1–2 s per 6-chain", came from the replay study. The study merged into a repo that kept its `target/` and did no lane acquisition.
- The PRD then routed chains through the spec warm-lane pool, which always re-seeds. That cost was never added back into the bound.

### Why 7190's tip verify took 8644 s: a cold lane on a slow host
- The `_spec-1` seed failed on its lock, so the tip ran in `_merge-31303890` with an **empty `target/`**: a full cold cargo build plus tests, with only sccache to help.
- Reify local speculative verifies: **warm p50 5729 s (n=15), cold p50 8504 s (n=5)**. 7190 took 8644 s.
- Host load was the same for both groups (CPU PSI about 55%, runqueue about 4.5), so the difference is the lane, not load.
- The local host is also slow in general: laptop speculative verify p50 is 1592 s.
- The verify was redundant for 7266, which finished its own laptop verify 83 minutes earlier.

### Did chain tips use warm lanes?
| tip | lane | verify runner | warm? |
|---|---|---|---|
| 7098 | `_spec-0` | local | yes |
| 7190 | `_merge-31303890` | local | **no** |
| 7482 (633e563f) | `_merge-55b308ba` | local | **no** |
| 8648 | `_merge-c7410a65` | laptop | n/a: the remote runner builds in the laptop's own worktree and never reads the local lane |

All DF chain lanes were cold `_merge-*`. DF does not set `git.merge_spec_warm_lane_pool`, so the default `False` applies.

### Structural problems the chain arm creates
- **Catch-22.** A warm build almost never fits 120 s. A cold build fits in 2–3 s, but its verify then costs about +2800 s on reify's local host.
- **Lane leak (correctness, live on reify while `chain_cap > 0`).**
  - Why it leaks:
    - `build_chain` calls `acquire_chain_build_lane` *before* its `try/except BaseException`.
    - `acquire_chain_build_lane` catches only `Exception`.
    - `GitOps.acquire_spec_lane` has no cancel or finally path between `pool.try_acquire()` and the seed.
  - Result: a timeout during the seed **leaves the `_spec` lane ASSIGNED until restart**, because there is no reclaimer for spec lanes.
  - Evidence (inferred from logs):
    - after 8558's timeout (10-26 22:42), `_spec-0` was never re-acquired;
    - after 7490's timeout (10-28 06:29), there was no spec acquisition at all until the 10-29 06:28 restart;
    - meanwhile ordinary local speculative verifies 9217 and 6501 ran cold (7182 s and 7754 s).
  - So the canary has slowed reify's **ordinary** pipeline, not just its chains.
- **Pool pressure.** The chain arm takes a local spec lane even when the tip verify is remote.
- **Silent cold fallback.** "Pool exhausted, cold fallback" logs at DEBUG only.

### Not chain-specific: startup Recovery double-uses spec lanes
In 4 of 4 reify restarts examined, `harness.py::Harness._recover_crashed_tasks` logged "lane _spec-N has no plan — releasing back to pool" 3–12 s *after* the merge worker had acquired that lane.

On 10-22 the consequence: 9169's chain build reset, cleaned and re-seeded `_spec-0` at 12:30:35Z, while 7098's tip verify had been running in it since 11:54Z. This is now **task 7745** (high priority).

### DF since the 10-27T22:59Z reload (1.6 days)
- **Funnel:** 10 speculative verifies, then 4 chain builds (1.8, 2.5, 6.9 and 31 s; **no timeouts**, since there are no seeds on cold lanes), then 2 deep verifies. Both passed: 6102 (3 items, local 2845 s) and 6175 (2 items, laptop 1013 s).
- **Cost:** a deep tip costs about the same as an ordinary verify there (local mean 3078 s, laptop 1160 s). The builds stalled dispatch for about 42 s in total.
- **Trains are DF's main preemptor.**
  - 2 of 4 builds returned `built=0` with `reason=train_request`.
  - A third was truncated, and had its first link absorbed, by a train forming mid-build.
  - Only 1 of 4 attempts was clean, and that one landed its link.
- **Throughput can't be read.**
  - The laptop was quarantined for 54% of heartbeats, against 28% the week before.
  - Arrivals fell by a third: 42 a day against 62.6.
- **No chain-attributed red main** and no rebase-conflict cluster, so the stop rule did not fire.

### Task updates
- **7720 amended:** covers both removal mechanisms, the ordering hole, chain overtaking chain, and SIGTERM-durable link bookkeeping.
- **7721 amended, priority now high:** root cause is the seed inside the bound. It must fix the lane leak on cancel and skip the local lane for remote tips.
- **7745 filed:** Recovery double-use.
- **7722** is already in progress and was not touched.

---

## Round 3 (2026-10-30): timeout value, filed fixes, and items for Leo

### Corrections to rounds 1–2
- **"A deep round lands one item" and "items_per_verify can't exceed 1.0" were wrong.** A passing tip also lands the slot-1 head on its authority. Over the 4 passing tips in the current window: 7190 (1 landing), 7482 + 8648 (2), 6102 (1), 5626 + 6175 + 3125 (3). That is 7 landings, 1.75 per passing tip. The predicate counts 5 (1.25), because adopted heads and the SIGTERM'd link are unstamped.
- **"Chain overtakes chain" is not waste.** 7482's cancelled local cold verify (3062 s) would have run far longer. The overtaking laptop tip landed it sooner.
- **Round 1's "the HOLD is correct because the walk lands nothing" overstated it.** Still, only 1 of 7 chained links landed by walk.

### Timeout value
Whole-process basis, reify: seed p50 148 s, p99 899 s, max 1272 s (`data/verify-logs/verify-runs.jsonl`, distinct seed_id, n=401, 10-15→10-30). Last 7 days p99 1003 s; ISO week 44 p99 1168 s. Add reset/clean ≤28 s, lock wait ≤30 s, merges ≈0.5 s/link: whole-process p99 ≈ 1230 s, worst observed ≈ 1335 s. Cross-check: reset→verify-start on ordinary `_spec` acquires, n=28, p99 719 s, max 760 s.

Decision: knob `merge_deep.chain_build_timeout_secs`, default **1800 s**, DF yaml **300 s** (8 builds, max 31 s, no seed). Carried as task 7809.

### Filed
- 7809 timeout knob (hand-carried).
- 7721 re-scoped: lane leak on cancel; a remote tip takes no warm lane or seed; visibility.
- 7810 yield undercount: stamp adopted heads; link landing survives SIGTERM.
- 7745 startup Recovery double-use (from round 2).
- 3190 now also depends on 7809 and 7810.

### Items for Leo
See the session reply for the full mechanism / impact / options per item; recorded here in brief:
1. GC sweep re-seeds every FREE lane endlessly (2,158 resets in 5.5 days, busy 97%, 23 lane-hours/day, 4–7 CPU-hours/day) for a seed the next acquire throws away. Rec: reclaim by delete.
2. 7720 design: reserve links vs adopt links on tip pass. Rec: reserve, bounded.
3. Inline build stalls `_verifier_loop` up to the new timeout. Rec: move off-loop with 7720.
4. Where tips run: local tip median 5729–8504 s vs laptop 1592 s. Rec: chain only on a remote lease on reify.
5. S+1 ordering hole. Rec: defer the look-ahead pop until S has dispatched.
6. A 30 s seed-lock wait falls back to a cold verify (+2775 s median). Rec: wait longer.
7. Seed cost growing: p99 562→904→1168 s over 3 weeks. Rec: file a /deb study; alert at 80% of the timeout.
8. Deployment: watchdog `ORCH_RESTART_MIN_INTERVAL_SECS=259200` (3 days), not 8 h; nothing above is live until about 11-01 without a manual sweep. DF builds 6 of 8 `train_request`, coalesce-off not yet live.
9. Gate sample: 4 deep verifies in 7 days vs min 8. Rec: 14-day window.

---

## Round 4 (2026-10-30): what the warm-lane GC is for (seat G; evidence in /tmp/3190g/)

Correction to Round 3 item 1: "the next acquire throws the seed away" is false for 57% of task-lane acquires. `git_ops.py::GitOps._reuse_warm_lane` never seeds; routes in syslog are recycle 65, reset_in_place_reattach 54, reuse 30. Reclaim-by-delete alone would make those cold. Also, Round 3's volumes were one arm only: a second arm, the orchestrator's `warm-lane-gc` BackgroundService, runs the same script concurrently and its output is discarded.

- Purpose: stop lane `target/` accretion reaching ENOSPC (incidents: 06-20→22, 06-26/27, 07-10, reify 5326). Pass 1 resets, Pass 2 orphan worktrees, a `.reseed-trash` reaper, an emergency `--disk-pressure` arm.
- Useful work: 175 of 2144 timer resets (8%) followed lane activity, freeing about 19.6 GiB each, about 607 GiB/day. The other 92% free nothing (mean −0.34 GiB, SE 0.87). Pass 2 removed 0. Emergency arm fired 0 times (floor 2 GiB; min free since 08-17 is 1603 GiB).
- Population: not idle lanes. All 56 records say `assigned`; reify's script copy ignores the record, so it resets lanes of live tasks (about 20/day between phases). The DF copy has the record gate (3075), but cutover 3078 is still pending.
- Harm: lock timeouts forcing cold verifies (4 in window, holder inferred), 5.5+ CPU-h/day, warm state destroyed.
- If stopped: both arms off, about 2.6–5.2 days to the floor at lost-reclaim rates, 1.1 days at the worst-observed decline; plateau estimate leaves about 0.9 TB free, so ENOSPC is plausible, not proven. On-demand disk guard is OFF on reify, so nothing else catches it short of the 10 GiB merge-verify skip.
- Options: see the session reply and seat G's table. Lead's recommendation: skip-unchanged-and-same-gen in reify's script, reduce the timer arm to trash + emergency, log the orchestrator arm's summary; then measure before considering delete + seed-if-absent or the 3078 cutover.

---

## Round 5 (2026-10-30): /deb on seed-cost growth (seat S; evidence in /tmp/3190s/)

Clone p50 W40→W44: 30.8→118.6 s, ×3.85 = ×1.46 (units) × ×2.63 (ms per unit). Total p50 rose 105 s, 83% of it clone.

- Part A, established: the base accumulates cargo units with new hashes (13,883→22,415 units, +61%). `reify scripts/prune-unconsulted-target.sh` removes only depth-1 deps, debug only; `.fingerprint` is never deleted and dead `.dwo` names fail its grammar. 107,680 of 164,255 entries in gen.602 (66%) belong to units unconsulted for 7 days. cp and the background rm walk them per file.
- Part B, located but onset unexplained: each file costs about one sleep; under verify load that sleep is 0.75–1 ms in xfs_buf_lock/iowait, mostly in the buffered loop layer. On 10-26, same gen and units: 27–31 s with no verifies, 144–168 s either side. Candidates not separated: saturation of the loop path, the deep-merge canary from 10-20, base staleness.
- Refuted as drivers: extent fragmentation, XFS free space/metadata depth, image fragmentation, fstrim, CPU load, the sweep (duty flat at 23 h/day), concurrency, script changes, profile change.

Timeout check: seeds not matching a sweep-logged seed, W44, p50 221, p90 696, p99 1223, max 1272. Whole-process p99 about 1285 s, so 1800 s still clears it.

History the seat did not cover, added by the lead: reify 7426/8352/8366 (a deps prune cut live variants; Leo 10-08 "CPU is much more scarce than space", reverted first), 9520 (pending: the 7-day liveness rule still deletes units lanes build), 8361 (pending: base prune frees no real disk while _merge-verify holds the extents). So "dead by the 7-day rule" is contested evidence for anything with deps still present. Fix 1 (remove fingerprint + .dwo of units whose deps are already gone) has no such rebuild exposure; fix 2 (release prune) does.

---

## Round 6 (.fingerprint trace; seat F)

Answer: CONDITIONAL. A dead compile unit's `.fingerprint` need not outlive its deps (12,065 of 12,199 dead units); 134 dead build-script units must keep theirs, because their `build/<unit>/` outputs survive and cargo can reuse them. `.dwo` never needs to: cargo doesn't track it, and 0 of 1,817 sampled references from live binaries hit a dead unit. Removes 80,118 of 164,254 entries (48.8%) with no added compile. No correctness reason for the old rule was ever stated; the recorded reasons are bytes (7426), change size, and the evidence signal, which doesn't apply to a base staging copy because each refresh re-imports from `_merge-verify`. Not run against real cargo: reify 9614 must confirm in a scratch target.

Filed: reify 9614 (fix 1), 9615 (fix 2, conditional, after 9520 and 9614), 9616 (pruned-gen experiment, delayed 3 days), 9617 (human gate: GC-off window).

---

## Round 7 (warm-lane GC cutover plan; seat C; pinned at DF a1eb49ebc3, reify 75c533fcea)

Leo's goal (most resets stop; no between-phase reset of live tasks) does not need 3076, 3077 or κ. The benefit comes from the record gate already in DF's copy (`warm-lane-gc.sh::_do_reclaim`, task 3075): `lane_state_class` maps assigned and in_use both to ASSIGNED, preserved up to 14 days (task 5504), reseeded past that, and only `--disk-pressure` reaches a younger one. PRD B4 as written would UNDO the benefit, and 3077 would close the 5504 emergency valve; both must not ship as written. 3078's item 1 was a real chicken-and-egg (project copy wins until κ, κ depended on 3078), so a reversible preference knob flips first.

B alone doesn't stop between-phase resets; the cutover alone gives about a 48% cut. Both are needed.

Chain: 7857 logging → 7862 port (discard, gen pin) → 7859 B+F in DF copy; 7863 record freshness on reuse; 7864 preference knob; reify 9611 B+F, reify 9618 soft floor → **7865 human gate + flip** → 7866 week's re-measure → 7867 human reassessment incl. A+. → 3078 rewritten as the post-soak gate for κ (14 days after 7865) → reify 5614, 5740, 5616. 3077 now waits on 3078 and needs a design pass.

Risk: accretion is bounded, not prevented. 43 preserved lanes × 62 GiB leaves about 640 GiB; × 113.6 GiB runs 1,580 GiB short. Turnover (32 of 56 records re-keyed in 1.5 days) likely settles it; only the soak confirms. Guards: soft floor on (500 GiB), emergency floor raised 2 → 200 GiB, disk guard and release-thin stay off. 7862 and 7859 go live on merge for eight other units running DF's copy from the main checkout.

---

Seat analysis scripts and key tables are saved in `plans/3190-deep-canary-few-verifies-2026-10-29/` (seed-study, gc-study, acquire-study). The scripts are one-off evidence, not maintained code, so they carry a `.py.txt` extension to stay outside lint and whole-tree `.py` scans; run one with `python3 -I <file>`. They expect the /tmp extracts. Raw log extracts stay in /tmp/3190{b,c,d,g,s} and are reproducible from syslog while it is retained.
