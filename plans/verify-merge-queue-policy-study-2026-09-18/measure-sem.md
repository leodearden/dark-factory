# Seat measure-sem — the task-verify admission queue, measured

Read-only study, 2026-09-18. No tracked file edited, no test suite or verify
command run, no MCP write, every sqlite connection opened
`file:...?mode=ro`. All scripts in `scripts/`, all datasets in `data/`.

Label on every number: **[M]** measured by me from a store/log now ·
**[D]** derived by arithmetic from [M] numbers (the arithmetic is shown) ·
**[E]** estimated with a stated assumption.

Host during measurement: 32 threads, load average 80–380 throughout (the
figure every historical duration below was recorded under).

---

## 0. Headline

| # | Finding | Confidence |
|---|---|---|
| 1 | The contradiction resolves as **(a) the config comment's number is wrong**, and by a factor of **6.3**: the slot-gated task-leg volume is **107.5 legs/day** on DF, not the ~240 legs/14 d (17.1/day) `dark-factory-orchestrator.yaml` asserts. Offered slot work is **≥16.4 h/day**, i.e. **ρ ≥ 0.68**, not 23 %. [M/D] | high |
| 2 | **Mean wait is invariant across FIFO, random and LIFO.** On the measured 578-leg trace, replaying random order gives mean wait 0.59 h and FIFO 0.57 h — a 3 % difference, inside replicate noise. FIFO buys **tail** only: p99 5.69 h → 3.77 h, max 10.15 h → 4.12 h. Leo's hunch that FIFO would improve *mean* task duration is **refuted at the semaphore**; the hunch about *variance* (and therefore about tasks dying before the restart) stands. [M/D] | high |
| 3 | **SJF halves the mean and is implementable today.** Same trace: mean 0.59 h → 0.29 h, p50 0.26 → 0.13 h, p90 1.54 → 0.54 h. It is implementable because the discriminator is already computed and logged *before* the leg runs: `(module, scope_kind)` explains **62.2 %** of service-time variance (module alone 25.7 %), and the extreme cells differ 45×: `orchestrator/full_suite` mean 1959 s vs `orchestrator/file_scoped` mean 43 s. SJF ≈ SRPT here (identical to 2 d.p. on cell means), so nothing is lost to prediction error. [M/D] | high |
| 4 | The **random winner is confirmed directly**: among 2 801 observed contending pairs, the later arrival was served first **50.3 %** of the time — exactly uniform-random, and 0 % under FIFO. [M] | high |
| 5 | **Reify is the saturated one and SJF cannot help it.** Reify offers ~62 gated legs/day at ~1 300 s each → **ρ ≈ 0.93** [D], and every leg is the *same* class (one unscoped whole-workspace `&&` chain): service CV² = **0.15** vs DF's 3.18. There is no short job to promote. Reify needs capacity or demand reduction, DF needs ordering. [M/D] | medium |
| 6 | **Reify's own bash semaphore is already redundant.** Its 58 recorded `test_slot_starvation` waits (2026-08-20 → 09-18) have p50 **3 s**, mean 183 s, max 1 686 s — because the DF-side flock serialises reify's verifies *before* they reach it. Reify's real queue is the DF-side flock, which emits nothing. [M] | high |
| 7 | **A factor of ~5 in observed wait is NOT explained by the measured trace.** On the *same* 112 legs, observed mean wait is 5.01 h but the replay of the complete arrival census gives 0.62 h. Since mean wait is discipline-invariant, the only possible cause is **more work on the slot than the census sees** — ρ_true ≈ 0.94 rather than 0.74. ~4.8 h/day of slot occupancy is unattributed. This is the biggest open question and it changes the *magnitude* of every policy gain, not its sign. [M/D] | high that the gap is real; low on its cause |
| 8 | The admission slot dominates the DF verify phase: verify-phase wall p50 **5.26 h**, of which agent time is **1.0 %** and instrumented verify service **1.7 %**; **97.3 %** is non-agent, non-service residual. [M] | high |
| 9 | Priority tiers are *not* equally served, though nothing in the semaphore intends it: wait p50 by tier is critical **393 s** < high **1 539 s** < medium **3 405 s** < low **7 861 s** [M]. Monotone, but confounded (see §7) — do not read it as "priority already works". | medium |
| 10 | **11.7 % of DF verify phases and 23.2 % of reify's end in `cancelled`**, and across the 6-day window DF logged 48 `cancelled` dispatches against 47 `done`, reify 132 against 73. [M] | high |

---

## 1. Window, and why

`verify_admission_task_slots` is the study's subject, so the window must not
straddle a change to it or to anything that changes slot-hold length.
From `git log` on `dark-factory-orchestrator.yaml` [M]:

| when (UTC) | change |
|---|---|
| 2026-09-08T10:17Z | `verify_admission_task_slots` **3 → 1**; merge `-n auto` pinned 16 |
| 2026-09-10T19:05Z | model/budget routing changes |
| 2026-09-11T16:05Z | `max_concurrent_tasks` **48 → 24** |
| 2026-09-12T07:24Z | **verify budget 3600 → 7200 warm, 5400 → 10800 cold** |
| 2026-09-12T07:40Z | per-test pytest timeout 60 → 300 |
| 2026-09-17T11:39Z | merge-leg `PYTEST_XDIST_AUTO_NUM_WORKERS` 8 → 16 (merge leg only, ungated) |

The 09-12 budget raise triples the worst-case slot hold, so it is the binding
boundary. **Event-store window: 2026-09-12T08:00Z → 2026-09-18T08:00Z (6 d).**
The syslog-derived arrival census starts later because `/var/log/syslog`
rotated on 09-13: **2026-09-13T00:10:48Z → 2026-09-18T09:11:12Z (5.38 d)**.
Both are inside one regime. Quoted "all time" figures span 2026-05-09 →
2026-09-18 and are labelled as such.

Reify's own config sets `REIFY_TEST_SEMAPHORE_CONCURRENCY: "1"`,
`REIFY_TEST_SEMAPHORE_WAIT: "unlimited"`, `max_concurrent_tasks: 24`,
`concurrent_verify: false`, and never sets `verify_admission_task_slots`, so it
takes the code default 1 (`config.py::OrchestratorConfig`). [M]

Both slot dirs exist and both `slot-1` files were **HELD** when I probed them
(non-blocking `flock -n`, no write): `/tmp/df-verify-slots-1000-dark_factory`
(DF's explicit override) and `/tmp/df-verify-slots-1000-65bc7aa5cef0`, which is
`sha256('/home/leo/src/reify')[:12]` per
`config.py::_default_verify_admission_slots_dir`. Five further hash dirs exist
for the other registered projects; DF's own hash dir does not, confirming the
override took. **The two projects do not share a slot.** [M]

---

## 2. Where the data is, and what each source cannot tell you

There is **no event, anywhere, for a slot acquisition.** I enumerated every
`event_type` in both stores over 09-04 → 09-18 (67 kinds in DF, 56 in reify):
none mentions admission. `verify.py::run_verification` stamps `started_at`
*inside* `async with _admission_slot(...)`, i.e. **after** the slot is won, so
the wait is invisible to `workflow_verify`, `rebase_verify_cost` and
`merge_verify` alike. `rebase_verify_cost.data.next_verify_wall_secs` is
`VerifyResult.duration_secs` (`workflow.py::_emit_rebase_verify_cost`), i.e.
service only.

Three usable sources, each with a different hole:

1. **`.worktrees/*/.task/verify/attempt-*[.<module>].summary.json`** — 916 DF
   files across 555 worktrees; 33 reify files. Carries per-leg
   `started_at` + `duration_secs`. **Overwritten**: `workflow.py` passes
   `attempt_id=verify_attempt + 1`, which resets to 1 on every re-entry of the
   verify phase, so only the last attempt per (worktree, attempt id, module)
   survives. Reify's `.worktrees/_lane-N` are **symlinks** into
   `/home/leo/src/warm-lanes`, so a plain `os.walk` finds nothing — `find -L` /
   `Path.glob` is required, and because lanes are recycled only the current
   occupant's attempt exists.
2. **`data/verify-logs/<task>/attempt-*.summary-<ts>.json`** — 224 DF, 65 reify.
   Timestamped, so these accumulate. But archival is gated on
   `verify_categories.should_archive` (via `verify.py::_should_archive_category`),
   so it is **failure-only for task legs** — and stronger than the project
   memory note says: **all 224 DF archive records have wait exactly 0**, because
   they are all merge-path, where no leg is gated. The archive contributes
   **zero** task-lane wait data.
3. **`/var/log/syslog`** — `run_scoped_verification` logs
   `Verify plan: {…}` (a python dict repr) per call, listing every per-module run
   with its tool and `scope_kind`. Every `cmd.tool == 'pytest'` run is exactly
   one `acquire_task_slot`. This is the only **census** of arrivals. Truncated at
   ~8 122 chars (96 of 510 DF lines; all of them ≥8-module merge plans).
   journald shows no `Suppressed … messages` lines and `journald.conf` is stock,
   so no evidence of dropped lines.

### The wait proxy, and the one trap in it

`run_verification` launches the three legs in one
`asyncio.gather(test, lint, type)` under `concurrent_verify: true`, test first;
only the test leg takes the slot. So

> `wait = test.started_at − min(lint.started_at, type.started_at)`

is the arrival→start wait, tight to within event-loop scheduling. **Validated
independently**: of the observed legs whose derived arrival falls inside the
plan-log window, **119/122 (98 %) match a `Verify plan:` line within 120 s, with
|Δ| p50 = 0 s, p90 = 7 s** [M]. The proxy's arrival *is* the plan timestamp.

**The trap.** `run_verification`'s bounded env-recovery retry re-runs only the
test command and rebuilds the attempt as
`VerifyAttempt([new_test, attempt.lint, attempt.type])` — keeping the **first**
pass's lint/type `started_at`. On such a record the proxy measures the first
pass's whole duration (up to the 7 200/10 800 s budget) plus the retry's own
wait. They are identifiable: `_serial_pytest_str` rewrites the recovery command,
so the test `cmd` carries `-p no:xdist -o addopts=`. **28 of 897 DF records
(3.1 %)**; their wait mean is **29 377 s** vs **10 901 s** for the 869 clean ones
[M]. Dropped everywhere below. It does **not** explain the headline — but any
future study using this proxy must drop them.

**The proxy is unusable for reify.** Reify sets `concurrent_verify: false`, so
its legs run **sequentially, test first**: its 33 lane records have
wait ≈ −service (mean −1 394 s) [M]. Its 45 archive records show wait ≈ +170 s,
which is the *lint* duration under the merge role's `sequential_lint_first`, not
a slot wait. **Reify's DF-side admission wait is unobservable in any artefact
that exists today.**

---

## 3. (1) Slot utilisation and holder mix

**Arrival census, DF** (`scripts/60_load.py`, 09-13 → 09-18, 5.38 d) [M]:

| | n | per day |
|---|---|---|
| slot-**gated** pytest runs (task + background plans) | **578** | **107.5** |
| merge-gate pytest runs (never acquire — C-merge-priority) | 663 | 123.3 |

The merge/task split is by plan width: a plan naming ≥8 distinct
`module_prefix` values is a `merge_verify_breadth: full` gate. Cross-check:
99 such plans vs **89 `merge_verify` events** in the same span [M] — the two
agree, and 92 of the 99 are the truncated lines.

**Service, DF** — from the 914-leg disk sample, weighted by the *arrival* joint
mix over `(module, scope_kind)` (17 cells, 100 % of arrivals covered):
**E[S] = 548 s, CV² = 3.18** [D from M]. Unweighted pooled mean 514 s, p50 86 s.

> **ρ = 107.5 × 548 / 86 400 = 0.68**, i.e. **16.4 h/day** of offered work on a
> 24 h/day single slot. [D]

Plus, on the *same* slot but absent from the census:

- **7 `run_full_verification` fan-outs** in the window (`Full verification:
  running 9 subprojects in parallel` [M]) — the main-tip sweep at
  `role='background'` (`main_tip_sweep_interval_secs` default 1800, never
  overridden; the SHA-dedup gate means it fires far less often than that) plus
  review checkpoints. Each is 9 **unscoped** module legs; summing my measured
  `full_suite` cell means gives **≈0.8 h of slot per fan-out**, ≈1.3 h with the
  retry-on-flake second pass. → **≈1.0–1.7 h/day, ρ +0.04…0.07** [E].
- **15 `Verification mode: subproject-scoped` verifies** [M], service unmeasured.
- Zero `Verification mode: per-subproject fan-out` lines in the window [M], so
  the branch in `run_scoped_verification` that runs every module config without
  logging a plan never fired — it is not a hidden consumer here.

**Accounted total ρ ≈ 0.72–0.74.** Holder mix: task-role legs dominate
(107.5/day vs ≈1.3/day background-sweep legs), so the
`run_main_tip_sweep`-shares-the-task-pool trade-off recorded in
`verify.py::run_full_verification` costs ~5 % of the slot, not the majority.

**Reify** [M/D]: `Verification mode: global` lines number 85/99/83/80/57 over
09-13…09-17 = **80.8/day**; subtracting the 18.5 merge gates/day
(`merge_verify` n=111/6 d) leaves **≈62 gated legs/day**, each ONE segmented
whole-workspace chain. Service: 1 394 s (lane sample n=33, CV² 0.15) or 1 209 s
(archive n=45, failure-biased). → **ρ ≈ 62 × 1 300 / 86 400 = 0.93** [D].

**Sampled occupancy, for a floor that assumes nothing**: the union of the
observed legs' `[start, start+service]` intervals in the 6-day event window is
**40.6 h / 144 h = 28.2 %** for DF and 16.9 h = 11.7 % for reify, on 161 and 47
sampled intervals respectively [M]. Overlap among sampled DF legs is 2 961 s —
non-zero, which is expected: the sample spans a period that includes the
3-slot regime and legs from the ungated merge path in `_iact`/`_merge` trees.

---

## 4. (2) Service time and how predictable it is in advance

DF, worktree sample, env-recovery records excluded [M]:

| module | n | mean | p50 | p90 | CV² |
|---|---|---|---|---|---|
| orchestrator | 335 | 1 147 s | 564 | 3 361 | 1.37 |
| fused-memory | 243 | 233 | 143 | 633 | 1.16 |
| scripts | 95 | 132 | 121 | 265 | 0.92 |
| shared | 77 | 96 | 56 | 249 | 1.52 |
| tests/scripts | 64 | 42 | 22 | 100 | 1.89 |
| escalation | 43 | 50 | 34 | 141 | 1.08 |
| dashboard | 42 | 81 | 40 | 165 | 2.14 |
| cockpit | 13 | 37 | 25 | 79 | 0.71 |
| sampler | 2 | 49 | — | — | 0.53 |

Split by the class the **planner already knows and logs**:

| module × class | n | mean | p50 | CV² |
|---|---|---|---|---|
| orchestrator × full_suite | 193 | **1 959 s** | 1 784 | 0.41 |
| orchestrator × file_scoped | 142 | **43 s** | 18 | 6.89 |
| fused-memory × full_suite | 158 | 337 | 263 | 0.57 |
| fused-memory × file_scoped | 85 | 38 | 30 | 0.45 |
| scripts × full_suite | 57 | 187 | 168 | 0.37 |
| shared × full_suite | 50 | 70 | 57 | 1.08 |
| tests/scripts × file_scoped | 52 | 24 | 11 | 1.05 |
| scripts × file_scoped | 34 | 22 | 11 | 2.68 |
| dashboard × full_suite | 19 | 121 | 76 | 1.10 |

**Variance of service explained: module alone 25.7 %, module × class 62.2 %**
(n=914) [D]. Both labels come free — they are in the `plan` object
`run_scoped_verification` derives and logs before it spawns anything, and
`verify_plan` already carries `scope_kind` per run. **A shortest-expected-first
admission order needs no new measurement infrastructure, only the ordering
hook.**

Reify: one class, CV² **0.15**, variance explained 0 %. [M]

---

## 5. (3) Wait, queue length, burstiness

DF observed wait, 869 clean records [M]:

| sample | n | mean | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| all time (2026-05-09 → 09-18) | 869 | 3.03 h | 0.99 h | 3.54 h | 8.25 h | 27.6 h | 44.8 h |
| in 6-day window | 124 | **4.85 h** | **3.54 h** | 6.26 h | 10.8 h | 25.4 h | 28.4 h |

`wait > 60 s` in 712/869; `> 1 h` in 434; `> 6 h` in 120; `> 24 h` in 14. [M]
This **reproduces the prior study's p50 2.41 h / p90 12 h** (F-demand.md) rather
than refuting it — the wait is real and, in the current regime, worse.

By module, the wait is **uncorrelated with that module's own service** — `scripts`
(mean service 132 s) has the *longest* mean wait, 23 647 s, and `orchestrator`
(mean service 1 147 s) the shortest, 7 964 s [M]. That is the signature of a
size-blind queue: small jobs inherit the queue's aggregate backlog.

**Burstiness** (gated legs only, `scripts/` inline) [M]:

| bin | mean | var | index of dispersion | max |
|---|---|---|---|---|
| 10 min | 0.75 | 1.33 | **1.79** | 8 |
| 1 h | 4.45 | 10.28 | **2.31** | 15 |
| 6 h | 26.27 | 100.11 | **3.81** | 49 |

IDC rises with the aggregation scale — long-range load variation, not point
bursts. **Hypothesis (d) is not supported**: no 10-minute bin holds more than
8 arrivals, so a fleet restart does *not* re-queue every in-flight verify at
once. Arrivals come in small batches (one plan = one batch): batch size 1 in
293 of 407 plans, E[X] = 1.42, E[X²]/E[X] = 1.82, inter-batch gap p50 724 s,
CV 1.08. [M]

Queue length at arrival is not directly recorded; the replay in §6 reconstructs
it from the census.

---

## 6. (4) The random winner, and what ordering actually buys

**Direct evidence.** Among the 869 observed acquisitions there are 2 801 pairs
where *j* arrived while *i* was still waiting. The later arrival was served
first in **1 410 of them = 50.3 %** [M]. Uniform random predicts 50 %; FIFO
predicts 0 %. Observed wait CV = 1.75.

**Replay** (`scripts/80_replay.py`) — the same 578-leg census through one
server, four disciplines, service from the empirical `(module, scope_kind)`
cells; `random` models the 0.1 s shuffle-poll loop in
`verify_admission.py::_acquire`/`_try_once`:

service = cell mean:

| discipline | mean | p50 | p90 | p99 | max | ρ |
|---|---|---|---|---|---|---|
| random | 0.59 h | 0.26 h | 1.54 h | 5.69 h | 10.15 h | 0.68 |
| **fifo** | 0.57 h | 0.37 h | 1.39 h | **3.77 h** | **4.12 h** | 0.68 |
| **sjf** (expected service) | **0.29 h** | **0.13 h** | **0.54 h** | 3.25 h | 3.93 h | 0.68 |
| srpt (actual service — unachievable bound) | 0.29 h | 0.13 h | 0.54 h | 3.25 h | 3.93 h | 0.68 |

service = bootstrap resample from the cell: random 0.75/0.31/1.92/6.63/9.76;
fifo 0.73/0.49/1.67/4.33/4.75; sjf 0.38/0.18/0.88/3.79/4.74; srpt
0.32/0.11/0.83/2.64/5.50. [M/D]

Three things fall out, and they are the policy answer for this seat:

1. **FIFO cannot move the mean** — it is discipline-invariant for any
   work-conserving, size-blind, non-preemptive order. Measured difference 3 %,
   and the *median* actually gets **worse** (0.26 → 0.37 h) because FIFO
   removes the lucky-short-wait draws. What FIFO removes is the tail:
   **p99 −34 %, max −59 %**. Given §8 (a dispatch is as likely to be cancelled
   as to finish, and cancellations cluster at restarts), tail removal is worth
   having on its own terms — but it must be sold as a variance and
   survival-probability change, never as a throughput or mean-duration change.
2. **SJF halves the mean** (−51 %) and cuts p90 by 65 %, and it does so with a
   62 %-accurate predictor that already exists. SJF ≈ SRPT to 2 d.p., so the
   prediction error costs essentially nothing. Its cost is the classic one:
   `orchestrator × full_suite` (1 959 s, 130 of 578 arrivals) is the job class
   that gets deferred, and p99 stays ~3.3 h.
3. **Neither changes ρ**, so neither changes throughput at the semaphore.
   Any throughput claim has to come from second-order effects (less rebase
   drift, fewer restart kills), which this seat did not measure.

---

## 7. (5) Priority

The semaphore is priority-blind by construction (`_try_once` shuffles slot
*indices*, never waiters). Joining the 869 observed acquisitions to
`tasks.priority` in the live store (`.taskmaster/tasks/tasks.db`, columns from
`scripts/tasks_db_schema.py`; 0 unmatched holders) [M]:

| priority | n | mean wait | p50 | p90 | max |
|---|---|---|---|---|---|
| critical | 16 | 0.39 h | **393 s** | 1.95 h | 2.56 h |
| high | 289 | 2.36 h | **1 539 s** | 7.05 h | 28.4 h |
| medium | 304 | 2.45 h | **3 405 s** | 6.34 h | 32.4 h |
| low | 260 | 4.61 h | **7 861 s** | 11.8 h | 44.8 h |

Store-wide priority mix: medium 2 046, low 2 043, high 1 371, critical 69,
polish 18 (n=5 547). [M]

The medians are perfectly monotone in priority, which is surprising for a
priority-blind queue — **and it is confounded, three ways**:

- `priority` is read **live**, not as-of the acquisition, so overrides since
  then are invisible.
- The sample is one leg-set per surviving worktree. A low-priority task lingers
  longer, so its *last* surviving attempt is more likely to have landed in a
  saturated epoch. Era, not tier, could produce this.
- Priority correlates with scope: a critical task is more often a small
  file-scoped fix, and a file-scoped leg arrives in a short batch.

So: **high-priority tasks do wait less today, but there is no measured evidence
that this is anything other than selection.** A priority policy would be
starting from ~4:1 (low:high) at the median, not from parity.

---

## 8. (6) Share of end-to-end duration; kills

DF verify-phase decomposition, 256 paired `phase_enter/phase_exit(phase='verify')`
intervals in the 6-day window (`scripts/30_verify_phase_decompose.py`) [M]:

| component | sum | mean | p50 | p90 |
|---|---|---|---|---|
| wall | 1 951.8 h | 27 448 s | **18 930 s (5.26 h)** | 65 495 s |
| agent (union of `invocations` spans) | 19.0 h | 267 s | 0 s | 967 s |
| instrumented verify service (`rebase_verify_cost`) | 33.2 h | 467 s | 0 s | 2 230 s |
| **residual** | **1 899.7 h** | 26 714 s | 18 300 s | 63 202 s |

**agent 1.0 % · verify service 1.7 % · residual 97.3 %.** Reify is the same
shape: wall p50 15 631 s, agent 0.3 %, service 1.8 %, residual 97.8 %.
Only 39 debugger and 11 steward invocations appear inside DF's 256 verify
phases, so the debug-fix loop is not where the phase goes.

Acquisitions per task in the surviving sample: mean 1.68, p50 1, p90 3, max 8;
summed wait per task mean 5.08 h, p50 1.97 h, max 70.0 h (n=518 tasks) [M].
Both are **floors** — the sample keeps only the last persisted attempt.

Slot-wait share of dispatch wall is **not computable from these stores**: only
46 DF dispatches produced a `task_results` row in the 6-day window (mean wall
2.19 h, p50 0.68 h) against 290 verify-phase entries, and summed wait per task
spans *multiple* dispatches while `duration_ms` covers one. The naive join gives
p50 = 100 % (capped), which is an artefact and should not be quoted.

**Kills.** Phase after a verify phase [M]: DF review 239 / **cancelled 32** /
escalated 2; reify review 262 / **cancelled 81** / escalated 6. `task_completed`
outcomes in the window: DF cancelled 48, done 47, requeued 34, merge-deferred
30, blocked 28; reify cancelled 132, done 73, requeued 59, blocked 34. **A
dispatch is as likely to be cancelled as to finish.**

Restart interval is **not 8 h**: `runs` rows start at 2026-09-12T09:39,
09-14T12:31, 09-15T23:00 for DF (≈51 h then ≈34 h) and five times for reify [M].
Cancellations cluster at those boundaries (e.g. 32 DF verify phases exited to
`cancelled`, with a visible cluster at 2026-09-15T23:00:00). Against a verify
phase whose p50 is 5.26 h and p90 18.2 h, a 34–51 h restart interval is not the
binding constraint it was when `max_concurrent_tasks` was 48 — but the
cancellation rate says something else is killing dispatches at that rate, and
this seat did not identify it.

---

## 9. The residual, stated plainly

The single result a simulator consumer must not skip.

Matching observed acquisitions to their census arrival within 120 s gives 112
pairs. On **those same legs** [M/D]:

| | mean | p50 | p90 |
|---|---|---|---|
| observed | **5.01 h** | 3.59 h | 11.51 h |
| replay of the census (random order) | **0.62 h** | 0.46 h | 1.50 h |
| replay over all 578 census legs | 0.56 h | 0.23 h | — |

A factor of **8.1** on the mean. It is **not** sample bias — the comparison is
leg-for-leg. It **cannot** be ordering — mean wait is discipline-invariant
(§6). Therefore it must be **work**: the slot is busy far more than the census
accounts for. Inverting M/G/1 at E[S] = 548 s, CV² = 3.18 for a 5.0 h mean wait
gives **ρ ≈ 0.94**, i.e. **22.6 h/day** of occupancy against the **16.4 h/day**
the census sees plus ~1.5 h/day of fan-outs — about **4.8 h/day unattributed**.

An independent check says the slot really was saturated during those waits: for
each of the 90 observed legs with wait > 1 h inside the census window, I summed
the expected service of every census arrival in `[arrival − 2 h, start]`. The
ratio wait / offered-work has **p50 = 1.05** (p10 0.63, p90 1.58), and only
**2/90** cases have wait > 2× the available work. The queue was genuinely
busy — the census just does not show all of what was in it. [M/D]

Ruled out as the missing work:
- **journald drops** — no `Suppressed … messages` lines, stock `journald.conf`. [M]
- **A second concurrent DF orchestrator** — the three DF pids' plan-line spans
  are strictly sequential (7047 → 09-14T13:29; 1553979 09-14T13:31 →
  09-15T23:34; 1937083 09-16T00:00 →), with one 26-minute gap. [M]
- **The unlogged `per-subproject fan-out` branch** — zero occurrences. [M]
- **Timeout holds** — all 11 timed-out legs in the sample sit at 3 600–3 605 s,
  i.e. the *pre*-09-12 budget; zero timeouts among the 124 in-window legs. [M]
- **The merge gate** — excluded by construction, and cross-checked against 89
  `merge_verify` events. [M]

Still open (§11).

---

## 10. Reify, separately

- ρ ≈ **0.93** [D] from 62 gated legs/day × ~1 300 s.
- Verify-phase wall p50 **15 631 s (4.34 h)**, p90 78 114 s, residual 97.8 % [M]
  — consistent with a near-saturated single server.
- Service CV² **0.15** across 33 records; one class only. **SJF has nothing to
  order.** FIFO would still cut its tail.
- Its bash semaphore (`scripts/lib_test_semaphore.sh` →
  `lib_slot_acquire.sh::slot_acquire`) is measured *directly* by the
  `@@REIFY_CLOCK_{STOP,START}@@` markers DF's clock-stop seam consumes, and it
  is **almost never contended**: 58 `test_slot_starvation` waits over a month,
  p50 **3 s**, mean 183 s, max 1 686 s, 2.94 h total. Its sibling
  `psi_pressure` gate (`cpu-admit.sh`) waited 30 times, p50 111 s, max 1 037 s,
  1.93 h total. [M]
  Both figures are from **failure-archived logs only**, so they are red-only
  floors — but the direction is unambiguous: the DF-side flock holds the queue,
  so the inner bash gate sees a nearly-empty one. Reify's own semaphore is
  where the *instrumentation* is and the DF-side flock is where the *queue* is.
- `REIFY_SLOT_EVENT_LOG` (the ACQUIRE/RELEASE nanosecond event log
  `lib_slot_acquire.sh::slot_emit_event` writes) is **unset in reify's
  `verify_env`**, so the one purpose-built trace facility that exists in either
  project is switched off. Turning it on is the cheapest possible
  instrumentation win, and it is a `verify_env` leaf, hence green-tier
  hot-reloadable.

---

## 11. Open questions

1. **Where are the missing ~4.8 h/day of DF slot occupancy?** (§9) Highest-value
   next measurement. Candidates I could not close read-only: real service of the
   `Full verification` fan-outs (unmeasured — their legs persist no summary
   because `run_full_verification` passes no `attempt_id`); the 15
   `subproject-scoped` verifies; any process that loads a DF-worktree
   `dark-factory-orchestrator.yaml` (every worktree carries the explicit
   `verify_admission_slots_dir`, so an agent or `/unblock` session rooted in a
   worktree shares DF's slot and logs nothing to syslog). A one-line ACQUIRE/
   RELEASE append in `acquire_task_slot`, mirroring reify's
   `slot_emit_event`, would settle it in a day of data.
2. **Is the priority gradient in §7 causal or selection?** Needs
   priority-as-of-acquisition, which no store retains.
3. **Does reify's ρ ≈ 0.93 survive a better service estimate?** Its 33-record
   sample is the current lane occupants only.
4. **What is killing ~50 % of dispatches**, if not the restart clock (restart
   interval measured at 34–51 h, verify-phase p90 18 h)?
5. **Does the 3-slot regime's evidence survive this correction?** The config
   comment's 1-vs-3-slot comparison (median 2 300 s → censored 3 600 s) was
   argued under "slot time is not the constraint". At ρ ≈ 0.7–0.9 the right
   reading of that experiment may be different — more slots raised *service*
   time through CPU contention while *queueing* was already the larger term.
   That is a re-analysis, not a new measurement.

## 12. Files

```
plans/verify-merge-queue-policy-study-2026-09-18/
  measure-sem.md                     this report
  scripts/00_schema.py               read-only schema dump, both run stores
  scripts/10_harvest_legs.py         harvest every on-disk per-leg summary
  scripts/20_demand.py               event-store demand: phases, fan-out, restarts
  scripts/30_verify_phase_decompose.py  verify phase -> agent / service / residual
  scripts/40_admission.py            wait + service + occupancy, all sources
  scripts/50_arrivals_from_syslog.py parse 'Verify plan:' into arrival rows
  scripts/60_load.py                 rho, arrival mix, service cells, variance explained
  scripts/70_wait_clean.py           env-recovery contamination split
  scripts/80_replay.py               1-server replay: random / fifo / sjf / srpt
  scripts/90_priority_overtake.py    priority tiers + overtake rate
  scripts/95_endtoend.py             dispatch wall, acquisitions/task, kills
  scripts/99_export_trace.py         build data/slot_trace.csv
  data/slot_trace.csv                THE SIMULATOR TRACE (578 arrival + 897 observed)
  data/slot_trace.README.md          columns + every known censoring
  data/verifyplan_syslog.txt         the syslog grep (syslog rotates weekly)
  data/legs_raw.jsonl                1 238 raw per-leg summary records
  data/wait_clean.jsonl              897 DF observed acquisitions, contamination flagged
  data/arrivals_df.jsonl             578 DF gated arrivals
  data/arrivals.jsonl                745 arrivals, all projects
  data/admission_trace.jsonl         1 216 rows, all projects/sources
```
`data/README.md` and `data/merge_trace.csv` in that directory belong to another
seat; I did not touch them.
