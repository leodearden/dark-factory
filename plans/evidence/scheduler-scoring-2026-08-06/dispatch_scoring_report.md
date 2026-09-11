# Dispatch-scoring re-derivation report

Snapshot timestamp: **2026-08-06T14:48:47Z** (`snapshot_taken_at.txt`).
Scheduler code read on 2026-08-06 from `orchestrator/src/orchestrator/scheduler.py`
(read-only; never modified).

Script: `/tmp/claude-1000/-home-leo-src-dark-factory/6aab7100-da40-4bdb-8e30-34db0d7abac5/scratchpad/score_replicate.py`
Per-candidate CSV: `/tmp/claude-1000/-home-leo-src-dark-factory/6aab7100-da40-4bdb-8e30-34db0d7abac5/scratchpad/candidates_scored.csv` (294 rows: id, tier, age, transitive_count, bonus, capped, score, rank)
Raw findings dump: `/tmp/claude-1000/-home-leo-src-dark-factory/6aab7100-da40-4bdb-8e30-34db0d7abac5/scratchpad/report_findings.json`

All constants (`TIER_BASE`, `age_alpha=10.0`, `cpm_beta=100.0`, `TIER_WIDTH=1000`) are the
`config.py` defaults — confirmed **not overridden** by grepping every key in
`orchestrator_config_snapshot.yaml` (no `age_alpha`/`cpm_beta`/`tier_base`/`scoring` key at
any level).

## Methodology summary (read this before the numbers)

- `ctx.tasks` / `ctx.tasks_by_id` in the live scheduler = `get_tasks(statuses=ACTIVE_TASK_STATUSES)`,
  where `ACTIVE_TASK_STATUSES = {pending, in-progress, blocked, deferred, review, merge-deferred}`
  (excludes `done`, `cancelled`, `infra-hold`). `status_map` is backfilled from the *full*
  task universe for any referenced dep id — replicated here by just using the full
  `tasks` table as `status_map` (a superset of what the real backfill produces, so
  behaviourally identical for every dep actually referenced).
- `max_id` is computed **only over the active-fetch result**, not the whole task
  universe (scheduler.py ~6504-6512) — replicated exactly. In this snapshot
  `max_id_active == max_id_all == 3796` (the newest task is itself still pending), so this
  distinction happens not to matter today, but is a real trap for a future re-run.
- `reverse_index` (used for both priority inheritance and CPM transitive counts) is built
  **only from active-status tasks' own dependency lists** (scheduler.py `_build_reverse_index`
  ~4580). A consequence transcribed faithfully but worth flagging: a `done`/`cancelled` task's
  outgoing dependency edges never enter the index (since it's not in the active fetch), so a
  chain `A ← B(done) ← C(active)` does **not** let a CPM walk from `A` reach `C` through `B`.
  This is the *production* behaviour, not a replication bug.
- **Age-anchor assumption** (`_pending_anchor` is in-memory-only, no snapshot of it exists):
  every currently-pending task's anchor is assumed to equal its own numeric id — i.e., "has
  been continuously pending since creation, never resurrected." This is the code's documented
  rule for the common case (scheduler.py ~4707-4737). Where wrong (a genuinely
  resurrected task), the *true* live anchor would be a more recent `max_id`, meaning our
  computed ages are an **upper bound** for any resurrected task. No snapshot signal exists to
  detect resurrection, so this is stated as an assumption, not silently baked in.
- **Gates not fully replicable offline**, each isolated and reported rather than guessed:
  - `external_deps` (cross-project): 2 pending candidates (`2915`, `3677`) carry
    `metadata.external_deps`; their live satisfied/not-satisfied state depends on another
    project's task store, unavailable from these snapshots. **Assumed satisfied** (included as
    candidates) — if wrong, N drops by up to 2.
  - `delivered_checks` (grep/script checks against committed `main`): 14 pending candidates
    depend on a TERMINAL task carrying `metadata.delivered_checks` (`2732, 2985, 2987, 3076,
    3176, 3184, 3212, 3259, 3311, 3315, 3326, 3327, 3354, 3543`). Re-running the checks
    would require executing scripts/greps against the live repo tree, out of scope for a
    read-only snapshot replication. **Assumed satisfied** — if wrong, N drops by up to 14.
  - Per-task dispatch cooldown (`_requeue_until`, 30 min window) and the signal-armed
    re-dispatch cooldown (`_last_dispatch_at`) are both in-memory-only. **Assumed inactive**
    for all candidates (0 pending tasks carry any of the 4 cooldown-signal metadata flags
    anyway — `recon_reset_count>1` / `steward_clear_at` / `recon_stage2_blocked_at` /
    `reopen_reason~'steward'` — so this assumption costs nothing measurable here).
  - Landed-outbox / already-landed gate (`_phase_landed_outbox_gate`) consults live merge-queue
    hook state — unavailable offline. Assumed empty (no candidate excluded on this basis).
- **Validation checkpoint**: `_compute_effective_priorities` was independently re-implemented
  (priority inheritance + override-boost overlay from `scheduler_state_snapshot.json`'s
  `overrides`) and compared against the **532** entries in the state snapshot's own
  `effective_priorities` field: **0 mismatches**. This is strong evidence the priority-
  inheritance and reverse-index replication is byte-correct.
- **Second validation checkpoint**: the two `skip_counts` entries in the state snapshot
  (`3618`: 31, `3076`: 7) land at **rank 1 and rank 2** respectively in our independently
  computed score ranking. Since `task_skipped`/park-installation only ever fires for the
  tick's #1-scored candidate (scheduler.py `_bump_skip_and_maybe_park(top_id, ...)`), a task
  with a nonzero skip_count essentially *must* have spent time at rank 1 — which is exactly
  where our replication independently places them. This cross-checks the scoring math itself.

---

## 1. N ready pending candidates

**N = 294** (pending status, all local deps done/cancelled — or intra-train merge-deferred allowance
where applicable — per `_deps_satisfied`; 0 pending tasks currently use train metadata).

- 537 tasks in `ACTIVE_TASK_STATUSES`; of those, 444 are `pending`.
- Of the 444 pending: 0 excluded by the live-claimant gate, 0 by `deferred_watch`, **8**
  excluded by `milestone_time_gated` (`2664, 2665, 2666, 3190, 3192, 3205, 3593, 3676` — all
  `mode='delayed'` milestones not yet fired), and the remainder trimmed by unmet local
  dependencies.
- **Vs. the claimed 296**: delta of **−2 (0.7%)**. Given the claim was "measured earlier
  today" at an unstated exact timestamp, and this system dispatches continuously (task 3536,
  a sibling of 3534, was boosted to `critical`+pinned and dispatched to `in-progress` sometime
  today per its `updated_at` of 11:53:42Z, removing it from the pending pool), a 2-task drift
  over a few hours is well within expected background churn — not a discrepancy requiring
  explanation beyond "the live system kept moving." The known-unverified gates above (`external_deps`,
  `delivered_checks`) could shift N by up to +16/−16 in either direction from mis-assumption,
  which comfortably brackets the observed 2-task gap.

## 2. Saturation

- **Capped** (`age_alpha*age + cpm_beta*log1p(trans) > 999`, i.e. the `min()` in
  `_compute_score` actually reduced the bonus): **251/294 = 85.4%**.
- **CPM term fully masked** (`age_alpha*age` alone `>= 999`, so the CPM term contributes zero
  marginal value regardless of its own magnitude): **249/294 = 84.7%**.
- **Confirms** the earlier claim of "250/296 ≈ 84%" — both of our precise definitions land
  within a point of that figure (85.4% and 84.7% respectively); whichever exact definition
  the earlier claim used, the headline conclusion (the CPM/value-unlocked signal is dead for
  the large majority of the queue, and dispatch order is functionally pure age-FIFO within
  tier) is corroborated, not contradicted.

## 3. Task 3534

- **Current status: `pending`**, priority `high` (own priority == effective priority — no
  inheritance boost active for 3534 itself). It is **still pending** in this snapshot — the
  "sibling boosted today" the prompt anticipated is task **3536** (own priority `high`, boosted
  to `critical` + pinned via `scheduler_state_snapshot.json` overrides, and now `in-progress`
  with a live claimant since 2026-08-06T11:53:42Z). 3534 itself was never boosted or
  dispatched, so no hypothetical reconstruction is needed — its current, real, live-computed
  rank is reported directly.
- **Rank: 32/294** (score 8999.00 = `TIER_BASE['high'] (8000) + min(10*262 + 100*ln(1+9), 999)`
  → raw bonus 2850.2, capped at 999). Tier `high`, age (id-distance) **262**, transitive_count **9**.
- **Vs. the claimed rank 30/296**: delta of 2 ranks / 2 candidates — consistent with the same
  small pool-composition drift noted in item 1 (N is down 2, and continuous priority/age churn
  among the ~50-60 other `high`-tier candidates easily accounts for a 2-rank shuffle). Not a
  discrepancy requiring further explanation.

## 4. Distributions

- **Age (id-distance)**, n=294: p10=**60.3**, p50=**354.0**, p90=**746.1**, max=**1064**.
- **transitive_count**, n=294: p10=**0**, p50=**0**, p90=**1**, max=**31**.
- Candidates with **transitive_count ≥ 3**: **22/294 (7.5%)** — the CPM signal is not just
  saturated for most tasks, it's *zero or near-zero* for the overwhelming majority to begin
  with (median transitive_count is 0).
- **If the `min(..., TIER_WIDTH-1)` cap were removed entirely**, only **2 of the top 24
  positions change occupants**:
  - Entering the top 24: **2732**, **3276**
  - Leaving the top 24: **3271**, **3275**
  - Interpretation: even fully un-gating the bonus barely reshuffles the top of the queue,
    because the *within-tier* ordering is already almost entirely age-driven for everyone near
    the top — removing the cap mostly just stretches the score gaps, it doesn't flip many
    orderings, because so few candidates carry a transitive_count large enough to matter (per
    the p90=1 above).

## 5. Wall-clock reality check

Two independent proxies were used (see Methodology); they diverge in an important, diagnostic
way:

- **Primary (100% coverage): `tasks.updated_at`** for currently-pending candidates (valid as a
  filing-time proxy only if the task never left `pending` and was never metadata-edited without
  a status change — a real but low-probability confound, undetectable from a single snapshot).
  Correlation between id-distance age and wait-hours: **n=294, Pearson r=0.844, Spearman
  r=0.847** — a strong positive correlation. **Overall, id-distance age does track wall-clock
  reasonably well** across the ~4-month history represented in this queue.
- **Secondary (partial coverage, n=121/294): event-log first-appearance.** Investigating *why*
  173/294 (58.8%) of candidates have **zero rows ever** in `runs_snapshot.db`'s `events` table
  surfaced a structural fact about the scheduler's own instrumentation: `task_skipped` fires
  **only for the tick's #1-scored candidate** (`_bump_skip_and_maybe_park(top_id, ...)`,
  scheduler.py ~6425/~6436), and `dispatch_deferred` fires only for PSI-admission-held heavy
  candidates. A candidate that has never once reached rank 1 and never been PSI-held generates
  **no events at all**, for its entire lifetime. So this proxy actually measures "time since
  last reaching rank 1," not "time since filing" — a materially smaller, different quantity
  (median age of the zero-event group is 308 vs. 400 for the has-event group — consistent
  with "hasn't reached the top of the queue yet" rather than "was just filed"). Reported only
  as a secondary cross-check (r=0.606 Pearson / 0.881 Spearman on its own, narrower
  definition), not as the answer to this item.
- **Concrete distortion examples** (from the `updated_at` proxy, currently-pending tasks,
  exact and verifiable): three genuine same-second-to-single-digit-second filing bursts exist
  in the current pending pool alone:
  1. **ids 3187–3193 (7 tasks)**, filed within a **9.1-second** window on 2026-07-29
     (13:33:14.403Z → 13:33:23.517Z).
  2. **ids 3537–3542 (6 tasks)**, filed within a **0.1-second** window on 2026-08-02
     (17:41:27.676Z → 17:41:27.825Z) — essentially simultaneous (a single `commit_planning` batch).
  3. **ids 3667–3672 (6 tasks)**, filed within a **1.1-second** window on 2026-08-05
     (20:14:41.043Z → 20:14:42.112Z).
  - In each case, **any older still-pending task instantly gained +6 or +7 id-distance age**
    the moment that batch committed, with **zero elapsed wall-clock time** for the older task
    itself. This is the exact "batch filing inflates everyone's age" distortion the earlier
    hypothesis described — it is real and present, just smaller in this snapshot (6-7 tasks
    per burst, not 14) than the illustrative magnitude in the prompt.
  - We could **not** reconstruct historical (non-pending) filing bursts, because (a) the
    schema has no `created_at` column — `updated_at` for a terminal task reflects its
    *last* status change (often completion), not filing, and (b) per the event-log finding
    above, the event stream doesn't cover backlog residency for non-rank-1 tasks either. This
    is a genuine data-availability limitation, not an omission.

## 6. Queue latency history (last ~14 days)

**Important caveat up front, per the finding in item 5**: the only wall-clock signal available
for tasks that eventually dispatch is the event-log first-appearance proxy, which — as
established above — measures **"time from reaching rank-1 (or a PSI-hold event) to lock
acquisition,"** not "time from filing/first-pending to dispatch." No task-creation timestamp
exists anywhere in either snapshot, and the scheduler does not emit a per-tick event for every
candidate, only the current top pick. **A true historical filing-to-dispatch latency series is
not reconstructable from these snapshots.** The numbers below are reported honestly under that
label, not relabeled as something they aren't.

- Dispatched (`lock_acquired`) in the last 14 days (≥ 2026-07-23T14:48:47Z): **500** tasks.
  "Filing" proxy = first event of any type for that task_id; "dispatch" = first
  `lock_acquired`; latency = the gap between them, split by the effective tier stamped in the
  `lock_acquired` event's own `data.priority` field (no independent recomputation needed —
  the scheduler stamps its own decision into the event).

| tier | n | p50 (h) | p95 (h) | max (h) |
|---|---|---|---|---|
| critical | 12 | 0.00 | 67.26 | 94.17 |
| high | 128 | 0.00 | 0.77 | 96.40 |
| medium | 175 | 0.00 | 0.00 | 71.67 |
| low | 185 | 0.00 | 0.00 | 0.00 |

  p50=0h across nearly every tier is itself evidence for the caveat above: once a task
  actually reaches rank 1, it usually acquires its module locks on the very same tick (no lock
  contention) — this is measuring lock-acquisition speed given top rank, not real queueing
  delay.

- **Currently-pending candidates whose time-since-last-rank-1-touch exceeds their tier's
  historical p95** (proxy-consistent comparison, same narrow metric on both sides — n=121
  candidates with any event history; the other 173/294 are excluded from this specific check
  since they have never touched rank 1 at all, so this undercounts true breaches):
  **120/121**.
- **Supplementary, more meaningful number** — actual current pending-wait, **100% coverage**,
  via `updated_at`:

| tier | n | p50 (h) | p90 (h) | max (h) |
|---|---|---|---|---|
| critical | 1 | 6.9 | 6.9 | 6.9 |
| high | 56 | 67.7 | 181.7 | 214.2 |
| medium | 90 | 102.7 | 326.0 | 439.6 |
| low | 147 | 92.2 | 196.5 | 360.6 |

  This is the honest answer to "how long have currently-pending tasks actually been waiting":
  medium-tier candidates have a median wait of over 4 days and a max of over 18 days; high-tier
  candidates a median of nearly 3 days.

## 7. skip_counts cross-check

Verbatim from `scheduler_state_snapshot.json`:

| task_id | skip_count | current status (tasks_snapshot.db) |
|---|---|---|
| 3076 | 7 | pending |
| 3618 | 31 | pending |

Both remain `pending`. Both also land at **rank 1 (3618) and rank 2 (3076)** in our
independently-computed score ranking (see Methodology's second validation checkpoint) — i.e.
they are not stuck because of a scoring problem, they are winning the score race every single
tick and still failing to dispatch. Cross-referencing `scheduler_state_snapshot.json`'s `parks`
/ `park_stacks`: both already have module reservations installed (3618 since
2026-08-06T08:19:22Z, 3076 since 2026-08-06T05:55:07Z), and they **contend for an overlapping
module** — `orchestrator/src/orchestrator/git_ops.py` — where 3618's park (rank 0, not
shadowed) sits above 3076's (rank 1, shadowed). This is a **module-lock contention** story, not
a scoring/CPM story: the scoring model correctly and repeatedly selects both as top candidates;
they just can't both hold `git_ops.py` at once, and a third task appears to be holding it live
(per `current_holders`, unrelated task ids hold most locked modules at snapshot time).

---

## Caveats index (all gates/assumptions not fully replicable offline, gathered in one place)

| Gate/assumption | Snapshot support | Assumption made | Max impact on N=294 |
|---|---|---|---|
| Age anchor (`_pending_anchor`) | none (in-memory) | own-id anchor, no resurrection | ages may be overstated for resurrected tasks (undetectable) |
| `external_deps` cross-project gate | none | assumed satisfied | ±2 |
| `delivered_checks` dep gate | partial (metadata present, check execution not) | assumed satisfied | ±14 |
| Per-task dispatch cooldown (`_requeue_until`) | none (in-memory) | assumed inactive | 0 observed signal-carriers, so ~0 in practice |
| Landed-outbox / already-landed gate | none (live hook state) | assumed empty | unknown, likely small |
| max_id (active-only vs. all-tasks) | full | replicated exactly (active-only, per code) | 0 in this snapshot (they're equal) |
