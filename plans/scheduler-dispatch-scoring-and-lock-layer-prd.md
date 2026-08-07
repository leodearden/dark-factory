# PRD: Scheduler dispatch scoring + lock layer

**Status:** active — authored 2026-08-06, ready to decompose.
**Approach:** B + H (contract section + two-way boundary tests). G5 triggered on all
four heuristics: cross-package blast radius 3 (`orchestrator/`, `fused-memory/`,
`shared/`), ~10 mechanisms, the load-bearing seam (the dispatch decision of every
project the fleet runs), and ≥2 consumer surfaces.
**Evidence:** `plans/evidence/scheduler-scoring-2026-08-06/` — the four investigation
reports and the CSVs they cite, copied into the repo so every number below is
checkable without leaving it.

---

## 1. Goal and user-observable surface (G1)

A PRD batch's chain head must reach the dispatch window on the strength of what it
unblocks and how long it has waited — **without a human pinning it**.

Three named consumers:

| Consumer | What it observes today | What it observes after |
|---|---|---|
| The orchestrator scheduler's dispatch decision (every project in the fleet) | Intra-tier order is FIFO-by-id; 117 candidates tie on one score | A graded order in which fan-out and wall-clock wait both move rank |
| The operator reading starvation escalations | Nothing — the watchdog has never fired under current config, emits no event, and its only consequence is an unread level-0 INFO | An emitted `starvation_detected` event and a visible auto-pin in the override store, surviving restarts |
| PRD-batch authors (this skill's own output) | Chain heads queue behind the entire backlog; 12 tasks and 54 verified-green commits held hostage on 2026-08-06 | Chain heads compete on unblocked value; pinning becomes a lever, not a prerequisite |

The immediate mitigation is already in place and is **not** part of this PRD: task
3534 is pinned at `pin_order 1` with `boost_tier critical`, with 3535/3537/3538/3539/
3540/3541/3545/3546 at orders 5–12 (verified live in the pin queue 2026-08-06). This
PRD is the durable fix that makes that intervention unnecessary next time.

## 2. Background — the measured defect

All numbers re-measured 2026-08-06 by a four-agent investigation plus adversarial
critique plus a trace-driven parking model. Reify calibration Spearman ρ = 0.913;
dark-factory calibration ρ = 0.50 (PSI holds and requeue churn unmodeled), so **DF
simulation conclusions below are relative-only and are labelled as such**.

**2.1 The scorer.** `Scheduler._compute_score` (`orchestrator/src/orchestrator/scheduler.py:4739`):

```
score = TIER_BASE[tier] + min(age_alpha·age + cpm_beta·log1p(D), TIER_WIDTH − 1)
```

with `age_alpha=10.0`, `cpm_beta=100.0`, `TIER_WIDTH=1000` (`config.py:4004`/`:4014`/`:52`),
and `age = max_id − task_id` (id-distance, `_compute_age` :4695).

The age term alone reaches the 999 cap at `age ≥ 100`. Median candidate age is 355.
Consequences, measured on the 2026-08-06T14:48:47Z snapshot (294 ready candidates):

- **249/294 (84.7%)** have the CPM term *fully masked* — `age_alpha·age` alone already
  exceeds the cap, so `D` contributes exactly zero.
- 291 candidates collapse onto **46 distinct scores**, with a **117-way tie** at 2999.0.
  Every task in the top-24 except the single critical one ties at exactly 8999.0.
- Intra-tier dispatch order is therefore decided entirely by the `task_id` **string**
  tie-break (`scheduler.py:6349`) — i.e. pure FIFO-by-id.
- **Removing the cap does not fix it.** Un-gating the bonus entirely changes only
  **2 of the top 24** occupants. Median `D` is 0, p90 is 1: the α:β ratio gives CPM no
  authority even un-saturated.
- Twelve of the 24 top slots are held by `D=0, L=0` leaves that unlock nothing, while
  the two largest fan-outs in the set (3197 at D=31, 3132 at D=26) sit at ranks 15 and 8
  only because their ids are low.

Task **3534** — `high`, D=9, L=7, chain head gating 12 stranded tasks and 54 unmerged
verified-green commits — ranked **32/294** against a 24-slot window, has **zero events
of any type** in its entire life, and was never dispatched in four days.

The constants were never derived: single commit `e037adb6fc` (2026-04-17), no PRD, no
saturation test.

**2.2 The starvation watchdog is structurally dead for this class.**
`_apply_starvation_watchdog` (`scheduler.py:3433`) fires on `skip_count ≥ 50 AND idle ≥ 72h`
(unreachable — `_bump_skip_and_maybe_park` runs only for the tick's rank-1 candidate, so
3534 has 0 skips) OR on `idle ≥ idle_only_secs` (72h). The idle clock is
`_streak_starvation.first_seen`, which is **in-memory only** — not among the eleven keys
`get_state_snapshot` persists, and with **no restore path anywhere**. The median
orchestrator process era is **2.2h** (last 30d, 149 eras). A 72h in-memory clock inside a
2.2h process cannot run out. Two `service_restart` events fell inside the 93h incident
window alone. Additionally, a single tick where the task drops out of the candidate list
wipes every anchor (`scheduler.py:3480-3485`).

On fire, the consequence is `harness._file_starvation_info` — a level-0 `severity=info`
escalation with no consumer, and **no `event_store.emit` at all**, so the firing is
invisible in the event stream.

**2.3 The lock layer.**

- **Occupancy is real but over-reported.** In the 93h window, `harness.py` was **74.0%**
  occupied and `workflow.py` **73.3%**. `task_ground_truth.py`'s widely-cited **96.5%**
  is a **stuck-lock artifact**: task 3563 acquired at 2026-08-03T10:16:53Z, `task_completed
  outcome=requeued` fired 7 minutes later, and **no `lock_released` ever followed** — while
  live `current_holders` showed the module FREE. True occupancy ≈ **14–21%**.
  Root cause located: `Scheduler.handle_blast_radius_expansion` releases via
  `self.lock_table.release(task_id)` **directly** (`scheduler.py:7038`), bypassing
  `Scheduler.release()` — the only site that emits `lock_released` (:7041-7058). It is the
  **only** such bypass in the file (`release_subset` at :6953 does emit). Fleet-wide the
  class is large: 1,283 DF / 4,776 reify stuck-at-era-end spans, 3,594 / 5,780 orphan
  releases.
- **Parks and skip counts die silently at every restart.** `skip_counts` and `parks` are
  written to `scheduler_state.json` but **never read back** — that file is a snapshot for
  readers (dashboard, fused-memory `read_scheduler_state`), not a restore source.
  **36–46%** of modern park episodes die at process-era end with no event. Since 2026-08-01,
  DF installed 19 parks and used 3.
- **62% of DF's and 82% of reify's `reservation_installed` events are no-ops** — every
  requested module was already blocked by a same-or-higher-tier foreign park (INV-3
  install blocking), yet the event fires and `has_parks` stays False, so the attempt
  re-fires every tick. The fairness mechanism live-locks against itself.
- **Only one hold predictor works.** Log2-space R² on a 70/30 time-ordered split:
  global median −0.22 / tier median −0.36 / tier+width −0.31, versus **module-history
  median (last 10 holds on the task's modules) 0.26 (DF) and 0.68 (reify)**. Static task
  features predict nothing. Safety multipliers: ×2.9 (DF) / ×2.0 (reify) covers 80% of
  realized durations, ×4.8 / ×4.0 covers 90%.
- **Rank is scan order, not entitlement.** `_phase_select_scored` tries `try_acquire` in
  score order and the first candidate that succeeds dispatches and returns. Three
  historical inversions were reconstructed where a strictly higher-ranked candidate was
  skipped and a lower-ranked one dispatched **3–5 ms later, same tick**. Only `top_id` is
  ever skip-tracked; every candidate between rank 1 and the winner loses its turn with no
  record at all.

## 3. Sketch of approach

Four coupled changes, one batch:

1. **Replace the saturating `min()` with fixed non-tier-crossing sub-budgets** — a CPM
   term, a wall-clock age term anchored on a new durable `pending_since`, and a small
   continuity credit — so no term can mask another and the order is graded rather than tied.
2. **Repair the starvation watchdog** — derive its idle clock from durable anchors instead
   of an in-memory streak that cannot outlive a 2.2h process, and give it a real
   consequence (auto-pin via the override store, the only restart-durable prioritization
   state) plus an emitted event.
3. **Make the lock layer truthful and durable** — one bypassing release site routed through
   the emitting path; skip counts and parks persisted across restarts.
4. **Let narrow work through parks when it provably fits** — EASY-backfill gated on the one
   predictor that works, plus below-rank-1 skip *counting* for observability.

The four compose deliberately: the scorer decides ordering, the watchdog catches whatever
ordering starves anyway, truthful lock events feed the predictor, and the predictor is what
makes durable parks affordable.

## 4. Resolved design decisions

**D1 — Sub-budgets, not a clipped sum.** The disease is a scale mismatch: a term with range
0–10,640 summed against a term with range 0–347 and then clipped at 999. Each term gets a
fixed budget and a transform that cannot leave it; the sum of budgets is validated below
`TIER_WIDTH`. No `min()` on the sum.

**D2 — Wall-clock age from a new durable `pending_since`, not id-distance, not in-memory.**
Id-distance is corrupted by batch filing: three same-second filing bursts exist in the
current pending pool alone (ids 3187–3193 in 9.1s, 3537–3542 in 0.1s, 3667–3672 in 1.1s),
each of which instantly aged every older pending task by +6 or +7 with zero elapsed time.
The anchor is durable because in-memory anchors are re-derived on every restart and the
median process era is 2.2h.

**D3 — `pending_since` is reset ONLY on `cancelled → pending`** (Leo's continuity ruling,
2026-08-06). A requeue, an unblock, or a `deferred → pending` commit leaves it untouched: a
task that has already waited must not lose its accrued wait because the machine dropped it.

**D4 — Back-fill from `updated_at`, eyes-open.** ~444 pending tasks carry no anchor. Seeding
`pending_since = updated_at` gives the true filing time for a never-touched task and a
**younger-than-truth** anchor for a previously-requeued one. The mis-aging direction is
conservative — the back-fill can under-age a task but never over-age one, so it cannot
manufacture a queue jump; the under-aged tail is exactly what the watchdog (§C4) covers.
Every back-filled row is stamped `pending_since_backfilled: true` so the population is
countable and the distortion is auditable rather than invisible. Rejected alternatives:
falling back to id-distance when the anchor is absent (keeps two incommensurable clocks
alive indefinitely, which is the bug), and synthesising a filing time from the id (no
`created_at` column exists anywhere in the schema — confirmed).

**D5 — Continuity is detected by WORK PRODUCT, not by dispatch history.** A branch with
commits ahead of its merge-base, gated on the zero-progress detector not having fired for
that task. The events DB is unusable for this: it carries both false positives (orphan
releases, the 3563-class stuck locks) and false negatives (173/294 candidates have **zero
events ever**, because a task that never reaches rank 1 and is never PSI-held generates no
events for its entire lifetime).

**D6 — The continuity credit sits strictly below `cpm(D≈4)`.** A task with prior work
beats a `D=1` task but loses to a genuine chain head. The flat-credit-above-CPM variant was
measured and **inverts the fix**: 115/291 candidates carry prior-dispatch history and 3534
drops to rank 35–40, worse than baseline.

**D7 — Numeric-id final tie-break.** The current `(-score, task_id_string)` sort inverts
FIFO at the 9999 → 10000 boundary. Every candidate id is 4 digits today, so this is latent,
not live — fix it while the sort key is already being touched.

**D8 — Scoring knobs are hot-reloadable.** `age_alpha`/`cpm_beta` are top-level
`OrchestratorConfig` fields and are **not** in `RELOADABLE_FIELDS` (verified — the file
lists `fairness.skip_threshold` and `starvation_watchdog.*` but no scoring key), so
today a scoring retune costs a fleet restart. The new knobs ship as a submodel group in
`RELOADABLE_FIELDS`, matching the `fairness`/`starvation_watchdog` precedent.

**D9 — Watchdog consequence is an auto-pin, bounded and owned (INV-7).** The override store
is the only restart-durable prioritization state. The auto-pin carries `ttl_until`, is
capped at `auto_pin_max_concurrent`, is tagged with its source so an operator's manual pin
is never clobbered or miscounted, and is released when the task dispatches. Hitting the cap
is itself the escalation — a systemically starved queue is an operator-grade signal, not a
per-task one. This matters concretely: the pin queue already holds 12 entries, 4 of them
operator pins.

**D10 — Ordering: backfill lands before park persistence.** Making parks durable amplifies
whatever the current parking regime does. For dark-factory the sim says the current regime
(P0) costs −1.9 disp/day versus no parking at all and roughly doubles the never-dispatched
count (53 → 104) for no aggregate wide-task gain; P4 (backfill) recovers 84% of that tax
while improving targeted protection. Park persistence is **not simulated** — the sim wipes
parks at era boundaries exactly as production does — so its benefit is an inference, not a
measurement, and it should not land ahead of its compensator. A real dependency edge, not a
config flag, enforces the order.

**D11 — Truthful lock events are a prerequisite for the predictor, not just hygiene.** The
predictor reads hold durations from `lock_acquired` → `lock_released` pairs. A missing
release reads as an infinite hold, which would make the admission test refuse every
backfill — the failure is silent and total. Fix the emission first; and handle the
historical residue by closing any span still open at a `service_restart` boundary, in one
shared helper (INV-5), never per-consumer.

**D12 — The new age term is armed by anchor coverage, not by deployment order.** β's
fail-safe (absent anchor ⇒ age 0) is correct per-task but catastrophic in bulk: if the
orchestrator loads the new scorer before fused-memory's α has stamped and back-filled, every
candidate scores age 0 at once and the fleet silently enters a pure-CPM regime. So the scorer
computes `anchor_coverage` — the fraction of this tick's candidates carrying a parseable
`pending_since` — and falls back to the **legacy** formula for the whole tick when coverage
is below `new_scoring_min_anchor_coverage` (default 0.90), emitting a
`scoring_anchor_coverage_low` event and escalating on a streak. Coverage is a live-measured
premise, not an assumption about who restarted first (INV-3, INV-4). The fallback is
self-clearing: once α's back-fill has run, coverage is 1.0 and the arming condition never
fires again.

**D13 — One PRD, one batch.** Splitting scoring+watchdog from the lock layer would create a
cross-PRD seam on a brand-new field (`pending_since` feeds both the scorer and the watchdog
clock), would not buy parallelism (both halves are dominated by `scheduler.py`, which
serializes on the module lock regardless of PRD boundary), and would double the number of
chain heads exposed to exactly the starvation this PRD exists to fix.

## 5. Contract (H)

### C1 — `metadata.pending_since` (produced by fused-memory, consumed by the orchestrator)

**Shape.** `task.metadata.pending_since` — an ISO-8601 UTC timestamp string in the same
format `updated_at` uses. Registered in `_BLESSED_METADATA_KEYS`
(`shared/src/shared/task_metadata.py:746`) as a Tier-A load-bearing key; without that
registration every write raises an `unknown_key` census warning.

**Write rules**, applied at the single status chokepoint
`SqliteTaskBackend.set_task_status` (`fused-memory/src/fused_memory/backends/sqlite_task_backend.py:1938`,
which already reads `old_status` inside the same transaction) and at any other write path
that can land a row in `pending` (`add_task`, `commit_planning`) — via **one shared helper**,
never duplicated logic (INV-5):

| Transition | Effect on `pending_since` |
|---|---|
| `* → pending`, key absent | stamp `now` |
| `cancelled → pending` | **overwrite** with `now` — the only reset (D3) |
| `in-progress/blocked/review/deferred → pending`, key present | unchanged |
| `pending → *` (any exit) | unchanged — never cleared |

**Invariants.** Monotone non-decreasing per task except across a `cancelled → pending`
reset. Never cleared. A `commit_planning` batch stamps one identical timestamp across the
batch (it is one atomic flip); intra-batch order therefore falls to CPM then numeric id,
which is the intended FIFO.

**Back-fill.** One shot, at migration: every task currently `pending` with no anchor gets
`pending_since = updated_at` and `pending_since_backfilled = true`. The migration logs the
count of rows touched.

**Reader contract.** `pending_age_secs(task, now) = max(0, now − parse(pending_since))`.
Absent or unparseable ⇒ **0** (fail-safe: the task loses age rather than jumping the queue)
**and** increments a counter that escalates on a rate threshold (INV-4) — never a bare log
line.

### C2 — `Scheduler._compute_score`

```
score(t) = TIER_BASE[tier(t)] + cpm(t) + age(t) + continuity(t)

cpm(t)        = CPM_BUDGET · log1p(min(D(t), CPM_D_CAP)) / log1p(CPM_D_CAP)
age(t)        = AGE_BUDGET · a / (a + AGE_HALF_SECS),  a = pending_age_secs(t, now)
continuity(t) = CONTINUITY_CREDIT  if has_work_product(t)  else 0
```

Defaults: `CPM_BUDGET = 300.0`, `CPM_D_CAP = 32`, `AGE_BUDGET = 500.0`,
`AGE_HALF_SECS = 259200` (72h), `CONTINUITY_CREDIT = 120.0`.

**Machine-checked invariants (INV-1 — pydantic model validators on the config, not comments):**

- **I-1 no tier crossing.** `CPM_BUDGET + AGE_BUDGET + CONTINUITY_CREDIT ≤ TIER_WIDTH − 1`
  (920 ≤ 998 at defaults). Priority always wins; a bonus can never bump a task across a
  tier boundary.
- **I-2 non-saturating.** Every term is strictly monotone in its input across its whole
  domain. `age` approaches its budget asymptotically and is never clipped; there is no
  `min()` on the sum.
- **I-3 continuity subordination.** `CONTINUITY_CREDIT < CPM_BUDGET · log1p(4) / log1p(CPM_D_CAP)`
  (120 < 138.1 at defaults) — D6.
- **I-4 tie-break.** Sort key `(-score, numeric_id, task_id)` where
  `numeric_id = int(tid) if tid.isdigit() else math.inf` — D7.
- **I-5 locality.** `score` is a pure function of `(tier, pending_since, D, work_product, now)`.
  No candidate-set-relative term — see §7 for why percentile normalization is rejected.
  (`anchor_coverage` below is an arming *gate* on which formula runs, not a term in the
  score; within an armed tick, locality holds.)
- **I-6 arming.** Per tick, `anchor_coverage = |candidates with parseable pending_since| / |candidates|`.
  When `anchor_coverage < new_scoring_min_anchor_coverage` (default 0.90) the tick scores
  with the **legacy** formula, emits `scoring_anchor_coverage_low` carrying the measured
  fraction, and escalates on a consecutive-tick streak — D12.

**Worked arithmetic at defaults** (exact given the inputs; the resulting *rank* is measured
by task κ, not asserted here):

| a (wait) | age term | | D | cpm term |
|---|---|---|---|---|
| 24h | 125.0 | | 1 | 59.5 |
| 72h | 250.0 | | 3 | 118.9 |
| 93.1h (3534 at snapshot) | 281.9 | | 4 | 138.1 |
| 168h | 350.0 | | 9 (3534) | 197.6 |
| 214.2h (oldest measured high-tier wait) | 374.2 | | 24 | 276.2 |
| 720h | 454.5 | | ≥32 | 300.0 |

So 3534 scores `281.9 + 197.6 + 0 = 479.5` of bonus, while the *maximum* any `D=0`
high-tier candidate in the snapshot can reach is 374.2 (its measured maximum wait was
214.2h). A `D=3` candidate needs a wait beyond ~200h to displace it. Symmetrically, a
`D=0` task overtakes a brand-new `D=32` task after 108h — the age budget deliberately
dominates in the long run, and the watchdog (C4) backstops whatever it still misses.

### C3 — `has_work_product(t)`

True iff **both**:

1. a branch `task/<id>` exists with ≥1 commit ahead of its merge-base with `main`; **and**
2. no `EventType.zero_progress_requeue` row exists for that task in
   `fetch_events_by_type_all_runs(..., task_id=tid)` (the durable, restart-safe read —
   `ZeroProgressRequeueTracker` itself is pure in-memory and must not be consulted).

**Bounded-occupancy requirement (hard).** The probe must **not** fork git per candidate per
tick on the event-loop thread. One `git for-each-ref --format='%(refname:short) %(ahead-behind:main)'
refs/heads/task/` covers every branch in a single process (verified working on the installed
git 2.43.0), refreshed off-thread on a TTL (`continuity_probe_ttl_secs`, default 300) into an
in-memory map that the tick path reads synchronously. Asynchrony alone is not the property
being asserted — bounded subprocess occupancy is.

**Interaction with task 3317** (jittered transient-requeue backoff, pending): a crash-looping
task accumulates work product and must not camp at tier-top on the strength of it. The
zero-progress gate is the primary guard; additionally the credit decays to zero after
`continuity_credit_max_age_secs` since the branch's newest commit, so stale work product
stops paying rent.

### C4 — Starvation watchdog

**Idle clock (durable).** `eligible_age_secs(t, now) = now − max(pending_since(t), deps_satisfied_at(t))`,
where `deps_satisfied_at(t) = max(updated_at)` over `t`'s terminal dependencies — derivable
from the task set the scheduler already fetches, with no new schema. A dependency edited
after completion over-estimates the anchor, which under-ages the task: conservative, and the
same direction as D4. The in-memory `_starvation_first_seen` streak is retained only as a
this-tick candidacy signal; it is no longer the clock, so a single-tick candidacy gap no
longer wipes 72h of accrued idleness and a 2.2h process era no longer bounds a 72h threshold.

**Fire consequence.**
1. Emit `EventType.starvation_detected` with structured facts (INV-2):
   `{task_id, eligible_age_secs, skip_count, gate: 'dual'|'idle_only', auto_pinned, pin_order, source}`.
   Today the firing emits nothing at all.
2. Re-read the task's live status from the backend before acting (INV-3) — a task that went
   terminal between candidate-build and fire must not be pinned.
3. Auto-pin via `OverrideStore.set_override(..., pinned=True, ttl_until=now + auto_pin_ttl_secs,
   source='starvation_watchdog')`, appending at the queue tail so operator pins keep
   precedence. Requires a new `source TEXT DEFAULT 'operator'` column on `overrides` — a
   small additive migration, and the only way to distinguish a watchdog pin from an operator
   pin at release time.
4. Keep the existing level-0 INFO escalation as the human trail; it now has a consequence.

**Bounds (INV-7 / INV-4).** At most `auto_pin_max_concurrent` (default 3) watchdog-owned pins
at a time. At the cap, do not pin — escalate at L1 instead, naming the starved cohort. Every
auto-pin carries `ttl_until` (default 24h) and is released at both dispatch sites (where
`_resolve_starvation_escalation` already runs), clearing **only** pins whose `source` is the
watchdog.

This supersedes task **2755**'s idle-only backstop, which is `done` but structurally
unreachable as shipped.

### C5 — Truthful lock release

`Scheduler.handle_blast_radius_expansion`'s direct `self.lock_table.release(task_id)`
(`scheduler.py:7038`) routes through `self.release(task_id, requeued=True)`. Status is
written before the release, preserving the existing ordering (INV-6).

**Machine-checked invariant.** Every module leaving `lock_table._held` emits exactly one
`lock_released` carrying that module — enforced by a test asserting `lock_table.release(`
has no call site anywhere outside `Scheduler.release` (one grep-anchored assertion, not
prose). Scope-checked against task **3538** ("Truthful REQUEUED/CANCELLED/infra exits",
pending, pinned): 3538 owns **task-status** truthfulness in `workflow.py` and `harness.py`;
this owns **lock-event** truthfulness in `scheduler.py`. Disjoint files, disjoint mechanisms,
no dependency edge — but the two are causally adjacent, so 3538 is named here so a future
reader does not merge them.

### C6 — Durable fairness state

New table in `orchestrator/src/orchestrator/run_store.py` (a sibling, **not** a migration of
the existing 4-column pause-only `scheduler_state` table):

```sql
CREATE TABLE IF NOT EXISTS scheduler_fairness_state (
    project_id TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    skip_count INTEGER NOT NULL DEFAULT 0,
    parks_json TEXT,               -- {"modules": [...], "installed_at": iso}
    updated_at TEXT NOT NULL,
    PRIMARY KEY (project_id, task_id)
);
```

Written under the same throttle/dedup discipline as the JSON snapshot. Read **once** at
scheduler construction to rehydrate `_skip_count` and re-install parks, immediately followed
by the existing owner-state park-GC so a restored park whose owner is now
terminal / deps-unsatisfied / missing is dropped with the usual `reservation_expired` event
(INV-3, INV-7). `scheduler_state.json` keeps its current role as a reader-facing snapshot and
is not repurposed.

### C7 — EASY-backfill through parks

Admit narrow candidate `c` through a park owned by `p` over module set `M` iff **all** hold:

```
c's modules are otherwise free  and  c.modules ∩ M ≠ ∅
park_age(p) ≤ backfill_max_park_age_secs
predicted_hold(c) · backfill_safety_factor ≤ provable_assembly_delay(p)
```

- `predicted_hold(c)` = median of the last `N=10` hold durations observed on any module in
  `c.modules`. **Fewer than `backfill_min_samples` (default 3) samples ⇒ refuse.** An empty
  history must refuse, not admit — a predicate that accepts the empty case certifies
  structure, not capability.
- `backfill_safety_factor` default **2.5** (measured: ×2.9 DF / ×2.0 reify covers 80% of
  realized durations; ×4.8 / ×4.0 covers 90%).
- `provable_assembly_delay(p)` = min over `p`'s still-blocked modules of
  `max(0, predicted_hold(holder) − elapsed_hold)`. If `p` is blocked on nothing, the delay is
  0 ⇒ refuse (`p` can assemble now).
- `backfill_max_park_age_secs` exists because the model named a casualty: without an
  age cutoff, one reify starver (4956) flips to never-dispatched.
- The final acquisition still goes through `try_acquire` under the lock — the free-module
  set read during scoring is never trusted as the acting basis (INV-3).
- Emit `park_backfill_granted` and `park_backfill_overstay` carrying predicted vs realized
  durations, so the measured 7–15% overstay rate is tracked in production rather than
  assumed, with a rate-threshold escalation (INV-4).

Park **installation** stays rank-1-gated — see §7.

### C8 — Below-rank-1 skip counting (observability only)

A **separate** counter `_passed_over_count[tid]`, incremented for every candidate the scan
passes over, surfaced in the state snapshot and in `scheduler_fairness_state`. It **must not**
feed `_bump_skip_and_maybe_park` — a test asserts that, because feeding it would silently
convert observability into below-rank-1 park installation, which §7 rejects.

**Storm escape (INV-4).** ~291 candidates × 4 ticks/min makes per-tick-per-task events a
storm. A task emits `task_passed_over` only when its count crosses a configurable stride
(default 100); the full counter set is always available in the snapshot.

## 6. Boundary-test sketch (H)

Each row faces **both** sides of a seam. Rows 1–3 face the fused-memory ↔ orchestrator seam;
4–8 face the scheduler ↔ lock-table seam.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Anchor survives a requeue | Task pending with `pending_since = T`; dispatched; requeued to `pending` | `pending_since == T` unchanged; scored age is measured from `T`, not from the requeue |
| 2 | Anchor resets only on un-cancel | Task `cancelled` with `pending_since = T`; set to `pending` at `T+Δ` | `pending_since == T+Δ`; the same task moved `blocked → pending` keeps `T` |
| 3 | Back-fill is complete and marked | Store holds N pending tasks with no anchor | After migration all N carry `pending_since == updated_at` and `pending_since_backfilled == true`; the migration reports N |
| 4 | Order is graded, not tied | The committed realistic candidate fixture (≥250 candidates, ≥3 tiers, ≥20 with D≥1 — asserted first, so a degenerate fixture fails loudly) | distinct scores ≥ 0.90·n (baseline 46/291 = 0.158); max tie group ≤ 10 (baseline 117); **0** candidates saturate (baseline 249/294 masked); and **every** candidate with D≥1 scores strictly above its own D=0 counterfactual — the CPM-is-live floor |
| 5 | No tier crossing at any input | Extremal inputs: `a → ∞`, `D = CPM_D_CAP`, continuity on | Total bonus < `TIER_WIDTH`; 0 inversions across the full ordering; a config whose budgets violate I-1 or I-3 is rejected at load, not at dispatch |
| 6 | Watchdog fires across a restart | Task continuously eligible past `idle_only_secs`, with a simulated process restart mid-window | `starvation_detected` emitted with the structured facts; a `source='starvation_watchdog'` pin visible in the override store; the operator's pre-existing pins keep their order; at the cap, an L1 fires instead of a pin |
| 7 | Fairness state survives a restart | Non-zero skip counts and ≥1 live park; snapshot; restart | Counts and parks rehydrated; a park whose owner went terminal during the restart is GC'd with `reservation_expired`; a restored park never blocks its own owner |
| 8 | Backfill admits only what fits, and every release is observable | A park whose beneficiary is blocked on a long-held module; one narrow candidate with ≥3 module-history samples and one with none | The predicted-fit candidate is granted with `park_backfill_granted`; the sample-less candidate is refused; a park older than the cutoff refuses; and across the whole run, every module leaving `_held` has exactly one matching `lock_released` |
| 9 | New scorer refuses to run un-anchored | Candidate set where only 50% carry `pending_since` (the orchestrator-ahead-of-fused-memory deployment order) | The tick scores with the **legacy** formula — not with every age at 0 — and emits `scoring_anchor_coverage_low` with the measured fraction; at 100% coverage the new formula runs and the event stops |

## 7. Explicitly rejected — do not re-propose

| Proposal | Why rejected (measured) |
|---|---|
| **Below-rank-1 park INSTALLATION** | Net-negative in every simulated variant: −4% to −11% throughput, idle reservation ×3.6–4.8, never-dispatched ×1.5–3. It gridlocks same-tier chains through INV-3 park collisions — 3534/3538 never dispatched across 4 seeds under K24-B32. A module budget trims the idle explosion ~2.7× but flips no sign. The safe discriminator would be park-set **disjointness**, not rank — that variant is **unsimulated**; it is named here as a gated follow-up only, not adopted. |
| **Park leases / expiry** | A no-op at 2–3h median process eras (P3 ≡ P0 within noise in both projects); a 24h lease is literally dead code — the process dies first. Confirms task 1228's lease removal. |
| **Naive chain-successor handoff** (reserve released modules across the merge window) | Worse on throughput **and** latency in both projects — the merge window's 0.5–2h median blocks more than it saves. |
| **α-retune as the fix** | α=1 leaves 3534 at rank 29 (still outside 24); the tipping point is α ≤ 0.5 or β ≥ 200. α=0.33 causes 9/24 window churn demoting 5 previously-dispatched tasks, violating the continuity ruling and whipsawing policy. And `age_alpha` is not in `RELOADABLE_FIELDS`, so it is not even hot-tunable. |
| **Percentile / within-set normalization (S2)** | Scores stop being a function of the task. Measured drift under candidate churn is up to **68 points** — larger than the 78-point margin that kept 3534 out — so a rank can cross the dispatch boundary because an unrelated task completed. Breaks "why didn't X run?" reproducibility from a task's own record. |
| **Path length L as a drop-in for D** | `L ≤ D` for **all 291** candidates, necessarily, so `log1p(L) ≤ log1p(D)` and the S4 score is provably ≤ S1's for every task — a uniform *shrink* of the CPM term that hands more relative weight to age. It **lowers** 3534's bonus (230 → 208). D never hides a deep chain (no L≥4 candidate has D≤3); it mis-*ranks* them. Depth-weighting needs its own scale and is out of scope unless a replay shows need. |
| **Lexicographic CPM buckets (S3)** | `floor(log2(1+D))` lets 3585 (D=1, age 211) displace 3090 (D=0, age **706**) on a bucket boundary. Within-tier starvation exposure rises to 706, the worst of any variant, and aging is structurally defeated — the same failure class as today's saturation with the polarity flipped. |
| **Flat continuity credit ≥ cpm max** | Inverts the fix: 115/291 candidates carry prior-dispatch history, and 3534 drops to rank 35–40, worse than baseline. |
| **Detecting continuity from dispatch-event history** | The events DB has both false positives (orphan releases, 3563-class stuck locks) and false negatives (173/294 candidates have no events at all, ever). |

## 8. Pre-conditions and substrate verification (G3)

Verified against `main` on 2026-08-06. No assumed capability is unresolved.

| Assumed capability | Status | Evidence |
|---|---|---|
| `_compute_score` / `_compute_age` / sort key are the sole ordering path | **exists** | `scheduler.py:4739`, `:4695`, `:6349` |
| `age_alpha`/`cpm_beta` are top-level config fields **not** hot-reloadable | **confirmed absent from `RELOADABLE_FIELDS`** | `config.py:4004`, `:4014`; reloadable list carries `fairness.skip_threshold` + `starvation_watchdog.*` only |
| A durable `pending_since` field | **DOES NOT EXIST** | `tasks` table has `metadata TEXT` + `updated_at`, no `created_at`, no `pending_since` (`sqlite_task_backend.py:91-106`) → **owned prerequisite task α** |
| A single status-write chokepoint that already sees `old_status` in-transaction | **exists** | `sqlite_task_backend.py:1938`, reads `old_status` at `:1993` |
| Tier-A metadata key registration | **exists, requires an add** | `_BLESSED_METADATA_KEYS`, `shared/src/shared/task_metadata.py:746` → deliverable of α |
| Override store supports pin + order + TTL | **exists** | `overrides.py:94`, `set_override(pinned=, pin_order=, ttl_until=)`; live pin queue confirms 12 entries |
| A way to distinguish a watchdog pin from an operator pin | **DOES NOT EXIST** | `overrides` has no `source` column → additive migration, deliverable of δ |
| Scheduler can emit events | **exists** | `self.event_store.emit(...)`, ~20 call sites in `scheduler.py` |
| A starvation event type | **DOES NOT EXIST** | no starvation member in `EventType`; `_file_starvation_info` emits nothing → new members, deliverable of δ |
| Durable cross-run event query for the zero-progress gate | **exists** | `EventStore.fetch_events_by_type_all_runs(type, task_id=)`, `event_store.py:697`; `EventType.zero_progress_requeue` at `:221` |
| `git for-each-ref %(ahead-behind:main)` | **exists** | git 2.43.0 installed; verified live against this repo |
| Exactly one lock-release bypass site | **confirmed** | `scheduler.py:7038` bypasses; `:6953` (`release_subset`) emits; `:7041` is the emitting path |
| Durable per-project orchestrator SQLite store to extend | **exists** | `run_store.py:70` (`scheduler_state`, pause-only) — sibling table added rather than migrated |
| No restore path for `skip_counts`/`parks` | **confirmed absent** | `scheduler_state.json` is written at `scheduler.py:6829`; the only reader is fused-memory's `read_scheduler_state` |
| Realistic candidate fixture for the cardinality test | **available** | `plans/evidence/scheduler-scoring-2026-08-06/candidates_scored.csv` (294 rows) pins the **baseline**; the new-formula fixture is regenerated by a committed dump script (deliverable of β) so no data is hand-transcribed |

## 9. Cross-PRD and cross-package seam ownership (G4)

No cross-**PRD** seams: a `search_tasks` sweep and a `plans/` grep on 2026-08-06 found no
owning PRD and no filed task for any item here (`age_alpha`/`cpm_beta`/`_compute_score`
appear in no document under `plans/` or `docs/`).

| Seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| fused-memory ↔ orchestrator | fused-memory produces, scheduler consumes | `metadata.pending_since` write rules + reader contract (C1) | **this PRD, task α** | queued |
| Task 3538 (pending, pinned) | adjacent, not overlapping | 3538 owns task-status truthfulness (`workflow.py`, `harness.py`); C5 owns lock-event truthfulness (`scheduler.py`) | 3538 / this PRD respectively | no edge; scope-checked |
| Task 3317 (pending) | consumes | transient-requeue backoff vs continuity credit — a crash-looper must not camp at tier-top | **this PRD, task γ** (zero-progress gate + credit decay) | no edge; C3 names the interaction |
| Task 2755 (done) | superseded | the idle-only starvation backstop, structurally unreachable as shipped | **this PRD, task δ** | superseded in place |
| `docs/legibility/design-invariants.md` | consumes | INV-1..7 walked in §11; no waivers | that doc | normative, unchanged |

## 10. Decomposition plan

Greek labels; real ids assigned at decompose time. `L` = leaf, `I` = intermediate.

**α — Durable `pending_since` anchor + back-fill** (I → β, δ)
Modules: `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
`shared/src/shared/task_metadata.py`, tests.
Implements C1 in full: one shared stamp helper called from every pending-landing write path,
the Tier-A key registration, and the one-shot back-fill.
*Unlocks:* β (the scorer's age term) and δ (the watchdog clock).
*Signal:* boundary rows 1–3 — a requeued task's anchor is unchanged and an un-cancelled
task's is reset, both observed through `get_task`; the migration reports the back-filled row
count.

**β — Scoring restructure + non-saturation known-answer test** (I → δ, ι, κ)
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/config.py`,
`orchestrator/tests/`. Depends α.
Implements C2: the three-term score, the config submodel with the I-1/I-3 validators, the
I-4 numeric tie-break, the I-6 anchor-coverage arming gate (D12), `RELOADABLE_FIELDS`
registration (D8), plus the committed fixture-dump script and the boundary-row-4/9 tests.
*Signal:* boundary rows 4 and 9 — on a realistic ≥250-candidate fixture, distinct scores ≥ 0.90·n,
max tie group ≤ 10, zero saturating candidates, and every D≥1 candidate strictly above its
own D=0 counterfactual; baseline 46-distinct / 117-tie / 249-masked reproduced from
`plans/evidence/.../candidates_scored.csv` in the same test as the comparison.

**γ — Continuity credit via work-product detection** (L)
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/tests/`. Depends β.
Implements C3: the single-fork TTL-cached branch probe off the loop thread, the durable
zero-progress gate, the credit decay, and the 3317 interaction.
*Signal:* a task with commits ahead of merge-base scores exactly `CONTINUITY_CREDIT` higher
than the same task without; a task with a `zero_progress_requeue` event scores zero credit;
and a test asserts at most one git subprocess per TTL window regardless of candidate count.

**δ — Starvation watchdog repair: durable clock, auto-pin, emitted event** (L)
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/harness.py`,
`orchestrator/src/orchestrator/overrides.py`, `orchestrator/src/orchestrator/event_store.py`,
`orchestrator/src/orchestrator/config.py`, `orchestrator/tests/`. Depends α, β.
(Depends β so the auto-pin's thresholds are chosen against the fixed scorer and the cap is
not consumed by a cohort the scorer was about to rescue.)
Implements C4 including the `overrides.source` column and the new `EventType` members.
*Signal:* boundary row 6 — `starvation_detected` observable in the event stream with its
structured facts, a `source='starvation_watchdog'` pin observable in `get_pin_queue`, both
across a simulated restart, and an L1 (not a pin) at the cap.

**ε — Truthful lock release: route the one bypass through the emitting path** (I → ζ)
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/tests/`. No deps.
Implements C5, including the single-writer grep-anchored invariant test.
*Unlocks:* ζ (the predictor's data quality — D11).
*Signal:* a blast-radius-expansion requeue emits `lock_released` (today it emits nothing),
and the single-writer assertion fails if any future call site bypasses again.

**ζ — Module hold-history predictor** (I → η)
Modules: `orchestrator/src/orchestrator/scheduler.py` (or a new
`orchestrator/src/orchestrator/hold_history.py`), `orchestrator/tests/`. Depends ε.
Rolling last-N per-module hold durations, seeded once at startup from
`fetch_events_by_type_all_runs` and fed in-process thereafter, with **one shared** helper
closing any span still open at a `service_restart` boundary (INV-5, D11). Exposes
`predicted_hold(task)` and `predicted_remaining(holder)`.
*Unlocks:* η (the admission test).
*Signal:* against a fixture event trace containing orphan releases and stuck-at-era-end
spans, the predictor reproduces the measured module medians and returns *no prediction*
below `backfill_min_samples` rather than a fabricated one.

**η — EASY-backfill admission through parks** (L). Depends ζ.
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/config.py`,
`orchestrator/tests/`. Implements C7.
*Signal:* boundary row 8 — a predicted-fit narrow task passes through a park and emits
`park_backfill_granted`; a sample-less candidate and an over-age park both refuse; overstays
emit `park_backfill_overstay` carrying predicted vs realized.

**θ — Durable fairness state across restarts** (L). Depends η (D10).
Modules: `orchestrator/src/orchestrator/run_store.py`, `orchestrator/src/orchestrator/scheduler.py`,
`orchestrator/tests/`. Implements C6.
*Signal:* boundary row 7 — snapshot → restart → skip counts and parks rehydrated and visible
in `get_scheduler_state`, with a now-terminal owner's park GC'd on the first tick.

**ι — Below-rank-1 passed-over counting (observability only)** (L). Depends β.
Modules: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/tests/`. Implements C8.
*Signal:* after a tick in which rank-1 is lock-blocked and rank-7 dispatches, ranks 2–6 show
a non-zero `passed_over` count in `get_scheduler_state` (today they leave no record at all);
a test asserts the counter never reaches `_bump_skip_and_maybe_park`; the stride limits
`task_passed_over` emission.

**κ — Integration gate: the whole seam, end to end** (L, the batch leaf).
Depends β, γ, δ, η, θ, ι.
Modules: `orchestrator/tests/`, `scripts/`.
Drives the **real** scheduler across the full seam in one test — a fused-memory-stamped
`pending_since` flowing into the score, a watchdog fire producing an auto-pin, a backfill
grant through a park, and a restart rehydrating fairness state — plus a committed script
that scores the **live** candidate set through the new function and prints the rank of the
highest-D `high`-tier chain head.
*Signal:* the script's output, recorded in the task's done-provenance, showing a
3534-class chain head (tier `high`, D ≥ 9, wait ≥ 48h) ranked inside `max_concurrent_tasks`
**with no pin and no boost applied** — the claim in §1 turned into a measurement.

**Contention note (not a dependency claim).** β, γ, δ, ε, ζ, η, θ and ι all touch
`scheduler.py`; they serialize on the module lock regardless of the edges drawn above.
Edges are drawn only where there is a functional prerequisite. The batch is deliberately
kept to one chain head (α) rather than split, per D12.

## 11. Design invariants walk (G7, advisory at author time)

Walked against `docs/legibility/design-invariants.md`. **No waivers required.**

- **INV-1 `contracts-machine-checked`** — the budget and subordination invariants are
  pydantic validators (C2 I-1/I-3), not comments; the `pending_since` write rule is one
  helper plus a test; the single-writer lock rule is a grep-anchored assertion (C5).
- **INV-2 `structured-facts-at-failure`** — `starvation_detected` carries the facts the
  emitter already holds (C4); today only prose exists. `park_backfill_overstay` carries
  predicted vs realized rather than requiring a log reconstruction.
- **INV-3 `corroborate-before-acting`** — auto-pin re-reads live task status before acting;
  backfill still acquires through `try_acquire` under the lock; restored parks are
  re-corroborated by the existing owner-state GC on the first tick; the new scorer arms on
  *measured* anchor coverage rather than assuming a deployment order (D12).
- **INV-4 `storm-escape-required`** — absent `pending_since` fails safe **and** counts
  toward a rate-threshold escalation, with the bulk case caught by the I-6 arming gate and
  its streak escalation; auto-pin is capped with the cap itself escalating; overstay rate
  carries a threshold escalation; `task_passed_over` emits on a stride.
- **INV-5 `no-lockstep-duplication`** — one `stamp_pending_since` helper across all
  pending-landing writes; one `pending_age_secs` that `eligible_age_secs` calls; one
  era-boundary span-closing helper shared by the predictor and any reporting.
- **INV-6 `status-matches-liveness`** — C5 touches an exit path; the status write stays
  ahead of the release, and routing through `Scheduler.release` restores the
  `_dispatched` discard and park clear that the bypass skipped.
- **INV-7 `holds-owned-and-bounded`** — the auto-pin names its owner (`source`) and carries
  a TTL, a cap, and an auto-release on dispatch; a restored park is bounded by the GC on
  restore and by `backfill_max_park_age_secs`; a backfill grant is bounded by its admission
  bound with an overstay event when the bound is exceeded.

## 12. Out of scope

- **Park-set disjointness as a below-rank-1 discriminator.** The one variant the model did
  not simulate, and the only one with a plausible path to being safe. Gated follow-up —
  file it only if a replay after this batch shows below-rank-1 protection is still wanted.
- **Depth-weighting (L) with its own scale.** D mis-ranks deep chains but never hides them.
  Revisit only if a post-landing replay shows need (§7).
- **PSI-admission interaction.** `dispatch_deferred` holds (8,215 events) are unmodeled in
  the simulation and untouched here; they throttle dispatch, they do not order it.
- **Restart cadence itself.** The 2.2h median process era is upstream of every parking knob,
  but reducing it is the fleet-redeploy/watchdog domain, not this PRD's.
- **`plan.files` vs `metadata.files` divergence** (task 3429) and merge-lane behaviour.
- **Reify-side rollout.** The changes are project-agnostic and ship in shared orchestrator
  code, but tuning any default for reify's file-granular crate tree is separate work; the
  measured cost of parking is **demand-conditional**, not proportional to reservation
  (reify reserves the same idle module-hours as DF and pays ~0 dispatches).

## 13. Open questions (tactical — decide during implementation)

1. **`AGE_HALF_SECS = 72h` vs a per-tier value.** Measured pending waits differ by tier
   (high p50 67.7h, medium p50 102.7h). *Suggested:* ship one global value; the knob is
   hot-reloadable, so retuning costs nothing. Decide in β.
2. **Where `HoldHistory` lives** — a new module versus more surface on `Scheduler`.
   *Suggested:* a new `hold_history.py`, since `scheduler.py` is already the batch's
   contention hot spot. Decide in ζ.
3. **`backfill_safety_factor` 2.5 vs 2.9.** 2.5 sits between the two measured 80%-coverage
   multipliers (DF 2.9, reify 2.0). *Suggested:* ship 2.5 and let the overstay events settle
   it with production data. Decide in η.
4. **Whether `task_passed_over`'s stride should be absolute or proportional to candidate
   count.** *Suggested:* absolute (100), simplest to reason about. Decide in ι.
5. **Auto-pin queue position.** Tail-append keeps operator pins ahead but means an auto-pin
   waits behind 12 existing entries. *Suggested:* tail-append; revisit only if κ's live
   measurement shows the auto-pin never reaching the front. Decide in δ.
