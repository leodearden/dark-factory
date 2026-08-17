# Would fixing dispatch SCORING have moved task 3534 — or is lock-aware scheduling the real lever?

Window analysed: **2026-08-02T17:41:00Z → 2026-08-06T14:48:47Z** (93.13h), i.e. batch-filing time
(tasks_snapshot `updated_at` for 3534/3535/3538) through `snapshot_taken_at.txt`.
Scripts: `derive_modules.py`, `lock_occupancy.py` (raw stdout captured in
`lock_occupancy_output.txt`), all in this scratchpad dir.

---

## 1. Required module sets (as the scheduler derives them)

Code path: `Scheduler._get_modules` (scheduler.py:7412-7449) reads **`metadata.files`
only** (never `metadata.modules`), and if non-empty calls
`module_charter.derive_modules(files, lock_depth)` (module_charter.py:44-66) →
`strip_directory_locks` (α-strip: drop directory-shaped entries) → `files_to_modules`
→ per-file `normalize_lock(f, depth)` (shared/locking.py:226-234: keep first `depth`
`/`-separated path components) → dedup + sort. Snapshot's `lock_depth = 4`.

| Task | `metadata.files` | Derived module-lock set (depth=4) |
|---|---|---|
| **3534** | `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_already_landed_dispatch_gate.py` | **same 2** (both paths are ≤4 components, so depth-4 coarsening is a no-op) |
| **3535** | + `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/task_ground_truth.py`, `orchestrator/tests/test_recovery_emission.py` | `harness.py`, `scheduler.py`, `task_ground_truth.py`, `test_recovery_emission.py` (4) |
| **3538** | `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_truthful_requeue_exits.py` | `harness.py`, `workflow.py`, `test_truthful_requeue_exits.py` (3) |

**Chain dependency (tasks_snapshot `dependencies` table)**: 3535 `depends_on` {3533 (done), **3534 (pending)**}; 3538 `depends_on` {3537 (pending), **3535 (pending)**}. So 3535 and 3538 are **not yet dispatch-eligible at all** — they're gated by the unmet-dependency check, not by scoring or locks, until 3534 (then 3535) completes. Only 3534 is a live candidate today.

**Data-quality flag**: `metadata.modules` (a separate field the task carries) diverges from the scheduler-derived set on **all three** tasks — e.g. 3534's `metadata.modules = ['harness.py']` only, missing the test file; 3535's omits the test file; 3538's omits nothing but is still a distinct list object. `_get_modules` never reads this field, so it's a stale/incomplete mirror, not a bug in dispatch itself, but a trap for any human or tool that trusts it instead of `metadata.files`.

---

## 2. Lock occupancy over the window

Reconstructed by pairing `lock_acquired`/`lock_released` events per module (full 4-month
history walked first to seed state at window start; **not** just the window slice — see
script). Cross-checked against `scheduler_state_snapshot.json.current_holders` (live state
at 2026-08-06T14:46:19Z, 2.5 min before the window end used here).

| Module | Occupied | Holders (task : span) |
|---|---|---|
| `harness.py` | **74.0%** (68.93h) | 3256 (4 spans, 42.3h total) → 3121 (8.23h) → 3256 (8.35h) → **3727** (10.09h, still held at snapshot — matches live state) |
| `scheduler.py` | **8.8%** (8.23h) | 3121 only |
| `workflow.py` | **73.3%** (68.23h) | 3113→3143→3110→**3536** (still held at snapshot — matches live state) |
| `task_ground_truth.py` | **96.5% raw** (89.87h) — **see data-quality flag below** | 3533 (2 legit spans, 13.33h) → 3563 (74.0h "held", **does not match live state**) |
| `test_already_landed_dispatch_gate.py` (3534-only) | 0.0% | never touched by any other task |
| `test_recovery_emission.py` (3535-only) | 0.0% | ″ |
| `test_truthful_requeue_exits.py` (3538-only) | 0.0% | ″ |

Cross-check to earlier cited figures: harness.py 74.0% vs. cited 73.8% and workflow.py
73.3% vs. cited 75.1% — both close (small deltas explained by window-boundary choice,
not a methodology error). **task_ground_truth.py reproduces the cited 96.5% exactly — but
that figure is an artifact, not real contention** (next section).

### Data-quality finding: `task_ground_truth.py`'s 96.5% is a stuck-lock artifact

`lock_acquired` for task 3563 fires at 2026-08-03T10:16:53Z; 7 minutes later `task_completed`
fires with `outcome: "requeued"` — **no `lock_released` event ever follows**. My
reconstruction is therefore forced to carry 3563 as the "holder" for 74 continuous hours,
which is exactly what produces the 96.5% figure. But `scheduler_state_snapshot.json`'s live
`current_holders` (ground truth, sampled at 14:46:19Z) shows `task_ground_truth.py` **FREE**
— a direct reconstructed-vs-live **MISMATCH** (flagged by the cross-check). The most
likely explanation: the requeue path does not call `Scheduler.release()` (which is the only
call site that emits `lock_released`, scheduler.py:7041-7058), so the lock sat "stuck" in
memory until the next **orchestrator process restart** (`service_restart` events at
2026-08-03T16:17:18Z and 02:14:37Z the next day silently rebuild `_held`/`current_holders`
from live in-progress claims — 3563 was no longer in-progress, so it simply drops out,
with no compensating event). Notably, **task 3538's own PRD title is literally "Truthful
REQUEUED/CANCELLED/infra exits"** — this looks like a live instance of exactly the bug
class that task is meant to fix. Net effect: task_ground_truth.py's *true* occupancy for
this window is bounded by 3533's two clean spans (13.33h) plus an indeterminate 3563 hold
(somewhere between ~7 minutes and ~5h53m, i.e. until the 16:17:18Z restart) — **roughly
14–21%, not 96.5%**. This doesn't touch 3534's own numbers (see below) but materially
changes the picture for 3535 once it's dep-unblocked.

**Other data-quality notes** (full history, all 7 modules, 162 total anomalies logged by
the script — not silently absorbed): a recurring pattern of `ORPHAN-RELEASE` (release with
no tracked holder — a missed/lost `lock_acquired`) and `DOUBLE-ACQUIRE` (a new holder
appears with no release from the previous one) appears roughly every few days across the
full 4-month log, on `harness.py`/`workflow.py`/`scheduler.py` specifically — the same
class of bug as the task_ground_truth.py case above, just usually resolved fast enough not
to distort a % figure materially. Only **3** anomalies fall inside the analysis window
itself, and none of them alter the 3534-headline numbers (one is a benign same-owner
"re-acquire" on `harness.py` at the 16:17:18Z restart with no gap in occupancy; the other
two are on `workflow.py`, outside 3534's own module set).

### The key number: 3534's own module set, joint free-time

Modules: `harness.py` + `test_already_landed_dispatch_gate.py` (the latter never
contended — 0% occupied always, so this reduces to harness.py's own timeline).

- **Union-occupied (either module held): 74.0% (68.93h)**
- **ALL-FREE (both simultaneously free): 26.0% (24.20h)**, across **7** free windows
- Free-window length: **min 0.035h (2.1min), p50 0.048h (2.9min), mean 3.457h, max 23.868h**
- **98.6% of all free time (23.868h of 24.20h) is a single contiguous window running from
  the moment of filing (2026-08-02T17:41:00Z) to 2026-08-03T17:33:05Z.** The remaining 69.2h
  of the window offers only 6 slivers totalling **~0.33h (≈20 minutes)**, each 2–6 minutes
  long, isolated hours apart.

For context, the same calc across the full 7-module union (3534+3535+3538's combined
footprint): **ALL-FREE only 3.5% (3.26h)**, 3 windows, p50 23.6min, max 2.51h — i.e. once
3535/3538 are folded in (adding the now-corrected-but-still-real `task_ground_truth.py`
and `workflow.py` contention), the joint-free ceiling for the *whole chain* is far tighter
than for 3534 alone.

---

## 3. Rank vs. locks: scanning order and skip semantics (code)

`_phase_select_scored` (scheduler.py:6327-6438):

- Builds `scored`: every dispatch-eligible candidate, sorted **descending by score, ties
  broken ascending by task_id string** (line 6349).
- `top_id = scored[0]` (line 6368) — the single highest-ranked candidate this tick.
- Iterates `scored` **in that exact order** (line 6372); for each candidate, tries
  `lock_table.try_acquire` (line 6381). **The first candidate — by score order — that can
  acquire wins and dispatches immediately** (return, line 6433). Lower-ranked candidates
  further down the list are never even tried once someone wins.
- Fairness bookkeeping (`_bump_skip_and_maybe_park`, called at line 6425 or 6436) is applied
  **only to `top_id`** — once, either when a lower-ranked task wins instead (6423-6425,
  "a lower-ranked task won — top was passed over this tick"), or when the whole loop
  exhausts with no dispatch at all (6436). **Every candidate strictly between rank #1 and
  the eventual winner is neither skip-tracked nor reserves anything** — it simply loses its
  turn silently, with no persistent record.
- `_bump_skip_and_maybe_park` (4506-4576): increments `_skip_count[task_id]`; once that
  count reaches the task's **tier** threshold (config.py:491-497 — `critical: 0, high: 1,
  medium: 2, low: 4, polish: 9999`) **and** it doesn't already hold a park, installs a
  "park" (reservation) on its **entire** requested module set — including modules that are
  currently free — so lower-or-equal-priority competitors can't grab them while it waits
  (scheduler.py:918-998, `install_parks`; `try_acquire`'s `_is_parked_blocks` check,
  1248-1271). **A park never evicts or preempts the current holder** — it only blocks *new*
  acquisitions by rivals; if the module is genuinely occupied, the parked task still just
  waits for the real holder to finish.

### Concrete historical inversions (from events)

All three show a strictly higher-ranked candidate skipped, then a strictly lower-ranked
candidate dispatching **within single-digit milliseconds, same tick**:

1. **2026-08-05T18:54:20.959Z** — task **3536** (pinned + boosted to effective priority
   **CRITICAL**, `skip_count=39`) skipped: could not acquire `workflow.py` +
   `test_workflow_merge_gating_strand.py`. **3ms later** (18:54:20.962Z) task **3554**
   (priority **LOW**) acquired 3 unrelated modules and dispatched.
2. **2026-08-06T14:46:19.244Z** — task **3618** (priority **HIGH**, `skip_count=31`)
   skipped on its module set (`OPERATIONS.md`, transcript-archive files, `git_ops.py`, …).
   **4ms later** task **3779** (priority **MEDIUM**) acquired an unrelated set
   (`CONTRIBUTING.md`, `docs/legibility/*`, `skills/*`) and dispatched.
3. **2026-08-06T07:06:26.653Z** — task **3076** (priority **HIGH**, `skip_count=7`)
   skipped on `harness.py`/`lane_lifecycle.py`/`workflow.py` etc. **5ms later** task
   **3561** (priority **MEDIUM**) acquired `fused-memory/src/fused_memory/services` and
   dispatched.

This is the mechanism working exactly as designed (rank determines *scan order*, not an
entitlement) — but it also means rank alone is no protection against a persistently
lock-blocked top candidate losing tick after tick to whatever unrelated, disjoint-module
candidate happens to rank just below it.

---

## 4. Starvation watchdog

**(a) Increment condition**: `_skip_count[task_id]` is incremented **only** for `top_id`
(the literal #1-ranked candidate that tick), and only by `_bump_skip_and_maybe_park`
(called from exactly the two sites above, scheduler.py:6425 and 6436). No other code path
touches `_skip_count`.

**(b) Can a never-top-ranked task accrue skips at all? No.** Confirmed three ways:
- By construction: only `top_id` is ever passed to `_bump_skip_and_maybe_park`.
- By the code's own documentation (scheduler.py:3502-3506, config.py:631-637): "A task
  that can never outscore a higher-tier candidate is never the top-scored candidate,
  accrues **ZERO** skips" — an explicitly named, known structural gap (cites reify-5166 /
  task 2755), which is *why* the separate `idle_only` backstop exists.
- Empirically: **3534/3535/3538 have zero rows of any event type — including
  `task_skipped`, `lock_acquired`, `phase_enter` — across the entire 4-month
  `runs_snapshot.db`**, and are absent from the live `skip_counts` snapshot (which holds
  only `{'3076': 7, '3618': 31}`). Task 3534 has been pending, dep-free, and dispatch-
  eligible for ~93h and has **never once** been the top-ranked candidate.

**(c) What `_resolve_starvation_escalation` does, and has it ever fired**:
- `_resolve_starvation_escalation` (3573-3589) is the **resolve**, not the fire, side —
  called at both dispatch sites to auto-close an open watchdog escalation once the task
  finally dispatches; no-op if never escalated.
- The **fire** path is `_apply_starvation_watchdog` (3433-3571), gated by an OR of two
  conditions (config.py:562-654, defaults: `skip_threshold=50`, `idle_secs=idle_only_secs=
  259200s/72h`, no override in `orchestrator_config_snapshot.yaml`):
  - **dual gate**: `skip_count ≥ 50` AND continuously-eligible ≥ 72h (the lock-contention
    path — structurally unreachable for a task with 0 skips, i.e. unreachable for 3534);
  - **idle-only backstop**: continuously-eligible ≥ 72h regardless of skip_count (the
    "never-top-scored" path — 3534's actual class).
  - On fire, calls `on_starvation_warn` → harness.py's `_file_starvation_info` (6435-6497),
    which files a level-0 `severity=info, category='risk_identified'` escalation directly
    onto `self._escalation_queue` — **without ever calling
    `self.event_store.emit(EventType.escalation_created, …)`**. None of the 26
    `escalation_created`-emitting call sites in harness.py are inside this function.
- **Data-source caveat**: this means `runs_snapshot.db`'s `events` table is **not** the
  authoritative channel for this specific escalation category — the escalation server's
  own store (not included in this snapshot) would be the ground truth. Within that caveat:
  zero of the 1,854 `escalation_created` events in the 4-month history carry a
  starvation-shaped category (observed categories: `task_failure, review_suggestions,
  review_issues, infra_issue, post_merge_verify, preexisting_main_break, stranded_blocked,
  merge_ff_failed, design_concern, wip_conflict, milestone_gate` — no `risk_identified`
  row at all), and a full-text scan of all event `data` blobs for "starvation" found zero
  matches among `escalation_created` rows. Combined with config.py's own note that all 209
  prior firings (through 2026-07-15) happened under a now-superseded, far more sensitive
  threshold before being deliberately retuned to today's 72h bar specifically so it would
  stop firing on ordinary contention — the defensible read is that **under current config
  the watchdog has not fired at all in this snapshot's history**.
- **Why not, specifically for 3534**: neither `_skip_count` nor the idle-clock anchor
  (`_streak_starvation`/`_starvation_first_seen`) is included in the 11 keys
  `get_state_snapshot` persists (scheduler.py:6554-6637 — `skip_counts, parks,
  park_stacks, effective_priorities, pin_queue, overrides, current_holders, lock_depth,
  is_paused, pause_reason, snapshot_at`), and there is **no restore/load path** anywhere in
  scheduler.py or harness.py that reads that JSON back into `_skip_count` or the streak
  registry — both are initialized empty in `__init__` (line 1499) and live purely in
  process memory. The window contains **2** `service_restart` events with
  `"service": "orchestrator"` (2026-08-03T16:17:18Z, 2026-08-04T02:14:37Z). From the later
  of those to the snapshot is only **~60.6h** — under the 72h `idle_secs`/`idle_only_secs`
  default. If either restart reset 3534's idle-clock anchor (plausible, since nothing
  persists it), 3534 simply hasn't yet accumulated a clean 72-continuous-hour run since the
  last restart to trip the backstop — consistent with, though not conclusively provable
  from, this snapshot alone.

**Verdict**: the watchdog is designed to cover both starvation classes (dual gate =
lock-contention-while-top-ranked; idle-only = never-top-ranked/pure-ranking), but for
**this** incident it protects against **neither in practice**: 3534 is squarely in the
never-top-ranked class (dual gate structurally unreachable, 0 skips), and the idle-only
backstop that's supposed to catch exactly this case is undermined by the same in-memory,
non-persisted state that (separately) resets `_skip_count` on every orchestrator restart —
and restarts are frequent enough (2 inside a single 93h window) that a clean 72h run may
never accumulate.

---

## 5. Bottom line: would a scoring fix alone have gotten 3534 dispatched materially earlier?

**Partially, and only by luck of timing — not robustly.**

- 3534's own module set sat **completely free for ~23.9 continuous hours immediately after
  filing** (17:41 08-02 → 17:33 08-03) — 98.6% of all the free time this window offers.
  Tick cadence looks sub-second when the scheduler has work to place (the three inversion
  examples above show 3–5ms between a skip and the next dispatch), and a "high"-tier task's
  skip-threshold is 1 (i.e. it parks on its very first skip). So **if a scoring fix had
  made 3534 rank #1 from the moment of filing, it would very plausibly have dispatched
  within minutes** — call it a counterfactual delay of **~0h**, against an actual delay of
  **>93h and counting**.
- But that conclusion is fragile, not structural. It depends on the fix landing 3534
  literally at rank **#1** (not merely "into the top 24") **during that one early window**
  — a candidate ranked #2–#24 gets no lock-acquisition attempt and no fairness credit at
  all in a given tick if #1 succeeds; only the single top-ranked candidate is ever tried.
  Once that first window closed, the ceiling collapses for *any* candidate, however ranked:
  harness.py alone is occupied 74% of the remaining time, with free slivers of 2–6 minutes
  isolated hours apart, and the wider chain (adding scheduler.py/task_ground_truth.py/
  workflow.py for 3535/3538) drops the joint-free ceiling to 3.5% with a 23-minute median
  window. A scoring fix, by construction, orders candidates by tier/age/CPM — it has no
  mechanism to notice "your module is free right now, dispatch now" versus "your module is
  occupied, don't bother trying" between ticks.
- Since 3535 and 3538 are still dependency-gated (not even candidates yet) and will inherit
  this same lock landscape the moment they become eligible — and since the *real* driver of
  future starvation on this chain is that 74–97%-occupied regime, not the scoring formula —
  **lock-aware scheduling (dispatch-time preference for whichever eligible candidate's
  modules are free right now, or a scoring boost keyed to lock availability rather than
  pure age/CPM) is the durable fix**; the scoring-alone fix would only have rescued this one
  case, and only because it happened to file into a rare, unusually long free window.

---

## Scripts (scratchpad)
- `derive_modules.py` — Q1 module derivation (verbatim reimplementation of
  `shared/locking.py` + `orchestrator/module_charter.py` logic, run read-only against
  `tasks_snapshot.db`)
- `lock_occupancy.py` — Q2 occupancy/free-window reconstruction from `runs_snapshot.db`
  events, with anomaly detection and a cross-check against
  `scheduler_state_snapshot.json.current_holders`
- `lock_occupancy_output.txt` — full run output (anomaly log, per-module intervals, both
  free-window analyses)
