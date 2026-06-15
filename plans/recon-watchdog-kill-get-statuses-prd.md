# PRD: recon watchdog-kill root cause — `get_statuses` read amplification + suppression-storm alarm

**Status:** authored 2026-06-15. Decompose target: `dark_factory` (`/home/leo/src/dark-factory`).

## Problem / consumer

Over a multi-hour window on 2026-06-15, **~90% of full memory reconciliation runs
failed** (41 `run_started` / 4 `run_completed` in 8h). The chain, established by a
live `py-spy` dump:

1. `fused-memory.service` is killed by its systemd watchdog (`WatchdogSec=120`)
   roughly every ~10 min — **38 watchdog timeouts in 8h** (`Failed with result
   'watchdog'`), each a SIGABRT that aborts the in-flight full recon cycle (a cycle
   takes ~10–15 min across three LLM stages, so it rarely survives to
   `run_completed`).
2. The watchdog fires because the asyncio **event-loop MainThread is CPU-bound
   holding the GIL**, which starves the dedicated `fused-memory-sd-watchdog`
   heartbeat thread (the task-1731 fix) past the 120 s deadline. A dedicated OS
   thread escapes *asyncio* scheduling but **not** the GIL; the GIL-convoy effect is
   worse, not better, on this 32-core host.
3. `py-spy` caught the hot path red-handed, `active+gil`:
   `get_external_statuses → get_statuses → get_tasks → _get_tasks_internal →
   _row_to_task → json.loads` (`json/decoder.py`).

**Root cause.** `task_interceptor.get_statuses(project_root, ids=[…])`
(`fused-memory/src/fused_memory/middleware/task_interceptor.py:3194-3230`) answers a
**status-only** query by loading the **entire** task tree (`tm.get_tasks`) and
filtering to `ids` *in Python*. `get_tasks → _get_tasks_internal`
(`sqlite_task_backend.py:497-510`) issues `SELECT * FROM tasks WHERE tag=?` and runs
`_row_to_task` → **`json.loads` per row** (`sqlite_task_backend.py:251`) — on the
event-loop thread (only the SQL fetch is off-thread; the decode comprehension is
not). For **reify** that is `SELECT *` over **4,576 rows + 4,576 `json.loads`
decoding 3.03 MB of metadata, per call**, to read a handful of status fields. The
orchestrator scheduler calls this **every dispatch tick** via `get_external_statuses`
(`server/tools.py:2184-2205`) to resolve cross-project `external_deps`, with **no
caching**. `status` is a top-level indexed column, so the correct cost is O(K).

The failures were **invisible**: a watchdog SIGABRT kills the process mid-run, so it
can't self-report a `recon_failure`; on restart the stale-run reaper recovers the
orphan but classifies it `dead_owner_shielded` and **suppresses** the
`recon_stale_run` escalation (`harness.py:708-725`, task 1731). That suppression is
correct for the ~6/**day** benign-restart baseline, but it silently absorbed ~38
systemic failures in 8h with **zero** escalations to the recon-watcher (port 8103).
The only surfaced signal — `ReconciliationBacklogExceeded` for reify (501→541, all 5
still pending) — is routed by `backlog_policy._maybe_write_escalation` to **reify's
own** L1 queue, not the recon queue. (`load ~17 on 32 hardware threads with ample
free RAM is light and will not get lighter; host load is NOT the cause — the tree
only grows.`)

**Consumers (G1).** Task α: the orchestrator scheduler's external-dependency
resolver (`get_external_statuses`) and the systemd-watchdog liveness of
`fused-memory.service` (and any status-only `get_statuses` caller). Task β: the
**recon-escalation-watcher** loop (`/recon-escalation-watcher`, port 8103) — it must
receive an aggregate alarm when the suppression rate indicates a systemic break.

## Relationship to prior work (G4 — no contested ownership)

- α **preserves** the `get_statuses` contract introduced for status-only callers
  (`{id_str: status_str}`); it changes the *implementation*, not the seam. No
  reciprocal ownership ambiguity.
- β is **additive** to the task-1731 `dead_owner_shielded` suppression — it keeps
  per-event suppression and layers a rate-aggregate alarm on top. It reuses the
  existing `_escalate` routing (A7b contract: harness submits, the 8103 watcher is
  the sole closer — `plans/afk-A7-recon-closure.md`). Owner of the recon-escalation
  path = this harness work.
- Aligns with the standing directive `feedback_prefer_loud_escalation_over_silent_degradation`:
  a guard that detects a systemic break must escalate loudly; degradation must never
  suppress the escalation.

## Approach (G5 — B, not B+H)

Both tasks are behavior-preserving hardening on **existing** seams within a single
package (fused-memory); no new cross-module contract. α is a strict
contract-preserving substitution of the read path (covered by two-way contract
tests on the `get_statuses` boundary — see its signal). β is an additive aggregate
counter feeding the existing `_escalate`. Per-task blast radius ≤ 1 package → **B**.

## Pre-conditions for activating

None — all substrate exists today (G3 verified, see capability manifest):

- α: `tasks.status` is `TEXT NOT NULL` (`sqlite_task_backend.py:62`) with index
  `ix_tasks_status (tag, status)` (`:69`) and PK `(tag, id)` (`:66`); the targeted
  single-row query pattern to mirror is `get_task` (`:522-547`,
  `SELECT * FROM tasks WHERE tag=? AND id=?`). `id` column is `INTEGER` — the new
  path must cast incoming `ids` to int for the `IN (…)` clause and re-key the result
  to `str` (mirroring `get_statuses`' current `str(...)` keys).
- β: `self._escalate(category, run_id, summary, detail, *, finding=…)` exists and
  routes to the 8103 recon queue with dedup-fold-on-ingest (`harness.py:838`);
  `self.config` is `config.reconciliation` (`harness.py:272`) and recon knobs live
  under `reconciliation:` in `config/config.yaml:87` (e.g. `stale_lock_seconds`,
  `max_staleness_seconds`) — the new threshold/window knobs slot there with in-code
  defaults so β works even if config is absent. The suppression site is
  `harness.py:708-725` (the `dead_owner_shielded` branch).

## Out of scope

- Widening `WatchdogSec`, or re-architecting the heartbeat to be GIL-independent
  (e.g. a subprocess/`timerfd` notifier). Worth a follow-up — the heartbeat staying
  GIL-dependent is a latent fragility — but α removes the only known trigger, so it
  is not queued here.
- Removing or reducing the per-event `dead_owner_shielded` suppression (β is purely
  additive).
- The leaked test collection `fused_test_cost_store_failure_does_n0` polled by every
  `get_status` health check — a separate cleanup, not load-bearing here.
- Re-routing the `backlog_policy` escalation off reify's L1 queue.
- Caching `get_external_statuses` results (α makes the uncached call cheap enough
  that a cache is unnecessary).

## Decomposition plan

Two leaves, **independent** (no ordering dependency); α is the actual fix, β is the
visibility guard so a recurrence can't hide again.

- **α — `get_statuses` answers status-only in O(K) without decoding metadata**
  (leaf). Add a lightweight backend method on `SqliteTaskBackend` (e.g.
  `get_statuses_raw(project_root, tag, ids)`) that issues
  `SELECT id, status FROM tasks WHERE tag=?` plus `AND id IN (…)` when `ids` is
  given (omit the `IN` clause for `ids=None` → all), returning `{str(id): status}`
  directly from the two selected columns. Route `task_interceptor.get_statuses`
  through it. The new path must **never** call `_get_tasks_internal` / `_row_to_task`
  / `json.loads`. Preserve the exact current contract: returns `{id_str: status_str}`;
  a row whose status is NULL/missing surfaces `'unknown'`; `ids=[]` → `{}`;
  `ids=None` → all tasks; unknown ids silently omitted; `str`-keyed.
  - **Signal:** unit/integration tests on the `get_statuses` boundary assert (1)
    contract preservation — for a seeded multi-row tree, `get_statuses(ids=subset)`,
    `ids=None`, `ids=[]`, and an unknown id all return exactly what the pre-change
    implementation returns; and (2) **no metadata decode** — `_row_to_task` /
    `json.loads` is not invoked on the `get_statuses` path (spy/patch asserts zero
    calls), or equivalently the SQL issued projects only `id, status`. Motivating
    operational outcome (not the pass/fail oracle): with α deployed, resolving
    `external_deps` stops decoding the foreign tree, `get_statuses` no longer appears
    as a GIL-holding hot path, the `Watchdog timeout` rate falls back toward the
    ~6/day baseline, full recon runs reach `run_completed`, and reify's
    reconciliation backlog drains below the 500 threshold.

- **β — aggregate alarm when `dead_owner_shielded` suppressions storm** (leaf;
  independent of α). At the suppression site (`harness.py:708-725`) keep suppressing
  the individual `recon_stale_run`, but record each suppression in a rolling-window
  counter (per-process; project label carried in the payload). When the count within
  the window crosses a configurable threshold (**default: ≥ 6 suppressions per
  rolling 60 min** — ~24× the ~6/day baseline; expose `*_storm_threshold` and
  `*_storm_window_seconds` under `reconciliation:` with these in-code defaults),
  emit **one** loud, non-suppressed escalation via `self._escalate(...)` with a
  stable `finding` identity (so dedup folds it to a single pending item) summarizing
  the storm — count, window, affected project(s), and the cause hint
  *"watchdog SIGABRT churn — full recon runs not completing"*. Rate-limit to at most
  ~once per window so it does not re-fire per suppression.
  - **Signal:** unit test drives the suppression path N times within the window and
    asserts exactly one `_escalate` call is made at the threshold crossing; N−1
    within the window asserts **zero**; continued suppressions in the same window do
    not re-fire (≤ 1 per window). The escalation reaches the recon queue (8103) — i.e.
    it is emitted via `_escalate` (the path the recon-watcher consumes), not the
    suppressed branch.

## Open questions (tactical)

- α: `id IN (…)` parameter binding vs. a temp-table/`json_each` join for very large
  `ids` lists — architect's call; external-dep `ids` lists are small (handful), so
  a bound `IN` is fine. Keep the off-thread/aiosqlite execution model `get_task`
  uses.
- α: whether to also short-circuit `get_external_statuses`' per-project batch when a
  project's `ids` list is empty — minor, local.
- β: rolling-window representation (timestamp deque vs. bucketed counter) and
  per-project vs. global threshold — local, recoverable; default to global count
  with per-project labels in the payload unless a per-project view proves needed.
- β: exact escalation `category` string — reuse `recon_stale_run` with a distinct
  storm `finding` identity, or a new `recon_watchdog_kill_storm` category — pick the
  one the 8103 watcher surfaces most legibly.
