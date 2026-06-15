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

## Follow-up γ (post-deploy finding 2026-06-15): `get_tasks` encode hot path

After α+β merged and deployed, the watchdog kills **continued** — ~22-min intervals
against `WatchdogSec=600` (raised as an interim mitigation), event-loop thread still
~78–88% CPU. A second `py-spy` dump caught the residual GIL-hog on the **encode**
side: pydantic `model_dump_json` (`mcp/server/streamable_http.py:_create_json_response`)
serializing a large MCP tool **response** on the event loop — a full `get_tasks`
payload. α fixed the status-only **read/decode** path; this is the same full-tree cost
on the **serialize-out** path for callers that fetch whole task dicts.
`scheduler.get_tasks()` (`orchestrator/scheduler.py:1148`) dispatches MCP `get_tasks`
with only `{project_root}` — no filter — pulling reify's entire tree (**4,577 tasks ≈
9.84 MB** column text) every fetch, across 6 orchestrators. Of those **4,333 are
terminal** (done/cancelled); only ~244 are dispatchable.

**Decision (user, 2026-06-15):** server-side filtered `get_tasks` (option 1) — opt-in
status filter pushed into SQL (default contract unchanged), adopted by the scheduler's
hot paths. Mirrors α (lean query at the source), fixes all callers.

**Correctness crux (G6/G2):** active-only filtering must NOT hide terminal tasks from
the scheduler's **dependency-satisfaction** checks — those read the lean `get_statuses`
(status-only, made cheap by α), not the filtered `get_tasks`. The γ2 audit must confirm
every `scheduler.get_tasks()` consumer either tolerates active-only or is rerouted to
`get_statuses`.

Split per the multi-package rule (fused-memory + orchestrator exceed one architect budget):

- **γ1 — fused-memory: opt-in status filter on `get_tasks`** (leaf; no deps). Thread a
  `statuses: list[str] | None` (or `exclude_terminal: bool`) param through the server
  tool (`server/tools.py:2020`) → interceptor (`task_interceptor.py:3187`) → backend
  (`sqlite_task_backend.py:514` / `_get_tasks_internal:497`), pushed into SQL as
  `... WHERE tag=? AND status IN (…)` (uses `ix_tasks_status`). Param omitted = today's
  full unfiltered tree, byte-identical — contract unchanged. Shrinks both the per-row
  decode and the `model_dump_json` encode proportionally.
  - **Signal:** backend/tool unit tests — `get_tasks(statuses=[…])` returns only matching
    rows and the issued SQL carries a `status IN` predicate; omitting the param returns
    the full tree identical to today; an empty list returns no tasks.

- **γ2 — orchestrator: scheduler adopts active-only on hot paths** (leaf; depends on γ1).
  Audit the four `scheduler.get_tasks()` consumers (`scheduler.py:1129, 2392`;
  `workflow.py:918, 7208`); per-tick/dispatch hot callers request non-terminal statuses;
  confirm dependency-satisfaction reads `get_statuses` (`scheduler.py:1429`) not the
  filtered fetch; leave any consumer that genuinely needs terminal tasks on the full fetch.
  - **Signal:** scheduler tests — the dispatch path fetches active-only (asserted via the
    `statuses` argument / stubbed tool call), AND a task whose dependency is a *terminal*
    (done) task still dispatches correctly because dep-satisfaction reads `get_statuses`
    — proving the filter didn't blind the scheduler to done deps.

## Follow-up γ3 (post-deploy 2026-06-15): residual unfiltered `get_tasks` callers

After γ1+γ2 deployed (all 5 orchestrators restarted) the watchdog kills **stopped**, but
the fused-memory event loop still ran ~83% CPU (warm, no longer pegged/killing) — there are
residual **unfiltered** `get_tasks` callers γ2 didn't touch. Non-urgent (no kills); the goal
is a truly-idle loop. Two packages → split.

- **γ3a — dashboard: stop re-pulling every project's full tree on each refresh** (leaf;
  dashboard package). `dashboard/src/dashboard/data/fetch_tasks` (`data/tasks.py:85`) calls
  `get_tasks` with only `{project_root}` (full tree) on every render. **The γ1 active-only
  filter does NOT apply here** — `_shape_task` (`tasks.py:29`) keeps `updated_at` as the
  recency key for ordering/displaying **done** tasks, so the dashboard genuinely needs all
  statuses. Lever instead: **cache `fetch_tasks` per project with a short TTL** (decouples
  the frontend poll cadence from full-tree MCP serialization — a monitoring view tolerates
  ~15–30 s staleness). Alternative (architect's call, heavier — cross-package): a server-side
  field projection on `get_tasks` so the dashboard list fetches only the columns it renders
  (drop `details`/heavy `metadata`, lazy-load on expand) while keeping all statuses.
  - **Signal:** dashboard tests — repeated `fetch_tasks`/render calls within the TTL issue at
    most one `get_tasks` MCP call per project (asserted via a spy/mock counting calls); the
    rendered task set (incl. done tasks and their completed timestamps) is unchanged.

- **γ3b — orchestrator: filter the remaining unfiltered `get_tasks()` callers** (leaf;
  orchestrator package; uses γ1's filter, task 1758 done). Audit and apply `statuses=` where
  terminal tasks aren't needed: `scheduler.py:1147` (`_get_train_members` — train members are
  active work), `harness.py:1173/1201/1544`. Skip `cli.py:271` (manual one-shot, not a hot
  loop). Same correctness crux as γ2 — don't filter a caller that needs terminal tasks; route
  any done-status need to `get_statuses`.
  - **Signal:** orchestrator tests — each converted caller issues `get_tasks` with the
    non-terminal `statuses` set (captured via stub) and its behavior (train-member ordering,
    harness logic) is unchanged for an active-only tree.
