# Dashboard availability — supervision correctness + the unindexed-journal defect

**Status:** active · authored 2026-07-30 · approach **B + H** (contract +
boundary-test sketch for the supervision seam)

## Goal

The dashboard stays continuously available under normal fleet load, and its
supervision fails **loud** rather than flapping silently.

User-observable end state:

- Reloading `http://127.0.0.1:8080/` renders the SPA every time — never a
  black screen, never "No tasks" on an orchestrator that has tasks.
- `journalctl --user -u dark-factory-dashboard.service --since -2h | grep -c
  'Started dark-factory-dashboard'` returns `0` on a healthy host.
- If the dashboard genuinely becomes unhealthy, an operator finds a
  **born-at-L2 escalation** saying so — instead of discovering a 24-hour
  outage by eye.

## Background — the 2026-07-30 incident (all figures measured)

The dashboard appeared "hung, black screen on reload". It was not hung: the
watchdog was killing it every ~30s. **192 restarts in 3h**, peak 113/hour,
degrading for 24h+. Dashboard code was unchanged for 12 days.

The measured causal chain, in the order it actually bites:

1. **`write_ops` (16.3M rows / 6.5 GB) has no `created_at`-leading index.**
   The schema carries `idx_wo_causation`, `idx_wo_project_time`,
   `idx_wo_operation`, `idx_wo_kind_time`, `idx_wo_agent_time` — every one
   with `created_at` in *second* position, so a bare `WHERE created_at >= ?`
   cannot range-seek. `EXPLAIN QUERY PLAN` confirms `SCAN` (not `SEARCH`) for
   all three `dashboard/data/write_journal.py` queries. The sibling table
   already has `idx_bo_created ON backend_ops(created_at)`; `write_ops`
   simply never got the equivalent.
   - measured: `SEARCH` (seekable prefix) **0.00s** vs `SCAN` **20.94s** on
     the same 24h window — a ≥2000× gap.
   - measured: `get_operations_breakdown` **16.51s**, `get_agent_breakdown`
     **21.47s**.
2. **`/api/v2/dashboard/memory-graphs` therefore takes 108 seconds** to
   return 1,185 bytes — and `static/redux/data.js:165` polls it **every 3
   seconds** with no in-flight guard. ~36 full scans of a 6.5 GB table stay
   permanently stacked, all serialised onto the single aiosqlite worker
   thread that `DbPool` keeps per database path.
3. **`/healthz` probes that same `write_journal.db`**, so its `SELECT 1`
   queues behind the pile: measured **28.8s**, then a 60s timeout, then
   **503 at 50.6s**.
4. **The watchdog restarts on a single 5s miss.**
   `curl -sf --max-time 5 /healthz || systemctl --user restart` on a 30s
   timer. Step 3 guarantees that miss, every time.
5. **The restart cannot complete gracefully.** uvicorn logs `Waiting for
   connections to close`, but the browser's keep-alives never idle out under
   a 3s poll → `TimeoutStopSec=15` → SIGKILL. Measured **~27% downtime in a
   contiguous ~16s dead window** per cycle.
6. **Black screen.** The SPA is unbundled: `index.html` pulls 5 CDN scripts,
   5 JS files and **15 `.jsx` files Babel-compiles in-browser**. A reload
   crossing the dead window loses `app.jsx`; React never mounts; `#root`
   stays empty against the dark `--bg-0`. CDN reachability and all 21 local
   assets were verified healthy — the only variable was the dead window.
7. **"No [object Object] tasks."** `ACTIVE_TASKS` is the largest payload
   (5.0 MB vs 60 KB for orchestrators) and so the most likely to be
   SIGKILLed mid-flight; `refreshOne` deliberately keeps prior values on
   failure, which on a fresh load is the empty default. The garbled text is
   a separate stale-refactor bug (below).

The 4.1–5.4 GB cgroup peaks are page cache from repeatedly scanning 6.5 GB
(cgroup v2 accounts page cache), not a leak: a 6-cycle stacked-poll probe
moved anon RSS only 450 MB → 823 MB.

**Two premises were tested and refuted**, and are recorded here so they are
not re-litigated:

- *"Each restart re-walks 8,670 escalation JSONs cold, so analytics needs a
  persistent rollup."* Measured cold walk: **0.855s / 76 MB peak** — inside
  its own ratified 2000ms budget. Not a cause; rollup dropped (see Out of
  scope).
- *"The 6.5 GB journal must be pruned to restore latency."* With the index,
  size stops driving latency. Retention downgrades to a growth **alarm**.

Prior art that this incident re-broke: task **326** already documented the
exact aiosqlite-serialisation mechanism ("DbPool caches one aiosqlite
connection per DB file and aiosqlite serializes through a single background
thread … page load latency becomes the sum of all queries rather than the
max") and fixed it by staggering HTMX loads; task **185** added polling
jitter. The redux SPA rewrite reintroduced simultaneous fan-out via
`Promise.all`. Task β re-establishes the lesson **with a regression test** so
it cannot be lost a third time.

## Sketch of approach

Four independent lines, smallest-blast-radius first:

1. **Fix the defect** (α) — add `idx_wo_created`. One idempotent line in the
   existing `WriteJournal._migrate()`, which already does
   `CREATE INDEX IF NOT EXISTS`.
2. **Stop any single slow endpoint from stacking** (β) — in-flight guard +
   error backoff + jitter in `data.js`, with a regression test.
3. **Make supervision correct** (γ, δ, ε) — shallow liveness probe,
   consecutive-failure hysteresis, startup grace, a restart-rate ceiling that
   escalates and *stops*, a bounded shutdown drain, and a `/healthz` whose
   deadline is structurally deliverable.
4. **Keep it honest over time** (ζ, η, θ) — journal growth alarm, unit-file
   drift check, and the empty-state legibility fix.

## Resolved design decisions

1. **Liveness ≠ readiness.** The watchdog probes the shallow `/api/health`
   (`app.py:386`, a bare `{'status':'ok'}`), never `/healthz`. A check that
   queries three databases measures *load*; wiring it to a restart action
   makes the supervisor fire hardest exactly when restarting is most harmful.
   `/healthz` stays a deep **diagnostic/readiness** surface, wired to nothing
   that kills.
2. **Hysteresis before action.** `N=3` consecutive probe failures plus a
   startup grace window before any restart. A single-sample kill has no
   hysteresis and fires on any transient.
3. **Storm escape, INV-4 (`storm-escape-required`).** On exceeding the
   restart-rate ceiling the watchdog files a **born-at-L2** escalation via
   `escalation submit` and **stops restarting**. A cleanly-down dashboard
   that says why beats one flapping at 27% availability — which is precisely
   the failure that went unnoticed for 24h. Ratified by the user 2026-07-30.
4. **Probe budgets must be structurally deliverable.** `/healthz` today has
   `_DB_PROBE_TIMEOUT = 5.0` × 3 databases = up to 15s behind a
   `curl --max-time 5`, so its `degraded` verdict was **never deliverable** —
   it could only ever manifest as a timeout (measured: 503 at 50.6s). Every
   probe gets a whole-handler deadline strictly below its caller's, and
   `curl -sf` treating 503 as failure means a "degraded" answer must never
   itself be the restart trigger (decision 1 already ensures this).
5. **The index, not retention, is the latency fix.** Retention is reduced to
   a growth alarm (ζ); pruning is deferred until the alarm fires. Ratified
   by the user 2026-07-30.
6. **Analytics stays as ratified.** `escalation-lifecycle-dashboard-prd.md`
   §Resolved-4 (per-request walk + TTL cache) is untouched; its tripwires
   (2664 +30d, +90d, +180d) remain the correct trigger for the rollup
   revisit, on evidence rather than on this incident's false premise.
7. **The frontend fix ships with a regression test.** Tasks 185/326 learned
   this lesson and the rewrite lost it; a fix without a test invites a third
   recurrence (INV-5 in spirit — the constraint must live somewhere
   machine-checked, not in reviewer memory).

## Pre-conditions

All verified on main, 2026-07-30:

- `WriteJournal._migrate()` already issues `CREATE INDEX IF NOT EXISTS`
  (`fused-memory/src/fused_memory/services/write_journal.py:168-174`) — α is
  an exact-pattern addition.
- Shallow `/api/health` exists (`dashboard/src/dashboard/app.py:386-388`).
- uvicorn **0.44.0** supports `--timeout-graceful-shutdown` and
  `--timeout-keep-alive` (`--help` verified).
- `escalation submit` is a script-callable born-at-L2 writer that works
  **without the MCP server** — its module docstring names "a detached systemd
  OnFailure unit" as the intended caller. Interface: `--queue-dir --task
  --severity {critical,urgent} --category --summary [--detail --agent-role]`;
  console script `escalation` (`escalation/pyproject.toml:21`).
- Unit-parity precedent: `scripts/check_fused_memory_unit_parity.py`.
- Prune precedent in the same class (for a future retention task, not ζ):
  `WriteJournal.prune_idempotent_ops` / `prune_mem0_intents`.

## Cross-PRD relationship (G4)

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/escalation-lifecycle-dashboard-prd.md` | consumes | `/api/v2/dashboard/escalation-analytics` availability; `scripts/check_esc_analytics_perf.sh` predicate (tasks 2664 +30d / +90d / +180d) | **other-prd** | wired — this PRD changes nothing there; §Resolved-4 stands and the tripwires keep their meaning. Restoring availability is what lets 2664 measure a warm cache as designed. |
| `plans/escalation-lifecycle-dashboard-prd.md` | — | escalation-archive rollup / incremental index | **other-prd** | not taken — measured 0.855s cold; revisit stays gated on that PRD's own tripwires |

No reciprocal-ownership ambiguity: this PRD explicitly declines the analytics
seam rather than claiming it.

## Contract — the supervision seam (B + H)

The load-bearing seam is **watchdog ↔ dashboard service**. Today it is three
words of inline shell in a unit file; that is a contract in prose (INV-1),
which is why it was wrong and stayed wrong.

**Probe contract**

| Field | Value |
|---|---|
| Endpoint | `GET /api/health` (shallow; no DB access) |
| Success | HTTP 200 within `PROBE_TIMEOUT` (default 5s) |
| Failure | non-200, connection refused, or timeout |
| Never used | `/healthz` — deep, and its 503 must not trigger restarts |

**Actuation contract**

| Rule | Value |
|---|---|
| Startup grace | no probe counts as failure within `GRACE_SECS` (default 60s) of `ActiveEnterTimestamp` |
| Hysteresis | restart only after `FAIL_STREAK` (default 3) **consecutive** failures |
| Streak reset | any success resets the counter to 0 |
| Rate ceiling | `MAX_RESTARTS` (default 3) within `RATE_WINDOW_SECS` (default 3600) |
| On ceiling | file born-at-L2 via `escalation submit`, then **stop restarting** (INV-4) |
| Escalation dedup | at most one open L2 per ceiling episode |
| State | persisted across oneshot invocations (the timer runs a fresh process each tick) |

**Invariants**

- I1 — A healthy dashboard produces **zero** restarts. (The regression this
  whole PRD exists to prevent.)
- I2 — A single transient failure produces **zero** restarts.
- I3 — A sustained failure produces **exactly one** restart per
  `FAIL_STREAK` episode, never a loop.
- I4 — Hitting the ceiling produces **exactly one** L2 and **no** further
  restarts until an operator intervenes.
- I5 — The watchdog never restarts a unit it did not observe failing
  (no action on a probe it never ran — e.g. its own startup error).

## Boundary-test sketch (B + H)

Facing both sides of the seam. Task γ names this table as its signal.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Healthy steady state | dashboard up, normal load | 0 restarts over ≥30 probe ticks; streak stays 0 |
| B2 | Single transient miss | one probe forced to fail, then recovery | 0 restarts; streak returns to 0 |
| B3 | Sustained outage | probe fails `FAIL_STREAK` times consecutively | exactly 1 restart; streak resets after |
| B4 | Restart storm | forced failure persists past `MAX_RESTARTS` in `RATE_WINDOW_SECS` | exactly 1 born-at-L2 record in the queue dir; **no** further restarts |
| B5 | Startup grace | unit restarted, probe fails during `GRACE_SECS` | 0 restarts attributable to the grace window |
| B6 | Slow-but-alive | `/api/health` 200 while `/healthz` is 503/slow | 0 restarts (proves liveness ≠ readiness) |
| B7 | Bounded drain | `systemctl restart` with a live 3s-polling client attached | restart completes within the configured drain; **no** `Killing process … SIGKILL` line in the journal |

## Decomposition plan

Labels are PRD-local; task IDs assigned at decompose.

| Label | Title | Modules | Prereqs | Observable signal |
|---|---|---|---|---|
| **α** | Index `write_ops(created_at)` — remove the timeseries full scan | `fused-memory/services/write_journal.py` | — | `EXPLAIN QUERY PLAN` reports `SEARCH … (created_at>?)` not `SCAN` for the **timeseries** and **agent-breakdown** queries; `get_operations_breakdown` **remains `SCAN` by design** (see note); `GET /api/v2/dashboard/memory-graphs` latency **measured and recorded** as materially improved from the 108s baseline — **not** asserted `< 2s` |
| **β** | Frontend poll flow control: in-flight guard + backoff + jitter | `dashboard/static/redux/data.js`, `dashboard/tests/` | — | With an endpoint stubbed to exceed the poll interval, at most **one** in-flight request per endpoint exists at any time (regression test asserts it; today 36 stack) |
| **γ** | Hysteresis watchdog with storm escape → L2 + stop | `scripts/`, `dashboard/*.service`, `dashboard/*.timer` | δ | Boundary-test table B1–B6 pass against a live unit; B4 leaves exactly one born-at-L2 record and zero further restarts |
| **δ** | Bound the shutdown drain so a restart can't cost SIGKILL | `dashboard/dark-factory-dashboard.service` | — | B7: `systemctl restart` with a live polling client completes within the configured drain and the journal shows **no** `Killing process … with signal SIGKILL` |
> **Note on α's corrected signal (amended 2026-07-31, watcher-df, on Leo's sign-off of esc-3304-2).**
> Row α originally asserted "all three queries `SEARCH`" and "endpoint `< 2s`". Both are
> unreachable by the single mandated index, and the error originated here rather than in the
> decomposed task — task **3304** inherited it verbatim, and **ζ depends on α**, so it is corrected
> at the source.
> Measured by the architect on the real DB (6.97 GB, 16.4 M rows, never `ANALYZE`d) and reproduced
> offline on an identical schema: `get_operations_breakdown`'s `GROUP BY operation` is satisfiable
> in-order by walking `idx_wo_operation`, so SQLite's planner keeps that plan to avoid a temp
> B-tree and pays a per-row `created_at` test across all 16.4 M rows. It never switches to
> `idx_wo_created` — unchanged at 0 rows, at 5 000 rows, after `ANALYZE`, and with a covering
> `(created_at, operation)` variant. That is planner behaviour, not a tuning miss.
> `api_memory_graphs` (`dashboard/src/dashboard/app.py:589-598`) calls exactly
> `get_memory_timeseries` + `get_operations_breakdown` — **`get_agent_breakdown` is not on this
> endpoint's path** — so only one of the endpoint's two queries benefits, and the other measured
> ~34.5 s. An endpoint carrying an unchanged ~34.5 s query cannot return in under 2 s.
> **α is therefore necessary but not sufficient for the availability goal, and that is fine:** the
> outage mechanism was ~36 *stacked* polls saturating the single aiosqlite worker thread, and
> **β** (in-flight guard + backoff + jitter) is what bounds concurrency to one request per
> endpoint regardless of query cost. α removes one scan; β removes the stacking. Neither alone
> closes the outage. If the residual ~34.5 s latency is judged unacceptable *after* β lands, that
> is a new decision (query reshape, cache, materialised rollup, or dropping the breakdown from
> this endpoint) and needs its own row — do not silently widen α to chase it.

| **ε** | `/healthz`: whole-handler deadline, deliverable budget | `dashboard/src/dashboard/app.py`, `dashboard/tests/` | — | With a DB made unresponsive, `/healthz` returns **503 `degraded` within its stated budget (< 5s)** instead of hanging (measured today: 50.6s); `conn.execute` is covered by the deadline, not just `fetchone` |
| **ζ** | write_journal growth alarm (no pruning) | `fused-memory/`, `dashboard/` | α | When `write_journal.db` exceeds the configured size/growth threshold, a loud WARNING **and** an escalation appear naming the measured size (today 6.5 GB / 16.3M rows, ~50 MB/day) |
| **η** | Dashboard unit-file parity check (in-repo vs installed) | `scripts/`, `dashboard/tests/` | γ, δ | Script exits non-zero and names the drifting directive when the installed unit differs from the repo copy (today: `DASHBOARD_KNOWN_PROJECT_ROOTS` = 9 roots installed vs 1 in repo) |
| **θ** | Fix `No [object Object] tasks` empty state | `dashboard/static/redux/tabs.jsx` | — | With all three orchestrator filters off, the table reads a plain-English empty state; the string `[object Object]` appears nowhere in rendered output |

**Out-of-batch dependent:** existing task **3289** ("Restore
`dark-factory-dashboard-watchdog.timer`…", currently `deferred`) depends on
α, β, γ, δ, ε, ζ, η. It is the batch's terminal gate: it re-arms the timer
only after the preconditions it already enumerates are met, and carries the
`disable --now` fallback if they will not be.

**G7 walk (docs/legibility/design-invariants.md).** No unwaived hits.
γ is the direct remedy for the INV-4 violation that caused the incident (a
fail-soft restart path with no rate/streak escalation — it fired 113×/hour
silently) and moves the probe contract out of unit-file prose (INV-1).
β and ε add machine-checked constraints rather than prose. η closes an INV-5
lock-step duplication between the repo and installed unit copies. ζ is a
loud-over-silent growth counter (INV-4 shape). α is a pure index addition —
no new contract, no fail-soft path.

## Out of scope

- **Escalation-archive rollup / incremental index.** Measured 0.855s cold —
  no availability justification. Stays owned by
  `escalation-lifecycle-dashboard-prd.md`, gated on tripwires 2664/+90d/+180d.
- **write_journal row pruning + aggregate rollup.** Deferred behind ζ's
  alarm. The `prune_idempotent_ops` / `prune_mem0_intents` precedent in the
  same class makes it cheap to add later; irreversible deletion should wait
  for the alarm to justify it.
- **Bundling the SPA.** In-browser Babel over 15 `.jsx` files makes every
  cold load slow and fragile, but it is not why the dashboard was down.
  Separate concern.
- **Metrics/burndown write amplification.** `_burndown_loop` and
  `_metrics_loop` each run an immediate snapshot on startup, so at 113
  restarts/hour they wrote ~19× their design rate. This self-resolves once
  restarts stop; no task needed.
- **`_split_queue_stats` / `_lifespan_block` log spam.** Noisy at every
  startup; cosmetic, and mostly a symptom of the restart rate.

## Open questions (tactical)

1. **Index build cost at startup.** `CREATE INDEX` on 16.3M rows will hold
   the write lock and stall `fused-memory` `initialize()` once. Estimated
   tens of seconds (a full scan of the same table measured ~21s), one-time —
   `IF NOT EXISTS` makes every later start free. **Suggested resolution:**
   accept the one-time cost, but if it measures > ~2 min, pre-build
   out-of-band during a quiet window before deploying α. Decide in α.
2. **Watchdog implementation language.** `scripts/` holds both `.sh` and
   `.py`. State persistence across oneshot ticks and the `escalation submit`
   call argue for Python (`scripts/orchestrator-watchdog.py` precedent).
   **Suggested resolution:** Python. Decide in γ.
3. **Exact ζ thresholds.** Absolute size vs growth-rate. **Suggested
   resolution:** both — absolute (e.g. 10 GB) and a daily-delta ceiling
   anchored on the measured ~50 MB/day. Decide in ζ.
4. ~~**Whether η should also assert the two unit copies' `Environment=` values
   agree**, given the installed copy legitimately carries 9 project roots.~~
   **RESOLVED in η (task 3312).** Presence-and-shape everywhere would have
   been too weak to deliver the check's actual purpose — an installed
   `TimeoutStopSec=90` against a committed `15`, or a stale
   `--timeout-graceful-shutdown`, is *present but wrong*. So directives are
   split by CLASS instead, in `scripts/check_dashboard_unit_parity.py`:

   - **Value-compared** — host-INVARIANT literals carrying no paths (`Type`,
     `Restart`, `RestartSec`, `RestartMaxDelaySec`, `TimeoutStopSec`,
     `TimeoutStartSec`, `StandardOutput`/`Error`, `OnBootSec`,
     `OnUnitActiveSec`, `WantedBy`). Comparison is symmetric: a directive
     added only to the installed copy is drift too.
   - **Presence-only** — directives embedding host paths (`ExecStart`,
     `WorkingDirectory`, `Documentation`), which cannot be value-compared
     without firing on every machine that is not this one.
   - **ExecStart flags** — the uvicorn flags γ added are extracted from the
     backslash-continuation-joined logical `ExecStart` of both copies and
     compared individually, so the `uv` path and repo root are ignored while
     a stale `--timeout-keep-alive` is not. This is what makes "the unit
     change reached the running system" a checkable claim for γ and δ.
   - **`Environment=`** — compared by variable-NAME set (a dropped variable
     is always drift), with values compared only for names off a documented
     `DIVERGENCE_ALLOWLIST`. One entry today: `DASHBOARD_KNOWN_PROJECT_ROOTS`,
     carrying the committed unit's own justification verbatim. Allowlisting
     is scoped to a variable NAME, not to `Environment=` as a whole, so
     blessing the nine-root value does not also bless the variable
     disappearing.

   Expected VALUES are read from the committed unit at run time; only the KEY
   registry and the allowlist are curated. A hardcoded literal list would be
   a third site that must agree with the other two, and would silently defeat
   the purpose — a stale literal keeps passing against the OLD value on both
   sides while the repo edit again fails to reach the running system. The key
   registry is guarded against rot by tests asserting every listed key is
   really declared in the committed unit.

   Exit contract, matching `check_fused_memory_unit_parity.py` so
   `setup-host.sh` can branch on it: **0** parity / **1** drift / **2**
   installed unit absent, with drift DOMINATING absence so an unrelated
   uninstalled unit cannot mask an actionable finding. Wired into
   `setup-host.sh` as a warn-only report. No `--fix`: re-running the
   installer is the propagation path, and a `--fix` could silently re-arm a
   watchdog timer someone deliberately left disarmed.

   **Drift this surfaces on the host at landing time:** the installed
   `dark-factory-dashboard-watchdog.service` is still the pre-incident
   inline-shell copy — no `TimeoutStartSec`, no journal routing — so the
   checker exits 1 until task 3289 installs the post-δ units. That is the
   intended signal, not a defect: it is exactly the silent-propagation
   failure this check was built to expose. The deliberate 9-vs-1
   `DASHBOARD_KNOWN_PROJECT_ROOTS` divergence correctly stays silent, and
   the uvicorn flags compare clean.
