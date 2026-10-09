# RCA — curator semantic dedup silently degraded to `create` for three days (2026-08-15..18)

**Date:** 2026-10-09 (re-measurement of the 2026-08-24 filing)
**Task:** 4718 (root cause landed separately by task 4448, merge `85c6168bf7`)
**Affected:** `fused-memory` TaskCurator dedup, fleet-wide; the measured victims are in `dark_factory`

Every number below comes from a read-only query against the main checkout's
stores, which is quoted with the number and was re-run on **2026-10-09** unless
it says otherwise. Run every query from `/home/leo/src/dark-factory`, with
`sqlite3 -readonly`.

---

## 1. Verdict: root cause KNOWN

The `claude` CLI was not on the fused-memory systemd unit's inherited `PATH`.
So every curator LLM call raised `FileNotFoundError` before any CLI process
started. Each one landed in the bare `except Exception` arm of
`fused-memory/src/fused_memory/middleware/task_curator.py::TaskCurator.curate`.
At the time, that arm degraded the call to `action='create'` with justification
`llm-failed: …`. It filed no escalation, did not advance the zero-output
breaker, and wrote nothing durable.

Task 4448 established this from a journald capture:

- **209** consecutive `LLM call failed, falling through to create: [Errno 2] No such file or directory: 'claude'` lines, each paired 1:1 with a `decision=create` line.
- Per journal day (llmfail / create / combine / drop): 08-15 38/38/0/0, 08-16 68/68/0/0, 08-17 67/67/0/0, 08-18 35/54/3/4.
- Recovery came at the 2026-08-18 18:20 BST (17:20Z) restart.

**Evidence caveat (measured 2026-10-09).** The capture 4448 cites,
`/home/leo/.claude/fleet/sessions/explore-reify-597526/curator_journal_aug15-19.txt.gz`,
no longer exists: that session directory is gone. Also,
`journalctl --user -u fused-memory --since 2026-08-15 --until 2026-08-19`
returns `-- No entries --`, because the unit's journal now begins at
2026-09-15T20:49Z. So the
primary evidence survives only as the counts in task 4448's record. Its per-day
figures sum to 208, against its stated 209 total; with the journal gone, that
one-line difference cannot be reconciled. Section 2 corroborates the verdict
from durable artifacts.

## 2. Corroboration from durable artifacts

**The ticket count matches the journal exactly.** Between the first fast ticket
and the restart, tickets.db holds 209 curated tickets, all `created`, and none
`combined`:

```sql
-- data/reconciliation/tickets.db
SELECT COUNT(*), SUM(status='created'), SUM(status='combined'),
       MIN(created_at), MAX(created_at),
       ROUND(MAX((julianday(resolved_at)-julianday(created_at))*86400.0),2)
FROM tickets
WHERE created_at BETWEEN '2026-08-15T08:50' AND '2026-08-18T17:20:04'
  AND status IN ('created','combined');
-- 209 | 209 | 0 | 2026-08-15T08:50:02Z | 2026-08-18T17:20:03Z | 33.19
```

The next ticket, at 17:31:28Z, is the first `combined` (44.4s), and resolves
from then on run at 35-126s. The last ticket before the window was a healthy
63s combine at 2026-08-13T04:09:43Z, and then **no tickets at all** were filed
for 52.7h. The onset therefore lies somewhere in that gap; tickets.db cannot
place it more precisely.

**Latency rules out a CLI call.** Bucketing the 209 resolves by whole seconds
gives 182 in 0-6s, 22 in 11-13s, 3 in 21-22s and 2 in 32-33s
(`GROUP BY CAST((julianday(resolved_at)-julianday(created_at))*86400.0 AS INTEGER)`
over the same predicate). On healthy days a real curator call averages
53-357s. A ~10s quantum on top of a ~2s floor fits ticket-worker scheduling, not
model latency. Hypothesis: the quantum is the worker's wait/poll cadence; this
was not traced.

**The other create-without-LLM paths are refuted** (the arms of
`TaskCurator.curate`, plus the interceptor's no-curator path):

| Path (justification) | Verdict | Evidence |
|---|---|---|
| `zero-output-breaker-open` | refuted | It opens only after `zero_output_breaker_threshold` (2) consecutive zero-output timeouts, each a call hung for the full curator timeout, and stays open `zero_output_breaker_cooldown_seconds` = 600s. The window's maximum resolve is 33.19s, so nothing in it hung, and a breaker opened earlier cannot survive three days. |
| `all-accounts-capped` | refuted | `data/reconciliation/curator_events.db`: `SELECT COUNT(*) FROM account_events WHERE created_at >= '2026-08-14' AND created_at < '2026-08-18';` returns **0**. A capped gate writes `cap_hit` rows, and a capped ticket waits, so its latency rises rather than falls. |
| `llm-error-escalated` (`CuratorFailureError`) | refuted | That arm calls `curator_escalator.py::CuratorEscalator.report_failure`, which rewrites its 1h burst log on every generic call. On 2026-08-24 that file held a single 2026-08-08 entry, so no write happened between 08-08 and 08-24; 4448 cites the same reading independently. **Not re-runnable today:** the file is `data/reconciliation/curator_escalator_state.json` (NOT `data/escalations/`), and on 2026-10-09 it holds one reify entry dated 2026-10-07T22:37:54Z. It is a rolling log, so the 08-24 reading survives only as recorded. |
| `corpus-failed` | refuted by journal only | No durable artifact separates it. 4448's 209 `LLM call failed` lines, 1:1 with 209 creates, leave no ticket for it. |
| curator absent (`_get_curator()` → None) | refuted by journal only | That path logs no `LLM call failed` line, and the 1:1 pairing leaves no room for it. It also left no trace of its own, which this task closes (section 7). |
| `llm-failed` (bare `except Exception`) | **confirmed** | The journal (section 1). |

## 3. The measurement

Ticket outcomes per day, all projects, keyed by filing day. `combined` is the
ticket status, which covers the curator's `drop:` and `combine:` verdicts plus
the deterministic `idempotency_hit` / `candidate_key_collision`.

```sql
-- data/reconciliation/tickets.db
SELECT substr(created_at,1,10) AS day,
       SUM(status='created') AS created, SUM(status='combined') AS combined,
       ROUND(AVG((julianday(resolved_at)-julianday(created_at))*86400.0),1) AS mean_s,
       ROUND(MAX((julianday(resolved_at)-julianday(created_at))*86400.0),2) AS max_s
FROM tickets
WHERE created_at >= '2026-08-10' AND created_at < '2026-08-21'
  AND status IN ('created','combined')
GROUP BY day ORDER BY day;
```

| day | created | combined | mean s | max s |
|---|---|---|---|---|
| 08-10 | 47 | 21 | 103.2 | 309.76 |
| **08-11** | 27 | **0** | **4.6** | 14.05 |
| 08-12 | 243 | 27 | 356.7 | 5039.09 |
| 08-13 | 19 | 6 | 101.5 | 163.54 |
| 08-14 | — | — | — | — (no tickets) |
| **08-15** | 40 | **0** | **4.7** | 33.19 |
| **08-16** | 69 | **0** | **3.0** | 13.36 |
| **08-17** | 65 | **0** | **3.9** | 32.41 |
| 08-18 | 58 | 12 | 53.1 | 259.29 |
| 08-19 | 72 | 20 | 90.9 | 241.70 |
| 08-20 | 114 | 28 | 115.2 | 318.60 |

The 2026-08-24 filing's table differs by a few tickets on some days. It
detected combines with `reason/result_json LIKE '%combine%'`, which misses drops.
The signature is the same either way.

**Window.** 2026-08-15T08:50:02Z → 2026-08-18T17:20:03Z by tickets.db, which
agrees with the journal's 17:20Z restart. The plan's earlier "~12:00 on 08-18"
estimate was wrong: dark_factory tickets kept resolving in 2-5s until 17:20Z.

**The 08-11 recurrence is longer than the daily table shows.** Measured against
the first combine after it:

```sql
SELECT COUNT(*), SUM(status='combined'),
       ROUND(MAX((julianday(resolved_at)-julianday(created_at))*86400.0),1)
FROM tickets
WHERE created_at >= '2026-08-11T09:30' AND created_at < '2026-08-12T11:41:45'
  AND status IN ('created','combined');
-- 185 | 0 | 24.0
```

185 tickets ran from 2026-08-11T09:30:31Z to 2026-08-12T11:26:12Z with no
combine, and none took longer than 24.0s. 144 of them were dark_factory tickets
filed 08-12 07:00-11:00Z, which 08-12's daily mean hides. `account_events` has
no row in that span. The next one, at 2026-08-12T11:41:51Z, comes 6s after the
first combine. The cause of this recurrence is **not established**: no journal
covers it. Hypothesis: it is the same PATH defect, since it has the identical
signature, including the absent account events.

## 4. Victims

```sql
-- data/reconciliation/tickets.db
SELECT ticket_id, task_id, created_at,
       ROUND((julianday(resolved_at)-julianday(created_at))*86400.0,2),
       json_extract(candidate_json,'$.metadata.files')
FROM tickets WHERE task_id IN ('4235','4236','4239','4240') ORDER BY created_at;
-- data/orchestrator/runs.db
SELECT task_id, COUNT(*), ROUND(SUM(cost_usd),4) FROM invocations
WHERE task_id IN ('4235','4236','4239','4240') GROUP BY task_id;
```

| task | parent | ticket | filed (08-15) | resolve | files at filing | final status (task store) | runs.db spend |
|---|---|---|---|---|---|---|---|
| 4235 | 4123 | `tkt_0RSFYR2VRNZ5NQGH83QP1BV0N2` | 13:26:49Z | 2.14s | 2 files | `deferred` | $0.0000 (0 rows) |
| 4236 | 4126 | `tkt_0RSFZ0E1D5KFB3KGVXGSEK3WC4` | 13:36:23Z | 2.29s | `[]` | `cancelled` (2026-08-20) | $5.8599 (6 rows) |
| 4239 | 4123 | `tkt_0RSFZV8KYWBDRG6XA4H967N9QZ` | 14:07:07Z | 1.10s | same 2 files | `cancelled` 2026-08-28, backlog sweep: duplicate of 4235 | $0.0000 (0 rows) |
| 4240 | 4126 | `tkt_0RSG04VRHRBAD7QY0NS7J7V1JK` | 14:18:06Z | 1.73s | `[]` | `cancelled` (2026-08-24, Leo on esc-4240-5) | $35.9662 (9 rows) |

The second pair (4235/4239) is the same `canonical_labels._QUALIFIED_NODE_NAME_PATTERN`
colon-padding follow-up, filed twice from 4123. It cost **$0**: 4239 was swept
as a duplicate before either was dispatched, and 4235 is still deferred. Across
both pairs, the measured duplicate spend is the first pair's **$41.83**.

## 5. Stream-2 exclusion (independent of the outage)

4236 and 4240 were filed with `files=[]`. Both later received the bulk tagger's
`files_tagged_at` of 2026-08-17T15:43:14Z. Stream 2 of
`task_curator.py::TaskCurator._build_corpus` (the module-lock pool) skips any
task with no files, so neither could have entered the other's module pool even
with the LLM reachable. Whether Stream 3 (embeddings) would have surfaced 4236
for 4240 is **UNVERIFIED and untestable** from surviving artifacts: the
embedding corpus and scores of 2026-08-15 are not retained. This RCA assumes
neither outcome.

The second pair was different. 4235 and 4239 were filed with identical
non-empty `files`, so Stream 2 would have put 4235 in 4239's pool. That pair is
a pure outage victim.

## 6. Instrument census (the 2026-09-10 D12a ruling)

Each class is one of U = unwritten, A = written but unattributed, or Z = written
and genuinely zero.

| Instrument | Class | Measured 2026-10-09 | Disposition |
|---|---|---|---|
| `curator_events.db::invocations` | U | `SELECT COUNT(*) FROM invocations;` → **0**, ever. `TaskCurator` passed no `cost_store`. | **Revived** (this task). |
| `curator_events.db::account_events.run_id` | A | `SELECT COUNT(*), SUM(project_id IS NULL), SUM(run_id IS NULL) FROM account_events;` → 3,359 / 3,359 / 3,359, spanning 2026-05-14..2026-10-08 | **Revived**: each process mints a curator run id onto the gate. |
| `curator_events.db::account_events.project_id` | A | as above | **Retired as NULL by design** for the gate's own events: one gate serves every project. The per-call `cap_hit` rows that `shared/src/shared/cli_invoke.py::invoke_with_cap_retry` writes now carry `project_id`. |
| `burndown/metrics.db::curator_snapshots` p50/p90/p99 | written, but zeroed by a dashboard bug | `SELECT COUNT(*), SUM(p50_active_ms=0), SUM(p50_active_ms IS NULL) FROM curator_snapshots;` → 1,023 / 1,010 / 8, while raw ticket latency averages 70.7-104.6s per day over 10-02..10-09 | **Follow-up task 6540**: the cap-window merge never closes an "all capped" window. |
| `burndown/metrics.db::curator_refusal_snapshots.refused_count` | Z | 1,023 rows summing to 0; `SELECT COUNT(*) FROM tickets WHERE status='refused';` → 0 | **Kept**: a live wire for a rare deterministic event. |
| `tickets.reason` on `created` | U | `SELECT COUNT(*), SUM(reason IS NULL) FROM tickets WHERE status='created';` → 9,330 / 9,330 | **Revived** (this task). |

A related ledger gap, `PathScopeAdjudicator` calling through the curator gate
without a `cost_store`, is follow-up **task 6541**.

## 7. What was fixed, and what each fix would have shown on 2026-08-15

**Task 4448** (merged 2026-10-08):

- The catch-all arm now reports through the escalator.
- A class-agnostic per-project degraded-streak alarm, `CuratorEscalator.report_consecutive_degraded`, fires after `degraded_streak_threshold` = 5 degraded curations.
- `TaskCurator.startup_self_check` escalates an unresolvable backend binary at wiring time; `CLAUDE_BINARY` overrides the lookup.
- The unit template pins `PATH`.

On 08-15, the self-check would have escalated at service start, before any
ticket. The streak alarm would have fired at the 5th ticket, which resolved at
~10:57:51Z.

**This task:**

- **Ticket provenance.** All 209 rows would have carried
  `reason = 'create: llm-failed: FileNotFoundError: [Errno 2] No such file or directory: 'claude''`.
  The vocabulary is `task_interceptor.py::_create_reason`. A combine or drop
  that could not be executed reads `create: combine-failed: …` /
  `create: drop-failed: …`. The one path 4448's streak cannot see, a curator
  that was never constructed, now reads
  `create: curator-unavailable: closed | disabled | construction-failed: <ExcType>: <first line>`.
  Exception text enters a reason only through
  `task_curator.py::exception_summary`: the type and the message's first
  line, cut to 120 characters.
- **Invocation ledger.** `curator_events.db::invocations` would have shown
  **zero** rows, because the call raised before the CLI ran, against 209
  tickets. It is threaded `server/main.py::run_server` → `TaskInterceptor` →
  `TaskCurator`, with roles `task_curator` / `task_curator_batch`. Rows also
  appear in the dashboard cost view through
  `dashboard/src/dashboard/project_dbs.py::_cost_sources`.
- **Signature detector.** `ticket_janitor.py::TicketJanitor` checks the
  signature on each tick, through `TicketStore.dedup_health` and
  `is_dedup_outage`. The signature is at least 10 resolved tickets in a 6h
  window, 0 `combined`, and a median raw resolve ≤ 15s (config
  `curator.janitor.dedup_outage`). When it holds, the janitor files a
  **blocking** `infra_issue` whose detail lists `top_create_reasons`. The 10th
  window ticket resolved at 2026-08-15T12:56:49Z, so the escalation would have
  queued by ~12:58Z, 29 minutes before the first victim ticket at 13:26:49Z.
  Whether anyone would have acted in time is unknowable.

**Detector back-test.** The back-test below runs the production predicate,
`ticket_janitor.py::is_dedup_outage`, with the shipped defaults. It scores every
resolved `created`/`combined` ticket: 11,597 tickets in 9 projects, from
2026-04-24 to 2026-10-09. Each project is evaluated at every hour boundary over
the preceding 6h:

| project | firing hours (UTC day: count) |
|---|---|
| dark_factory | 08-11: 5, 08-12: 5, 08-15: 11, 08-16: 24, 08-17: 24, 08-18: 4 (73 total) |
| reify | 05-01: 5, 08-11: 4, 08-18: 7 |
| know_live | 05-01: 3 |

There are **0 firing hours anywhere from 2026-08-19 to 2026-10-09.** Every
dark_factory hour falls inside the two outage windows of section 3. The last
one, 08-12 13:00Z, falls before the first post-recurrence combine resolved, at
13:02:25Z. reify's
08-18 hours run past the restart, because the 6h window still holds pre-restart
tickets; the per-window rate limit keeps that to one escalation. The
**2026-05-01** reify/know_live window (16:00-20:00Z) is **unexplained**: it is
not verified as either an outage or a false positive.

<details><summary>Back-test script (read-only; run from a worktree with <code>uv run --project fused-memory python -I backtest.py data/reconciliation/tickets.db</code>)</summary>

```python
import sqlite3, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, UTC
from fused_memory.config.schema import DedupOutageDetectorConfig
from fused_memory.middleware.ticket_janitor import is_dedup_outage
from fused_memory.middleware.ticket_store import DedupHealth

cfg = DedupOutageDetectorConfig()
con = sqlite3.connect(f'file:{sys.argv[1]}?mode=ro', uri=True)
by_pid = defaultdict(list)
for pid, status, reason, c, r in con.execute(
        "SELECT project_id, status, reason, created_at, resolved_at FROM tickets "
        "WHERE resolved_at IS NOT NULL AND status IN ('created','combined')"):
    by_pid[pid].append((datetime.fromisoformat(r), datetime.fromisoformat(c), status, reason))
first = min(x[0] for v in by_pid.values() for x in v)
last = max(x[0] for v in by_pid.values() for x in v)
window, fires = timedelta(seconds=cfg.window_seconds), defaultdict(list)
t = first.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
while t <= last + timedelta(hours=1):
    for pid, items in by_pid.items():
        sel = [x for x in items if t - window <= x[0] < t]
        if sel and is_dedup_outage(DedupHealth(
                resolved=len(sel),
                combined=sum(x[2] == 'combined' for x in sel),
                median_resolve_seconds=statistics.median((r - c).total_seconds() for r, c, _, _ in sel),
                top_create_reasons=()), cfg):
            fires[pid].append(t)
    t += timedelta(hours=1)
for pid, ts in sorted(fires.items()):
    print(pid, len(ts), dict(sorted(Counter(x.date().isoformat() for x in ts).items())))
print('after 08-19:', sum(x >= datetime(2026, 8, 19, tzinfo=UTC) for ts in fires.values() for x in ts))
```

</details>

## 8. What was not fixed

- **`suggestion_hash` shape validation.** The exact-match R4 gate could not
  bridge `4126-dangling-test-class-reference-guard` (4236) and a 64-hex sha256
  (4240). That is sibling **task 4719**, which is independent of this outage.
- **Applying triage.py's SKIP rules on the agent-followup path** (the steward's
  proposal, esc-4240-4) was refuted, so it is not done:
  - SKIP rule 1 requires a remedy that is purely prose editing, and both tasks
    proposed new code.
  - Rule 2 requires strengthening an existing check, and none existed.
  - The TRIAGE role's only consumer is gated to `review_suggestions`
    escalations above a suggestion threshold.

## 9. Residual risk

- **A raising invocation still writes no ledger row.** A missing binary or
  all-accounts-capped raises before `invoke_with_cap_retry` reaches its write,
  so the ledger's silence is itself the signal. The tickets.db signature
  detector is the backstop for that case.
- **Any `combined` counts as health.** That includes the deterministic
  `idempotency_hit` (118 ever) and `candidate_key_collision` (26 ever). A
  window holding one such combine among LLM-less creates would not fire.
- **Low-volume projects are invisible to the detector.** A project filing fewer
  than 10 tickets per 6h never trips it: reify filed 5 tickets over 08-15..08-17
  and never fired. 4448's 5-curation streak is the guard there.
- **The detector stays silent during an all-accounts-capped period, by
  design.** Capped tickets wait, so their wall-clock latency rises.
- **The detector is not wired while `curator.enabled` is false.** A
  deliberately disabled curator produces the signature itself, so wiring it
  would page every window (`server/main.py::_dedup_outage_detector_config`).
  A curator disabled by mistake therefore raises no alarm here; its tickets
  still read `create: curator-unavailable: disabled`.
- **The ledger exists only while `usage_cap.enabled`.** Without it,
  `server/main.py::_setup_curator_usage_gate` opens no CostStore, so an empty
  `invocations` table says nothing about the curator.
- **Both alarms' state resets on restart.** The detector's rate limit and
  4448's streak live in process memory. The detector re-derives its evidence
  from tickets.db on each tick, so a restart costs at most one duplicate
  escalation; the streak restarts its count.
