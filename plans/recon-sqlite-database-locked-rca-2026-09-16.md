# RCA: reconciliation runs dying on SQLite "database is locked" (2026-09-16)

Read-only investigation, run as an agent team (see §9). Brief:
`~/.claude/spawn-briefs/2026-09-16-recon-sqlite-database-locked.md`. The evidence and reproduction scripts are
in the session scratchpad `/tmp/claude-1000/-home-leo-src-dark-factory/4b8838b3-e315-4ecf-8c3a-bc8ee8d8ca0f/scratchpad/`
(abbreviated `SP/` below). The scratchpad is session-local, so every figure a decision rests on is restated here.

## 0. The whole chain in one paragraph

The locked file is `data/reconciliation/reconciliation.db`. The failing connection is
`ReconciliationJournal._db`, a single aiosqlite connection shared by every coroutine in the fused-memory
server. A journal write fails **immediately**, in milliseconds, **not after the 5 s busy_timeout**, when three things coincide:

1. Another coroutine's read on that same connection is between hops. `async with db.execute(SELECT…) as cur:
   await cur.fetchall()` is several queued worker-thread calls, and SQLite holds a read snapshot open from the
   execute hop until the last row is fetched.
2. The write is queued into that gap.
3. A *different* connection to the same file commits or holds the write lock inside the window. That connection
   is EventBuffer or ReconLedgerStore, both in the same process.

SQLite never invokes the busy handler for a connection that already holds a read transaction. It returns
`SQLITE_BUSY_SNAPSHOT` or `SQLITE_BUSY` at once, so raising `busy_timeout` cannot help.

**By what:** the dominant pinning reader is the stale-run reaper,
`harness.py::ReconciliationHarness._recover_stale_runs` → `journal.py::ReconciliationJournal.get_stale_runs`,
every ~5 s. It pins only while some run is `running` past `stale_run_recovery_seconds`, and then for about the
60 ms its execute hop takes on the live-size table. All 16 lock deaths occurred inside such a window. The
kind-matched base rate gives P ≈ 7e-8, and the evidence peaks exactly at the configured 1800 s cutoff.

**Why it dies:** no layer retries a journal write. The bookkeeping writes (`record_run_session`,
`clear_run_session`, the post-completion `update_run_stage_reports`) are fatal to the run. Nothing re-triggers
the run explicitly; the normal triggers simply fire again because the watermark never advanced.

## 1. Population — corrected

- **7 lock-killed runs since 08-24, not 6.** Run `0a623163` (solar_challenge_platform, full, 08-26) died on the
  lock error, but 0.8 s later the stale-run reaper overwrote its row with a `StaleRunRecovery` `_error`. The
  brief's `stage_reports LIKE '%database is locked%'` query therefore misses it. **16 all-time.**
- **The rate is ~0.7 %, not 0.23 %.** Only full and remediation runs can die this way: 7 of ~960 full/remediation
  completions from 08-24 to 09-16. The 2,774 denominator is dominated by targeted runs.
- **Most lock errors kill no run.** 08-16→09-16 fused-memory logged **142 lock lines in 85 incidents** (60 s clusters):
  - `Project loop error` ×73 (≈6 of them re-log a run death)
  - `Failed to record run_action` ×17 (audit rows lost)
  - EventQueue drain transient ×12 (retried)
  - `Reconciliation loop error` ×11
  - `Heartbeat failed` ×8
  - `Drained event cleanup failed` ×5
  - background/targeted ×6
  - judge ×3
  - `update_run_stage_reports failed … (second-cancellation write path)` ×1 (run a2dcd4bd, 09-15)

  A further 228 syslog lines come from the orchestrator's own `event_store.db` and are a separate subsystem.
- **Three things hide deaths from the runs table:**
  - the reaper overwriting a row it has just read (0a623163 → task 5546);
  - `_run_remediation_pass` storing no traceback (4 rows);
  - `_escalate` returning silently once the escalation queue is torn down at shutdown (F4 filed no escalation).
- The failure-path seat's estimate that the 12 remediation-row `StaleRunRecovery` records were hidden lock deaths is
  **refuted**. Those rows are cancellation orphans, already owned by **task 5545**, and no lock line sits near any
  of them except 0a623163.

## 2. Per-failure table (the 7 since 08-24, plus today's candidate)

Times UTC. The write-duration column is the log-measured bound from the event that precedes the failing write
to the failure line; microsecond syslog prefixes.

| # | run · project · type | died | DB file / connection | failing site | write took | holder | evidence strength |
|---|---|---|---|---|---|---|---|
| F1 | 96dad36a · reify · full | 08-24 17:05:07.59 | reconciliation.db / journal | `stages/base.py::BaseStage.run` finally → `journal.py::clear_run_session`, after the TKS agent exited 17:05:04.26 | ≤ 3.34 s (bound) | reaper pin; the dying run itself was eligible (2,154 s > 1,800) | site **confirmed** (traceback) · immediacy **bounded** · holder **inferred** |
| F2 | bd68c5cd · reify · full | 08-27 12:46:13.77 | reconciliation.db / journal | `BaseStage.run` → `clear_run_session` | **0.147 s** | reaper pin; self-eligible (1,998 s) | site + timing **confirmed** · holder **inferred** |
| F2b | 0a623163 · solar_challenge_platform · full | 08-26 10:38:20.67 | reconciliation.db / journal | journal write at the integrity_check boundary; traceback destroyed by the reaper overwrite | **1.38 s** | reaper: 3 eligible runs incl. self. The reaper's read **demonstrably spanned the failure**: 0.8 s later it rewrote the row from a snapshot taken before the run terminalised it | timing **confirmed** · holder **strong** · site unknown |
| F3 | c814a6d0 · dark_factory · remediation | 08-28 14:57:24.10 | reconciliation.db / journal | inside `_run_remediation_pass` at task_knowledge_sync; no traceback stored | **0.041 s** | reaper pin; other project's run reify 96caca2f at 2,103 s | timing **confirmed** · site + holder **inferred** |
| F4 | 22e63e5a · dark_factory · full | 08-28 20:12:13.98 | reconciliation.db / journal | `BaseStage.run` → `record_run_session` before the TKS launch | **≤ 0.16 s** | reaper pin; reify 151a0ef8 at 3,849 s. The write came right after a **67 s event-loop stall** during SIGTERM shutdown | timing **confirmed** · holder **inferred**, confounded by shutdown |
| F5 | a33dea89 · solar_challenge · full | 09-10 20:16:33.16 | reconciliation.db / journal | `harness.py::run_full_cycle` → `update_run_stage_reports`, AFTER `update_watermark` and `complete_run('completed')` had landed | **0.49 s** | reaper pin; dark_factory 51988a83 at 3,082 s plus self at 2,218 s. 91 ms earlier `checkpoint recon_journal failed: database table is locked` shows the journal had a statement in flight. SIGTERM came 1.8 s earlier | timing **confirmed** · in-flight journal statement **sampled** · holder identity **inferred** |
| F6 | a4309c6a · solar_challenge · full | 09-14 15:07:28.69 | reconciliation.db / journal | `BaseStage.run` → `record_run_session` for the integrity_check launch | **0.90 s** | reaper pin; reify 5284adbf at 1,872 s. 4 EventBuffer commits in the same second; steady state, no shutdown | timing **confirmed** · holder **inferred** |
| today | none; not a run death | 09-16 11:06:57 | reconciliation.db / **EventBuffer** | `Drained event cleanup failed` and `Project loop error` ×3, all within 7 ms | n/a | a backlog of writers released after a **2 m 55 s event-loop wedge** (11:04:01→11:06:56, watchdog "wedged" ×3) | **not a member** of the run-death class: a consequence of the wedge (owners: task 3778, watchdog) |

The pre-08-24 deaths behave the same way. 11 of the 15 run rows carry tracebacks, and every one fails in `journal.py`.
Two of those deaths cannot be bounded below 5 s: a5a7f78d (≤ 6.3 s) and 6bed633b (≤ 187 s).

## 3. Class-level root causes

| # | Claim | Label |
|---|---|---|
| RC1 | **Locked file and connection.** reconciliation.db via `ReconciliationJournal._db`: one aiosqlite connection shared by every coroutine; legacy isolation, WAL, busy_timeout 5000. Not write_journal / event_journal / tickets / tasks.db / FalkorDB / Qdrant. | **confirmed**. All 11 stored tracebacks fail in `journal.py` `_txn` writes. |
| RC2 | **Immediate BUSY from cross-coroutine snapshot pinning.** See §0. Reproduced three independent ways with the production connection helpers: (a) the lead's minimal cases in `SP/lead/aio_shapes.py`; even a PK lookup pins between its execute and fetchone hops. (b) The skeptic's `SP/skeptic/decisive.py` with the **unmodified** `get_stale_runs` + `record_run_session` on a 24k-row table: 25/40, 26/40 and 34/40 writes fail with `SQLITE_BUSY_SNAPSHOT` within 5 ms when one run is past the cutoff; 0/40 when none is. (c) Stress: multi-hop reads fail 544–558 of 600 writes; single-hop `execute_fetchall` fails 0/600. | **confirmed** |
| RC3 | **Why 5 s wasn't enough: it was never used.** SQLite's btree calls the busy handler only when the connection has no open transaction. A connection already holding a read snapshot gets `SQLITE_BUSY_SNAPSHOT` (another connection committed) or `SQLITE_BUSY` (another connection holds the write lock) at once. All 7 recent deaths are bounded under 5 s, most well under 1 s; true lock-hold reproductions never fail before 5.013 s. | **confirmed** |
| RC4 | **By what: the stale-run reaper is the dominant pinning reader.** `get_stale_runs` returns no rows, and pins nothing, unless a run is past the cutoff. Otherwise its execute hop (median ~60 ms on a live-size copy) stops at the first match and the snapshot stays pinned until fetchall, every ~5 s. Evidence: (i) 16/16 deaths inside eligible windows; kind-matched base rate 13–31 % after 05-28, Poisson-binomial **P ≈ 6.8e-8** (recent 7 alone ≈ 1.6e-5). (ii) 4/4 deaths whose own run was not eligible still had another eligible run in flight. (iii) Out of sample, the non-fatal journal-write failures (`Failed to record run_action`) fall in eligible windows in 14/16 incidents against a ~23–29 % base, p ≈ 8e-8. (iv) Sweeping a hypothetical cutoff over 32 journal-connection incidents, the likelihood ratio **peaks exactly at the configured 1,800 s** (LR 33.5; 22.4 at 1,200 s; 6.9 at 3,600 s); that parameter is read only by the reaper. (v) F2b: the reaper's read verifiably spanned the failure. Secondary readers: `ReconciliationJournal.checkpoint()` (two hops, cursor never closed, every 300 s), `get_recent_runs`, and judge reads. 3 of 32 incidents had no eligible run. | **confirmed by inference** for the class. The reader is never directly logged per failure, so the per-failure holder stays "inferred". |
| RC5 | **Required co-factor: an in-process second connection writing the same file.** EventBuffer: push plus burst_state per event (median 62 events/h, max 738/h), per-project heartbeats, and lock-table txns from every idle project loop's `should_trigger` every 5 s. ReconLedgerStore: stage-end upserts and gc. No out-of-process writer (the dashboard opens read-only). | **confirmed** (code plus `lsof`) |
| RC6 | **Why the run dies.** `journal.py::ReconciliationJournal._txn` rolls back and re-raises. `BaseStage.run` calls `record_run_session` (pre-launch) and `clear_run_session` (in `finally`, so it also masks the stage's own exception) with no handling. `run_full_cycle`'s `except Exception` marks the run failed, restores drained events, escalates `recon_failure` and re-raises. `_run_remediation_pass` swallows the error with no traceback. The only retry in the subsystem is `event_queue.py::EventQueue._commit_with_retry`, and it covers only event pushes; that is what the WP-B docstring's "locked reconciliation.db" fix actually covered. | **confirmed** |
| RC7 | **Why nothing re-triggers.** No retry path reads "last run failed". `update_watermark` runs only on success, so max_staleness, quiescent and buffer_size fire again; the next run started 0.1–55 min later. It did not always redo the work. F5's and F6's immediate successors were themselves interrupted by restarts, so the next *completed* full run came 4 h and 8 h later. The failed remediation c814a6d0 was never retried. F5 was worse than lost: its watermark had advanced and its drained events were restored, so they are **processed twice**, and its judge review and remediation were skipped. | **confirmed** |
| RC8 | **Amplifiers.** Event-loop stalls and shutdowns stretch every in-flight pin and release queued writers in a burst: F4 (67 s stall), F5 (SIGTERM), today (2 m 55 s wedge). The stalls are owned by **task 3778**, in progress: sync git probes in recon stage payload assembly. | **plausible**, confirmed as present at F4/F5 |
| RC9 | **Adjacent defect with the same root: silent transaction loss on shared connections.** Coroutine X's `_txn` UPDATE is in flight when coroutine Y's `_txn` fails and `_safe_rollback` rolls back the *connection*. X's commit then succeeds with no error and its write is gone (`SP/skeptic/alt.py` A7). Any cancellation or `wait_for` timeout inside a `_txn` can discard other coroutines' writes on the journal or EventBuffer connections. | **confirmed** (reproduced); production frequency unmeasured |
| RC10 | **H1: a writer or a TRUNCATE checkpoint holding the write lock > 5 s.** | **refuted** for 14/16 deaths (bounded < 5 s); undetermined for a5a7f78d and 6bed633b. The 300 s checkpoint loop takes part only as a secondary pinning reader (journal) or as a write-lock holder that a pinned write collides with immediately (other connections). |
| RC11 | **Another DB file**, or FalkorDB/Qdrant surfacing a SQLite error. | **refuted** |

Measured non-fixes. **Raising busy_timeout** does nothing for this class because the handler is never invoked. **`BEGIN IMMEDIATE`**
still raises `SQLITE_BUSY_SNAPSHOT`, in 0.2 ms: the pin belongs to the same connection. A **separate write
connection** leaves reads on the shared connection blind to its commits while any read is pinned (it returned `{}`
until the pin released), and turns immediate failures into 5 s stalls behind TRUNCATE checkpoints
(`SP/skeptic/fixes2.py` F1, `ckrace.py`).

Prediction, untested. `stale_run_recovery_seconds` was raised 1800 → 3600 today (cc191354dd) for unrelated reasons.
In-sample, 11 of 32 incidents would have been eligible at 3,600 s instead of 29, so expect **fewer** deaths
without the mechanism being fixed. Rows stuck in `running` for hours keep the reaper eligible at any cutoff. Do not
read a quiet period as a fix.

## 4. Decision briefings

> **RULED 2026-09-17 (Leo): D1 = B · D2 = B, plus C at low priority · D3 = B.** Filed as a committed batch:
> - **5560** (high): the shared atomic-access primitive, adopted by the journal, EventBuffer and ReconLedgerStore.
> - **5561** (medium, depends on 5560): bookkeeping writes become best-effort, with no completed→failed flip.
> - **5562** (low, depends on 5560): adopt the primitive in the task backend, tickets, write journal, durable
>   queue and planned-episode registry.
>
> Cockpit decision `recon-sqlite-database-is-locked-class-2026-09-08` was already `answered`.

### D1 — How to remove the pinned-snapshot collision

**Issue.** Store methods on a shared aiosqlite connection are not atomic units. A multi-hop read can pin a stale
snapshot under another coroutine's write, causing RC2 and 7 run deaths in 23 days. A failed write's rollback can
silently undo another coroutine's write (RC9).

**Decision.** Which remedy to adopt, or whether to accept the rate. This is also the answer to cockpit decision
`recon-sqlite-database-is-locked-class-2026-09-08`: its premise "busy_timeout=5000 already" is irrelevant to this class.

**Options.**
- **(A) Single-hop reads.** Every read runs execute and fetch in one worker call (`execute_fetchall`), including
  `checkpoint()`; async cursor iteration is banned. This closes the window rather than shrinking it: 0/600 under
  stress. It does not fix RC9, and it must cover *every* reader: converting only the reaper still left 85/600
  failures. Future readers must keep the shape, so it needs a structural guard.
- **(B) A plus atomic write units.** Each store routes all access through two primitives: `_read` (single hop) and a
  write primitive that holds a per-connection `asyncio.Lock` across execute and commit. `_read` takes the same lock
  so it never sees another coroutine's uncommitted writes. Fixes RC2 and RC9.
- **(C) Bounded retry.** It must wrap whole methods, since `_txn` is an asynccontextmanager that can't re-run its
  body, yield between attempts, and outlast pins of ≥ 2 s (a 1.55 s budget failed). It masks the symptom, keeps
  RC9, adds tail latency and can hide a wedge.
- **(D) Separate write connection.** Evidence-rejected: stale reads, and 5 s stalls behind checkpoints.
- **(E) Accept.** Keep dismissing the per-run escalations.

**Ramifications.**
- Each death costs a cycle, redone minutes to hours later. F5-type deaths also skip judge and remediation and
  double-process events.
- The non-fatal tail runs to about 85 incidents a month: lost `run_action` audit rows, heartbeat failures, judge
  failures, and project-loop errors that delay triggers.
- RC9's silent loss has unknown frequency.
- The recurring operator cost is five dismissed escalations, a cockpit decision and this RCA.
- B's cost is contained: a change to three store classes' access primitives, migrating about 40 methods in
  `journal.py` plus the `event_buffer.py` and `recon_ledger.py` call sites. No schema or config change, and it is
  testable at the store interface.
- B's pitfall: `asyncio.Lock` is not re-entrant. A primitive called from inside a write unit deadlocks; existing
  nesting such as `get_run_actions_combined` → `get_run_actions` must be restructured. The deadlock is loud in tests,
  not silent.
- A is smaller but leaves RC9.

**Recommendation: B.** It also enforces heuristic 10 (clear invariants uniformly enforced) at one place per
store rather than by call-site convention (heuristic 9, deep modules). Close the cockpit decision as "fix via D1-B".
C is optional defence in depth only after B, for the residual ≥ 5 s cases during wedges.

### D2 — Scope of the fix

**Issue.** The same multi-hop, shared-connection shape exists on the EventBuffer and ReconLedgerStore connections.
EventBuffer's own lock errors (73 project-loop lines, 8 heartbeats, 5 drained-cleanup lines) are *not* keyed to
reaper windows, so a journal-only fix won't reach them. Other persistent stores (tasks.db backend, write_journal,
tickets, durable_queue) share the shape. Their only observed symptom is `checkpoint … failed: database table is
locked` (SQLITE_LOCKED from a pending statement), and they have no observed lock deaths.

**Decision.** Which connections get D1's remedy now.

**Options.**
- **(A)** The journal only: the run-death path.
- **(B)** All three reconciliation.db connections (journal, EventBuffer, ReconLedgerStore), via one shared
  primitive, e.g. in `shared/src/shared/async_sqlite_base.py`.
- **(C)** B plus every persistent aiosqlite store in fused-memory, pre-emptively.

**Ramifications.** A fixes the deaths but leaves most of the non-fatal incidents and RC9 on EventBuffer. B covers
every connection to the file where the collision is observed, at about 3× A's migration surface. C spends effort
where nothing has failed and widens review blast radius.

**Recommendation: B**, built as a shared primitive so that C later becomes adoption rather than redesign, and only
if a store starts showing lock errors.

### D3 — Should journal bookkeeping writes be able to kill or flip a run?

**Issue.** Session snapshots are fatal today, although their own docstring calls them "best-effort" and the resume
gate already validates them: `record_run_session` runs before launch and `clear_run_session` in a `finally`, where
it masks the stage's real exception. A trailing `update_run_stage_reports` failure after `complete_run('completed')`
flips a finished run to `failed` (F5), which skips judge and remediation and re-queues events that were already
processed. Even after D1, ≥ 5 s lock holds remain possible during event-loop wedges and shutdowns.

**Decision.** Keep these writes fatal, or degrade.

**Options.**
- **(A)** Keep them fatal and rely on D1.
- **(B)** Make `record_run_session`/`clear_run_session` best-effort (log with errorname, continue; never mask the
  stage's exception). Treat a post-completion `update_run_stage_reports` failure as a logged degradation that
  neither flips the status nor restores drained events.
- **(C)** B plus D1-C's bounded retry on those writes.

**Ramifications.** A keeps a deterministic "journal state = run state" contract, at the cost of whole-cycle loss on
any residual lock. B changes run-status semantics: a `completed` run may carry incomplete stage_reports, and
`_finding_persistence_count` and the judge read those reports, so they must tolerate the gap. C adds latency on
every stage transition.

**Recommendation: B.** It is independent of D1, removes the fatal amplifier for the wedge and shutdown cases, and
fixes F5's double-processing. Sequence it after D1 so the two don't collide in `harness.py`.

## 5. What this RCA corrects in its own premises

- "6 failed runs": 7 since 08-24; the LIKE query misses reaper-overwritten rows.
- "~0.2 % failure rate": ~0.7 % of the runs that can die this way.
- "busy_timeout=5000": never consulted for this class.
- "may not be reconciliation.db": it is.
- "WP-B fixed the locked reconciliation.db": it fixed event pushes only; journal writes were never covered.
- "none after 09-14T16:15": not significant at this rate (148 runs), and the 3,600 s cutoff will suppress the rate further without a fix.

## 6. Existing owners consulted

- **5545** (pending): remediation child rows left `running` on cancellation. It owns the 12 remediation `StaleRunRecovery` rows.
- **3778** (in progress): event-loop stalls from sync git probes, amplifier RC8.
  This RCA's observability task was combined into **5545** (§8).
- **2711** (done): instance-aware stale recovery, the context for the reaper change filed below.

No existing task owned the shared-connection atomicity defect (D1/D2). It was filed as 5560–5562 once Leo ruled on 2026-09-17.

## 7. Caveats and what is not verified

- The pinning reader is never logged per failure. RC4 is a class-level inference; the per-failure holder column
  says "inferred" except F2b.
- Reproduction stress rates are ~1000× production. The mechanism and the fixes are verified; the production
  frequency is estimated from the measured ~60 ms window every 5 s, not reproduced.
- RC9's production frequency is unmeasured.
- a5a7f78d and 6bed633b (May and August) can't be bounded below 5 s, and 6bed633b predates syslog retention.
- F3 and F2b have no failing call site (no traceback).

## 8. Tasks filed

Both were submitted through `submit_task` by agent_id `claude-team-recon-sqlite-rca`.

1. **Task 5546** (created, medium, pending): the stale-run reaper overwrites a run its own coroutine already
   terminalised. The fix makes `_recover_one_run`'s terminalisation conditional on `status='running'` and skips
   the rest of recovery when 0 rows change. Evidence: 0a623163's own `complete_run('failed')` and escalation landed
   0.8 s before the reaper replaced its `_error` and all its stage reports with `StaleRunRecovery`.
2. **Task 5545** (combined by the curator into the existing task, now medium, pending): recon run failure records
   drop lock-classification evidence. The fix adds a traceback on `_run_remediation_pass`'s except path,
   `sqlite_errorname`/`sqlite_errorcode` in **both** drivers' `_error`, and a WARNING when `_escalate` drops a
   submission because the queue is gone (F4). The combine narrowed the scope to the remediation path; an appended
   amendment restores the `run_full_cycle` half and the verify steps. The fix also gives D1 a direct before/after
   measure.

## 9. Team

| Seat | Model / effort | Tokens · min | Outcome |
|---|---|---|---|
| logs | sonnet / high | 260k · 53 | Per-failure log windows with microsecond wait bounds; census; F2b and F4 shutdown/stall context. Good. |
| conn-map | opus / high | 293k · 48 | Connection inventory; named the reaper and its eligibility condition; first correlation. The key finding. |
| failure-path | sonnet / high | 264k · 37 | Exception and no-retry paths, next-run table, timestamp semantics. One wrong estimate: 12 hidden remediation deaths, refuted via 5545. |
| repro | sonnet / high | 154k · 35 | H1 timings correct. **Wrongly "ruled out" H2**: its competing UPDATE was a no-op and its `in_transaction` argument is invalid. The lead caught and re-ran it. Routing lesson: subtle-mechanism experiment design wanted opus, or a mandatory control. |
| skeptic | opus / high | 261k · 56 | Refuted none of C1–C10; weakened 4 (counts, phase statistics, fix caveats); added RC9, the checkpoint pin and fix pitfalls. |
| lead (inline) | opus | — | Scouting, the sub-second-gap discriminator, H2 reproduction and fix matrix, correlation, census clustering, synthesis. |

No fable seat was used or proposed; the problem was a concrete concurrency RCA with checkable evidence.
