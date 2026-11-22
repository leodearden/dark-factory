# Startup hardening: no startup step crashes the process, the merge lane starts after startup, and a crash loop pages within minutes

**Status:** active PRD, authored 2026-11-02 (`/prd`, fable session prd-df-806015, from the brief Leo approved
on 2026-11-02 "4: yes" after the restart-loops follow-up). Bare B: one chain in one file plus one host script.
**Code anchors** cite symbols; verified against main `922d9d3baa` (2026-11-02). Main moves fast — re-locate at
implementation time.
**Evidence:** `plans/portfolio-review-2026-11-01/followups/restart-loops.md` (§2 the loop, §6 options 1–5) and
`followups/restart-loops-seats/S3.md` (the per-step fatality table); `seats/Q5.md` (527 starts in 30 d: 334
deploys, 193 the loop, 0 repairs).

## 1. Goal

A startup fault of the 2026-10-26 shape — a deterministic exception from a startup repair sweep on durable
on-disk state — costs at most three process iterations instead of 193, costs the merge lane nothing instead of
one verify kill and N "Merge worker shutting down" blocks per iteration, and pages exactly one L2 naming the
failing startup step within minutes instead of never. A crash loop the process cannot record (SIGKILL, import
error, broken venv) is caught by a second, independent detector in the host watchdog. Every hosted project
gets this at once: the chain lives in `harness.py`, the detector in the one watchdog, and the unit sizing in
the nine committed `orchestrator-*.service` files.

### What an operator observes when this lands

- Inject a raise into `_recover_crashed_tasks` on three consecutive starts against one data dir: the three
  starts log `startup step 2c failed` and complete startup; one L1 `orchestrator-startup-2c` sits on the
  queue with `dedupe_count` 2; `runs` rows carry `startup_completed_at`. Zero `merge_finalized state=blocked`
  events; no verify subprocess started.
- Inject a process-killing fault (a `BaseException`, or a raise before the escalation queue exists) three
  times: the fourth start skips that step, comes up, and files one born-at-L2 critical
  `orchestrator-startup-crash-loop` naming the step and the exception; the three crashed `runs` rows read
  `exit_reason = startup_crash:<step>`.
- `scripts/orchestrator-watchdog.py --report` shows a `RESTARTS` column per unit (NRestarts, Result); when a
  unit's NRestarts grows by 5 within 30 min, one born-at-L2 critical `orchestrator-crash-loop-<unit>` appears
  in dark-factory's queue, and a unit in `Result=start-limit-hit` stays down (no `reset-failed`) with that
  same page open.
- `systemctl --user cat orchestrator-reify.service` reads `StartLimitIntervalSec=1800`, and the unit comment
  states the cycle it actually bounds.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| startup-step runner (α) and its `startup_step_failed` event + L1 per step | the L2 watcher (`skills/escalation-watcher`) through the queue; the crash-loop breaker (γ) reads its step names and outcomes; forensic reads of runs.db |
| merge loop launched after `finish_startup()` (β) | the merge lane itself (no shutdown blocks on a crashed start); task 7745's lane-reset class closes by construction |
| consecutive-startup-failure journal + quarantine + born-at-L2 (γ) | the L2 watcher; the next start of the same process (reads the journal); `runs.exit_reason` / `startup_completed_at` read by the OPERATIONS.md §Troubleshooting row this PRD adds and by η |
| watchdog restart-rate alarm, start-limit-hit honour, liveness min-interval, `--report` column (δ) | the L2 watcher via dark-factory's queue; the operator reading `--report`; systemd's StartLimit (δ stops undoing it) |
| `StartLimitIntervalSec=1800` (ε) | systemd on the host, once ζ installs the units; δ, which pages on the state ε now lets systemd reach |
| deploy gate (ζ) and observation (η) | the fleet operator; η is the INV-13 proof that the producers above wrote in production |

## 2. Background

- `harness.py::Harness.run` runs eleven hand-written `try/except` blocks around startup steps and none around
  2c `_recover_crashed_tasks`, 2d `_reconcile_stranded_in_progress`, 2e `_reap_orphan_worktrees` and Diff 7
  `_reconcile_terminal_lanes` (S3 §Q1). 2d states the opposite policy in its per-task handler: "Other
  (unexpected) exception types intentionally propagate so bugs in the sweep surface". Task 4053 wrapped 2c2
  only, after it looped reify for 5 h 12 m on 2026-10-26. Heuristics 10 (invariants uniformly enforced) and
  11 (SPOT) are broken by construction: the invariant "a startup repair sweep must never prevent startup" is
  restated per site and absent at four.
- The merge worker starts at 1b (`_build_lifecycle_registry`: loop-stall-monitor → escalation-server →
  merge-worker → …), `reap_merge_verify_survivors` runs inside `_start_merge_worker`, and
  `_recover_pending_merges` (1c0a) re-enqueues journaled requests into `self._merge_queue`, an
  `asyncio.Queue`, which the worker's `run()` consumes concurrently with every later step. A crash at 2c or
  later therefore kills the previous iteration's verify and resolves every queued request with
  `MERGE_WORKER_SHUTDOWN_REASON` via `SpeculativeMergeWorker.stop()` — 579 `merge_finalized` blocked events
  and ~190 verify kills on 10-26 (follow-up §1). `_stop_merge_worker` skips `stop()` when
  `_merge_worker_task is None`.
- Nothing records a startup crash: `Harness.run`'s `finally` stamps `completed_at` and `finish_run` writes
  `total_tasks` (0 before `finish_startup`); `runs` has no exit-reason column; `cli.py::run` catches only
  `CancelledError`. No event, no escalation, no counter (S3 §Q2).
- systemd: `Restart=on-failure RestartSec=10 RestartSteps=4 RestartMaxDelaySec=60 StartLimitIntervalSec=600
  StartLimitBurst=10`. The backoff reaches 60 s by the fifth restart, so ten cycles need more than 600 s
  whatever the crash time; the limit trips only for a crash within ~15 s of a fresh start (S3 §Q3a). The unit
  comment "blocks a true crash loop" is false for the case it names.
- `scripts/orchestrator-watchdog.py`: `main()` skips a unit started under `STARTUP_GRACE_SECS` (120 s), so a
  ~100 s cycle is never probed (295 skips on 10-26); it reads neither `NRestarts` nor `Result`; `restart_unit`
  runs `reset-failed`, which clears start-limit-hit, so a tripped limit is a 2–3 min pause; the orchestrator
  liveness tier has no revive min-interval (fused-memory's has `FM_LIVENESS_RESTART_MIN_INTERVAL_SECS`); the
  script has no escalation-filing path. `scripts/dashboard-watchdog.py::file_ceiling_escalation` is the
  precedent for a stdlib-only watchdog filing a born-at-L2 through `escalation submit` into
  dark-factory's `data/escalations`.
- Related open tasks: 7701 (per-lane guard in 2c2; folded into α), 7745 (recovery releases a `_spec` lane the
  worker holds; folded into β), 7957 (refused-WIP lane stays pinned with only a WARNING; stays separate),
  7277 (2d re-pends a task whose merge request was recovered; stays separate, rebases over α), 5700 (watchdog
  restart_unit log wording; stays separate), 7247 (shutdown drain bound; stays separate).

## 3. Resolved design decisions

1. **One startup-step runner, non-fatal by default, fatal by explicit opt-in.** New module
   `orchestrator/src/orchestrator/startup_steps.py` with `StartupStep(name, run, fatal=False)` and a runner
   `run_startup_step(step, *, journal, event_store, escalation_queue) -> StepOutcome`. A step that raises
   `Exception` yields `StepOutcome(status='failed', exception_type, message)` — the caller sees the failure in
   the value (INV-11), the runner logs ERROR with `exc_info`, emits `EventType.startup_step_failed` with
   `{step, exception_type, message}` (INV-2), and files or refreshes one L1 keyed on the sentinel task id
   `orchestrator-startup-<step>` (dedup by `get_by_task(..., status='pending', level=1)`, refresh in place
   with `dedupe_count += 1`, the `_file_dirty_tree_escalation` idiom). A `fatal=True` step re-raises; the
   only member is the merge-liveness config check (`MergeLivenessConfigError`), whose fatality is already
   ratified in `_start_merge_worker`. `BaseException` is never caught by the runner; γ catches it one level
   out. The eleven ad-hoc `try/except Exception` blocks in `Harness.run` startup are replaced by runner calls;
   `Harness.run` names the steps in one ordered table (heuristics 4, 9, 11).
2. **Per-entry isolation inside the three sweeps that iterate durable state.** `_recover_crashed_tasks` (per
   worktree entry), `_reap_orphan_worktrees` (per worktree), and `_reconcile_stranded_in_progress` (per task:
   the existing `except SetTaskStatusRejected` widens to `except Exception`, keeping the rejected-write
   branch's escalation) continue past one bad entry, log it with the entry name, and add it to the step's
   `StepOutcome.failed_entries`; a step with failed entries but no raise is `status='partial'` and files the
   same L1 naming the entries. 7701's per-lane guard for `_reconcile_lane_checkouts` is this rule applied to
   2c2 and lands in α.
3. **2d is no longer fatal at startup; mid-run is untouched.** The ruling the brief left open. Reasoning: the
   stated policy ("bugs surface") was written before a sweep crash could loop a unit for five hours unseen;
   surfacing is now the L1 + event + journal, which is louder than a traceback in a journal nobody paged.
   Skipping 2d on one start is safe: the mid-run sweep (`_reconcile_stranded_in_progress(mid_run=True)` on
   the empty-queue branch) runs the same reconcile, and a stranded in-progress task meanwhile is held, not
   lost. The mid-run call keeps its propagate policy — the main loop's exposure is S3's row "3" and out of
   scope here. Leo may overrule in-thread; if he keeps 2d fatal, α lists it `fatal=True` and γ's quarantine is
   what bounds it.
4. **The merge worker is constructed at 1b and its loop launched after `finish_startup()`.** `ManagedService
   'merge-worker'` keeps its slot and name; its `start_fn` now constructs `SpeculativeMergeWorker`, the three
   restart coordinators and the liveness checks only. A new `Harness._launch_merge_worker_loop()` runs
   `reap_merge_verify_survivors` then `create_task(worker.run())`, called immediately after
   `scheduler.finish_startup()` and before the first `acquire_next`. Premise verified in code: recovery
   enqueues into `self._merge_queue` (an `asyncio.Queue` created in `__init__`) and registers in
   `_merge_inflight_registry`; neither needs the consumer loop. **Invariant, stated for β and pinned by a
   test:** `_stop_merge_worker` MUST NOT call `worker.stop()` when `_merge_worker_task is None` — a crash
   between construction and launch must resolve nothing, so no `merge_finalized state=blocked` is emitted and
   the journal re-recovers the same requests next start (C1 §A5). Under `dry_run` the loop is never launched.
   7745 closes by construction: no spec lane is acquired before 2c runs. The 1b docstring contract becomes
   "the recovery block needs the escalation server live and the worker constructed".
5. **In-process crash-loop breaker: a durable consecutive-failure journal, K = 3, quarantine, one L2.** File
   `data/orchestrator/startup_failures.json` beside `runs.db` (resolved from `runs_db(project_root)`'s
   parent), pydantic-modelled (INV-1): `{step, exception_type, message, consecutive, first_seen, last_seen}`.
   Written synchronously by the runner on every `failed` outcome and by one `except BaseException` around the
   whole startup block in `Harness.run` (step name = the step in flight, recorded by the runner before each
   step starts; a `CancelledError` from SIGTERM is NOT a failure and writes nothing). `consecutive` increments
   when the same step fails on the next start and resets to 1 on a different step. On the K-th consecutive
   failure of a step the next start marks it quarantined: the runner skips it, logs WARNING, and files one
   born-at-L2 critical on sentinel `orchestrator-startup-crash-loop` (category `infra_issue`, summary names
   the step and exception, detail carries the journal; refreshed in place if pending). `finish_startup()`
   clears the journal; the L2 stays as the record and refreshes with `dedupe_count` on the next episode.
   Failures before the escalation queue exists (`mcp.start`, `start_all`, the escalation bind) are journaled
   and the page is filed by the first later start that reaches the queue (the `_file_restored_pause_escalation`
   deferred-filing shape). The quarantine covers only non-fatal steps; a `fatal=True` step at K files the L2
   and still raises, because starting without it is unsafe by its own ratification.
6. **A crashed start is distinguishable in runs.db.** Two additive columns via `add_column_if_missing_sync`
   (the `_migrate_runs_code_sha` idiom): `startup_completed_at TEXT` (stamped at `finish_startup()`) and
   `exit_reason TEXT` (`startup_crash:<step>` from the `except BaseException`, `cancelled` for a SIGTERM
   `CancelledError`, NULL for a clean return). `finish_run` writes both. OPERATIONS.md §Troubleshooting gains
   the row "an orchestrator unit restarts every ~100 s" whose recipe reads these columns and the journal.
7. **The watchdog gets an independent detector and stops undoing StartLimit.** `main()` reads `systemctl
   --user show -p NRestarts,Result` for every WATCHED unit on every tick, grace window or not, and persists
   `{unit: [(epoch, n_restarts)]}` samples in `data/orchestrator/watchdog-restart-rate.json` (the
   `FM_LIVENESS_STREAK_PATH` atomic-write idiom). When NRestarts grows by ≥ `ORCH_RESTART_RATE_ALARM_COUNT`
   (5) within `ORCH_RESTART_RATE_ALARM_WINDOW_SECS` (1800), or `Result=start-limit-hit`, it files one
   born-at-L2 critical into **dark-factory's queue** via `escalation submit` (sentinel
   `orchestrator-crash-loop-<unit>`, category `infra_issue`, agent role `orchestrator-watchdog`), at most one
   per window per unit (the dashboard-watchdog dedup rule, persisted epoch). On `start-limit-hit` the liveness
   pass does NOT `reset-failed` or restart. A liveness revive of an orchestrator unit is capped at one per
   `ORCH_LIVENESS_RESTART_MIN_INTERVAL_SECS` (default 900) per unit, mirroring the fm cap; a wedged unit is
   revived no later than that. `--report` gains a `RESTARTS` column (`NRestarts/Result`). NRestarts counts only
   `Restart=` restarts (task 5739 constraint 6), which is exactly the crash-loop signal; watchdog and sweep
   restarts flush it, so the sample list, not a bare delta against the last tick, is what the window reads.
8. **The alarm's home is dark-factory's queue as the host's own.** The brief's open question. A new host
   sink would have no consumer (G1): no watcher reads it, no dashboard lists it. The dashboard watchdog already
   files host-level born-at-L2s into `data/escalations` of this repo, the df L2 watcher reads that queue, and
   the crash-looping project's own queue is served by the very process that is looping. The record names the
   unit in its sentinel and summary so the watcher routes by hand.
9. **StartLimit sized to the real cycle.** `StartLimitIntervalSec=1800`, `StartLimitBurst=10` on the nine
   committed `scripts/orchestrator-*.service` files and `skills/factory-init/references/supervised-unit.md`;
   the comment states the bound (ten starts at the 60 s backoff fit under 1800 s for a crash up to ~120 s into
   startup). Cap-rotation churn and sweeps are one start each per unit, far under ten per 30 min. Restart-tier:
   takes effect when ζ installs the files. Without δ it would be cosmetic (the watchdog resets it); with δ it is
   the backstop that fires when the process cannot record its own crash.
10. **Deploy is one human gate.** Installed units are patched by copying the committed files over the installed
    copies and `daemon-reload` (`scripts/check_orchestrator_unit_parity.py` is full symmetric equality for these
    units, and the reify warm-lane drop-in is a separate file that the copy does not touch), then one drained
    sweep. A `systemctl restart` by the sweep against a unit sitting in `start-limit-hit` fails and the sweep
    reports it; that is the intended trade.

## 4. Out of scope

- The main loop's own fatality (S3 row "3"), the mid-run 2d policy, and `cli.py`'s exception handling beyond
  passing the exit reason through.
- The drain covering queued entries, the `merge_finalized blocked` signal for a kept request (follow-up
  option 6), reify's >90 s shutdown (option 7, task 7247), the fm tier's lease (5719), the coordinator's debt
  (4763 et al.), and the 11-04 eligibility-clock check (option 8).
- A fused-memory restart-rate alarm; fm keeps its own streak-gated tier.
- Dashboard rendering of the new `runs` columns.

## 5. Cross-PRD relationship and seams (G4)

| other PRD / task | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/harness-supervision-prd.md` (LifecycleRegistry, task 2241) | this PRD consumes | `ManagedService 'merge-worker'` start/stop; `test_harness_lifecycle_registry.py::_CANONICAL_ORDER` | this PRD (β restates the 1b contract; the order list is unchanged) | wired |
| watchdog liveness tier (OPERATIONS.md §8; tasks 4754/4755 lease) | this PRD changes | `main()` revive path; `_live_fleet_lease` current_unit skip | this PRD (δ keeps the lease skip and adds the min-interval and start-limit-hit arms) | wired |
| drain + fleet redeploy clock (5371, 7297, 7172) | untouched | `restart-all-orchestrators.sh` restarts; 72 h clock | other | wired — a sweep meeting a start-limit-hit unit reports failure (decision 10) |
| task 7277 (2d re-pends a recovered-merge task) | adjacent | `_revert_in_progress_if_no_live_claimant` inside 2d | 7277 | pending; rebases over α's per-task isolation, no conflict in semantics |
| task 7957 (refused-WIP lane WARNING only) | adjacent | `_reconcile_lane_checkouts` detach-refused branch | 7957 | pending; α's per-lane guard does not file 7957's L1 |
| task 7247 (shutdown drain bound) | adjacent | `Harness.run` finally ladder step 4a | 7247 | pending; γ's `except BaseException` sits above the ladder and does not touch it |
| `plans/cancel-with-children-prd.md` | this PRD consumes | cancelling 7701 and 7745 as superseded | n/a (neither has children) | decompose cancels them with `superseded_by` |
| `scripts/dashboard-watchdog.py` (dashboard-availability PRD) | this PRD imitates | `file_ceiling_escalation` argv shape | other | wired — δ copies the shape, not the code (INV-5: the sentinel-role and severity rules are enforced by `escalation submit` itself) |

## 6. Decomposition plan

Bare B, one DAG. Labels are the sidecar's; task ids are stamped at decompose.

| label | title | modules | observable signal (G2) | prereqs |
|---|---|---|---|---|
| α | Startup-step runner: every startup sweep non-fatal by default, per-entry isolation, one L1 per failed step | `orchestrator/src/orchestrator/startup_steps.py` (new), `harness.py` (run startup block, 2c/2d/2e/2c2 loops), `event_store.py` (`startup_step_failed`), `orchestrator/tests/test_harness_startup_steps.py` (new) | fault-injection test: an injected raise in each of 2c, 2c2, 2d, 2e, Diff 7 leaves `finish_startup()` called, one `startup_step_failed` event with the step name, one pending L1 `orchestrator-startup-<step>`; a second run refreshes it (`dedupe_count` 1) instead of filing twice; one bad worktree in 2c and one bad task in 2d leave the other entries processed. The eleven ad-hoc blocks are gone (`git grep -c 'Failed to recover pending merges' harness.py` drops to 0). Folds 7701 | — |
| β | Merge worker constructed at 1b, loop and survivor reap launched after `finish_startup()` | `harness.py` (`_start_merge_worker`, new `_launch_merge_worker_loop`, `_stop_merge_worker`), `orchestrator/tests/test_harness_lifecycle_registry.py`, `orchestrator/tests/test_recovery_spec_lane_hold.py` (new, from 7745's test) | fault-injection test: a journaled merge request plus a raise in 2c → zero `merge_finalized` events with `MERGE_WORKER_SHUTDOWN_REASON`, no verify started, `worker.stop()` not called; a healthy start logs `Speculative merge worker started` after `Task tree populated`; 7745's test: a spec lane acquired by the worker is never released by recovery. Folds 7745 | α |
| γ | Crash-loop breaker: consecutive-startup-failure journal, K=3 quarantine, one born-at-L2, `exit_reason`/`startup_completed_at` on runs | `startup_steps.py`, `harness.py` (run `except BaseException`, journal clear at `finish_startup`), `run_store.py` (migration + `finish_run`), `OPERATIONS.md` §Troubleshooting, `orchestrator/tests/test_startup_crash_loop_breaker.py` (new) | fault-injection test against one tmp data dir: three starts raising `SystemExit` from 2e → three `runs` rows `exit_reason='startup_crash:2e'`, journal `consecutive=3`; the fourth start skips 2e, completes (`startup_completed_at` set), files exactly one pending L2 `orchestrator-startup-crash-loop` whose summary names `2e` and `SystemExit`, and clears the journal; a fifth healthy start files nothing. A SIGTERM `CancelledError` mid-startup writes `exit_reason='cancelled'` and no journal entry. This is the PRD's integration gate for the in-process chain | α, β |
| δ | Watchdog: NRestarts restart-rate alarm into dark-factory's queue, honour start-limit-hit, per-unit liveness min-interval, `--report RESTARTS` | `scripts/orchestrator-watchdog.py`, `tests/scripts/test_orchestrator_watchdog.py`, `OPERATIONS.md` §8 | tests with a fake `systemctl show`: NRestarts samples 0→5 inside 1800 s file one `escalation submit` argv (sentinel `orchestrator-crash-loop-<unit>`, severity critical, category infra_issue, agent role prefixed `orchestrator-`) into `data/escalations`, a second tick in the window files nothing; `Result=start-limit-hit` → no `reset-failed`, no restart, one page; two dead probes 10 min apart → one revive; `--report` prints `RESTARTS`. Live: `--report` on this host shows the column | — |
| ε | StartLimitIntervalSec=1800 on the nine committed orchestrator units, the factory-init template and the unit-file tests | `scripts/orchestrator-*.service` (9), `skills/factory-init/references/supervised-unit.md`, `tests/scripts/test_orchestrator_service_files.py`, `tests/scripts/test_systemd_restart_backoff.py` if it pins the value | `git grep -c StartLimitIntervalSec=600 -- scripts/orchestrator-*.service` is 0 and `=1800` is 9; the tests pin 1800; `scripts/check_orchestrator_unit_parity.py` reports the nine installed units as drifted until ζ. `complexity=simple` | — |
| ζ | Deploy: install the resized units, daemon-reload, drained fleet sweep (human gate) | none (host) | `check_orchestrator_unit_parity.py` clean; `systemctl --user show -p StartLimitIntervalUSec orchestrator-reify.service` = 30min; every unit restarted onto a main containing α–δ; `--report` shows `RESTARTS`. `task_kind=deterministic`, `always_escalates=True`, `execution_class=operational` | α, β, γ, δ, ε |
| η | Observation 7 days after ζ: the producers wrote in production (INV-13) | `plans/startup-hardening-prd.md` §9 (appended record) | for every `runs` row started after ζ: `startup_completed_at` non-NULL or `exit_reason` non-NULL; `data/orchestrator/watchdog-restart-rate.json` carries ≥ 7 days of samples for all WATCHED units; no `orchestrator-crash-loop-*` or `orchestrator-startup-crash-loop` record filed falsely (each one that exists is matched to a journal `Scheduled restart job` burst); the record is appended to §9 and committed. `metadata.milestone = {mode: delayed, after_secs: 604800}` | ζ |

G7 walk (advisory here, re-walked at decompose): INV-4 — the runner is a fail-soft path; its storm escape is
the K counter (γ) plus the L1's `dedupe_count` and the watchdog's window (δ). INV-11 — `StepOutcome` carries
the failure to `Harness.run`; `exit_reason` carries it to runs.db. INV-2 — step, exception type and message
are structured in the event and the journal. INV-7 — the quarantine hold is owned by the open L2 and bounded
by `finish_startup()` clearing the journal. INV-9 — the step list has one home (the runner table in
`Harness.run`); the sidecar and this doc cite it, never copy it. INV-10 — every leaf's guard executes a fault
against the real startup block. INV-13 — η. INV-12 — the `fatal=True` list is the ratified-exception list,
with one member and its ratification cited. INV-5 — δ reuses `escalation submit`'s enforced boundary rather
than re-implementing the sentinel rules. No waivers.

## 7. Pre-conditions

None: no novel substrate (G3 N/A beyond what §2 verified — `asyncio.Queue` enqueue without a consumer,
`add_column_if_missing_sync`, `escalation submit`, `systemctl show -p NRestarts,Result`, the
`EventType` StrEnum, the queue's `get_by_task`/`make_id`/`submit`).

## 8. Open questions (tactical)

1. **K and the alarm thresholds.** K=3, 5 restarts / 30 min, revive min-interval 900 s are the follow-up's
   numbers; make each a module constant with an env override in the watchdog and a config leaf in the harness
   only if a second project needs a different value. Decide during γ/δ.
2. **Where the `RESTARTS` column sits in `--report`.** Beside `VERDICT`; decide during δ.
3. **Whether the journal also records `partial` outcomes.** Suggested: no — a partial step did not crash and
   its L1 already names the entries. Decide during γ.

## 9. Observation record (η appends here)

### 2026-11-22 — η observation (task 8274)

**Window:** 2026-11-15T10:24:00Z to 2026-11-22T11:19:11Z, 7 d 0 h 55 m (7.04 days). The start is when ζ's
drained sweep began: the esc-8273-1 resolution reads "restart-all-orchestrators.sh --drain ran 10:24-11:00Z".
The end is the first measurement. Each item below states its own measurement time.

**Context:** the live stores in the MAIN checkouts (`/home/leo/src/dark-factory/data/…`,
`/home/leo/src/reify/data/…`), read read-only (sqlite `file:…?mode=ro` URIs from system `python3`) from a
dispatched agent shell (task 8274's worktree) on host leo-MS-7C35. The journal was read with
`journalctl --user` and live unit state with `systemctl --user show`.

| item | rule | figure | verdict |
|---|---|---|---|
| 1. `runs` rows started in the window | every row has `startup_completed_at` or `exit_reason` non-NULL; a both-NULL row is a γ defect | dark-factory: 6 rows, 4 `cancelled`, 2 both-NULL. reify: 6 rows, 4 `cancelled`, 2 both-NULL. All 4 both-NULL rows had completed startup: 2 are live, 2 were SIGKILLed after startup. 0 `startup_crash:*` | **FAIL**: γ (task 8269) persists `startup_completed_at` only at exit. Follow-up: _pending in this commit_ |
| 2. watchdog restart-rate sampling, `--report RESTARTS` | every WATCHED unit has an entry with `sampling_since` ≤ window start and a last sample ≤ ~3 min old; the watchdog timer ran across the whole window with no gap far above its 60 s tick; `--report` prints `RESTARTS` for every running unit. (δ keeps only the last 1800 s of samples by design, so §6's "≥ 7 days of samples" is read as `sampling_since` plus timer continuity.) | 7/7 WATCHED units present, each with `sampling_since` 2026-11-11T17:55:04Z and a last sample 2 s old (30 samples, all NRestarts 0). 10 095 timer ticks in the window, median gap 60.2 s, largest 120.4 s (one missed tick). `RESTARTS` printed for 9/9 running units, all `0/success`. Not sampled: scoping-intake and stream-touch, which are not in WATCHED | **PASS**. The WATCHED gap is owned by pending tasks 9321 and 9815 |
| 3. crash-loop pages | each in-window `orchestrator-crash-loop-*` or `orchestrator-startup-crash-loop` record matches a journal `Scheduled restart job` burst (≥ 5 within 1800 s, or `start-limit-hit`; ≥ 3 consecutive startup crashes for γ's page). Conversely, every such burst has a record | 0 records in the window across the 9 hosted queues. 0 `Scheduled restart job` lines and 0 start-limit hits on all 9 units. 0 `startup_crash:*` rows in all 9 runs.db | **no episode, detector unexercised**: no false page, no missed page |
| 4. startup-step L1s | list each in-window `orchestrator-startup-<step>` L1 with its `dedupe_count`, and say whether it names a real fault | 0 L1s in the window across the 9 queues. 0 `startup_step_failed` events in the window in all 9 runs.db | **no step failure in the window, runner's failure path unexercised** |
| 5. merge-lane cost per restart | per restart, at most one `MERGE_WORKER_SHUTDOWN_REASON` burst, with events = distinct request_ids; no burst from a `startup_crash:*` run and none per crash iteration; every burst attributable to one sweep or one deliberate restart | 6 fleet-sweep waves in the window, ζ's included. dark-factory: 4 bursts, 57 / 52 / 66 / 33 events. reify: 5 bursts, 2 / 3 / 6 / 4 / 6 events. Every burst lasts < 1 s and has events = distinct request_ids. 0 `startup_crash:*` runs. 3 run ends were SIGKILLed and emitted no burst | **PASS** |

**Item 1 evidence** (measured 2026-11-22T11:19–11:26Z). The query was `SELECT run_id, started_at, completed_at,
startup_completed_at, exit_reason, code_sha FROM runs WHERE started_at >= '2026-11-15T10:24:00' ORDER BY
started_at`, run on each project's `data/orchestrator/runs.db`.

- dark-factory, 6 rows. `exit_reason='cancelled'`: 4 (run-fc741c6ac05d, run-7c108b742770, run-9a6d5c55b0a4,
  run-c1a1cfef15e7). Each has `startup_completed_at` 24–83 s after `started_at`, and each ended in a systemd
  stop with `status=130`. NULL: 2, both with NULL `startup_completed_at`. No other value appears. Across the
  whole table only those 4 rows carry `startup_completed_at`.
- reify, 6 rows. `exit_reason='cancelled'`: 4 (run-579bc7a3e1c9, run-bd31446a3fb5, run-3fc27174c232,
  run-d774a0682070), with `startup_completed_at` 35–90 s after `started_at`, each ending in a `status=130`
  stop. NULL: 2, both with NULL `startup_completed_at`. Across the whole table only those 4 rows carry it.
- run-64d39f3b97db (dark-factory, started 2026-11-15T11:00:28Z: ζ's own restart, unit MainPID 3122636) is
  class (ii), KILLED AFTER STARTUP. Orchestrator pid 3122652 logged `Started run run-64d39f3b97db` at
  11:00:28, then `Task tree populated: 3382 pending tasks` and `Speculative merge worker started` at 11:01:13.
  On 2026-11-17 the unit logged `Stopping` at 07:12:12, then `State 'stop-sigterm' timed out. Killing.` and
  `status=9/KILL` at 07:13:42. The next line, `Started run run-fc741c6ac05d`, is at 07:13:49.
- run-be51184228f6 (dark-factory, started 2026-11-21T01:38:14Z) is class (i), LIVE. The unit has been active
  since 01:33:38 with MainPID 4058288. Its orchestrator child, pid 4058311, is the process that logged
  `Started run run-be51184228f6`, then `Task tree populated: 3989 pending tasks` and `Speculative merge worker
  started` at 01:41:01. The unit has logged no `Stopping` or `Started` line since.
- run-7dfa13be7c9b (reify, started 2026-11-19T03:24:53Z) is class (ii), KILLED AFTER STARTUP. Pid 2452643
  logged `Task tree populated: 2583 pending tasks` at 03:25:18 and `Speculative merge worker started` at
  03:25:19. On 2026-11-21 the unit logged `Stopping` at 01:27:02, then `State 'stop-sigterm' timed out.
  Killing.` and `status=9/KILL` at 01:28:32. The next line, `Started run run-de3958d3e6bf`, is at 01:28:43.
- run-de3958d3e6bf (reify, started 2026-11-21T01:28:43Z) is class (i), LIVE. The unit has been active since
  01:28:33 with MainPID 3888978. Its child, pid 3888996, logged `Task tree populated: 2708 pending tasks` at
  01:29:55 and `Speculative merge worker started` at 01:29:57.
- Class (iii) (no startup-complete line before a kill): none.
- Excluded, because it started before the window: dark-factory run-0dac71f2fe0f (started 2026-11-14). ζ's
  sweep SIGKILLed it at 2026-11-15T11:00:19Z (stop-sigterm timeout), and it has NULL `completed_at`.
- Mechanism. `orchestrator/src/orchestrator/harness.py::Harness.run` assigns `startup_completed_at` to a local
  right after `self.scheduler.finish_startup()`. It hands that local to
  `orchestrator/src/orchestrator/run_store.py::RunStore.finish_run` only from its `finally`, whereas §3.6 above
  says "stamped at `finish_startup()`". So a row reads both-NULL for as long as its run is live, and forever
  once a SIGKILL skips the `finally`. That is exactly the row a SIGKILL during startup leaves. Until the process
  exits through its `finally`, the column cannot tell §3.6's crashed start from a healthy one.
- The two class (ii) kills are systemd's 90 s stop-sigterm bound: `Stopping` came 90 s before `Killing` both
  times. So the shutdown drain overran, which is task 7247 and out of scope under §4.
- The same shape appears across the fleet, from the runs.db of all 9 hosted projects (measured 2026-11-22T11:31Z
  for the items 3–4 scan). Each project has 6 rows in the window and 0 `startup_crash:*`. Seven projects have 1
  both-NULL row, the live run. reify and dark-factory have 2, as above. solar-challenge-platform also has 2: one
  is live (run-37f056969250) and one, run-6d9d7fe5ccbb (started 2026-11-19T03:25:58Z), was SIGKILLed at
  2026-11-21T01:31:11Z by the same stop-sigterm timeout, about 46 h after it started.

**Item 2 evidence** (measured 2026-11-22T11:27–11:28Z, main checkout at `8436bb9b60`).

- `data/orchestrator/watchdog-restart-rate.json`. Its key set equals the 7 units in
  `scripts/orchestrator-watchdog.py::WATCHED`, as read from the main checkout's file. The watchdog unit sets no
  `ORCH_RESTART_RATE_*` override, so `scripts/_watchdog_crash_loop.py`'s defaults apply:
  `RESTART_RATE_ALARM_COUNT` 5 and `RESTART_RATE_ALARM_WINDOW_SECS` 1800.

  | unit | `sampling_since` | samples | first / last sample | NRestarts | `last_escalation_epoch` | `start_limit_paged` |
  |---|---|---|---|---|---|---|
  | each of the 7 WATCHED units (dark-factory, reify, my-solar-challenge, know-live, autopilot-video, solar-challenge-platform, pump-web-ui) | 2026-11-11T17:55:04Z | 30 | 2026-11-22T10:58:23Z / 11:27:30Z | 0 in every sample | none | false |

- Why the samples span only 29 min. `scripts/_watchdog_crash_loop.py` keeps a sample only while `now - epoch <=
  window_secs`, so `samples` is pruned to the alarm window, while `sampling_since` "is never pruned". So δ
  cannot persist seven days of samples; that is a restatement of the rule, not a defect. The value of
  `sampling_since` (four days before the window opened) together with the timer's continuity is what shows
  the whole window was sampled.
- Timer continuity. `journalctl --user USER_UNIT=orchestrator-watchdog.service --since '2026-11-15 10:24'`
  shows 10 095 `Starting` and 10 096 `Finished` lines, from 2026-11-15T10:24:21Z to 2026-11-22T11:27:30Z, and
  no failed run (the only other lines are 33 `Consumed … CPU time`). The timer is
  `OnUnitActiveSec=60` with `AccuracySec=5s`. The median gap between starts is 60.2 s, the largest are 120.4 s
  (2026-11-19T05:48:23Z) and 120.2 s (2026-11-21T02:58:10Z), and only 3 gaps exceed 90 s.
- `python3 /home/leo/src/dark-factory/scripts/orchestrator-watchdog.py --report` exited 1. Per its docstring,
  `report()` "return[s] 1 iff any unit is stale": all 9 units were `stale` against a main commit from
  2026-11-22T08:17Z, inside the 72 h deploy cadence, so the exit is not a failure. The `RESTARTS` column sits
  beside `VERDICT` and reads `0/success` for all 9 running units: autopilot-video, dark-factory, know-live,
  my-solar-challenge, pump-web-ui, reify, scoping-intake, solar-challenge-platform, stream-touch.
- orchestrator-scoping-intake.service and orchestrator-stream-touch.service appear in `--report` but not in
  WATCHED, so they get no crash-loop sampling. Pending tasks 9321 (add both to WATCHED) and 9815 (S24 δ2a:
  derive the liveness set from installed sockets) already own that gap, so nothing new is filed for it.

**Items 3–4 evidence** (measured 2026-11-22T11:30–11:33Z).

- Queues scanned. The roots come from the `--config <root>/dark-factory-orchestrator.yaml` of every installed
  `~/.config/systemd/user/orchestrator-*.service` except the watchdog: autopilot-video, dark-factory, know-live,
  solar-challenge (unit my-solar-challenge), pump-web-ui, reify, scoping-intake, solar-challenge-platform,
  stream-touch. In each `<root>/data/escalations` the scan read every top-level `esc-*.json` and every
  `archive/<date>/esc-*.json`: 8 764 files in all, none unreadable. It selected records whose `task_id` starts
  with `orchestrator-crash-loop-` or `orchestrator-startup-` (the latter covers `orchestrator-startup-crash-loop`).
- In-window records with `timestamp` ≥ 2026-11-15T10:24:00Z: none, in any queue.
- Excluded as out of window: 8 records, all `orchestrator-startup-2c3-streak-alarms` (an L1 `-1` and an L2 `-2`),
  filed 2026-11-10T20:57Z–2026-11-11T10:48Z.
  - In dark-factory both are archived 2026-11-11 as `dismissed`. That L2 was resolved by task 9322, which made
    both `_alarm_subject` helpers tolerate `task_id=None`; it is done and merged as `0a2ff8bb91`.
  - In autopilot-video, know-live and pump-web-ui both are still `pending`, with `dedupe_count` 0 and no
    `updated_at`.
  - Each of those units restarted 6 times in the window. A recurrence would have refreshed the pending L1 in
    place (§3.1), so `dedupe_count` 0 is in-window evidence that the 2c3 fault did not recur. The six pending
    records are stale, not live faults.
- Journal converse check (`journalctl --user USER_UNIT=orchestrator-<unit>.service --since '2026-11-15 10:24'`,
  all 9 units). 0 `Scheduled restart job` lines and 0 start-limit lines on every unit. Each unit has 6 `Started`
  lines, and every one follows a `Stopping` line, so each start was a stop/start: a fleet sweep or a manual
  restart, never a `Restart=` restart. That agrees with NRestarts 0 throughout item 2. No burst reaches the
  5-in-1800 s threshold, so no page was missed.
- runs.db corroboration, all 9 projects: `SELECT run_id, timestamp, data FROM events WHERE
  event_type='startup_step_failed' AND timestamp >= '2026-11-15T10:24:00'` returns 0 rows everywhere. The
  all-time counts are 1 each in autopilot-video, dark-factory, know-live and pump-web-ui, all on 2026-11-10 (the
  2c3 fault above), and 0 elsewhere.

**Item 5 evidence** (measured 2026-11-22T11:34–11:45Z).

- The query, on both runs.db: `SELECT run_id, timestamp, task_id, data FROM events WHERE
  event_type='merge_finalized' AND timestamp >= '2026-11-15T10:24:00' AND json_extract(data,'$.reason') =
  'Merge worker shutting down'`. The literal is
  `orchestrator/src/orchestrator/merge_lane/intake_buffers.py::MERGE_WORKER_SHUTDOWN_REASON`. Events are
  bucketed by event timestamp, so reify's pre-window run ac72adcd8509, which ζ's sweep stopped, is counted.
  This gives 208 events in dark-factory and 21 in reify, all `state='blocked'`.
- Restart waves. Each wave stopped all 9 units once, in the same unit order (autopilot-video, know-live,
  my-solar-challenge, pump-web-ui, reify, scoping-intake, solar-challenge-platform, stream-touch,
  dark-factory), and every unit has exactly those 6 `Stopping` lines in the window. In the table, "sweep"
  means `scripts/restart-all-orchestrators.sh`. The wave identification is:
  - W1, 2026-11-15T10:24–11:00Z: ζ's `--drain` sweep (esc-8273-1).
  - W2, 11-17T05:51–07:13Z: a sweep, journal identifier `restart-all-orchestrators.sh[1819989]`, which ended
    "All 9 orchestrator unit(s) restarted and verified fresh".
  - W3, 11-17T19:40–20:48Z and W4, 11-18T17:30–18:34Z: the same order and pacing, but the driver's own lines
    are not in the journal under the identifiers searched (`restart-all-orchestrators.sh`, `sh`, `bash`). So
    the driver is unidentified, though the shape is a fleet sweep.
  - W5, 11-19T02:28–04:25Z: a sweep, `sh[3752634]`, which ended "All 9 … verified fresh".
  - W6, 11-21T00:16–01:33Z: unit `orch-redeploy-restart-10802` ran `restart-all-orchestrators.sh --drain`
    (`sh[4182317]`). It is the sweep that `data/orchestrator/last_redeploy_orchestrator.json` records
    (2026-11-21T01:33:38Z).

  | project | run_id | burst at (UTC) | events | distinct requests | cause |
  |---|---|---|---|---|---|
  | reify | run-ac72adcd8509 | 2026-11-15 10:37:25.16–.19 | 2 | 2 | W1, ζ's sweep |
  | dark-factory | run-0dac71f2fe0f | none | 0 | 0 | W1, ζ's sweep; SIGKILLed at 11:00:19 (stop-sigterm) |
  | reify | run-579bc7a3e1c9 | 2026-11-17 06:48:24.81–.87 | 3 | 3 | W2 sweep |
  | dark-factory | run-64d39f3b97db | none | 0 | 0 | W2 sweep; SIGKILLed at 07:13:42 (stop-sigterm) |
  | reify | run-bd31446a3fb5 | 2026-11-17 20:14:46.67–.80 | 6 | 6 | W3, fleet sweep (driver not journaled) |
  | dark-factory | run-fc741c6ac05d | 2026-11-17 20:48:47.48–.99 | 57 | 57 | W3, fleet sweep (driver not journaled) |
  | reify | run-3fc27174c232 | 2026-11-18 18:17:31.11–.13 | 4 | 4 | W4, fleet sweep (driver not journaled) |
  | dark-factory | run-7c108b742770 | 2026-11-18 18:34:20.87–21.08 | 52 | 52 (49 tasks) | W4, fleet sweep (driver not journaled) |
  | reify | run-d774a0682070 | 2026-11-19 03:24:46.54–.58 | 6 | 6 | W5 sweep |
  | dark-factory | run-9a6d5c55b0a4 | 2026-11-19 04:25:34.69–35.21 | 66 | 66 | W5 sweep |
  | reify | run-7dfa13be7c9b | none | 0 | 0 | W6 `--drain` sweep; SIGKILLed at 01:28:32 (stop-sigterm) |
  | dark-factory | run-c1a1cfef15e7 | 2026-11-21 01:33:19.61–20.39 | 33 | 33 | W6 `--drain` sweep |

- Per restart: at most one burst, and none spans more than 0.8 s. No event count exceeds its distinct
  `request_id` count, so no request was resolved twice. In run-7c108b742770, 3 tasks each had two distinct
  requests in the queue.
- Per run: every burst comes from a run with `exit_reason='cancelled'`, or, for ac72adcd8509, from a run
  that started before the window. No run in the window has `exit_reason` `startup_crash:*`, so there is no
  crash-iteration burst.
- Cost per sweep, as burst size (the queued entries resolved as blocked). dark-factory: 0 / 0 / 57 / 52 / 66 /
  33. reify: 2 / 3 / 6 / 4 / 6 / 0. The three zeros at a SIGKILLed run end are not cheap stops: the SIGKILL
  skipped `SpeculativeMergeWorker.stop()`, which is the same 90 s stop-sigterm bound as item 1 (task 7247).
