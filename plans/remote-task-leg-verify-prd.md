# Remote task-leg verifies: a `TaskVerifySpec` transport and a host pool shared by the merge lane and the task leg

**Status:** authored 2026-10-09 (`/team --fable` + `/prd` author mode, unattended; brief `~/.claude/spawn-briefs/remote-task-leg-verify-prd-2026-10-09.md`, item 12 of `plans/verify-rate-improvement-2026-10-09/MEMO.md`; three code-reader seats and one fable critic pass whose 21 findings are folded in). Decomposed 2026-10-09 into tasks 6667–6674 plus reify:8430; see §14, which re-scopes γ, δ1 and δ2 and adds one serialising edge.
**Type:** extension of the shipped multi-host merge-verify lever (Lever C, `plans/concurrent-merge-verify-prd.md`, `plans/merge-lane-throughput-prd.md` D1/D2) to the task role, plus one new ownership seam (a harness-owned host allocator).
**Approach:** B+H. G5 applies on every count: two seams (transport and allocation), five hub files (`config.py`, `workflow.py`, `harness.py`, `merge_lane/worker.py`, `verify_runner.py`), three consumers of the allocator (merge dispatch, drift check, task leg), and a strand class Leo ranks first (a false red in the VERIFY phase drives the debugger).
**Code anchors** verified against main `00725ff2b6` (2026-10-09). Main moves fast — cite-by-symbol; re-locate at implementation time.

**Rulings this PRD executes (Leo, 2026-10-09, verify-rate study MEMO §4):** pile the laptop work on now; size after task 6580 (the advisory merge-phase verify) lands, but author now; never build host-wide verify admission across projects (RED-TIER, `plans/cpu-load-robust-verify-prd.md` §6); a static leg-identity class is allowed, a load-derived count is not (`plans/verify-oversubscription-control-prd.md` §3 C-no-load-derived-count; the 09-11 synthesis ruling R3); a BINARY pressure hold in the runner's health probe is the one allowed load-shaped lever; local is the trust anchor and a laptop task verdict is a lower integrity stake than a laptop merge verdict because the merge gate re-verifies.

## 1. Goal

A task's VERIFY-phase test leg can run on leo-laptop instead of the workstation. The workflow's single task-role verify site routes through a dispatcher that sends a statically identified class of long legs (on Dark Factory: any task verify whose module set includes `orchestrator`) to the laptop when the merge lane does not want it, and runs locally otherwise. The verdict comes back as the same `VerifyResult` the workflow consumes today; a transport failure, a no-verdict kill or a dirty tree falls back to the local run with its local admission slot. The merge lane keeps priority by cgroup CPU weight on the laptop (merge 100, task 33), never by arbitration.

What an operator observes once this lands and DF is flipped:

- `data/verify-logs/<task>/attempt-N.dispatch-<stamp>.json` exists for every DF task verify the dispatcher considered (policy not `local`): `decision: remote`, `runner: laptop` when it ran there, beside the `attempt-N.remote-laptop.*` stream and pass-summary files; a fallback records `decision: local` and a `reason` naming why (`merge_demand`, `no_host`, `host_parked`, `runner_unavailable`, `infra_transient`, `dirty_tree`, `sync_stale`, `not_eligible`, `dispatcher_error`). With `verify_task_host_policy: local` no such file is written and `data/verify-logs` is byte-identical to today.
- The `workflow_verify` green event for that task carries `runner: laptop`.
- On the laptop, `systemctl --user list-units 'df-verify-dark-factory-*'` shows task-tier scopes (`CPUWeight=33`) beside merge-tier ones.
- `mcp__escalation__get_merge_queue` `hosts` block shows the laptop `busy` with `held_by: task:<id>` while a task leg holds it, and merge items arriving meanwhile dispatch `local` (not parked).
- `mcp__escalation__reload_config` after setting `verify_task_host_policy: local` turns it all off without a restart.

## 2. Premise (G6) — measured, skeptic-verified

All figures are from `plans/verify-rate-improvement-2026-10-09/` (W14 = 2026-09-25→10-09; `s6-skeptic.md` re-derived every load-bearing one). Rules are stated with each figure.

- **The workstation is CPU-saturated.** atop over all 336 h: 30.9 of 32 cores busy (97 %), 17.5 cores niced verify test legs. Load at verify start p50 108 / p90 217 (n = 2,668). [S6 claim 5] Consequence: any added local verify concurrency is near zero-sum (task 5797 raised the task slot 1→2 and reverted on measured service inflation: orchestrator-suite leg median 3,233 s → 4,995 s).
- **The task VERIFY phase is mostly waiting for the one per-project admission slot.** Archived `slot_wait_secs` (10-04→10-09, post-5671): DF 95 %, reify 89 %, SCP 88 %, SC 81 % of verify-phase residence; DF residence p50 3.5 h for a ~6 min scoped leg; reify 8.75 h for a 37 min leg. Debugger time < 0.5 %. [S6 claim 1]
- **The long leg is the DF orchestrator full suite:** 86 % of slot work at 3,066 s mean local (09-18 study), versus the laptop running DF's whole 9-module merge gate in ~12.7 min (2026-09-30, `dark-factory-orchestrator.yaml` comment). Laptop:local ≈ 1:3 for DF solo verifies under load ~108; **the DF orchestrator task leg's laptop wall is NOT yet measured** (S4 assumed ~19 min). §9 ε measures it.
- **WIP residence is the binding limit.** All 24 workflow slots per project are held by tasks mostly waiting in verify or merge (rc-c-skeptic.md headline 1: DF 9 merge / 12 verify / 3 execute). Shorter verify-phase residence frees slots and locks.
- **The laptop idles 55 % of its wall** (10-01→10-08, W = 182 h): DF lane genuinely empty 8.3 pp, DF single local-only item 3.8 pp, gate re-verifies held-idle 8.1 pp (owned by 6278), skew incident 5.4 pp (6565). [FINDINGS.md; S6 claim 8]
- **Demand is being deleted first.** Task 6580 removes the advisory merge-phase verify (≈20 h/day fleet, 16–27 % of each project's task slot). This PRD is the capacity lever that remains after that deletion. Its sizing baseline is the first measurement after 6580 lands (§9 ε), per Leo's ruling.

What this PRD does not claim: a throughput number. S4's "~60 extra task verifies per week" rests on an unmeasured laptop wall and is not an acceptance figure anywhere below.

## 3. Sketch of approach

Two seams, deliberately orthogonal (heuristic 3):

**Transport seam.** `TaskVerifySpec` (new module) carries everything verdict-deciding about a task-role verify. `RemoteRunner` gains `run_task_verify(spec)` over a factored `_dispatch` that `run_merge_verify` also uses. The laptop gains `orchestrator verify-task`, a sibling of `verify-merge` sharing one extracted prologue (pgid, watchdog, config) but running `run_scoped_verification(role='task')` on an ephemeral worktree. The verdict is the ordinary `VerifyResult`; red streams are archived on the workstation and the result's log paths point there, never at laptop paths.

**Allocation seam.** The `HostAllocator` moves from a lazily built, worker-owned object to a harness-owned one injected into the merge worker. A `HostPool` wraps it with the one role-aware primitive the task leg needs, `acquire_for_task()`, which hands out the laptop only when the slot is free and the merge lane reports no demand (`SpeculativeMergeWorker.merge_lane_wants_host()`). Non-preemptive: a merge item arriving while a task holds the laptop dispatches local, exactly as `prefer_remote` already does when the remote slot is busy.

**Dispatcher.** `TaskVerifyDispatcher` (new module) is the single consumer of both seams and the only thing the workflow calls. Its decision is a pure function of static config and the module set. Every fallback is a local `run_scoped_verification` with the local admission slot, and every decision is recorded.

Nothing here is a cross-project gate. The laptop's own per-project admission slot (`/tmp/df-verify-slots-<uid>-<hash>`, n = 1, taken on the executing host by `verify.py::run_verification`) is the only thing a remote task leg contends on there, and the merge role never takes it.

## 4. Resolved design decisions

Numbered; each names the heuristic or ruling it turns on.

1. **`TaskVerifySpec` is a new type, not a reuse of `MergeVerifySpec`.** Six of `MergeVerifySpec`'s nine fields are merge-shaped (`unscoped_typecheck`, `cold_timeout_secs`, `is_merge_verify`, `merge_verify_workspace`, `merge_verify_breadth`, the pass-summary scope). A task spec with those fields would lie about what the laptop runs (heuristic 1) and `LocalRunner.run_merge_verify` hard-codes `role='merge'`, `is_merge_verify=True`, `max_retries=0` and the unscoped typecheck gate, none of which the task role runs. The two specs share `VerifyCommand`, the JSON codec helpers and one spec-agnostic `ModuleConfig` rebuild helper (heuristic 11; today `verify_runner.py::_module_config_from_command` is merge-typed and is generalised in α).
   - No `role` field: the type is the role (heuristic 12, structured data over meaningful strings). S5's draft field is dropped.
   - `pytest_n` is **host-read, not spec-carried**, departing from S5 §1. It is load-shaping and verdict-neutral (5295's classification), and the laptop thread split is a laptop-side static ruling (synthesis 09-11 Q2: 8/8 per project). The laptop's `verify_admission_pytest_n` governs. The workstation's "8" is coincidentally the same value today.
   - `task_files` is always a tuple, never `None`: the dispatcher derives it on the workstation (`run_scoped_verification` would otherwise run `git diff <main>...HEAD` on the laptop against a main ref the laptop does not maintain for task branches).
   - `verify_cold_preprovision_command` **is** carried, closing the 6127-class race for task legs at the source: every remote task verify is a cold ephemeral worktree. One rule: a non-empty spec value replaces the laptop's; an empty spec value leaves the laptop's own config value in force (host fallback, the same "host keys absent from the spec are preserved" shape as `verify_env`). 6127's item 1 (carry it on `MergeVerifySpec`) is the merge-side twin (§8).
   - `verify_env` is one spec-level mapping, as on `MergeVerifySpec`. `verify.py::_resolve_verify_env` merges a module's own `verify_env` on top of the top-level one per module, so a project with per-module env would run a different, verdict-deciding env on the laptop (the 5496 class). No DF subproject sets one today. The spec module's docstring states the collapse, and `eligible()` refuses (reason `not_eligible`, logged once at load) any project whose module configs carry per-module `verify_env`, so the fiction cannot ship silently (INV-1; 5295's audit records it).
   - No `base_sha` on the wire: the laptop never reads it, and the workstation pairs the green with its own `_base_commit`. It lives on the dispatch record only (heuristics 5, 6).
   - `scope_cargo` is carried (verdict-deciding: it forks scoping in `run_scoped_verification`). Timeouts, `concurrent_verify`, `verify_timeout_retries`, cgroup knobs and clock-stop stay host-shaped, exactly as for merge.

2. **The laptop entry is a new subcommand, `orchestrator verify-task`, not a role flag on `verify-merge`.** `verify-merge` is 290 lines whose worktree acquisition (`GitOps.acquire_host_verify_worktree`: persistent `_merge-verify` lane, safety valve, lane flock) and run (`run_merge_verify_on_worktree`) are merge-only end to end. A flag would be a 2^N inline-check design (heuristic 3). The shared part — `load_config`, `GitOps`, pgid file, `setsid`, stdin-heartbeat watchdog, exit-code contract — is extracted into one prologue helper both subcommands use (heuristic 11). The task subcommand always uses an ephemeral worktree (there is no persistent task lane on the laptop; the DF laptop yaml already has `persistent_merge_worktree: false`).
   - **The ephemeral worktree gets a `.task/` directory before the run.** `verify.py::_is_verify_cold` reads cold-vs-warm from `.task/verify_warmed`; a worktree with no `.task/` is classified WARM, so preprovision would be skipped and `_persist_attempt_logs` would no-op. A fresh `.task/` makes the verify cold (preprovision runs) and gives the logs a home. This is a named helper with a docstring stating that invariant, not a bare `mkdir` (heuristic 10, informatively). Adding an explicit cold flag to `run_verification` is a tactical option recorded in §13, deliberately not taken here because `verify.py` is in 5651/5653/5628/5139's footprint.
   - Exit-code contract is `verify-merge`'s: `passed=False` exits 0 with the result on stdout; a raised `VerifyInfraError` or any exception exits 1, which the dispatcher reads as `RunnerUnavailable` and falls back local. That is the intended mapping (infra → local), so no second wire encoding is added.
   - `cancel-verify` keys on the pgid file by request id, which `verify-task` writes identically. Because `cancel_request` SIGKILLs the pgid tree, the subcommand's `finally` never runs on a cancel, and each soft-cancelled task verify (146 verify-phase cancels in W14 at restarts) would orphan an ephemeral worktree plus its `.venv` on a laptop whose root is 90 % full; a full disk then reads as `RunnerUnavailable`, quarantine, reprobe, retry. So β records the worktree path beside the pgid file and `cancel_request` removes that worktree on its success path (for both subcommands; `verify-merge`'s ephemeral runs gain the same hygiene). 4196/4437 remain the sweep for everything else. `RemoteRunner.probe_clean`'s `pgrep -f verify-merge` becomes one pattern constant matching both subcommands.
   - The result returned over the wire has `worktree_log_paths=[]` and `archive_log_paths=[]` (laptop paths are meaningless on the workstation); the laptop runs with `archive_root=None`.

3. **One transport, two callers.** `RemoteRunner._dispatch(sha, subcommand, spec_json, *, request_id)` owns: the `refs/merge-verify/<id>` push, the ssh argv with `_SSH_BASE_OPTS`, the stdin-heartbeat run, `result_from_json`, the `RunnerUnavailable` mapping (ssh rc ≠ 0, unparseable stdout), the ref delete and `_inflight_request_id`. `run_merge_verify` = main-mirror push + `_dispatch('verify-merge')` + merge archiving. `run_task_verify(spec, *, task_id, archive_root)` = `_dispatch('verify-task')` + task archiving (streams on red, pass-summary on green, both under `attempt-N.remote-<host>.*`, matching the merge precedent `_archive_failure_streams` / `_archive_pass_summary`).
   - **No-verdict is `RunnerUnavailable` at `_dispatch`, for both callers.** A parsed result whose summary carries a no-verdict fragment (signal kill, xdist worker death) raises `RunnerUnavailable(kind='no_verdict')`. The predicate is a new public `verify.py::is_no_verdict_summary(summary) -> bool` over the existing private marker tuple, so `verify_runner.py` reaches no private name (heuristic 13); α therefore declares `verify.py` (a three-line addition; the file is in 5651/5628/5139's lock footprint).
   - The remote archive helpers currently hard-code `attempt-1` in every file name; α parametrises the attempt id (merge callers keep 1, task callers pass `spec.attempt_id`). This is 6106's first work item, delivered once at the shared point instead of per caller (heuristic 11; §8 amends 6106 to its remaining items: the MemoryMax config field and drop-in retirement). A genuine red stays red.
   - The ref namespace stays `refs/merge-verify/` in α (the laptop-side pgid and ref hygiene, 4196/4437, key on it). Renaming to `refs/verify/` is §13.
   - **Invariant: a `HostLease` on a remote host grants exclusive use of that host's `RemoteRunner` instance.** `_inflight_request_id` is a single slot; two concurrent dispatches on one runner would clobber it and make `cancel_verify` a silent no-op. `HostAllocator` already guarantees one lease per host; `HostPool` preserves it for the task path by never handing out a lease the allocator does not hold. Enforced redundantly: `_dispatch` raises if `_inflight_request_id` is already set (heuristic 10, redundantly).

4. **The allocator is harness-owned and injected; the merge lane's code paths are unchanged.** `harness.py::Harness._start_merge_worker` builds one `HostAllocator` (via `build_remote_runners`, moved out of `merge_lane/drift.py`) and one `runner_quarantine` set for the process lifetime and passes both into `SpeculativeMergeWorker(..., host_allocator=, runner_quarantine=)`. `_ensure_host_allocator` returns the injected one (and still copies the reprobe knobs); its lazy build remains only for bare-worker tests. The allocator keeps no role knowledge (heuristic 6: its purpose stays "one slot per host"). PARKED/quarantine state therefore survives an in-process worker rebuild, which today resets it; §13 records that as the intended reading.
   - `build_remote_runners` lives in the new `orchestrator/verify_hosts.py`, not `verify_runner.py` (3,759 lines, already past heuristic 14's alarm) and not `merge_lane/drift.py`: the workflow side must not import the merge-lane package at all (layering, heuristic 9; `scripts/merge_lane_metrics.py`'s `external_importers` measure counts such imports), and `worker.py` imports `orchestrator.workflow._select_train_members` at module level, so `workflow → merge_lane.worker` would be a cycle. `drift.py` and `worker.py` import it from there; no re-export shim is kept in `worker.py` (heuristic 13; the `reexport_names` ratchet counts shims). `test_merge_lane_alias_names.py` and `test_merge_queue_multihost_wiring.py` are updated to the new home.
   - The quarantine set, allocator and pool are built in `Harness.__init__` (process lifetime), not in `_start_merge_worker`, which only injects them: the merge-worker service is registered unconditionally and `_start_merge_worker` can refuse fail-closed (`MergeLivenessConfigError`), and the task leg must not depend on that outcome. `merge_demand` resolves the worker lazily (`False` while none exists).

5. **`HostPool` is the role-aware layer, in `verify_hosts.py`, composing the allocator rather than extending it.** It holds the allocator, a `MergeDemand = Callable[[], bool]` supplied by the harness (which is the only module that may know both the worker and the task side), the task-hold registry, and two unavailability hooks. `acquire_for_task(task_id)` returns a remote `HostLease`, or a frozen `Refusal(reason)` value (`no_host`, `host_parked`, `merge_demand`, `pressure_hold`) — a value return, never a side channel read after the call, because up to `max_concurrent_tasks` dispatchers share the pool (heuristic 7). It returns a lease iff an acquirable remote exists and `merge_demand()` is false; it never returns a local lease (a local task verify holds no host slot; it is gated by the flock admission semaphore, an independent mechanism). Non-preemptive by construction: there is no path that cancels a task hold on merge demand. Preemption is §12 out of scope.
   - **A task-path unavailability must enter the merge lane's RU tracker, or the laptop is lost to both lanes.** `HostAllocator.quarantine_and_release` only adds the name to the shared quarantine set; readmission is tracker-driven — `SpeculativeMergeWorker._reprobe_quarantined_hosts` iterates `_runner_unavailable`, whose only writer is the merge-lane path `_quarantine_unreachable_host` → `_record_runner_unavailable` (task 3043), and `_host_states_block` classifies an untracked quarantined host as `divergence`, "cleared only by an operator". So `HostPool` takes `on_unavailable(name, reason)` and `on_recovered(name)` callables; the harness wires them to a public worker method δ1 adds (`note_runner_unavailable(name, reason)`, the existing tracker entry point made public) so the existing streak, `verify_host_unreachable` escalation ladder and reprobe sweep cover task-path RU and stale-sync exactly as merge-path RU. The pool never calls `quarantine_and_release` directly on a task lease.
   - **`SpeculativeMergeWorker.merge_lane_wants_host()`** is the worker's one new read: true when any item is parked awaiting a host (`_redispatch` non-empty), built but undispatched (`_verifier_queue` non-empty, or `_pending_verifier_get` done), or being built under a live merge-ahead permit (`_merge_ahead_ledger.live > 0`). It over-counts `DecidedItem` passthroughs that need no host; over-counting yields the laptop to a merge that then dispatches local, which is the cheap side of the asymmetry. It never reads the allocator (that would re-create `_dispatch_opportunity_exists`'s conjunction, which is the wrong half).
   - `HostPool.holders()` returns `{host: 'task:<id>'}` for task-held leases; `_host_states_block` in the worker's snapshot gains `held_by` from an injected reader so a task-held laptop never reads as "busy with no merge item" (INV-7, 3275's observability surface).

6. **The dispatcher's decision is static.** `verify_task_host_policy: local | remote_modules | remote_all` (default `local`) and `verify_task_remote_modules: list[str]` (module prefixes; DF: `[orchestrator]`), both green-tier. With policy `local` the dispatcher short-circuits before recording anything, so production is byte-identical to today including `data/verify-logs`. Otherwise a verify is eligible iff the policy is `remote_all`, or `remote_modules` and the **executed** module set's prefixes intersect the list — the set `verify_plan.derive_verify_plan(…, role='task')` narrows the assigned modules to (the same narrowing `run_scoped_verification` applies), so a task assigned `orchestrator` whose diff touches no orchestrator file does not ship a short leg to the laptop. `force_workspace` verifies (train members) are never eligible in this PRD. Nothing in the decision reads load, queue depth, slot wait or time of day (C-no-load-derived-count). `remote_modules` with an empty list validates as a config error at load (INV-1: the contract is machine-checked, not prose).

7. **Every non-remote outcome is one local call with the local slot.** Order inside `TaskVerifyDispatcher.run`: eligibility → `pool.acquire_for_task(task_id)` (a `Refusal` → local with its reason) → `GitOps.has_uncommitted_work` (dirty → release, `dirty_tree`; the pushed ref would silently omit the WIP) → `RemoteRunner.sync_if_stale` (not ok → `on_unavailable(name, 'sync_stale')` + release, `sync_stale`) → `run_task_verify` → `RunnerUnavailable` → `on_unavailable(name, kind)` + release, `runner_unavailable`. An `INFRA_TRANSIENT` category on a remote result is treated like `RunnerUnavailable` (local fallback inside the same call, reason `infra_transient`) so laptop infra never consumes the workflow's `verify_infra_retry_max_attempts` budget. A remote red is returned as red unless the parity knob in §4.9 is on; `verify_failure_is_preexisting_on_main` then runs its local main probe exactly as today (it is leaseless and deliberately never remote).
   - **The lease is released on every exit.** The dispatch body runs under one `try/finally`: `release(lease)` when no dispatch was in flight, `cancel_and_release(lease)` under `asyncio.shield` when one was (cancellation or an exception mid-ssh). An unexpected exception is re-raised after the release and recorded with reason `dispatcher_error`. Nothing else may leave the laptop `busy` with `held_by: task:<id>` for a dead owner (INV-7; `HostAllocator.readmit` deliberately cannot touch a BUSY slot, so a leak here would be permanent until restart).
   - The decision is recorded in the dispatcher's **own** sidecar, `data/verify-logs/<task>/attempt-N.dispatch-<stamp>.json`: `{decision, runner, reason, tip_sha, base_sha, remote_wall_secs, lease_held_secs, cross_check, red_confirmed_locally}`. It does not edit the per-module summaries `verify.py::_build_summary_payload` writes (one builder per record, heuristic 11; a verify produces one summary per module while the decision is per attempt, and the summary path is not returned on the result). INV-2, INV-11: a fallback is distinguishable from a remote run by a reader of that file, not only of the log; INV-13: ε's reader keys on it and the pre-flight is its first production row.

8. **Cancellation.** The workflow's soft cancel reaches a running verify as `asyncio.CancelledError` on the body task (`workflow_types.py::CancellationScope.supervise`), before `_on_terminal_cleanups` run. The dispatcher's `finally` (§4.7) therefore calls `pool.cancel_and_release(lease)` under `asyncio.shield` BEFORE the dispatch coroutine unwinds (the `_inflight_request_id` is cleared on unwind; `merge_lane/worker.py::_abort_remote_verify` documents the same ordering). The remote stdin-heartbeat watchdog remains the backstop for a dead workstation process. A failed cancel parks the slot; 3310 (in progress) owns PARKED recovery and is a prerequisite for the DF flip, since restarts soft-cancel every verify-phase task (146 in W14).

9. **Trust anchor and parity, in both directions.** A laptop task verdict is re-gated by the merge lane, so a false GREEN is an integrity cost the lane catches. A false RED is the strand class: `_verify_debugfix_loop` sends any red to the debugger after a main probe that classifies against main, not the branch tip, so a laptop-only red (6127's 171 `reportMissingImports` with every pytest leg green is a measured instance) would be debugged as real. Two static parity knobs, both green-tier, both set by ε for a bounded window and then turned off, following the 5416 precedent (adopt at zero disagreements):
   - `verify_task_remote_red_confirms_locally: bool` (default false): a remote red runs the ordinary local verify (with its local slot) before the verdict is returned; the local verdict is returned, and the sidecar records `red_confirmed_locally: {remote_passed: false, local_passed}`. This reads every red, which is the trip-wire's direction. It is off by default because in a debug loop most verifies are red and each confirmation is a local long leg; whether to keep it on after the window is Leo's call on ε's report (§12).
   - `verify_task_remote_cross_check_every_n: int` (0 = off): every n-th eligible verify runs remote AND local (local authoritative), recording `cross_check: {remote_passed, local_passed, agree}` — the false-green direction.

10. **Observability without a new event type.** Where a verify ran has two grains: per attempt (the dispatch sidecar, §4.7) and per green (the `workflow_verify` event, which gains `runner`). No `task_verify_dispatch` event is added: task 5651 owns the verify event shape (run wall, `slot_wait_secs`, role, checkpoint hit) and a second event for the same fact would be a second home (INV-9). The existing `runner_stale` / `runner_synced` / `verify_host_unreachable` events already fire from `sync_if_stale` and the reprobe ladder for the task path unchanged.

11. **Pressure hold (phase 2, η).** The one load-shaped lever Leo allows: a binary hold for task legs only — `verify_task_remote_pressure_hold: {cpu_some_avg10: float, probe_interval_secs: float, max_age_secs: float}` (null = off). `acquire_for_task` is synchronous, and `RemoteRunner.health()` is `ssh host true` run only by the reprobe sweep, so the reading needs a producer: a `verify_hosts.py` probe loop refreshes a cached `PressureReading(value, read_at)` per remote host on the static interval (`ssh host cat /proc/pressure/cpu`). `acquire_for_task` compares the cached `some avg10` against the static threshold: above → `Refusal('pressure_hold')`; absent or older than `max_age_secs` → no hold, recorded as `pressure_hold_stale` (fail-open, INV-11). A static threshold on a static-cadence reading is the allowed binary shape; it never derives a count and never applies to merge legs. η is filed `deferred` at decompose, flipped only by ε's step 6 if the report shows overload (F11 of the critic pass: a status dependency would auto-dispatch it on ε's completion regardless of the evidence).

12. **Placement (heuristics 9, 13, 14).** New files: `orchestrator/src/orchestrator/task_verify_spec.py` (spec, codec, rebuild), `verify_hosts.py` (runner construction, `HostPool`, `MergeDemand`), `task_verify_dispatch.py` (the dispatcher and `TaskVerifyPort`). Each reads in isolation: `task_verify_spec` imports downward from `verify_runner` (`VerifyCommand`, codec helpers) and `config`; `verify_hosts` imports `verify_runner` (`HostAllocator`, `RemoteRunner`, `resolve_local_df_checkout`); `task_verify_dispatch` imports both plus `verify` and `git_ops`. None imports `merge_lane.*` or `workflow`. `workflow.py` imports `task_verify_dispatch` for the `TaskVerifyPort` type only; the default port stays the module-level `run_scoped_verification` so an un-injected workflow is byte-identical and every existing `patch('orchestrator.workflow.run_scoped_verification')` keeps working.

13. **Hub-file carry chain.** α, β, γ touch no hub file and are dispatchable. δ1 (`harness.py` + `merge_lane/worker.py`) and δ2 (`config.py` + `workflow.py`) are hand-carries, chained δ1 → δ2. `worker.py` cannot grow in lines without `--authorize-raise` (merge-lane ratchet); `build_workflow` has a keyword-set tripwire (`test_workflow_factory.py::_BUILD_WORKFLOW_PARAMS`). δ2 waits for 6580 so there is exactly one task-role call site to route and the two `workflow.py` edits do not collide.

## 5. Contract

### 5.1 `orchestrator/src/orchestrator/task_verify_spec.py`

```python
@dataclass(frozen=True)
class TaskVerifySpec:
    verify_commands: tuple[VerifyCommand, ...]   # the module set; spec-authoritative (4536 rule)
    global_verify_command: VerifyCommand | None  # zero-module projects (INV-1 global gate)
    task_files: tuple[str, ...]                  # never None; derived on the dispatcher
    force_workspace: bool
    verify_env: Mapping[str, str]                # dispatcher's effective env; spec wins, host-only keys preserved (5496)
    scope_cargo: bool
    verify_cold_preprovision_command: str        # non-empty replaces the host's; '' keeps the host's own (fallback)
    attempt_id: int
    task_id: str
    tip_sha: str                                 # the pushed SHA; equals the ref target

def build_task_verify_spec(config, module_configs, task_files, *, force_workspace, attempt_id, task_id, tip_sha) -> TaskVerifySpec
def apply_task_verify_spec(config, spec) -> tuple[OrchestratorConfig, list[ModuleConfig]]
    # laptop side: config.model_copy(update={verify_env: {**host, **spec.verify_env}, scope_cargo,
    # verify_cold_preprovision_command only when spec's is non-empty,
    # global commands when spec.global_verify_command}) and the rebuilt module list; mirrors run_merge_verify_on_worktree
def spec_to_dict / spec_from_dict / task_spec_to_json / task_spec_from_json
```

Invariants: `from_dict` rejects unknown keys (a wire-shape mismatch is a `TypeError` the dispatcher maps to `RunnerUnavailable`, the 6565 skew class — bench, do not loop); `task_files` is a tuple even when empty; `tip_sha` is a full 40-hex SHA. Timeouts, `-n`, nice, cgroup weight, clock-stop and `verify_timeout_retries` are deliberately absent: host-shaped, the laptop config wins.

### 5.2 `verify_runner.py::RemoteRunner`

```python
async def _dispatch(self, sha: str, subcommand: Literal['verify-merge', 'verify-task'], spec_json: str) -> VerifyResult
    # push refs/merge-verify/<id>; ssh `orchestrator <subcommand> --sha --spec --config --request-id`;
    # result_from_json; RunnerUnavailable on rc != 0 / unparseable / no-verdict summary; delete ref; clear inflight.
    # Raises if _inflight_request_id is already set (exclusive-use invariant).
async def run_merge_verify(...)   # unchanged signature; main-mirror push + _dispatch('verify-merge') + merge archiving
async def run_task_verify(self, spec: TaskVerifySpec, *, task_id: str, archive_root: Path | None) -> VerifyResult
    # _dispatch('verify-task') + archive: red → attempt-{spec.attempt_id}.remote-<name>.{test,lint,type}-<stamp>.log
    # + .summary-<stamp>.json; green → .pass-summary-<stamp>.json; sets result.archive_log_paths to those files.
class RunnerUnavailable(Exception): kind: Literal['transport', 'unparseable', 'no_verdict', 'wire_shape']
```

`probe_clean` matches `verify-(merge|task)`. `cancel_verify` is unchanged.

### 5.3 `cli.py` — `orchestrator verify-task`

```
orchestrator verify-task --sha <sha> --spec <TaskVerifySpec JSON> --config <laptop yaml> [--request-id <id>]
```

Shared prologue (extracted from `verify_merge`): `load_config`, `GitOps`, pgid file + `start_own_process_group` + `write_pgid_file`, stdin watchdog, `_hand_exit_to_fired_watchdog`, stdout = `result_to_json`, stderr = logs; exit 0 on any `VerifyResult`, 1 on config/spec/exception/watchdog. Task body: `GitOps.create_ephemeral_verify_worktree(sha)` (public; the ephemeral branch of today's `_create_merge_worktree`; `git_ops.py` is in the merge-lane ratchet cluster with its line count frozen, so β authorises the raise), `write_worktree_path(pgf, wt)` beside the pgid file so `cancel_request` can reap it, `prepare_task_verify_worktree(wt)` (creates `.task/`; docstring states the cold invariant), `apply_task_verify_spec`, `run_scoped_verification(wt, config', modules', list(spec.task_files), attempt_id=spec.attempt_id, task_id=spec.task_id, archive_root=None, force_workspace=spec.force_workspace, role='task')`, strip `worktree_log_paths`/`archive_log_paths`, `cleanup_merge_worktree` in `finally`. No lane flock (no persistent task lane), no unscoped typecheck, no flake suppression.

### 5.4 `orchestrator/src/orchestrator/verify_hosts.py`

```python
def build_remote_runners(config, cwd, *, quarantine: set[str] | None = None) -> list[RemoteRunner]   # moved verbatim from merge_lane/drift.py
MergeDemand = Callable[[], bool]

@dataclass(frozen=True)
class Refusal:
    reason: Literal['no_host', 'host_parked', 'merge_demand', 'pressure_hold']

class HostPool:
    def __init__(self, allocator: HostAllocator, *, merge_demand: MergeDemand,
                 on_unavailable: Callable[[str, str], None], on_recovered: Callable[[str], None]) -> None
    @property allocator -> HostAllocator                      # the merge lane's object, unchanged API
    def acquire_for_task(self, task_id: str) -> HostLease | Refusal
        # remote only; records the hold under task_id; `host_parked` when the only remote is PARKED (HostAllocator.is_parked)
    async def release(self, lease) -> None
    async def cancel_and_release(self, lease) -> bool          # delegate; drop the hold record either way
    def mark_unavailable(self, lease, reason: str) -> None     # on_unavailable(name, reason): the merge lane's RU tracker, never a bare quarantine
    def holders(self) -> Mapping[str, str]                    # {host: 'task:<id>'}
```

Invariants: `acquire_for_task` is synchronous and atomic on the loop (as `HostAllocator.acquire_remote` is); a hold record exists iff the allocator slot is BUSY for a task; `merge_demand` is read per call and never cached; the pool never writes the quarantine set itself.

### 5.5 `orchestrator/src/orchestrator/task_verify_dispatch.py`

```python
class TaskVerifyPort(Protocol):
    async def __call__(self, worktree, config, module_configs, task_files=None, *, attempt_id, task_id, archive_root, force_workspace, role='task', event_store=None) -> VerifyResult
    # exactly run_scoped_verification's call shape at the workflow site

@dataclass(frozen=True)
class TaskVerifyPolicy:       # decompose 2026-10-09 (§14): γ's value; δ2's config-backed reader builds it
    host_policy: Literal['local', 'remote_modules', 'remote_all']; remote_modules: tuple[str, ...]
    cross_check_every_n: int; red_confirms_locally: bool

class TaskVerifyDispatcher:   # satisfies TaskVerifyPort
    def __init__(self, pool: HostPool, git_ops: GitOps, *, policy_of: Callable[[OrchestratorConfig], TaskVerifyPolicy],
                 run_local: TaskVerifyPort = run_scoped_verification) -> None
    async def __call__(...) -> VerifyResult
    def decision_for(self, task_id: str, attempt_id: int) -> DispatchDecision | None   # keyed, never a shared "last" slot

@dataclass(frozen=True)
class DispatchDecision:
    decision: Literal['remote', 'local']; runner: str
    reason: Literal['remote', 'not_eligible', 'no_host', 'host_parked', 'merge_demand', 'pressure_hold', 'dirty_tree', 'sync_stale', 'runner_unavailable', 'infra_transient', 'cross_check', 'dispatcher_error']
    tip_sha: str | None; base_sha: str | None; remote_wall_secs: float | None; lease_held_secs: float | None
    cross_check: CrossCheck | None; red_confirmed_locally: RedConfirmation | None

def eligible(policy: TaskVerifyPolicy, executed_prefixes: Iterable[str], *, force_workspace, per_module_env_present: bool) -> bool   # pure; the static leg class
```

Ordering, as in §4.7, inside one `try/finally` that releases the lease on every exit. The dispatcher writes its own sidecar `attempt-N.dispatch-<stamp>.json` under `archive_root/<task_id>/` (nothing when policy is `local`); it never edits files other modules write. The policy is read per call, as `policy_of(config)` on that call's `config`, so a hot-reloaded policy reaches the next dispatch and never a running one.

### 5.6 Workflow, worker, harness seams

- `TaskWorkflow.__init__(..., *, task_verify: TaskVerifyPort | None = None)`, forwarded by `build_workflow`. `_run_scoped_verification_with_infra_retry` calls `self._task_verify or run_scoped_verification` with the unchanged argument list, still inside `git_ops.task_verify_lease` (the lane lease keeps warm-lane GC off the workstation worktree the workflow still uses). After `_verify_debugfix_loop` returns DONE, `_verify_green_runner` is read from the dispatcher's `decision_for(task_id, attempt_id)` (`'local'` when none) and emitted on `workflow_verify` as `runner` (through 5409's `emit_workflow_verify` if it has landed; readers ignore unknown keys).
- `SpeculativeMergeWorker.__init__(..., host_allocator: HostAllocator | None = None, runner_quarantine: set[str] | None = None, host_holder_reader: Callable[[], Mapping[str, str]] | None = None)`; `merge_lane_wants_host() -> bool`; `note_runner_unavailable(name, reason)` and `note_runner_recovered(name)` (public names over the existing `_quarantine_unreachable_host` / `_record_runner_recovered` tracker paths); `_host_states_block` adds `held_by`.
- `Harness.__init__` owns `_runner_quarantine`, `_host_allocator` (via `build_remote_runners`), `_host_pool` and `_task_verify_dispatcher` for the process lifetime; `_start_merge_worker` injects the allocator and set into each worker it builds; `merge_demand` and the two hooks resolve `self._merge_worker` lazily (no worker → `False` / no-op). `build_workflow(..., task_verify=self._task_verify_dispatcher)`. When `enabled_verify_runners` is empty, `acquire_for_task` always returns `Refusal('no_host')` and every eligible verify runs local. (Decompose split, §14: δ1 builds the quarantine set, allocator and pool; δ2 builds `_task_verify_dispatcher` with `policy_of=policy_from_config` and adds the `task_verify=` argument at the harness `build_workflow` call.)

### 5.7 Config (`config.py`, green tier unless stated)

| key | type / default | tier |
|---|---|---|
| `verify_task_host_policy` | `Literal['local','remote_modules','remote_all']` = `local` | green |
| `verify_task_remote_modules` | `list[str]` = `[]`; validator: non-empty when policy is `remote_modules` | green |
| `verify_task_remote_cross_check_every_n` | `int` = 0, ge 0 | green |
| `verify_task_remote_red_confirms_locally` | `bool` = false | green |
| `verify_task_remote_pressure_hold` | `PressureHoldConfig \| None` = None (`cpu_some_avg10`, `probe_interval_secs`, `max_age_secs`) | green (η) |
| `verify_runners`, `verify_host_policy`, `verify_drift_check_every_n_lands` | existing; unchanged | restart / green / restart |

Laptop side (not in this repo, `~/.config/orchestrator/dark-factory-laptop.yaml`): `verify_admission_task_slots: 1` (default), `verify_admission_pytest_n: "8"` (Q2 split), `verify_use_cgroup_scope: true` (already), `verify_cold_preprovision_command` (already set 2026-10-01). The DF scope drop-in `df-verify-dark-factory-<hash>-.scope.d/` covers task scopes too because the scope tag derives from the project root, not the role (seat-verified via `_scope_tag_for`).

## 6. Boundary-test sketch

Both sides of each seam are faced; the integration gate ι executes every row. Fakes follow `test_multihost_verify_integration.py::_RemoteFakeRunner` and `test_host_allocator.py::_FakeRemoteRunnerCancellable`; the CLI round trip follows `test_cli.py::test_verify_merge_cli_wrapper_transparency` on `_setup_verify_repo`.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Spec round trip | a `TaskVerifySpec` with two modules, env, `task_files=()` | `task_spec_from_json(task_spec_to_json(s)) == s`; an extra key raises `TypeError` |
| 2 | Laptop body on a fixture repo | fixture repo at SHA, spec with `true`/`false` test commands, `--request-id` | exit 0 both ways; parsed result equals a direct in-process `run_scoped_verification(role='task')`; `.task/verify_warmed` absent before → the SPEC's preprovision command observed when non-empty, the host's when the spec's is `''`; worktree removed; pgid and worktree-path files removed; log path lists empty |
| 3 | Infra on the laptop | the fixture's test command raises `VerifyInfraError` via a killed leg | CLI exits 1; `RemoteRunner.run_task_verify` raises `RunnerUnavailable(kind='transport')` |
| 4 | No-verdict kill | fake stdout is a parseable result whose summary carries `killed by signal 9` | `_dispatch` raises `RunnerUnavailable(kind='no_verdict')` for BOTH `run_merge_verify` and `run_task_verify`; nothing red is returned |
| 5 | Eligible, free host, no merge demand | policy `remote_modules: [orchestrator]`, executed module set includes `orchestrator`, fake remote free, `merge_demand` False | remote result returned; sidecar `attempt-N.dispatch-*.json` has `decision=remote, runner=laptop`; lease released; `archive_log_paths` name workstation files; no per-module summary was modified |
| 6 | Eligible, merge demand | as 5 but `merge_demand` True | local `run_local` called once with the identical arguments; `reason=merge_demand`; remote never called; allocator slot FREE throughout |
| 7 | Merge arrives while a task holds the laptop | task lease held; the real `SpeculativeMergeWorker._dispatch_item` under `prefer_remote` | merge dispatches `local`, is not parked; snapshot `hosts` shows the laptop `busy`, `held_by: task:<id>` |
| 8 | Not eligible | policy `local`, or `remote_modules: [fused_memory]` with an `orchestrator` set, or `force_workspace=True` | `run_local` called; `reason=not_eligible`; the pool is never consulted |
| 9 | RunnerUnavailable fallback | fake remote `unavailable=True`, real `SpeculativeMergeWorker` wired as the hook target | local result returned; `reason=runner_unavailable`; snapshot `hosts` shows the laptop `quarantine_class: ru`, streak 1 (not `divergence`); after `health()` passes, `_reprobe_quarantined_hosts` re-admits it with no restart; the `verify_host_unreachable` ladder fires at the configured streak |
| 10 | Infra-transient remote result | fake remote returns `category` in `INFRA_TRANSIENT_CATEGORIES` | local run in the same call; the workflow wrapper's retry counter is untouched; `reason=infra_transient` |
| 11 | Dirty tree | worktree has an uncommitted file | local; `reason=dirty_tree`; no push, no lease held after |
| 12 | Stale sync | `sync_if_stale` returns `ok=False` | local; `reason=sync_stale`; host enters the RU tracker (as row 9), not a bare quarantine; `runner_stale` event emitted |
| 13 | Soft cancel mid-remote | dispatch awaiting the fake ssh; `_cancel_event` set | `cancel_verify` called BEFORE unwind (fake records order); slot FREE; `CancelledError` propagates; workflow outcome `SOFT_CANCELLED` |
| 14 | Cancel fails | fake `cancel_verify` rc 1, `probe_clean` False | slot PARKED; `holders()` empty; a later `acquire_for_task` returns `Refusal('host_parked')` and the sidecar says so (3310's class is countable) |
| 15 | Cross-check | `verify_task_remote_cross_check_every_n=2`, two eligible verifies | first remote only; second remote then local, local verdict returned, sidecar `cross_check={remote_passed, local_passed, agree}` |
| 16 | Un-injected workflow | `build_workflow` without `task_verify` | `run_scoped_verification` called via the module-level name (existing patch target), byte-identical argument list |
| 17 | `workflow_verify` carries runner | scenario 5 through a real `TaskWorkflow` to REVIEW | event payload has `runner: 'laptop'` beside `tip_sha`; `green_checkpoint_at_tip` still hits on the next run |
| 18 | Config contract | `remote_modules` with `[]` | `load_config` rejects with a message naming both keys; `reload_config` reports the field as applied when valid |
| 19 | Exclusive runner use | two `_dispatch` calls on one `RemoteRunner` concurrently | the second raises before pushing; `_inflight_request_id` unchanged |
| 20 | Hub wiring | real `Harness` with one fake runner config | one `HostAllocator` object is shared by `worker._host_allocator`, `pool.allocator` and the dispatcher; a worker rebuild keeps the quarantine set; a `_start_merge_worker` refusal leaves the pool usable (`merge_demand` False) |
| 21 | Dispatcher error | fake `run_task_verify` raises `RuntimeError` | slot FREE; `holders()` empty; the exception propagates; sidecar `reason=dispatcher_error` |
| 22 | Remote red, confirm on | `verify_task_remote_red_confirms_locally=True`, fake remote red, local green | local verdict (green) returned; sidecar `red_confirmed_locally={remote_passed: false, local_passed: true}`; the debugger is not invoked |
| 23 | Policy `local` | `verify_task_host_policy: local` | `run_local` called with the identical argument list; no sidecar written; the pool is never consulted |
| 24 | Cancelled laptop run reaps its worktree | `verify-task` running on the fixture repo; `cancel-verify --request-id` | pgid tree killed; the ephemeral worktree and its `.task/` are gone; pgid and worktree-path files removed |
| 25 | Per-module env refusal | a module config carrying `verify_env` | `eligible()` False with reason `not_eligible`; the refusal is logged once at config load naming the module |

## 7. Pre-conditions (G3) — substrate verified on `00725ff2b6`

Each line was read in code by a seat on 2026-10-09; symbol-cited.

- `verify_runner.py::RemoteRunner.run_merge_verify` does push → ssh → `result_from_json` → archive → delete-ref with a single `_inflight_request_id`; `cancel_verify` returns 0 with nothing in flight; `probe_clean` greps `verify-merge`; `sync_if_stale` returns `ok=True, synced=False` when a dispatch is in flight (one-lease-per-host is what keeps that safe within a process; cross-process is 6565's).
- `verify_runner.py::HostAllocator`: one slot per host, `acquire(policy)` atomic, no roles; `cancel_and_release` idempotent on FREE, parks on cancel rc ≠ 0; `readmit` un-parks; quarantine set shared by reference. No injection kwarg on `SpeculativeMergeWorker`; `_host_allocator` is a plain attribute, built lazily by `_ensure_host_allocator` from `merge_lane/drift.py::_build_remote_runners`, whose only dependencies are `RemoteRunner` and `resolve_local_df_checkout` (no cycle on moving).
- `harness.py::Harness._start_merge_worker` constructs the worker; `build_workflow` is the single `TaskWorkflow` construction site (keyword tripwire `test_workflow_factory.py`); no harness-owned object is injected into both the worker and workflows today beyond `git_ops`, `event_store`, `scheduler`, `usage_gate`, `cost_store`, `merge_worker`.
- `cli.py::verify_merge`: prologue steps are as §5.3 lists; `passed=False` exits 0; `acquire_host_verify_worktree` is merge-shaped; `run_merge_verify_on_worktree` hard-codes the merge profile. `verify_cancel.py` has `pgid_file`, `start_own_process_group`, `write_pgid_file`, `start_stdin_watchdog`, `fire_watchdog_kill`, `cancel_request`.
- `verify.py::run_scoped_verification` signature as in §5.5; admission wraps only the `test` leg via `_admission_slot`, gated roles `task`/`background`, slot acquired on the executing host; `_is_verify_cold` keys on `.task/verify_warmed`; `_preprovision_shared_venv` reads `config.verify_cold_preprovision_command` (host); `VerifyResult` has no `runner` or `slot_wait_secs` field (that is `CheckRun`, archived in the summary by 5671); `_NO_VERDICT_SUMMARY_MARKERS` exists; `INFRA_TRANSIENT_CATEGORIES` is in `verify_categories.py`.
- `workflow.py::TaskWorkflow._run_scoped_verification_with_infra_retry` is the VERIFY-phase site, inside `task_verify_lease`, module-level `run_scoped_verification` import (patch point); `_run_merge_phase` is the second task-role site (deleted by 6580); `verify_failure_is_preexisting_on_main` is local-only by docstring; `workflow_verify` is emitted in `_enter_phase` with `{passed, tip_sha, base_sha, branch}`; soft cancel arrives as body-task `CancelledError` via `CancellationScope.supervise`; `_on_terminal_cleanups` is a fixed list (no registration API); `GitOps.has_uncommitted_work` exists.
- Config: `verify_host_policy` green; `verify_runners`, `verify_drift_check_every_n_lands`, `verify_use_cgroup_scope` restart; `VerifyRunnerConfig` has `config_path`, `df_checkout_path`, `enabled`. DF yaml: `laptop` runner enabled, `prefer_remote`, drift every 10, `verify_admission_pytest_n: "8"`, `verify_use_cgroup_scope: true`, `verify_admission_task_slots: 1`.
- Ratchets: `scripts/merge_lane_metrics.py::CLUSTER_PATHS` covers `merge_lane/**`, the alias modules, `git_ops.py` (size-ceiling exempt but line-frozen: "cannot grow unwatched") and two test files — not `verify_runner.py`, `harness.py`, `workflow.py`, `cli.py`, `verify_cancel.py`; new cluster functions ≤ cognitive 15; `worker.py` and `git_ops.py` line growth needs `--authorize-raise <task>` recorded in the same commit; `test_merge_lane_alias_names.py` pins alias-reachable names; `reexport_names` is structural (imported and unused), so γ's import-and-use hop in `drift.py` is not a shim.
- Laptop (brief, 2026-10-09): 16 threads, 60 G RAM, 136 G swap; DF scope drop-in switched to `MemoryHigh` throttle + higher `MemoryMax` + swap; `verify_use_cgroup_scope: true`; preprovision command set 2026-10-01; laptop configs exist for DF and reify only.

No fiction found. Two substrate items are **queued prerequisites**, not assumptions: 6565 (sync to the deployed SHA / RW-flock on the shared DF checkout) and 3310 (PARKED recovery, in progress).

## 8. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| 6580 stop the advisory merge-phase verify | prerequisite of δ2 | the second task-role `run_scoped_verification` site in `_run_merge_phase` disappears; δ2 routes the one remaining site | 6580 | pending high; δ2 depends on it |
| 6565 INV-2 sync to the deployed SHA + RW-flock on the laptop DF checkout | prerequisite of ε | every remote task leg is another `sync_if_stale` caller against the shared checkout | 6565 | pending high; ε depends on it |
| 6106 MemoryMax kill reads RED | **amended**: α delivers "no-verdict → `RunnerUnavailable`" once at `_dispatch` for both callers (via the public `is_no_verdict_summary`); 6106 keeps the MemoryMax/MemorySwapMax config field (5205's `systemd-run -p` seam) and the drop-in retirement | `_NO_VERDICT_SUMMARY_MARKERS` classification at `RemoteRunner._dispatch` | α (classification), 6106 (config field) | decompose amends 6106's text; if 6106 lands first, α adopts its classifier |
| 6127 cold-venv pyright false red on the laptop | twin | `TaskVerifySpec` carries `verify_cold_preprovision_command` from the start; 6127 item 1 adds it to `MergeVerifySpec`; items 2–3 (branch_bug misclassification of third-party `reportMissingImports`, regression tests) stay with 6127 | this PRD (task spec), 6127 (merge spec, classifier) | 6127 pending; ε depends on it for the DF flip (a phantom red in VERIFY drives the debugger) |
| 3310 HostAllocator PARKED recovery | prerequisite of ε | task verifies are soft-cancelled at every restart; a failed cancel parks the host | 3310 | in progress (claimed 2026-10-09); ε depends on it |
| 5295 host-vs-spec field audit | pattern | §5.1's field classification (carried / host-shaped) is the task-role instance of 5295's merge-path audit; 5295's guard test should cover `TaskVerifySpec`'s reads when it lands | 5295 extends; this PRD records the classification in `task_verify_spec.py`'s docstring | pending low |
| 5651 verify events (plan hash, per-unit outcomes, task-leg duration, slot wait, role) | consumer of this PRD's `runner` | `workflow_verify` gains `runner` here (δ2); 5651 adds its own fields to the same payload; neither restates the other | this PRD owns `runner`; 5651 owns duration/slot-wait/role | pending high, hub-heavy; no ordering constraint, keys are disjoint |
| 5409 stamp `tip_sha` on every `workflow_verify` green (shared `emit_workflow_verify`) | shared emit site | δ2 uses 5409's constructor if landed, else adds `runner` at `_enter_phase` and 5409 absorbs it | 5409 | pending medium |
| 5628 ordered admission gate with recorded wait | complements | the in-process gate sits inside the local branch (`verify.py`), which a remote leg bypasses by construction; no shared code | 5628 | pending high; hub overlap with δ2 (`workflow.py`, `config.py`) — lock-serialised, sequencing noted for the carry |
| 6278 gate re-verify on the head's leased host; 6160 train verify on an acquired lease | sibling allocator consumers | both read the worker's `HostAllocator`; after δ1 that object is the injected one (same API) | each task | 6278/6160 share a `worker.py`/`gates.py` chain; δ1's `worker.py` edit must land before or be rebased under theirs |
| 6648 drift check: concurrent legs, defer instead of skip when a slot is busy | prerequisite of ε | `drift.py::_run_drift_check` calls `acquire_remote()` directly and skips the check, consuming the every-N cadence, when the laptop is busy; a task hold would silently thin the standing fidelity guard under `prefer_remote` | 6648 (dispatchable: `drift.py` + `verify_runner.py`) | pending medium; ε depends on it |
| `plans/cpu-load-robust-verify-prd.md` §6, 5052 | ruling | no host-wide admission; this PRD's pool is per-process and allocator-slot based, not a shared slots dir | — | honoured |
| `plans/verify-oversubscription-control-prd.md` C-merge-priority, C-no-load-derived-count | ruling | merge never takes the semaphore; the leg class is static; η's hold is binary | — | honoured |
| 4537 env-fingerprint divergence detector; 4196/4437 remote GC | costs, not blockers | more laptop runs raise the value of the drift-10 guard and of pgid/worktree GC; β's cancel-time worktree reap covers the cancel path, 4196 the rest; ε adds laptop free space to its trip-wires | those tasks | pending |
| reify adoption | downstream | `verify_task_host_policy: remote_all` + reify laptop yaml; a reify task is filed at decompose as an external dependent of ε | reify | out of scope here (§12) |

No reciprocal ambiguity: every row names one owner.

## 9. Decomposition plan

Sizes follow the overlay bands. Labels are Greek; ids are assigned at decompose. "Hub" marks a hand-carry (feedback_hand_carry_procedure). No two leaves edit the same file except where the chain serialises them.

- **α — Task spec and transport (no hub; dispatchable).** [~550 LOC; files: `task_verify_spec.py` (new), `verify_runner.py` (`_dispatch`, `run_task_verify`, no-verdict → RU, spec-agnostic module rebuild, attempt-parametrised archive names, `probe_clean` pattern), `verify.py` (`is_no_verdict_summary`, three lines), `orchestrator/tests/test_task_verify_spec.py` (new), `test_verify_runner.py`]. Signal: scenarios 1, 3, 4, 19 pass against a fake `ssh_run`; `run_merge_verify`'s existing tests are unchanged. Unlocks β, γ. Note: `verify_runner.py` is also in 3310's and 6565's footprints and `verify.py` in 5651/5628/5139's — lock-serialised, no dependency edge needed.
- **β — `orchestrator verify-task` (no hub; dispatchable, with one ratchet raise).** [~500 LOC; files: `cli.py` (prologue extraction, new subcommand), `git_ops.py` (`create_ephemeral_verify_worktree`; in the ratchet cluster, so `merge_lane_ratchet_baseline.json` + `merge_lane_ratchet_authorized_raises.json` move in the same commit), `verify_cancel.py` (worktree-path file beside the pgid file; `cancel_request` reaps it; the probe pattern constant), `test_cli.py`, `test_verify_cancel.py`]. Depends α. Signal: scenarios 2 and 24 execute the real subcommand through `CliRunner` on a fixture repo, and `verify-merge`'s nine existing CLI tests pass unchanged after the extraction. Unlocks ι.
- **γ — Host pool and dispatcher (no hub; dispatchable).** [~900 LOC; files: `verify_hosts.py` (new; `build_remote_runners` moved in, `HostPool`, `Refusal`), `task_verify_dispatch.py` (new; dispatcher, `TaskVerifyPort`, `DispatchDecision`, sidecar writer, `eligible`), `merge_lane/drift.py` (import from the new home), `orchestrator/tests/test_verify_hosts.py` (new), `test_task_verify_dispatch.py` (new), `test_merge_queue_multihost_wiring.py` (import)]. Depends α. Signal: scenarios 5, 6, 8, 10–15, 21–23, 25 pass with fake runners and fake hooks through the real `HostAllocator`; scenario 9's tracker half waits for δ1. `worker.py`'s import stays in place (it imports the name from `drift.py`, which imports and uses it from `verify_hosts`; the `reexport_names` measure is structural, so this hop is not a shim). Unlocks δ1, δ2.
- **δ1 — Harness-owned allocator, injected (HUB: `harness.py`, `merge_lane/worker.py`).** [~250 LOC; plus `test_harness.py`, `test_merge_queue_host_observability.py`, `test_merge_lane_alias_names.py`, `merge_lane_ratchet_baseline.json` + authorized raise]. Depends γ. Hand-carry stage 1. Signal: scenarios 20, 7 (`held_by` in the snapshot) and 9 (task-path RU enters the tracker and is re-admitted by the sweep); worker's direct import of `build_remote_runners` moves to `verify_hosts`; `_ensure_host_allocator` returns the injected object; `note_runner_unavailable` / `note_runner_recovered` are public. Unlocks δ2.
- **δ2 — Workflow consumer and config (HUB: `workflow.py`, `config.py`).** [~450 LOC; plus `test_workflow_factory.py` keyword set, `test_workflow_verify_infra_resume.py`, `test_config_reload.py` or its equivalent, `OPERATIONS.md` config-reference section, `SETUP.md` §12 laptop section, `dark-factory-orchestrator.yaml` comment block only (value stays `local`)]. Depends δ1, 6580. Hand-carry stage 2. Signal: scenarios 16, 17, 18, 23; `reload_config` applies the four keys; with `verify_task_host_policy: local` every verify is byte-identical to today, `data/verify-logs` included. Unlocks ι.
- **ι — Integration gate (no hub; dispatchable).** [~400 LOC; `orchestrator/tests/test_task_verify_integration.py` (new)]. Depends β, γ, δ2. Signal: all twenty-five §6 scenarios execute in one module against the real `Harness` + `SpeculativeMergeWorker` + `TaskWorkflow` with the fixture-repo CLI wired behind `RemoteRunner`'s `ssh_run` seam (the "remote" is the real `verify-task` subcommand run in-process). Unlocks ε.
- **ε — DF rollout, parity window and the post-6580 baseline (operational gate, `execution_class: operational`, `task_kind: deterministic` where its checks are mechanical).** Depends ι, 6565, 6127, 3310, 6648, and 6580 (baseline). Steps: (1) confirm the laptop yaml values in §5.7, that the drop-in covers a task scope (start one by hand), and the laptop's root free space; (2) hand-run one `verify-task` through the production dispatcher on a scratch task branch (the 6127 confirmation shape) and read its sidecar; (3) record the baseline — the first full week after 6580 is live — for DF: archived `slot_wait_secs` p50, verify-phase residence p50, orchestrator-leg local wall p50, from `data/verify-logs` and runs.db; (4) `reload_config` with `verify_task_host_policy: remote_modules`, `verify_task_remote_modules: [orchestrator]`, `verify_task_remote_red_confirms_locally: true`, `verify_task_remote_cross_check_every_n: 5`; (5) after seven days report Leo's three trip-wires — laptop wall per orchestrator leg vs local, RU/no-verdict rate (sidecars with `reason in {runner_unavailable, infra_transient, host_parked}` over eligible verifies), and parity disagreements in both directions (every remote red via `red_confirmed_locally`, every n-th green via `cross_check`) — plus the two residence figures, drift-check deferrals, and laptop free space against the baseline; (6) decide, for Leo's ruling: cross-check to 0 and red-confirm off if disagreements are zero (5416 precedent), or hold at `local` and file the finding; and whether the report shows overload (wall per leg rising against local, RU/no-verdict rate rising), in which case flip η from `deferred` to `pending`. Signal: a dated note under `plans/merge-lane-throughput-prd.measurements/` with those figures, and the DF yaml carrying the chosen policy. No numeric threshold is asserted in advance (G6): the decision is Leo's on the report.
- **η — Binary pressure hold for task legs (HUB: `config.py`; hand-carry).** [~300 LOC; `verify_hosts.py` (probe loop, `PressureReading`), `task_verify_dispatch.py`, `config.py`, tests]. Filed `deferred` with an `x_deferral` naming ε's report as its flip condition; ε step 6 is the only thing that flips it. Signal: with threshold 60 and a cached reading of 70 younger than `max_age_secs`, `acquire_for_task` returns `Refusal('pressure_hold')` and the verify runs local; a reading older than `max_age_secs` yields no hold and `pressure_hold_stale` in the sidecar; merge dispatch never consults the reading.
- **Companion corrections (filed at decompose as amendments, not leaves):** amend 6106 per §8; note on 6127 that the task spec already carries the preprovision command; note on 5651/5409 that `workflow_verify.runner` is added by δ2; file the reify adoption task on reify with an external dependency on `dark_factory:ε`.

Dependency DAG: α → β; α → γ; γ → δ1 → δ2 (δ2 also ← 6580); β, γ, δ2 → ι; ι, 6565, 6127, 3310, 6648, 6580 → ε; η deferred, flipped by ε step 6 (no status edge). Decompose added β → γ and ι → η; §14 gives the reasons.

## 10. G7 walk (advisory, author mode)

- INV-1: the leg class is typed config with a validator, not prose. INV-2: every fallback carries a structured `reason` and the RU kind. INV-3: the spec's `tip_sha` is read at dispatch and is the pushed ref's target; the result is for that SHA by construction (immutable ref). INV-4: RU and stale-sync storms enter the merge lane's RU tracker through the pool's hooks, so the existing streak and `verify_host_unreachable_escalate_after_n` ladder hear them; a `merge_demand`/`no_host` storm is not a failure but is visible as the sidecar `reason` distribution ε reports. INV-6/7: a task hold has an owner (the dispatch coroutine), a bound (the laptop's own verify timeout plus the heartbeat watchdog), a release on every exit (`finally`), and a surface (`held_by`); a failed cancel parks, which is countable as `host_parked`, and 3310 owns parked recovery. INV-8: no new loop-thread work; ssh runs through the existing heartbeat subprocess path; η's probe loop is its own task with a static cadence. INV-9: per-attempt home is the dispatch sidecar; per-green home is `workflow_verify.runner`; no third copy, and no other module's record is edited. INV-10: ι executes the CLI and the fakes; no text guards. INV-11: a fallback returns a real local verdict and is distinguishable by the sidecar's `decision`; a stale pressure reading is recorded, not silently ignored. INV-12: no allow-lists. INV-13: ε's reader (the measurement note) names its producer (γ's sidecar) and the pre-flight in step 2 is the first production row. No waiver needed.

## 11. Out of scope

- Preemption of a task hold by merge demand (non-preemptive first; revisit with ε's data).
- Remote task legs for train workspaces (`force_workspace=True`), for `verify_failure_is_preexisting_on_main`'s main probe, and for the background role.
- A persistent per-task laptop lane (warm worktrees on the laptop); every remote task verify is cold.
- Shipping full laptop logs back (only the `VerifyResult` streams come back, as for merge).
- Reify and other projects' adoption (config shape is provided; their laptop yamls and clones are theirs; SCP needs Python 3.11 and a GitHub-sourced dep: S5 §2).
- Any host-wide or cross-project verify admission (RED-TIER), any load-derived slot or `-n` count, and an in-process second task slot.
- The verify event shape beyond `runner` (5651), the MemoryMax config field (6106), the merge spec's preprovision field (6127), PARKED recovery (3310), the deployed-SHA sync (6565).
- The merge-phase advisory verify site (deleted by 6580, never routed).

## 12. Open questions (tactical)

1. **Explicit cold flag on `run_verification`** instead of the `.task/` directory invariant. Suggested resolution: keep the directory in β; add the flag under whichever of 5653/5628 next edits `run_verification`'s signature. Decide at β.
2. **Ref namespace** `refs/merge-verify/` for task dispatches versus `refs/verify/<id>` for both. Suggested resolution: keep in α; rename only together with 4196's ref GC. Decide at α.
3. **`merge_lane_wants_host` and merge-ahead permits.** Whether a live build-ahead permit with no built item yet should count as demand. Suggested resolution: count it (yield early); measure refusals by reason in ε. Decide at δ1.
4. **Quarantine/PARKED survival across an in-process worker rebuild** (δ1 makes it process-lifetime). Suggested resolution: keep; `readmit` and restart are the exits, and 3310 adds the un-park path. Decide at δ1.
5. **Cross-check cadence and window** for ε (5 and seven days are the 5416 shape). Decide at ε.
6. **`apply_task_verify_spec`'s handling of host env keys absent from the spec** — 5496's rule (preserve) is assumed; confirm against the laptop's `effective_verify_env` on the first pre-flight. Decide at ε step 2.
7. **Whether `verify_task_remote_red_confirms_locally` stays on after the parity window.** On: every laptop false red is caught before the debugger, at the price of one local long leg per remote red (most verifies in a debug loop are red). Off: the trip-wire is sampled only by the cross-check. Suggested resolution: off after a zero-disagreement window, as for 5416; Leo rules on ε's report.

## 13. META check

If decomposed and queued without further oversight: every mechanism has a named consumer (the dispatcher consumes the spec, transport and pool; the workflow consumes the dispatcher; the merge lane consumes the injected allocator unchanged); every leaf names an executable signal that is load-independent; the substrate was verified symbol by symbol; the two seams are contracted with signatures and invariants and faced by twenty-five boundary scenarios one gate executes; the hub carries are two chained stages with their ratchet and tripwire costs named; the rulings against host-wide admission and load-derived counts are honoured by construction; and the rollout is a measured, reversible config flip gated on the four prerequisites that each remove a strand path. Yes.

## 14. Decompose record (2026-10-09)

Decomposed by a sibling `/prd` decompose session against main `f188b4b568`. `orchestrator/src`
had not changed since the verify SHA `00725ff2b6`. The G1, G3, G4 and G7 re-walks found no
drift, and every substrate symbol §7 names still exists. Manifest:
`plans/remote-task-leg-verify-prd.capability-manifest.md`, with its stamped YAML twin beside it.

| Label | Task | Status at filing | Prereqs |
|---|---|---|---|
| α | 6667 | pending | — |
| β | 6668 | pending | α |
| γ | 6669 | pending | α, β |
| δ1 | 6670 | pending (hub carry, stage 1) | γ |
| δ2 | 6671 | pending (hub carry, stage 2) | δ1, 6580 |
| ι | 6672 | pending | β, γ, δ2 |
| ε | 6673 | pending (operator gate: execution_class operational) | ι, 6565, 6127, 3310, 6648, 6580 |
| η | 6674 | **deferred** (`x_deferral` names 6673 step 6) | ι |
| reify adoption | reify:8430 | pending (operator gate) | external `dark_factory:6673` |

Companion amendments applied in the task store:
- 6106 was re-scoped to the memory config field and drop-in retirement. Its no-verdict
  classification moved to α. Its new field must express MemoryHigh as well as MemoryMax and
  MemorySwapMax, per the 10-09 throttle ruling.
- 6127 now notes the shared preprovision rule, and that ε depends on it.
- 5651 and 5409 now note that δ2 adds `runner`. 5651 is told that a remote leg's slot wait
  belongs to the laptop.

Re-scopes made at decompose. None changes a §4 decision.

1. **γ does not read config keys that δ2 produces.** As written, γ's `eligible(config, …)`
   read the `verify_task_*` fields, which land in δ2, downstream of γ. That is a
   producer-downstream G6 fail, and pyright would reject the attribute reads.
   - γ owns a frozen `TaskVerifyPolicy`.
   - `eligible()` takes it.
   - The dispatcher reads it per call via an injected `policy_of(config)` (§5.5 updated).
   - δ2 adds the keys and `policy_from_config`. It therefore also constructs the dispatcher
     in `Harness.__init__` and adds `task_verify=` at the harness `build_workflow` call
     (§5.6 updated). `build_workflow`'s kwarg is δ2's, so that edit could never have been δ1's.
   - δ1 keeps the quarantine set, the allocator, the pool and the worker injection.
2. **New edge β → γ.** β (`git_ops.py`), γ (`merge_lane/drift.py`) and δ1 (`worker.py`) each
   regenerate `orchestrator/tests/merge_lane_ratchet_baseline.json`, and
   `test_merge_lane_ratchet.py::test_baseline_matches_a_fresh_measurement` demands an exact
   match. The overlay's same-file rule serialises them, so the chain is α → β → γ → δ1 → δ2.
   β and γ are no longer parallel.
3. **Rows split by producer.**
   - Row 3: transport half α, CLI exit-1 half β.
   - Rows 9 and 12: hook call γ, RU-tracker half δ1.
   - Row 13: dispatcher cancel order γ; the workflow `SOFT_CANCELLED` half needs δ2's
     wiring and is asserted by ι.
   ι joins all of them.
4. **Two single homes.**
   - The `verify-(merge|task)` probe-pattern constant lives in α's `verify_runner.py`.
   - `runner` is read from `TaskVerifyDispatcher.decision_for(task_id, attempt_id)`, never from
     a shared "last decision" slot, because up to `max_concurrent_tasks` workflows share one
     dispatcher.
5. **ι → η.** η depends on ι, so a premature flip cannot dispatch it before the dispatcher
   exists. This is not the ε status edge §4.11 forbids.

Delivered checks: 16 mechanical checks (`grep`/`path`) were copied by `commit_planning` onto
α, β, γ, δ1, δ2 and ι. Each was linted absent on main before filing. ε's and η's checks are
`manual`.
