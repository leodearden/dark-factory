# S8 — Harness composition root and slot registry (PRD)

**Status:** draft, authored 2026-11-01 by the S8 seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S8, agent mode, wave 2, size L, risk
high, ruling R6); critic findings folded 2026-11-02 (two passes); lead rulings of 2026-11-02 applied (OQ-4, OQ-6, OQ-10, S3 OQ-7, the S8↔S9 `_on_stranded_merge_done` conflict and its edge direction); critic third pass folded; the rulings of the wave-2 sitting of 2026-11-10 applied (§Rulings) and every anchor re-checked at today's main; fold-verified 2026-11-10 (the lead's sequencing rulings of 2026-11-10 applied, the cross-leaf collaborator seams of ε1a…ε3 and the leaves' test-file reach made explicit). B+H: contracts C1–C7 (with C3b) and a two-way boundary-test
sketch. Twenty-two leaves including the integration gate ω, none held (OQ-10 ruled (a)): the sitting
added η (the `convert_to_blocked_enforce` default flip) and ι2 (the `EventStore` read surface and its
guard), and κ split into κ1/κ2 when its declared files re-measured at 18. Remediates run
`review-all-dark_factory-20261008`.

**Code anchors** verified against main `e634cfe658` (2026-11-10). The run's anchors were verified at
`6551b75aba`; the author pass at `ce0f71a384`, the critic at `0935e63d88`, the fold at `d40ba048e9`
(2026-11-02). Main moved 2,532 commits from `d40ba048e9` to `e634cfe658`; 41 of them touched
`harness.py` (+1,554/−1,100). The ones this plan builds on: the startup-step runner (task 8267:
`Harness.run` drives its startup through `startup_steps.py::StartupStep`, and crash recovery, the lane
checkouts, the stranded sweep and the orphan-worktree reaper gained per-entry bodies), inherited
streak alarms (task 4147: `_inherit_pending_streak_alarms`, `_seeded_recovery_veto_tracker`,
`_veto_streak_release_sites`), the scheduled self-deploy settle pass (task 7276), the harness-owned
`HostPool`/`TaskVerifyDispatcher` (task 6670, built in `__init__`), and two new readers of the sandbox
process global (`dry_run_unblock.py`, `evals/shadow_isolation.py`). No finding closed in that window.
Every count is labelled `measured at e634cfe658` unless it says otherwise; cite by symbol and re-locate
at implementation time. Re-checked at main `bb1c659562` (2026-11-10, fold-verify): `harness.py`,
`scheduler.py`, `task_ground_truth.py`, `steward.py`, `config.py`, `mcp_lifecycle.py` and
`sandbox_dispatch.py` are unchanged since `e634cfe658`; `event_store.py` changed in comments only; the
counts re-measured there (173 `event_store.emit(` sites, 17 raw events lines in 8 modules, 392
`Scheduler(` sites in 60 test files, 6 `is_workflow_active` code sites in `escalation/server.py`, 13
`_workflow_cancel_at` test files) are unchanged, and all 80 sidecar checks still fail. Counts labelled
`measured at bb1c659562` were first taken in the fold-verify pass. Re-stamp this header to the
then-current main at decompose.

## Goal

`Harness` becomes a composition root: it constructs every collaborator with every hook, owns one
registry of live slots, and delegates recovery, escalation filing and session-resume policy to classes
it builds. The next agent who adds a per-slot fact, a Harness-to-Scheduler hook, a recovery case, an
escalation filer or a background service edits one declared place, and a mistake fails at construction
or in a boundary test instead of leaking a slot, silently disabling a gate, or re-dispatching a task that
is still running.

What an operator observes when this lands:

- `release_workflow` on a task whose deterministic gate is still executing reports `released=False,
  was_active=True, slot_cleared=False, parked=None` and parks nothing, where today it parks the task as
  `blocked` while `DeterministicRunner.run` is still running (fk-c1ac3df8e44d). The escalation server
  (through S3 C2 `WorkflowSlots`, which `SlotRegistry` implements), the resolve path, the orphan-L0
  reaper and the terminal-status watcher answer "is a slot live for task X" from one registry.
- A `Harness` built without a dispatch safety gate does not start: `SchedulerCallbacks` requires both
  gates, so the silent `return set()` path for an unwired gate no longer exists.
- Shutdown tears down `usage_gate`, `cost_store` and the MCP lifecycle after every service that uses
  them, by registration order, and `Harness.run`'s `finally` holds no hand-ordered teardown.
- Both recovery sweeps (D15: `CrashRecovery` at boot, `StrandedReconciler` at startup and mid-run) take
  task state from `TaskGroundTruth.derive_truth`; the stranded sweep applies `recovery_for`'s answer
  verbatim; the four sweep-side rewrites of that answer are rows of the one table, and the scheduler's
  stranded-blocked phase (S17) can consume the same table.
- `harness.py` loses its recovery-sweep, escalation-filing and session-resume strata to classes that
  tests construct directly; `Harness.__new__(Harness)` disappears from `orchestrator/tests` (39 sites in
  19 files, measured at `e634cfe658`), and the 24 `getattr(self, '_…')` tolerance reads in `harness.py`
  (of 26 `getattr(self, '…')` reads) go with it.
- `EventStore.emit` stays a fire-and-forget call at its 173 `event_store.emit(` sites but no longer
  opens a connection and fsyncs on the event loop per event; a read through `EventStore` still sees
  every row emitted before it. Inside `orchestrator/src` every events read is an `EventStore` method:
  a guard test fails on raw `FROM events`/`JOIN events` SQL anywhere else in the package outside a
  named allow-list of offline tools (S8-3).
- An orchestrator whose event, run or cost store cannot be built does not start: it retries a locked
  database briefly, then files one critical born-at-L2 escalation naming the store, the path and the
  exception, logs ERROR and exits (S8-8). Today it runs on with the store unset.
- A task's steward spend is counted per task, durably, against a $25 cap in every project (today $12
  per steward instance, $20 in this repo); it is forgotten 7 days after the task's last steward spend
  (S8-1).
- The stranded sweep converts an escalation-pinned, unclaimed `in-progress` task to `blocked` in every
  project by default, and `convert_to_blocked_enforce` no longer exists (S8-7).
- An agent invocation carries its sandbox backend and its MCP session; nothing about an invocation's
  isolation or its fused-memory target depends on whether a `Harness` was constructed earlier in the
  process.

## Background

`Harness` (`orchestrator/src/orchestrator/harness.py::Harness`) spans lines 1789–19562 of a 19,562-line
file (248 methods, 9,171 prose lines = 46.9% by tokenize comments plus AST docstrings, measured at
`e634cfe658`; 19,108 / 224 / 47.8% at `d40ba048e9`). Three earlier PRDs built the seams this stream finishes: `plans/harness-supervision-prd.md`
(W10: `SchedulerCallbacks`, `LifecycleRegistry`, `TaskGroundTruth` and its `_RECOVERY` table),
`plans/worktree-lane-lifecycle-prd.md` (W11: the record-driven branch of `_recover_crashed_tasks`) and
`plans/scheduler-pause-halt-retirement-prd.md` (α 4686: `PauseClass`, which closes one of this
stream's keys outright). Each stopped at a boundary its tasks named "out of scope": the two private
Scheduler gates, the GitOps hooks, the `run()`-time writes, the sweep-side rewrites of the recovery
answer, and the `__new__` fixtures that production code bends around. Task 6660 (`SlotLease`, landed
`6d09554707`) added a seventh per-slot container and the first identity-guarded release path, which is
the shape the registry generalises.

Program rulings this PRD implements without relitigating: R6 (the stream ruling: option 1, both sweeps
consume `derive_truth` verbatim with condition rows), R23 (the two weakened keys re-verified in
§Findings disposition), R2 (S9 owns `stranded_verified_green.py`'s classification and filer; S8's
sweep consumes them), R5 (S5 owns `GitOpsCallbacks` and `LaneLifecycle`; S8 constructs), R10 (S3's port
exposes the registry to the escalation server), R16 (S17 consumes C2 and C3). R22 promotes no invariant
for S8; INV-8 `loop-thread-occupancy-bounded` is cited where it binds. The wave-2 sitting of 2026-11-10
ruled every open question of this plan (§Rulings); none of those rulings amends a 2026-10-10 ruling.

## Findings disposition

One row per key of the stream. `fixed:<sha>` names the commit on main that removed the cost; `stale`
names a premise that was false at the run's as_of; a `weakened` key is re-verified under G3/G6 before it
earns a leaf.

| Key | Sev | Verdict | Re-verified 2026-11-10 at `e634cfe658` (2026-11-02 at `d40ba048e9`) | Disposition |
|---|---|---|---|---|
| `fk-c1ac3df8e44d` two liveness containers | medium | confirmed | Still present, unchanged since `d40ba048e9`. `is_workflow_active` is `task_id in self._workflow_cancel_events`; `_run_slot`'s deterministic early return (`if self.scheduler.is_deterministic(...)`) precedes the `_workflow_cancel_events[...] = cancel_event` registration; `_escalation_events` is registered in `_launch_slot` before `create_task`. Task-keyed containers in `__init__`: 15 (task 6660 added `_slot_leases`). Task 6660 also added `_unregister_slot`, one release edge for 6 of them — the shape the registry generalises. `escalation/server.py` reads `harness.is_workflow_active` at 6 code sites (plus one comment). | leaf α |
| `fk-d8f08696dd66` six hook mechanisms | medium | **weakened** | Kept. The six install mechanisms stand: `SchedulerCallbacks` (12 hooks), `scheduler._landed_outbox_gate` and `._already_landed_gate` written after construction, 2 public GitOps attributes, 2 private GitOps attributes, `warm_lane_pool.set_on_lane_record_drift`, module-global `set_archival_failure_hook`. `run()` still writes `usage_gate._cost_store/_project_id/_run_id`, `review_checkpoint.cost_store/run_id/event_store`, `scheduler.event_store`; `_start_escalation_server` late-binds `escalation_queue` on the scheduler, the checkpoint and the provenance sink. The skeptic's point holds — both existing gates have wiring tests — so the cost is the six precedents the next hook copies, not a live silent gate. Under G3/G6 that cost is real and removable. The GitOps half (the 4 attribute installs, `set_on_lane_record_drift`, the 10 `git_ops._lane_lifecycle` reads, 9 at `d40ba048e9`) is delivered by S5 γ (C2 `GitOpsCallbacks`) and S5 ι (C6 `LaneLifecycle`) (OQ-10 ruled (a)). Task 6670's `HostPool`/`TaskVerifyDispatcher` are already built in `__init__` (no new post-construction write). | leaf β (GitOps half: S5 γ + ι, OQ-10 ruled (a)) |
| `fk-fc3c4cba6a0f` teardown order by prose | medium | **weakened** | Kept, narrowed, unchanged. `run()`'s `finally` is slot cancel → tallies → `finish_run` → WAL checkpoint → `usage_gate.shutdown()` → `cost_store.close()` → `_lifecycle.stop_all()` → `mcp.stop()` → straggler sweep, while merge-worker and watcher-supervisor still receive `usage_gate=`/`cost_store=`. Task 2142 made `UsageGate.shutdown` self-guarding, so the probe hazard is gone; the residual is a dropped cost row with a warning and an ordering that only prose states. Two live tasks now want to add steps to the same `finally` (7107 `Scheduler.aclose()`, 8559 `flush_state_snapshot`): a hand-ordered list grows a step per task. Cheap to remove inside β. | leaf β |
| `fk-b01dac97e6e9` pause class as a string prefix | medium | confirmed at as_of | **Gone.** Task 4686 (`730794ca3b`, authored 2026-10-22, committed 2026-10-23) landed `orchestrator/src/orchestrator/pause_policy.py::PauseClass`, the `scheduler_state.pause_class` column, `Scheduler.pause(reason, pause_class)`, and routes resume/restore/refresh on `STORED_EWA_PREDICATE_CLASSES`. `git grep "startswith('ewa_trip_')"` = 0 at `e634cfe658`; `run_store.refresh_scheduler_pause_ewa` filters `pause_class IN (...)`. `_ewa_trip_reason` remains as the display string only. The residual memory-first-wins/disk-last-wins divergence is owned by live task 4691 (pending). | `fixed:730794ca3b` |
| `fk-d0c02dc767dd` storm detector embedded, reach-back import | low | confirmed | Still present. `storm_window_bound.py` does `from orchestrator.harness import _BY_DESIGN_RESTORE_OUTCOMES`; `test_crash_recovery.py` imports the two private constants at 3 sites. The 8 session-resume methods γ moves (`_session_resume_reasons`, `_archive_available`, `_decay_session_resume_run`, `_retire_session_resume_run`, `note_resume_succeeded`, `note_resume_failed`, `_render_resume_failure`, `_file_session_resume_storm_escalation`) total 653 lines at both `d40ba048e9` and `e634cfe658` (the author's earlier "9 methods, 809 lines" is not reproducible from a named list and is superseded). Live task 5536 (pending, low) asks for this extraction. | leaf γ (subsumes 5536) |
| `fk-1e13b183c961` extractable strata | medium | re-observed | Still present. Name-partition of `Harness` methods measured at `d40ba048e9`: escalation filing 52 methods / 3,605 lines; recovery sweeps 31 / 3,528; session-resume 12 / 777; pause 8 / 585; slot 19 / 1,218; everything else 102 / 7,251 (24 methods added since, most of them per-entry bodies of the recovery strata). By AST, 31 `Harness` methods construct an `Escalation` or call `.submit(` at both shas (task 5516's generic `_file_harness_l1` replaced `_block_and_file_dispatch_gate_l1` in that set); ζ1a/ζ1b/ζ2 move 23 of them (§Decomposition plan lists every method per leaf). 39 test files call the recovery methods directly, 35 call the filers (measured at `d40ba048e9`). | leaf ζ1b (ζ1a, ζ2, ε1a–ε3 contribute) |
| `fk-b1c6965066ed` h14 measurement | medium | re-observed | Still present, grown: 19,562 lines, 9,171 prose (46.9%) by tokenize comments plus AST docstrings, 248 `Harness` methods (19,108 / 9,139 / 47.8% / 224 at `d40ba048e9`). | leaf ω (final re-measurement recorded in its task record) |
| `fk-d470c7828682` `__new__` fixtures and tolerance reads | medium | re-observed | Still present, grown: `Harness.__new__(Harness)` 39 sites in 19 files; `getattr(self, '…')` 26, of which 24 are `'_'`-prefixed (what ω's check matches; one more `_escalation_queue` read since `d40ba048e9`); `getattr(self.config, '…')` 5; five lazily-creating accessors exist for the fixtures (`_cancel_causes`, `_slot_lease_registry`, `_force_released_slot_tasks`, `_get_ground_truth`, and task 4147's `_seeded_recovery_veto_tracker`); 9 `__new__` comments in `harness.py` justify code shape by fixtures. | leaf ω (α–ζ2 remove the targets) |
| `fk-17b789fe3ce6` four rewrites of the recovery answer | high | re-observed | Still present, unchanged. `_reconcile_one_stranded` (861 lines) rewrites `recovery_for`'s action four times: LEAVE→MARK_DONE_WITH_PROVENANCE (blocked, on-main kinds, `report_pins_blocked_done_flip` false), LEAVE→RE_FILE_ESCALATION (blocked, EXISTS_OFF_MAIN, `report_pins_blocked_recovery` false), CONVERT_TO_BLOCKED→LEAVE when `report_blocked_arm_would_move`, CONVERT_TO_BLOCKED→LEAVE when `getattr(self.config, 'convert_to_blocked_enforce', False)` is false. Its MARK_DONE handler re-decides further (degenerate branch, `validate_landing_evidence`, a `failed` delivered-checks block; §D16, S8-6). | leaf δ |
| `fk-e6a18c90d2fe` crash recovery decides twice | medium | re-observed | Still present. Task 8267 split `_recover_crashed_tasks` into an 81-line driver and the per-entry `_recover_worktree_entry` (542 lines; 634 lines in one method at `d40ba048e9`); the two branches still decide separately: 6 comments say "mirror"/"mirrored" (7 at `d40ba048e9`), with 2 direct `self.scheduler.get_status(...)` and 2 `self.scheduler.get_task(...)` plus `read_worktree_title` reads beside `plan.json`. | leaf ε1a |
| `fk-bf4197a9a7d9` ground truth re-derived by sweeps | high | re-observed | Still present. Harness defines `_recover_crashed_tasks`, `_reconcile_stranded_in_progress` (and task 8267's `_reconcile_stranded_at_startup` entry), `_reconcile_one_stranded`, `_reap_orphan_worktrees`, `_reconcile_lane_checkouts`, `_reconcile_terminal_lanes`, `_recover_stranded_deterministic_task/_gate`, `_run_deterministic_recon_sweep` plus the merge-lane set; `recovery_for`/`derive_truth` have one production caller (`_reconcile_one_stranded`). | leaf ε2a (δ, ε1a, ε1b, ε2b, ε3 contribute) |
| `fk-5cad48d24b6f` no transition authority | high | re-observed | **Half `stale`, half present.** The legality half was a false premise at the run's as_of: `shared/src/shared/task_transitions.py::TRANSITIONS` (`cb99e2c70e`) and `ActorClass` (`8b3fbf5bc0`) are both dated 2026-07-07, before `6551b75aba`, and enforced at the fused-memory interceptor (task 2175; `task_status.enforce_transitions: true` in `fused-memory/config/config.yaml`); S10-3 (ruled 2026-11-10) moves that enforcement into the backend only, which leaves this disposition unchanged. The present half — dead-end `in-progress`/`blocked` rows recovered by sweeps that override the classifier — is what δ and ε2a remove. | legality half `stale`; recovery half leaf δ |
| `fk-5d0a66a2d32e` Harness↔Scheduler private seam | medium | re-observed | Still present (weakened), unchanged: the 9 `_on_*` assignments are gone (task 2235); `scheduler._landed_outbox_gate`/`._already_landed_gate` written after construction; 3 reads of `scheduler._module_cache`; the `_workflow_cancel_at` property/setter shim over `scheduler._workflow_cancel_at` (the same 13 test files use it); `event_store` and `escalation_queue` attribute-injected after construction. | leaf β (shim: β0) |
| `fk-a1085af3a290` steward lifetime budget is per instance | medium | re-observed | Still present. `steward.py::TaskSteward._remaining_lifetime_budget` returns `config.steward_lifetime_budget - self.metrics.total_cost_usd` (per instance); task 3566's amendment measured 114 of 499 tasks with more than one instance. Default `12.0` in `config.py::OrchestratorConfig.steward_lifetime_budget` and `defaults.yaml`; this repo's yaml sets `20.0`; no other project yaml sets it (S8-1 discussion, 21 yamls read 2026-11-10). | leaf θ (S8-1 ruled A; 3566 cancelled) |
| `fk-f1a7f57fa58e` `EventStore.emit` connects per event | medium | re-observed | Still present. `EventStore.emit` calls `self._connect()` (`sqlite3.connect` + `apply_full_durability_pragmas_sync`), INSERTs, commits, closes, per call; 173 `event_store.emit(` sites in `orchestrator/src` (167 at `d40ba048e9`). In-process readers through `EventStore` methods span 11 modules (unchanged); raw-SQL `FROM events` readers: 17 lines in 8 modules (§ι2). Live task 4443 (pending, low) asks for the writer; live task 4197 (pending, high) names this site as its finding 2 and recommends the same writer shape. | leaves ι (writer, subsumes 4443) and ι2 (read surface and guard, S8-3) |
| `fk-a97501641592` sandbox backend as a process global | low | confirmed | Still present, wider: one writer (`Harness.__init__` → `set_backend`); readers `invoke.py` (2 deferred imports), `workflow.py` (1), and since `d40ba048e9` `dry_run_unblock.py::_investigation_sandbox_modules` (`get_backend`, `resolve_active_backend`, `resolve_backend_or_refuse`) and `evals/shadow_isolation.py` (`resolve_active_backend` as a default argument); two process globals; two autouse `_restore_sandbox_backend` fixtures (`conftest.py`, `orchestrator/tests/conftest.py`); 11 test files call the process-global API (8 at `d40ba048e9`). | leaf κ1 |
| `fk-538940c8d347` `mcp_call` ignores its url; `rstrip` misparse | low | confirmed | Unchanged: `mcp_lifecycle.py::_session` module global set by `McpLifecycle.start`, cleared by `stop`; `mcp_call` uses `url.rstrip('/mcp').rstrip('/')`; callers `scheduler.py::Scheduler.dispatch_tool` and `agents/memory_recall.py::MemoryRecall._call_tool`. Out-of-harness consumers rely on `mcp_call`'s per-call fallback: `cli.py` (`Scheduler(config, override_store=…)`, `ReplayFramedBriefingAssembler`), `evals/runner.py` (×2), `evals/architect_eval.py`, `evals/reviewer_trial/runner.py`. | leaves μ1, μ2 |
| `fk-58d1f3489c98` backends write into the worktree | low | confirmed | Narrowed, unchanged (`agents/invoke.py` untouched since `d40ba048e9`): `_invoke_codex` writes nothing under `cwd`; `_invoke_gemini` still writes `GEMINI.md` and `.gemini/settings.json` at the worktree root; `_invoke_pi` still renames an existing `.mcp.json` to `.mcp.json.pi-invoke-backup`. Root `.gitignore` covers `.mcp.json` and `.codex`, not `GEMINI.md`/`.gemini`. Neither CLI is installed on this host, so their config levers are unverifiable here. | leaf κ2 |
| `fk-9854bd9a928f` claude spawn recipe duplicated | medium | re-observed | Still present, unchanged: `_invoke_claude_with_sandbox`'s sandboxed branch builds env, calls `_run_subprocess` and `_parse_claude_output` itself (importing the private names), with two "lockstep obligation" comments and no `VllmBridge`; `shared.cli_invoke.invoke_claude_agent` already accepts `sandbox_wrap`. | leaf κ2 |

## Pre-conditions for activating

- **S5 `GitOpsCallbacks` (C2) / `LaneLifecycle` (C6)** (G4: S5 owns). OQ-10 is ruled (a)
  (lead, 2026-11-02; S5 OQ-4 = A): S5 γ and ι perform the minimal harness-side GitOps wiring; S8 has no
  λ; β depends on S5 γ and S5 ι for the shared `Harness.__init__` site and re-homes their construction
  into the composition root.
- **S9 contracts** (G4: S9 owns): C1 `Plan`/`PlanState` (S9 β) read by ε1a; C5
  `classify_merge_future`/`file_stranded_merge_failed` (S9 θ) consumed by ε2b's `StrandedReconciler`;
  C6 `corroborated_landing` (S9 ι) called by δ's `_resolve_branch_state` journal rung; C7
  `RestoreOutcome`/`ResumeFailure`/`ResumeOutcomeSink` (S9 ζ1, module per S9 OQ-8) imported by γ. Each is
  a dependency edge if the S9 leaf is filed first, else an xref; whichever of the pair is filed second
  takes the edge — except S9 θ, which is one-way: ε2b depends on S9 θ (lead ruling 2026-11-02; S8 is filed second and takes the edge): S9 θ re-points `Harness._on_stranded_merge_done` onto C5 while the method still lives in `harness.py`, and ε2b then moves the already re-pointed method into `StrandedReconciler`. γ and ε1a are written for both orders (C6, C7): each consumes the S9 contract when it
  has landed and otherwise carries today's value forward for the S9 leaf to replace in the new module.
  The `_on_stranded_merge_done` ownership is ruled in S8's favour (lead, 2026-11-02): it moves into
  `StrandedReconciler` (ε2b, after S9 θ) and S9's consumers import from `stranded_reconcile.py`.
- **S6 C4 `LandedOutbox` (γ1, γ2), C6 `LaneTimingBudgets` (α), `LaneLimits` (κ)** (G4: S6 owns, S8
  constructs): S6 makes the construction edits at today's sites, including
  `TaskGroundTruth(..., landed_outbox=outbox)` in γ1; β re-homes them; δ keeps the `landed_outbox`
  parameter S6 γ1 adds. S6 θ edits `harness.py` before S8's leaves (S6's shared-files table); β depends
  on it. Under S6-2 = B (2026-11-10) task 5040 (verify-dispatch extraction, pending with a half-built
  branch) finishes first and S6's lifecycle and limits tasks wait for it, so β's edge on S6 κ also waits
  on 5040.
- **S4** (same files): S4 β edits `review_checkpoint.py`, which β also edits; S4 ι3 adds a
  `reap_leftover_verify_scopes` call in `harness.py`; S4 κ builds `AdmissionWatch(host)` in `harness.py`
  with the one process host; S4 ξ edits what `_run_main_tip_sweep` calls
  (`run_main_tip_sweep`/`confirm_main_tip_failure_is_real`). `_run_main_tip_sweep` stays in `harness.py`;
  no S8 leaf moves it. Per-file order on `harness.py` (lead ruling, 2026-11-10): β depends on S4 κ, so
  S4's host wiring lands first and β re-homes it. Under S4-1 (2026-11-10) the process `VerifyHost` is
  resolved once, by `VerifyHost.process()`, at the boundary objects the harness builds; the composition
  roots obtain it through `VerifyHost.process()`, never `VerifyHost.fresh()` (S4 D8), and production code
  outside the composition roots never calls `VerifyHost.fresh()` (S4 guard 2): β keeps the one
  `VerifyHost.process()` call in the composition root and never moves it into a collaborator.
- **S3 C2 `WorkflowSlots`** (`orchestrator/mcp/ops_tools.py`; S3 OQ-7 ruled (a), lead 2026-11-02):
  S3 owns the Protocol `WorkflowSlots(is_active, cancel, wait_cleared)`; `Harness` implements it until α
  lands, then `SlotRegistry` implements it, `release_workflow` awaits `slots.wait_cleared`, and
  `LadderLookups.workflow_active` binds to `slots.is_active`. S8 defines no `WorkflowSlots` of its own.
  α deletes `Harness.cancel_workflow` (S3 δ already moves `release_workflow` onto `slots.cancel`) and keeps
  `Harness.is_workflow_active` as a one-line `self._slots.is_active` forwarder, because the pin
  classifier's read in `escalation/server.py` (`_annotate_pins_recovery`'s `harness.is_workflow_active`)
  is rebound onto `LadderLookups.workflow_active` only in S3 ζ. α depends on S3 δ (which lands C2), and
  β on S3 β (C4 `set_resolve_callback(..., deliver_on=loop)`) and S3 δ — none of them held under S3-1 = A.
- **The S3-1 / S8-2 cycle and its forced cut (2026-11-10).** S3-1 = A reverses task 4651's edge: S3's
  first merge-tool move (S3 ε1) now depends on 4651 (4651 lands first). S8-2 = A makes 4651 depend on
  ε2a. S3 ε1 → ε2 → ζ is S3's serial chain, so an S8 edge from β (or any leaf at or before ε2a) onto S3
  ε1, ε2 or ζ closes the cycle S3 ε1 → 4651 → ε2a → … → β → S3 ζ → S3 ε2 → S3 ε1. β therefore does
  NOT depend on S3 ε1, ε2 or ζ: β edits `_start_escalation_server` first (guards, late-binds), S3 ζ
  later turns it into its composition call on top of β's result, and S3 ε1 wires the native lane-query
  class into the composition root β built. The `Harness.is_workflow_active` forwarder stays until S3 ζ
  rebinds the pin classifier; ω (which depends on S3 ζ) deletes it. The lead CONFIRMED this cut and
  ruled the per-file order on `harness.py` (lead sequencing rulings, 2026-11-10): S3 ε1 depends on S8
  ε2b, and S8 ε3 depends on S3 ζ, so the serial order reads … ε2a → ε2b → S3 ε1 → S3 ε2 → S3 ζ → ε3 →
  ζ1a …; S8's three high leaves (δ, ε2a, ε2b) never wait on the S6 native-port chain, and ε3…ω do
  (through S3 ε1/ε2/ζ on S6 μ, S7 δ and task 5409).
- **Live tasks on the same files.** 7277 (high, pending; `task_ground_truth.py`, `harness.py`,
  `recovered_merge_failure.py`) and 6615 (in-progress since 2026-11-01, claimed and heartbeating on
  2026-11-10; `task_ground_truth.py::_pid_alive`) are dependencies of δ. β depends on 7247 (pending;
  `run()`'s shutdown step 4a), 8559 (pending high; adds `flush_state_snapshot` to the same `finally`),
  8268 (in-progress; merge worker built at startup step 1b) and 8269 (pending high; the crash-loop
  breaker around `Harness.run`'s startup block), on 8926 (pending low; the one file-or-refresh idiom β's
  store-failure L2 uses) and, by S8-8, on the watchdog crash-loop alarm 8270 (pending high). 5819 (high,
  pending; `steward.py`) is a dependency of θ. 8563 (pending high, planned 2026-11-10; `digest.py`'s
  done-in-window count) is a dependency of ι2. 4651 depends on ε2a (S8-2 = A). Until ε2a is filed,
  4651 is held by a TEMPORARY parking edge on task 5040 that the lead added on 2026-11-10 (read on
  4651 at `bb1c659562`: its dependencies include 5040); S8's decompose seat adds 4651 → ε2a and then
  removes 4651's parking edge on 5040, nothing else (lead sequencing ruling).
- No novel substrate: `SchedulerCallbacks`, `LifecycleRegistry`/`ManagedService`, `TaskGroundTruth.recovery_for`,
  `recovery_pins`, `PauseClass`, `PersistentMap`, `SlotLease`, `cli_invoke.invoke_claude_agent(sandbox_wrap=…)`,
  `shared.task_transitions.TRANSITIONS`, `already_landed_tick_view.py::AlreadyLandedGate`,
  `startup_steps.py::StartupStep` (task 8267) and `born_at_l2.py::file_deduped_born_at_l2` all exist and
  are wired on main (§Capability manifest). `LandedOutboxGate` does not exist; β defines it.

## Sketch of approach

**α — `SlotRegistry`.** New `orchestrator/src/orchestrator/slot_registry.py`. One frozen `SlotRecord`
per dispatched slot: `task_id`, `kind` (`WORKFLOW` | `DETERMINISTIC`), `modules`, `cancel_event`,
`escalation_event`, `slot_task`, `lease` (the task-6660 `SlotLease`, folded in with its wedged-slot
diagnostics), `dispatched_at`. `Harness._launch_slot` creates the task and registers the record in one
synchronous section — no await between `create_task` and `register`, so the resolve path running on the
loop can never observe a task without its record (today's "register before create_task" property,
restated for a complete record); a duplicate registration is handled by D1a. `_run_slot`'s `finally`
releases the record at today's unregister point (D2a); the slot task's done-callback releases only a
record that is still current (the leak guard for a slot cancelled before its first step);
`_force_release_wedged_slot` releases early through the same call. `SlotRegistry` implements S3 C2
`WorkflowSlots` (`is_active`, `cancel`, `wait_cleared`); the composition root constructs it with
`on_cancel_accepted=scheduler.note_workflow_cancelled`, so an accepted `cancel` still stamps the R3
grace window `release_workflow`'s park logic relies on. `Harness.cancel_workflow` is deleted and
`Harness.is_workflow_active` becomes a one-line `self._slots.is_active` forwarder kept until S3 ζ
rebinds the pin classifier's read (ω deletes it; §Pre-conditions, the S3-1 / S8-2 cycle); their in-harness callers (the terminal-status scan and
`_action_teardown_and_set_status`) call the registry. `hard_cancel_workflow` stays a `Harness` method
returning `bool`, and `Harness.hard_cancel_refusal(task_id) -> HardCancelRefused | None` gives the
reason to callers that want it; both call sites (the terminal-status scan and
`_action_teardown_and_set_status`) keep their `if result:` reading. `hard_cancel_workflow`, the
terminal-status scan, `_reap_orphan_l0_escalations`, `_on_escalation_resolved` and
`_modules_unlocked_by_live_slot` read the registry. Deterministic slots are registered like any other,
with `kind=DETERMINISTIC`; `SlotRegistry.cancel` refuses them (S8-4), and the terminal-status scan
skips `kind=DETERMINISTIC` records (today's behaviour: it iterates `_workflow_cancel_events`, which never
holds one), so a terminal row never cancels or force-releases a running gate or deploy; a wedged
deterministic slot is the deterministic recon sweep's concern (ε3). The terminal-status watcher's
poll streak becomes watcher state keyed by record (heuristic 5), not a Harness container. Deletes
`_escalation_events`, `_workflow_cancel_events`, `_workflow_slot_tasks`, `_slot_leases`,
`_workflow_cancel_causes`, `_terminal_cancel_counts`, `_task_modules`, `_force_released_slots`,
`_cancel_causes`, `_slot_lease_registry`, `_force_released_slot_tasks`, `_unregister_slot`, `_owns_slot`,
the dispatch-time done-callback leak guard, and `slot_lease.py` (merged).

**β0 — one test factory for `Scheduler`; the cancel-recency shim goes.** `SchedulerCallbacks` gains the
two gate fields (`landed_outbox_gate`, `already_landed_gate`, still defaulting to `None` here) and
`SchedulerCallbacks.passthrough_gates(**hooks)`; new `orchestrator/tests/_scheduler_factory.py::make_scheduler(config,
**overrides)` passes `callbacks=SchedulerCallbacks.passthrough_gates()` and is the one place β and μ1 add the arguments
they make required. A mechanical codemod moves the 392 `Scheduler(` constructions in 60 test files (measured
at `e634cfe658` by `git grep -cE '(^|[^A-Za-z_])Scheduler\(' -- orchestrator/tests`; 387 in 57 at `d40ba048e9`) onto the factory;
`cli.py` and `evals/runner.py` pass `passthrough_gates()` explicitly. The `Harness._workflow_cancel_at`
property/setter shim is deleted and its 13 test files are re-pointed at
`scheduler.note_workflow_cancelled`/`scheduler.workflow_cancel_recent`. No production behaviour change.

**β — composition root.** `Harness.__init__` mints `run_id` (or takes `run_id=`) and builds
`EscalationQueue` FIRST, then `EventStore`, `RunStore` and `CostStore` with no degrade (D4a, S8-8: a
store that cannot be built stops the process loudly), then `TaskGroundTruth` and S6's `LandedOutbox`,
`LaneTimingBudgets` and `LaneLimits` (re-homed from the sites S6 γ1/γ2/α/κ edit), and hands each
collaborator its hooks and stores at construction: `Scheduler(config, event_store,
callbacks=SchedulerCallbacks(..., landed_outbox_gate=, already_landed_gate=), escalation_queue=)`,
`UsageGate(config.usage_cap, cost_store=, project_id=, run_id=)`, `ReviewCheckpoint(config, mcp,
usage_gate, cost_store=, run_id=, event_store=, escalation_queue=)`, `ProvenanceConflictSink(queue)`.
The two gate fields become required (D3). `RunStore.start_run()` and `CostStore.open()` stay in `run()`
and lose their `try/except` degrade too. The queue's resolve/notify callbacks need the running loop (S3
C4 `set_resolve_callback(..., deliver_on=loop)`), so they are registered at the top of `run()` — the
one deliberate `run()`-time binding, named in C3's invariant. β edits `_start_escalation_server`
before S3 ζ does (the S3-1 / S8-2 cycle, §Pre-conditions): it deletes the queue construction, the three
late-binds and both `HAS_*` guards there and leaves the server start for S3 ζ to turn into its
composition call. `_build_lifecycle_registry` registers `usage_gate`, `cost_store` and `mcp` first as
`ManagedService`s, then the scheduler's own shutdown (`Scheduler.aclose()` and the
`flush_state_snapshot` call task 8559 adds), so `stop_all` tears the shared stores down last; `run()`'s
`finally` keeps only the slot drain (task 7247's bounded `asyncio.wait`, a dependency), the tallies,
`finish_run`, the WAL checkpoint, `stop_all`, the straggler sweep and the lock release. The module-global
`set_archival_failure_hook` becomes a constructor argument of the archive collaborator GitOps receives
(exact shape tactical; the invariant is that `Harness.__init__` writes no module global). The 3
`scheduler._module_cache` reads go through a `Scheduler` accessor. The `HAS_ESCALATION` and
`HAS_STEWARD` `except ImportError` guards are deleted (escalation is a declared workspace dependency;
INV-11), and the 28 function-local `from escalation.models import` lines are hoisted to the one
top-level import (measured at `e634cfe658`, unchanged; live task 6625's remedy). Task 6670's
`HostPool`/`TaskVerifyDispatcher` construction in `__init__` already has the composition-root shape and
stays. Subsumes live tasks 7698, 6625, 6768 (the `UsageGate` private writes) and 7107
(`Scheduler.aclose()` at shutdown).

**γ — `SessionResumeStorm`.** New `orchestrator/src/orchestrator/session_resume_storm.py` owning
`BY_DESIGN_SESSION_RESUME_REASONS`, `MAX_RECORDED_RESUME_FAILURES`, the streak state,
`note_resume_failed`/`note_resume_succeeded` (the `ResumeOutcomeSink` implementation, imported from
S9 C7's module per S9 OQ-8), decay/retire, the renderer and the storm L1 filer, as a class receiving
`(queue, config, event_store, clock)`. "By design" for a restore outcome is written for both orders
(C6): if S9 ζ1 has landed, γ reads S9 C7's `RestoreOutcome.is_by_design` and keeps no set of its own; if
not, γ moves today's set into `session_resume_storm.py` as `BY_DESIGN_RESTORE_OUTCOMES`, and S9 ζ1
replaces it there (as S9's ζ1 already anticipates). The same applies to `ResumeFailure`/
`ResumeOutcomeSink`: imported from S9's module if it exists, else from `workflow.py`. The session-resume reasons
predicate is a function of config, sidecar and config dir; `archive_available` is a
`SessionResumeStorm` method, because it sets the log-once latch `_archive_available_fault_logged`,
which becomes instance state of the class. `storm_window_bound.py` imports downward from it;
`harness.py` imports it; no session-resume constant is importable from `orchestrator.harness`. Not
`StormCounter` (task 5536's constraint: the semantics are a circuit breaker with reset-on-success).
Subsumes live task 5536.

**δ — one classifier, condition rows.** `task_ground_truth.py` replaces the dict keyed on a five-tuple
with an ordered rule table whose rows carry a column for every dimension of today's `_RecoveryShape`
plus a named condition (C2). The four sweep-side rewrites become rows and `recovery_for` returns a
`RecoveryDecision(action, leave_reason, rule)` that the sweep applies without inspection. Under S8-7
(option B) the log-mode rewrite is deleted together with `convert_to_blocked_enforce` itself: δ deletes
the field from `config.py` and this repo's `dark-factory-orchestrator.yaml` line, after η has flipped
its code default to True and the lead has removed the key from the eight other project yamls (D18); no
policy reaches the classifier. `TruthReport` gains `merge_in_flight: bool`
(task 7277's in-flight-merge ownership, filled by `derive_truth` from the constructor's
`merge_in_flight` callable, replacing the guard in `_revert_in_progress_if_no_live_claimant`) and, under
S8-6 (option a), the landing-evidence verdict the MARK_DONE handler re-decides today
(`validate_landing_evidence`, degenerate branch, delivered-checks verdict). `_resolve_branch_state`'s
journal rung calls S9 C6 `corroborated_landing` (S9 ι edits that rung first). `TaskGroundTruth`
keeps the `landed_outbox` parameter S6 γ1 adds (S6 C4). `recovery_for` accepts an already-fetched row (task 6682). The applier
in the sweep becomes a dispatch over `RecoveryAction` with one handler per action, each returning a
`HandlerOutcome` (D16). An equivalence test generated from today's effective outcome pins the table (D7).

**ε1a — `CrashRecovery`: the boot worktree scan as a class reading `TruthReport`.** New
`crash_recovery.py` (`CrashRecovery`: the boot worktree scan — task 8267's `_recover_crashed_tasks`
driver and its per-entry `_recover_worktree_entry` body, which keeps 8267's per-entry isolation —
`_adopt_recovered_session` and the orphaned-transcript sweep), returning a frozen `CrashRecoveryResult`
(task 7436). The four recovery maps are not read-only after boot: `_run_slot` pops a task's entries
when it dispatches it, `_mark_in_progress_done` pops them when the stranded sweep marks the task done,
and `_reap_orphan_entry` and `_revert_in_progress_if_no_live_claimant` test membership (read at
`bb1c659562`). So the composition root seeds ONE `crash_recovery.py::RecoveredWork` ledger from the
result (`claim(tid)` for `_run_slot`, `discard(tid)` for the done-writer, `holds(tid)` for the reaper
and the revert) and injects it into `Harness`, `LaneReconciler` (ε1b) and `StrandedReconciler` (ε2a);
until those leaves land, the moved readers receive the ledger and the Harness methods still in
`harness.py` read it through `self`. Task-state facts — db status, live claimant, branch state, open escalations — come from
`TaskGroundTruth.derive_truth(tid)` (D15, S8-5); the lane-local facts `TruthReport` lacks (durable
lane-record state, plan state, stored title) are a `LaneEvidence` value. Written for both orders (C7):
with S9 β landed, `LaneEvidence.plan_state` is S9 C1's `PlanState`; without it, `LaneEvidence` carries
today's plan.json read as `plan_steps` and S9 β re-types it. Adopt/preserve/clean/quarantine is decided
ONCE by `decide_lane(truth, lane) -> LaneDecision` (C7), so the record-driven and heuristic branches
collapse to two readers of one decision and the 2 direct `get_status` and 2 `get_task` +
`read_worktree_title` sites go. Task 3155's `_task_recovery_artifact` (also read by `_locate_plan_artifacts`,
which stays) becomes a module-level function of `crash_recovery.py` that `harness.py` imports downward.

**ε1b — `LaneReconciler`.** New `lane_reconcile.py` (`LaneReconciler`: `_reconcile_lane_checkouts`
with its per-lane body `_reconcile_one_lane_checkout`, `_reap_orphan_worktrees` with its per-entry body
`_reap_orphan_entry`, `_reconcile_terminal_lanes`, `_reclaim_terminal_lane_records`,
`_assigned_durable_records_with_statuses`, the lane-assignment census; 644 lines moved, measured at
`e634cfe658`; 598 at `d40ba048e9` before task 8267 split out the per-entry bodies). The orphan reaper
files and resolves the pool-storage-absent L1 (`_file_pool_storage_absent_escalation`,
`_resolve_pool_storage_absent_escalation`, which ζ1b moves later), so `LaneReconciler` receives
`file_pool_storage_absent` and `resolve_pool_storage_absent_if_pending` callables and the
`RecoveredWork` ledger (ε1a); the `_pool_storage_absent_maybe_pending` flag is the filer pair's state
and moves with it in ζ1b.

**ε2a — `StrandedReconciler`: the class and the action dispatch.** New `stranded_reconcile.py`
(`StrandedReconciler`: `_reconcile_stranded_in_progress` and task 8267's startup entry
`_reconcile_stranded_at_startup`, `_reconcile_one_stranded` as the action dispatch,
`_revert_in_progress_if_no_live_claimant`, `_mark_in_progress_done`, `_delivered_checks_block`,
`_RECONCILE_SWEEP_STATUSES`; 1,390 lines moved before δ's deletions, measured at `e634cfe658`).
`_mark_in_progress_done` and `_delivered_checks_block` are not the sweep's alone: the already-landed
dispatch gate (`_already_landed_dispatch_gate`, which stays in `harness.py`) calls both (AST at
`bb1c659562`), and 4651's reconciler must reuse the same gated write. ε2a therefore lands them as one
`stranded_reconcile.py::GatedDoneWriter` (the delivered-checks gate, then `scheduler.mark_done`, then
the `RecoveredWork.discard`), which the composition root constructs once and injects into
`StrandedReconciler` and `Harness`. Its final constructor (after ζ1b) receives `scheduler`, `git_ops`,
`escalation_queue`, `event_store`, `config`, `merge_queue`, `provenance_conflict_sink`, `ground_truth`,
`dispositions` (`RecoveryDispositions`), `done_writer` (`GatedDoneWriter`), `recovered_work`
(`RecoveredWork`), `landing_rejects` (the `LandingRejectRecorder` the gate and `_launch_slot` share),
`lane_reconciler` (its pass calls `reconcile_terminal_lanes`, ε1b), `escalate_reconcile_failure`
(`InfraFilers`, ζ1b), a `resolve_task_worktree` callable, a `branch_is_degenerate` callable,
`filing_claimant_run_id`, and a `file_stranded_merge_failed` callable — today's
`Harness._file_stranded_merge_failed` until S9 θ lands C5, then
`stranded_verified_green.file_stranded_merge_failed`. In ε2a, the disposition emitter, the
veto-streak release, the verified-green submit and `_escalate_reconcile_failure` are passed as Harness
callables that ε2b (the first three) and ζ1b (the last) replace. Task 4651 (S8-2 = A) is
re-scoped onto this class: a second `StrandedReconciler` instance over `pending`/`merge-deferred` whose
rows live in `RECOVERY_RULES`, with a test that no row's `statuses` intersect across the two
reconcilers (the 2026-08-24 partition, made a checked invariant).

**ε2b — `RecoveryDispositions` and the verified-green trio.** `recovery_emission.py::RecoveryDispositions`
holds `_emit_recovery_disposition` plus the veto tracker, streak memo and process latch
(`_recovery_veto_tracker`, `_recovery_streak_filed_at`, `_recovery_process_notices`, their helpers
`_seeded_recovery_veto_tracker`, `_recovery_streak_memo`, `_recovery_process_latch`,
`_clear_recovery_veto_streak`, `_release_recovery_veto_streaks`, `_rearm_recovery_process_notice` (it
writes the process latch; its callers are ε3's), and the veto half of task 4147's startup inheritance;
377 lines measured at `e634cfe658`, plus the 20-line `_rearm_recovery_process_notice` measured at
`bb1c659562`). `_inherit_pending_streak_alarms` seeds two memos from one `queue.get_pending()` read:
the zero-progress one (`_zero_progress_filed_at`, which stays with `_apply_retry_cap`, D8) and the veto
one. It splits: startup step `2c3-streak-alarms` reads the pending alarms once, seeds the zero-progress
memo in `harness.py` and calls `RecoveryDispositions.inherit_pending(alarms)` for the veto half; the
release sites `_veto_streak_release_sites` computes from `_deterministic_recon_sweep_runs()` (the
composition root's registration predicate, which stays) are a constructor argument, not a reach-back.
The composition root constructs it once and injects it into `StrandedReconciler`,
`DeterministicReconSweep` (ε3) and `Harness` (for `_escalation_hold`, which stays). The trio
`_maybe_submit_stranded_verified_green`, `_stranded_merge_marker_is_fresh` and `_on_stranded_merge_done`
(242 lines) moves into `StrandedReconciler`. ε2b depends on S9 θ (lead ruling 2026-11-02; S8 is filed second and takes the edge): S9 θ re-points `Harness._on_stranded_merge_done` onto C5 while the method still lives in `harness.py`, and ε2b then moves the already re-pointed method into `StrandedReconciler`; ε2b never precedes S9 θ, so the
`file_stranded_merge_failed` collaborator is C5's `stranded_verified_green.file_stranded_merge_failed`
from the start of ε2b.

**ε3 — `DeterministicReconSweep`.** New `deterministic_recon.py`: the deterministic recon sweep, the two
stranded-deterministic recoveries, deploy-health revalidation, with the unit inspector and
`RecoveryDispositions` injected by constructor (the unit-inspector half of live task 9030). The
`scheduled-deploy-settle` service (task 7276) stays a separate registration: its logic already lives in
`scheduled_deploy_settle.py`, and `_deterministic_recon_sweep_runs` stays the composition root's
registration predicate. The unit inspector is shared: `_run_scheduled_deploy_settle_pass` (stays) reads
it too, so the composition root builds it once and injects it into both. The sweep calls
`_revalidate_open_l2`, which ζ2 moves later, so ε3 receives it as a Harness callable that ζ2 rebinds
onto `EscalationRecordSweeps`. ε3 depends on S3 ζ (lead sequencing ruling 2026-11-10): S3 ε1, ε2 and ζ
edit `harness.py` between ε2b and ε3.

Harness registers each sweep class as a `BackgroundService` and calls the boot passes through today's
startup-step table (8267) in today's order.
Merge-lane recovery methods (`_recover_pending_merges`, `_reap_orphaned_merge_worktrees`,
`_reconcile_landed_outbox`, `_recover_abandoned_merge_deferred`) stay in `harness.py` (D14).

**ζ1a / ζ1b / ζ2 — filers as classes.** `dispatch_gate_filers.py` (`DispatchGateFilers`, ζ1a) and
`infra_filers.py` (`InfraFilers`, ζ1b), each constructed with `(queue, config, event_store, scheduler,
filing_claimant_run_id)` and exposing one method per filing; `escalation_record_sweeps.py`
(`EscalationRecordSweeps`, ζ2), reading liveness from the `SlotRegistry` (α) and the Scheduler. The
method lists are in each leaf. The pause filers `_file_scheduler_pause_escalation` and
`_file_restored_pause_escalation` stay with the pause stratum that live tasks 4687/4691 own; task 5516's
restart-recut stratum (`_issue_restart_recut_ruling`, `_reassert_restart_recut_ruling`,
`_file_restart_recut_unrecorded`, `_block_if_still_pending`) stays with the slot teardown it serves and
files through `DispatchGateFilers`' generic L1 method once ζ1a moves `_file_harness_l1`.
Resolution appliers (`_on_escalation_resolved`, `_escalation_hold`, `_action_teardown_and_set_status`,
`_cascade_unblock_member`, `_fold_granted_files`) stay in `harness.py`: they act on slots and the
scheduler, not on the queue. The filing-identity tripwire (`test_harness_escalation_filing_identity.py`)
is extended to the new modules, never lowered. ζ2 subsumes live task 7262.

**η — `convert_to_blocked_enforce` defaults to True (S8-7, step 1).** `config.py::OrchestratorConfig`
flips the field's default from False to True and rewrites its description: enforce is the shipped
behaviour, and log mode remains only for a yaml that sets `false` until δ deletes the field. This is
step 2 of the field's own promotion comment, never done. No behaviour change in any of the nine
projects, which all set `true`. Independent of the `harness.py` chain; it deploys on the normal
cadence, and its fleet-wide deploy is what releases the lead's yaml edits (D18).

**θ — durable steward spend, $25 per task (S8-1).** The steward's lifetime spend becomes a
`PersistentMap` under `guards/steward_lifetime_spend/<task_id>.json` (the task-5352 shape, same TTL:
`steward.py::_STEWARD_GUARD_TTL` = `timedelta(days=7)`; `guard_state.py::PersistentMap`'s TTL tracks
the last bump, so a task's spend is forgotten 7 days after its last steward spend), incremented at the
two `total_cost_usd +=` sites and read by `_remaining_lifetime_budget`; `StewardMetrics.total_cost_usd`
stays the per-instance report. In the SAME commit (D17): `config.py::OrchestratorConfig.steward_lifetime_budget`
and `defaults.yaml` go from `12.0` to `25.0`, and this repo's `steward_lifetime_budget: 20.0` line and
its 22-line comment are deleted from `dark-factory-orchestrator.yaml`; no other project sets the key,
so no other repository is edited. The budget-exhausted L1, the `config.py`/`defaults.yaml` comment and
`OPERATIONS.md` state "$25 per task, durable, forgotten 7 days after the last steward spend". Durable
spend starts from zero at deploy (no backfill). Subsumes live task 3566 (cancelled).

**ι — one event writer.** `EventStore.emit` enqueues onto a thread-safe queue (callable from the
FastMCP threadpool, where `escalation/server.py` calls it) and returns; one writer thread owns one
connection with the durability pragmas applied once and commits batches in one transaction (group
commit; `synchronous=FULL` kept, so one fsync per batch, not per event) (S8-3). FIFO order is
preserved. Read-your-writes is a property of `EventStore` itself (C5): every `EventStore` read method
first drains the pending queue, bounded on the loop thread by one batch interval plus one commit (and by
`flush`'s `timeout_secs` off the loop); a sync `flush(timeout_secs)` callable from any thread and an
`async aflush(...)` exist for shutdown and tests. No call site changes: the in-process readers through
`EventStore` methods — `escalation/server.py` (5 calls, sync tool bodies on the FastMCP threadpool),
`merge_lane/completion.py` (2), `merge_lane/disposition.py`, `merge_lane/worker.py`, `hold_history.py`,
`task_runtime.py`, `rebase_cost_readout.py`, `stranded_verified_green.py`, `verify_checkpoint.py` (2),
`shadow_eval.py` (which already calls `asyncio.to_thread(self._deps.events.emit, …)`) — are covered by
construction (11 modules, measured at `e634cfe658`). Until ι2 lands, the raw-SQL readers on their own
connections see lag of at most the batch interval, which C5 states. Shutdown drains through a
`ManagedService` that ι prepends to β's ordering so it stops last. Writer failure is loud (INV-11):
WARNING with the count dropped and the connection reopened. Subsumes live task 4443 and delivers
finding 2 of live task 4197 (whose recommendation (b) is this writer).

**ι2 — every in-process events read is an `EventStore` method (S8-3).** `digest.py`'s raw readers
(`query_count_done_in_window`, `merge_disposition_counts`, `chain_telemetry_in_window`,
`_compute_turn_cap_saturation`, `repended_during_current_pause`: 7 SQL lines) and
`strict_event_reads.py::task_ids_with_event` (1 line; read by `Scheduler._load_work_product` on a
worker thread) move onto typed `EventStore` read methods (about six: done count in a window, merge
disposition counts, chain telemetry, turn-cap saturation rows, repended-during-pause rows, task ids
carrying an event type), each draining first (C5). The digest calls them through `asyncio.to_thread`
from its coroutine, the pattern task 8563 sets for `landings()` (INV-8). A guard test fails on raw
`FROM events`/`JOIN events` SQL anywhere in `orchestrator/src` except `event_store.py` and a named
allow-list whose every entry states why: the offline tools `storm_window_bound.py`,
`resume_age_bound.py`, `analyze_modules.py`, `sandbox_soak.py` and `evals/reviewer_trial/mining.py`
(run by hand or from tests on their own connection; no read-your-writes need), and the operator recipe
`scheduler.py` prints into its retry-cap report (a string, not a read). Measured at `e634cfe658`: 17
matching lines in 8 modules, 8 of them in the two runtime readers. The "flush at named sites" writer
option is subsumed: draining inside the read method IS the named site. The cross-package `runs.db` read
model (dashboard, fused-memory, client feed, quality series, scripts) is a follow-up routed to the
dashboard-readers plan (S22), not a task here. Depends on ι (same file) and on task 8563 (same
`digest.py` function).

**κ1 — the invocation carries its sandbox backend.** `sandbox_dispatch` loses `set_backend`,
`get_backend`, `_preferred`, `_escalated_backend_state` and `reset_escalation_dedupe`;
`resolve_active_backend(backend)` and `resolve_backend_or_refuse(backend, dedup)` take the configured
backend and a `SandboxRefusalDedup` object that the composition root owns and `build_workflow` hands each
`TaskWorkflow`; `invoke_agent` gains `sandbox_backend` beside `sandbox_modules`. The two readers that
appeared after `d40ba048e9` take the backend the same way: `dry_run_unblock.py::_investigation_sandbox_modules`
reads `config.sandbox.backend` and the injected dedup, and `evals/shadow_isolation.py`'s
`resolve_backend` default binds the configured backend at its composition site. Both autouse
`_restore_sandbox_backend` fixtures are deleted. κ1 edits `_invoke`'s region after S9 ζ1.

**κ2 — non-claude backends write under `.task/`; one claude spawn recipe.** `_invoke_claude_with_sandbox`
collapses onto `invoke_claude_agent(..., sandbox_wrap=…, env_overrides=…)`, deleting the duplicated
env/subprocess/parse recipe and the private imports — sandboxed runs gain `VllmBridge` for free.
`_invoke_gemini` and `_invoke_pi` write per-invocation config under the gitignored
`.task/<backend>-<invocation>/` wherever the CLI accepts a path or env var; where a CLI insists on cwd
discovery (pi's `.mcp.json`), the backend refuses loudly when the target is tracked (`git ls-files
--error-unmatch`) instead of renaming a committed file. The three non-claude backends adopt
`shared.secret_env.agent_subprocess_env` (task 7311, subsumed).

**μ1 — the harness path carries its MCP session.** `McpLifecycle` owns its `McpSession` as an attribute
created at construction and initialised by `start()`; the module global `_session` and `get_session`
go. `Scheduler` receives the session at construction, now required (C3's `mcp_session` moves from
optional to required here); `cli.py` and `evals/runner.py` open their own `McpSession` in an
`async with` around the run, and β0's factory supplies a typed stub. `mcp_call(url, …)` becomes the pure
one-shot helper its signature promises, with `removesuffix('/mcp')`.

**μ2 — briefing and recall carry the session (D13a).** `MemoryRecall(session: McpSession | None)`:
`None` means recall is disabled and is logged once at WARNING. `BriefingAssembler` and
`ReplayFramedBriefingAssembler` receive the session (or `None`), not a url; the out-of-harness entry
points (`cli.py`, `evals/runner.py`, `evals/architect_eval.py`, `evals/reviewer_trial/runner.py`, with
`evals/replay_frame.py` defining the subclass) open and close their own `McpSession` in an
`async with` around the run.

**ω — integration gate.** The boundary-test matrix below runs green against real fixtures; the
`__new__` fixtures, `_init_harness_state_for_test`, the `getattr(self, '_…')` reads and the 9 comments
justifying code shape by fixtures are gone; the `Harness.is_workflow_active` forwarder α kept is deleted
(ω depends on S3 ζ, which rebinds the pin classifier onto `LadderLookups.workflow_active`); every test
that builds a real `Harness(config)` uses a tmp `project_root` (`EventStore.__init__` mkdirs and runs
`_ensure_schema` DDL against `runs.db`, and under D4a a construction failure is a raise, not a
degrade); `harness.py`'s heuristic-14 measurement is re-taken and recorded in the task record.

## Resolved design decisions

- **D1 — registration is complete and synchronous.** A `SlotRecord` is registered with its `asyncio.Task`
  already created, in the same synchronous section of `_launch_slot`; the record is frozen; the only
  mutable per-slot objects are the two `asyncio.Event`s and the lease's released flag. Rejected: a
  two-phase open/bind (a half-registered record is today's defect in a new coat).
- **D1a — a duplicate registration never dispatches.** On `SlotAlreadyLive`, `_launch_slot` cancels the
  just-created task before it runs (it has not started), releases the lease it acquired, logs ERROR
  naming both records (task id, kinds, `dispatched_at`), and returns without dispatching. The live
  record is untouched.
- **D2 — one release call, identity-guarded.** `SlotRegistry.release(record) -> bool` is the only
  removal; the slot body, the done-callback and the watcher's force-release all call it; a slot body
  checks `registry.current(task_id) is record` before acting on the task (task 6660's `_owns_slot`,
  kept). Deterministic slots are released the same way, so `registry.is_active` is true exactly while
  `DeterministicRunner.run` executes; `cancel` refuses such a slot (S8-4). The
  terminal-status scan skips `kind=DETERMINISTIC` records (today's behaviour), so it never cancels or
  force-releases a live gate or deploy; a wedged deterministic slot is ε3's concern.
- **D2a — release happens at today's point.** `_run_slot`'s `finally` calls `registry.release(record)`
  first, where `_unregister_slot` runs today — before `_apply_retry_cap`, `_maybe_auto_eval`,
  `_maybe_bump_routing_tier`, the claimant clear and `lease.release(...)` — so `registry.is_active`
  turns False at the same moment it does today. Its result travels with the slot's report: `False`
  means the watcher force-released the record first, and `_collect_slot_report` drops that late report
  (this replaces `_force_released_slot_tasks`). The done-callback calls `release(record)` only when the
  record is still current, i.e. the body never reached its `finally` (a slot cancelled before its first
  step, boundary row 2). Rejected: release in the done-callback alone (it runs via `call_soon` after the
  dispatcher woken by `sem.release`, so the slot would read live past `lease.release`).
- **D3 — required gates, named passthrough.** `SchedulerCallbacks.landed_outbox_gate` and
  `.already_landed_gate` are non-Optional. The bare-Scheduler tests that today rely on `None` use
  `SchedulerCallbacks.passthrough_gates()` through β0's factory, which returns gates answering "nothing
  gated" and says so in its name. A `Scheduler` cannot be built with a silently absent gate (heuristic
  10, *informatively* and *redundantly*: construction refuses, and the gate's own consult has no `None`
  branch to fail open).
- **D4 — stores built at construction, opened in `run()`.** `run_id` is minted in `__init__`
  (`run_id: str | None = None` kwarg for tests), so `EventStore`, `RunStore`, `CostStore` exist when
  collaborators are built. `CostStore.open()` and `RunStore.start_run()` stay in `run()`. Rejected:
  factories passed to collaborators (every consumer would hold a latent-None path, which is the shape
  this stream removes).
- **D4a — a store that cannot be built stops the process, loudly (S8-8, option A).** Today
  `Harness.run` wraps `EventStore(...)`, `RunStore(...)`/`start_run` and `CostStore(...)`/`open()` each
  in `try/except` and keeps running with the store unset (zero such failures in `/var/log/syslog` and
  its rotations back to 2026-10-11). Under the ruling, `Harness.__init__` builds `EscalationQueue`
  first (mkdir plus lazy caches, so it exists before any store can fail), then the three stores with no
  degrade. Only a transient lock is retried: an `sqlite3.Error` whose `sqlite_errorcode` is
  `SQLITE_BUSY` or `SQLITE_LOCKED` (a typed code, never a message match: heuristic 12) gets about three
  attempts over about 30 s; the sleep happens in the synchronous constructor, before any event loop
  exists (INV-8 is not engaged). Any other failure, or the last retry, files ONE born-at-L2 critical
  escalation on the fixed sentinel `store-unavailable-startup` naming the store, the path and the
  exception (file-or-refresh: a later start refreshes the pending record in place rather than filing a
  second, through the one idiom task 8926 establishes), logs ERROR and raises `StoreUnavailable`.
  `RunStore.start_run()` and `CostStore.open()` in `run()` lose their degrade the same way. Rejected:
  stand-in null stores (the plan's earlier recommendation; Leo: "fail loud and hard, don't degrade") and
  Optional stores (the latent-None shape). Fail closed is per project: a bad `runs.db` stops that
  project's dispatch and merges until a human fixes it. The raise happens before `Harness.run`, so the
  in-process crash-loop breaker (task 8269) does not see it; the watchdog's restart-rate alarm (task
  8270, a dependency of β by ruling) pages a unit that restarts every 10 s, and once deployed leaves a
  unit that hit its start limit stopped. Every test that builds a real `Harness(config)` uses a tmp
  `project_root` (ω).
- **D5 — teardown order is registration order.** `usage_gate`, `cost_store` and `mcp` register first
  as `ManagedService`s (`shutdown`, `close`, `stop`), so reverse-order `stop_all` tears them down after
  every service that holds them; ι later prepends the event writer before them. The scheduler's own
  shutdown (`Scheduler.aclose()`, task 7107's ask, and the `flush_state_snapshot` call task 8559 adds to
  the `finally`) registers next, so a new shutdown step is a registration, not a line in the `finally`. The hand-ordered
  `finally` statements are deleted, not kept as belt-and-braces: a second ordering is a second home
  (INV-9).
- **D6 — the escalation queue is a required collaborator, built first.** `EscalationQueue(queue_dir)`
  is a cheap construction (mkdir and lazy caches, re-read at `e634cfe658`); `__init__` builds it before
  the stores so a store failure can be filed (D4a). Both
  `HAS_ESCALATION` and `HAS_STEWARD` optional-import guards are deleted: `escalation` is a declared
  workspace dependency of `orchestrator`, so the guard only turns a genuine import error into a silent
  no-op ladder (INV-11; live task 6625, subsumed). `_run_deterministic_slot`'s
  `RuntimeError('_escalation_queue must be initialised…')` becomes unreachable and is deleted. The
  queue's callbacks are registered at the top of `run()` because S3 C4's `deliver_on=loop` needs the
  running loop.
- **D7 — rows, not rewrites (R6).** `_RECOVERY` becomes `RECOVERY_RULES: tuple[RecoveryRule, ...]`,
  first match wins, default `LEAVE`. Columns cover every dimension of today's `_RecoveryShape`
  five-tuple (`db_status`, live claimant, `branch_state.kind`, `_vetoes_done_flip(report)`,
  `deploy_phase`), with `None` meaning any; conditions are the existing named predicates in
  `recovery_pins` / `task_ground_truth` (`report_pins_blocked_done_flip`, `report_pins_blocked_recovery`,
  `report_blocked_arm_would_move`) plus `TruthReport.merge_in_flight` and (S8-6)
  `TruthReport.landing_evidence`. `report_would_duplicate_a_handoff` is not a row condition: it stays the
  RE_FILE handler's `deduped` outcome (D16).
  The sweep applies `decision.action`; a `LEAVE` carries its `leave_reason` for the disposition emitter.
  First-match over sets replaces an exact-key dict lookup, so δ proves equivalence: before the change it
  generates a golden table from today's *effective* outcome — `classify_recovery`, plus the four
  rewrites, plus the MARK_DONE handler's three fallbacks with their own guards (each reverts only
  when `status == 'in-progress'`, and the delivered-checks one only when also `block.reason ==
  'failed'`; re-read in `_reconcile_one_stranded` at `e634cfe658`) — over the full product `TaskStatus × claimant × BranchStateKind × veto ×
  (DeployPhase | None) × LandingEvidence × the three pin predicates × enforce`, where `LandingEvidence`
  carries the delivered-checks block reason (`failed`, and `errored`/`main_sha_unresolved` as
  `DELIVERED_CHECKS_UNMEASURED`), and asserts the new
  `recovery_for` matches it row for row. The enforce axis is generated at `enforce=True` only: η makes
  it the default, every project yaml sets it, and S8-7 deletes log mode with the field, so the
  `enforce=False` rows are retired behaviour, not equivalence targets. Rejected (R6 option 2): base verdict plus sweep policy.
- **D8 — the sweep classes receive, never reach.** Each sweep/filer class takes its collaborators by
  constructor and holds no reference to `Harness` once its chain completes (ε2a's two transitional
  Harness callables are replaced by ε2b). State that was a Harness container moves to the class that
  writes it (`_recovered_plans`, `_preserved_worktrees`, `_recovered_sessions`,
  `_recovered_session_config_dirs` → the frozen boot `CrashRecoveryResult`, seeded into one mutable
  `RecoveredWork` ledger that `_run_slot` claims from, `GatedDoneWriter` discards from and the orphan
  reaper and the revert read; `_recovery_veto_tracker`, `_recovery_streak_filed_at`,
  `_recovery_process_notices` → `recovery_emission.py::RecoveryDispositions`, constructed once and
  shared by `StrandedReconciler`, `DeterministicReconSweep` and `Harness._escalation_hold`;
  `_zero_progress_filed_at` stays with `_apply_retry_cap`; `_pool_storage_absent_maybe_pending` →
  `InfraFilers` with its filer pair). A collaborator two classes use is built once by the composition
  root and injected into both, never reached through the other (`GatedDoneWriter`, `RecoveredWork`,
  `RecoveryDispositions`, the unit inspector, `LandingRejectRecorder`). A moved method that calls a
  method a LATER leaf moves receives it as a constructor callable bound to the Harness method, and the
  later leaf rebinds the composition root onto its class. The pairs, measured by AST at `bb1c659562`:
  ε1a and ε1b → ζ1b's pool-storage filer pair; ε2a → ε2b's disposition emitter, veto-streak release and
  verified-green submit, and ζ1b's `_escalate_reconcile_failure`; ε3 → ζ2's `_revalidate_open_l2`.
  Heuristic 13: every new module imports downward only; none imports `orchestrator.harness`.
- **D9 — the filing-identity tripwire widens.** `test_harness_escalation_filing_identity.py`'s AST
  scan covers `harness.py` plus the three filer modules; its floor is not lowered when a site moves.
- **D10 — prose stance.** Every docstring a leaf moves or rewrites follows the Comments stance in
  `docs/code-quality.md`: task-number histories, review-finding answers and rejected alternatives go to
  the commit message; a moved method keeps at most a one-line pointer. Moved regions are counted by
  AST-moved region, not diffstat (the project's measured convention from task 5305).
- **D11 — one writer, same call shape (ι; S8-3 option A).** `emit` keeps its signature and never raises;
  durability is per batch with `synchronous=FULL`; every `EventStore` read method drains the pending
  queue first (bounded on the loop thread by one batch interval plus one commit), so read-your-writes needs no call-site discipline; shutdown drains. Rejected:
  `asyncio.to_thread` per emit (a cancellation point that loses the write — the repo's own
  counter-example in `workflow.py`), relaxing to `synchronous=NORMAL` (a durability decision the ruling
  did not take), and explicit `flush()` at enumerated read sites (subsumed by D19: once every
  in-process read is an `EventStore` method, draining in that method is the named site).
- **D12 — backend travels with the call (κ1).** `invoke_agent(..., sandbox_backend=)`; the fail-closed
  escalation dedup moves to a `SandboxRefusalDedup` the composition root owns, because that dedup is
  per orchestrator process, not per module. `fused-memory`'s `reconciliation/sandbox_guard.py` mirrors
  the resolution order by comment today; S19's shared sandbox module adopts the parameterised API when
  it moves.
- **D13 — session, not url (μ1, μ2).** Consumers receive an `McpSession`; a url appears only where a
  session is built (`McpLifecycle.__init__`, `fm_session`, and the out-of-harness entry points' own
  `async with`). Rejected: a session provider callable (temporal coupling on `start()` survives in a new
  form).
- **D13a — recall without a session is off, and says so (μ2).** `MemoryRecall(session: McpSession |
  None)`; `None` disables recall with one WARNING per process. `cli.py` and the evals open and close their
  own `McpSession` around the run; none relies on `mcp_call`'s per-call fallback any more.
- **D14 — merge-lane recovery methods stay; the journal's construction is β's.** The four
  `_recover_pending_merges`-family methods and `build_train_callback_factory` are S6's territory; ε
  touches neither. Constructing S6's `LandedOutbox`, `LaneTimingBudgets` and `LaneLimits` is the
  composition root's job (G4: "harness (S8) constructs the journal"), so β re-homes them.
- **D15 — R6's two sweeps (S8-5, confirmed by Leo).** "Both recovery sweeps consume `TaskGroundTruth.derive_truth`"
  names `CrashRecovery` (boot) and `StrandedReconciler` (startup and mid-run), the G4 row's "both harness
  sweeps". `CrashRecovery` takes its task-state facts from `TruthReport`; `LaneEvidence` holds only what
  `TruthReport` lacks.
- **D16 — handlers apply, never re-decide.** Each `RecoveryAction` handler returns a typed
  `HandlerOutcome` (`applied | deduped | submitted_verified_green`), which the disposition emitter
  records; a handler never calls another action's handler. Under S8-6 (option a) the evidence checks the
  MARK_DONE handler runs today
  (`_branch_is_degenerate`/`warm_lane_ref_is_degenerate`, `validate_landing_evidence`, a `failed`
  delivered-checks block) are computed by `derive_truth` into `TruthReport`, so the rows decide
  MARK_DONE versus REVERT; the RE_FILE handler's verified-green submit and
  `report_would_duplicate_a_handoff` dedupe are application outcomes (`submitted_verified_green`,
  `deduped`), not re-decisions.
- **D17 — steward spend is per task, durable, capped at $25 (θ; S8-1 option A plus Leo's raise).** The
  spend lives in one `PersistentMap` entry per task (INV-9), forgotten 7 days after the task's last
  steward spend (the map's TTL tracks the last bump). The default rises to `25.0` in `config.py` and
  `defaults.yaml`, and this repo's `20.0` line with its comment is deleted, in the SAME commit as the
  durable spend, so no deploy carries the raise without the durable count (several steward instances
  per task would otherwise each get $25; task 4810 had 7 in the September measurement) and no reload
  window drops this repo to $12. The per-call cap `budgets.steward` ($5) still shapes spending and the
  last call may overshoot by up to $5. Task 5822's "re-derive `steward_lifetime_budget`" item becomes a
  read-only report of the durable spend distribution against the $25 cap (lead amendment); its main
  question (fresh session versus resume) is unchanged. Rejected: renaming the key per instance (task
  3566, cancelled) and shipping the raise alone first (Leo kept them in one commit).
- **D18 — `convert_to_blocked_enforce` is deleted, in a safe order (S8-7 option B).** (1) η flips the
  code default to True; (2) after η is deployed fleet-wide, the lead removes the key from the eight
  other project yamls (autopilot-video, know-live, pump-web-ui, reify, scoping-intake, solar-challenge,
  solar-challenge-platform, stream-touch), one yaml-only `git commit --only` per repository on its
  `main`, best effort, recorded on δ's task description; no task dependency and no hold on any deploy;
  (3) δ deletes the log-mode rewrite, the field and this repo's yaml line. Removing a yaml line before η
  deploys would put that project into log mode at its next restart (the field is restart-tier), which
  is why η goes first. A project whose yaml still carries the key when δ deploys gets one self-closing
  born-at-L2 unknown-key escalation from `Harness._file_config_unknown_keys_escalation` (startup step
  `1c0-config-keys`; `OrchestratorConfig` ignores unknown keys, so startup continues) that closes when
  the line goes and the project restarts or reloads. Rejected: a deprecated, ignored field (the plan's
  earlier recommendation) and editing the yamls before the default flips (a log-mode window).
- **D19 — `EventStore` is the only events reader in the orchestrator package (ι2; S8-3).** Inside
  `orchestrator/src`, an events read is an `EventStore` method; a guard test fails on raw `FROM
  events`/`JOIN events` SQL outside `event_store.py` and a named allow-list (offline tools on their own
  connections and one printed operator recipe), each entry with its reason. Read-your-writes is then
  complete for every reader in the writer's process. Readers in other processes (dashboard,
  fused-memory, client feed, quality series, scripts) see committed rows at most one batch interval
  late under any writer option; their shared read model is the follow-up routed to S22. No design
  invariant is promoted (R22): a repo-wide rule can come with that follow-up, once it can pass.

## Contracts

```python
# C1 — orchestrator/src/orchestrator/slot_registry.py (α)
class SlotKind(StrEnum):
    WORKFLOW = 'workflow'; DETERMINISTIC = 'deterministic'

@dataclass(frozen=True)
class SlotRecord:
    task_id: str
    kind: SlotKind
    modules: tuple[str, ...]
    cancel_event: asyncio.Event
    escalation_event: asyncio.Event
    slot_task: asyncio.Task
    lease: SlotLease                      # module locks + semaphore permit; release() idempotent
    dispatched_at: float                  # monotonic

class SlotAlreadyLive(Exception):
    live: SlotRecord; rejected: SlotRecord                 # D1a

class HardCancelRefused(StrEnum):
    NO_SLOT = 'no_slot'; DETERMINISTIC = 'deterministic'   # S8-4; a reason, never a return value of hard_cancel_workflow

class SlotRegistry:                                         # implements S3 C2 WorkflowSlots (orchestrator/mcp/ops_tools.py)
    def __init__(self, *, on_cancel_accepted: Callable[[str], None]): ...   # bound to scheduler.note_workflow_cancelled
    def register(self, record: SlotRecord) -> None        # raises SlotAlreadyLive on a duplicate task id
    def current(self, task_id: str) -> SlotRecord | None
    def release(self, record: SlotRecord) -> bool          # False when record is no longer current
    def live_task_ids(self) -> frozenset[str]
    def is_active(self, task_id: str) -> bool              # WorkflowSlots
    def cancel(self, task_id: str) -> bool                 # WorkflowSlots; sets cancel_event and calls on_cancel_accepted;
                                                           # False with no slot or kind=DETERMINISTIC (S8-4)
    async def wait_cleared(self, task_id: str, timeout_secs: float) -> bool   # WorkflowSlots; True once no record is current

# harness.py — cancel_workflow deleted in α; is_workflow_active a one-line self._slots.is_active
# forwarder until S3 ζ, deleted in ω (S3 OQ-7 (a); the S3-1 / S8-2 cycle keeps it out of β); in-harness callers use the registry
class Harness:
    def hard_cancel_workflow(self, task_id: str) -> bool   # True only when a task.cancel() was issued
    def hard_cancel_refusal(self, task_id: str) -> HardCancelRefused | None

# C2 — orchestrator/src/orchestrator/task_ground_truth.py (δ)
class LandingEvidence(StrEnum):                            # S8-6
    ACCEPTED = 'accepted'; DEGENERATE_BRANCH = 'degenerate_branch'
    EVIDENCE_REJECTED = 'evidence_rejected'; DELIVERED_CHECKS_FAILED = 'delivered_checks_failed'
    DELIVERED_CHECKS_UNMEASURED = 'delivered_checks_unmeasured'   # block.reason errored | main_sha_unresolved → a LEAVE row
    NOT_APPLICABLE = 'not_applicable'

@dataclass(frozen=True)
class TruthReport:                                          # existing fields unchanged, plus:
    merge_in_flight: bool                                   # filled by derive_truth (task 7277's ownership)
    landing_evidence: LandingEvidence                       # S8-6; filled by derive_truth

@dataclass(frozen=True)
class RecoveryRule:
    name: str                                               # carried on recovery_* events as `rule`
    statuses: frozenset[TaskStatus]
    claimant_live: bool | None                              # None = any
    branch_kinds: frozenset[BranchStateKind]
    vetoes_done_flip: bool | None                           # None = any
    deploy_phases: frozenset[DeployPhase | None] | None     # None = any
    condition: Callable[[TruthReport], bool]                # a NAMED module-level predicate; never a lambda; reads no config
    action: RecoveryAction
    leave_reason: LeaveReason | None = None

RECOVERY_RULES: tuple[RecoveryRule, ...]                        # first match wins; default LEAVE

@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    leave_reason: LeaveReason | None
    rule: RecoveryRule | None                                   # None for the default

class HandlerOutcome(StrEnum):                                  # D16
    APPLIED = 'applied'; DEDUPED = 'deduped'; SUBMITTED_VERIFIED_GREEN = 'submitted_verified_green'

class TaskGroundTruth:
    def __init__(self, *, scheduler, git_ops, escalation_queue, landed_outbox: LandedOutbox,
                 merge_in_flight: Callable[[], frozenset[str]], ...): ...
    async def derive_truth(self, tid: str, *, row: dict | None = None) -> TruthReport
    async def recovery_for(self, tid: str, *, row: dict | None = None) -> tuple[TruthReport, RecoveryDecision]

# C3 — orchestrator/src/orchestrator/scheduler.py (β0 adds the fields, β makes them required)
LandedOutboxGate = Callable[[str], Awaitable[bool]]             # β defines it here
@dataclass(frozen=True)
class SchedulerCallbacks:
    landed_outbox_gate: LandedOutboxGate
    already_landed_gate: AlreadyLandedGate                      # already_landed_tick_view.py::AlreadyLandedGate
    ...                                                         # the 12 existing optional hooks
    recovery_for: Callable[[str], Awaitable[tuple[TruthReport, RecoveryDecision]]] | None = None   # δ; S17's phase
    @classmethod
    def passthrough_gates(cls, **hooks) -> SchedulerCallbacks   # gates that gate nothing, by name

class Scheduler:
    def __init__(self, config, event_store, *, callbacks: SchedulerCallbacks,
                 escalation_queue: EscalationQueue | None = None,
                 mcp_session: McpSessionLike | None = None, ...): ...   # μ1 makes mcp_session required
    def module_cache_summary(self) -> ModuleCacheSummary       # replaces the 3 private reads

# C3b — orchestrator/src/orchestrator/harness.py (β): stores fail closed (D4a, S8-8)
STORE_UNAVAILABLE_SENTINEL = 'store-unavailable-startup'   # one pending born-at-L2, refreshed in place
class StoreUnavailable(RuntimeError):
    store: str; path: Path; cause: BaseException               # raised from Harness.__init__ / run() after the L2 is filed
# retried only for sqlite3.Error with sqlite_errorcode in {SQLITE_BUSY, SQLITE_LOCKED}: ~3 attempts, ~30 s

# C4 — orchestrator/src/orchestrator/background_service.py (β; ι prepends): ordering only
#   β registers usage-gate, cost-store and mcp first, then the scheduler's shutdown, loop-stall-monitor, escalation-server, …;
#   ι prepends event-writer. stop_all() is the reverse; Harness.run's finally tears down no registered service.

# C5 — orchestrator/src/orchestrator/event_store.py (ι)
class EventStore:
    def emit(self, event_type, *, task_id=None, phase=None, role=None, data=None,
             cost_usd=None, duration_ms=None) -> None           # unchanged; thread-safe; never raises; enqueues
    def flush(self, timeout_secs: float = 5.0) -> bool          # sync, any thread; True when every earlier emit is committed
    async def aflush(self, timeout_secs: float = 5.0) -> bool
    async def close(self) -> None                               # drains, then closes the one connection
    # every EventStore read method (latest_merge_finalized, merge_finalized_absorbing, …) drains first, bounded
    # ι2 adds typed reads for today's in-process raw readers (names tactical), each draining first:
    #   done count in a window, merge disposition counts, chain telemetry, turn-cap saturation rows,
    #   repended-during-pause rows, task ids carrying an event type
# orchestrator/tests/test_event_store_encapsulation.py (ι2): RAW_EVENTS_SQL_ALLOWED: Mapping[str, str]
#   module path → reason; any other orchestrator/src module holding FROM/JOIN events SQL fails the test

# C6 — orchestrator/src/orchestrator/session_resume_storm.py (γ)
BY_DESIGN_SESSION_RESUME_REASONS: frozenset[str]
class SessionResumeStorm:                                       # implements S9 C7's ResumeOutcomeSink
    def __init__(self, *, queue: EscalationQueue, config: OrchestratorConfig,
                 event_store: EventStore, clock: Callable[[], float]): ...
    def note_resume_failed(self, failure: ResumeFailure) -> None   # by-design = failure.restore.is_by_design (S9 C7 landed)
# if S9 ζ1 has not landed: BY_DESIGN_RESTORE_OUTCOMES: frozenset[str] lives here (moved from harness.py)
#   and S9 ζ1 replaces it with RestoreOutcome.is_by_design in this module
    def note_resume_succeeded(self, task_id: str) -> None
    def archive_available(self, ...) -> bool                       # owns the log-once fault latch

# C7 — orchestrator/src/orchestrator/crash_recovery.py (ε1a)
@dataclass(frozen=True)
class LaneEvidence:                                             # only what TruthReport lacks
    lane_record: LaneRecord | None; plan_state: PlanState | None; stored_title: str | None
    # if S9 β has not landed: plan_steps: tuple[Mapping, ...] | None (today's plan.json read); S9 β re-types it

class LaneDecision(StrEnum):
    ADOPT = 'adopt'; PRESERVE = 'preserve'; CLEAN = 'clean'; QUARANTINE = 'quarantine'; RELEASE = 'release'

def decide_lane(truth: TruthReport, lane: LaneEvidence) -> LaneDecision

@dataclass(frozen=True)
class CrashRecoveryResult:
    recovered_plans: Mapping[str, Plan]; preserved_worktrees: frozenset[str]
    # if S9 β has not landed: recovered_plans: Mapping[str, dict] (today's _recovered_plans shape); S9 β re-types it
    recovered_sessions: Mapping[str, str]; recovered_session_config_dirs: Mapping[str, Path]

@dataclass(frozen=True)
class RecoveredEntry:
    plan: Plan | None; session: str | None; session_config_dir: Path | None; preserved: bool

class RecoveredWork:                                            # seeded once from CrashRecoveryResult (D8)
    def __init__(self, result: CrashRecoveryResult): ...
    def holds(self, task_id: str) -> bool                       # orphan reaper, stranded revert
    def claim(self, task_id: str) -> RecoveredEntry | None      # _run_slot: removes and returns
    def discard(self, task_id: str) -> None                     # GatedDoneWriter after a done-write

# stranded_reconcile.py (ε2a)
class GatedDoneWriter:                                          # one gated attribution done-write (D8)
    async def write_found_on_main(self, task_id: str, *, main_sha: str, ...) -> DoneWriteOutcome
    # delivered-checks gate first (today's _delivered_checks_block), then scheduler.mark_done, then
    # recovered_work.discard; used by StrandedReconciler, Harness._already_landed_dispatch_gate and 4651
```

Invariants. C1: a task id has at most one current record; `register` and the slot task's creation happen
in one synchronous section; `release` is idempotent and returns False for a stale record; `is_active`
is `current(task_id) is not None` and is the only liveness definition in `orchestrator/src`;
`SlotRegistry.cancel` calls `on_cancel_accepted` exactly when it accepted the cancel; S8 defines no
`WorkflowSlots` Protocol (S3 C2 is the one definition, INV-9); the terminal-status scan never cancels or
force-releases a `kind=DETERMINISTIC` record; `hard_cancel_workflow` returns a plain `bool`.
C2: no code outside `task_ground_truth.py` assigns a `RecoveryAction`; every condition is a module-level
named predicate; a `LEAVE` decision always names its `leave_reason`; `RECOVERY_RULES` reproduces the
golden table generated from today's classifier plus the four rewrites (D7); a handler returns a
`HandlerOutcome` and never calls another action's handler. C3: `Scheduler` cannot be constructed with
an absent safety gate; the consult methods have no `is None` branch; `Harness` passes every collaborator
at construction, and the queue's loop-bound callbacks registered at the top of `run()` are the only
`run()`-time binding. C3b: no store is ever unset or a stand-in; a store that cannot be built leaves
exactly one pending born-at-L2 on `STORE_UNAVAILABLE_SENTINEL` and a raised `StoreUnavailable`; only
`SQLITE_BUSY`/`SQLITE_LOCKED` is retried. C4: every collaborator a registered service holds is itself
registered earlier.
C5: events are committed in emit order; `flush()` returning True means durable; a read through an
`EventStore` method sees every row emitted before it; a raw-SQL reader on its own connection (another
process, or an allow-listed offline tool) lags by at most the batch interval; a read method called on
the loop thread waits at most one batch interval plus one commit; a writer failure is logged with the
dropped count; after ι2 no `orchestrator/src` module outside `event_store.py` and the named allow-list
holds raw events SQL. C6: `orchestrator.harness`
defines no session-resume constant; `storm_window_bound` imports only from C6. C7: `CrashRecovery`
reads no task status except through `TruthReport`; after ε1a no `harness.py` attribute holds a
recovery map (only the injected `RecoveredWork`), and after ε2a every attribution done-write of the
stranded sweep and the already-landed gate goes through the one `GatedDoneWriter`.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | registry ↔ server | `release_workflow` while a deterministic gate runs | task dispatched, `kind=DETERMINISTIC`, runner awaiting | `slots.cancel` False (S8-4); `slots.wait_cleared` returns False at `timeout_secs`; response `released=False, was_active=True, slot_cleared=False, parked=None`; no `blocked` write while the slot is live |
| 1b | registry | `slots.cancel` on a live workflow slot | `kind=WORKFLOW` record | `cancel_event` set and `scheduler.workflow_cancel_recent(tid)` True (the R3 grace stamp, via `on_cancel_accepted`) |
| 1c | registry ↔ watcher | terminal row, live `DETERMINISTIC` record | task row `done`, `DeterministicRunner.run` still executing, poll count past `terminal_status_force_release_polls` | no cancel, no hard-cancel, no `_force_release_wedged_slot`; claimant and module locks held |
| 1d | registry | `hard_cancel_workflow` with no slot | no record for tid | returns `False`; `hard_cancel_refusal(tid)` is `HardCancelRefused.NO_SLOT`; the terminal-status scan neither logs nor counts a hard-cancel |
| 2 | registry | slot task cancelled before its first step | `_launch_slot` returned, loop not yet run | done-callback finds the record still current and releases it; `is_active` False; no leaked event or lease |
| 2b | registry | duplicate registration (D1a) | record R live for tid; second `_launch_slot` for tid | second task cancelled before it runs, its lease released, ERROR names both records; R untouched |
| 3 | registry | force-release then late slot exit (D2a) | watcher force-released record R; slot body still running | body sees `current(tid) is not R`, acts on nothing; its `finally`'s `release(R)` returns False and `_collect_slot_report` drops the report; a re-dispatch's record R2 is untouched |
| 3b | registry | ordinary slot exit (D2a) | slot body returns | `registry.is_active` False before `_apply_retry_cap` runs and before `lease.release` |
| 4 | registry ↔ reaper | orphan-L0 reaper during a live slot | record present, `escalation_event` set | reaper classifies the record as live from the registry, not from a second container |
| 5 | composition | `Scheduler(...)` without a gate | `SchedulerCallbacks()` with a gate omitted | construction raises naming the field; `passthrough_gates()` constructs and gates nothing |
| 6 | composition | gate reachable at first tick | real `Harness(config)` on a tmp `project_root`; no `run()` yet | `scheduler.callbacks.landed_outbox_gate` is the harness's bound method; no post-construction write in `run()` |
| 7 | lifecycle | shutdown with a live merger holding `cost_store` | merge-worker stopping slowly | `cost_store.close()` runs after merge-worker's `stop()`; `usage_gate.shutdown()` after watcher-supervisor's |
| 8 | lifecycle | `run()` raises before `start_all()` | injected failure at MCP start | `stop_all()` still tears down the registered stores; no `AttributeError` on an unbuilt store |
| 8b | composition | `runs.db` unusable at construction (D4a, S8-8) | `project_root` whose `data/` is read-only | `Harness(config)` raises `StoreUnavailable` naming the store and the path; exactly one pending born-at-L2, severity critical, on `store-unavailable-startup` names store, path and exception; one ERROR line; a second construction refreshes that record in place (still one) |
| 8c | composition | `runs.db` locked by another writer at construction (D4a) | a connection holding a write lock; injected clock and sleep | construction retries on `SQLITE_BUSY`/`SQLITE_LOCKED` only, succeeds once the lock is released inside the ~30 s budget, and files nothing; a non-busy `sqlite3.Error` is not retried |
| 9 | classifier | blocked, on main, no live claimant, no blocking pin | `TruthReport` built by fixture | `recovery_for` → MARK_DONE_WITH_PROVENANCE from a named rule, no sweep rewrite |
| 10 | classifier | blocked, EXISTS_OFF_MAIN, pin lets go | fixture | → RE_FILE_ESCALATION from a rule |
| 11 | classifier | in-progress CONVERT shape whose blocked arm would move | fixture | → LEAVE with `leave_reason=escalation_pinned` |
| 12 | classifier | CONVERT shape, no pin (S8-7) | fixture | → CONVERT_TO_BLOCKED from a rule; `OrchestratorConfig` has no `convert_to_blocked_enforce` field and no rule reads config |
| 12a | config | default config before δ (η) | `OrchestratorConfig()` with no yaml key | `convert_to_blocked_enforce is True`; a yaml setting `false` still selects log mode until δ |
| 12c | composition | a project yaml still carrying `convert_to_blocked_enforce` after δ (D18) | real `Harness(config)` start on a tmp `project_root` whose yaml sets the key | startup continues; exactly one born-at-L2 unknown-key escalation names the key; a restart with the line removed resolves it |
| 12b | classifier | equivalence (D7) | golden table generated before δ from today's effective outcome (`classify_recovery` + the four rewrites + the three MARK_DONE fallbacks with their guards), `LandingEvidence` in the product | every row of the product matches `recovery_for` |
| 13 | classifier ↔ sweep | in-progress task with a recovered merge request in flight (7277) | `TruthReport.merge_in_flight` True | → the disposition 7277 landed; no revert to pending |
| 13b | classifier | in-progress, on main, degenerate branch (S8-6) | `landing_evidence=DEGENERATE_BRANCH` | → REVERT from a rule; the MARK_DONE handler is never entered |
| 13c | classifier | in-progress, on main, landing evidence rejected (S8-6) | `landing_evidence=EVIDENCE_REJECTED` | → REVERT from a rule; blocked with the same evidence → today's effective outcome for blocked (golden row) |
| 13d | classifier | delivered checks failed (S8-6) | `landing_evidence=DELIVERED_CHECKS_FAILED`, once `in-progress`, once `blocked` | in-progress → REVERT from a rule; blocked → LEAVE (today's fallback is guarded by `block.reason == 'failed' and status == 'in-progress'`; a blocked task is held, no done-write) |
| 13e | classifier | in-progress, on main, delivered checks `errored` (S8-6) | `landing_evidence=DELIVERED_CHECKS_UNMEASURED` | → LEAVE with `leave_reason=delivered_checks_unmeasured`; no done-write, no revert |
| 14 | sweep | `StrandedReconciler.run_pass()` over a store with one row per action | real sqlite task store, tmp git repo | each handler called once and returns a `HandlerOutcome`; dispositions emitted; no `getattr` fallback path exercised |
| 15 | crash recovery | lane with durable record AND plan.json | tmp lane, record state ASSIGNED | one `decide_lane` call; adopt/quarantine identical whether the record or the plan is read first |
| 15b | crash recovery | boot pass over a lane whose task is `done` (D15) | `TruthReport.db_status=done` | release decided from `TruthReport.db_status`; no direct `scheduler.get_status` call |
| 15c | crash recovery ↔ sweep ↔ slot | a recovered task marked done by the stranded sweep, another dispatched (D8) | `RecoveredWork` seeded with tasks A and B | the sweep's `GatedDoneWriter` write discards A; `_run_slot` claims B's plan and session once; `holds(A)` and `holds(B)` are False afterwards; the orphan reaper skips a held worktree |
| 15d | already-landed gate ↔ done-writer (ε2a) | the dispatch gate auto-dones a landed task whose delivered checks fail | real `GatedDoneWriter`, a `failed` delivered-checks verdict | no done-write; the same block the stranded sweep's MARK_DONE handler takes for that verdict |
| 16 | storm | failure below the window then one success | `SessionResumeStorm` with fake clock | streak resets on success; `storm_window_bound` sampler imports C6, not harness |
| 17 | writer | 1,000 emits from the loop and 100 from a thread | fresh `EventStore` | `flush()` True; rows in emit order per producer; one connection opened |
| 17b | writer | emit then read on the same store (C5) | fresh `EventStore`, no explicit flush | `latest_merge_finalized` returns the row, called from the loop and from a thread |
| 18 | writer | writer connection fails mid-batch | injected sqlite error | WARNING names the dropped count; next batch commits |
| 17c | reader guard (ι2, D19) | a raw `FROM events` read planted in a non-allow-listed `orchestrator/src` module | `test_event_store_encapsulation.py` | the guard fails naming the module; with the plant removed it passes, and every allow-list entry names an existing module and a reason |
| 17d | reader (ι2) | digest window counts through the typed reads | fixture `runs.db` with events written by `emit`, no explicit flush | each typed read returns the same rows the pre-move raw SQL returned on the golden fixture, including rows emitted just before the read |
| 19 | sandbox | `invoke_agent(..., sandbox_backend='landlock')` with no `Harness` built | process never called the old `set_backend` | the wrapper is landlock's; `resolve_backend_or_refuse` refuses once and dedups on the injected object |
| 20 | backends | pi invocation in a repo with a tracked `.mcp.json` | fixture repo | refusal names the tracked path; the file is not renamed |
| 21 | mcp | harness `Scheduler.dispatch_tool` before `McpLifecycle.start()` | session constructed, not initialised | a typed not-started error, not a silent one-shot handshake to a different url (harness path only) |
| 21b | mcp | `MemoryRecall(None)` (D13a) | briefing built outside the harness | recall disabled; one WARNING per process; no `mcp_call` fallback |
| 22 | steward | second `TaskSteward` for the same task after `$15` spent (S8-1) | default config, durable spend file present, last bump younger than `_STEWARD_GUARD_TTL` (7 days) | remaining budget is `$10` of the `$25` cap, not `$25`; the budget-exhausted L1 text says "$25 per task, durable, forgotten 7 days after the last steward spend" |
| 22b | steward | spend older than the TTL | durable spend file whose last bump is 8 days old | the entry is gone; remaining budget is the full `$25` |
| 22c | config | default steward cap (D17) | `OrchestratorConfig()`; this repo's `dark-factory-orchestrator.yaml` | `steward_lifetime_budget == 25.0`, `defaults.yaml` agrees, and this repo's yaml sets no `steward_lifetime_budget` |

## Cross-PRD relationship

| Other PRD / stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S3 escalation server and queue boundary | consumes / implements | S3 C2 `WorkflowSlots(is_active, cancel, wait_cleared)` (S3 OQ-7 ruled (a)): `SlotRegistry` implements it, S8 defines none; α deletes `Harness.cancel_workflow` and keeps `Harness.is_workflow_active` as a one-line forwarder until S3 ζ rebinds the pin classifier, ω deletes it; `LadderLookups.workflow_active` binds to `slots.is_active`; S3 C4 `set_resolve_callback(..., deliver_on=loop)` | S3 owns the Protocol; S8 the registry | α depends on S3 δ (C2); β depends on S3 β (C4) and S3 δ; under S3-1 = A with S8-2 = A, S3 ε1 depends on 4651 which depends on ε2a, so NO S8 leaf at or before ε2a may depend on S3 ε1/ε2/ζ (cycle); β edits `_start_escalation_server` first and S3 ζ composes on top; S3 ε1 wires the native lane-query class into β's composition root; per-file order on `harness.py` (lead sequencing ruling, 2026-11-10): S3 ε1 depends on S8 ε2b and S8 ε3 depends on S3 ζ; ω depends on S3 ζ (also transitively through ε3) |
| S4 verify run context | same files | S4 β edits `review_checkpoint.py` (`VerifyContext.background`); S4 ι3 adds a `reap_leftover_verify_scopes` call in `harness.py`; S4 κ builds `AdmissionWatch(host)` in `harness.py`; S4 ξ edits `run_main_tip_sweep`/`confirm_main_tip_failure_is_real`, which `Harness._run_main_tip_sweep` (stays in `harness.py`) calls; S4-1 (2026-11-10): the process `VerifyHost` is resolved once, by `VerifyHost.process()`, at the boundary objects the harness builds; composition roots use `VerifyHost.process()`, never `fresh()`, and production code outside them never calls `VerifyHost.fresh()` | S4 | β depends on S4 β, S4 ι3 and S4 κ if filed (per-file order on `harness.py`, lead ruling 2026-11-10: S4's host wiring lands first, β re-homes it); β keeps the one `VerifyHost.process()` call in the composition root; ζ2 (beside `_run_main_tip_sweep`) ↔ S4 ξ, whichever is filed second |
| S5 GitOps ports | consumes | C2 `GitOpsCallbacks`, C6 `LaneLifecycle` | S5 | OQ-10 ruled (a) with S5 OQ-4 = A: S5 γ/ι do the minimal harness wiring; S8 has no λ; β depends on S5 γ and ι (same file) and re-homes the construction |
| S6 merge lane | consumes / constructs | C4 `LandedOutbox` (γ1; γ2 makes it required in `TaskWorkflow` constructions), C6 `LaneTimingBudgets` (α), `LaneLimits` (κ); S6 θ's `harness.py` edits; merge-lane recovery methods and `build_train_callback_factory` untouched | S6 | β re-homes the constructions (depends on S6 γ1, γ2, α, κ and θ if filed — S6's shared-files table puts S8's `harness.py` leaves after S6 θ); under S6-2 = B, S6 κ waits for task 5040, so β does too; δ keeps the `landed_outbox` parameter S6 γ1 adds to `TaskGroundTruth` (depends on S6 γ1); rule with S6 OQ-3 |
| S9 workflow spine | consumes | C1 `Plan`/`PlanState` (S9 β) | S9 | ε1a depends on S9 β if filed first; else `LaneEvidence` carries today's plan read and S9 β re-types it (C7, both orders) |
| S9 | consumes | C5 `classify_merge_future`, `file_stranded_merge_failed` (S9 θ) | S9 | ruled in S8's favour (lead, 2026-11-02): ε2b moves the verified-green trio, `_on_stranded_merge_done` included, into `StrandedReconciler`; S9's consumers import from `stranded_reconcile.py`; ε2b depends on S9 θ (lead ruling 2026-11-02; S8 is filed second and takes the edge): S9 θ re-points `Harness._on_stranded_merge_done` onto C5 while the method still lives in `harness.py`, and ε2b then moves the already re-pointed method into `StrandedReconciler` |
| S9 | consumes | C6 `corroborated_landing` (S9 ι) | S9 | δ depends on S9 ι (δ's `_resolve_branch_state` journal rung calls C6) |
| S9 | consumes | C7 `RestoreOutcome.is_by_design`, `ResumeFailure`, `ResumeOutcomeSink` (S9 ζ1; module per S9 OQ-8) | S9 | γ depends on S9 ζ1 if filed first; else γ lands first with `BY_DESIGN_RESTORE_OUTCOMES` moved into `session_resume_storm.py`, and S9 ζ1 replaces it there (C6, both orders) |
| S9 | same region | `_invoke` (S9 ζ1/ζ2), the reaper freshness arms `_has_fresh_dispatch`/`_has_fresh_merge_phase` (deletable after S9 ε), `steward.py` (S9 ε, ζ2, ξ, ω) | S9 | κ1 → S9 ζ1; μ1 → S9 ζ2; ζ2 → S9 ε (ζ2 deletes the two freshness arms); θ ↔ S9 ε/ζ2/ξ/ω, whichever is filed second |
| S9 follow-ups (S9-5, S9-7; 2026-11-10) | xref | S9-5: a periodic (about 2-minute) pass over the existing landed-merge journal with a one-hour give-up to an L1, after task 4646, listed beside this plan's sweeps; S9-7: the L0 settle-outcome check names "a joint L0 end-of-life redesign with the harness plan (S8)" as its follow-up when its thresholds are crossed | program (lead files) | xref only; no S8 task. When filed, the journal pass is one more `BackgroundService` the composition root registers |
| S10 task-write contract | same file | S10 β1 (the required `metadata_mode` signature) and β2 (the `MetadataWriteFailed` raise) at the 9 `update_task` call sites, and ξ (task-status literals), edit `harness.py` | S10 | β → S10 β1, β2; ζ1a/ζ1b/ζ2 → S10 ξ, whichever is filed second |
| S17 scheduler seams | produces (to) | C2 (`recovery_for` for the stranded-blocked phase) through `SchedulerCallbacks.recovery_for`, bound by the composition root so `TaskGroundTruth(scheduler=…)` and the scheduler's phase form no construction cycle; C3 (required gates, `escalation_queue=`, `module_cache_summary`) | S8 | β lands C3; δ lands C2 and the hook; S17 consumes |
| S22 dashboard readers | follow-up (S8-3) | a shared typed `runs.db` read model (events, run results, cost rows) for the dashboard, fused-memory, the client feed, the quality series and scripts; about 67 matching raw-SQL lines outside `orchestrator/src` (S8-3 discussion, not re-measured here) | program follow-up, routed to S22 | xref only; ι2 bans raw events SQL inside `orchestrator/src` only |
| S1 / S2 escalation | produces (to) | ζ1a/ζ1b/ζ2's filer modules are the repoint site for `escalation/filing.py::file` and S2's typed vocabulary | S1/S2 | cross-reference only; no S8 edit of `escalation/` |
| S3 / S7 event vocabulary | same file | `EventType` additions (`merge_deduplicated`, `merge_gate_unmeasured`) and S7-3's deletion of the narrowed-retry event marker fields in `event_store.py` | S3 / S7 | ι, ι2: enum and row edits do not conflict with the writer or the read methods; xref only |
| S19 package layering | boundary | `sandbox_dispatch`'s parameterised API (κ1) | S19 later moves the module | cross-reference |
| `plans/scheduler-pause-halt-retirement-prd.md` (4686 done, 4691/4687 pending) | consumed / boundary | `PauseClass`, `pause_class` column; the 8-method pause stratum incl. `_file_scheduler_pause_escalation`, `_file_restored_pause_escalation` | that PRD | `fk-b01dac97e6e9` fixed by 4686; the pause filers stay in `harness.py` for 4687/4691 |
| `plans/startup-hardening-prd.md` (8267 done; 8268 in-progress; 8269, 8270 pending) | consumed | the startup-step table β keeps; the crash-loop breaker around `Harness.run`'s startup block (8269); the watchdog restart-rate alarm (8270) | that PRD | β depends on 8270 (S8-8) and on 8268/8269 (same `run()` region); 8269's breaker is not widened to wrap `Harness` construction (not ruled; 8270 pages a constructor raise) |
| `plans/instruments-honesty-prd.md` (8559 β1, 8563 ε pending high) | same region | 8559 adds `flush_state_snapshot` to `run()`'s `finally`; 8563 turns `digest.py::query_count_done_in_window` into a `landings()` count via `asyncio.to_thread` | that PRD | β depends on 8559 and registers that flush as part of the scheduler's shutdown service (D5); ι2 depends on 8563 |
| `docs/prds/landed-not-done-recovery.md` η (4651, pending high) | boundary | a new landed reconciler over `pending`/`merge-deferred` | that PRD | S8-2 = A (Leo, 2026-11-10): 4651 depends on ε2a and is re-scoped to a second `StrandedReconciler` over `pending`/`merge-deferred` with rows in `RECOVERY_RULES`, under a tested `RecoveryRule.statuses` partition; its two `harness.py` delivered checks move to `stranded_reconcile.py`; its done-write goes through ε2a's `GatedDoneWriter` (the shared capability gate its 2026-08-24 amendment requires); S3 ε1 depends on 4651 (S3-1); 4651's temporary parking edge on 5040 (lead, 2026-11-10) is removed by S8's decompose seat after it adds 4651 → ε2a |
| Task 7277 (pending high) | consumed | in-flight-merge ownership guard | 7277 | δ depends on 7277 and folds the guard into `TruthReport.merge_in_flight` |
| Task 6615 (in-progress) | same file | `task_ground_truth.py::_pid_alive` | 6615 | δ depends on 6615 |
| Task 7247 (pending) | same region | `run()`'s shutdown step 4a | 7247 | β depends on 7247 |
| Task 8926 (pending low) | consumed | one file-or-refresh idiom for pending sentinel escalations (`startup_steps.py::_file_or_refresh_l1`, `Harness._file_dirty_tree_escalation`) | 8926 | β depends on 8926 and files its store-failure L2 through it |
| Task 4197 (pending high) | overlap | INV-8 restoration; its finding 2 (per-event connect and fsync on the loop) is ι's writer, its recommendation (b) | 4197 | lead amends 4197: finding 2 is delivered by ι; findings 1 and 3–6 stay with 4197 |
| Task 5819 (pending high) | same file | `steward.py` | 5819 | θ depends on 5819 |
| Task 5822 (pending, deterministic gate) | consumed | steward-mode gate; its "re-derive `steward_lifetime_budget`" item | 5822 | lead amends it to a read-only report of the durable per-task spend distribution against the $25 cap (S8-1) |

## Decomposition plan

Overlay bands (`.claude/skills/prd/project.md`): target leaf ~300–1,500 changed LOC counting moved
regions once (D10), ≤ 10–12 declared files; a leaf above 15 files is a review trigger and says why. β
declares 14 and ω 13; β0, δ, κ1 and μ2 exceed 15 as one mechanical mechanism over many sites and say
so; η is below the ~100 LOC floor and is filed `complexity='simple'`. The extraction leaves (γ, ε1a,
ε1b, ε2a, ε2b, ε3, ζ1a, ζ1b, ζ2) also declare every test file that reaches a moved name in code
(patch sites, direct calls and attribute writes on a `Harness`), because those files break when the
name leaves `Harness` and must be re-pointed onto the class's constructor or the composition root in
the same leaf (tests stance). Each leaf names them as candidates measured at `bb1c659562` by
`git grep -nE "(\.|')(<the leaf's moved names>)\b" -- 'orchestrator/tests/*.py'` with comment lines
dropped; the candidates include some docstring and same-named-attribute hits, which the decompose seat
drops after reading each site. With them, ε1a, ε1b and ε2a cross the 15-file review trigger: examined,
each is one extraction whose test re-points are mechanical, and splitting would stitch one unit across
leaves. All leaves but η, θ and κ2 edit
`harness.py`, so they are serialised by real edges in the order that shrinks the file top-down:
α → β0 → β → δ → γ → ε1a → ε1b → ε2a → ε2b → [S3 ε1 → S3 ε2 → S3 ζ] → ε3 → ζ1a → ζ1b → ζ2 → ι → ι2 →
κ1 → μ1 → μ2 → ω (the bracketed S3 leaves by the lead's sequencing ruling of 2026-11-10), with κ2 after
κ1 (same `agents/invoke.py`) and before ω. Three leaves sit off that chain and share `config.py` (and
δ and θ this repo's `dark-factory-orchestrator.yaml`), so they are serialised too: η has no
dependency and goes first; δ depends on η (the same field: η flips its default, δ deletes it, the
order S8-7 rules); θ depends on η and on δ (θ is medium and waits on 5819 and S9's `steward.py`
leaves; the reverse edge would put those on the high δ's path). No leaf depends on the lead's yaml
edits: η's deploy, not its merge, releases them (D18). The high leaf δ runs first after the
composition root. `high` leaves (δ, ε2a, ε2b) become dependencies of trigger-chain gate 6698 (pending)
at decompose.

**Cross-stream rule.** A sibling leaf that edits a region an S8 leaf moves lands before that S8 leaf (the
S8 leaf depends on it), or is re-scoped onto the new module. Whichever leaf is filed second takes the
edge. Three exceptions are fixed by ruling: ε2b depends on S9 θ, never the reverse (lead, 2026-11-02; the
reverse closes the cycle S9 θ → ε2b → … → δ → S9 ι → S9 θ); no S8 leaf at or before ε2a depends on
S3 ε1, ε2 or ζ (forced by S3-1 = A with S8-2 = A, 2026-11-10: S3 ε1 → 4651 → ε2a; confirmed by the
lead); and on `harness.py` S3 ε1 depends on S8 ε2b while S8 ε3 depends on S3 ζ (lead sequencing
ruling, 2026-11-10). The full walk at `bb1c659562` (S8's chain, S3 ε1 → {ε2b, 4651, S6 μ, S3 κ, S10
β1/β2}, S3 ε2 → {S3 ε1, S6 μ, 5409, S7 δ}, S3 ζ → S3 ε2, 4651 → {ε2a, 4623, 4646, 4647, 4673, 4887},
S6 μ → {S6 κ, S3 α}, β → S6 κ → {5040, S4 β, S5 θ}, β → S4 κ → {S4 ι3 → {S4 ι2, S4 θ → {S4 η, S6 ζ,
S6 λ → S6 κ}}, 5292}, ε2b → S9 θ, δ → S9 ι) has no cycle: no task S4 κ transitively depends on depends on
β.

| S8 leaf | Sibling leaves in its region |
|---|---|
| α | S3 δ (C2 `WorkflowSlots`) |
| β | S3 β (C4) and S3 δ, S4 β (`review_checkpoint.py`), S4 ι3, S4 κ (`harness.py`, lead ruling 2026-11-10), S5 γ/ι, S6 γ1/γ2/α/κ/θ, S10 β1/β2; tasks 7247, 8268, 8269, 8270, 8559, 8926 |
| δ | S6 γ1, S9 ι; tasks 7277, 6615; η (same field in `config.py`, S8-7's order) |
| γ | S9 ζ1 |
| ε1a | S9 β |
| ε2a | task 4651 depends on ε2a (S8-2); S8's decompose seat then removes 4651's temporary parking edge on 5040 |
| ε2b | S9 θ (one-way: ε2b depends on S9 θ, lead ruling); S3 ε1 depends on ε2b (lead sequencing ruling, 2026-11-10) |
| ε3 | S3 ζ (lead sequencing ruling, 2026-11-10: S3 ε1 → ε2 → ζ edit `harness.py` between ε2b and ε3) |
| ζ1a, ζ1b, ζ2 | S10 ξ; ζ2 also S9 ε and S4 ξ (`_run_main_tip_sweep` stays in `harness.py` beside ζ2's `_close_superseded_main_sweep_escalations`; no S8 leaf moves it); tasks 7170 (ζ1a), 8826 (ζ1b), 7385 and 8659 (ζ2) land first |
| κ1 | S9 ζ1 |
| μ1 | S9 ζ2 |
| θ | S9 ε, ζ2, ξ, ω (`steward.py`); task 5819; η and δ (`config.py`, this repo's yaml) |
| ι, ι2 | S3 (`merge_deduplicated`), S7 (`merge_gate_unmeasured`, S7-3's marker-field deletion) — enum and row edits, xref only; ι2 also task 8563 (`digest.py`) |
| ω | S3 ζ (the forwarder deletion; also implied through ε3) |

**Critical path to gate 6698.** The high leaves δ, ε2a and ε2b sit behind α → β0 → β. β waits on S3 β
and δ (neither held under S3-1 = A), S4 β, ι3 and κ (κ adds only itself and task 5292 to what ι3
already brings), S5 γ/ι, S6 γ1/γ2/α/κ/θ (S6 κ itself behind task 5040 under S6-2 = B), S10 β1/β2 and
tasks 7247, 8268, 8269, 8270, 8559 and 8926; δ adds S9 ι, S6 γ1 and tasks
7277 and 6615; ε2b adds S9 θ (lead ruling 2026-11-02). Cutting β's old edge on S3 ζ (forced by the
cycle) takes the S6 native-port task and S3's two merge-tool moves and ladder split off this path. The rest of the 19-leaf serial chain follows δ, so
6698 moves by the slowest of those upstream leaves plus α, β0, β, δ, γ, ε1a, ε1b, ε2a and ε2b, plus η
(a one-file `complexity='simple'` leaf with no dependency) through δ. Task 4651 (high) waits on ε2a,
eight leaves into the chain (S8-2's accepted cost). After ε2b the chain leaves this path: ε3 … ω wait
on S3 ζ, so on S6 μ (behind S6's chain and task 5040), S7 δ, task 5409 and S3 ε1/ε2/ζ (the lead's
2026-11-10 order; the price of keeping the high leaves off the native-port chain). θ waits on δ.

**η — `convert_to_blocked_enforce` defaults to True (S8-7, step 1)**
(`task_kind=normal`, `complexity='simple'`, priority medium, ~60 LOC; first in this plan and independent of the `harness.py` chain)
- Files: `orchestrator/src/orchestrator/config.py`, `orchestrator/tests/test_convert_to_blocked.py`
  (`test_…` pinning `OrchestratorConfig().convert_to_blocked_enforce is False` flips).
- Keys: none of its own; step 1 of D18 toward `fk-17b789fe3ce6`'s fourth rewrite. Deps: none. δ and θ
  depend on it (same `config.py`; δ deletes the field η flips). Its fleet-wide deploy, not an edge,
  releases the lead's yaml edits (D18 step 2), which are best effort.
- Changes: the field's default becomes True; its description says enforce is the shipped behaviour and
  log mode is selected only by a yaml that sets `false`, until δ deletes the field; the promotion
  comment's step 2 is marked done.
- Signal: boundary row 12a; no project's behaviour changes (all nine set `true`, measured 2026-11-10).

**α — `SlotRegistry`: one frozen per-slot record, one identity-guarded release**
(`task_kind=normal`, priority medium, ~950 LOC)
- Files: `orchestrator/src/orchestrator/slot_registry.py` (new), `orchestrator/src/orchestrator/slot_lease.py`
  (deleted; merged), `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_slot_registry.py`
  (new), `orchestrator/tests/test_harness_cancel_workflow.py`, `orchestrator/tests/test_harness_force_release_wedged_slot.py`,
  `orchestrator/tests/test_harness_terminal_status_watcher.py`, `orchestrator/tests/test_harness_deterministic_dispatch.py`,
  `orchestrator/tests/test_workflow_cancellation.py`, `orchestrator/tests/test_cancel_scope_bounded_cleanup.py`.
- Keys: `fk-c1ac3df8e44d`. Deps: S3 δ (C2 `WorkflowSlots`) if filed. Contract: lands C1 (implementing S3
  C2); D1a, D2a.
- Deletes: the 8 slot containers, `_cancel_causes`, `_slot_lease_registry`, `_force_released_slot_tasks`,
  `_unregister_slot`, `_owns_slot`, the dispatch-time leak-guard callback, `slot_lease.py`, and
  `Harness.cancel_workflow` (S3 OQ-7 (a)); `Harness.is_workflow_active` stays as a one-line
  `self._slots.is_active` forwarder until ω (after S3 ζ). Both `hard_cancel_workflow`
  call sites (`_scan_for_terminal_active_tasks`, `_action_teardown_and_set_status`) keep a `bool` result.
- Signal: boundary rows 1–4 (with 1b, 1c, 1d, 2b, 3b). Operator-visible: `release_workflow` on a running
  deterministic gate returns `released=False, slot_cleared=False, parked=None` (today: parks `blocked`);
  the journal's `workflow_slot_force_released` path still fires for a wedged slot (6660's test suite stays
  green).
- G7: INV-6 `status-matches-liveness` — one liveness definition; INV-3 — `release_workflow`'s re-check
  after `get_status` is preserved through `slots.wait_cleared` + `slots.is_active`.
- Live task 5153 (idle-workflow reaper) lands FIRST, in today's code (Leo, 2026-11-11, program doc R59): α
  (task 9216) depends on 5153 and folds the container 5153 adds into `SlotRegistry`, so the reaper then reads
  `SlotRecord.dispatched_at` and a per-record last-event stamp. The draft's reverse edge (5153 → α) is withdrawn.

**β0 — `make_scheduler` test factory and `passthrough_gates()`; the `_workflow_cancel_at` shim goes**
(`task_kind=normal`, priority medium, ~1,200 LOC, mechanical codemod)
- Files: `orchestrator/src/orchestrator/scheduler.py` (the two gate fields, defaulted, and
  `passthrough_gates`), `orchestrator/src/orchestrator/harness.py` (shim deleted), `orchestrator/src/orchestrator/cli.py`,
  `orchestrator/src/orchestrator/evals/runner.py`, `orchestrator/tests/_scheduler_factory.py` (new), the 60
  test files constructing `Scheduler(` (enumerated at decompose by
  `git grep -lE '(^|[^A-Za-z_])Scheduler\(' -- orchestrator/tests`; 392 sites measured at `e634cfe658`),
  and the 13 `_workflow_cancel_at` test files (unchanged since `d40ba048e9`): `orchestrator/tests/_orch_helpers.py`,
  `test_deterministic_task.py`, `test_harness_cancel_workflow.py`, `test_harness_claimant_clear.py`,
  `test_harness_deterministic_recon_sweep.py`, `test_harness_merge_registry_wiring.py`,
  `test_harness_reblock_guard.py`, `test_reconcile_stranded.py`, `test_repend_state_machine.py`,
  `test_scheduler_callbacks.py`, `test_stranded_blocked_sweep.py`, `test_stranded_verified_green.py`,
  `test_workflow_cancellation.py`. Exempt from the 15-file review trigger: one mechanical rewrite per
  construction site, no behaviour change.
- Keys: contributes to `fk-5d0a66a2d32e`. Deps: α (same file).
- Deletes: the `Harness._workflow_cancel_at` property/setter; every direct `Scheduler(` construction in
  `orchestrator/tests` outside the factory.
- Signal: `git grep -cE '(^|[^A-Za-z_])Scheduler\(' -- orchestrator/tests` counts only
  `_scheduler_factory.py` and the scheduler's own constructor tests; the full orchestrator test leg is
  green with no production diff beyond the two defaulted fields.

**β — Harness as a composition root: hooks and stores at construction, stores fail closed, teardown by registration order**
(`task_kind=normal`, priority medium, ~1,450 LOC)
- Files: `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/review_checkpoint.py`, `shared/src/shared/usage_gate.py`,
  `shared/src/shared/transcript_archive.py`, `orchestrator/src/orchestrator/git_ops.py` (archive hook
  argument only), `orchestrator/tests/test_scheduler_callbacks.py`, `orchestrator/tests/test_harness_landed_dispatch_gate_wiring.py`,
  `orchestrator/tests/test_harness_already_landed_gate_wiring.py`, `orchestrator/tests/test_harness_lifecycle_registry.py`,
  `orchestrator/tests/test_harness_scheduler_seam.py`, `orchestrator/tests/test_harness_starvation_watchdog.py`,
  `orchestrator/tests/test_harness_store_fail_closed.py` (new; rows 8b, 8c), `shared/tests/test_usage_gate.py`.
- Keys: `fk-d8f08696dd66`, `fk-fc3c4cba6a0f`, `fk-5d0a66a2d32e`. Deps: β0; out-of-batch 7247, 8268
  (if still live), 8269, 8559, 8926 and 8270 (S8-8: the watchdog alarm pages a constructor raise the
  in-process breaker cannot see); S3 β and δ, S4 β, ι3 and κ (κ: per-file order on `harness.py`, lead
  ruling 2026-11-10 — S4's host wiring lands first and β re-homes it, carrying the one
  `VerifyHost.process()` call), S5 γ and ι, S6 γ1/γ2/α/κ/θ, S10 β1 and β2 if filed (cross-stream rule). NOT S3 ε1/ε2/ζ (the S3-1 / S8-2 cycle). Contract: lands C3 (required gates,
  `LandedOutboxGate`), C3b, C4; D4a, D5, D6.
- Deletes: the 2 private scheduler gate writes, the 7 `run()` writes, the 3 `escalation_queue`
  late-binds, `set_archival_failure_hook`'s call, the hand-ordered `finally` statements, the three
  `try/except` store degrades (`Failed to create event store`, `task_results persistence is DISABLED`,
  `Failed to create cost store`), the 3 `_module_cache` reads, `_get_ground_truth`'s lazy build, the
  deterministic slot's `_escalation_queue is None` raise, the `HAS_ESCALATION`/`HAS_STEWARD` guards and
  the 28 function-local `from escalation.models import` lines (measured at `e634cfe658`).
- Re-homes: S6's `LandedOutbox`, `LaneTimingBudgets` and `LaneLimits` construction and S5's
  `GitOpsCallbacks`/`LaneLifecycle` wiring into the composition root.
- Signal: boundary rows 5–8, 8b, 8c. Operator-visible: `journalctl` shows `usage-gate stopped` and
  `cost-store stopped` after every sweep's `stopped` line on a clean shutdown; a config with the usage
  cap enabled starts with the gate already bound to the cost store (the first `cap_detected` row has a
  `run_id`); a broken `escalation.server` import stops startup instead of logging "Escalation package not
  installed"; an unusable `runs.db` leaves one critical born-at-L2 on `store-unavailable-startup`
  and a stopped unit instead of a running orchestrator with no event or run store.
- G7: INV-11 — an unwired gate, a broken escalation import and an unbuildable store are loud
  construction errors (D3, D4a, D6); INV-9 — one teardown order (D5).
- Subsumes 7698, 6625, 6768 and 7107 (cancel with reason). Edges at decompose: 6143 → β (its
  landed-outbox-gate item lands here; its PSI-reader and hold-history items are S17's); 7603 → β and
  8778 → β (β builds the queue in `__init__`, so their "public queue injection" item is delivered; their
  test migrations and `drain_background_tasks` stay); 8779 → β (its accessors re-scope onto the
  composition root's constructor arguments).

**δ — `RECOVERY_RULES` with condition columns; the sweep applies the decision verbatim (R6); `convert_to_blocked_enforce` deleted (S8-7)**
(`task_kind=normal`, priority **high**, ~1,400 LOC)
- Files: `orchestrator/src/orchestrator/task_ground_truth.py`, `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/src/orchestrator/recovery_pins.py`, `orchestrator/src/orchestrator/recovery_emission.py`,
  `orchestrator/src/orchestrator/scheduler.py` (`SchedulerCallbacks.recovery_for`),
  `orchestrator/src/orchestrator/config.py` and `dark-factory-orchestrator.yaml` (the field and this
  repo's line deleted), `orchestrator/tests/test_task_ground_truth.py`,
  `orchestrator/tests/test_recovery_rules_equivalence.py` (new), `orchestrator/tests/test_reconcile_stranded.py`,
  `orchestrator/tests/test_convert_to_blocked.py`, `orchestrator/tests/test_harness_ground_truth_wiring.py`,
  and the other readers of the deleted field measured at `e634cfe658`:
  `orchestrator/tests/test_clear_claim_then_set_status.py`, `orchestrator/tests/test_reconcile_revert_status_race.py`,
  `orchestrator/tests/test_recovery_emission_wiring.py`, `ARCHITECTURE.md`,
  `skills/escalation-watcher-auto/SKILL.md`. 17 files, above the 15-file review trigger: the ruled field
  deletion reaches five files that pin or describe log mode, each a mechanical removal.
- Keys: `fk-17b789fe3ce6`, `fk-5cad48d24b6f` (recovery half). Deps: β, η (the same `config.py` field
  and `test_convert_to_blocked.py`; S8-7's order); out-of-batch 7277, 6615; S6 γ1 (`landed_outbox`),
  S9 ι (journal rung) if filed. No dependency on the lead's yaml edits (D18): they are best effort.
  Contract: lands C2, D7, D16, D18 step 3.
- Deletes: the four rewrite blocks in `_reconcile_one_stranded`, the `getattr(self.config,
  'convert_to_blocked_enforce')` read, the `convert_to_blocked_enforce` field with its promotion comment,
  this repo's `convert_to_blocked_enforce: true` line, the `_only_merge_remediable` alias (5 references),
  the second `get_task` (6682), the 7277 guard in `_revert_in_progress_if_no_live_claimant` (becomes
  `TruthReport.merge_in_flight`), and the MARK_DONE handler's three evidence re-decisions (S8-6).
- Lead action recorded in this task's description (D18 step 2): after η is deployed fleet-wide, the lead
  removes `convert_to_blocked_enforce` from the eight other project yamls, one yaml-only `git commit
  --only` per repository, best effort, and records each commit sha (or "skipped: <why>") here.
- Signal: boundary rows 9–13 (with 12, 12b, 12c, 13b–13e); `recovery_vetoed`/`recovery_left` events carry
  `rule` (the matched row's name) so an operator reading `get_scheduler_events` sees which row decided a
  hold; a project yaml still carrying the deleted key produces one self-closing unknown-key L2, not a
  failed start.
- G7: INV-5 — one table; INV-2 — the decision names its rule and reason.
- Subsumes 6682 (cancel with reason).

**γ — `SessionResumeStorm`: the detector and vocabulary leave `harness.py`**
(`task_kind=normal`, priority low, ~800 LOC, 653 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/session_resume_storm.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/src/orchestrator/storm_window_bound.py`, `orchestrator/tests/test_session_resume_storm.py` (new),
  `orchestrator/tests/test_crash_recovery.py`, `orchestrator/tests/test_storm_window_bound.py`,
  `orchestrator/tests/test_session_resume_integration_gate.py`.
  Plus 2 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_orchestrator_governed_lists.py`, `test_recut_ruling_dispatch.py`.
- Keys: `fk-d0c02dc767dd`. Deps: δ (same file); S9 ζ1 (C7) if filed. Contract: lands C6.
- Deletes: `_BY_DESIGN_SESSION_RESUME_REASONS`, `_MAX_RECORDED_RESUME_FAILURES` (and
  `_BY_DESIGN_RESTORE_OUTCOMES`, moved into `session_resume_storm.py` as `BY_DESIGN_RESTORE_OUTCOMES`
  unless S9 ζ1 already replaced it with `RestoreOutcome.is_by_design`) from `harness.py`, the 4 streak
  attributes, the 8 methods named in §Findings disposition, `_archive_available_fault_logged` (now instance state), the reach-back
  import, the 3 test imports of harness privates.
- Signal: boundary row 16; `python -c 'import orchestrator.storm_window_bound'` loads without importing
  `orchestrator.harness` (measured by `sys.modules`); the storm L1 text is byte-identical for the same
  failures (golden fixture from `test_crash_recovery.py`).
- G7: INV-4 — the circuit breaker's reset-on-success is kept (not `StormCounter`).
- Subsumes 5536 (cancel with reason). Note: 4952 (deferred) is a future consumer of C6.

**ε1a — `CrashRecovery`: the boot worktree scan as a class reading `TruthReport` (D15)**
(`task_kind=normal`, priority medium, ~1,250 LOC, 879 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/crash_recovery.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_crash_recovery.py`, `orchestrator/tests/test_project_layout_adoption.py`,
  `orchestrator/tests/test_crash_recovery_lane_decision.py` (new), `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  Plus 21 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `_lane_test_helpers.py`, `test_harness_interactive_reaper.py`, `test_harness_lifecycle_registry.py`, `test_harness_park_stop.py`, `test_harness_parked_loop.py`, `test_harness_scheduler_seam.py`, `test_harness_startup_get_statuses.py`, `test_harness_startup_git_maintenance.py`, `test_harness_startup_steps.py`, `test_harness_verify_scope_reaper.py`, `test_harness_warm_lane_wiring.py`, `test_harness_watcher_supervisor.py`, `test_lane_in_use.py`, `test_lane_lifecycle_integration.py`, `test_merge_worktree_lifecycle_integration_gate.py`, `test_reconcile_stranded.py`, `test_session_resume_integration_gate.py`, `test_singleton_lock.py`, `test_stale_escalation_cleanup.py`, `test_transcript_archive_startup_sweeper.py`, `test_warm_lane_pool_shrink.py`.
- Keys: `fk-e6a18c90d2fe`. Deps: γ (same file; γ moves `_adopt_recovered_session`'s eligibility
  callees first), δ (C2, transitively); S9 β (C1 `PlanState`) if filed. Collaborators: `TaskGroundTruth`,
  `git_ops`, `scheduler`, `config`, `event_store`, and a `file_pool_storage_absent` callable (the Harness
  method until ζ1b rebinds it, D8). Contract: lands C7 (S8-5), including the `RecoveredWork` ledger the
  composition root seeds from the result and injects into `Harness` (`_run_slot` claims from it).
- Deletes: `_recover_crashed_tasks`' and `_recover_worktree_entry`'s mirrored branches (one
  `decide_lane` call; task 8267's per-entry isolation is kept), their 2 direct `get_status` and 2
  `get_task`/`read_worktree_title` reads, the 4 recovery containers from `Harness.__init__` (every
  reader and writer of them, `_run_slot`, `_reap_orphan_entry`, `_revert_in_progress_if_no_live_claimant`
  and `_mark_in_progress_done` included, goes through `RecoveredWork`),
  `_sweep_orphaned_transcripts`, `_adopt_recovered_session` from `harness.py`;
  `_task_recovery_artifact` becomes a `crash_recovery.py` function `harness.py` imports.
- Signal: boundary rows 15, 15b, 15c; after a simulated SIGKILL mid-implementer (the `test_crash_recovery.py`
  restart idiom) the boot log's `Crash recovery: adopted N, preserved M, cleaned K` line is unchanged
  and `CrashRecoveryResult` reports the same sets the private containers held.
- G7: INV-5 — the duplicated identity guard and terminal release exist once.
- Subsumes 7436 (cancel with reason). Edge 7349 → ε1a at decompose (same method text).

**ε1b — `LaneReconciler`: lane reconciliation as a class**
(`task_kind=normal`, priority medium, ~800 LOC, 644 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/lane_reconcile.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_harness_warm_lane_wiring.py`, `orchestrator/tests/test_harness_lane_record_drift.py`,
  `orchestrator/tests/test_harness_interactive_reaper.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  Plus 14 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_atomic_train_merge.py`, `test_harness_lifecycle_registry.py`, `test_harness_parked_loop.py`, `test_harness_scheduler_seam.py`, `test_harness_startup_get_statuses.py`, `test_harness_startup_git_maintenance.py`, `test_harness_terminal_lane_record_reclaim.py`, `test_harness_warm_lane_gc.py`, `test_harness_watcher_supervisor.py`, `test_lane_checkout_reconcile_conflict.py`, `test_orphan_reaper.py`, `test_reconcile_stranded.py`, `test_singleton_lock.py`, `test_warm_lane_pool_shrink.py`.
- Keys: contributes to `fk-bf4197a9a7d9`. Deps: ε1a (same file). Collaborators beyond the stores:
  `RecoveredWork` (the reaper's skip set), `file_pool_storage_absent` and
  `resolve_pool_storage_absent_if_pending` callables (Harness methods until ζ1b rebinds them, D8).
- Deletes: `_reconcile_lane_checkouts` and `_reconcile_one_lane_checkout`, `_reap_orphan_worktrees` and
  `_reap_orphan_entry`, `_reconcile_terminal_lanes`, `_reclaim_terminal_lane_records`,
  `_assigned_durable_records_with_statuses`, `_stale_lane_assignment_census` from `harness.py`; startup
  steps `2c2`, `2e` and `diff7` call the class.
- Signal: the lane-reconcile log lines and `lane_*` events for a fixture pool match the pre-move golden
  set; tests construct `LaneReconciler` directly.

**ε2a — `StrandedReconciler`: the stranded sweep as a class dispatching on `RecoveryDecision`**
(`task_kind=normal`, priority **high**, ~1,500 LOC, 1,390 moved before δ's deletions, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/stranded_reconcile.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_reconcile_stranded.py`, `orchestrator/tests/test_stranded_reconciler.py` (new),
  `orchestrator/tests/test_retry_cap.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  Plus 27 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_clear_claim_then_set_status.py`, `test_convert_to_blocked.py`, `test_crash_recovery.py`, `test_deterministic_task.py`, `test_harness_abandoned_merge_deferred_wiring.py`, `test_harness_already_landed_gate_wiring.py`, `test_harness_deterministic_recon_sweep.py`, `test_harness_ground_truth_wiring.py`, `test_harness_infra_hold_repend.py`, `test_harness_interactive_reaper.py`, `test_harness_lifecycle_registry.py`, `test_harness_park_stop.py`, `test_harness_parked_loop.py`, `test_harness_scheduler_seam.py`, `test_harness_startup_get_statuses.py`, `test_harness_startup_git_maintenance.py`, `test_harness_verify_scope_reaper.py`, `test_harness_warm_lane_wiring.py`, `test_harness_watcher_supervisor.py`, `test_reconcile_revert_status_race.py`, `test_recovery_emission_wiring.py`, `test_repend_state_machine.py`, `test_singleton_lock.py`, `test_stale_escalation_cleanup.py`, `test_status_authority_gate.py`, `test_stranded_blocked_sweep.py`, `test_stranded_verified_green.py`.
- Keys: `fk-bf4197a9a7d9`. Deps: ε1b (same file), δ. Task 4651 depends on ε2a (S8-2 = A).
- Deletes: `_reconcile_stranded_in_progress`, `_reconcile_stranded_at_startup`, `_reconcile_one_stranded`,
  `_revert_in_progress_if_no_live_claimant`, `_run_stranded_reconcile_pass` from `harness.py`;
  `_mark_in_progress_done` and `_delivered_checks_block` become `GatedDoneWriter` (C7), which
  `_already_landed_dispatch_gate` (stays) now calls through the injected writer, so its 57 code
  references in `test_harness_already_landed_gate_wiring.py` re-point onto it;
  `_RECONCILE_SWEEP_STATUSES` moves with the class. Above the 15-file trigger with its test re-points
  (examined: one extraction, §Decomposition plan); ~1,700 LOC with them, above the band for the same
  reason.
  The class is built to take a status partition at construction so 4651's second instance over
  `pending`/`merge-deferred` needs no change to it (S8-2).
- Signal: boundary rows 13–14, 15d; the sweep's summary log line (`Reconcile: N reverted; M marked done; …`)
  is unchanged for the same store; `test_reconcile_stranded.py` constructs `StrandedReconciler` with a
  real sqlite task store and tmp git repo — no `Harness.__new__`, no `patch.object(harness, '_mark_in_progress_done')`.
- G7: INV-8 — `derive_truth`'s synchronous filesystem reads per task are noted (4651's observation) and
  bounded by the existing sweep cadence; offloading them is out of scope and recorded as a follow-up
  line in the task record, not a new task here.

**ε2b — `RecoveryDispositions` and the verified-green trio join the reconciler**
(`task_kind=normal`, priority **high**, ~800 LOC, 619 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/stranded_reconcile.py`, `orchestrator/src/orchestrator/recovery_emission.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_recovery_emission_wiring.py`,
  `orchestrator/tests/test_stranded_reconciler.py`, `orchestrator/tests/test_stranded_verified_green.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  Plus 3 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_harness_deterministic_recon_sweep.py`, `test_reconcile_stranded.py`, `test_zero_progress_requeue.py`.
- Keys: contributes to `fk-bf4197a9a7d9`. Deps: ε2a; S9 θ (C5) — one-way, lead ruling 2026-11-02: S9 θ
  re-points `Harness._on_stranded_merge_done` onto C5 in `harness.py` first; ε2b moves the re-pointed
  method.
- Deletes: `_emit_recovery_disposition` and its 8 memo/latch methods (task 4147's
  `_seeded_recovery_veto_tracker` and `_veto_streak_release_sites` and `_rearm_recovery_process_notice`
  included), the veto half of `_inherit_pending_streak_alarms` (the startup step keeps one pending-alarm
  read and the zero-progress half, §Sketch ε2b), the 3 veto-tracker containers,
  `_maybe_submit_stranded_verified_green`, `_stranded_merge_marker_is_fresh`, `_on_stranded_merge_done`
  from `harness.py`; ε2a's three transitional Harness callables (disposition emitter, veto-streak
  release, verified-green submit).
- Signal: `StrandedReconciler` holds no Harness reference (constructor signature test); the verified-green
  submit for a fixture stranded task emits the same `merge_enqueued` / `stranded_merge_failed` sequence
  as before (golden fixture).

**ε3 — `DeterministicReconSweep`: the deterministic recon sweep as a class**
(`task_kind=normal`, priority medium, ~700 LOC, 514 moved plus the module-level helpers, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/deterministic_recon.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_harness_deterministic_recon_sweep.py`, `orchestrator/tests/test_deterministic_recon.py` (new).
  Plus 3 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_harness_escalation_revalidation_sweep.py`, `test_recovery_emission_wiring.py`, `test_scheduled_deploy_hold_e2e.py`.
- Keys: contributes to `fk-bf4197a9a7d9`. Deps: ε2b (`RecoveryDispositions`); S3 ζ (lead sequencing
  ruling 2026-11-10: S3 ε1 → ε2 → ζ edit `harness.py` after ε2b and before ε3). Collaborators beyond the
  stores: the unit inspector (built once by the composition root and shared with
  `_run_scheduled_deploy_settle_pass`, which stays) and a `revalidate_open_l2` callable (the Harness
  method until ζ2 rebinds it, D8).
- Deletes: `_run_deterministic_recon_sweep`, `_recover_stranded_deterministic_task`,
  `_recover_stranded_deterministic_gate`, `_revalidate_deterministic_deploy_health`, the
  `_recon_unit_inspector` attribute and the module-level `_deterministic_*_stranded` helpers from
  `harness.py`.
- Signal: the sweep's `deterministic_recon_*` events for a fixture store match the pre-move golden set;
  the test suite injects the unit inspector by constructor, not by attribute (the about 30 private
  `_recon_unit_inspector` writes in `test_harness_deterministic_recon_sweep.py` go).
- Edge 9030 → ε3 at decompose: ε3 delivers 9030's unit-inspector seam (its item 2); its helper-module
  and backfill-test items stay.

**ζ1a — `DispatchGateFilers`: the dispatch-gate filers as a class**
(`task_kind=normal`, priority medium, ~800 LOC, 628 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/dispatch_gate_filers.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_harness_escalation_filing_identity.py`, `orchestrator/tests/test_harness_starvation_watchdog.py`,
  `orchestrator/tests/test_dispatch_gate_filers.py` (new).
  Plus 4 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_cross_project_dispatch_integration.py`, `test_cross_repo_gate.py`, `test_delivered_check_gate_e2e.py`, `test_substrate_gate.py`.
- Moves (9 methods, measured at `e634cfe658`): `_block_and_escalate_external_dep`,
  `_block_and_escalate_delivered_check`, `_block_and_escalate_cross_repo`,
  `_block_and_escalate_substrate_flip`, `_block_and_file_dispatch_gate_l1`, the generic
  `_file_harness_l1` it now files through (task 5516; `_file_restart_recut_unrecorded` stays with the
  restart-recut stratum and calls the class), `_file_starvation_info`, `_resolve_starvation_info`,
  `_file_unattributed_landing_escalation`.
- Keys: contributes to `fk-1e13b183c961`. Deps: ε3 (same file); out-of-batch 7170 (pending low; folds
  the three block-and-L1 copies onto `_block_and_file_dispatch_gate_l1` in today's file); S10 ξ,
  whichever is filed second.
- Signal: the filing-identity tripwire scans `harness.py` + the new module and counts the same sites
  (D9); each filed record for a fixture trip carries the same `summary`/`category`/`level` (golden).

**ζ1b — `InfraFilers`: the infrastructure filers and resolvers as a class**
(`task_kind=normal`, priority medium, ~1,500 LOC, 1,375 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/infra_filers.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_harness_escalation_filing_identity.py`, `orchestrator/tests/test_harness_config_unknown_keys.py`,
  `orchestrator/tests/test_dirty_tree_escalation.py`, `orchestrator/tests/test_harness_no_landings_breaker.py`,
  `orchestrator/tests/test_harness_watcher_supervisor.py`, `orchestrator/tests/test_warm_lane_structural_exhaustion.py`.
  Plus 7 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_crash_recovery.py`, `test_harness_lane_record_drift.py`, `test_harness_warm_lane_wiring.py`, `test_orphan_reaper.py`, `test_pool_storage_guard.py`, `test_singleton_lock.py`, `test_transcript_archive_startup_sweeper.py`.
- Moves (19 methods): `_file_archival_storm_escalation`, `_file_config_dir_ambiguous_escalation`,
  `_file_pool_storage_absent_escalation`, `_resolve_pool_storage_absent_escalation`,
  `_file_warm_base_hard_down_notice`, `_promote_warm_base_hard_down_l2`, `_resolve_warm_base_hard_down`,
  `_file_no_landings_info_escalation`, `_resolve_no_landings_info_escalation`,
  `_file_lane_record_drift_l2`, `_file_structural_exhaustion_l2`, `_file_dirty_tree_escalation`,
  `_file_config_unknown_keys_escalation`, `_file_watcher_outage_l2`, `_resolve_watcher_outage_l2`,
  `_file_reblock_guard_l2`, `_file_slot_wedged_l1`, `_escalate_reconcile_failure`,
  `_escalate_cascade_status_rejection`. The pause filers stay (§Out of scope). The
  `_pool_storage_absent_maybe_pending` flag becomes `InfraFilers` state beside its filer pair, and the
  composition root rebinds the transitional callables of ε1a and ε1b (pool-storage pair) and ε2a
  (`_escalate_reconcile_failure`) onto `InfraFilers` (D8).
- Keys: `fk-1e13b183c961`. Deps: ζ1a (same file); out-of-batch 8826 (pending low; moves four of these
  filers onto `born_at_l2.py::file_deduped_born_at_l2` in today's file). 8926's file-or-refresh helper
  already precedes ζ1b through β.
- Signal: as ζ1a; the archival-storm counters move with their filer (no counter lost).
- G7: INV-2 unchanged; INV-4 — the storm counters move with their filers.

**ζ2 — `EscalationRecordSweeps`: orphan-L0 reaper, stale strands and revalidation as a class**
(`task_kind=normal`, priority medium, ~1,000 LOC, 827 moved, measured at `e634cfe658`)
- Files: `orchestrator/src/orchestrator/escalation_record_sweeps.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_stale_escalation_cleanup.py`, `orchestrator/tests/test_harness_escalation_revalidation_sweep.py`,
  `orchestrator/tests/test_orphan_l0_reaper.py` (exists at `e634cfe658`),
  `orchestrator/tests/test_harness_escalation_filing_identity.py`.
  Plus 5 candidate test files in `orchestrator/tests/` reaching a moved name in code (measured at
  `bb1c659562`; §Decomposition plan): `test_harness_deterministic_recon_sweep.py`, `test_harness_main_tip_sweep.py`, `test_harness_watcher_supervisor.py`, `test_info_l0_router_harness.py`, `test_info_l0_router_workflow.py`.
- Moves (12 methods): `_reap_orphan_l0_escalations`, `_run_orphan_l0_reaper_pass`,
  `_snapshot_stale_l0_strands`, `_report_stale_l0_strands`, `_open_stale_l0_strand_records`,
  `_cite_open_stale_l0_strand_records`, `_file_stale_l0_strand_escalation`,
  `_route_info_l0s_across_restart`, `_close_superseded_main_sweep_escalations`, `_revalidate_open_l2`,
  `_revalidate_open_deterministic_escalation`, `_watcher_has_actionable_l1`; the composition root
  rebinds ε3's `revalidate_open_l2` callable onto the class (D8). After S9 ε, ζ2 also deletes the reaper
  freshness arms `_has_fresh_dispatch`/`_has_fresh_merge_phase`.
- Keys: contributes to `fk-1e13b183c961`. Deps: ζ1b; reads C1 (α); out-of-batch 5200 (pending; the
  strand-sweep retune 7262 waited on), 7385 (pending low; the reaper's L0→L1 promotion onto
  `l0_promotion.promote_l0_to_l1`, its prerequisite 7096 done) and 8659 (pending low; the public
  strand sentinel), each landing first in today's file; S9 ε, S10 ξ, whichever is filed second.
- Signal: boundary row 4; the reaper's `orphan_l0_*` events for a fixture queue match the golden set.
- Subsumes 7262 (cancel with reason; 5223 is done and 5200 is an explicit dependency).

**θ — durable steward spend per task, $25 default (S8-1)**
(`task_kind=normal`, priority medium, ~450 LOC)
- Files: `orchestrator/src/orchestrator/steward.py`, `orchestrator/src/orchestrator/config.py` (default
  `25.0` and its comment), `orchestrator/src/orchestrator/defaults.yaml` (`25.0`),
  `dark-factory-orchestrator.yaml` (the `steward_lifetime_budget: 20.0` line and its 22-line comment
  deleted), `orchestrator/src/orchestrator/workflow_types.py` (the budget-exhausted docstring),
  `OPERATIONS.md`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_steward_session_cost.py`,
  `orchestrator/tests/test_config.py`.
- Keys: `fk-a1085af3a290`. Deps: η and δ (`config.py`, this repo's yaml; §Decomposition plan); out-of-batch
  5819 (same file); S9 ε/ζ2/ξ/ω (`steward.py`), whichever is filed second. No `harness.py` edit; runs in parallel.
- One commit (D17): the durable spend, the two `25.0` defaults and the yaml deletion land and deploy
  together; the key is hot-reloadable, so a reload between a yaml edit and the code deploy would
  otherwise drop this repo to $12. No other repository is edited (no other project yaml sets the key,
  21 read on 2026-11-10).
- Deletes: this repo's `steward_lifetime_budget` override and its comment; adds the one durable home.
- Signal: boundary rows 22, 22b, 22c; the budget-exhausted L1 for a task names the durable total, the
  per-instance share, the $25 cap and "forgotten 7 days after the last steward spend".
- G7: INV-9 — spend has one durable home; INV-7 — the guard file carries the 7-day TTL every 5352 guard
  carries.
- Subsumes 3566 (cancel with reason, S8-1). Lead amends task 5822's budget item into a read-only spend
  distribution report (D17).

**ι — one event writer with read-your-writes inside `EventStore`**
(`task_kind=normal`, priority medium, ~650 LOC)
- Files: `orchestrator/src/orchestrator/event_store.py`, `orchestrator/src/orchestrator/harness.py`
  (event-writer `ManagedService` prepended), `orchestrator/tests/test_event_store.py`,
  `orchestrator/tests/test_event_store_writer.py` (new; rows 17, 17b, 18), `orchestrator/tests/test_sqlite_pragmas.py`.
- Keys: `fk-f1a7f57fa58e`. Deps: ζ2 (same file). Contract: lands C5 (S8-3, option A).
- Deletes: the per-event `_connect()` in `emit`.
- Signal: boundary rows 17, 17b, 18; measured in the task record: per-emit loop occupancy before/after
  on the verify host (`emit` returns in microseconds; one fsync per batch in `strace -c` of the writer
  thread).
- G7: INV-8 — the loop no longer blocks on fsync per event; worst case for a read through an
  `EventStore` method on the loop thread is one batch interval plus one commit (one fsync), never
  `flush`'s 5 s default; INV-11 —
  writer failures are loud; INV-13 — the dashboard still reads committed rows (lag bounded by the batch
  interval, stated in C5).
- Subsumes 4443 (cancel with reason). Lead amends 4197: its finding 2 is delivered here.

**ι2 — every in-process events read is an `EventStore` method; the raw-SQL guard (S8-3)**
(`task_kind=normal`, priority medium, ~600 LOC)
- Files: `orchestrator/src/orchestrator/event_store.py` (typed read methods), `orchestrator/src/orchestrator/digest.py`,
  `orchestrator/src/orchestrator/strict_event_reads.py` (its one reader moves onto `EventStore`),
  `orchestrator/src/orchestrator/scheduler.py` (`_load_work_product`'s call), `orchestrator/src/orchestrator/harness.py`
  (the digest service passes its `EventStore`), `orchestrator/tests/test_event_store_encapsulation.py`
  (new; rows 17c, 17d), `orchestrator/tests/test_digest.py`, `orchestrator/tests/test_digest_chain_telemetry.py`,
  `orchestrator/tests/test_digest_pause_and_storm.py`, `orchestrator/tests/test_strict_event_reads.py`,
  `orchestrator/tests/test_event_store.py`.
- Keys: `fk-f1a7f57fa58e` (closes, with ι). Deps: ι (same files); out-of-batch 8563 (pending high; its
  `landings()` rewrite of `query_count_done_in_window` lands first, and ι2 moves whatever raw readers
  remain).
- Deletes: the 7 raw SQL lines in `digest.py` and the 1 in `strict_event_reads.py` (measured at
  `e634cfe658`); the allow-list holds exactly the five offline tools and the printed recipe in
  `scheduler.py`, each with its reason.
- Signal: boundary rows 17c, 17d; `git grep -niE '(from|join)\s+events\b' -- orchestrator/src` lists
  only `event_store.py` and the allow-listed modules.
- G7: INV-8 — the digest's aggregate reads run through `asyncio.to_thread` from its coroutine (8563's
  pattern), so the move adds no loop occupancy; INV-9 — one reader surface for the events table in the
  writer's process. No invariant is promoted (R22; D19).
- Follow-up (not a task here): the shared `runs.db` read model for other packages, routed to S22.

**κ1 — the invocation carries its sandbox backend**
(`task_kind=normal`, priority medium, ~550 LOC)
- Files: `orchestrator/src/orchestrator/agents/sandbox_dispatch.py`, `orchestrator/src/orchestrator/agents/invoke.py`,
  `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/harness.py` (owns `SandboxRefusalDedup`),
  `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/src/orchestrator/evals/shadow_isolation.py`,
  `conftest.py`, `orchestrator/tests/conftest.py`, and the 9 other test files calling the deleted API
  (measured at `e634cfe658`): `orchestrator/tests/test_sandbox_dispatch.py`,
  `orchestrator/tests/test_workflow_sandbox_refusal.py`, `orchestrator/tests/test_workflow_sandbox_events.py`,
  `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_sandbox_git_config.py`,
  `orchestrator/tests/test_shadow_invocation_sandbox.py`, `orchestrator/tests/test_workflow_task_root.py`,
  `orchestrator/tests/test_landlock.py`, `orchestrator/tests/test_sandbox.py`. 17 files, above the
  15-file review trigger: one process global becomes a parameter, and every test site that set or reset
  it changes mechanically.
- Keys: `fk-a97501641592`. Deps: ι2 (same file `harness.py`); S9 ζ1 (`_invoke`) if filed.
- Deletes: `set_backend`, `get_backend`, `_preferred`, `_escalated_backend_state`, `reset_escalation_dedupe`,
  both `_restore_sandbox_backend` fixtures.
- Signal: boundary row 19; `dry_run_unblock` and the shadow-isolation probe resolve the configured
  backend with no `Harness` in the process.
- G7: INV-9 — the backend has one home per process, the composition root.

**κ2 — non-claude backends write under `.task/`; one claude spawn recipe**
(`task_kind=normal`, priority medium, ~450 LOC)
- Files: `orchestrator/src/orchestrator/agents/invoke.py`, `orchestrator/tests/test_invoke.py`,
  `orchestrator/tests/test_invoke_argv_parity.py`, `orchestrator/tests/test_invoke_agent_python_env.py`,
  `orchestrator/tests/test_invoke_agent_credentials.py`.
- Keys: `fk-58d1f3489c98`, `fk-9854bd9a928f`. Deps: κ1 (same file `agents/invoke.py`).
- Deletes: `_invoke_claude_with_sandbox`'s duplicated recipe and its two private imports, the `.mcp.json`
  rename/restore in `_invoke_pi`, the worktree-root writes in `_invoke_gemini`.
- Signal: boundary row 20; `git status --porcelain` in an agent worktree during a gemini or pi
  invocation lists nothing outside `.task/`; a sandboxed claude invocation with `ANTHROPIC_BASE_URL`
  overridden starts the `VllmBridge` (today it does not).
- G7: INV-11 — a tracked `.mcp.json` is a refusal, not a rename; INV-2 — the refusal names the path.
- Gemini/pi config levers are `OPEN` in the manifest (neither CLI is installed on the verify host); the
  fail-loud floor is the guaranteed deliverable. Subsumes 7311 (cancel with reason). Edge 6993 → κ2 at
  decompose (same functions).

**μ1 — the harness path carries its MCP session**
(`task_kind=normal`, priority low, ~450 LOC)
- Files: `orchestrator/src/orchestrator/mcp_lifecycle.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/cli.py`,
  `orchestrator/src/orchestrator/evals/runner.py`, `orchestrator/tests/_scheduler_factory.py`,
  `orchestrator/tests/test_mcp_lifecycle.py`, `orchestrator/tests/test_scheduler.py`.
- Keys: `fk-538940c8d347` (harness half). Deps: κ1 (same file `harness.py`); S9 ζ2 if filed.
- Deletes: `mcp_lifecycle._session`, `get_session`, the `rstrip('/mcp')` misparse, `Scheduler._memory_url`'s
  fallback branch; `Scheduler`'s `mcp_session` becomes required.
- Signal: boundary row 21; `mcp_call('http://fm.example.com/mcp', …)` targets `http://fm.example.com`
  (unit test on the derived base url).

**μ2 — briefing and recall carry the session; out-of-harness entry points own theirs (D13a)**
(`task_kind=normal`, priority low, ~600 LOC)
- Files: `orchestrator/src/orchestrator/agents/memory_recall.py`, `orchestrator/src/orchestrator/agents/briefing.py`,
  `orchestrator/src/orchestrator/cli.py`, `orchestrator/src/orchestrator/evals/runner.py`,
  `orchestrator/src/orchestrator/evals/architect_eval.py`, `orchestrator/src/orchestrator/evals/reviewer_trial/runner.py`,
  `orchestrator/src/orchestrator/evals/replay_frame.py`, `orchestrator/tests/test_memory_recall.py`, and the
  14 test files that build `BriefingAssembler`/`MemoryRecall` (measured at `e634cfe658`; 12 at `d40ba048e9`):
  `orchestrator/tests/_briefing_helpers.py`, `orchestrator/tests/conftest.py`, `test_architect_all_committed_plan.py`,
  `test_architect_exit_enumeration.py`, `test_briefing_debugger.py`, `test_briefing_implementer_variant.py`,
  `test_briefing_judge.py`, `test_briefing_progress_spot.py`, `test_briefing_steward.py`, `test_eval_driver.py`,
  `test_eval_replay_frame.py`, `test_harness_wip_step_detection.py`, `test_one_step_variant_boundary.py`,
  `test_reviewer_trial_runner.py`.
  Above the 15-file review trigger: one constructor-signature change over its construction sites.
- Keys: `fk-538940c8d347` (closes). Deps: μ1.
- Deletes: `MemoryRecall`'s url argument; every out-of-harness reliance on `mcp_call`'s per-call
  `fm_session` fallback.
- Signal: boundary row 21b; a `cli.py` replay run logs one `async with McpSession` open/close pair.

**ω — integration gate: boundary matrix green, `__new__` fixtures retired, the liveness forwarder deleted, harness.py re-measured**
(`task_kind=normal`, priority medium, ~800 LOC, mostly test deletion)
- Files: `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/test_harness_watcher_supervisor.py`,
  `orchestrator/tests/test_auto_eval.py`, `orchestrator/tests/test_conftest_helpers.py`,
  `orchestrator/tests/test_harness_reblock_guard.py`, `orchestrator/tests/test_harness_merge_registry_wiring.py`,
  `orchestrator/tests/test_harness_main_tip_sweep.py`, `orchestrator/tests/test_harness_claimant_clear.py`,
  `orchestrator/tests/test_watcher_rotation_cap_retry.py`, `orchestrator/tests/test_sqlite_pragmas.py`,
  `orchestrator/tests/test_harness_integration_gate.py` (new), `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-d470c7828682`, `fk-b1c6965066ed`. Deps: μ2 and κ2 (and every leaf, transitively); S3 ζ
  (the pin classifier's read is rebound before the forwarder goes).
- Deletes: the 39 `Harness.__new__(Harness)` sites (19 files, the same set as at `d40ba048e9`; the other
  nine files' sites go with the leaves that move their subjects), `_init_harness_state_for_test`, the
  remaining `getattr(self, '_…')` and `getattr(self.config, '…')` reads, the 9 fixture-justifying
  comments, and the `Harness.is_workflow_active` forwarder α kept.
- Signal: `git grep -c 'Harness.__new__(Harness)' orchestrator/tests` = 0; `grep -c "getattr(self, '_"
  harness.py` = 0; `def is_workflow_active` absent from `harness.py`; boundary rows 1–22c pass in CI
  against real fixtures (sqlite store, tmp git, tmp queue, tmp `project_root` for every real
  `Harness(config)`); the task record carries the post-batch heuristic-14 measurement of `harness.py`
  (lines, prose share, method count) beside the `e634cfe658` baseline (19,562 / 46.9% / 248).
- G7: INV-10 — the gate exercises behaviour against fixtures, never prose.

## Mechanical tasks and live-task dispositions (contract §8)

Step 1 (`find_tasks_by_metadata(key="x_finding_key")`) returned no task for any of the 19 keys
(author pass 2026-11-01; critic 2026-11-02; re-run 2026-11-10). Step 2 (`search_tasks` ≥ 0.6,
confirmed by `get_task`) found the live tasks below; the 2026-11-10 re-run also walked every live task
declaring `harness.py` in `metadata.files` (201 on 2026-11-10) and confirmed the ones a leaf's work
covers with `get_task`. Each is a dependency, an amendment or a cancellation naming the subsuming leaf,
never a second task. Statuses read with `get_statuses` on 2026-11-10.

| Task | Status | Relation | Action at decompose |
|---|---|---|---|
| 7698 construction seam for queue + already-landed gate | pending, low | subsumed by β | cancel, `x_acceptance_reason` names β |
| 6625 INV-11: `HAS_ESCALATION`/`HAS_STEWARD` guards in `_start_escalation_server` (key `fk-69b77cb9d60f`) | pending, medium | subsumed by β (D6: both guards deleted, the function-local `escalation.models` imports hoisted) | cancel, `x_acceptance_reason` names β |
| 6768 public way to attach `UsageGate`'s cost store | pending, low | subsumed by β (`UsageGate(..., cost_store=, project_id=, run_id=)` at construction; the private writes go) | cancel, reason names β |
| 7107 call `Scheduler.aclose()` at harness shutdown | pending, low | subsumed by β (D5: the scheduler's shutdown is a registered service) | cancel, reason names β |
| 7247 shutdown drain: bounded `asyncio.wait` for cancel-immune tasks | pending, low | same `finally` β rewrites | `add_dependency(β, 7247)` |
| 8559 scheduler snapshot flush at pause/resume/restore/shutdown | pending, high | same `finally` β rewrites | `add_dependency(β, 8559)`; β registers its shutdown flush (D5) |
| 8268 merge worker built at 1b, its loop after `finish_startup()` | in-progress | same `run()` startup region | `add_dependency(β, 8268)` if still live at decompose |
| 8269 crash-loop breaker around `Harness.run`'s startup | pending, high | same `run()` startup region | `add_dependency(β, 8269)` |
| 8270 watchdog restart-rate alarm | pending, high | pages a constructor raise (S8-8) | `add_dependency(β, 8270)` (ruled) |
| 8926 one file-or-refresh idiom for pending sentinel escalations | pending, low | β's store-failure L2 files through it | `add_dependency(β, 8926)` |
| 6143 Scheduler constructor seams (PSI, landed-outbox gate, hold history) | pending, low | landed-outbox item subsumed by β; rest S17 | `add_dependency(6143, β)`; note to S17 |
| 7603 public injection point for the harness escalation queue | pending, low | the seam is delivered by β; the 56-file test migration stays | `add_dependency(7603, β)`; description note |
| 8778 harness public seams: attach a queue, drain background tasks | pending, low | item 1 delivered by β; items 2–4 stay | `add_dependency(8778, β)`; description note |
| 8779 harness public test seams (verify transport, host pool, dispatcher, merge worker) | pending, low | re-scopes onto β's constructor arguments | `add_dependency(8779, β)` |
| 5536 extract the session-resume storm streak | pending, low | subsumed by γ | cancel, reason names γ |
| 6682 fold the second `get_task` into `derive_truth` | pending, low | subsumed by δ | cancel, reason names δ |
| 7277 startup reconcile re-pends a recovered-merge task | pending, high | δ consumes its guard as `TruthReport.merge_in_flight` | `add_dependency(δ, 7277)` |
| 6615 `_pid_alive` OverflowError (`task_ground_truth.py`) | in-progress (claimed, heartbeating 2026-11-10) | same file as δ | `add_dependency(δ, 6615)` if still live |
| 4651 landed reconciler for pending/merge-deferred | pending, high; held by the lead's temporary parking edge on 5040 (2026-11-10; its dependencies read 4623, 4646, 4647, 4673, 4887, 5040) | S8-2 = A: re-scoped onto `StrandedReconciler` | `add_dependency(4651, ε2a)`, then `remove_dependency(4651, 5040)` (the parking edge; nothing else removed); amend its description and two `harness.py` delivered checks onto `stranded_reconcile.py`/C2 spellings under the `RecoveryRule.statuses` partition invariant, its done-write through ε2a's `GatedDoneWriter` |
| 7436 public crash-recovery result accessor | pending, low | subsumed by ε1a | cancel, reason names ε1a |
| 7349 false "resume is lost" claims in `_adopt_recovered_session` | pending, low | same method as ε1a | `add_dependency(7349, ε1a)` |
| 7459 public seam for already-landed gate + stranded sweep | pending, low | sweep half delivered by ε2a; gate half stays | `add_dependency(7459, ε2a)`; description note |
| 9030 shared e2e helpers; harness unit-inspector seam | pending, low | item 2 delivered by ε3 | `add_dependency(9030, ε3)`; description note |
| 7170 fold the block-and-L1 copies onto `_block_and_file_dispatch_gate_l1` | pending, low | same methods ζ1a moves | `add_dependency(ζ1a, 7170)` |
| 8826 four born-at-L2 filers onto `file_deduped_born_at_l2` | pending, low | same methods ζ1b moves | `add_dependency(ζ1b, 8826)` |
| 7262 extract the stale-L0 strand cluster | pending, low | subsumed by ζ2 | cancel, reason names ζ2 |
| 5200 stale-L0-strand sweep retune | pending, medium | same cluster ζ2 moves (7262 waited on it) | `add_dependency(ζ2, 5200)` |
| 7385 reaper's L0→L1 promotion onto `l0_promotion` | pending, low | same reaper tail ζ2 moves | `add_dependency(ζ2, 7385)` |
| 8659 public stale-L0 strand sentinel constant | pending, low | same cluster ζ2 moves | `add_dependency(ζ2, 8659)` |
| 4443 `EventStore.emit` synchronous sqlite | pending, low | subsumed by ι | cancel, reason names ι |
| 4197 restore INV-8: blocking rmtree and event-store write | pending, high | its finding 2 is ι (its own recommendation (b)); findings 1 and 3–6 are not S8's | amend: finding 2 delivered by ι, `event_store.py` leaves its files |
| 8563 return brief; digest done-in-window via `to_thread` | pending, high | same `digest.py` function ι2 moves | `add_dependency(ι2, 8563)` |
| 7311 secret-env rule for codex/gemini/pi | pending, low | subsumed by κ2 | cancel, reason names κ2 |
| 6993 gemini ignores tool lists | pending, low | same functions as κ2 | `add_dependency(6993, κ2)` |
| 3566 rename `steward_lifetime_budget` | pending, low | S8-1 = A: subsumed by θ | cancel, reason names θ and the 2026-11-10 ruling |
| 5822 steward-mode gate | pending (deterministic, waits on 5393, 5820) | its "re-derive `steward_lifetime_budget`" item is superseded (S8-1) | amend: that item becomes a read-only report of durable per-task spend against the $25 cap; the main question is unchanged |
| 5819 steward fresh session per L0 | pending, high | same file as θ | `add_dependency(θ, 5819)` |
| 5153 registered-but-idle workflow reaper | pending, high | lands first; α absorbs its container into C1 | `add_dependency(9216, 5153)` wired 2026-11-11 (Leo's ruling, program doc R59); the reverse edge is withdrawn |
| 4686 PauseClass | done | closes `fk-b01dac97e6e9` | none (disposition `fixed:730794ca3b`) |
| 2168 / 2175 transition table + enforcement | done | the legality half of `fk-5cad48d24b6f` predates the run (`stale`) | none |
| 7332 already-landed gate branch-exists arm cost | done (`5e61becdf7`, 2026-11-06; was deferred on 2026-11-02) | out of scope (β only wires the gate) | none |

## Out of scope

- `escalation/src/escalation/server.py` and `orchestrator/mcp/ops_tools.py` are S3's: S3 C2 owns
  `WorkflowSlots`, and `release_workflow` moves onto `slots.wait_cleared` in S3's leaf. `escalation/` is not edited
  by this stream. Reporting `slot_kind` from `release_workflow` is S3's choice (xref).
- GitOps's own constructor ports, `GitOpsCallbacks` and `LaneLifecycle`, and (OQ-10 ruled (a)) their
  minimal harness-side wiring (S5 γ, ι).
- `stranded_verified_green.py`'s classification and the stranded-merge filer (S9 θ, R2); S8 moves the
  sweep's callers of them.
- Merge-lane recovery methods and `build_train_callback_factory` (S6; D14).
- The already-landed dispatch gate's internals (`_already_landed_dispatch_gate`, 381 lines measured at
  `e634cfe658`) — task 7332 landed its cost fix (`5e61becdf7`); live tasks 8075 and 8807 (pending) own the
  rest; β only wires it through C3, and ε2a only re-points its two done-write calls onto the injected
  `GatedDoneWriter`.
- The pause/digest stratum (8 methods, 585 lines at `d40ba048e9`), including `_file_scheduler_pause_escalation` and
  `_file_restored_pause_escalation`: live tasks 4687 and 4691 own it; task 2987's lane-record divergence
  digest (`_advance_lane_record_divergence_and_alarm`) belongs to it too.
- Task 5516's restart-recut stratum (`_issue_restart_recut_ruling`, `_reassert_restart_recut_ruling`,
  `_restart_recut_ruling`, `_file_restart_recut_unrecorded`, `_block_if_still_pending`) and task 7276's
  `scheduled-deploy-settle` service (its logic already in `scheduled_deploy_settle.py`).
- Offloading `derive_truth`'s synchronous filesystem reads (INV-8 at the per-task level; noted in ε2a).
- `fused-memory/src/fused_memory/reconciliation/sandbox_guard.py`'s mirror (S19).
- Raw `runs.db` SQL outside `orchestrator/src` (dashboard, fused-memory, client feed, quality series,
  scripts): the S8-3 follow-up routed to S22.
- Widening task 8269's breaker to wrap `Harness` construction in `cli.py` (offered in the S8-8
  discussion as optional, not ruled; 8270 pages a constructor raise).
- Any change to `docs/legibility/design-invariants.md`: R22 promotes nothing for S8, and the 2026-11-10
  sitting added none (INV-15/16/17 are S6's, S10's and S7's).

## Rulings (sitting of 2026-11-10)

Every open question of this plan was ruled on 2026-11-10. Sources: `digests/wave2-rulings-applied.md`
(§S8 and its cross-stream consequences), `digests/discussion/S8.md` (whose "what changes in the plan"
text is part of each changed ruling), `digests/wave2-leo-answers-2026-11-10.md` and
`digests/wave2-lead-rulings.md` (with its "revised 2026-11-10" block). None of them amends one of Leo's
2026-10-10 rulings; S8-5 confirms the lead's reading of R6.

| Question (sitting key) | Ruling | By |
|---|---|---|
| OQ-1 (S8-1) steward lifetime budget | Option A, durable per-task spend (θ, D17), 3566 cancelled; PLUS default cap $25 in `config.py` and `defaults.yaml` and this repo's `20.0` line deleted with its comment, all in the durable-spend commit; task 5822's budget item amended to a read-only spend-distribution report | Leo |
| OQ-2 (S8-2) task 4651 versus ε2a | Option A as recommended: 4651 depends on ε2a and is re-scoped onto `StrandedReconciler` with a tested status partition (the re-raise Leo's 2026-08-24 stop rule asked for) | Leo |
| OQ-3 (S8-3) event writer and read-after-write | Option A (batched `synchronous=FULL` writes, reads drain first, ι) PLUS encapsulating the orchestrator process: typed `EventStore` reads for `digest.py` and `strict_event_reads.py`, offline tools allow-listed, a guard test on raw events SQL in `orchestrator/src` (new leaf ι2, D19); "flush at named sites" dropped as subsumed; the cross-package read model is a follow-up to S22 | Leo |
| OQ-5 (S8-4) cancel on a deterministic slot | As recommended: refused (`SlotRegistry.cancel` False, `HardCancelRefused.DETERMINISTIC`) | Leo |
| OQ-6 (S8-5) R6's "both recovery sweeps" | As recommended (a): `CrashRecovery` and `StrandedReconciler` (D15); confirms the lead's 2026-11-02 reading | Leo |
| OQ-7 (S8-6) landing-evidence checks | As recommended (a): `derive_truth` computes `TruthReport.landing_evidence`; rows decide | Leo |
| OQ-8 (S8-7) `convert_to_blocked_enforce` | Option B, delete the field, in Leo's confirmed order: η flips the default to True; after its fleet-wide deploy the lead removes the key from the eight other yamls, best effort; δ deletes the field and this repo's line (D18) | Leo |
| OQ-9 (S8-8) store construction failure | Option A, fail closed (the recommendation was B): queue first, no degrade, retry only `SQLITE_BUSY`/`SQLITE_LOCKED` (~3 attempts, ~30 s), one critical born-at-L2 on a fixed sentinel, ERROR, raise; stand-in stores dropped; β depends on 8270 (D4a) | Leo |
| OQ-4 module placement | Flat modules under `orchestrator/src/orchestrator/` (S8-m1, accepted with Part 2) | lead (2026-11-02) |
| OQ-10 GitOps wiring | (a) = S5 OQ-4 A: no λ; β re-homes behind S5 γ/ι (S8-m3) | lead (2026-11-02) |
| S3 OQ-7 `WorkflowSlots` home | (a): S3 C2 is the one Protocol; `SlotRegistry` implements it (S8-m6). Amended here by the S3-1 / S8-2 cycle: the forwarder is deleted in ω, not β | lead (2026-11-02; amendment forced 2026-11-10) |
| `_on_stranded_merge_done` ownership and the ε2b ↔ S9 θ edge | moves into `StrandedReconciler`; ε2b depends on S9 θ, one-way (S8-m4, S8-m5) | lead (2026-11-02) |
| Cross-stream: S3-1 = A | no adapter; S3 ε1 depends on 4651 (reversed edge) — with S8-2 this forbids any S8 edge at or before ε2a onto S3 ε1/ε2/ζ, so β drops its S3 ζ edge | Leo (consequence derived here) |
| Cross-stream: S6-2 = B | 5040 first; β's S6 κ edge now also waits on 5040 | Leo |
| Cross-stream: S4-1 = A (VerifyHost) | the process host is resolved once by `VerifyHost.process()`; the composition roots use `process()`, never `fresh()`, and production code outside them never calls `VerifyHost.fresh()`; β keeps the one `VerifyHost.process()` call in the composition root; per-file order on `harness.py` β → S4 κ (lead, 2026-11-10) | Leo; order: lead |
| Cross-stream: S9-5 / S9-7 follow-ups; S8-3 follow-up | program-level follow-ups the lead files or routes; xref only here | Leo |
| Lead sequencing rulings after the ruling folds (2026-11-10) | the S8 ↔ S3 cycle cut CONFIRMED (β depends only on S3 β and δ; ω deletes the forwarder after S3 ζ); per-file order on `harness.py`: S3 ε1 depends on S8 ε2b and S8 ε3 depends on S3 ζ; task 4651 is parked on 5040 until S8's decompose seat adds 4651 → ε2a and removes that parking edge; `harness.py` order S6 α → S3 β, δ, κ → S6 β1 … μ → S3 ε1, ε2, ζ (S8 β follows S6 κ, α follows S3 δ) | lead |
| Lead rulings after the fold-verifiers (2026-11-10 late) | θ → δ stands; β → S4 κ on `harness.py` (no cycle: §Decomposition plan's walk); S10's split leaves named β1/β2 wherever S8 consumes them | lead |

## Open questions

None. No ruling raised a new design question, and the lead's sequencing rulings of 2026-11-10 settled
the one sequencing point the ruling fold left open (S3 ε1/ε2/ζ sit between ε2b and ε3 on `harness.py`).
Task 5153 versus α was ruled by Leo on 2026-11-11 (program doc R59): 5153 lands first and α (task 9216)
depends on it.

Tactical (decide in the leaf): the batch interval of ι (the loop-thread drain bound follows from it); whether
`SlotRecord.dispatched_at` is monotonic or wall-clock (5153 will want wall-clock for its L1 text); the
exact `LaneEvidence` fields in ε1a; where `_RECONCILE_SWEEP_STATUSES` is spelled once the sweep is a
class; the name of the archive hook argument in β; the typed read method names in ι2; the retry
spacing inside D4a's ~30 s budget.
