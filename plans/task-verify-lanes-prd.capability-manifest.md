# Capability manifest: task-verify lanes

PRD: `plans/task-verify-lanes-prd.md`. Pointers were resolved against main `1fbd702a1c` (2026-11-18).
Rendered from the YAML sidecar beside this file, which is the home of the capability list and verdicts.

Signal leaf: ε.

## α. Task-verify lanes on the executing host

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| verify-task-worktree-acquisition-seam | `grep:orchestrator/src/orchestrator/cli.py::_verify_task_body` wired: the one place verify-task obtains its worktree today | PASS | none (on-main or upstream evidence) |
| clean-retaining-build-artifact-dirs | `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.reset_persistent_offline_deep_worktree` wired: an existing persistent lane already resets through GitOps._clean_lane_retaining_artifacts with git.reap_build_artifact_dirs | PASS | none (on-main or upstream evidence) |
| lane-lock-helper | `grep:orchestrator/src/orchestrator/verify_cancel.py::acquire_merge_verify_flock` `grep:orchestrator/src/orchestrator/verify_cancel.py::lane_lock_path` wired: non-blocking with timeout 0; descriptor not inherited | PASS | none (on-main or upstream evidence) |
| live-reference-scan | `grep:shared/src/shared/proc_group.py::scan_process_groups_under_path` exists; used read-only (nothing is reaped) | PASS | none (on-main or upstream evidence) |
| cancel-leaves-non-merge-band-alone | `grep:orchestrator/src/orchestrator/cli.py::cancel_verify` wired: _recorded_merge_worktree_remover refuses anything but a _merge- entry under the worktree base | PASS | none (on-main or upstream evidence) |
| cold-classification-after-reset | `grep:orchestrator/src/orchestrator/task_verify_run.py::run_task_verify_on_worktree` wired: prepare_task_verify_worktree creates .task/ fresh and verify.py::_is_verify_cold then reports cold | PASS | none (on-main or upstream evidence) |
| wire-fingerprint-pin | `grep:orchestrator/tests/test_remote_gate.py` exists: its pinned wire vector must be re-pinned when VerifyResult gains fields | PASS | none (on-main or upstream evidence) |
| real-git-fixture-for-verify-task | `grep:orchestrator/tests/_task_verify_fixtures.py::seed_verify_repo` exists | PASS | none (on-main or upstream evidence) |
| lane-config-keys | built here: GitConfig.task_verify_lanes, task_verify_lane_root, task_verify_lane_min_free_gib (C1) | PASS | grep `task_verify_lane_root` present in orchestrator/src/orchestrator/config.py |
| lane-claim-prepare-fallback | built here: the lane module with the note enum (C2-C6) | PASS | grep `lane_root_unavailable` present in orchestrator/src/orchestrator/ |
| lane-facts-on-result | built here: three optional fields on VerifyResult (C7) | PASS | grep `host_lane_kept` present in orchestrator/src/orchestrator/ |
| verify-lanes-status-and-reset | built here: the verify-lanes command group (C10) | PASS | grep `verify-lanes` present in orchestrator/src/orchestrator/ |
| lane-boundary-rows-1-17 | built here: orchestrator/tests/test_task_verify_lane.py (PRD section 6 rows 1-17) | PASS | path orchestrator/tests/test_task_verify_lane.py present |

## β. Dispatcher record of lane facts and the degraded-lane alarm

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| lane-facts-on-result | producer:task-α upstream: VerifyResult carries host_lane, host_lane_kept, host_lane_note | PASS | none (on-main or upstream evidence) |
| remote-result-received-here | `grep:orchestrator/src/orchestrator/task_verify_dispatch.py::TaskVerifyDispatcher._remote_verdict` wired: every remote result passes through it before _settle_remote discards it on cross-check and red-confirm | PASS | none (on-main or upstream evidence) |
| dispatch-record-writer | `grep:orchestrator/src/orchestrator/task_verify_dispatch.py::_write_sidecar` wired: writes DispatchDecision to data/verify-logs/<task>/attempt-N.dispatch-*.json | PASS | none (on-main or upstream evidence) |
| dispatcher-construction-site | `grep:orchestrator/src/orchestrator/harness.py::Harness.__init__` wired: the only place TaskVerifyDispatcher is constructed; the escalation queue is None there | PASS | none (on-main or upstream evidence) |
| escalation-queue-late-binding | `grep:orchestrator/src/orchestrator/harness.py::Harness._start_escalation_server` wired: creates the queue and late-binds the sinks that need it | PASS | none (on-main or upstream evidence) |
| l1-dedup-and-resolve-precedent | `grep:escalation/src/escalation/queue.py::EscalationQueue.has_open_l1` `grep:orchestrator/src/orchestrator/merge_lane/liveness.py::_alarm_verify_host_skew` wired: sentinel dedup and resolve-on-recovery | PASS | none (on-main or upstream evidence) |
| registered-category-infra-issue | `grep:escalation/src/escalation/categories.py` (INFRA_ISSUE = 'infra_issue') | PASS | none (on-main or upstream evidence) |
| lane-facts-on-dispatch-record | built here: DispatchDecision carries the three facts, remote_runner and recorded_at for every dispatch that obtained a remote result (C8) | PASS | grep `remote_runner` present in orchestrator/src/orchestrator/task_verify_dispatch.py |
| degraded-lane-alarm | built here: the alarm module and its sentinel (C9) | PASS | grep `__task_verify_lane_degraded__` present in orchestrator/src/orchestrator/ |
| integration-row-21 | built here: real verify-task behind the ssh seam, two dispatches, second record kept true | PASS | grep `host_lane` present in orchestrator/tests/test_task_verify_integration.py |

## γ. Host provisioning script, in-service check and operations notes

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| lane-config-keys | producer:task-α upstream: the three keys the installer writes | PASS | none (on-main or upstream evidence) |
| verify-lanes-status-and-reset | producer:task-α upstream: the command OPERATIONS.md documents | PASS | none (on-main or upstream evidence) |
| lane-facts-on-dispatch-record | producer:task-β upstream: the lane facts, remote_runner and recorded_at the in-service check reads | PASS | none (on-main or upstream evidence) |
| ssh-shim-test-pattern | `grep:scripts/tests/test_enable_laptop_persistent_worktree.py` `grep:scripts/deploy/enable_laptop_persistent_worktree.sh` exist: fake ssh shims and yaml edit with backup | PASS | none (on-main or upstream evidence) |
| installer | built here: scripts/deploy/install_laptop_task_verify_lanes.sh with --check | PASS | path scripts/deploy/install_laptop_task_verify_lanes.sh present |
| in-service-check | built here: scripts/deploy/check_task_verify_lanes_in_service.py, exits 0 / 75 / 2 / 1 | PASS | path scripts/deploy/check_task_verify_lanes_in_service.py present |
| operations-section | built here: OPERATIONS.md section on task-verify lanes | PASS | grep `verify-lanes reset` present in OPERATIONS.md |

## ε. Enable task-verify lanes for reify on leo-laptop (operator gate)

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| installer | producer:task-γ upstream | PASS | none (on-main or upstream evidence) |
| in-service-check | producer:task-γ upstream | PASS | none (on-main or upstream evidence) |
| degraded-lane-alarm | producer:task-β upstream: fallbacks after enabling are visible | PASS | none (on-main or upstream evidence) |
| lane-claim-prepare-fallback | producer:task-α upstream | PASS | none (on-main or upstream evidence) |
| dispatcher-code-commit-readable | `grep:scripts/orchestrator-watchdog.py` (--report prints each orchestrator's CODE-SHA) | PASS | none (on-main or upstream evidence) |
| lanes-in-service-for-reify | built here (by the operator): the in-service check exits 0 against /home/leo/src/reify/data/verify-logs | PASS | manual: host state on leo-laptop and reify dispatch records; not expressible as a check on the dark-factory tree |
