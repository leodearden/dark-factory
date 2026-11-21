# Capability manifest: one canonical build path and one shared compiler cache for reify

PRD: `plans/canonical-build-path-shared-cache-prd.md`. Pointers were resolved against main `e12bf0532c` (2026-11-18).
Rendered from the YAML sidecar beside this file, which is the home of the capability list and verdicts.
Blocks δ, β1, β2, γ and λ are produced by reify tasks (`external_task_id`); their evidence is in the reify repository.

Signal leaf: κ.

## α. Measure the drafted shape on real reify, on both hosts (operator gate)

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| trial-harness | built here (by the operator): the study's trial harness at plans/sccache-path-key-study-2026-11-18/trial/ (untracked, beside the study), adapted to the PRD's shape | PASS | manual: an experiment on two hosts using untracked study scripts; its record is the gate's resolution |
| real-reify-cross-host-measurement | built here (by the operator): hit and miss counters on the second host, missed units, read-only server count, a sandboxed agent session behind the bind prefix | PASS | manual: host measurements recorded in the resolution; not expressible as a check on the dark-factory tree |

## δ. Liveness checks see processes inside a mount namespace (reify lib_live_refs.sh)

Producer: reify task 10228.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| namespace-path-translation-shell | built here: reify scripts/lib_live_refs.sh live_ref_present translates paths through a process's mount namespace (C11) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |

## β1. The primitive: build-directory modes, namespace entry, trampoline

Producer: reify task 10229.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| measured-shape | producer:task-α upstream (dark-factory): the drafted shape passes on real reify before the primitive is built | PASS | none (on-main or upstream evidence) |
| liveness-before-namespaces | producer:task-δ upstream: reify's deletion gates see namespaced processes before anything creates them | PASS | none (on-main or upstream evidence) |
| primitive-interface | built here: reify scripts/canon_run.py with exec, cargo and status (C1-C3), incremental policy end to end | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| cargo-trampoline | built here: reify scripts/canon-bin/cargo with the pass-through list of decision 10 | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |

## β2. Private sccache server, guard, store relay, local fallback and the build_cache event

Producer: reify task 10232.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| primitive-interface | producer:task-β1 upstream | PASS | none (on-main or upstream evidence) |
| private-cache-server | built here: a foreground sccache server per outermost invocation on a unix socket (C4) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| store-relay-and-fallback | built here: probe, loopback relay, watchdog cut-off, local fallback (C5) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| machine-flag-guard | built here: reify scripts/canon-bin/guard/sccache (decision 20) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| build-cache-event | built here: one build_cache event per outermost invocation on REIFY_VERIFY_EVENT_LOG (C6) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |

## γ. Hooks and warm-base scripts call the primitive; the seed honours the build-root stamp

Producer: reify task 10230.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| primitive-interface | producer:task-β1 upstream | PASS | none (on-main or upstream evidence) |
| owned-launch-sites | built here: reify hooks and warm-base scripts run through canon_run.py exec with class verify (C12) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| seed-honours-stamp | built here: reify scripts/seed-warm-lane.sh skips relocation and relink for a stamped base, and relocates from the canonical path for a lane without the primitive (C12) | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |

## ε. Process-liveness scans translate paths through mount namespaces

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| process-group-scan | `grep:shared/src/shared/proc_group.py::scan_process_groups_under_path` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.reap_merge_verify_survivors` wired: the scan feeds the survivor reap, so a positive answer signals processes | PASS | none (on-main or upstream evidence) |
| cpu-ticks-probe | `grep:shared/src/shared/proc_group.py::cpu_ticks_under_path` `grep:orchestrator/src/orchestrator/merge_lane/liveness.py::verify_tree_cpu_ticks` wired: the merge progress watch reads working directories only | PASS | none (on-main or upstream evidence) |
| shell-library-copy | `grep:orchestrator/scripts/warm-lane/lib_live_refs.sh` `grep:orchestrator/tests/test_live_refs_lib.py` exist: dark-factory's copy of live_ref_present and its test | PASS | none (on-main or upstream evidence) |
| namespace-path-translation | built here: translation through a process's mount namespace in proc_group.py and the shell copy (C11) | PASS | grep `ns/mnt` present in shared/src/shared/proc_group.py |

## ζ. Launch wiring: verify legs through the build primitive; sandboxed agents bound before landlock

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| verify-leg-spawn-site | `grep:orchestrator/src/orchestrator/verify.py::run_verification` wired: its nested _run_leg_command starts each leg as /bin/bash -c <string>, after the command rewriters | PASS | none (on-main or upstream evidence) |
| offline-lane-launch | `grep:orchestrator/src/orchestrator/offline_lane_runs.py::launch` wired: the offline lane starts its runs here, outside run_verification | PASS | none (on-main or upstream evidence) |
| sandbox-wrap-site | `grep:orchestrator/src/orchestrator/agents/sandbox_dispatch.py::wrap_command` `grep:orchestrator/src/orchestrator/agents/invoke.py::_sandbox_wrap` wired: where the landlock helper is placed in front of a sandboxed agent command | PASS | none (on-main or upstream evidence) |
| agent-environment | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._build_agent_env` wired: the environment every agent launch gets | PASS | none (on-main or upstream evidence) |
| reloadable-config | `grep:orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` exists: a section listed here is hot-reloadable | PASS | none (on-main or upstream evidence) |
| vanished-worktree-detection | `grep:orchestrator/src/orchestrator/verify.py::run_main_tip_sweep` wired: its nested _enoent_on_self matches the worktree path in error text | PASS | none (on-main or upstream evidence) |
| primitive-interface | producer:task-β1 upstream (reify): scripts/canon_run.py exec -- <cmd> | PASS | none (on-main or upstream evidence) |
| same-file-predecessor | producer:task-10736 upstream: task-verify lanes α edits config.py and verify.py first | PASS | none (on-main or upstream evidence) |
| build-canon-config | built here: the build_canon section (root, shim_dir, expect_canon) (C8) | PASS | grep `build_canon` present in orchestrator/src/orchestrator/config.py |
| leg-prefix-and-session-bind | built here: legs started through the primitive; bwrap bind before the landlock helper (C8) | PASS | grep `canon_run` present in orchestrator/src/orchestrator/ |

## η. build_cache facts on the verify result, the observation log and the degraded-cache alarm

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| event-channel-parser | `grep:orchestrator/src/orchestrator/verify_event_channel.py::_channel_from` wired: unknown event names are ignored, so a new event is back-compatible | PASS | none (on-main or upstream evidence) |
| result-aggregation | `grep:orchestrator/src/orchestrator/verify.py::_aggregate_results` `grep:orchestrator/src/orchestrator/verify.py::VerifyResult` wired: legs are folded here field by field | PASS | none (on-main or upstream evidence) |
| remote-wire | `grep:orchestrator/src/orchestrator/verify_runner.py::result_from_dict` `grep:orchestrator/src/orchestrator/remote_gate.py::wire_fingerprint` wired: the strict decoder and the fingerprint computed from field shapes | PASS | none (on-main or upstream evidence) |
| task-verify-tap | `grep:orchestrator/src/orchestrator/task_verify_dispatch.py::TaskVerifyDispatcher.__call__` wired: every task verify, local and remote, returns here | PASS | none (on-main or upstream evidence) |
| merge-verify-tap | `grep:orchestrator/src/orchestrator/verify_runner.py::VerifyRunnerPool.dispatch` wired: every merge verify returns here | PASS | none (on-main or upstream evidence) |
| late-bound-sinks | `grep:orchestrator/src/orchestrator/harness.py::Harness._start_escalation_server` wired: where the escalation queue exists and sinks are bound | PASS | none (on-main or upstream evidence) |
| build-cache-event | producer:task-β2 upstream (reify): the event the primitive writes | PASS | none (on-main or upstream evidence) |
| build-canon-config | producer:task-ζ upstream: build_canon.expect_canon | PASS | none (on-main or upstream evidence) |
| lane-alarm-ladder | producer:task-10737 upstream: the lane alarm whose ladder is extracted | PASS | none (on-main or upstream evidence) |
| build-cache-result-field | built here: VerifyResult.build_cache folded from the leg events (C9) | PASS | grep `build_cache` present in orchestrator/src/orchestrator/verify.py |
| observation-log | built here: one row per observed result in data/verify-logs/build-cache-observations.jsonl (C10) | PASS | grep `build-cache-observations` present in orchestrator/src/orchestrator/ |
| shared-streak-ladder | built here: one ladder module used by both alarms | PASS | path orchestrator/src/orchestrator/streak_alarm.py present |
| degraded-cache-alarm | built here: the alarm under the sentinel __build_cache_degraded__<host> (C10) | PASS | grep `__build_cache_degraded__` present in orchestrator/src/orchestrator/ |

## θ. Shared cache store service, pruning, drift probe, in-service check and operations notes

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| measured-shape | producer:task-α upstream: the store is proven on real reify before it is installed as a service | PASS | none (on-main or upstream evidence) |
| deploy-scripts-home | `grep:scripts/deploy` `grep:scripts/tests` exist: host installers and their tests live here | PASS | none (on-main or upstream evidence) |
| store-installer | built here: scripts/deploy/install_build_cache_store.sh with --check (C13) | PASS | path scripts/deploy/install_build_cache_store.sh present |
| store-pruner | built here: scripts/deploy/build_cache_store_prune.py (C13) | PASS | path scripts/deploy/build_cache_store_prune.py present |
| drift-probe | built here: scripts/deploy/build_cache_drift_probe.sh (C13) | PASS | path scripts/deploy/build_cache_drift_probe.sh present |
| in-service-check | built here: scripts/deploy/check_build_cache_in_service.py, exits 1 / 0 / 2 / 75 (C13) | PASS | path scripts/deploy/check_build_cache_in_service.py present |
| operations-section | built here: OPERATIONS.md section on the canonical build path and the shared store | PASS | grep `reify-canon` present in OPERATIONS.md |

## λ. Interactive sessions: setup-dev installs the trampoline; GUI launch scripts; CLAUDE.md

Producer: reify task 10231.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| private-cache-server | producer:task-β2 upstream | PASS | none (on-main or upstream evidence) |
| interactive-install | built here: reify scripts/setup-dev.sh installs the trampoline into ~/.local/share/reify-canon/bin | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |
| gui-launch-at-canon | built here: reify run-gui.sh and run-gui-dev.sh launch through canon_run.py exec | PASS | manual: the producer is a reify task; its files are in the reify repository, outside this tree |

## ι. Enable the canonical build path and shared cache for reify on leo-laptop (operator gate)

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| real-reify-cross-host-measurement | producer:task-α upstream | PASS | none (on-main or upstream evidence) |
| private-cache-server | producer:task-β2 upstream (reify) | PASS | none (on-main or upstream evidence) |
| owned-launch-sites | producer:task-γ upstream (reify) | PASS | none (on-main or upstream evidence) |
| namespace-path-translation-shell | producer:task-δ upstream (reify) | PASS | none (on-main or upstream evidence) |
| namespace-path-translation | producer:task-ε upstream | PASS | none (on-main or upstream evidence) |
| leg-prefix-and-session-bind | producer:task-ζ upstream | PASS | none (on-main or upstream evidence) |
| observation-log | producer:task-η upstream | PASS | none (on-main or upstream evidence) |
| degraded-cache-alarm | producer:task-η upstream: fallbacks after enabling are visible | PASS | none (on-main or upstream evidence) |
| store-installer | producer:task-θ upstream | PASS | none (on-main or upstream evidence) |
| in-service-check | producer:task-θ upstream | PASS | none (on-main or upstream evidence) |
| laptop-in-service | built here (by the operator): the in-service check exits 0 for leo-laptop | PASS | manual: host state on leo-laptop and reify observation rows; not expressible as a check on the dark-factory tree |

## κ. Enable the canonical build path and shared cache for reify on the workstation (operator gate)

Producer: dark-factory task.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| laptop-in-service | producer:task-ι upstream | PASS | none (on-main or upstream evidence) |
| seed-honours-stamp | producer:task-γ upstream (reify) | PASS | none (on-main or upstream evidence) |
| interactive-install | producer:task-λ upstream (reify) | PASS | none (on-main or upstream evidence) |
| workstation-in-service | built here (by the operator): the in-service check exits 0 for the workstation and an interactive build of main in a never-built worktree shows no workspace-library misses | PASS | manual: host state on the workstation and reify observation rows; not expressible as a check on the dark-factory tree |
