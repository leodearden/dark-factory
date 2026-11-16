# Capability manifest: record CPU and memory use by consumer on the factory hosts

**PRD:** `plans/per-consumer-resource-recording-prd.md` · **Sidecar:** `plans/per-consumer-resource-recording-prd.capability-manifest.yaml`
**Code anchors** verified against main `fad550f26f` (2026-11-16; the PRD was written at `3b490319d9`). Cited by symbol.

## Resolution of the first draft's FAIL bindings (2026-11-16)

A first draft found three FAIL bindings, all gaps in the PRD's section 8 graph rather than missing substrate:

- **Leaf 2.** `project-slice-name` imported task 3394's slice-name rule. C2 was relaxed instead: the project consumer
  is keyed by the last segment of the hold's `own_cgroup`. Leaf 2 does not wait on the slices.
- **Leaf 7b.** `cgroup-file-parser-import` needed leaf 1's parser. Edges 1 → 7a and 1 → 7b were added.
- **Leaf 8.** `report-script-and-its-scaffolding` needed leaf 2's script. Edge 2 → 8 was added.

Every row below is now PASS.

**Notes that are not FAIL** (no capability is missing; these are batch-wiring points for decompose):

- Leaf 7a: wire task 10429 (allowance reserve 1b) and task 9850 (stream 20 task 9850, moves the spawn to `shared/claude_runner.py`) as same-file serialization. The task the PRD calls "stream 20's per-attempt writer" is **9846** (stream 20 task 9846, from that PRD's sidecar), which depends on 9812.
- Leaf 4 (existing task 6355): review-all stream 4's moves already wait on it (task 9130 depends on 6355), so it serializes behind tasks 3394 and 8078 instead. `_run_cmd` pointers resolve on main today.
- Leaf 3: a pure gate (`always_escalates`, no `before_done`) cannot run the report; its escalation body carries the command. The PRD says "Check: runs ...". If the runner must run it, the leaf is act-then-ask.
- Leaf 7b: task 9985 is the production signal, not a code dependency; no edge 9985 -> 7b. The needed edge 6 -> 9985 is on the existing task.
- Leaf 2 "merged and writing in production" (INV-13) is a rollout condition, not a code capability.
- `signal_leaves`: `["3", "8"]`. Every chain ends in one of them (1 -> 2 -> 3; 1 -> 4 -> 5 -> 8; 3394 -> 6 -> 7b -> 8; 7a -> 7b), so a delivered mark needs the whole graph. Leaf 3 carries the goal "whose memory causes the holds" (goal items 1 to 4); leaf 8 carries "what verify runs and agents cost" (goal items 5 to 7). Leaf 3 is dated 7 days after leaf 2, so the PRD reads delivered no earlier than that; leaf 8 alone would flip it sooner, so both are named.

Each grep pointer was resolved against main with `git cat-file -e main:<path>` and the name's definition looked up in `git show main:<path>` (an ast walk: module-level def, class or assignment, or Class.method). There is no manifest-pointer resolver script in this repo. Each mechanical `delivered_check` was run with `git grep -E` against main and with `shared.delivered_check_polarity.lint_delivered_checks`: every `present` check matches nothing on main, the one `absent` check matches once (`sampler/src/sampler/metrics.py`), and none is filename-shaped, self-referential or comment-only.

Legend: `producer:task-N upstream` is a leaf label of this PRD (or a real task id when 4 digits) in the transitive closure; `built here` is delivered by the leaf itself and carries a `delivered_check` (mechanical where a name is PRD-fixed, manual otherwise).


---

## Leaf 1: Record every consumer's memory, swap, reclaim, stall and CPU

*New leaf; absorbs task 7858 part 2 and task 7006 item 2 (re-scoped at decompose). No prerequisites.*

- `metric-table-and-vocabulary-guard` → `grep:sampler/src/sampler/__main__.py::main` `grep:sampler/src/sampler/metrics.py::METRIC_GROUPS` `grep:sampler/tests/test_vocabulary_one_table.py::_scope` wired: main runs every group of the table; the scan is the guard. **PASS**.
- `due-time-state-substrate` → `grep:sampler/src/sampler/store.py::LoadSampleStore.should_cleanup` `grep:sampler/src/sampler/sampler.py::run_tick` wired: should_cleanup reads the meta-table due time through _is_due; run_tick is the tick write. **PASS**.
- `pressure-file-parser-reused` → `grep:sampler/src/sampler/metrics.py::_read_leaf_some_avg10` `grep:shared/src/shared/psi.py::parse_pressure_file` wired: the sampler's reader calls the shared parser. **PASS**.
- `served-group-table` → `grep:dashboard/src/dashboard/data/load.py::_SERVED_GROUPS` `grep:dashboard/src/dashboard/static/redux/tab_overview.jsx` the module-level table the verify-scope group is added to; the tiles are JSX names, not Python-checkable. **PASS**.
- `old-verify-scope-tag-rule` → `grep:orchestrator/src/orchestrator/verify.py::reap_leftover_verify_scopes` `grep:orchestrator/src/orchestrator/verify.py::_scope_tag_for` wired: reap_leftover_verify_scopes calls _scope_tag_for, the rule the shared one is cross-checked against. **PASS**.
- `stateful-group-contract` → built here; previous state in, samples and new state out, persisted with the rows and the next due time in one transaction. **PASS**.
  - delivered_check: manual. The new names are the architect's; the boundary rows for a crash mid-tick and a recorder gap prove it, not a grep.
- `cgroup-file-parser` → built here; shared/src/shared/cgroup_stats.py, the one parser for memory.stat, cpu.stat and memory.events (D7). **PASS**.
  - delivered_check: path present: shared/src/shared/cgroup_stats.py
- `consumer-classifier-module` → built here; sampler/src/sampler/consumers.py, the one rule table, walk, residuals and consumer-name encode and parse (D2). **PASS**.
  - delivered_check: path present: sampler/src/sampler/consumers.py
- `refault-and-swap-in-counters-read` → built here; memory.stat keys pswpin, pgsteal_kswapd, pgsteal_direct and workingset_refault_anon and _file read per cgroup (C1); absent on main. **PASS**.
  - delivered_check: grep present, pattern `workingset_refault_anon`, paths sampler/src/sampler/, shared/src/shared/
- `reclaim-thread-cpu-recorded` → built here; the background reclaim and compaction threads' CPU from /proc/<pid>/stat, kswapd and kcompactd (C1); absent on main. **PASS**.
  - delivered_check: grep present, pattern `kswapd`, paths sampler/src/sampler/
- `two-zero-series-retired` → built here; the verify_rss_total_bytes metric declaration leaves the table (D6); present in metrics.py on main. **PASS**.
  - delivered_check: grep absent, pattern `MetricSpec\('verify_rss_total_bytes'`, paths sampler/src/sampler/metrics.py
- `verify-scope-tiles-swapped` → built here; the host load card reads live verify scopes and their memory in place of the two zero tiles. **PASS**.
  - delivered_check: manual. Operator-visible: while a verify scope is live the card shows non-zero values. The tile list is JSX and its old keys also sit in a comment, so no grep is sound.
- `shared-scope-tag-rule` → built here; one public home in shared/src/shared/ that builds and parses the verify scope tag, with a cross-check test against _scope_tag_for. **PASS**.
  - delivered_check: manual. The module and function names are the architect's; guessing them in a pattern could wedge leaves 2 and 4 forever.
- `consumer-rows-in-the-store` → built here; the consumer rows for the named consumers appear in data/load-samples.db within a minute of the merge. **PASS**.
  - delivered_check: manual. The goal signal; an sqlite read of the live store on the host after deploy.
- `one-journal-summary-line-per-collection` → built here; sampler __main__ prints one summary line for the group, never its rows. **PASS**.
  - delivered_check: manual. Observed in the user journal; no mechanical form.
- `retention-cleanup-timing` → built here; one cleanup timed against a copy of the store at about 8 times its rows, an index led by ts added above 2 seconds. **PASS**.
  - delivered_check: manual. A measurement recorded in the task report.

---

## Leaf 2: Report which consumers grew and which paid before each memory hold

*C2 relaxed 2026-11-16: the project consumer is keyed by the last segment of the hold's own_cgroup, so this leaf imports nothing from task 3394.*

- `consumer-samples-written` → producer:task-1 upstream. **PASS**.
- `shared-scope-tag-rule-for-the-verify-mapping` → producer:task-1 upstream. **PASS**.
- `hold-events-carry-own-cgroup` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._note_heavy_deferral` `grep:orchestrator/src/orchestrator/event_store.py::EventType` wired: _note_heavy_deferral emits dispatch_deferred with the whole pressure sample, own_cgroup included. **PASS**.
- `project-id-from-the-root` → `grep:orchestrator/src/orchestrator/config.py::OrchestratorConfig._derive_fused_memory_project_id` `grep:orchestrator/src/orchestrator/config.py::load_config` wired: the config derives the project id through shared.project_id.project_id_for_root. **PASS**.
- `default-project-roots` → `grep:dashboard/src/dashboard/config.py::DashboardConfig.all_project_roots` the dashboard's known roots, which the script defaults to. **PASS**.
- `vocabulary-scan-covers-the-report` → `grep:sampler/tests/test_vocabulary_one_table.py::_scope` the scan list the script is added to; the script's name is not in it on main. **PASS**.
  - delivered_check: grep present, pattern `resource_report`, paths sampler/tests/test_vocabulary_one_table.py
- `project-mapping-by-own-cgroup` → built here; C2 maps service, agents and project consumers by the last segment of the hold's sample.own_cgroup. **PASS**.
  - delivered_check: manual (exercised by the PRD section 7 'Hold mapping, both shapes' tests)
- `holds-subcommand` → built here; scripts/resource_report.py with the holds subcommand, --summary, --json and repeated --project-root. **PASS**.
  - delivered_check: path present: scripts/resource_report.py
- `recorder-not-live-state` → built here; a hold before the first consumer sample prints recorder not live, distinct from an empty ranking (INV-13); absent on main. **PASS**.
  - delivered_check: grep present, pattern `recorder not live`, paths scripts/

---

## Leaf 3: Decision task: a week of consumer data on the memory holds

*Signal leaf with 8. A pure gate (task_kind deterministic, always_escalates, no before_done): its only dispatch action is the escalation, so the report command rides in the escalation body and the reader runs it. If the runner must run it, the leaf is act-then-ask instead; the PRD's word is 'Check: runs'.*

- `holds-report-with-summary-and-json` → producer:task-2 upstream. **PASS**.
- `seven-day-delay-honoured` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._milestone_time_gated` wired: holds a task whose metadata.milestone is not yet due; delayed mode after_secs of 7 days. **PASS**.
- `pure-gate-shape-accepted` → `grep:fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::deterministic_task_error` wired: accepts task_kind deterministic with always_escalates and no before_done. **PASS**.
- `escalation-to-leo-carries-the-summary` → built here; a pure gate's born-at-L2 escalation naming holds per project, top consumers per ranking, the unattributed share and the reclaim threads' share. **PASS**.
  - delivered_check: manual. A human gate: Leo receives the escalation. The task is the deliverable and carries no code.

---

## Leaf 4: Every verify command records its CPU, peak memory and stall from inside its scope

*Existing task 6355, retitled and re-planned at decompose (store edit; task_id stays null here and is stamped at commit). Hooks land wherever review-all stream 4 has left _run_cmd at dispatch.*

- `cgroup-file-parser` → producer:task-1 upstream. **PASS**.
- `shared-scope-tag-rule` → producer:task-1 upstream. **PASS**.
- `scope-script-runs-the-command` → `grep:orchestrator/src/orchestrator/verify.py::_run_cmd` `grep:orchestrator/src/orchestrator/scope_launch_handshake.py::ScopeLaunchHandshake.payload` wired: _run_cmd runs handshake.payload(cmd) under bash in the scope. **PASS**.
- `kill-path-reaps-the-scope` → `grep:orchestrator/src/orchestrator/verify.py::_run_cmd` `grep:orchestrator/src/orchestrator/verify.py::_reap_command_subtree` wired: _run_cmd calls _reap_command_subtree on timeout, cancel and error. **PASS**.
- `relocated-scope-detection` → `grep:orchestrator/src/orchestrator/verify.py::_reap_command_subtree` wired: it calls _discover_relocated_scopes. **PASS**.
- `summary-whitelist-and-run-result` → `grep:orchestrator/src/orchestrator/verify.py::_build_summary_payload`; `grep:orchestrator/src/orchestrator/verify.py::CheckRun`. **PASS**.
- `census-reads-local-summaries` → `grep:scripts/verify_budget_census.py::load_records` wired: it reads the summaries under data/verify-logs. **PASS**.
- `c4-record-fields` → built here; the C4 dataclass with user_usec, system_usec, the three stall totals and the identity fields, one home, listed in remote_gate wire_classes; absent on main. **PASS**.
  - delivered_check: grep present, pattern `user_usec`, paths orchestrator/src/orchestrator/, shared/src/shared/
- `kill-path-capture-state` → built here; the before_kill capture state, read from the scope's sysfs path before the kill; absent on main. **PASS**.
  - delivered_check: grep present, pattern `before_kill`, paths orchestrator/src/orchestrator/
- `census-prints-cpu-and-peak` → built here; scripts/verify_budget_census.py prints CPU and peak memory per command from the record; absent on main. **PASS**.
  - delivered_check: grep present, pattern `cpu_usec`, paths scripts/verify_budget_census.py
- `capture-never-changes-the-verify` → built here; the subshell suffix keeps the exit code and adds nothing to the command's output; an unavailable capture is a record that says so. **PASS**.
  - delivered_check: manual. A behavioural property proven by the capture-at-exit, timeout, relocated and unavailable boundary rows, not a grep.

---

## Leaf 5: Carry verify records back from the laptop, and store each command run as an event

- `c4-record-and-run-result-carrier` → producer:task-4 upstream. **PASS**.
- `strict-wire-reader` → `grep:orchestrator/src/orchestrator/verify_runner.py::result_from_dict` `grep:orchestrator/src/orchestrator/remote_gate.py::wire_classes` `grep:orchestrator/src/orchestrator/remote_gate.py::wire_fingerprint` result_from_dict is the strict reader of the VerifyResult wire; wire_classes is the registry the record is added to. **PASS**.
- `pass-summary-payloads` → `grep:orchestrator/src/orchestrator/verify_runner.py::_task_pass_summary_payload`; `grep:orchestrator/src/orchestrator/verify_runner.py::_merge_pass_summary_payload`; `grep:orchestrator/src/orchestrator/verify_runner.py::RemoteRunner._archive_failure_streams`. **PASS**.
- `event-registry-and-local-emit-precedent` → `grep:orchestrator/src/orchestrator/event_store.py::EventType` `grep:orchestrator/src/orchestrator/verify.py::_recorded_admission_plan` `grep:orchestrator/src/orchestrator/task_verify_dispatch.py` wired: EventType is the registry; _recorded_admission_plan is the local emit point beside the admission-plan event; remote dispatch is in task_verify_dispatch. **PASS**.
- `event-type-registered` → built here; verify_command_resources in the EventType registry; absent on main. **PASS**.
  - delivered_check: grep present, pattern `verify_command_resources`, paths orchestrator/src/orchestrator/event_store.py
- `event-emitted-per-command-run` → built here; one verify_command_resources event per command run, local and remote, task id or merge sha and request id; absent on main. **PASS**.
  - delivered_check: grep present, pattern `EventType\.verify_command_resources`, paths orchestrator/src/orchestrator/
- `laptop-and-local-events-and-pass-summaries` → built here; runs.db holds events with runner laptop and runner local, and a remote green pass summary carries the same records. **PASS**.
  - delivered_check: manual. Observed in the live runs.db after deploy; the remote-carry boundary row proves it in CI.

---

## Leaf 6: Delegate each orchestrator service's subtree, start it in daemon/, and keep the slice weight check working

*Intermediate leaf. Add the edge 9985 -> this leaf at decompose (an edge on the existing task). Prerequisite task 3394 supplies the slice module, the weight check, the Slice= unit lines, the unit-file test and the topology doc (all in its declared files).*

- `slices-weight-check-and-slice-lines` → producer:task-3394 upstream. **PASS**.
- `own-reading-resolver` → `grep:shared/src/shared/psi.py::read_own_cgroup_pressure` `grep:shared/src/shared/psi.py::resolve_own_cgroup` wired: read_own_cgroup_pressure calls resolve_own_cgroup. **PASS**.
- `unit-file-test` → `grep:tests/scripts/test_orchestrator_service_files.py` exists on main; task 3394 extends it, this leaf adds the three-lines-together rule. **PASS**.
- `units-delegate-the-memory-controller` → built here; Delegate=memory in every scripts/orchestrator-*.service; absent on main. **PASS**.
  - delivered_check: grep present, pattern `Delegate=memory`, paths scripts/
- `units-start-in-the-daemon-subgroup` → built here; DelegateSubgroup=daemon in every scripts/orchestrator-*.service; absent on main. **PASS**.
  - delivered_check: grep present, pattern `DelegateSubgroup=daemon`, paths scripts/
- `weight-check-accepts-a-daemon-child` → built here; ensure_project_slice_weight accepts a daemon child of the service. **PASS**.
  - delivered_check: manual. The file is task 3394's and may already mention the word daemon, so a pattern could pass before this leaf; the weight-check boundary row proves it.
- `no-unit-delegates-cpu-and-slice-comes-with-delegate` → built here; no unit carries Delegate=memory or DelegateSubgroup=daemon without Slice=, and none delegates cpu. **PASS**.
  - delivered_check: manual. A negative property across all units; the unit-file test is its check, and a grep for a cpu delegation is absent today (vacuous).

---

## Leaf 7a: The shared spawn path accepts a placement and records each attempt's reading

*Intermediate leaf. Producers outside this batch: task 9812 (row schema) and task 9846 (stream 20 task 9846, the per-attempt writer, named in plans/review-all-s20-cli-invocation-and-usage-gate-prd.capability-manifest.yaml). Same-file serialization to wire at decompose: task 10429 (allowance reserve 1b) and task 9850 (stream 20 task 9850, moves the spawn to shared/claude_runner.py); the pointers here resolve on main today and are re-located if theta lands first.*

- `one-invocations-row-per-attempt-schema` → producer:task-9812 upstream. **PASS**.
- `one-invocations-row-per-attempt-writer` → producer:task-9846 upstream. **PASS**.
- `spawn-prefix-chain` → `grep:shared/src/shared/cli_invoke.py::run_cli_subprocess` `grep:orchestrator/src/orchestrator/agents/invoke.py::invoke_agent` wired: run_cli_subprocess composes the govern, nice and sandbox prefixes before the argv; invoke_agent reaches it through its runner default. **PASS**.
- `retry-wrapper-threads-a-keyword` → `grep:shared/src/shared/cli_invoke.py::invoke_with_cap_retry` wired: it calls the usage gate's invoke_slot and run_cli_subprocess; the optional placement parameter rides here. **PASS**.
- `added-columns-stay-last-and-nullable` → `grep:shared/src/shared/cost_store.py::_ADDED_INVOCATION_COLUMNS` `grep:shared/src/shared/cost_store.py::CostStore.save_invocation` the table the C5 columns are appended to, and the writer. **PASS**.
- `c5-columns-on-invocations` → built here; cgroup_capture, cgroup_capture_reason, cpu_usec, memory_peak_bytes, swap_peak_bytes and the four stall totals; absent on main. **PASS**.
  - delivered_check: grep present, pattern `cgroup_capture_reason`, paths shared/src/shared/cost_store.py
- `outermost-join-prefix` → built here; a launch prefix writes its own pid into the child's cgroup.procs and execs, outside the govern and nice prefixes and the sandbox wrap; absent from shared/src/shared on main. **PASS**.
  - delivered_check: grep present, pattern `cgroup\.procs`, paths shared/src/shared/
- `placement-protocol-and-stdin-survives` → built here; the placement protocol (prefix and env before, reading after) declared in shared code; the stdin-starvation pin in shared/tests/test_cli_invoke.py stays green through the new prefix. **PASS**.
  - delivered_check: manual. The protocol's names are the architect's; the CI spawn through the full prefix chain with a fixture placement is the leaf's signal.

---

## Leaf 7b: Each agent attempt gets its own cgroup inside its orchestrator service

*Leaf 1 is upstream (edges 1 → 7a and 1 → 7b added 2026-11-16). Task 9985's reinstall is the production signal, not a code dependency.*

- `daemon-subgroup-and-delegated-memory-controller` → producer:task-6 upstream. **PASS**.
- `placement-protocol-and-c5-columns` → producer:task-7a upstream. **PASS**.
- `topology-doc-and-slice-weight-check` → producer:task-3394 upstream, through leaf 6. **PASS**.
- `cgroup-file-parser-import` → producer:task-1 upstream (shared/src/shared/cgroup_stats.py, D7). **PASS**.
- `startup-step-hook` → `grep:orchestrator/src/orchestrator/harness.py::Harness._run_startup_steps` wired: it runs the daemon's startup steps, the site the +memory write joins. **PASS**.
- `agent-row-writers` → `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._invoke` `grep:orchestrator/src/orchestrator/review_checkpoint.py::ReviewCheckpoint._run_review` `grep:orchestrator/src/orchestrator/agents/invoke.py::invoke_agent` wired: _invoke and _run_review write the invocations row through cost_store.save_invocation; invoke_agent is the spawn entry. **PASS**.
- `harness-guidance-advice` → `grep:orchestrator/src/orchestrator/agents/harness_guidance.py` carries the advice to read one's own memory.events, checked in prose against the new layout. **PASS**.
- `agent-cgroup-manager-module` → built here; orchestrator/src/orchestrator/agent_cgroups.py: startup checks, +memory, the placement implementation, remove and sweep. **PASS**.
  - delivered_check: path present: orchestrator/src/orchestrator/agent_cgroups.py
- `memory-controller-enabled-at-startup` → built here; the daemon writes +memory to the service's cgroup.subtree_control; absent from orchestrator/src on main. **PASS**.
  - delivered_check: grep present, pattern `subtree_control`, paths orchestrator/src/orchestrator/agent_cgroups.py
- `per-project-switch` → built here; defaults on, reloadable, in the review-all stream 16 config layout; decides only whether new spawns get a child. **PASS**.
  - delivered_check: manual. The key's name and place depend on stream 16's layout at dispatch; the switch-off boundary row proves it.
- `agent-children-in-production` → built here; after task 9985's restart the service holds daemon/ and one agent-* child per running agent, new invocations rows carry cgroup_capture exit, and the recorder shows agents@df-dark_factory.slice. **PASS**.
  - delivered_check: manual. Operator-observable on the host after task 9985's reinstall.

---

## Leaf 8: Report what verify runs and agent attempts cost

*Signal leaf with 3. Leaf 2 is upstream (edge 2 → 8 added 2026-11-16), which is also the same-file serialization.*

- `verify-command-events` → producer:task-5 upstream. **PASS**.
- `agent-attempt-columns` → producer:task-7b upstream, which carries 7a's columns. **PASS**.
- `report-script-and-its-scaffolding` → producer:task-2 upstream (scripts/resource_report.py). **PASS**.
- `invocations-carry-role-and-model` → `grep:shared/src/shared/cost_store.py::CostStore.save_invocation` wired: it writes the model and role columns of the invocations table the report groups by. **PASS**.
- `runs-db-event-reader` → `grep:orchestrator/src/orchestrator/event_store.py::EventStore` the events store the report reads read-only per project root. **PASS**.
- `costs-subcommand-reads-verify-events` → built here; the costs subcommand groups verify_command_resources events by project, role, leg and host; absent from scripts on main. **PASS**.
  - delivered_check: grep present, pattern `verify_command_resources`, paths scripts/resource_report.py
- `no-producer-written-state` → built here; a source with no rows prints no producer has written here (INV-13), the agent rows before task 9985; absent on main. **PASS**.
  - delivered_check: grep present, pattern `no producer has written here`, paths scripts/
- `percentiles-and-labels` → built here; CPU seconds and peak-memory percentiles per project, verify role, leg and host and per agent role and model, with peak memory labelled as including file cache. **PASS**.
  - delivered_check: manual. Output shape on the live stores; the report's own fixture tests assert it.
