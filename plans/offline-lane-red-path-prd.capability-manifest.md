# Capability manifest — the offline lane's red path

PRD: `plans/offline-lane-red-path-prd.md`. Decomposed 2026-10-12 by a sibling `/team` + `/prd` session;
substrate re-read on main `2178c2badb` by five drafting seats and again on `040be27994` by the critic
(no file the batch cites changed between the two). Machine-readable twin:
`plans/offline-lane-red-path-prd.capability-manifest.yaml`, stamped by `commit_planning`.

Task ids (ε0 added 2026-10-13, after the first filing; see the PRD's §14 item 24): α 6857 · β 6858 · γ 6859 · δ1 6860 · δ2 6861 · δ3 6862 · π 6863 · ε1 6864 · ε2 6865 · ι 6866 · ζ0 6867 · ζ 6868. Outside the PRD's plan, and so outside the sidecar: follow-up 1 6869
(merge-gate evidence), follow-up 2 6870 (deferred PRD stub), follow-up 3 6871 (suite census).

Binding vocabulary is `skills/prd/references/gates.md` → *Capability Manifest*. Every mechanical check
below was linted against `main` with `shared.delivered_check_polarity.lint_delivered_checks` before
filing: 33 checks at the first filing and 3 more for ε0, zero findings. Each asserts a new symbol or file positively, or the removal of a
named definition scoped to one file; none matches prose.

## Resolutions made at decompose

The re-scopes, and the reason for each, have one home: the PRD's §14 decompose record. The bindings
that failed as §9 was written, and how each was resolved without changing a §4 decision:

| Binding as written | Failure | Resolution (§14 item) |
|---|---|---|
| α's scenario 1 asserts kind and first line; the typed parse was β's | producer-downstream | the typed parse moved to α (1) |
| ζ's mechanical checks had no producer | producer-absent | new leaf ζ0, upstream of ζ (12) |
| ζ0 read a pass's outcome and ended episodes; nothing persisted them | producer-extent-short | two pass-level blocks on the run record: α declares, δ2 writes, ε2 extends (15) |
| ε1's report partition read lane rows before any existed | reader before producer | a never-written line, distinct from zero (13) |
| δ1 tested the tri-state ancestry read that δ2 builds | producer-downstream | δ1 declares a one-method port and tests it with a fake; nothing calls the engine before δ2 |

## G7 walk — every task against `docs/legibility/design-invariants.md`

| Invariant | Where it binds, and the reading |
|---|---|
| `contracts-machine-checked` | γ (one typed submit outcome, one fixture read by both packages); β (typed pass outcome); δ3 and π (typed, bounded config fields; a probe that breaks its exit contract is a broken probe with an open escalation); ζ0 (exit codes are the contract ζ reads). No hit. |
| `structured-facts-at-failure` | α (exit code, kind, first line kept at the emitter); δ1 (the D11 record names the task, its status and the record paths); π (the escalation names the dependency). **Hit, resolved:** ε1 put the lane's call site only in an escalation's prose; it now carries it as a structured evidence entry. ζ0 reads stores, never the journal. |
| `corroborate-before-acting` | δ1 (covering task read live; ancestry tri-state; an unreadable answer defers; `planning_committed` set only after a live read, because the client's commit never raises); γ (commit only a deferred survivor); π (a stored escalation id is re-read before another is opened); ε2 (live status before eligibility). No open hit. |
| `storm-escape-required` | **Waived on ε2** for the lane's clear rate (D6 call 1; `metadata.g7_waivers`). **Hit, resolved:** δ1's `filing_due` was retried for ever with a log line; a due entry older than `recheck_secs` now raises one level-1 record per command. ε0's debt drain has its own stale-item escalation (30 minutes). β (a persistent unconfirmable is a pseudo-test bounded by D5 and D11); π (a broken probe keeps one escalation open). |
| `no-lockstep-duplication` | **Hits, resolved:** ε2's third copy of the host-pressure mapping (one public helper in `shared/psi.py`); the reproduce line rendered in three leaves (β's `reproduce_line`); exit code 75 in two packages (one constant in `shared`). δ2 duplicates the lane's role string into `PROMOTE_SENTINEL_BOUND_ROLES`, which the escalation package cannot import: pinned by a lockstep test, that registry's existing pattern. One home each: the kind rule (β), the slug (α), the recorder body (ε1), the fixture (γ), policy (δ1). |
| `status-matches-liveness` | The lane writes task status only by filing; a half-filed deferred task is completed on the next decision (scenario 19). No hit. |
| `holds-owned-and-bounded` | δ1 (a held episode: recheck and expiry; a due filing: the stuck-write record). **Hits, resolved in π:** the outage hold had no exit when the worktree is absent after a restart, when the probe is removed from the command, or when the command is disabled. ζ (the milestone). Follow-up 2 (a deferral record naming its flip condition; its enforcement, task 6525, has not landed). |
| `loop-thread-occupancy-bounded` | **Hit, resolved:** δ1 refreshed one store entry per episode per pass on the loop thread, each a whole-file rewrite; now one write per pass, off the loop, and one batched status read. α and β do record and junit I/O in one off-loop hop; π's probe is a subprocess with a timeout. |
| `one-fact-one-home` | Re-scopes live in PRD §14 and task texts point there; δ3's pointers are dated pointers; a debt's origin lives on its owner task; a pass's outcome and effect live on its run record. No hit. |
| `guards-exercise-behaviour` | β pins the round trip with real pytest output; π runs its script against live, unreachable and write-refusing stores; ζ0's tests build their stores with the real writers; δ3's pointer binding is manual, not a grep over prose. No hit. |
| `no-silent-fail-soft` | γ (failure in band); β (Unconfirmable, inconclusive); ζ0 (`no_data` and exit 2; an interrupted pass is counted). **Hit, resolved:** a debt-eligible test the filing cap did not reach left no trace; ε0 retires the cap, and every eligible test stays on the ledger's due list until it has an owner. |
| `exceptions-owned-or-ratified` | δ2 moves a role between two capability registries (out of this invariant's scope by its own text) and records one authorized ratchet raise with its task id and reason. No hit. |
| `readers-prove-their-producer` | **Hit, resolved:** ε1 (above). ζ0 (`no_data` is not a pass); ζ reads production rows. Follow-up 3 exists because the suite census reads every occurrence row and the lane becomes a producer it did not expect. |

One waiver: `storm-escape-required` on ε2 (6865) — the PRD's D6, call 1, an author's call the owner had
not reviewed when the batch was filed. Until task 3792 lands nothing escalates on the lane's clear rate.

## Per-leaf bindings

### α — Offline-lane red path α: every lane sub-run streams its output to disk and leaves a run record (records, one streaming launcher, the typed failure-line parse)

Decompose re-scope: the typed failure-line parse moved here from β (scenario 1 is this leaf's signal and asserts kind and first line). Supersedes task 4292.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| run-record-module | capability→producer — α creates orchestrator/src/orchestrator/offline_lane_records.py (the §5.1 run record, the record store with an injectable clock, 14-day retention, the command-name slug). Absent on main 2178c2badb. The record root derives from config.project_root, which orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker.__init__ already holds (orchestrator/src/orchestrator/guard_state.py::guard_path is the precedent for the unresolvable-root guard). | PASS | path `orchestrator/src/orchestrator/offline_lane_records.py` present |
| run-seams-module | capability→producer — α moves the three detection seams (orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker._default_run_suite, ::_default_run_infra, ::_default_run_command) into orchestrator/src/orchestrator/offline_lane_runs.py behind one streaming launcher that all six subprocess launches use. Measured on main: six create_subprocess_exec sites in offline_lane.py, each PIPE + communicate() keeping a 2000-character tail. | PASS | path `orchestrator/src/orchestrator/offline_lane_runs.py` present |
| typed-failure-parse | capability→producer — α adds FailureKind / PytestFailure / parse_pytest_failures / node_ids_of beside orchestrator/src/orchestrator/verify.py::_extract_failing_test_ids, which becomes a projection of it (four production callers; the name is patched by string in tests, so it stays). As §9 was written this was producer-downstream (β); resolved by re-homing. | PASS | grep `^def parse_pytest_failures\(` present in orchestrator/src/orchestrator/verify.py |
| signal-ended-classification | substrate — orchestrator/src/orchestrator/verify_classify.py::is_external_kill_rc exists (negative codes, allow-list of four signals). Both forms measured on this host: a signal-killed child reports -9; through `sh -c` a killed grandchild reports 137. α accepts 128+n only where that allow-list accepts n. | PASS | manual — behaviour pinned by scenario 2's fixture commands, not greppable |
| junit-for-structured-commands | substrate — orchestrator/src/orchestrator/verify_cmd.py::with_junitxml and orchestrator/src/orchestrator/verify.py::_with_junitxml_str exist; both dark-factory lane commands parse as structured pytest. | PASS | manual — consumed substrate |
| pass-record-read-side-and-blocks | capability→producer — α writes a pass header at allocation, declares the pass-level `outcome` and `effect` blocks (None in α; δ2 writes them) and exposes the reads later leaves need: one pass by id, a command's passes since a time, a well-formedness check, a re-write of the detect record with a probe result. Added at decompose: as §5.1 was written, no durable record said what a pass reproduced, cleared or counted. | PASS | manual — a read API; exercised by δ1, δ2, π and ζ0 |

### β — Offline-lane red path β: the confirm narrows or intersects, and a red pass ends in Confirmed, Unconfirmable or (defined only) Outage

Folds task 4226 (unparseable non-zero exits) and the lane half of task 3880. Defines all three pass-outcome arms; the Outage arm is produced only by π.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| confirm-module | capability→producer — β creates orchestrator/src/orchestrator/offline_lane_confirm.py (Confirmed | Unconfirmable | Outage; confirm(); the narrowed and intersected modes; the leg-native rules of D13). Absent on main. Outage is defined here and produced by π: β's signal does not assert it. | PASS | path `orchestrator/src/orchestrator/offline_lane_confirm.py` present |
| reproduction-rule | capability→producer — β adds orchestrator/src/orchestrator/verify.py::reproduced_failures as the one home of the kind rule (a confirm ERROR reproduces only a detection ERROR). Absent on main. Consumes α's typed parse (upstream). | PASS | grep `^def reproduced_failures\(` present in orchestrator/src/orchestrator/verify.py |
| whole-command-confirm-removed | removal — orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker._default_confirm_command (present on main) and the `<name>::nonzero-exit` sentinel go. | PASS | grep `def _default_confirm_command\(` absent in orchestrator/src/orchestrator/offline_lane.py |
| narrowed-command-construction | substrate — orchestrator/src/orchestrator/verify_cmd.py::parse_config_command, ::scope_to, ::serial_pytest, ::with_pytest_timeout, ::with_junitxml, ::render exist; rendered and read back on main for both dark-factory lane commands (`pytest -m integration <ids>` keeps the CLI -m). | PASS | manual — consumed substrate |
| junit-outcome-reader-and-normaliser | capability→producer — absent on main (orchestrator/src/orchestrator/verify.py::_extract_test_ids_from_junit returns failing and skipped ids spelled classname::name, with no passed vector); β builds the reader and the junit→node-id normaliser. | PASS | manual — names are the implementer's; behaviour is pinned by scenarios 6 and 10 |
| round-trip-rejection | rejection-mechanism — built by β; scenario 11 observes it fire on real pytest output for an id containing whitespace. | PASS | manual — rejection quality is judged by scenario 11 |
| legacy-leg-contracts | substrate, external — the numeric leg's "exit 100 when it printed ids" and the infra leg's `RESULT: PASS | FAIL | SKIP (name)` lines are reify's scripts' contracts, read on that repo; dark-factory parses only FAIL lines today. | PASS | manual — external repo; scenario 22's fixture is the pin |

### γ — Offline-lane red path γ (HUB carry): one typed submit seam for machine filers — Scheduler.submit_task returns a tagged outcome; chronic_flake.extract_task_id and the tkt_ prefix guard retire

Hub hand-carry. Decompose re-scope: Harness._extract_task_id is NOT retired here; it loses its auto-eval caller and is deleted with its last caller (the lane's old client) by δ2, so γ edits no lane test file. Makes auto-eval live: see the decision point in the task.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| typed-scheduler-submit | capability→producer — γ adds orchestrator/src/orchestrator/scheduler.py::Scheduler.submit_task returning a tagged outcome. Absent on main: no submit method exists; every filer calls dispatch_tool and reads the response with its own parser. | PASS | grep `async def submit_task\(` present in orchestrator/src/orchestrator/scheduler.py |
| submit-outcome-types | capability→producer — γ adds shared/src/shared/task_submit.py (SubmittedTicket / SubmittedTask / SubmitFailed and one parser over shared/src/shared/mcp_envelope.py::parse_tool_result, which exists and is strict). In `shared` because flake_ledger.py imports `shared` only. | PASS | grep `^class SubmittedTask[(:]` present in shared/src/shared/task_submit.py |
| chronic-flake-extractor-retired | removal — orchestrator/src/orchestrator/chronic_flake.py::extract_task_id (present on main; it accepts `ticket` as an id) and orchestrator/src/orchestrator/flake_ledger.py::_TICKET_ID_PREFIX go. | PASS | grep `^def extract_task_id\(` absent in orchestrator/src/orchestrator/chronic_flake.py |
| one-fixture-two-readers | capability→producer — γ adds shared/tests/fixtures/submit_task_responses.json, read by an orchestrator test and by fused-memory/tests/test_task_interceptor.py. The four shapes are produced on main by fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor.submit_task, ::_submit_task_planning_mode and fused-memory/src/fused_memory/server/tool_errors.py::mcp_tool_errors. | PASS | manual — scenario 24 executes both readers against the one file |

### δ1 — Offline-lane red path δ1: the episode engine (LaneEngine.observe and LaneEngine.tick), the D11 escalation record and the filed fix task — intermediate, unwired until δ2

Intermediate, roped into ι. Settles PRD §12 question 1 at decompose: the D11 record, the role constant and the sentinel live here, in offline_lane_escalations.py. G7 resolutions: one store write per pass off the loop thread; a stuck-write record when a filing stays due.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| episode-engine | capability→producer — δ1 creates orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine (observe, tick → TickResult.commands_due, lane_policy). Absent on main. Store: orchestrator/src/orchestrator/guard_state.py::PersistentMap exists (it rewrites the file on every assignment, hence the one-write-per-pass rule in the task). | PASS | grep `^class LaneEngine[(:]` present in orchestrator/src/orchestrator/offline_lane_episodes.py |
| fix-task-payload-builder | capability→producer — δ1 creates orchestrator/src/orchestrator/offline_lane_filing.py::build_lane_fix_task_arguments (the §5.5 payload). Absent on main. `spawn_context` and `merge_lane` are blessed keys (shared/src/shared/task_metadata.py::_BLESSED_METADATA_KEYS). | PASS | grep `def build_lane_fix_task_arguments\(` present in orchestrator/src/orchestrator/offline_lane_filing.py |
| lane-escalation-records | capability→producer — δ1 creates orchestrator/src/orchestrator/offline_lane_escalations.py (the role, the per-command sentinel, the D11 record). Pattern on main: orchestrator/src/orchestrator/harness.py::Harness._file_pool_storage_absent_escalation; per-key sentinels: orchestrator/src/orchestrator/recovery_emission.py::RECOVERY_VETO_STREAK_SENTINEL_PREFIX. | PASS | path `orchestrator/src/orchestrator/offline_lane_escalations.py` present |
| owner-liveness-shared-home | capability→producer — δ1 moves orchestrator/src/orchestrator/flake_ledger.py::_owner_liveness (one call site, ::_ensure_owner_task) to shared/src/shared/task_statuses.py::owner_liveness. | PASS | grep `^def owner_liveness\(` present in shared/src/shared/task_statuses.py |
| effect-returned-by-observe | capability→producer — LaneEngine.observe returns the pass's effect (tests counted, episodes ended with reason, length and reproduced heads, tests that reached the threshold, task ids filed). The store keeps no history of ended episodes (§5.3), so this is the only durable trace; δ2 writes it to the pass record. | PASS | manual — asserted by δ1's engine tests; read by ζ0 |
| stuck-write-record | capability→producer — δ1 raises one level-1 record on the command's sentinel when a filing has stayed due for recheck_secs, and resolves it itself. G7 resolution (storm-escape-required, holds-owned-and-bounded): a retry period is not a bound, and a filing that fails every time was the PRD's own premise failure. commit_planning on the task client returns nothing and never raises, so planning_committed is set only after a live read. | PASS | manual — behaviour asserted by δ1's engine tests |
| typed-submit-outcome | producer:γ — upstream of δ1. | PASS | manual — consumed; γ's own checks gate it |
| pass-outcome-and-run-record | producer:β and producer:α — both upstream of δ1. | PASS | manual — consumed |
| ancestry-read-port | capability→producer — δ1 declares a one-method Protocol (is_ancestor_checked) and tests with a fake; the real read is δ2's. δ1's signal is unit tests over fakes and nothing in production calls the engine before δ2, so no capability δ1 asserts is downstream of it. | PASS | manual — a declared port, exercised by δ1's own fakes |

### δ2 — Offline-lane red path δ2 (HUB carry): the cutover — the engine replaces the fingerprint red path; the per-pass pipeline moves to offline_lane_pass.py with the §5.7 hook; one tick in the worker loop; the lane files through the machine-filer client

Hub hand-carry; 25 files. Decompose re-scopes: the per-pass pipeline moves to a new non-ratcheted offline_lane_pass.py so that π and ε2 never edit the ratcheted worker; the §5.7 hook list lives there; Harness._extract_task_id is deleted here. Carries review finding fk-0238ff63263c (from task 6639).

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| per-pass-pipeline | capability→producer — δ2 creates orchestrator/src/orchestrator/offline_lane_pass.py::run_pass (detect → confirm → hooks → LaneEngine.observe) and lane_pass_outcome_hooks. Absent on main. | PASS | grep `async def run_pass\(` present in orchestrator/src/orchestrator/offline_lane_pass.py |
| pass-outcome-and-effect-recorded | capability→producer — run_pass writes β's outcome form and the effect δ1's observe returns into the two blocks α declared on the pass record. All three producers are upstream. | PASS | manual — asserted by δ2's worker tests; read by ζ0 |
| ancestry-read-tells-error-from-no | capability→producer — δ2 adds orchestrator/src/orchestrator/git_ops.py::GitOps.is_ancestor_checked. On main ::GitOps.is_ancestor returns rc == 0, so a git error reads as "not an ancestor". git_ops.py is ratcheted and grows: an authorized raise. | PASS | grep `def is_ancestor_checked\(` present in orchestrator/src/orchestrator/git_ops.py |
| old-red-path-removed | removal — orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker._maybe_promote_blocker and the rest of the fingerprint red path (present on main). | PASS | grep `def _maybe_promote_blocker\(` absent in orchestrator/src/orchestrator/offline_lane.py |
| old-lane-client-removed | removal — orchestrator/src/orchestrator/harness.py::_OfflineLaneTaskClient (present on main). | PASS | grep `^class _OfflineLaneTaskClient[(:]` absent in orchestrator/src/orchestrator/harness.py |
| harness-extractor-removed | removal — orchestrator/src/orchestrator/harness.py::Harness._extract_task_id (present on main; after γ its only caller is the class above). | PASS | grep `def _extract_task_id\(` absent in orchestrator/src/orchestrator/harness.py |
| lane-role-sentinel-bound | capability→producer — escalation/src/escalation/authority.py::PROMOTE_SENTINEL_BOUND_ROLES exists with one role; δ2 adds the lane's and removes it from escalation/src/escalation/classify.py::INFO_L0_MECHANICAL_ROLES. | PASS | grep `orchestrator-offline-lane` present in escalation/src/escalation/authority.py |
| machine-filer-client | substrate — orchestrator/src/orchestrator/chronic_flake.py::SchedulerChronicFlakeTaskClient is constructible from what orchestrator/src/orchestrator/harness.py::Harness._start_offline_lane holds; its typed outcome is producer:γ (upstream). | PASS | manual — consumed |
| engine | producer:δ1 — upstream. | PASS | manual — consumed |

### δ3 — Offline-lane red path δ3 (HUB carry): the three policy knobs in and red_advances_before_blocker out, the workflow.py helpers and their metadata keys retired, stale prose, the runbook, ARCHITECTURE §5.4 and seven dated PRD pointers

Hub hand-carry; 26 files, 16 of them prose-only. Decompose re-scope: the policy read is in offline_lane_episodes.py::lane_policy, not in offline_lane.py as §9 listed.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| policy-knobs | capability→producer — δ3 adds git.offline_lane_persist_passes / _recheck_secs / _red_unlanded_hours to orchestrator/src/orchestrator/config.py::GitConfig and ::RELOADABLE_FIELDS. Absent on main. | PASS | grep `offline_lane_persist_passes` present in orchestrator/src/orchestrator/config.py |
| advance-count-knob-removed | removal — git.offline_lane_red_advances_before_blocker (present on main in config.py; no project yaml sets it). | PASS | grep `offline_lane_red_advances_before_blocker` absent in orchestrator/src/orchestrator/config.py |
| fingerprint-helper-removed | removal — orchestrator/src/orchestrator/workflow.py::compute_failing_test_set_fingerprint and ::build_offline_lane_fix_task_arguments (present on main; δ2 deletes their last callers). | PASS | grep `def compute_failing_test_set_fingerprint\(` absent in orchestrator/src/orchestrator/workflow.py |
| reload-reaches-the-next-pass | substrate — orchestrator/src/orchestrator/config.py::apply_reload mutates the live config in place and the worker holds that object (pattern: orchestrator/tests/test_offline_lane.py::TestOfflineLaneCommandsReload); the policy seam is producer:δ1 (lane_policy, upstream). | PASS | manual — behaviour asserted by the new reload test |
| dated-pointers | substrate — the seven target documents exist on main (two under docs/prds/, five under plans/). | PASS | manual — prose; a grep over prose is not a guard |

### π — Offline-lane red path π (HUB carry): the outage probe — LaneCommand.outage_probe, the probe step in the pass, Outage and broken-probe handling, the per-command escalation the lane closes itself, the probe loop on the tick, and fused-memory's lane_dependency_probe.py

Hub hand-carry. Decompose re-scope: the probe step goes into offline_lane_pass.py::run_pass and the records into offline_lane_escalations.py; π does not edit offline_lane.py. Adds probe_broken_escalation_id beside §5.3's outage_escalation_id.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| lane-command-probe-field | capability→producer — π adds outage_probe / outage_probe_timeout_secs to orchestrator/src/orchestrator/config.py::LaneCommand. Absent on main. git.offline_lane_commands is already hot-reloadable as a list leaf. | PASS | grep `outage_probe_timeout_secs` present in orchestrator/src/orchestrator/config.py |
| dependency-probe-script | capability→producer — π creates fused-memory/scripts/lane_dependency_probe.py. Absent on main. Embedder and chat model through fused-memory/src/fused_memory/backends/graphiti_client.py::build_embedder and ::build_llm_client over fused-memory/src/fused_memory/config/schema.py::FusedMemoryConfig. | PASS | path `fused-memory/scripts/lane_dependency_probe.py` present |
| qdrant-integration-declares-a-probe | capability→producer — π adds the probe to the qdrant-integration entry of dark-factory-orchestrator.yaml. Absent on main. | PASS | grep `outage_probe:` present in dark-factory-orchestrator.yaml |
| escalation-queue-calls | substrate — escalation/src/escalation/queue.py::EscalationQueue.resolve, ::amend, ::has_open_l1, ::get_by_task, ::get, ::submit exist; a level-2 with members is minted in process through escalation/src/escalation/server.py::create_server and its promote_to_l2 tool. | PASS | manual — consumed substrate |
| pipeline-engine-and-sentinel | producer:δ2 (run_pass; the role in PROMOTE_SENTINEL_BOUND_ROLES) and producer:δ1 (LaneEngine.observe / tick, had_outage_pass, the sentinel) — both upstream. | PASS | manual — consumed |
| probe-script-against-live-stores | scenario 39's live-store half is marked `integration`, so the merge gate deselects it and the lane itself runs it; the unreachable and write-refusing halves carry no marker and run in the gate. | PASS | manual — the live half cannot be exercised by this task's own gate |

### ε0 — Offline-lane red path ε0 (HUB carry): flake-debt filing leaves the observation path — a durable due list in the ledger, drained by a background service; the per-observation filing cap and budget retire

Added 2026-10-13 on Leo's ruling (PRD §14 item 24): de-flake filing leaves the observation path. Hub hand-carry (harness.py). Retires the per-observation filing cap and budget; supersedes the result type §14 item 19 first gave the recorder.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| debt-drain-module | capability→producer — ε0 creates orchestrator/src/orchestrator/flake_debt_drain.py: one pass reads due items oldest first (50 per pass), calls orchestrator/src/orchestrator/flake_ledger.py::open_debt (exists) with the wall-clock stamp, deletes an item once the debt is closed or has an owner that reads live and not deferred, backs off exponentially otherwise, bounds each pass (50 items, the first transport failure, 5 minutes), runs its sqlite calls off the loop thread, and keeps one stale-item escalation. Its pass reads the harness escalation queue when it runs, because the registry is built before the queue exists. Registered as a orchestrator/src/orchestrator/background_service.py::BackgroundService in orchestrator/src/orchestrator/harness.py (the lifecycle registry pattern exists, eleven services use it). | PASS | path `orchestrator/src/orchestrator/flake_debt_drain.py` present |
| durable-due-list | capability→producer — ε0 adds flake_debt_due to orchestrator/src/orchestrator/flake_ledger.py::_SCHEMA (additive CREATE TABLE IF NOT EXISTS, the schema's own idiom), keyed like flake_occurrence on (test_id, observed_at, call_site), written in the occurrence rows' transaction. | PASS | grep `CREATE TABLE IF NOT EXISTS flake_debt_due` present in orchestrator/src/orchestrator/flake_ledger.py |
| filing-cap-retired | removal — orchestrator/src/orchestrator/flake_recorder.py::_DEBT_FILINGS_PER_OBSERVATION_CAP, ::_DEBT_FILING_BUDGET_SECS and ::_unowned_first (present on main; task 4974 via 5304) and the recorder's task_client. The merge worker stops threading a client to the recorder (orchestrator/src/orchestrator/merge_lane/worker.py::_record_flake_observation; ratcheted, shrinks). | PASS | grep `_DEBT_FILINGS_PER_OBSERVATION_CAP =` absent in orchestrator/src/orchestrator/flake_recorder.py |
| stale-head-rule-in-the-drain | capability→producer — open_debt gains one optional check: when the lazy close reads the owner done and the due item's head does not contain that owner's fix commit, the cycle closes without a re-entry or a regression report. The check is δ1's containment function in orchestrator/src/orchestrator/flake_ledger.py (upstream) over δ2's GitOps.is_ancestor_checked (upstream). Leo 2026-10-13 (PRD §14 items 23 and 24); applies to every call site. | PASS | manual — behaviour asserted by scenario 40 in its merge-gate and drain-timing forms |
| retry-files-no-second-task | substrate — orchestrator/src/orchestrator/flake_ledger.py::open_debt writes the owner pointer before commit_planning and treats a live owner as a repeat (::_ensure_owner_task), so retrying an item cut off mid-filing files no second task. ε0 pins it. | PASS | manual — behaviour asserted by the drain tests |
| owner-invariant-meaning-kept | the flake-ledger PRD §5.9 owner invariant is still enforced inside open_debt; only the time open_debt runs moves. The report renders an unowned row with a due item as filing pending, not the breach. | PASS | manual — a semantic property; asserted by the merge-gate before/after test and the report test |

### ε1 — Offline-lane red path ε1: one flake recorder body for every call site — debt_eligible, debt context, per-test evidence, cycle in the de-flake title, report counters per call site

Rests on D6 call 1 (author's call, unreviewed). G7 resolution: the report prints a never-written line for a call site with no row, because the partition lands before any lane row exists. Tasks 3792, 3794, 4976, 5089, 5875 and 6343 depend on this leaf.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| shared-recorder-body | capability→producer — ε1 extracts the body of orchestrator/src/orchestrator/flake_recorder.py::record_merge_flake_suppression (one caller: orchestrator/src/orchestrator/merge_lane/worker.py::_record_flake_observation) into record_call_site_observations with a separate debt_eligible parameter. Every merge-specific element of PRD §5.8's table was confirmed against the code. | PASS | grep `async def record_call_site_observations\(` present in orchestrator/src/orchestrator/flake_recorder.py |
| lane-call-site | capability→producer — ε1 adds FlakeCallSite.offline_lane in orchestrator/src/orchestrator/flake_ledger.py. Absent on main. | PASS | grep `offline_lane = 'offline_lane'` present in orchestrator/src/orchestrator/flake_ledger.py |
| debt-context | capability→producer — ε1 adds DebtContext and threads it through orchestrator/src/orchestrator/flake_ledger.py::open_debt and ::_ensure_owner_task to ::build_deflake_task_arguments (which takes a DebtRow only today and builds the same title on every cycle). | PASS | grep `^class DebtContext[(:]` present in orchestrator/src/orchestrator/flake_ledger.py |
| report-per-call-site | capability→producer — ε1 replaces the two single counters of orchestrator/src/orchestrator/flake_report.py::FlakeLedgerReport with gate_blind_by_call_site / systemic_by_call_site. | PASS | grep `gate_blind_by_call_site` present in orchestrator/src/orchestrator/flake_report.py |
| enqueues-instead-of-filing | record_call_site_observations puts every debt_eligible test, with its DebtContext, on ε0's due list in the rows' transaction; no cap, budget or task client (ε0, upstream). Regression escalations, raised from ε0's drain module, carry the call site as a structured evidence entry | PASS | manual — ε1's recorder tests and scenario 32 after a drain pass |
| evidence-on-the-wire | substrate — the `detail` column exists (orchestrator/src/orchestrator/flake_ledger.py::_SCHEMA) and ::flake_suppression_from_wire projects onto declared fields, so an added optional field is safe in both directions; ε1 builds the nested read. | PASS | manual — pinned by the wire round-trip and skew tests |
| one-connection-bound | substrate — orchestrator/tests/test_flake_ledger.py::TestOneConnectionPerCall holds unchanged because the builder stays pure (PRD §12 question 6 settled as suggested). | PASS | manual — an existing test is the check |
| typed-outcome-and-shared-classifier | producer:γ and producer:δ1 — both upstream; ε1 is written against the tree after them. | PASS | manual — consumed |

### ε2 — Offline-lane red path ε2: lane clears and recoveries reach the flake recorder — per-pass rows, the one-head recovery rule, each due item's evidence head, the debt-eligible list, the de-flake owner in the fix digest

Rests on all three D6 calls (author's calls, unreviewed). G7 waiver storm-escape-required (D6 call 1): nothing escalates on the lane's clear rate until task 3792 lands. G7 resolution: one home for the host-pressure mapping in shared/psi.py. Records what became of every debt it offered in the pass record's effect block.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| lane-calls-the-shared-recorder | capability→producer — ε2 adds the recorder calls to orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine.observe and ::tick. Absent on main (the module does not exist). | PASS | grep `record_call_site_observations\(` present in orchestrator/src/orchestrator/offline_lane_episodes.py |
| stale-head-for-a-done-de-flake-owner | Leo 2026-10-13: decided by ε0's debt drain against each due item's head (the owner can go done between observation and drain); ε2 gives each item its evidence head | PASS | manual — scenario 40 (ε0, ε2, ι) |
| recorder-context-and-evidence | producer:ε1 — upstream. | PASS | manual — consumed |
| engine-pipeline-and-outage-mark | producer:δ1 (the engine, reproduced_heads, last_green_head), producer:δ2 (the worker calls it), producer:π (had_outage_pass) — all upstream. | PASS | manual — consumed |
| host-pressure-read | substrate — shared/src/shared/psi.py::read_psi_sample exists; orchestrator/src/orchestrator/verify.py::_psi_cpu_some10_or_none wraps it for the discriminator. ε2 gives the mapping one public home there. | PASS | manual — a small helper; pinned by shared/tests/test_psi.py |
| debt-row-read | substrate — orchestrator/src/orchestrator/flake_ledger.py::read_debt_many exists. | PASS | manual — consumed |

### ι — Offline-lane red path ι (integration gate): PRD §6 scenarios 1–23 and 25–38 against the real OfflineLaneWorker, the typed submit client over the scenario-24 fixture and the real flake ledger

The B+H integration gate. Its two modules import orchestrator.offline_lane, so each adds a row to the merge-lane ratchet baseline: ι carries both ratchet files (§9 listed two files).

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| red-path-integration-module | capability→producer — ι creates orchestrator/tests/test_offline_lane_red_path_integration.py (scenarios 1–23). Every producer it exercises is upstream (α, β, γ, δ1, δ2, δ3). | PASS | path `orchestrator/tests/test_offline_lane_red_path_integration.py` present |
| ledger-and-outage-integration-module | capability→producer — ι creates orchestrator/tests/test_offline_lane_ledger_and_outage_integration.py (scenarios 25–38). Producers upstream: π, ε1, ε2. | PASS | path `orchestrator/tests/test_offline_lane_ledger_and_outage_integration.py` present |
| stateful-fake-task-store | capability→producer — absent on main (PRD §7 "Not assumed": no orchestrator-side fixture for a real fused-memory task backend); ι builds it over γ's scenario-24 fixture. | PASS | manual — a test helper |
| injectable-clock | producer:δ1 (the engine's clock) and producer:α (the record store's) — upstream; δ2 hands one clock to both. | PASS | manual — consumed |

### ζ0 — Offline-lane red path ζ0: the soak report script — ζ's mechanical checks and its re-measurement figures, read from the lane's run records, the flake ledger, the escalation queue and the task store

Added at decompose. As §9 was written ζ's mechanical checks had no producer (producer-absent); resolved by queueing this leaf upstream of ζ.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| soak-report-script | capability→producer — ζ0 creates scripts/offline_lane_soak_report.py. Every fact it reads has an upstream producer: pass headers and the record read API (α); the outcome block (β's form, written by δ2); the effect block (δ1's, written by δ2, with ε2 adding the debt results from ε1's DebtFilingResult); occurrence and debt rows at the lane's call site (ε1, ε2); the stuck-write and outage records (δ1, π); the structured call site on regression escalations (ε1); tasks through get_task. As first drafted this row was producer-extent-short — no leaf persisted a pass's outcome or an ended episode — and was resolved by the two blocks. | PASS | path `scripts/offline_lane_soak_report.py` present |
| no-data-is-not-a-pass | rejection-mechanism — built by ζ0: an empty tree or an unreadable store exits 2 with every check `no_data`. | PASS | manual — observed by ζ0's own tests |

### ζ — Offline-lane red path ζ (OPERATOR GATE, 7-day soak): re-measure persist_passes, recheck_secs and red_unlanded_hours on the deployed lane, and put the three unreviewed D6 calls in front of Leo

Operator gate (deterministic, always_escalates, delayed milestone of seven days). "Seven days after deploy" has no machine expression: the milestone anchors on ι and ζ0 being done and runbook step 0 checks the deployed window.

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| mechanical-checks | producer:ζ0 — upstream. | PASS | manual — an operator gate; the script's exit code is read by a person |
| seven-days-of-the-deployed-lane | docs/task-authoring.md §6: `delayed` anchors on dependencies being done, not on a deploy. Resolved in the runbook: T is the earliest run record of the cut-over lane, and the gate is not resolved before T + 7 days. | PASS | manual — a runbook step |
| rulings | the gate's output is Leo's ruling on three knobs and on the three D6 author's calls. | PASS | manual — a human decision |
