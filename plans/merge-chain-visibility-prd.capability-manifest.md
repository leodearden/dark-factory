# Capability manifest: merge-chain visibility

**PRD:** `plans/merge-chain-visibility-prd.md` · **Sidecar:** `plans/merge-chain-visibility-prd.capability-manifest.yaml`
**Code anchors** verified against main `fd3f84d7ed` (2026-11-18) and task 8528's branch `.worktrees/8528` at `be6a4d3fb8`. Cited by symbol.

Every binding below is PASS. Each `grep:` pointer (47 distinct) was resolved against main: `git cat-file -e main:<path>`,
and the name looked up as a module-level or class-level definition in `git show main:<path>`. Each built-here `grep`
check was run with `git grep -E` on main and is absent today, so it fails now and passes when its producer lands.
Instance attributes named in prose (`_next_dispatch_unchained`, `_train_verify_host`, `_shutdown_timeout`) are set in
`SpeculativeMergeWorker.__init__` and carry no pointer of their own.

**Signal leaves: γ and ε.** The PRD's goal (§1) has two parts. "Now" is delivered in its final form by γ, the live-seam
integration gate (L1, L5–L8); α is an interim copy that γ retires (D18). "Recently" is delivered by ε, the history-seam
integration gate (H22–H24), which also reads the records β writes. ε depends on γ, so ε done means the whole goal; γ
done alone means the live half, which the dashboard shows as partial.

**G5 (B + H).** The §9 rows are carried by the leaves that build each side: H1–H21 and H23's audit refusal by β; H22,
H23's panel reading and H24 by ε; L2–L4 and L8 by α; L1 and L5–L8 by γ.

**Fixes made at decompose (bindings that would not have held as the PRD reads):**

1. `ChainDepthBasis` is defined in `orchestrator/src/orchestrator/merge_verify_facts.py` on 8528's branch, and nothing
   under `shared/src/shared/` imports `orchestrator`. §8.1 types `ChainRoundStarted.basis` as `ChainDepthBasis` in a
   shared module restricted to stdlib and local types. As written that binding is producer-extent-short. Fixed in β:
   β moves the enum into `shared/src/shared/merge_chain.py` and `merge_verify_facts.py` imports it from there (one
   home, INV-5). Mechanical check added.
2. The live probe drops all but six keys per entry, and no `request_id`
   (`dashboard/src/dashboard/data/merge_queue.py::_normalize_entry`). γ's "Currently queued" columns join by request
   id, and nothing in γ's PRD file list carried `entries[].chain`, `entries[].unchained_mark` or `request_id` to the
   page. Fixed in γ: `data/merge_queue.py` is added to its files, and its brief chooses one path, widening the
   normalizer or carrying the keys in the `merge_chains` projection.
3. γ and δ both edit `static/redux/merge_queue.js` and `tabs.jsx`, and δ and ε both edit `merge_queue.js`, with no
   edge between γ and δ in the PRD's DAG. The overlay requires real edges for same-file pairs, so γ now depends on δ.
   δ's prerequisites (α, 8528, 10621) are a subset of γ's transitive ones, so γ is not delayed by more than δ's own
   run. ε reaches δ through γ.
4. δ's reader would print "bisect rounds: 0" over a window with no `chain_depth_reason` rows, which looks the same as
   "no bisect rounds" (INV-13). δ's brief now asks for "—" with the reason when no deep row in the window carries the
   key.

**Producers outside this batch.** 8528 (blocked at 2026-11-18, not cancelled), 10624, 10721, 10621 (with 10619 and
10620 upstream), 8222, 9852 (with 9800 upstream), 8547. Each was checked to exist and not be cancelled.

**G7.** All five tasks were walked against `docs/legibility/design-invariants.md`. No waivers.

- α: INV-13: the reader prints "orchestrator predates task 5371" when `restart_drain` is absent and an unknown with
  the probe error on a failed probe. INV-9: "is X verifying" has one home, `restart_drain.verifies_in_flight[]` (D21).
  INV-10: the fixture skeleton test executes the projector. INV-6: the panel claims no liveness beyond one reply.
- β: INV-1: constructor-checked typed records, I1–I5. INV-2: settles carry outcome, stop, cause and fault; rejected
  records name field and value. INV-5: one wire module, and the `ChainDepthBasis` move (fix 1). INV-7: open rounds
  are bounded by verifies in flight plus builds, with one open search; a restart leaves unsettled records that panels
  label. INV-9: D15. INV-10: lane-scene behaviour tests and an executable audit. INV-11: D16, a fail-safe ledger
  whose failures are recorded as events. INV-12: any ratchet raise names this task. INV-13: the audit refuses an
  unrecorded window.
- γ: INV-9: it extends the one `_verifies_in_flight` derivation and does not add a second; `chain_reservations[]`
  becomes tests-only (D22). INV-10: the fixture regenerator. INV-5: the fixture is the one pin of the live wire (D23).
- δ: INV-5: one shared rule, which the agreement test pins to the script. INV-13: fix 4.
- ε: INV-13: "β not deployed" (D26). INV-11: undecodable rows are counted and shown (I5, H24). INV-1: totality test
  for `CHAIN_OUTCOME_FAMILY`.
- INV-3, INV-4, INV-8 and INV-14 have no hit in any task.


## α — "Verifying now" from the live reply

_Interim live panel before 8528; its task-id decided-link flag is retired by γ (PRD D18, D24)._

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| verifies-in-flight-on-the-reply | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot` `grep:orchestrator/src/orchestrator/merge_lane/types.py::VerifyInFlight.to_wire` wired: snapshot serves restart_drain.verifies_in_flight from _verifies_in_flight (task 5371, done) | PASS | — |
| chain-reservations-on-the-reply | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot` `grep:orchestrator/src/orchestrator/merge_lane/chain_reservation.py::ChainReservations.snapshot` wired: snapshot serves chain_reservations[] (task 7720, done) and entries[].verify_base (task 5447) | PASS | — |
| probe-site-sees-the-raw-reply | `grep:dashboard/src/dashboard/data/merge_queue.py::_probe_live_one` holds the raw reply before `grep:dashboard/src/dashboard/data/merge_queue.py::_normalize_entry` drops all but six keys | PASS | — |
| live-freshness-bound | `grep:dashboard/src/dashboard/data/merge_queue.py::LIVE_QUEUE_FRESHNESS_BOUND_SECONDS` | PASS | — |
| jsdom-harness-and-wire-fixtures | `grep:dashboard/tests/js/dom_harness.mjs` `grep:dashboard/tests/js/wire_fixtures.mjs` | PASS | — |
| merge-tab-mount-point | `grep:dashboard/src/dashboard/static/redux/tabs.jsx` (MergeTab, a JSX component) | PASS | — |
| projector-module | built here: dashboard/src/dashboard/data/merge_chains.py, a pure projection serving the verifying_now Datum | PASS | grep `verifying_now` in dashboard/src/dashboard/data/merge_chains.py, present |
| verifying-now-panel | built here: dashboard/src/dashboard/static/redux/merge_chains.jsx | PASS | grep `verifying_now` in dashboard/src/dashboard/static/redux/merge_chains.jsx, present |
| live-wire-fixture | built here: shared/tests/fixtures/merge_queue_snapshot_chain.json, captured once from df's real get_merge_queue reply (D23) | PASS | grep `verifies_in_flight` in shared/tests/fixtures/merge_queue_snapshot_chain.json, present |
| decided-link-flag-and-unknowns | built here: the task-id flag (D18) and the two unknown readings (L3, L4) | PASS | manual: Rendering states pinned by L3, L4 and the jsdom states; the flag text is interim (γ replaces it), so a text grep would pin a copy γ retires. |

## β — the merge lane records chain rounds and culprit searches

_Fix at decompose: ChainDepthBasis lives in orchestrator merge_verify_facts.py on 8528's branch and shared never imports orchestrator, so β moves the enum into shared (one home) for merge_chain.py's basis field._

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| depth-policy-and-suspects | producer:task-8528 upstream (ChainDepthBasis, choose_depth, _set_chain_policy, _note_tip_result, _is_code_verdict, chain_policy.py, the reservation context managers) | PASS | — |
| red-tip-and-finally-sites-settled-first | producer:task-10624 upstream; producer:task-10721 upstream (same worker.py sites: red-tip exit, _run_inflight_verify's finally, _deep_chain_placement) | PASS | — |
| exit-sites | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._deep_chain_placement` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._run_inflight_verify` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._tear_down_inflight_entry` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._land_chain_prefix` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._void_and_remerge` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._land_rebased_or_park_gate` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._finalize_landed_by_successor` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._finalize_inflight` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._end_chain_reservation` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._expire_chain_reservations` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.stop` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._advance_candidate` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::_HALT_ADVANCE_RESULTS` | PASS | — |
| abort-causes | `grep:orchestrator/src/orchestrator/merge_lane/verify_cancellation.py::VerifyCancelCause` `grep:orchestrator/src/orchestrator/verify_runner.py::RunnerUnavailable` `grep:orchestrator/src/orchestrator/git_ops.py::MergeVerifyLeaseContended` | PASS | — |
| emit-seams | `grep:orchestrator/src/orchestrator/merge_verify_facts.py::MergeVerifyDispatchFacts` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._emit_verdict_voided` `grep:orchestrator/src/orchestrator/merge_lane/dead_base_ledger.py::DeadBaseKiller` `grep:orchestrator/src/orchestrator/merge_lane/types.py::ChainResult` | PASS | — |
| event-store-reads-and-emit | `grep:orchestrator/src/orchestrator/event_store.py::EventStore.fetch_events_by_type_all_runs` `grep:orchestrator/src/orchestrator/event_store.py::EventStore.emit` `grep:orchestrator/src/orchestrator/event_store.py::EventStore.__init__` (emit never raises; the constructor creates a missing store) | PASS | — |
| lifetime-budget-sources | `grep:orchestrator/src/orchestrator/verify.py::merge_verify_command_budget_secs` `grep:orchestrator/src/orchestrator/config.py::MergeDeepConfig.chain_build_timeout_secs` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.MAX_CAS_RETRIES` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES` | PASS | — |
| public-fault-seams-for-lane-scenes | `grep:orchestrator/tests/_merge_lane_fakes.py::make_lane` `grep:orchestrator/tests/test_merge_speculation.py::_AdvanceFailingGitOps` | PASS | — |
| ratchet-raise-home | `grep:orchestrator/tests/merge_lane_ratchet_raises` | PASS | — |
| chain-depth-basis-in-shared | built here: ChainDepthBasis moves from orchestrator merge_verify_facts.py (8528) into shared/src/shared/merge_chain.py; merge_verify_facts.py imports it | PASS | grep `^class ChainDepthBasis` in shared/src/shared/merge_chain.py, present |
| round-records | built here: shared/src/shared/merge_chain.py frozen records with to_event_data/from_event_data | PASS | grep `^class (ChainRoundStarted\|ChainRoundSettled\|ChainSearchSettled\|ChainRecordRejected)\b` in shared/src/shared/merge_chain.py, present |
| ledger | built here: orchestrator/src/orchestrator/merge_lane/chain_rounds.py::ChainRounds | PASS | grep `^class ChainRounds\b` in orchestrator/src/orchestrator/merge_lane/chain_rounds.py, present |
| chain-event-types | built here: merge_chain_round_started, merge_chain_round_settled, merge_chain_search_settled, merge_chain_record_rejected in EventType wherever it lives at dispatch | PASS | grep `merge_chain_record_rejected` in orchestrator/src/orchestrator/event_store.py, shared/src/shared/run_events.py, present |
| merge-verify-carries-chain-ids | built here: MergeVerifyDispatchFacts gains chain_id and origin_chain_id | PASS | grep `origin_chain_id` in orchestrator/src/orchestrator/merge_verify_facts.py, present |
| audit-script | built here: scripts/check_chain_round_records.py | PASS | grep `runs-db` in scripts/check_chain_round_records.py, present |
| ledger-fails-safe | built here: no ChainRounds method raises into the lane; a bad record becomes merge_chain_record_rejected (D16, H20) | PASS | manual: A behavioural property; H20 asserts it through a lane scene and a raising sink. |
| audit-refuses-vacuous-windows | built here: the audit refuses a missing path and a window with deep verifies and no round records (H21, H23) | PASS | manual: Behaviour of the script over fixture stores; H21 and H23 run it. |

## γ — snapshot round view; round, queue and controller lines

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| ledger-round-lookup | producer:task-β upstream (ChainRounds.round_of, chain_id on ChainResult) | PASS | — |
| live-panel-and-fixture | producer:task-α upstream (data/merge_chains.py, merge_chains.jsx, shared/tests/fixtures/merge_queue_snapshot_chain.json) | PASS | — |
| policy-state | producer:task-8528 upstream (merge_lane/chain_policy.py::ChainPolicyState, depth_table, suspects) | PASS | — |
| snapshot-and-in-flight-derivation | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot` `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._verifies_in_flight` `grep:orchestrator/src/orchestrator/merge_lane/types.py::VerifyInFlight` | PASS | — |
| entry-normalizer-drops-new-keys | `grep:dashboard/src/dashboard/data/merge_queue.py::_normalize_entry` keeps six keys and no request_id today; γ widens it or carries the keys in the projection | PASS | — |
| unchained-mark-on-the-wire | built here: entries[].unchained_mark and entries[].chain | PASS | grep `unchained_mark` in orchestrator/src/orchestrator/merge_lane/worker.py, present |
| controller-line-on-the-wire | built here: top-level chain_policy from ChainPolicyState.wire | PASS | grep `def wire\(` in orchestrator/src/orchestrator/merge_lane/chain_policy.py, present |
| fixture-regenerator | built here: orchestrator/tests/test_merge_queue_snapshot_chain_fixture.py | PASS | grep `merge_queue_snapshot_chain\.json` in orchestrator/tests/test_merge_queue_snapshot_chain_fixture.py, present |
| dashboard-reads-round-view | built here: the projection reads unchained_mark and chain_policy | PASS | grep `unchained_mark` in dashboard/src/dashboard/data/, present |
| task-id-flag-retired | built here: per-member decided replaces α's task-id flag (D18) | PASS | manual: L5 asserts the decided member renders; the retired flag's wording is α's choice, so an absent-grep would pin unknown text. |

## δ — chain telemetry counts bisect rounds apart

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| shared-telemetry-module-and-card | producer:task-10621 upstream (summarize_chain_telemetry moved to shared/src/shared/chain_telemetry.py; the Deep chains card) | PASS | — |
| basis-on-merge-verify | producer:task-8528 upstream (merge_verify.chain_depth_reason.basis; ChainDepthBasis.BISECT == 'bisect') | PASS | — |
| digest-and-throughput-script | `grep:orchestrator/src/orchestrator/digest.py` `grep:scripts/merge_lane_throughput.py` `grep:scripts/tests/test_chain_telemetry_agreement.py` | PASS | — |
| bisect-split-in-shared-module | built here: ChainTelemetry gains bisect_rounds and bisect_red | PASS | grep `bisect_rounds` in shared/src/shared/chain_telemetry.py, present |
| throughput-script-uses-shared-split | built here: scripts/merge_lane_throughput.py prints the bisect count from the shared function | PASS | grep `bisect_rounds` in scripts/merge_lane_throughput.py, present |
| agreement-test-covers-bisect | built here: the agreement test gains a bisect row | PASS | grep `bisect` in scripts/tests/test_chain_telemetry_agreement.py, present |

## ε — "Chain rounds" history panel

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| runs-read-model | producer:task-9852 upstream (shared/src/shared/runs_read/merge_lane.py, OUTCOME_FAMILY, OutcomeFamily, contract rows); producer:task-9800 upstream (shared.runs_read RunsQuery, orchestrator/tests/test_runs_read_contract.py) | PASS | — |
| round-records-and-events | producer:task-β upstream (shared/src/shared/merge_chain.py from_event_data, the four event types) | PASS | — |
| live-panel-current-shape | producer:task-γ upstream (data/merge_chains.py, merge_chains.jsx) | PASS | — |
| recent-deep-verify-list | producer:task-8547 upstream (the one list of recent deep verifies, A-8547) | PASS | — |
| outcome-colours-module | `grep:dashboard/src/dashboard/data/outcome_colors.py` `grep:dashboard/tests/test_outcome_colors.py` | PASS | — |
| chain-rounds-query | built here: chain_rounds, a RunsQuery in runs_read/merge_lane.py | PASS | grep `def chain_rounds\(` in shared/src/shared/runs_read/merge_lane.py, present |
| chain-outcome-family | built here: CHAIN_OUTCOME_FAMILY beside OUTCOME_FAMILY | PASS | grep `^CHAIN_OUTCOME_FAMILY` in dashboard/src/dashboard/data/outcome_colors.py, present |
| chain-rounds-datum | built here: the chain_rounds Datum in the /merge-queue payload | PASS | grep `chain_rounds` in dashboard/src/dashboard/data/merge_chains.py, present |
| pre-beta-store-and-undecodable-rows | built here: 'β not deployed' reading and undecodable count (H23, H24) | PASS | manual: Behaviour over fixture stores; H23 and H24 assert it. |
