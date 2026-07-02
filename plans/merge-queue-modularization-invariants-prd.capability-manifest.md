# Capability manifest — merge-queue-modularization-invariants-prd

Binds each task's RED/observable signal to substrate evidence (G3+G6
mechanization). Line anchors verified 2026-07-02 against main at `9559f8ab0f`
(study session) / `77c5da9dbc` (PRD commit). All bindings PASS; no
`declared-only`/`test-only`/`producer-downstream`/`fixture-ERROR`/`bound≤floor`/
`rejection-absent` findings.

## α — Extract merge_types.py
- Types to move exist & are wired: `grep merge_queue.py:3872` (MergeRequest),
  `:3962` (GroupMergeRequest), `:4002` (MergeOutcome), `:4056` (SpeculativeItem),
  `:4078` (InflightEntry), `:3516` (WaiterRecord), `:2885` (InFlightMergeRegistry),
  `:2775` (TerminalOutcomeRetention), `:527` (MergeBounceRegistry) — all
  referenced from the production loops (not test-only).
- Importers requiring the re-export shim: `workflow.py:5207` (MergeRequest
  construction), `workflow.py:939` (GroupMergeRequest), `harness.py:4958`
  (worker construction), `escalation/server.py:829` (MCP MergeRequest).
- Doc map target exists: `skills/merge-queue/references/two-layer-model.md`
  (as-built symbol map; confirmed by tests+docs survey 2026-07-01).

## β — Extract merge_gates.py
- Gate fns exist & are wired into the landing path: `_check_plan_targets_in_tree`
  merge_queue.py:1464 (called :9156, :5352), `_check_plan_files_touched_in_branch`
  :1602, `_check_post_merge_equivalence` :1884 (called :1239),
  `_check_post_merge_pyright` :2390 (called :1303), `_rebase_delta_touched_overlap`
  :2090, `_reverify_rebased_tree` :2222 (called :10640), `_finalize_advanced_merge`
  :1155 (called :10558, :5419), `_map_advance_failure` :1350.
- Reason prefixes are load-bearing consumer strings: constants :138–201; their
  docstrings name the workflow.py routing contract (frozen by this PRD's contract).

## γ — Extract shadow/drift/liveness modules
- Functions exist as free functions taking the worker as a parameter (clean
  extraction): shadow-compare block merge_queue.py:11395–12598
  (`_run_shadow_compare` :12346, `_maybe_schedule_shadow_compare` :12500),
  drift `_run_drift_check` :12599 / `_maybe_run_drift_check` :12732, liveness
  `check_merge_liveness_margin` :11179 / `enforce_merge_liveness_margin` :11301,
  warm-worktree `_acquire_warm_verify_worktree` :12785.
- Runtime consumer wiring: `_finalize_inflight` calls shadow :10597 and drift
  :10603 on every 'done' landing (journal-observable).

## δ — Extract SuffixConflictTracker
- State + methods to encapsulate: `_suffix_conflict_graph` :6155,
  `_suffix_conflict_signature` :6158, `_bounce_registry` :6170,
  `recompute_suffix_conflict_graph` :6918, `_bounce_conflicting_suffix_items`
  :7178, `SuffixConflictGraph` :4179.
- Production consumers: `_acquire_next_request` calls recompute :7363 and
  bounce :7370 each acquire cycle; `snapshot()` :7458 reads the graph;
  `_pop_next_pickable` :6620 reads footprint_neighbors.

## ε — Item-shape validation + replace-only rebuilds (I2, I3)
- Rebuild sites to convert: merge_queue.py:10683 (rebased_pending_reverify)
  and :10761 (cas_failed) — both currently hand-constructed (1928 fix carried
  `merged_branch_tip` through them: commits 3aa6308ba0, 1b7f609242).
- Shape invariants hold on main today (validation cannot break existing
  construction): immediate_outcome⟂merge_result documented SpeculativeItem
  :4064–4069; merger-loop constructions obey it (:8947, :8985, :9016, :9047,
  :9065, :9102, :9133, :9180, :9202; `_remerge` :10006–10113).
- Sentinels to enum-ify: InflightEntry.status :4125, InflightVerifyResult.status
  :4173 (values 'DROPPED'/'REQUEUED'/'RUNNER_UNAVAILABLE'/'*_PREDISPATCH').
- Rejection-mechanism check (branch 4): the raise-at-construction signal is
  produced by the `__post_init__` this task itself builds and is asserted by
  the same task's CI test — mechanism and assertion live on the same leaf.

## ζ — Resolve-and-release chokepoint
- The five duplicated handlers exist: verifier-loop BaseException blocks at
  :9532 (dispatch), :9578 (passthrough-finalize), :9614 (finalize-head),
  :9757 (cascade), :9814 (blocking-get) — each hand-replicates
  resolve+release+clean today.
- Release primitives exist: `_speculation_slot.release()` (:9548 et al.),
  `_host_allocator.cancel_and_release`/`release` (:10783–10787),
  `_cleanup_owned_merge_worktree` :6456, `_resolve_or_drop_abandoned` :10116.

## η — Request-liveness ledger (I1)
- Heartbeat substrate: `_heartbeat_loop` :7762 (30 s poll),
  `_maybe_log_queue_heartbeat` :7698.
- Escalation substrate: `_escalation_queue` injected at :5970/:6043 and wired
  by the harness (worker construction harness.py:4958–4966); dedup'd-alarm
  precedent `_alarm_verify_host_unreachable` :11859 (has-open-L1 dedup shape).
- Terminal/requeue transition points the ledger must hook: ζ's chokepoint
  (upstream in-batch producer, DAG-direction PASS), requeue sites
  `_queue.put_nowait` :10868 (pre-dispatch halt), :10304 (abort-poll REQUEUED),
  stop() drains :8107–8274.

## θ — SpeculationController (R3+R5)
- State to encapsulate (all currently merger-loop locals): spec_base/prefetched/
  held_spec_permit/pending_spec_base/pending_predecessor :8730–8756.
- Permit lifecycle sites: acquire :9248, transfer-on-put :9236–9238,
  late-arrival ATTACH/FALLBACK :8818–8841, cascade release :9729, finalize
  release :10788, stop() over-release :8114.
- Conservation identity documented (the property test's spec): cap invariants
  comment :217–228.

## ι — Conservation audits (I4, I6)
- Ledger substrate: `_owned_merge_worktrees` :6220 with scope exemptions
  documented :6206–6219; register :6436 / deregister :6449 / touch :6486.
- Semaphores to audit: `_speculation_slot` :6026, `_merge_ahead_cap` :6032
  (both plain Semaphore; audit is read-only accounting, upstream θ makes the
  merger-held count queryable — DAG-direction PASS).
- Surface: `snapshot()` :7458 (additive key), heartbeat (η upstream),
  escalation queue (see η bindings).

## κ — Shared guard pipeline (R4)
- The duplicated pipelines exist: `MergeWorker._do_merge` :5256–5480,
  merger-loop steps 0–3b :8963–9191, `_remerge` :9914–10114 (incl. the
  speculation-race retry duplicate :9969–10069).
- Shared guard helpers already extracted (pipeline composes them):
  `_classify_branch_presence` :2540, `_check_plan_targets_in_tree` :1464,
  `_build_merge_failure_diagnostic` :9868, `_request_abandoned` :5077.

## λ — Declarative gate chain (R6)
- Sequential gate calls to replace: `_finalize_advanced_merge` runs equivalence
  :1239 then pyright :1303 inline; both fns are β-relocated (upstream in-batch
  producer — DAG-direction PASS).

## μ — AdvanceOutcome value object (R7a)
- Side channels exist (the thing being deleted): documented git_ops.py:4404–4408;
  getattr reads merge_queue.py:10626–10628; AssertionError backstop :10629–10636.
- `advance_main` signature to change: git_ops.py:4348.

## ν — Retire MergeWorker (R7b)
- MergeWorker is production-unreferenced: sole production worker construction
  is SpeculativeMergeWorker at harness.py:4958; deprecation note harness.py:4882;
  every `MergeWorker(...)` construction is under orchestrator/tests/ (survey
  2026-07-01: test_merge_queue.py, test_atomic_train_merge.py,
  test_workflow_e2e.py, test_train_integration.py,
  test_merge_queue_equivalence.py). No config flag selects workers.
- Protocol to simplify: `_TrainMergeHost` :4262.

## ξ — Verify-base promotion (I5) + single-writer asserts (I7)
- The log-only guard exists and is wired: `_warn_if_verify_base_not_frozen_tip`
  :6878, called at dispatch :11051–11055 (fail-open, control-flow-neutral).
- Promotion target exists: `two_layer_invariants` :6804 (list[str] violation
  idiom; snapshot key landed with λ=1895).
- Rejection-mechanism check (branch 4): the task's precondition step greps the
  journal for spurious `ε=1890 §5.3 guard` WARNINGs and investigates hits
  before promoting — binding the "guard is quiet in production" premise to
  observed evidence at implementation time, not assumed here.
- Single-writer discipline currently comment-only: :6147 ("Accessed only from
  the merger coroutine") — the assert helper is new, produced by this task.

## ο — Full sum-type split (R2 deep)
- Consumers to convert are enumerable: `_dispatch_item` :10801,
  `_finalize_inflight` :10364, `_verifier_loop` :9402, stop() drain :8171–8223,
  `_remerge` returns. ε's factory whitelist + grep-guard (upstream) is the
  migration rail — DAG-direction PASS.
- Exhaustiveness proof substrate: unscoped pyright gate runs on every merge
  (`_check_post_merge_pyright` :2390; config full fan-out per config.yaml:14).

## π — Invariant integration gate (B+H leaf)
- Every asserted surface has an upstream in-batch producer (DAG-direction
  PASS): liveness ledger (η), resource_audit (ι), shape-raise (ε),
  two_layer_invariants verify-base row (ξ), chokepoint uniformity (ζ),
  permit conservation (θ).
- Fault-injection substrate exists: `VERIFY_ABANDON_POLL_SECS` monkeypatch
  convention :5927, `operator_halt` :4936, RunnerUnavailable path :10310,
  integration-harness precedent test_merge_queue_two_layer_integration.py
  (λ=1895 gate) and test_coalesce_integration_gate.py.
- 1928 regression row (sketch #8): carried-tip substrate landed 3aa6308ba0 /
  1b7f609242 (rebuild sites carry merged_branch_tip; `_commit_is_linear`
  fail-safe :1112).

## ρ — Deterministic all-orchestrators deploy
- Script exists, executable, committed: `scripts/restart-all-orchestrators.sh`
  (commit 77c5da9dbc, mode 100755; syntax-checked). Single-unit predecessor
  `scripts/restart-orchestrator.sh` landed via task 1969 (done).
- Deterministic runner + detached self-restart path exist: task_kind
  deterministic batch 1898–1904 all done; detached path
  `deterministic_runner.py::_default_schedule_detached_restart` (named in
  restart-orchestrator.sh header as the sanctioned self-restart mechanism);
  `before_done.script` existence validated at submit_task (guard landed with
  the deterministic batch — validation runs before the planning_mode branch).
- Numeric floor (branch 1): `timeout_secs: 900 >` worst-case
  6 running units × (TimeoutStopSec 90 s + RESTART_VERIFY_TIMEOUT 30 s) = 720 s.
  Six running `orchestrator-*.service` units enumerated 2026-07-02.
- Rejection/failure path: preset "before_done present + always_escalates=false"
  → escalate only on failure (CLAUDE.md deterministic field-combo table);
  done_provenance `kind='deterministic-deploy-scheduled'` stamped by the
  runner (never author-supplied).
