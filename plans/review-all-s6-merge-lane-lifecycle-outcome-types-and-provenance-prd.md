# S6 — Merge-lane lifecycle, outcome types and provenance (PRD)

**Status:** draft, authored 2026-11-01 by the S6 author seat and folded six times on 2026-11-02
against the critic's passes (`critique/S6.critic.md` to `critique/S6.critic6.md`; fold log
`critique/S6.fold.md`) of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S6, agent mode, wave 2, size L, risk
high). Implements ruling **R1** (option 1: class decomposition, typed facts first) and the R1/R22
invariant promotion. G5 B+H: contracts C1–C7 and a two-way boundary-test sketch. **Post-sitting fold
2026-11-10** (`critique/S6.rulingfold.md`): every open question is ruled (§Rulings); the invariant is
INV-15 and lands first, so its four-site lockstep edit rides in α and the former docs leaf ν is gone;
task 5040 finishes first and ζ, κ and δ build on it; a refused lifecycle transition is staged into
stop-the-line (new leaves φ, ξ, υ); the lane-query port is built natively by μ with no adapter.
Fifteen leaves (α, β1, β2, γ1, γ2, δ, θ, ζ, φ, κ, λ, μ, ι, ξ, υ), none held; ξ and υ are filed by the
lead once φ is on main. **Fold-verify 2026-11-10** (`critique/S6.foldverify.md`): the lead's sequencing
rulings after the ruling folds are applied — `harness.py` order S6 α, then S3 β, δ, κ, then S6 β1 … μ
(β1 → S3 κ); task 4833 lands first (ζ → 4833); μ follows λ; α's resolver keyword is `merge_cold`;
S6-2 = B is a program-doc amendment of R1 for 5040 only. **Alignment 2026-11-10**
(`critique/S6.align.md`, the lead's rulings after the fold-verifiers): S6 α lands first on every file
it shares with another plan and takes no cross-stream edge (S5 α, S5 λ0, S7 δ, S7 β and S7 ε depend on
it there); β1 → S5 λ0 and β1 → S7 β order the rest of S6's chain; κ → 4833 beside ζ → 4833.

**Code anchors** verified against main `38a200d213` (2026-11-10, fold-verify). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. The run's anchors were verified at `6551b75aba`
(2026-10-08), the author and critic passes at `ce0f71a384`/`0935e63d88`, the 2026-11-02 folds at
`d40ba048e9`. Between `d40ba048e9` and `e634cfe658` (2,532 commits) `merge_lane/` took 36 commits
(tasks 8523, 5302, 5290, 6670, 6455, 4496), `worker.py` grew net +58 lines, and `halt.py`,
`suffix_graph.py`, `main_health_actions.py`, `unblock_types.py`, the INV docs and
`plans/merge-queue-modularization-invariants-prd.md` did not change. Between `e634cfe658` and
`38a200d213` (78 commits) task 8524 (merge-lane policy: re-anchor and re-merge onto a dead base) landed
in `worker.py` (21,144 → 21,347 lines; `_dispatch_item` 610 → 367, its re-merge arms now
`_remerge_at_dispatch`/`_park_unhosted`) and `types.py` (`RemergeBase`, `NoHost`); no other file S6
cites changed in a counted symbol. Every count below is labelled `measured at 38a200d213` and names its counting rule where the rule changes the number. Cite by
symbol; re-measure at implementation time.

## Goal

The merge lane states its own facts once, in types, and the rest of the orchestrator reads them
instead of re-deriving them: an outcome carries its `kind`, a train carries its `train_kind`, a
landed row is a collaborator every reader is handed, the main-health auto-heal policy has one
home, the lane's timing budgets are derived from the merge-verify budget they must exceed and a
config that breaks the ordering refuses to load, a lane's halt is the union of its named holders,
and where a request IS in its life is the `ItemLifecycle` registry's answer — not the union of
five containers. A transition the registry refuses is a typed fact every caller branches on, and —
once the operative registry has proved itself quiet — it stops that request's lane until a human
resolves it. The `SpeculativeMergeWorker` decomposition already filed from
`plans/merge-lane-quality-prd.md` (tasks 5040 → 5041 → 5042 → 5306 → 5045) runs alongside: its
first step, 5040, finishes first (ruling S6-2), and S6's lifecycle, limits and auto-heal leaves
build on its result. `AdvanceOutcome.detail`, the `_map_advance_failure` reads, its outcome-taking
signature and both worker.py callers are S5 leaf η's (S5 C5; OQ-13 (a)).

What a user observes when this lands (the signal leaf is ι; before it, α–δ):

- `orchestrator run --config <yaml>` with `merge_verify_progress.no_progress_budget_secs: 9000`
  and `merge_verify_cold_command_timeout_secs: 10800` exits non-zero at config load with a message
  that names both values and the invariant slug `lane-budgets-exceed-verify-timeouts`; the same
  pin with merge-cold 7200 loads. With the knob unset, `get_merge_queue` reports
  `timing_budgets.no_progress_budget_secs` = the resolved merge-verify budget plus the margin
  (7800 on dark-factory, 11400 on reify), not a constant. This validator is R1's first leaf and,
  under the lead's revised order of 2026-11-10, lands first of the wave's three invariants: INV-15
  `lane-budgets-exceed-verify-timeouts` appears in `docs/legibility/design-invariants.md` in the
  same leaf and the four-site consistency guard passes. It has no cross-stream prerequisite: it lands
  first on every file it shares with another plan, the sibling tasks there (S5 α, S5 λ0, S7 δ, S7 β,
  S7 ε, S3 β, S10 θ) depend on it, and it is never held behind S3 or task 4833 (lead, 2026-11-10).
- A `merge_finalized` event and its `MergeOutcome` carry `kind: <OutcomeKind>`; `workflow.py`
  contains no `result.reason.startswith`; a `GroupMergeRequest` carries `train_kind`.
- A WIP halt on the project root and the auto-heal hold on `normal` are two named holders:
  `get_merge_halt_status` lists both, resolving one leaves the lane held by the other, and the
  lane runs only when its holder set is empty.
- `get_merge_queue`'s snapshot `entries` are read from the lifecycle registry, so a request that is
  TERMINAL never appears, a request in flight appears exactly once, and the dashboard's
  merge-queue panel shows the same keys it shows today.
- A refused lifecycle transition files its L1 and the lane carries on (policy `escalate`, the
  default). Fourteen days after the operative registry lands with zero refusals on dark-factory and
  reify, a dated gate arms `hold_lane` on dark-factory: a refusal then places a holder of kind
  `lifecycle_refused` on that request's lane, owned by the refusal's escalation, the other lane
  keeps landing, and the snapshot reports `lifecycle_refusal_policy: hold_lane`.
- A `TaskWorkflow` or `TaskGroundTruth` built without a merge worker still sees landed rows:
  the journal is a required constructor argument, not a class-level global bound by whoever was
  constructed first. The autouse conftest rebinding is gone.

## Background

The `/review-all` run `review-all-dark_factory-20261008` found that the merge lane encodes its
load-bearing facts in strings and containers: a `MergeOutcome` carries its failure class only as
a `reason` prefix routed by `startswith`; a train's kind is the `coalesce-` prefix of its id; an
advance failure's facts travel as `_last_*` attributes on the shared `GitOps` instance; the
landed-row journal reaches its readers through `MergeProvenance._outbox`, a class global bound by
the worker's constructor; the main-health auto-heal policy is written twice with its eligibility
predicate copied four times; three timing constants restate, in prose, an ordering against config
timeouts that every live config now violates; and the `ItemLifecycle` registry the 2026-07
reliability work made "the single lifecycle state" is the one terminal record not driven by the
request's own future — 21 raw `set_result` calls and 32 hand-placed
`_retire_item`/`_resolve_or_drop_abandoned`/`_note_requeue` calls (35 including the 3
definitions; AST call count, measured at 38a200d213, unchanged since the run) take the terminal
edge by hand.

R1 ruled option 1 — decompose the class, sequenced typed facts first — because the live hazards
(the silent train halt, the budget ordering) rest on the typed-facts leaves, and option 2 pays the
decomposition cost later without reducing it. R22 item 8 promotes the timing-budget ordering to a
configuration invariant owned here. R23 routes no weakened key to S6.

**A refused transition today.** `worker.py::SpeculativeMergeWorker._note_transition` catches
`IllegalLifecycleTransition`, and `worker.py::_alarm_illegal_lifecycle_transition` files one L1
per request id (category `merge_lifecycle_transition_rejected`, plain `EscalationQueue.submit` with
an id minted by `make_id`, deduplicated by `has_open_l1`) and emits `escalation_created` with
`request_id`, `from_state` and `to_state`; the caller carries on. That is decision 4 of
`plans/merge-queue-modularization-invariants-prd.md` ("Invariants escalate loudly, degrade never").
Measured on 2026-11-10 across all nine projects' `runs.db` (`digests/discussion/S6.md` §S6-5):
25 refusals ever, all between 2026-07-11 and 2026-08-07 (dark-factory 2, reify 23), none in the last
90 days; of 22 classified, 15 were real lane defects and 7 a missing edge in the registry's own
table; one real defect escalate-and-continue left open is task 4151 (pending since 2026-08-12, a
phantom re-merge). Leo ruled S6-5 = B staged into stop-the-line (§Rulings).

**Coalescing is off; declared trains run on reify.** `dark-factory-orchestrator.yaml` sets
`merge_train_coalesce_enabled: false` under "Retroactive coalescing OFF (Leo's ruling,
2026-10-29)" (`66e0b1978f`) and `merge_train_former_enabled: false` (`2515ca5bb3`, 2026-10-30);
`defaults.yaml` has both `false`. Reify's yaml (read 2026-11-10) has `merge_train_coalesce_enabled:
false` and `merge_train_former_enabled: true`, so reify still forms declared trains and θ's
holder-set rework of train advance failures changes live reify behaviour, not dormant code.
`plans/mcp-coalesce-trains-2026-10-26.md` §0 rules unwinding on exceptions and `stop()` "moot while
coalescing is off" (ruling 3) and "Stream 2 is moot while coalescing is off" (ruling 4). The
coalesce-only keys (`fk-c55b12f4b66d`, `fk-e8a03e4ec067`, the coalesce half of `fk-ba531799522b`)
therefore describe a cost that is dormant but present: they are accepted through one cancelled
acceptance task that reopens them if the flag flips (S6-6 = (a)); the author's leaf η (journal and
unwind coalesce trains) is not filed.

What changed since the run's `as_of`: the fourteen pre-package alias modules are deleted
(`9d0dfb0079`, task 5037); `worker.py` no longer imports `workflow._select_train_members`
(`merge_lane/train_selection.py`, task 5305); `_WipHaltMixin` is `merge_lane/halt.py::HaltState`
and the lane buffers, redispatch park and suffix graph are `merge_lane/intake_buffers.py::MergeIntake`
(task 5305, `f7f9957bb5`); `MergeOutcome` carries `block_category: MergeBlockCategory` and
`conflict_evidence` (task 7302) — a producer-stated fact in exactly the shape β1 generalises. Since
the 2026-11-02 folds: `snapshot()` reads the finalize head and the occupying in-flight entries
through `SpeculativeMergeWorker._rendered_inflight()` (task 5290, `848c020628`); the merge worker takes the
harness's quarantine set, holder reader and readmit hook (task 6670); `merge_lane/__init__.py`
became a typed lazy-export table (task 6455). Task 5040 was dispatched on 2026-11-03/04 and holds a
half-built branch (`task/5040`, 5 commits to `e1d8911c7a`, 2,315 lines added, not rebased since
2026-11-04): `HostQuarantine` is extracted to `merge_lane/host_quarantine.py`, and a RED test pins
`merge_lane/main_health_probe.py` (`classify_main_health_red`, `spawn_main_health_probe`,
`MainHealthResponder` — the worker's deferred auto-heal and escalation arm).

## Findings disposition

Every S6 key once, one disposition each (`docs/quality-findings-contract.md` §7 "Exactly one of").
`leaf <label>` closes the key; `(dep: <id>)` names a live task carrying part of the work that the
leaf depends on and that the lead stamps with the key (§Live tasks). §7 has no `stale`
disposition: a cost that is dormant but present is `accepted:<task>` (a cancelled acceptance task
carrying `x_acceptance_reason`) or `filed:<deletion leaf>`. Re-verified 2026-11-10: no key's cost is
gone since the 2026-11-02 folds; no new `fixed:` entries.

| Key | Sev | Disposition | Re-verified (measured at 38a200d213) |
|---|---|---|---|
| `fk-313f404f3cc6` lifecycle = container membership | high | leaf ζ | `ItemLifecycle` (worker.py) and `_LEGAL_TRANSITIONS` exist; `snapshot` still builds `entries` from containers: `_rendered_inflight()` (finalize head plus occupying in-flight entries), `_intake.pending_redispatch()`, the verifier queue, `_intake.buffered(lane)` and `_intake.queued_arrivals()`, then a `_lifecycle.non_terminal_items()` sweep for anything not already surfaced, de-duplicated by request id; `_finalizing_head_entry` (96 lines) derives the head from `_live_items` type plus deque absence. The containers remain the operative state. |
| `fk-a2984bf98f85` terminal edge by hand | high | leaf ζ | AST calls in worker.py (unchanged since the run): 21 `.set_result(` (all 21 on a `.result`; 9 in `stop`, 3 in `_run_inflight_verify`), `_retire_item` 21, `_resolve_or_drop_abandoned` 9, `_note_requeue` 2, `_note_transition` 25; `_note_transition` catches `IllegalLifecycleTransition`, files a dedup'd L1 and returns with the registry unchanged while the caller proceeds. Outside worker.py: `intake_buffers.py::MergeIntake._coalesce_reentrant_drain` and `suffix_graph.py::SuffixConflictTracker._escalate` resolve request futures; `types.py::InFlightMergeRegistry.attach` mirrors results onto waiter futures. Task 5040 moves `_run_inflight_verify` and `_dispatch_item` (4 of these sites) into its extracted modules; ζ follows them there. |
| `fk-e61eed486dbc` MergeOutcome has no kind | medium | leaf β1 (β2 completes) | `MergeOutcome` (plain `@dataclass`) `status: Literal[12]`, `reason: str`, no `kind`; `OutcomeKind` 23 members, 4 non-terminal per `types.py::_NON_TERMINAL_OUTCOMES`; `GateVerdict.emit_subtype: OutcomeKind` reaches only `merge_attempt`; workflow.py 8 `result.reason.startswith` (6 in `_submit_to_merge_queue`, 2 in `_maybe_enqueue_group_merge`); `worker.py::_derail_on_train_verify_failure` re-tags by prefix; `intake_buffers.py::MergeIntake.buffer_owned_request` routes the journal prune on `result.reason == MERGE_WORKER_SHUTDOWN_REASON`. |
| `fk-80a790e6f793` train kind is an id prefix | medium | leaf β1 | 3 `train_id.startswith(_COALESCE_TRAIN_ID_PREFIX)` sites (`_stand_down_train`, `_dissolve_coalesce_trains_if_halted`, `_merger_loop`); `GroupMergeRequest` has no kind field. The `_select_train_members` upward-import half is **fixed** by 5305 (`train_selection.py`). 0 test files reference the prefix. |
| `fk-c55b12f4b66d` coalesce train never journaled | high | `accepted:<A>` (S6-6 = (a)) | `queue_store.py::MergeQueueStore.record` returns early on `isinstance(req, GroupMergeRequest)`. Dormant: coalescing off on every project (`66e0b1978f`, 2026-10-29 ruling; reify's yaml also `false`). `<A>` is the acceptance task the decompose pass files cancelled with `x_acceptance_reason` (§Live tasks). |
| `fk-e8a03e4ec067` stop() strands coalesce members | high | `accepted:<A>` (S6-6 = (a)) | `stop()` drains `self._intake.drain_lanes_for_shutdown()` with no `_unwind_coalesce_train`. Its statement's premise "live in production: `merge_train_coalesce_enabled: true`" is false; the 10-26 plan §0 ruling 3 calls this unwind moot while coalescing is off and leaves task 6450 as-is. |
| `fk-ba531799522b` train halt has no owner | high | leaf θ | `_do_train_merge` → `_map_advance_failure(halt=worker.halt_for_wip)`; `HaltState.halt_for_wip` writes every lane's owner to `None`. The coalesce-train arm is dormant and named in `<A>`'s `x_acceptance_reason`; the declared-train arm is live on reify (`merge_train_former_enabled: true`) and registers through `workflow.py::TaskWorkflow._maybe_enqueue_group_merge`'s orphan-halt probe → `TaskWorkflow._escalate_train_halt`, after the every-lane wipe θ removes. |
| `fk-1432a52e4753` `_verifier_loop` 670 lines | high | leaf ι (dep: 5045) | `_verifier_loop` 663 lines; the head-failure cascade and the blocking-get path are intact. 5045 (ο, pending) makes both loops thin schedulers; ζ moves the cascade's terminal edges onto `retire`. |
| `fk-769c3bb58ce1` retry budgets are parallel dicts | medium | leaf κ (dep: 5040) | module-level `_run_post_merge_verify` 1,345 lines, 29 parameters (5040 moves it); 23 class-level `NAME: int|float =` constants on `SpeculativeMergeWorker`, 0 production assignments; 177 test-side pins (`.<NAME> =` plus `setattr(…, '<NAME>')` over the 23 names) in 17 files. α deletes 4 of the constants; over the remaining 19 names: 104 pins in 16 files; S7 δ (reify-narrowing deletion, S7-3 = B) deletes `MAX_POST_MERGE_VERIFY_NARROWED_RETRIES` (0 test pins), leaving κ 18 names. The per-task dicts and `MAX_*` pairs are unchanged. |
| `fk-153195971a14` worker.py h14 partition | medium | leaf ι (dep: 5045) | worker.py 21,347 lines; `SpeculativeMergeWorker` spans lines 7692–21126 (215 methods); `__init__` 718 lines. |
| `fk-e521254a5b71` worker.py h14 measurement | high | leaf ι (dep: 5045) | 54.2% prose (11,561 of 21,347 lines carry a comment token or sit inside a docstring; 54.6% at `e634cfe658`, 55.5% at `d40ba048e9` by the same rule); largest functions `_run_inflight_verify` 1,413, `_run_post_merge_verify` 1,345, `__init__` 718, `_finalize_inflight` 699, `_merger_loop` 676, `_verifier_loop` 663, `snapshot` 460, `_dispatch_item` 367, `stop` 326. |
| `fk-e7deb8bcfa95` tests reach lane internals | high | leaf κ (deps: 5446, 5777) | 46 `patch('orchestrator.merge_lane.worker.run_scoped_verification'`; conftest `_mock_merge_queue_verification` autouse still patches it; `PRODUCTION_VERIFIER` wires 7 `lambda:` resolvers. 5446 and 5777 carry the test-side work; κ carries the production side (the constant pins: 177 today, 104 after α). |
| `fk-e768d50f8363` archaeology prose | medium | leaf ι (dep: 5047) | 586 `\btask [0-9]{3,5}` (case-sensitive; 606 case-insensitive) mentions in worker.py; 70 `kappa`; 9 `esc-`; `_main_health_fingerprint`'s docstring claims "no module-level cycle" while the import is function-local (δ deletes it). 5047 (ρ, pending) relocates the prose; ι re-measures. |
| `fk-83625b943aa5` halt owner not single-owner | medium | leaf θ | `halt.py::HaltState` (205 lines, unchanged since the run's folds): `halt_for_wip` and `operator_halt` write `_lane_halt_owner[lane] = None` for every lane; `set_lane_halt_owner`/`set_halt_owner` carry 3 bare `assert`s (lines matching `^\s+assert `). Live because auto-heal can own `normal` while a high-lane WIP halt runs. |
| `fk-201d70aaf409` pipeline fns take the whole worker | medium | leaf κ | 35 `worker._` token accesses in worker.py; 3 `isinstance(worker, SpeculativeMergeWorker)` arms in `classify_and_merge` (task 6162, pending, folded into κ). |
| `fk-a0f6af37f0d6` satellites over worker privates | medium | leaf λ | reach-back half **fixed** (`9857475d2d`); `worker\._` lines: `drift.py` 12, `shadow.py` 10; both `TYPE_CHECKING`-import `SpeculativeMergeWorker`. |
| `fk-59b3df6f52a3` `_last_*` side channel | medium | `filed:<S5 η>` (OQ-13 (a)) | `_map_advance_failure(git_ops, result: str, …)` reads 5 `getattr(git_ops, '_last_…', None)`; git_ops.py 7 writes. S5 leaf η (C5) deletes writes and reads together, makes `_map_advance_failure` take the `AdvanceOutcome`, and rewrites its 2 worker.py callers (`_do_train_merge`, `SpeculativeMergeWorker._finalize_inflight`) in the same commit (its signal `grep -c "= getattr(git_ops, '_last_" gates.py = 0`); S5 disposes the key in η. |
| `fk-2c9b6486d554` journal via class global | medium | leaf γ1 (γ2 completes) | 1 `MergeProvenance.bind` (worker `__init__`); readers (AST calls) workflow.py `lookup` 4 + `consume` 3, task_ground_truth.py `lookup` 1, harness.py `consume` 2 plus 4 reads of `self._merge_worker._landed_outbox` (`Harness._reconcile_landed_outbox`, `Harness._landed_dispatch_gate`); `lookup` returns `None`/`consume` no-ops when unbound; `MergeProvenance` named in 16 `.py` files under `orchestrator/tests/` (plus the ratchet baseline and 3 authorized-raises records); `_reset_merge_provenance`/`_isolated_merge_provenance` defined in 12 files. |
| `fk-e501862ffe2c` landed→done non-atomic, detectors compensate | high | leaf γ1 (journal half); detectors → S9 ι xref | `LandedOutbox` write-ahead and `reconcile_landed_row/task/outbox` exist; the workflow ghost-loop guards remain. γ1 makes the journal a construction-time fact every reader holds; S9 ι consolidates the workflow-side detectors (xref). |
| `fk-d9e6c03dc259` landing_evidence reached through local imports | low | leaf μ | escalation imports `orchestrator.merge_lane.landing_evidence` function-locally at 3 sites (`git_authority.py` 2, `server.py` 1); the façade exports only `patch_content_contained`. Under S3-1 = A there is no adapter: S3 α creates `merge_lane/queries.py` (Protocol, records, error), μ builds the native implementation and `LIVE_STATE_OF`, and S3's two merge-tool moves (which take these imports into `orchestrator/mcp/`) depend on μ. |
| `fk-d41caf6b6331` 14 alias shims | medium | **fixed:`9d0dfb0079`** | all fourteen modules deleted 2026-10-16 (task 5037); 0 `*/src` importers of the old paths; the remaining test mentions are negative assertions. |
| `fk-4b76c9695674` timing budgets vs verify timeouts | high | leaf α | `INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 5400.0`, `RESOURCE_AUDIT_WORKTREE_GRACE_SECS = 9000.0`, `PERIODIC_REAP_MIN_AGE_SECS = 21600.0`; defaults.yaml warm 1800 / cold 5400 / merge-cold 7200; dark-factory warm 7200, cold 10800 (merge-cold inherits 7200); reify warm 7200, merge-cold 10800. `verify.py::_resolve_verify_timeout` returns merge-cold first for a merge verify, so `merge_verify_command_budget_secs` = 7200 on dark-factory, 10800 on reify; nothing compares it with the constants. Task 4581 (pending, no claimant) adds a knob and a WARNING; 7396 decides its value; both cancelled → α (S6-1). |
| `fk-7f2bce2d482c` auto-heal implemented twice | medium | leaf δ (dep: 5040) | workflow `_auto_heal_main_health` 109 + `_spawn_main_health_fix_task` 34 = 143 lines (`_post_submit_tasks`, 13, is shared with `_route_review_suggestions_to_curator` and stays); worker `_auto_heal_main_health_deferred` 89 + `_spawn_main_health_fix_task` 35 + `_post_submit_tasks` 38 + module `_file_main_health_escalation` 66 = 228; eligibility predicate at 4 sites (worker 3, workflow 1). 5040's branch moves the worker's arm into `main_health_probe.py::MainHealthResponder`; δ builds the one collaborator on that. |
| `fk-fd91b14b3ead` workflow routes on reason prose | medium | leaf β1 | see `fk-e61eed486dbc`; plus 9 `result.status == '<literal>'` switches in `_submit_to_merge_queue` (2 more in `_maybe_enqueue_group_merge`) and 2 function-local prefix imports in workflow.py. |
| `fk-75fc1b96098d` lane imports upward | medium | leaf δ | `_select_train_members` half **fixed** (5305); residue: module-level `from orchestrator.scheduler import StaleEvidenceRejection` (3 uses; S17's seam, xref) and function-local `from orchestrator.workflow import compute_preexisting_main_break_fingerprint` in `_main_health_fingerprint` (5040's RED test imports it from workflow as well). |

Synthesis `deletable` list accounted for: container-driven dispatch residue (ζ); reason-prefix
routes — 8 workflow prefix routes, 3 train-id routes, 1 shutdown-reason equality (β1); 4
`_last_*` attributes + 5 `getattr` reads and the `(git_ops, result)` signature (S5 η);
`MergeProvenance.bind` and the autouse fixture (γ1); the workflow auto-heal copy, 143 lines, and 3 of
4 predicates (δ); 3 hand-derived constants' prose (α); 14 shims (already deleted).

## Pre-conditions for activating

- **Task 5305 landed** (`f7f9957bb5`, 2026-11-01): `HaltState` and `MergeIntake` exist; ζ hands
  `MergeIntake` the lifecycle hop it already declares (`intake_buffers.py::IntakeLifecycle`).
- **Upstream streams decomposed first** (OQ-8 (a): S5, S4, S3, S7, then S6; nothing held).
  Decompose order is not dependency direction. S6 consumes four sibling contracts and depends on
  them: S5 α (`run_git` repoint of worker/gates/drift/shadow/landing_evidence) and S5 λ0, reached by
  β1 → S5 λ0 (λ0 follows S5 α in S5's chain); S7 β (the placement deletions land before S6's remaining
  worker.py leaves, S7 OQ-CQ5; β1 → S7 β, and the chain carries every later S6 worker.py leaf); S4 C1
  `VerifyContext` (leaf β; S6 κ → S4 β); and S3 C1 `MergeLaneQueries` (S3's port task α; S6 ζ, μ →
  S3 α). μ depends on no S3 merge-tool move: under S3-1 = A both moves depend on μ. **S6 α is the
  exception** (lead, 2026-11-10, after the fold-verifiers): it fixes a live misconfiguration, lands
  first on every file it shares with another plan and depends on no sibling task and on no task of
  S3's or task 4833 — S5 α → S6 α and S5 λ0 → S6 α (`worker.py`, `types.py`, `harness.py`, `verify.py`),
  S7 δ → S6 α and S7 β → S6 α (`worker.py`, `types.py`, `shadow.py`, `config.py`), S7 ε → S6 α (the
  resource-audit test and code), S3 β → S6 α (`harness.py`) and S10 θ → S6 α (the invariant files).
  Every other shared file is a same-file edit in a disjoint region, and the sibling leaf follows S6
  (§Cross-stream same-file edges): R1's typed facts come first. Three same-file exceptions, ruled by
  the lead on 2026-11-10: S5 λ0 (`QueuedBranch` → `shared.branch_names`, S5 OQ-2 A, second in S5's
  chain) precedes every S6 leaf other than α that edits `merge_lane/types.py`, `worker.py` or
  `harness.py` imports, so S6 β1 → S5 λ0 (β1 heads the rest of the chain); on `harness.py` S6 α lands
  first, then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ (S3 β → S6 α; S6 β1 → S3 κ); and task 4833
  lands first in today's files (S6 ζ → 4833 and S6 κ → 4833; S6 α is not held behind it; Live
  tasks). S6 imports `QueuedBranch` from `shared.branch_names` and never redefines it.
- **Task 5040 finishes first** (S6-2 = B). The merge-lane-quality chain (5040 → 5041 → 5042 → 5306
  → 5045 → 5046/5047/5048) is the live owner of the class decomposition and this PRD files no second
  extraction. 5040 (pending, high; no claimant at 2026-11-10) holds a half-built branch; nothing parks
  it and it waits on no S6 or S7 task, but its live dependencies are 5305 (done) and 8528 (merge-lane
  policy ζ, pending, high), which waits on 8527 → 7720 (in progress), 7721, 7824, 8526 — so δ, ζ and κ
  follow that chain too. ζ, κ and δ depend on it and edit its
  extracted modules (`merge_lane/verify_dispatch.py` and the modules its declared files name:
  `inflight_verify.py`, `post_merge_verify.py`, `host_quarantine.py`, `main_health_probe.py`); each
  re-derives that list from 5040's landed tree at filing.
- **`verify.py::merge_verify_command_budget_secs`** exists (task 5371) and
  `liveness.py::enforce_merge_liveness_margin` is the precedent for a start-time config refusal
  (`MergeLivenessConfigError`). The harness's merge-worker start already re-raises
  `MergeLivenessConfigError` and swallows only other exceptions; α's worker-start check runs
  outside that try, or `LaneTimingConfigError` is re-raised beside it.
- **The invariant number is free.** `docs/legibility/design-invariants.md` ends at INV-14
  `sampler-vocabulary-one-table`; the fixtures doc's cumulative rehearsal count is 28/28
  (measured at 38a200d213). α takes INV-15 with no upstream wait.
- **Halt work in flight**: task 4838 (pending, high; halt KIND beside `halt_owner_esc_id`, bounds
  `_handle_wip_conflict`/`_handle_wip_recovery`, fixes `harness.py::Harness._rehydrate_merge_halt`)
  and 3876 (pending, high; rehydration re-tests a preserved wip_conflict halt; its declared files add
  `orchestrator/src/orchestrator/merge_halt_rehydration.py`). θ depends on both (S6-8 = (b)).
- **The stop-the-line gate needs its measurement on main before it is filed.** ξ's
  `before_done` names `scripts/lifecycle_refusal_census.py`, which φ creates; `submit_task` refuses a
  `before_done.script` that does not exist, so the lead files ξ and υ once φ has landed.

## Sketch of approach

**Phase 1 — typed facts (R1's first leaves; α–δ).** α moves the merge-verify timeout cascade into
`orchestrator/verify_budget.py` (one definition of "authorised"), derives the lane timing budgets
from it plus margins in one frozen `LaneTimingBudgets`, lets an operator pin the no-progress
budget through the red-tier `merge_verify_progress` knob, refuses a config whose pin or margins
break an ordering at `load_config` (over the global fallback and every discovered module) and
again at worker start, and performs the four-site lockstep edit for INV-15 in the same commit (the
invariant lands first of the wave's three, so nothing makes it wait). β1 puts `kind: OutcomeKind` on
`MergeOutcome` and `train_kind: TrainKind` on `GroupMergeRequest`; every src producer states the
kind (or `__post_init__` derives it from status); workflow, the train derail re-tagger, the three
train-id sites and the shutdown prune route on them; the reason strings stay byte-identical. β2
moves the 51 test files onto one builder and makes `kind` mandatory for `blocked`/`error`. γ1 makes
`LandedOutbox` a required constructor argument of the worker, `build_workflow` and
`TaskGroundTruth`, constructed once by the harness, and deletes `MergeProvenance`; γ2 moves the 136
direct test constructions of `TaskWorkflow` onto the helper and makes its keyword required. δ, after
5040, grows 5040's main-health responder into `merge_lane/main_health.py::MainHealthAutoHeal` and
makes both arms call it; the fingerprint and eligibility predicate get one home in the lane.

**Phase 2 — halt holders, the lifecycle and its refusal policy (θ, ζ, φ).** θ makes a lane's halt
the set of its named holders, with the operator, WIP, auto-heal, train-advance and lifecycle-refused
halts each a holder kind, after 4838 and 3876 land. ζ, after 5040, moves `ItemLifecycle` to
`merge_lane/item_lifecycle.py` as the operative registry: it owns `_live_items`, resolves request
futures, and exposes one synchronous terminal chokepoint `retire(...)` whose hooks `drain_hooks()`
awaits, one `resolve_duplicate(...)` and one `requeue(...)`; the 21 raw `set_result` sites and 32
hand-placed calls route through it, wherever 5040 left them; `snapshot`, `_finalizing_head_entry`
and the merger's admission reads derive from `items_in(state)`; a refused transition is a typed
result every caller branches on, and the one default branch files the L1 and continues. φ makes
that default branch read the restart-only `merge_lane.lifecycle_refusal_policy` (`escalate` |
`hold_lane`, default `escalate`), builds the `hold_lane` arm (a `lifecycle_refused` holder on the
request's lane, owned by the refusal's escalation, the request left in place, one holder per lane
per refused edge), writes the decision-4 amendment, and ships the read-only refusal census the gate
runs. ξ, filed by the lead once φ is on main, fires 14 days after φ lands and passes only on zero
refusals on dark-factory and reify; υ then arms `hold_lane` on dark-factory.

**Phase 3 — collaborators over typed facts (κ, λ, μ; the live chain; ι).** The merge-lane-quality
chain extracts `verify_dispatch.py` (5040, first), `speculation.py`, `landing.py`, `worktrees.py`,
`telemetry.py` and the residual worker. κ, after 5040 and φ, replaces the class constants α leaves
(19, or 18 once S7 δ deletes the narrowed-retry cap) with a frozen `LaneLimits` (timing fields from
α's `LaneTimingBudgets`), the ~10 per-task retry dicts with one `RetryBudget`, takes S4 C1
`VerifyContext` in `_run_post_merge_verify`, narrows the module-level pipeline functions' host to a
protocol with no private reach, and moves the 104 remaining test-side constant pins onto
`LaneLimits` construction. λ hands `drift.py` and `shadow.py` their own state objects. μ builds
S3 C1 natively over ζ's registry, with the total `LIVE_STATE_OF` table from `ItemLifecycleState` to
`MergeState`; there is no adapter to delete (S3-1 = A). ι is the B+H gate over S6's boundary rows.

**Not filed (S6-6 = (a)):** the author's leaf η (coalesce trains journaled as `TrainRow`, unwind
from a terminal hook on every exit, boot re-drive; ~900 LOC, high). If coalescing is ever re-enabled
the acceptance task `<A>` reopens the keys and η returns as authored at `ce0f71a384`.

## Resolved design decisions

- **D1 — `kind` reuses `OutcomeKind`; no second enum (R1; heuristic *SPOT*).** Producers already
  know the kind (`gates.py::GateVerdict.block(emit_subtype: OutcomeKind)`, `_emit_merge_attempt`).
  β1 adds the 8 terminal members the public reason prefixes lack (`worktree_missing`,
  `transient_infra`, `needs_rebase`, `trivial_pass_main_red`, `train_verify_failed`, `superseded`,
  `shutdown`, `error`) and classifies each in both terminal tables
  (`merge_lane/types.py::_NON_TERMINAL_OUTCOMES` and its frozen mirror
  `scripts/merge_lane_throughput.py::_NON_TERMINAL_OUTCOMES`). The prefix→kind equivalence table
  lives in β1's test module (`orchestrator/tests/test_outcome_kind.py`), not in src, so no runtime
  `REASON_PREFIX_KIND[...]` lookup can re-grow the parser β1 removes (heuristic *structured data
  instead of meaningful strings*). Task 5046's `MergeFailureKind` would be a parallel enum (INV-5);
  it is cancelled in favour of β1, its children repointed (§Live tasks; S6-4 = (a)).
- **D2 — reason strings are frozen; routing moves, wording does not.** The
  `plans/merge-lane-quality-prd.md` §Contract freeze on wire strings, events and `snapshot()` keys
  holds for every S6 leaf. β1's equivalence test builds its expectation table from the pre-β1
  routing, so a self-consistently wrong mapping is caught.
- **D3 — `TrainKind` is a field, `coalesce-` stays an id-minting convention.**
  `GroupMergeRequest.train_kind: TrainKind` (`declared | coalesced`) is set by the two formers
  (`workflow.py::TaskWorkflow._maybe_enqueue_group_merge`, `worker.py::_maybe_coalesce_waiting_singles`).
  β1 keeps it (S6-6 = (a)): it is cheap, the 3 prefix routes still exist in the dormant
  coalescer, and declared trains are live on reify.
- **D4 — the journal is a required collaborator (heuristic *prefer stateless interactions between
  modules*).** `LandedOutbox` is constructed once in `Harness` at the existing
  `SpeculativeMergeWorker(` site and passed to the worker, `build_workflow` and
  `TaskGroundTruth.__init__` as a required keyword; `TaskWorkflow.__init__` takes it keyword-only.
  γ1 lands it with `TaskWorkflow`'s keyword defaulting to a fresh `LandedOutbox.in_memory()` per
  instance (never `None`, so no silent no-journal path opens in the γ1–γ2 window) so γ1 stays
  green; γ2 rewrites the 136
  direct test constructions (95 files) onto `orchestrator/tests/_workflow_helpers.py`, which
  supplies `LandedOutbox.in_memory()`, and removes the default, so an absent journal is a
  construction-time `TypeError`. Production reaches `TaskWorkflow` only through
  `workflow.py::build_workflow` (callers `harness.py`, `orchestrator/evals/runner.py`).
  `MergeProvenance` is deleted. The G4 table says harness (S8) constructs the journal; γ1 makes the
  one construction edit at the existing site (S6-3 = (a)); the S8 xref asks S8 to keep it there or move
  it into its composition root.
- **D5 — one `MainHealthAutoHeal`, owned by the lane, returning a decision (heuristic *well-defined
  purpose for each entity*, *SPOT*).** `merge_lane/main_health.py` (OQ-9's name) holds
  `MainHealthAutoHeal` (registry, `is_auto_heal_eligible`, `preexisting_break_is_escalatable(config,
  result)` — the predicate written four times today — the fingerprint, filing through
  `escalation.dedupe::submit_or_dedupe` with the one `DedupeConfig` as the worker does today, the
  lane hold, the fix-task spawn) and returns `AutoHealDecision`. It grows out of 5040's
  `main_health_probe.py::MainHealthResponder` (S6-2 = B: 5040's part 2 moves the worker's deferred
  arm there): the responder's "what the lane does once bare main is red" moves into
  `main_health.py`, and `main_health_probe.py` keeps the question "does bare main fail the same
  way?" (`classify_main_health_red`, `spawn_main_health_probe`) — two modules, two purposes.
  `merge_lane/ports.py::EscalationPort` (only `file(record)`) is not widened; widening it is S1's
  seam. The workflow's MAIN_HEALTH_RED arm calls `merge_worker.auto_heal.heal(...)` and maps the
  decision to `_mark_blocked`; the deferred probe calls the same method.
  `compute_preexisting_main_break_fingerprint` moves into the module; workflow re-imports it from
  the lane, so the import path the three test files use (`test_merge_queue_main_health.py` 9,
  `test_merge_queue_disposition_wiring.py` 2, `test_workflow_main_health_routing.py` 10; measured
  at 38a200d213) and 5040's probe test still resolve. `TaskWorkflow._post_submit_tasks` is kept:
  `_route_review_suggestions_to_curator` also calls it.
- **D6 — `_map_advance_failure`, its callers and `AdvanceOutcome.detail` are S5's (S5 C5, leaf η;
  OQ-13 (a), ruled 2026-11-02).** S5 η deletes the 7 `_last_*` writes and the 5 `getattr` reads,
  changes `_map_advance_failure`'s leading `(git_ops, result: str, …)` to `outcome: AdvanceOutcome`
  (the other keywords unchanged) and rewrites both worker.py callers (`_do_train_merge`,
  `SpeculativeMergeWorker._finalize_inflight`) in one commit, so no window exists in which
  `getattr(…, None)` silently returns `None`. `AdvanceOutcome.detail` is the frozen union
  `AdvanceDetail = WipOverlap | PopConflict | StashFailed | ParkLockContended | NoDetail` (S5 OQ-7 B,
  ruled), read by a `match` on `outcome.detail`. S6 files no leaf on it (ε stays deleted). Because
  η edits worker.py and gates.py, it is serialized with S6's chain by edges (S5 γ → S6 ζ, ruled
  2026-11-02, so S5 γ, ε, ζ, η and θ follow S6's typed facts through S5's own chain; S6 κ → S5 θ;
  §Cross-stream same-file edges). The S5 xref asks that η keep a `push_main` path (the
  `pop_conflict` arm awaits `git_ops.push_main()`).
- **D7 — lane timing budgets are derived from the merge-verify budget; a pin must exceed it
  (R1 invariant; heuristic *clear invariants*, *SPOT*; S6-7 = (a)).** `orchestrator/verify_budget.py`
  (new, leaf module) holds the one cascade: `resolve_verify_timeout(config, module, *, is_cold,
  merge_cold)` (today's `verify.py::_resolve_verify_timeout`, moved; its merge-verify keyword is
  `merge_cold` from the start with no default — S4 β's name, lead 2026-11-10 — and α passes it
  explicitly at today's four call sites) and
  `merge_verify_command_budget_secs(config, covered)` over an explicit module list. It imports
  `OrchestratorConfig`/`ModuleConfig` under `TYPE_CHECKING` only, so `config.py` imports it at
  module level without a cycle (`verify.py` and `verify_plan.py` both import `orchestrator.config`
  at module level today). `verify.py::merge_verify_command_budget_secs` stays the worker-side
  entry: it computes the covered set with `verify_plan.effective_merge_module_configs` and calls
  the leaf. `load_config` calls the leaf over `[None, *all discovered module configs]` — a
  superset of any covered set, so the load check can only be stricter. `LaneTimingBudgets.derive`
  computes `no_progress_budget_secs = authorised + no_progress_margin_secs` (default 600),
  `no_progress_probe_secs` (the configured probe interval, default 60),
  `worktree_grace_secs = authorised + worktree_grace_margin_secs` (1800) and
  `periodic_reap_min_age_secs = 2 × authorised + reap_margin_secs` (3600).
  Orderings, each refused with an INV-2 message naming the slug and both values:
  `no_progress_budget_secs > authorised`; `no_progress_probe_secs < no_progress_budget_secs`;
  `authorised < worktree_grace_secs < periodic_reap_min_age_secs` — task 3018's
  detect-early/destroy-late ordering: `SpeculativeMergeWorker.reap_orphaned_merge_worktrees`
  destroys a leaked tree at the grace in the startup sweep and at the reap floor in the periodic
  sweep, while the resource audit reports from the grace; `periodic_reap_min_age_secs ≥ 2 ×
  authorised`; and, when `config.orchestrator_restart_min_interval_secs > 0`,
  `periodic_reap_min_age_secs <` that window (the constant's prose says "8h"; the live field is
  259200 by default and on dark-factory). `liveness.py::INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` is
  not one of these budgets: at 38a200d213 it is read only by
  `intake.py::coalesce_or_enqueue_merge_request` (the dedupe gate's alive-vs-`stale_inflight`
  classification), `liveness.py::check_merge_liveness_margin`/`enforce_merge_liveness_margin` (the
  heartbeat-floor guard) and the worker's stuck-request report (1.5×); it bounds no reaper, and α
  leaves it alone. `merge_verify_progress.*` is **red-tier**: the worker reads `self._timing`,
  built once at start, and the verify timeouts it derives from are not in
  `config.py::RELOADABLE_FIELDS` either, so a reload never changes a derived budget behind an
  `applied` report. The four class constants `LaneTimingBudgets` replaces
  (`INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS`, `INFLIGHT_VERIFY_PROGRESS_PROBE_SECS`,
  `RESOURCE_AUDIT_WORKTREE_GRACE_SECS`, `PERIODIC_REAP_MIN_AGE_SECS`) and their prose are deleted.
- **D8 — `ItemLifecycle` is the operative state; containers are indexes (R1; heuristic *clear
  invariants*).** `merge_lane/item_lifecycle.py::ItemLifecycle` (OQ-9) owns `_live_items`, the
  state map and request-future resolution. `retire(request_id, outcome) -> Retired` is
  synchronous: it transitions to TERMINAL, pops the live object and resolves the future unless the
  waiter is gone (`Dropped`), with no `await` (today's docstring guarantee that
  register/current/transition contain no await points carries over), then puts each registered
  `on_terminal` hook on an owned queue (a task set). `async drain_hooks()` awaits that queue in
  registration order; `stop()` and tests call it, and `stop()` returns only after it. The sync
  sites that take a terminal edge today stay sync and call `retire` directly — at 38a200d213 (AST;
  5040 moves `_run_inflight_verify` and `_dispatch_item`, and ζ follows their sites there):
  `_detach_parked_gate_head`, `_oob_deliver`, `_resolve_merging_requests`,
  `_resolve_or_drop_abandoned` (called from 9 async sites), the `retire_item=lambda rid: …`
  accessor `SuffixConflictTracker` receives, and `suffix_graph.py::SuffixConflictTracker._escalate`.
  `resolve_duplicate(obj, outcome)` resolves a re-entrant duplicate's own future by identity and
  never touches the registered entry (today's `MergeIntake._coalesce_reentrant_drain` case, where
  the live original shares the request_id). `requeue(...)` is the one QUEUED re-entry. A test
  asserts, by AST over `merge_lane/*.py` and `suffix_graph.py`, that no module but
  `item_lifecycle.py` calls `.set_result(` on a request future; the one named exemption is
  `types.py::InFlightMergeRegistry.attach`'s waiter mirror. `snapshot` reads `items_in(state)`;
  `_finalizing_head_entry` becomes `items_in(FINALIZING)`. `MergeIntake` keeps its buffers as
  queues to drain; their membership is checked against the registry on every `snapshot`, and a
  mismatch emits `merge_lifecycle_census_mismatch` plus a dedup'd L1 — reports, never repairs, and
  never an `assert` (stripped under `-O`).
- **D9 — an illegal transition is a typed refusal every caller branches on; one default branch
  decides what the lane does (INV-11; S6-5 = B staged).**
  `transition(request_id, to, *, expected, live_obj=None)` keeps today's from-state cross-check as
  `expected: ItemLifecycleState | None` (the twin-entry defence from tasks 2441/2852) and returns
  `Transitioned | TransitionRefused(request_id, expected, to_state, current)`. Each caller branches
  on it exhaustively (pyright-checked `match`); its refusal arm calls the one default branch,
  which returns `RefusalAction.CONTINUE` or `RefusalAction.FREEZE`, and the caller obeys. ζ lands
  the branch with today's behaviour only: it files the L1 (`_alarm_illegal_lifecycle_transition`'s
  per-request dedup and its `escalation_created` event with `request_id`, `from_state`,
  `to_state` kept byte-identical, the census's counting rule) and answers `CONTINUE` with the
  registry unchanged. φ makes it read `merge_lane.lifecycle_refusal_policy`: under `escalate` (the
  default) nothing changes; under `hold_lane` it places a `lifecycle_refused` holder (C7) on the
  request's lane — id minted before filing, held before the filing, filed through θ's
  `file_holder_escalation`, released alone if the filing fails — and answers `FREEZE`: the caller
  takes no action and the request stays in its registry state behind the hold. Storm guard: a lane
  already held for the same refused edge gets no second holder. The holder's owner is the refusal's
  escalation and its bound the L1→L2 ladder (INV-7), the bound WIP holders carry; it is rehydrated
  on restart from that pending escalation and released only by resolving it. A refusal never
  retires the request under either policy. Decision 4 of
  `plans/merge-queue-modularization-invariants-prd.md` ("Invariants escalate loudly, degrade never")
  is amended for this one invariant (φ's draft text); every other invariant check still escalates
  and continues. Why staged: all 25 refusals ever seen came in the four weeks after the registry
  was wired, and 7 of the 22 classified were bugs in its own edge table, so the hold is armed only
  after ξ's 14 quiet days.
- **D10 — (withdrawn under S6-6 = (a)).** The coalesce-train journal and unwind are not filed.
- **D11 — a lane's halt is the set of its holders (heuristic *clear invariants*; S6-8 = (b)).**
  `HaltState.hold(lane, holder)` / `release(lane, holder)`; a lane runs when its holder set is
  empty. The operator halt, the WIP halt (on every lane, because the project root is shared) and the
  auto-heal hold on `normal` are each a holder carrying its own owner escalation id and bound
  (INV-7). `halt_for_wip`/`operator_halt` add a holder and never touch another; resolving one
  escalation releases only its holder. The train advance-failure path holds FIRST and files after:
  the escalation record's id is minted before any call (as the workflow's
  `_submit_halt_owning_escalation` does today); `_map_advance_failure`'s `halt=` keyword stays a
  sync `Callable[[str], None]` (S5 η keeps its other keywords unchanged), and θ's caller passes a
  sync closure that calls `HaltState.hold(lane, HaltHolder(id=<minted id>, kind=train_advance,
  reason=...))`, so the lane is held before the next merge can be picked; `_do_train_merge` then
  awaits `HaltState.file_holder_escalation(record)` right after `_map_advance_failure` returns, and
  returns `_map_advance_failure`'s outcome unchanged whether that call answers `Filed(id)` or
  `FilingFailed(exc)`.
  Halt-owning filings use plain `queue.submit(record)`, called synchronously on the event loop, as
  `workflow.py::TaskWorkflow._submit_halt_owning_escalation` does today (the submit's notify
  callback, `harness.py::Harness._on_escalation`, sets an `asyncio.Event`, which is not
  thread-safe; off-loop filing would need S8 to make `_on_escalation` use
  `loop.call_soon_threadsafe(event.set)`, out of S6's scope) — never `escalation.dedupe::submit_or_dedupe`, which can fold the record into an
  existing pending parent so the minted id never becomes a record; `HaltState` takes that plain
  callable separately from `MainHealthAutoHeal`'s dedupe callable. If the returned id differs from
  the minted one, the holder is re-keyed to the returned id. A failed or raising submit keeps task
  1765's rule (and `gates.py::_map_advance_failure`'s task 1671 unhalt-on-raise): release that
  holder only (others untouched), emit `merge_halt_owner_filing_failed` carrying the exception
  (INV-11) and log — a holder whose id names no escalation could be released by nothing a human
  resolves (INV-7); the next advance re-detects the WIP overlap and holds again. In the lane
  `file_holder_escalation` never raises: it returns `FilingFailed(exc)`. A re-raise would reach
  `SpeculativeMergeWorker._merger_loop`'s generic `except Exception` arm, which resolves the
  request `blocked: Merger error` only if the registry still reads MERGING — dropping the outcome
  `_map_advance_failure` already built (for `pop_conflict`, `done_wip_recovery` with `merge_sha`:
  main advanced) or leaving the declared train's awaiter unresolved. With the lane released, the
  declared train's workflow takes its existing genuine-failure arm and `_mark_blocked` files the
  human-facing escalation. `async hold_with_escalation(lane, reason, record, kind)` is the same two
  steps for callers outside the merge path (holds synchronously before its first `await`, then
  files) and, unlike the lane path, re-raises the submit failure after releasing its holder — as
  `_submit_halt_owning_escalation` does today, where the merge outcome has already been delivered.
  Single-request WIP paths take the same lane path as trains: the lane mints the id, holds through
  the sync `halt=` closure, files through `file_holder_escalation` (the record built in the lane
  from the request; the workflow-only context fields — worktree, workflow state, filing claimant —
  become optional on it, as D5 does for the main-health payload), and returns the outcome with
  `MergeOutcome.halt_holder_id` set. The workflow files nothing for a WIP halt:
  `TaskWorkflow._handle_wip_conflict` and `TaskWorkflow._handle_wip_recovery` replace their
  `_submit_halt_escalation_and_wait(esc)` call with a wait on `outcome.halt_holder_id`'s
  resolution, and `_maybe_enqueue_group_merge`'s orphan probe (`is_wip_halted and
  halt_owner_esc_id is None` → `_escalate_train_halt`) is deleted because the lane now owns every
  advance-failure holder; on `FilingFailed` the outcome carries no holder id and the workflow's
  genuine-failure arm reaches `_mark_blocked`. Rehydration files nothing: the
  harness's `halt_for_wip` + `set_halt_owner` pair in `Harness._rehydrate_merge_halt` becomes
  `hold(lane, HaltHolder(id=<preserved escalation id>, kind=wip, …))`, so a restart never re-files
  the preserved L1; it lands after 3876 and 4838 (θ depends on both). 4838's halt KIND is a holder attribute. The holder
  kinds are `operator`, `wip`, `auto_heal`, `train_advance` and `lifecycle_refused` (S6-5: θ declares
  it and the optional structured `refused_edge`; φ is its only writer).
- **D12 — one `LaneLimits`, one `RetryBudget` (heuristic *orthogonal dimensions*, *minimum data
  access scopes*).** κ moves the 19 class constants α leaves (18 if S7 δ has deleted
  `MAX_POST_MERGE_VERIFY_NARROWED_RETRIES`) onto a frozen `LaneLimits` dataclass in
  `types.py` (timing fields taken from `LaneTimingBudgets`) constructed by the harness from config;
  the per-task counters become one `RetryBudget` keyed `(task_id, Counter)` with one
  `reset(task_id)` from ζ's terminal hook. `_run_post_merge_verify` (wherever 5040 placed it) takes `(ctx: VerifyContext,
  req, merge_wt, limits, budget)` (S4 C1, leaf β). The module-level pipeline functions take
  `TrainMergeHost`, a Protocol with no private attribute, and the
  `isinstance(worker, SpeculativeMergeWorker)` arms go (6162 folded in, including its item 3:
  `merge_lane/intake.py::_emit_merge_queued`'s `reason`/`position` params, and item 5,
  `_TrainMergeHost`; items 2 and 4 are already gone with 5305).
- **D13 — satellites own their state (heuristic *files make internal sense in isolation*).**
  `drift.py` and `shadow.py` take `DriftCheckState`/`ShadowCompareState` objects the worker
  constructs and holds; the `TYPE_CHECKING` imports of `SpeculativeMergeWorker` go.
- **D14 — prose stance.** Every docstring a leaf rewrites follows the Comments stance in
  `docs/code-quality.md`. Task 5047 relocates the existing prose; S6 leaves do not add to it.
- **D15 — nothing here raises a ratchet measure.** Every leaf lowers the keys it touches in
  `orchestrator/tests/merge_lane_ratchet_baseline.json` in the same commit, and no leaf adds a
  function-local import inside the package.

## Contracts

```python
# C1 — merge_lane/types.py (β1 lands; β2 makes kind mandatory for blocked/error)
class OutcomeKind(StrEnum): ...      # existing 23 members (4 non-terminal) + 8 terminal: worktree_missing,
                                     # transient_infra, needs_rebase, trivial_pass_main_red,
                                     # train_verify_failed, superseded, shutdown, error
class TrainKind(StrEnum):
    DECLARED = 'declared'; COALESCED = 'coalesced'

@dataclass                            # unchanged decorator (MergeOutcome is not frozen today)
class MergeOutcome:
    status: Literal[...]              # unchanged 12 literals
    reason: str = ''                  # byte-identical (D2)
    ...                               # existing fields unchanged (block_category, conflict_evidence, ...)
    halt_holder_id: str | None = None # θ: the holder an advance failure placed (its escalation id)
    kind: OutcomeKind | None = field(default=None, kw_only=True)   # β1 (transitional)
    # β2 narrows the static type to `kind: OutcomeKind`: a private sentinel default that
    # __post_init__ replaces with the derived member, or a `kind` property typed OutcomeKind over a
    # private optional field (β2 picks); no consumer type-checks an Optional after β2.
    def __post_init__(self) -> None:
        # derives kind from status where status alone decides (done, already_merged, conflict,
        # unknown_branch, superseded, ...); β2: raises ValueError for status in {blocked, error}
        # (and any status β1's table marks explicit) without kind; always raises ValueError when
        # kind is not kind.is_terminal or is inconsistent with status (one status↔kind table).

@dataclass
class GroupMergeRequest(MergeRequest):
    train_kind: TrainKind = field(kw_only=True)    # REQUIRED; set by both formers
# Invariants: every MergeOutcome constructed in orchestrator/src states or derives a terminal kind
# (β1 AST test over orchestrator/src; β2 enforces at runtime); `result.reason.startswith` does not
# appear in orchestrator/src/orchestrator/workflow.py; no code reads _COALESCE_TRAIN_ID_PREFIX.
# Consumers: workflow.py; S9 η/θ (`classify_merge_future` on kind; the member is `shutdown`,
# not `SHUTDOWN`); S3 via C5. unblock_types.py::classify_block_reason classifies a raw reason
# string with no MergeOutcome in hand and is S9 η's to retire.

# C2 — merge_lane/item_lifecycle.py (ζ): the operative state
@dataclass(frozen=True)
class Dropped:                        # new: the sole waiter is gone; no future to resolve
    reason: str
LiveItem = MergeRequest | SpeculativeItem | InflightEntry
class ItemLifecycle:
    def register(self, request_id: str, initial: ItemLifecycleState = ItemLifecycleState.QUEUED,
                 *, live_obj: LiveItem) -> None            # today's signature + live_obj (new keyword)
    def transition(self, request_id: str, to: ItemLifecycleState, *,
                   expected: ItemLifecycleState | None, live_obj: LiveItem | None = None
                   ) -> Transitioned | TransitionRefused
    def retire(self, request_id: str, outcome: MergeOutcome | Dropped) -> Retired      # sync
    async def drain_hooks(self) -> None       # awaits queued on_terminal hooks in registration order
    def resolve_duplicate(self, obj: MergeRequest, outcome: MergeOutcome) -> None   # identity-checked
    def requeue(self, request_id: str, *, expected: ItemLifecycleState,
                live_obj: LiveItem | None = None) -> Transitioned | TransitionRefused
    def current(self, request_id: str) -> ItemLifecycleState | None
    def items_in(self, *states: ItemLifecycleState) -> tuple[LiveItem, ...]
    def live(self, request_id: str) -> LiveItem | None
    def on_terminal(self, hook: Callable[[LiveItem, MergeOutcome | Dropped], Awaitable[None] | None]) -> None
# Invariants: retire and resolve_duplicate are the only places a request future is resolved
# (AST-guarded over merge_lane/ and suffix_graph.py; exemption InFlightMergeRegistry.attach);
# retire is synchronous (transition, pop, future resolution, hook enqueue; no await) so the sync
# terminal-edge sites call it directly; TERMINAL items are not in _live_items; a refused transition
# changes nothing and is returned, not raised; items_in is the only source snapshot() reads.
# Error semantics: retire on an unknown id returns Retired(already=True); hooks are queued on an
# owned task set and drain_hooks awaits them in registration order; a hook exception is logged with
# the request id, emits merge_lifecycle_hook_failed (INV-11) and does not prevent later hooks;
# stop() returns only after drain_hooks(). Consumers: worker.py and 5040's extracted modules,
# intake_buffers.py (IntakeLifecycle), suffix_graph.py, telemetry (5306), landing (5042), LaneQueries (C5).

# C2 (continued) — the refusal and its one default branch (ζ lands escalate; φ adds the policy)
@dataclass(frozen=True)
class TransitionRefused:
    request_id: str; expected: ItemLifecycleState | None; to_state: ItemLifecycleState
    current: ItemLifecycleState | None
class RefusalAction(StrEnum):
    CONTINUE = 'continue'             # escalate: L1 filed, the caller proceeds as today
    FREEZE = 'freeze'                 # hold_lane: holder placed, the caller takes no action
class LifecycleRefusalResponder:      # φ; constructed by the worker with the policy read once at start
    def __init__(self, *, policy: Literal['escalate', 'hold_lane'], halt: HaltState,
                 escalation_queue: EscalationQueue, event_store: EventStore | None) -> None
    def on_refused(self, refusal: TransitionRefused, req: MergeRequest) -> RefusalAction
# config.py (φ): class MergeLaneConfig(BaseModel): lifecycle_refusal_policy: Literal['escalate',
#   'hold_lane'] = 'escalate'; OrchestratorConfig.merge_lane: MergeLaneConfig (RED tier: not in
#   RELOADABLE_FIELDS); snapshot() gains the additive key 'lifecycle_refusal_policy'.
# Invariants: every refusal emits escalation_created with request_id, from_state and to_state
# (today's shape; the census counts it); under hold_lane the holder is placed before the escalation
# is filed and carries refused_edge = (current, to_state); at most one lifecycle_refused holder per
# (lane, refused_edge); a refusal never retires the request; no other invariant check calls on_refused.

# C3 — merge_lane/main_health.py (δ)
class MainHealthAutoHeal:
    def __init__(self, *, registry: MainHealthAutoHealRegistry, halt: HaltState,
                 submit: Callable[..., Awaitable[str | None]],    # escalation.dedupe::submit_or_dedupe, bound
                 spawn_fix_task: Callable[..., Awaitable[str | None]], max_attempts: int) -> None
    async def heal(self, *, brk: MainBreak, req: MergeRequest, source: Literal['sync', 'deferred'],
                   context: AutoHealContext) -> AutoHealDecision
def preexisting_break_is_escalatable(config: OrchestratorConfig, result: VerifyResult) -> bool
def is_auto_heal_eligible(category: str | None, cause_hint: str | None) -> bool
def compute_preexisting_main_break_fingerprint(category: str, cause_hint: str, probe_sha: str) -> str
@dataclass(frozen=True)
class AutoHealDecision:              # new
    kind: Literal['healing', 'ineligible', 'attempts_exhausted', 'already_in_flight',
                  'no_escalation_queue', 'fingerprint_unavailable']
    escalation_id: str | None; fix_task_id: str | None; held_lane: str | None
# Invariants: one DedupeConfig literal; the lane hold and its holder registration happen inside
# heal, never in a caller; the workflow maps kind → _mark_blocked and never touches the registry
# or HaltState. Consumers: workflow.py MAIN_HEALTH_RED arm (S9 η), the deferred probe
# (5040's main_health_probe.py, or the worker's arm if 5040 left it there).

# C4 — merge_lane/landed_outbox.py (γ1): injected journal
class LandedOutbox:
    def __init__(self, path: Path) -> None
    @classmethod
    def in_memory(cls) -> LandedOutbox
    def lookup(self, task_id: str) -> LandedRow | None
    def consume(self, task_id: str) -> None
    def record(self, row: LandedRow) -> None
# Harness: outbox = LandedOutbox(project_root / 'data/orchestrator/landed_outbox.json');
# SpeculativeMergeWorker(..., landed_outbox=outbox); build_workflow(..., landed_outbox=outbox);
# TaskGroundTruth(..., landed_outbox=outbox); the harness reads its own `outbox`, never
# self._merge_worker._landed_outbox. MergeProvenance is deleted. Consumers: workflow.py,
# task_ground_truth.py, Harness._reconcile_landed_outbox / _landed_dispatch_gate (S8), S9 ι.

# C5 — S3 C1 (S3's port task α) creates merge_lane/queries.py: the MergeLaneQueries Protocol,
#      LiveRequestState / TerminalRequestState (state: MergeState, never empty; no raw_state) and
#      QueryUnavailable. There is no adapter (S3-1 = A). μ adds, in that file:
LIVE_STATE_OF: dict[ItemLifecycleState, MergeState]   # total over the non-terminal members (S3-5)
class LaneQueries:                    # implements S3's MergeLaneQueries over C2
    def live_request_state(self, request_id: str) -> LiveRequestState | None       # from C2 items
    def live_states_for(self, *, branch: str | None = None, task_id: str | None = None) -> tuple[LiveRequestState, ...]
    def owns_merge_worktree(self, path: Path) -> bool
    def terminal_state(self, request_id: str) -> TerminalRequestState | None
        # from the journal (merge_lane/queue_store.py::MergeQueueStore) plus merge_finalized rows;
        # never TerminalOutcomeRetention (task 4833 deletes it)
# Invariants: read-only; built over C2, never over snapshot() dicts; state reaches a record only
# through LIVE_STATE_OF, and a test fails when a non-terminal ItemLifecycleState member has no row;
# raises S3's QueryUnavailable, never a silent False. S3's first merge-tool move (ε1) wires
# LaneQueries into the harness and retypes MergeCancelVerdict.state to ItemLifecycleState; both
# merge-tool moves depend on μ, and μ on S3 α only. snapshot()'s display mapping
# (worker.py::_REGISTRY_STATE_TO_WIRE) stays for get_merge_queue's human rendering.

# C6 — orchestrator/verify_budget.py and merge_lane/types.py (α): lane timing budgets
def resolve_verify_timeout(config: OrchestratorConfig, module: ModuleConfig | None, *,
                           is_cold: bool, merge_cold: bool) -> float   # moved from verify.py; no default
def merge_verify_command_budget_secs(config: OrchestratorConfig,
                                     covered: Sequence[ModuleConfig | None]) -> float
@dataclass(frozen=True)
class LaneTimingBudgets:
    authorised_command_budget_secs: float    # the max this was derived from (messages, snapshot)
    no_progress_budget_secs: float
    no_progress_probe_secs: float
    worktree_grace_secs: float
    periodic_reap_min_age_secs: float
    @classmethod
    def derive(cls, config: OrchestratorConfig, covered: Sequence[ModuleConfig | None]) -> LaneTimingBudgets
class LaneTimingConfigError(Exception): ...  # message names the slug, the violated ordering and both values
# config.py: class MergeVerifyProgressConfig(BaseModel): no_progress_budget_secs: float | None = None;
#   probe_interval_secs: float = 60.0; max_consecutive_aborts: int = 3; no_progress_margin_secs: float = 600.0;
#   worktree_grace_margin_secs: float = 1800.0; reap_margin_secs: float = 3600.0
#   OrchestratorConfig.merge_verify_progress: MergeVerifyProgressConfig  (RED tier: not in RELOADABLE_FIELDS)
# Invariants (D7): no_progress_budget_secs > authorised; no_progress_probe_secs <
# no_progress_budget_secs; authorised < worktree_grace_secs < periodic_reap_min_age_secs (detect
# early, destroy late); periodic_reap_min_age_secs >= 2 × authorised and, when
# orchestrator_restart_min_interval_secs > 0, < it. liveness.py::INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
# is not derived (it bounds no reaper). Checked in load_config over [None, *discovered modules] and
# at worker start over the module registry; both call derive.
# Consumers: worker (κ folds the fields into LaneLimits); get_merge_queue snapshot
# `timing_budgets` (additive key).

# C7 — merge_lane/halt.py (θ)
@dataclass(frozen=True)
class HaltHolder:                     # new
    id: str                           # owner escalation id, or 'operator'
    kind: HaltKind                    # operator | wip | auto_heal | train_advance | lifecycle_refused
                                      # (4838's halt kind; lifecycle_refused per S6-5, placed by φ)
    reason: str
    refused_edge: tuple[ItemLifecycleState, ItemLifecycleState] | None = None
                                      # set iff kind is lifecycle_refused: the storm guard's key
class HaltState:
    def __init__(self, lanes: tuple[str, ...], *, on_resume: Callable[[], None],
                 submit_halt_owner: Callable[[EscalationRecord], str]) -> None
        # submit_halt_owner: plain EscalationQueue.submit, synchronous (an fsync'd single-file write),
        # called on the event loop as every halt-owner submit is today — its notify callback
        # (Harness._on_escalation) sets an asyncio.Event, which is not thread-safe; never the dedupe
        # callable MainHealthAutoHeal receives (C3), which can fold the record into a parent
    def hold(self, lane: str, holder: HaltHolder) -> None
    async def hold_with_escalation(self, lane: str, reason: str, record: EscalationRecord,
                                   kind: HaltKind) -> HaltHolder
        # holds FIRST, synchronously, with HaltHolder(id=record.id, kind=kind, reason=reason) — record.id
        # is minted by the caller before the call — then calls file_holder_escalation(record); for
        # callers outside the merge path: on FilingFailed(exc) it re-raises exc after the release
    async def file_holder_escalation(self, record: EscalationRecord) -> Filed | FilingFailed
        # calls submit_halt_owner(record) synchronously (no await); Filed(id) re-keys the holder if id
        # differs from record.id; a failed or raising submit releases that holder only, emits
        # merge_halt_owner_filing_failed carrying the exception, logs, and returns FilingFailed(exc) —
        # never raises into the lane. Used directly after a sync hold() by _do_merge/_do_train_merge,
        # whose _map_advance_failure halt= closure stays sync and which return _map_advance_failure's
        # outcome unchanged either way (with halt_holder_id set)
    def release(self, lane: str, holder_id: str) -> Released | NotHeld
    def holders(self, lane: str) -> tuple[HaltHolder, ...]
    def is_lane_halted(self, lane: str) -> bool        # holders(lane) non-empty
    def lanes_held_by(self, holder_id: str) -> tuple[str, ...]
@dataclass(frozen=True)
class Filed:                          # new
    escalation_id: str
@dataclass(frozen=True)
class FilingFailed:                   # new
    exc: BaseException
# Invariants: a lane runs iff its holder set is empty; a holder is in place before its escalation is
# filed; a filing failure releases that holder only; no method removes a holder it was not asked to remove; every holder carries a machine-readable owner (INV-7); no assert in halt.py.
# Consumers: worker, MainHealthAutoHeal (C3), LifecycleRefusalResponder (C2, φ), workflow (S9),
# Harness._rehydrate_merge_halt / _on_escalation_resolved (S8), escalation halt/unhalt tools (S3).
```

Exact field names beyond those above, `AutoHealContext`'s fields and the status↔kind table's rows
are tactical and decided in the leaf that lands the contract.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1a | config (α) | yaml pins `merge_verify_progress.no_progress_budget_secs: 9000`, merge-cold 7200, warm ≤ 7200 | inline yaml | `load_config` succeeds; `LaneTimingBudgets.derive` reports authorised 7200, no-progress 9000 |
| 1b | config (α) | the same pin with `merge_verify_cold_command_timeout_secs: 10800` | `orchestrator/tests/fixtures/lane_budget_below_timeout.yaml` | `load_config` raises `LaneTimingConfigError`; message contains `lane-budgets-exceed-verify-timeouts`, `9000`, `10800` |
| 1c | config (α) | the same pin with `merge_verify_cold_command_timeout_secs: null` and `verify_cold_command_timeout_secs: 10800` | inline yaml | raises naming `9000` and `10800`: with merge-cold unset the cascade resolves `verify_cold_command_timeout_secs` as the governing merge timeout (R1's literal key) |
| 2 | config (α) | knob unset; a module yaml raises a per-module warm `verify_command_timeout_secs` above the root | registry with one such module | `load_config` and worker start derive `no_progress_budget_secs` = module value + 600; snapshot `timing_budgets` shows it; no refusal |
| 2b | config (α) | reify-shaped yaml (warm 7200, merge-cold 10800), knob unset | inline yaml | loads; `timing_budgets`: no-progress 11400, grace 12600 < reap floor 25200; `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` unchanged (10800) |
| 2c | config (α) | `reap_margin_secs` raised so the reap floor ≥ `orchestrator_restart_min_interval_secs` | inline yaml | `load_config` raises naming the reap floor and the window |
| 2d | config (α) | `reload_config` with a changed `merge_verify_progress.no_progress_margin_secs` | running harness | disposition `restart_required`; `timing_budgets` unchanged |
| 3 | workflow ← lane (β1) | every public reason prefix, as a `MergeOutcome` with its `kind` | pre-β1 routing table (test module) | `_submit_to_merge_queue` reaches the same branch as before β1 for every row; `result.reason.startswith` absent from workflow.py |
| 3b | lane (β1/β2) | `MergeOutcome('done')`, `MergeOutcome('blocked')`, `MergeOutcome('blocked', kind=cas_retry)` | after β2 | first derives `kind=done`; second raises `ValueError`; third raises (non-terminal kind) |
| 4 | lane (β1) | a coalesce train and a declared train under an admission halt | both buffered | coalesce train dissolved, declared train requeued, decided on `train_kind`; the id prefix unread |
| 4b | lane (β1) | shutdown with an owned request buffered | buffered request | journal prune decided on `kind == OutcomeKind.shutdown`; reason string unchanged |
| 5 | harness → readers (γ1/γ2) | `TaskWorkflow` and `TaskGroundTruth` built with an in-memory outbox holding a row; then built with no outbox | no worker constructed | first: `TaskWorkflow._recover_if_already_merged` and `TaskGroundTruth._resolve_branch_state` see the row; second (after γ2): `TypeError` at construction |
| 6 | workflow ↔ lane (δ) | MAIN_HEALTH_RED through the sync arm, then the same break through the deferred probe | fresh registry | one escalation (dedup), one fix task, one `auto_heal` holder on `normal`; second call returns `already_in_flight`; `_mark_blocked` reached from the decision |
| 7 | lane (ζ) | single request through merge → verify → land | real git fixture | exactly one `retire`; future resolved once; `items_in(TERMINAL)` empty; `snapshot()['entries']` never lists it after retire; keys byte-identical to today |
| 8 | lane (ζ) | sole waiter abandons mid-verify | detached future | `retire(..., Dropped)`; no `set_result`; worktree ledger matches disk |
| 8b | lane (ζ) | a re-entrant DUPLICATE submission drains while the original is VERIFYING | live original registered | `resolve_duplicate` resolves the duplicate `already_merged`; the original's registry entry and future are untouched |
| 8c | lane (ζ) | `stop()` while a terminal hook is queued, and a sync site (`_resolve_or_drop_abandoned`) retiring | hook blocked on an event | the sync site calls `retire` without awaiting; transition+pop+resolve happened at the call; `stop()` returns only after `drain_hooks()` completes the hook |
| 9 | lane (ζ; φ under `escalate`) | a caller requests an illegal transition | registry at VERIFYING, `expected=QUEUED`; policy `escalate` (the default) | `TransitionRefused` returned; registry unchanged; the L1 filed and `escalation_created` emitted with `request_id`, `from_state`, `to_state` (today's shape); no holder; the request is NOT retired and its merge continues |
| 9b | lane (φ) | the same refusal under `merge_lane.lifecycle_refusal_policy: hold_lane` | request on lane `normal`; lane `high` carrying a second request | a holder of kind `lifecycle_refused` on `normal`, id = the refusal's escalation id (minted before filing), `refused_edge` = (VERIFYING, the requested state); the caller takes no action and the request stays VERIFYING; `high` keeps landing; resolving the escalation releases that holder only; `get_merge_halt_status` lists it |
| 9c | lane (φ) | storm guard: five requests on `normal` refused on the same edge, then one on a different edge | policy `hold_lane` | one holder for the first edge, a second for the other edge, never more; each request still gets its own deduplicated L1 |
| 9d | harness (φ) | restart with a pending `merge_lifecycle_transition_rejected` escalation that owned a holder | policy `hold_lane`, holder placed before the restart | rehydration re-holds that lane under the preserved escalation id; no escalation is filed; resolving it releases the lane |
| 9e | config (φ) | `reload_config` with a changed `merge_lane.lifecycle_refusal_policy`; snapshot read | running harness | disposition `restart_required`; snapshot `lifecycle_refusal_policy` reports the value the worker started with |
| 12 | lane (θ) | auto-heal holds `normal` as esc_A; a WIP halt esc_B on the project root | normal held {esc_A} | every lane holds esc_B; releasing esc_A leaves `normal` halted by esc_B; releasing esc_B runs every lane; `get_merge_halt_status` lists holders per lane |
| 12b | harness (θ) | restart with a preserved pending wip_conflict L1 | holder persisted | rehydration calls `hold(lane, HaltHolder(id=<preserved escalation id>, kind=wip))`, per 3876's re-test; no escalation is filed (the queue's pending count is unchanged) |
| 13b | lane (θ) | a declared (coalesce-free) train's advance fails with `wip_overlap` while `submit_halt_owner` raises; auto-heal holds `normal` | normal lane held {auto-heal}; second request queued | the train's holder is in place before submit is called; when submit raises, that holder alone is released, `merge_halt_owner_filing_failed` emitted (carrying the exception), nothing re-raised; the auto-heal holder is untouched; the train's future resolves with `_map_advance_failure`'s outcome and the workflow's genuine-failure arm reaches `_mark_blocked`; the next merge's own WIP check holds again. Variant: submit returns a different id → the holder is re-keyed to it |
| 13c | lane (θ) | a declared train's advance returns `pop_conflict` (main advanced) while `submit_halt_owner` raises | train at MERGING | the awaiter receives `done_wip_recovery` with `merge_sha`, never `blocked: Merger error`; that holder alone is released; `merge_halt_owner_filing_failed` emitted |
| 15 | lane (κ) | a task exhausts `max_cas_retries` | `LaneLimits(max_cas_retries=2)` | third CAS loss resolves `cas_exhausted`; `RetryBudget.reset(task_id)` at retire; no class constant assignment in the test |
| 16 | satellites (λ) | drift check deferred twice then runs | `DriftCheckState` injected | state object records the deferrals; `worker._` absent from drift.py/shadow.py |
| 17 | mcp ← lane (μ) | `live_request_state` / `live_states_for(branch=)` for a VERIFYING request and `terminal_state` for a TERMINAL one; every `ItemLifecycleState` member | C2 registry; journal row + `merge_finalized` event | S3's typed records returned with `state: MergeState` mapped through `LIVE_STATE_OF`; `terminal_state` read from the journal and the event, no `snapshot()` dict parsed; a test fails when a non-terminal member lacks a `LIVE_STATE_OF` entry |
| 18 | census (φ) | `scripts/lifecycle_refusal_census.py --runs-db A --runs-db B --window-days 14 --now T` over fixture stores | A holds two refusal events inside the window and one outside; B none | exit 1, one `store <path>: refusals=<n>` line per store (A 2, B 0); with A's in-window events removed, exit 0; a missing store or one named twice exits non-zero naming the path; nothing is written |

Rows 10, 11 and 13 (coalesce journal, dequeue unwind, coalesce advance-failure owner; row 13b is
its declared-train successor, with 13c) and row 14 (the advance-failure mapping, now S5's B16) are
not S6's under S6-6 = (a) and OQ-13 (a).

## Design invariant (α lands the code and performs the four-site lockstep edit)

Minted here: **INV-15 `lane-budgets-exceed-verify-timeouts`** (lead ruling revised 2026-11-10: S6 α
INV-15, then S10 θ `metadata-slices-one-table` INV-16, then S7 ζ `merge-gate-unmeasured-typed` INV-17;
S10 θ depends on S6 α and S7 ζ on S10 θ on the three invariant files; it reverses the 2026-11-02 order because the
validator fixes a live misconfiguration and R1 made it a first leaf). The design-invariants doc ends
at INV-14 on main and `scripts/tests/test_design_invariants_consistency.py` requires contiguous
numbers from 1, so INV-15 waits on nothing and the lockstep edit rides in α (the former docs leaf ν
existed only to wait for INV-16 and is gone). The slug is the stable id and every check anchors on
it. α recomputes the dated paragraph and the cumulative rehearsal count (28/28 at 38a200d213; α's
2-row addendum makes it 30/30; S10 θ and S7 ζ add theirs after).

Sites (`scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`):
`docs/legibility/design-invariants.md` (section, header dating paragraph, `## Fixtures` date list);
`docs/legibility/design-invariants-fixtures.md` (`INV-15-PRD`, `INV-15-CODE`, a dated addendum
walk with its 2-row rehearsal table and the recomputed cumulative count); `skills/prd/references/gates.md`
(family-inventory row and the `inv-trigger-shapes` span); `CONTRIBUTING.md` gets nothing (pinned
as an absence). Draft normative text:

> **Rule**: Every merge-lane timing budget that bounds a verify — the in-flight no-progress budget,
> the resource-audit worktree grace and the periodic-reap minimum age — exceeds the merge-verify
> authorised budget: the maximum, over the global fallback and every registered module, of the warm
> per-command timeout (`verify_command_timeout_secs`, the per-module override) and the merge-cold one
> (`merge_verify_cold_command_timeout_secs`, else the per-module `verify_cold_command_timeout_secs`,
> else the root `verify_cold_command_timeout_secs`), computed only by
> `orchestrator.verify_budget.merge_verify_command_budget_secs`. The budgets are derived from it
> (`orchestrator.merge_lane.types.LaneTimingBudgets.derive`); the grace stays below the reap floor
> (detect early, destroy late) and the reap floor below the fleet-redeploy window; an operator pin or
> margin that breaks an ordering is refused at config load and again at merge-worker start, with a
> message naming both values.
>
> **Checkable design question(s)**: Does this change add or raise a verify timeout the merge
> cascade can resolve, or add a lane budget that watches a verify? Is the budget derived through
> `LaneTimingBudgets.derive`, or a constant that must be kept above the timeout by hand?
>
> **Evidence**: at `6551b75aba` `INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS` was 5400 while every live
> config authorised 7200–10800 per merge-verify command (unchanged at `38a200d213`); a healthy silent
> verify was torn down at 5400 s and resolved `blocked` after three strikes (`fk-4b76c9695674`, run
> `review-all-dark_factory-20261008`; the 2026-08-29 livelock `2e58a8fc28`).
>
> **House pattern**: `verify_budget.merge_verify_command_budget_secs` as the one definition;
> `LaneTimingBudgets.derive`; the `load_config` check and the worker-start check both raising
> `LaneTimingConfigError`; fixture yaml `orchestrator/tests/fixtures/lane_budget_below_timeout.yaml`
> fails `load_config` while the same pin under a lower governing timeout loads.
>
> **Family boundary**: INV-2 `structured-facts-at-failure` governs what a violation message
> carries; this invariant governs orderings between configuration values and requires them to hold
> by derivation. INV-9 `one-fact-one-home` governs world-facts in records; this governs a derived
> configuration value with one derivation.

**How this PRD reads R1's fixture** (S6-7 = (a), ruled 2026-11-10). R1 specifies "a yaml raising
`verify_cold_command_timeout_secs` above the budget fails load". Under `_resolve_verify_timeout`'s
precedence a merge verify's cold timeout is `merge_verify_cold_command_timeout_secs` first (7200 in
`defaults.yaml`), so `verify_cold_command_timeout_secs` governs a merge verify only when the merge
key is null. The PRD reads R1's fixture as "the timeout that governs a merge verify": row 1b raises
the merge-cold key, row 1c raises `verify_cold_command_timeout_secs` with the merge key null (the
literal key, where it actually governs), and row 1a shows the same pin loading under a lower
governing timeout — so a validator that ignored the governing timeout fails 1b/1c.

Trigger shape for gates.md: *a merge-lane timing budget set as a constant, or pinned below the
merge-verify authorised budget (`lane-budgets-exceed-verify-timeouts`)?*

The guard has two parts, both exercised against fixture inputs (INV-10): (1) behavioural —
boundary rows 1a–1c, 2–2d; (2) structural — an AST test that none of the names
`LaneTimingBudgets` derives (`INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS`,
`INFLIGHT_VERIFY_PROGRESS_PROBE_SECS`, `RESOURCE_AUDIT_WORKTREE_GRACE_SECS`,
`PERIODIC_REAP_MIN_AGE_SECS`) is assigned a literal anywhere under `merge_lane/`, with its negative
fixture: a seeded module containing `INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 5400.0` fails the test.
α deletes all four; the other 7 `*_SECS` class constants on `SpeculativeMergeWorker` (11 − 4) are κ's.

The S6-5 stop-the-line rule is not minted as an INV: it is an amendment to decision 4 of
`plans/merge-queue-modularization-invariants-prd.md` (φ writes it; draft text under φ), scoped to
one invariant check, as the 2026-08-17 resource-audit amendment there was.

## Cross-PRD relationship

| Other PRD / stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S5 GitOps ports, query contract and the git runner | consumes | S5 C5 (leaf η): `AdvanceOutcome.detail` (frozen union `AdvanceDetail`, S5 OQ-7 B), the `_map_advance_failure` reads, its outcome-taking signature and both worker.py callers, atomically (OQ-13 (a)); S5 α `run_git` repoint; S5 λ0 `QueuedBranch` → `shared.branch_names` (S5 OQ-2 A: S6 imports, never redefines) | S5 | S5 α → S6 α and S5 λ0 → S6 α where they share a file (lead 2026-11-10, after the fold-verifiers: S6 α lands first and takes no edge on S5). S6 β1 → S5 λ0: λ0 is second in S5's chain, and every S6 leaf other than α that edits `merge_lane/types.py`, `worker.py` or `harness.py` imports follows it; β1 heads the rest of S6's chain, so that one edge orders them all (and, through S5's chain, after S5 α). Ruled 2026-11-02 and unchanged by the sitting: S5 γ → S6 ζ (S5's constructor-ports chain, ε, ζ, η, θ, follows S6's typed facts); S6 κ → S5 θ (worker.py/gates.py, D6). `fk-59b3df6f52a3` is disposed in S5 η. Task 5042 decomposes `GitOps.advance_main` in place; S5 wires 5042 → η. |
| S4 Verify run context and runner contract | consumes | S4 C1 (leaf β) `VerifyContext`; S4 β (C1 contexts in gates.py, worker.py's infra branch), θ (pool sites) and μ0 (the `VerifyResult` import across worker.py, types.py, gates.py) edit S6's lane files. Under S4-1 the merge-lane worker is one of the boundary objects that resolves the process `VerifyHost` once; κ passes it on wherever S4 C1 carries it | S4 | κ → S4 β (a real C1 consumption). Ruled 2026-11-02, unchanged: S4 α, S4 β, S4 θ and S4 μ0 each depend on S6 ζ; S4 θ → S6 λ; S4 μ0 → S6 μ — labels unchanged and both edges standing (lead 2026-11-10). μ depends on λ (§DAG), so S4 μ0 → S6 μ also orders μ0 after S6's last `worker.py`/`shadow.py` leaf and no extra edge is requested. S6 α depends on no S4 leaf: it moves `verify.py::_resolve_verify_timeout` to `verify_budget.py::resolve_verify_timeout`, naming its keyword `merge_cold` (no default) from the start so S4 β renames nothing (lead 2026-11-10), and keeps `verify.py::merge_verify_command_budget_secs`'s name, module and signature (S4 C1); S4 ν imports the resolver from `orchestrator/verify_budget.py` (xref). Quarantine home: task 6670 (landed 2026-11-06) hands the merge worker the harness's quarantine set; 5040's branch (2026-11-04, pre-6670) extracts `HostQuarantine`; 5040's rebase and S4 θ's `HostAllocator` must agree one home (xref S4; not S6's to decide). |
| S7 Merge-lane gates, placement mechanisms and target adapters | consumes | S7 C1/C2 (leaves α1/α2) `GateMeasurement`, the worker's one `GateUnmeasuredReporter`; deletion leaves β (probe and auto-chain, S7-1 = A) and δ (reify retry narrowing, S7-3 = B); INV-17 (leaf ζ) | S7 | S6 α lands first (lead 2026-11-10, after the fold-verifiers): S7 δ → S6 α and S7 β → S6 α where they share a file (`worker.py`, `types.py`, `shadow.py`, `config.py`), S7 ε → S6 α on the resource-audit file; S6 α takes no S7 edge. S7 OQ-CQ5 for the rest: S6's remaining worker.py leaves depend on S7 β — S6 β1 → S7 β, the chain carries the rest; ι → S7 β. S7 δ heads S7's chain and β depends on δ; S7 β still depends on task 7720 (OQ-CQ7; in progress at 2026-11-10), so β1 follows δ, β and 7720. S6-2 = B: 5040 waits on no S7 task (the lead's 2026-11-02 union edge set for 5040 is void). S7 δ deletes `MAX_POST_MERGE_VERIFY_NARROWED_RETRIES`, so κ moves 18 constants if δ lands first. Under S7-1 = A and S7-3 = B about 470 lines (probe 275, auto-chain 194) plus the reify narrowing block leave the lane, not 1,450; S6 states no count that assumed the larger figure, and ι re-measures. S6 withdraws its 2026-11-02 request S7 δ → S6 ζ: S7 δ is now the reify-narrowing deletion, which should land before 5040 when it can, and ζ follows 5040; S7's α1 → … → ζ chain is ordered against S6's chain on worker.py, gates.py, types.py and config.py by the lead's per-file serial order (S9 Q-C3 option 1). Invariant order (lead, 2026-11-10): S6 α `lane-budgets-exceed-verify-timeouts` INV-15 first, then S10 θ `metadata-slices-one-table` INV-16, then S7 ζ `merge-gate-unmeasured-typed` INV-17; edges S10 θ → S6 α and S7 ζ → S10 θ (S10's and S7's to state); checks anchor on the slugs. |
| S3 Escalation server and queue boundary | consumes / produces (to) | S3 C1 (S3's port task α): `MergeLaneQueries`, `LiveRequestState`/`TerminalRequestState` (`state: MergeState`, no `raw_state`), `QueryUnavailable` in a new `merge_lane/queries.py` with no adapter (S3-1 = A); S6 μ adds the native `LaneQueries` and the total `LIVE_STATE_OF` table (S3-5); S3's first merge-tool move ε1 wires `LaneQueries` into the harness and retypes `MergeCancelVerdict.state` in `merge_lane/types.py`; S3 ε2 converts the dedupe gate onto the port; C7 holders exposed by the halt tools | S3 defines; S6 implements (μ) | ζ → S3 α (intake.py); μ → S3 α only — μ must not depend on any S3 merge-tool move (a cycle). Sibling side (S3's edges): S3 ε1 → S6 μ and S3 ε2 → S6 μ. μ keeps priority low; the scheduler's priority inheritance (`scheduler.py::Scheduler._compute_effective_priorities`: a task's effective priority is the highest of its own and its undone dependents') runs μ and S6's chain beneath it at ε1/ε2's priority once they are filed — not hand-raised (Leo, S3-1). harness.py per-file order (lead, 2026-11-10): S6 α first (never held behind S3), then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ — edges S3 β → S6 α and S6 β1 → S3 κ. intake.py: S3 α, S6 ζ, S6 κ, S3 ε2, with task 4833 ahead of S6 ζ and κ (S6 ζ → 4833, S6 κ → 4833; 4833 lands first in today's files and S3's merge_request move carries the result). Task 6827 is cancelled as covered by `LIVE_STATE_OF` (S3's lead action). |
| S9 Workflow phase spine and recovery ledger | produces (to) | C1 `MergeOutcome.kind` (S9 η/θ; member `shutdown`), C3 `MainHealthAutoHeal` (S9 η), C4 injected journal (S9 ι) | S6 | β1, γ1, δ, θ edit the workflow's consuming sites minimally; S9 η retires `unblock_types.py::classify_block_reason`'s prose sniff; S9 ι takes the journal-trusting guards (`fk-e501862ffe2c` residual). S9-5 follow-up (Leo, program level): a periodic (~2-minute) pass over the landed-merge journal (C4) with a one-hour give-up to an L1, after task 4646, listed with the harness plan's sweeps — xref only; the lead files or routes it. |
| S8 Harness composition root and slot registry | produces (to) | C4 journal construction; `LaneTimingBudgets`/`LaneLimits` construction; `MainHealthAutoHeal` wiring; φ's refusal-policy read and holder rehydration arm | S6 makes the edits at the existing `SpeculativeMergeWorker(` site | S8 ε leaves the journal territory to S6 (D14); request in the S8 xref. Under S8-8 (fail closed) the harness builds `EscalationQueue` first and its stores with no degrade; `LandedOutbox` is constructed at the merge-worker start, not in `Harness.__init__`, and S8 decides whether its composition root moves it. |
| S10 Task write contract | sibling | INV lockstep docs | lead | Revised 2026-11-10: S6 α `lane-budgets-exceed-verify-timeouts` (INV-15) lands first; S10 θ `metadata-slices-one-table` (INV-16) depends on S6 α; S7 ζ `merge-gate-unmeasured-typed` (INV-17) depends on S10 θ. Checks anchor on slugs. |
| S17 Scheduler seams and tick state | consumes | `scheduler.StaleEvidenceRejection` imported by worker.py (3 uses); `delivered_checks.run` | S17 | xref |
| S22 Dashboard readers (wave 3) | consumes | φ's census reads `runs.db` events directly, as `scripts/merge_deep_canary_predicate.py` does; S8-3's guard covers only `orchestrator/src` | S22 | xref: the shared runs.db read model (S8-3 follow-up) absorbs this reader when it lands |
| `plans/merge-lane-quality-prd.md` (tasks 5040, 5041, 5042, 5306, 5045, 5046, 5047, 5048) | the live decomposition | module ownership table; ratchet; frozen wire strings | that PRD | S6 files no second extraction. S6-2 = B: 5040 finishes first and waits on nothing of S6's; ζ, κ and δ depend on 5040 (δ because 5040's part 2 moves the deferred auto-heal arm into `main_health_probe.py::MainHealthResponder`); ι depends on 5045, 5047, 5048; 5046 cancelled → β1 with 5047 and 6416 repointed (S6-4); 6162 cancelled → κ. 5038/5039 cancelled as landed (5305). The lead orders 5040/5041 against S5 η/θ on worker.py at decompose (S5 wires 5042 → η). |
| `plans/merge-queue-modularization-invariants-prd.md` decision 4 | amended | "Invariants escalate loudly, degrade never" gains a written exception for refused lifecycle transitions (S6-5) | φ | φ writes the amendment beside the 2026-08-17 one (task 3622) |
| Task 4581 / 7396 (timing knob and its value) | subsumed | `MergeVerifyProgressConfig` knob; WARNING → validator | α | cancel both → α (S6-1); both pending, no claimant, no branch at 2026-11-10 |
| Tasks 4838, 3876 (halt hardening, rehydration) | upstream of θ | halt kind; `Harness._rehydrate_merge_halt` (3876 may move it to `merge_halt_rehydration.py`) | those tasks | θ depends on both (S6-8) |
| Task 4151 (phantom re-merge after a sole-waiter retire) | adjacent | a real lane defect the refusal alarm caught and escalate-and-continue left open for 90 days | its own | no edge; cited as the evidence for S6-5 |
| Tasks 6450, 7309, 6658 (coalesce strand, train cancel, member re-dispatch) | adjacent, coalesce path | — | their PRD | no S6 edge under S6-6 = (a); 6450 stays as-is per the 10-26 plan ruling 3; 6658 is that plan's "Stream 2", moot while coalescing is off. |
| Tasks 5446, 5777 (test seams) | upstream of κ | 0-patch-target bound; autouse → injection | those tasks | κ depends on both; both stamped with `fk-e7deb8bcfa95`. |
| Task 8164 (shared TaskGroundTruth/MergeProvenance test doubles) | subsumed | the doubles in `test_supervision_composition_integration_gate.py` | γ1 | cancel → γ1; γ1 shares all five doubles, not only the provenance pair. |
| Tasks 8527, 8528 (merge-lane policy ε, ζ; pending, high) | same-file neighbours | `worker.py` deep-chain placement; 8528 also `config.py` and `dark-factory-orchestrator.yaml` | that PRD | no semantic edge; ordered by the lead's per-file serial order at decompose (S9 Q-C3 option 1) |
| `plans/dashboard-prd-lanes-prd.md` | consumes | `snapshot()` keys frozen; `timing_budgets` and `lifecycle_refusal_policy` additive | S6 | unchanged keys asserted by row 7 and 5306's key-set test |

### Cross-stream same-file edges

Shared non-test files, from the author returns' `touches` (re-read 2026-11-02, re-checked against
the sitting's rulings and the lead's sequencing rulings 2026-11-10). R1 puts typed facts first, so S6
depends on a sibling where it consumes that sibling's contract (S5 α through S5 λ0, S7 β, S4 β, S3 α;
from β1 on) and where the lead ruled a same-file order the other way (S5 λ0 before β1; S3 κ before β1
on `harness.py`; task 4833 before ζ and κ). S6 α is first on every file it shares and depends on no
sibling: the sibling leaves on its files depend on it. On every other shared file the edits sit
in disjoint regions (α's drift.py/shadow.py edits are comment mentions, its worker.py edit is the
constant block, its verify.py edit moves one function) and the sibling leaf follows S6. "→" reads
"depends on"; sibling-side edges are requests in the xrefs.

| File | S6 leaves | Sibling leaves | Edge |
|---|---|---|---|
| `merge_lane/worker.py`, `drift.py`, `shadow.py` | α → … → λ (chain); 5040 before δ, ζ, κ | S4 α, β, θ, μ0; S5 α, λ0, γ, ε, η, θ; S7 β, δ, α1…ζ; tasks 5040, 8527, 8528 | S5 α, S5 λ0, S7 δ, S7 β, S7 ε → S6 α (α first, lead 2026-11-10); S6 β1 → S5 λ0, S7 β (S7 OQ-CQ5; the chain carries every later S6 worker.py leaf); ruled 2026-11-02: S4 α, β, θ, μ0 → S6 ζ; S4 θ → S6 λ; S4 μ0 → S6 μ (μ follows λ); S5 γ → S6 ζ; S6 κ → S4 β, S5 θ; S6 δ, ζ, κ → 5040 (S6-2 = B), and 5040 → 8528 → 8527 is a live edge, so they follow both; S7 α1…ζ by the lead's per-file order |
| `merge_lane/gates.py` | β1, θ, κ | S4 β, μ0; S5 α, γ, ζ, η; S7 α2, γ | S4 β, μ0 → S6 ζ; S4 μ0 → S6 μ; S5 γ → S6 ζ (S5 ζ, η follow through S5's chain); S6 β1 follows S5 α through S6 β1 → S5 λ0; S6 κ → S4 β, S5 θ; S7's gates.py leaves (α2, γ) by the lead's per-file order |
| `merge_lane/types.py` | α, β1, β2, δ, θ, ζ, κ | S5 λ0; S4 μ0; S7 α1; S3 α, ε1; task 4833 | S5 λ0 → S6 α and S7 δ, β → S6 α (α first, lead 2026-11-10); S6 β1 → S5 λ0 (every later S6 types.py leaf follows λ0 through the chain); S6 ζ, κ → 4833 (S6 α is not held behind 4833); S4 μ0 → S6 μ; S6 ζ → S3 α; S3 ε1 → S6 μ (ε1 retypes `MergeCancelVerdict.state`); S7 α1 by the lead's per-file order |
| `merge_lane/intake.py` | ζ, κ | S3 α, ε2; task 4833 | S6 ζ → S3 α; S6 ζ → 4833 and S6 κ → 4833 (4833's `retention=` removal lands first in today's files, lead 2026-11-10); S3 ε2 → S6 μ (after κ) |
| `merge_lane/intake_buffers.py`, `merge_lane/__init__.py` | β1, ζ / β1, δ, μ | S3 α | S6 ζ → S3 α; S6 μ → S3 α; β1 and δ also touch them, so the lead wires either S3 α → S6 δ or S6 β1 → S3 α at decompose (both acyclic) |
| `merge_lane/queries.py` | μ | S3 α (creates it), ε1, ε2 | S6 μ → S3 α; S3 ε1, ε2 → S6 μ |
| `merge_lane/halt.py` | θ | S3 (halt tools read holders) | S3's halt-tool edits follow S6 θ |
| `harness.py`, `workflow.py`, `config.py`, `defaults.yaml` | harness: α, β1, γ1, θ, φ, κ; workflow: β1, γ1, γ2, δ, θ; config/defaults: α, φ | every wave-2 stream (S3, S4, S5, S7, S8, S9, S10); 8528 (config.py) | sibling leaves follow S6 on these files: S4 via S4 α/β/θ/μ0 → S6 ζ; S5 λ0 follows S6 α and precedes the rest of S6 (S5 λ0 → S6 α; S6 β1 → S5 λ0; lead 2026-11-10); S7's config.py leaves by the lead's per-file order; S3's harness.py leaves interleave as the lead ruled on 2026-11-10 — S6 α, then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ (S3 β → S6 α; S6 β1 → S3 κ); S8, S9 and S10 leaves follow S6 θ (S9 η, θ, ι consume C1/C3/C4) |
| `verify.py` | α | S4 (16 leaves on verify.py), S5, S9 | S4's verify.py leaves follow S6 α through S4 α → S6 ζ (S4 α does not touch `_resolve_verify_timeout`); S4 ν imports `resolve_verify_timeout` from `verify_budget.py`; S5 α → S6 α (lead 2026-11-10); S9's follow S6 α by request |
| `task_ground_truth.py`, `evals/runner.py` | γ1 | S5, S8, S9 / S4, S8, S9 | sibling leaves on them follow S6 γ1 (request) |
| `suffix_graph.py`, `merge_lane/landing_evidence.py` | β1, ζ / μ | S5 α | via the chain (S6 β1 → S5 λ0, which follows S5 α) |
| `dark-factory-orchestrator.yaml` | υ | task 8528; S8 (S8-1, S8-7 yaml-line deletions) | υ is filed late (after ξ); the lead orders it at filing |
| shared docs (`OPERATIONS.md`, `review/briefing.yaml`, `skills/merge-queue/references/two-layer-model.md`) | α, φ, ι | S3, S7, S8 | docs-only; serialized by decompose order, no semantic edge |
| `merge_lane/verify_dispatch.py` and 5040's other extracted modules | δ, ζ, κ | 5040; S7 δ | δ, ζ, κ → 5040; S7 δ edits them only if 5040 lands first (S7's rule) |
| `test_merge_queue_resource_audit.py` and the resource-audit code | α | S7 ε | S7 ε → S6 α (ruled, lead 2026-11-10: α is the urgent leaf) |
| INV docs (`design-invariants.md`, `-fixtures.md`, `gates.md`) | α | S10 θ; S7 ζ | S6 α `lane-budgets-exceed-verify-timeouts` INV-15 first, then S10 θ `metadata-slices-one-table` INV-16, then S7 ζ `merge-gate-unmeasured-typed` INV-17: S10 θ → S6 α; S7 ζ → S10 θ (revised 2026-11-10; checks anchor on slugs) |
| `plans/merge-queue-modularization-invariants-prd.md` | φ | none | — |

## Decomposition plan

Overlay bands: every leaf targets ~300–1,500 changed LOC and ≤ 12 declared files except α, β2,
γ1, γ2 and κ, which cross the >15-file review trigger and are kept whole because each is one
mechanism with many call sites (the overlay's review trigger, not a split rule); β2 and γ2 are the
mechanical test migrations their src leaves split off so each src leaf lands green. φ is split off ζ
for the same reason: the refusal policy, its config value, its census and the decision-4 amendment
are one mechanism of ~550 LOC that would push ζ past 1,500. Same-file leaves are serialized by real
edges: the internal chain is α → β1 → β2 → γ1 → γ2 → δ → θ → ζ → φ → κ → λ → μ; it covers
worker.py, types.py, workflow.py, harness.py, gates.py, intake.py, config.py and the shared test
files. Task 5040 finishes first (S6-2 = B): δ, ζ and κ depend on it, and where 5040 has landed by
filing, every leaf that edits code 5040 moved declares 5040's extracted module beside or instead of
worker.py (the lead re-derives those file lists at decompose). `priority` = the highest severity of
the keys closed; `high` leaves (α, γ1, θ, ζ, κ, ι) become dependencies of trigger-chain gate 6698.
File lists for β2, γ2 and κ are in §Appendix.

**α — Lane timing budgets derived from the merge-verify budget; the load-time validator; INV-15**
(`task_kind=normal`, priority **high**, est. ~1,400 LOC; 26 files — review trigger, one mechanism)
- Files: `orchestrator/src/orchestrator/verify_budget.py` (new), `orchestrator/src/orchestrator/verify.py`,
  `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/defaults.yaml`,
  `orchestrator/src/orchestrator/merge_lane/types.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`,
  `orchestrator/src/orchestrator/merge_lane/liveness.py`, `orchestrator/src/orchestrator/merge_lane/drift.py`,
  `orchestrator/src/orchestrator/merge_lane/shadow.py` (the last three: comment mentions only),
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/test_config_merge_verify_progress.py` (new; rows 1a–2d and the AST guard),
  `orchestrator/tests/fixtures/lane_budget_below_timeout.yaml` (new; row 1b),
  `orchestrator/tests/test_merge_queue_request_liveness.py`, `orchestrator/tests/test_merge_queue_resource_audit.py`,
  `orchestrator/tests/test_merge_queue_orphan_reaper.py`, `orchestrator/tests/test_merge_queue.py`,
  `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_merge_queue_lifecycle_registry.py`,
  `orchestrator/tests/test_merge_queue_multihost_wiring.py`, `orchestrator/tests/test_merge_shadow.py`,
  `orchestrator/tests/test_merge_worktree_lifecycle_integration_gate.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json`,
  `OPERATIONS.md` (red-tier list gains `merge_verify_progress.*`),
  `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`,
  `skills/prd/references/gates.md` (the INV-15 lockstep edit; §Design invariant).
  The 9 test files carry 105 references to the four timing constants (measured at 38a200d213,
  unchanged since the run's folds). If 5040 has landed, its in-flight verify module reads the
  budgets too and joins the list.
- Keys: `fk-4b76c9695674`.
- Deps: none (lead 2026-11-10, after the fold-verifiers: α lands first on every file it shares with
  another plan; it takes no edge on S5 α, S5 λ0, S7 δ or S7 β and is never held behind S3 or task 4833).
  Subsumes 4581 and 7396 (S6-1). Sibling side (their edges): S5 α → S6 α and S5 λ0 → S6 α (`worker.py`,
  `types.py`, `harness.py`, `verify.py`); S7 δ → S6 α and S7 β → S6 α (`worker.py`, `types.py`,
  `shadow.py`, `config.py`); S7 ε → S6 α (`test_merge_queue_resource_audit.py` and the resource-audit
  code); S3 β → S6 α (`harness.py`); S10 θ (INV-16) → S6 α (the invariant files).
- Effective values after α (measured at 38a200d213; margins at their defaults), to be restated in
  the task description:

  | Value | dark-factory (authorised 7200) | reify (authorised 10800) | Today |
  |---|---|---|---|
  | no-progress budget | 7800 | 11400 | 5400 |
  | worktree grace (detect; startup reap) | 9000 | 12600 | 9000 |
  | periodic reap floor (destroy) | 18000 | 25200 | 21600 |
  | hung-verify hold (no-progress × `MAX_INFLIGHT_DEAD_VERIFY_ABORTS` = 3) | 6.5 h | 9.5 h | 4.5 h |

  dark-factory's reap floor drops by an hour; the hung-verify hold grows past the 4.5 h the
  constant's comment calls the "cost of the raise, accepted deliberately" — the price of never
  tearing down a verify the config authorised (accepted with S6-7). Grace < reap floor holds on both
  projects (9000 < 18000, 12600 < 25200). `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` (10800) is
  unchanged (OQ-14 withdrawal confirmed by the lead 2026-11-10).
- Signal: rows 1a–2d; `get_merge_queue` on a live orchestrator shows
  `timing_budgets.no_progress_budget_secs` = 7800 on dark-factory; `reload_config` on the knob
  reports it `restart_required`; `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py`
  passes with INV-15 in every pinned site.
- Deletes: the four class constants `LaneTimingBudgets` replaces and their derivation prose;
  `verify.py::_resolve_verify_timeout` (moved); the probe-vs-budget clamp's constant reads.
- G7: INV-2; INV-10 (fixture yaml + behavioural rows + AST guard with a seeded negative; the INV's
  own fixture); INV-11 (the worker-start check raises outside the harness's non-fatal wrapper);
  INV-12 (no suppression).

**β1 — `MergeOutcome.kind` and `GroupMergeRequest.train_kind`; consumers route on kind**
(`task_kind=normal`, priority medium, est. ~900 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/types.py`, `orchestrator/src/orchestrator/merge_lane/gates.py`,
  `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/merge_lane/intake_buffers.py`,
  `orchestrator/src/orchestrator/merge_lane/__init__.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/harness.py` (`Harness._on_stranded_merge_done`),
  `orchestrator/src/orchestrator/suffix_graph.py` (`SuffixConflictTracker._escalate`),
  `scripts/merge_lane_throughput.py` (the `_NON_TERMINAL_OUTCOMES` mirror),
  `orchestrator/tests/test_outcome_kind.py` (prefix→kind equivalence table, status↔kind table, AST
  test that every src `MergeOutcome(` states or derives a kind), `orchestrator/tests/test_merge_queue_frozen_prefix.py`,
  `orchestrator/tests/test_workflow_merge_blocked_surfacing.py`, `orchestrator/tests/test_merge_queue_coalesce.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`; plus 5040's `post_merge_verify.py` /
  `verify_dispatch.py` if 5040 has landed (the verify-failure producers move there).
- Keys: `fk-e61eed486dbc`, `fk-fd91b14b3ead`, `fk-80a790e6f793`.
- Deps: α, S3 κ (`harness.py`, lead 2026-11-10: S6 α, then S3 β, δ, κ, then S6 β1 … μ), S5 λ0
  (`types.py`, `worker.py`, `harness.py` imports; λ0 follows S5 α in S5's chain; lead 2026-11-10),
  S7 β (`worker.py`, `gates.py`, `types.py`; S7 OQ-CQ5 — S6's remaining worker.py leaves follow S7 β
  through the chain). Cancels 5046 → β1
  (children repointed; S6-4). Produces C1 for S9 and S3.
- Signal: rows 3, 4, 4b; `git grep -E 'result\.reason\.startswith' orchestrator/src/orchestrator/workflow.py`
  is empty; `git grep -E 'startswith\(_COALESCE_TRAIN_ID_PREFIX\)' orchestrator/src` is empty; a
  `merge_finalized` event on a live lane carries `kind`.
- Deletes: 8 workflow prefix routes, 3 train-id routes, 1 shutdown-reason equality, the
  2 function-local prefix imports in workflow.py, the `_derail_on_train_verify_failure` re-tag (it
  sets `kind=train_verify_failed` and keeps the prefix in `reason` for humans).
- G7: INV-2; INV-5 (one enum, one status↔kind table).

**β2 — Test constructions on one builder; `kind` mandatory for blocked/error**
(`task_kind=normal`, priority medium, est. ~500 LOC; 54 files — review trigger, mechanical)
- Files: `orchestrator/src/orchestrator/merge_lane/types.py` (`__post_init__` raises),
  `orchestrator/tests/_merge_lane_fakes.py` (`blocked_outcome(kind, reason, **fields)` builder),
  `orchestrator/tests/merge_lane_ratchet_baseline.json`, and the 51 files that construct a
  `blocked`/`error` `MergeOutcome` (§Appendix; ~154 constructions, measured at 38a200d213 by
  `MergeOutcome(` followed within two lines by a `blocked`/`error` status literal).
- Keys: none of its own (completes β1's contract).
- Deps: β1.
- Signal: row 3b; the suite passes with the `ValueError` arm live.
- G7: INV-11 (an unclassified failure outcome is refused at construction, not routed as `error`).

**γ1 — `LandedOutbox` injected by the harness; `MergeProvenance` deleted**
(`task_kind=normal`, priority **high**, est. ~800 LOC; 28 files — review trigger, one mechanism)
- Files: `orchestrator/src/orchestrator/merge_lane/landed_outbox.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/task_ground_truth.py`, `orchestrator/src/orchestrator/evals/runner.py`,
  `orchestrator/tests/conftest.py`, `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/_merge_lane_fakes.py`,
  `orchestrator/tests/test_harness_ground_truth_wiring.py`, `orchestrator/tests/test_harness_train_callbacks.py`,
  `orchestrator/tests/test_landed_outbox.py`, `orchestrator/tests/test_reconcile_stranded.py`,
  `orchestrator/tests/test_recovery_emission_wiring.py`, `orchestrator/tests/test_stranded_verified_green.py`,
  `orchestrator/tests/test_supervision_composition_integration_gate.py`, `orchestrator/tests/test_task_ground_truth.py`,
  `orchestrator/tests/test_workflow_e2e.py`, `orchestrator/tests/test_workflow_helpers.py`,
  `orchestrator/tests/test_workflow_merge_provenance.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`,
  `orchestrator/tests/test_workflow_train_attribution.py`, `orchestrator/tests/test_workflow_train_completion.py`,
  `orchestrator/tests/test_background_service.py` (8164's third file), `orchestrator/tests/test_merge_queue_landed_reconciler.py`,
  `orchestrator/tests/test_harness_landed_dispatch_gate_wiring.py`, `orchestrator/tests/test_harness_reconcile_landed_outbox_none_guard.py`
  (with `test_landed_outbox.py`, the three that reach the worker's `_landed_outbox`),
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  (The 16 `.py` files naming `MergeProvenance`, the 12 defining its fixtures, and 8164's files,
  measured at 38a200d213; the 3 authorized-raises records under `merge_lane_ratchet_raises/` are
  history and untouched.)
- Keys: `fk-2c9b6486d554`, `fk-e501862ffe2c`.
- Deps: β2. Cancels 8164 → γ1: γ1 shares all five doubles 8164 names (`_fake_git_ops`,
  `_fake_scheduler`, `_make_ground_truth`, `_bind_landed_row`, `_reset_merge_provenance` → deleted).
  Produces C4 (S6-3 = (a): γ1 makes the one construction edit at the existing site).
- Signal: row 5 (first half); `git grep MergeProvenance -- 'orchestrator/src' 'orchestrator/tests/*.py' escalation/`
  is empty; after a restart the startup `reconcile_landed_outbox` report names the harness-held
  journal; the dashboard's PRD-lane delivered mark for a landed task is unchanged.
- Deletes: `MergeProvenance` (class, `bind`, `_outbox`), `_isolated_merge_provenance`,
  `_reset_merge_provenance`, the 4 `self._merge_worker._landed_outbox` reads, the bind/`_outbox`
  test references.
- G7: INV-13 (the producer is `_journal_landed_then_advance`, unchanged); INV-11; INV-3.

**γ2 — Test constructions of `TaskWorkflow` through the helper; `landed_outbox` required**
(`task_kind=normal`, priority medium, est. ~400 LOC; 95 files — review trigger, mechanical)
- Files: `orchestrator/src/orchestrator/workflow.py` (default removed), and the 95 test files with
  a direct `TaskWorkflow(` (136 constructions, measured at 38a200d213, unchanged; §Appendix).
- Keys: none of its own (completes γ1's contract).
- Deps: γ1.
- Signal: row 5 (second half).
- G7: INV-11 (no silent in-memory journal outside tests).

**δ — One `MainHealthAutoHeal` collaborator, grown from 5040's responder; fingerprint and eligibility move into the lane**
(`task_kind=normal`, priority medium, est. ~900 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/main_health.py` (new), `orchestrator/src/orchestrator/merge_lane/main_health_probe.py`
  (5040's; its responder moves out, its classifier and probe spawner stay),
  `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/merge_lane/types.py` (`AutoHealDecision`),
  `orchestrator/src/orchestrator/merge_lane/__init__.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/main_health_actions.py`, `orchestrator/tests/test_merge_queue_auto_heal.py`,
  `orchestrator/tests/test_merge_queue_main_health_deferred.py`, `orchestrator/tests/test_workflow_main_health_routing.py`,
  `orchestrator/tests/test_merge_lane_main_health_probe.py` (5040's), `orchestrator/tests/test_merge_lane_main_health.py` (new),
  `orchestrator/tests/merge_lane_ratchet_baseline.json`. If 5040 lands without extracting the
  responder, δ edits the worker's `_auto_heal_main_health_deferred` instead.
- Keys: `fk-7f2bce2d482c`, `fk-75fc1b96098d`.
- Deps: γ2, 5040 (5040's part 2, RED on its branch at `e1d8911c7a`, moves the deferred auto-heal
  arm δ rewrites; S6-2 = B). Produces C3 for S9.
- Signal: row 6; `git grep 'from orchestrator.workflow import' orchestrator/src/orchestrator/merge_lane`
  is empty; a MAIN_HEALTH_RED on a live lane files one escalation and one fix task whichever arm
  sees it first; `get_merge_halt_status` shows the auto-heal holder on `normal`.
- Deletes: workflow `_auto_heal_main_health`, `_spawn_main_health_fix_task` (143 lines); the lane's
  second copy wherever 5040 left it (worker `_auto_heal_main_health_deferred`,
  `_spawn_main_health_fix_task`, `_post_submit_tasks`, `_file_main_health_escalation`, 228 lines at
  38a200d213, or their moved form in `MainHealthResponder`); 3 of 4 predicate copies;
  `_main_health_fingerprint`'s lazy import and fail-safe `''`. Workflow's `_post_submit_tasks` stays.
- G7: INV-5; INV-4 (the dedup window and `MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS` stay the storm escape);
  INV-11 (an empty fingerprint is `fingerprint_unavailable`, not a silent dedup-off).

**θ — A lane's halt is its holder set; train advance failures hold with an owner**
(`task_kind=normal`, priority **high**, est. ~750 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/halt.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`,
  `orchestrator/src/orchestrator/merge_lane/gates.py`, `orchestrator/src/orchestrator/merge_lane/types.py`
  (`MergeOutcome.halt_holder_id`), `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/harness.py` (`_rehydrate_merge_halt`, or the
  `merge_halt_rehydration.py` 3876 extracts it into; `_on_escalation_resolved`; the
  `HaltState(..., submit_halt_owner=queue.submit)` wiring),
  `orchestrator/tests/test_merge_lane_halt.py`, `orchestrator/tests/test_halt_owner.py`,
  `orchestrator/tests/test_workflow_halt_owner.py`, `orchestrator/tests/test_workflow_train_halt_owner.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-83625b943aa5`, `fk-ba531799522b`.
- Deps: δ, 4838, 3876 (S6-8 = (b)). Produces C7 (S9, S8, S3 consume; φ places the
  `lifecycle_refused` kind θ declares).
- Signal: rows 12, 12b, 13b, 13c; on a live lane `get_merge_halt_status` lists holders per lane; a WIP halt
  never clears the auto-heal holder on `normal`; on reify (declared trains live) a train advance
  failure holds with its own owner.
- Deletes: the three bare `assert`s, the every-lane owner wipes in `halt_for_wip`/`operator_halt`,
  `set_lane_halt_owner`/`set_halt_owner`, the harness's `halt_for_wip` + `set_halt_owner` pair,
  workflow's `_submit_halt_owning_escalation` (moved into the lane), the orphan owner probe in
  `TaskWorkflow._maybe_enqueue_group_merge` (`is_wip_halted and halt_owner_esc_id is None`) and
  `TaskWorkflow._escalate_train_halt` (obsolete once the lane owns train holders), and
  `TaskWorkflow._submit_halt_escalation_and_wait` — its callers `_handle_wip_conflict` and
  `_handle_wip_recovery` are rewritten to wait on `outcome.halt_holder_id` (D11).
- G7: INV-7 (every holder has a machine-readable owner and bound); INV-2; INV-11 (no `assert`).

**ζ — `ItemLifecycle` operative; one `retire`; containers derived; refusals typed**
(`task_kind=normal`, priority **high**, est. ~1,400 LOC; ~14 files)
- Files: `orchestrator/src/orchestrator/merge_lane/item_lifecycle.py` (new; OQ-9), `orchestrator/src/orchestrator/merge_lane/worker.py`,
  `orchestrator/src/orchestrator/merge_lane/types.py`, `orchestrator/src/orchestrator/merge_lane/intake_buffers.py`,
  `orchestrator/src/orchestrator/merge_lane/intake.py`, `orchestrator/src/orchestrator/suffix_graph.py`,
  `orchestrator/src/orchestrator/merge_lane/verify_dispatch.py` and the 5040 module that holds
  `_run_inflight_verify`'s terminal edges (its declared files name `inflight_verify.py`),
  `orchestrator/tests/test_item_lifecycle.py`, `orchestrator/tests/test_merge_queue_lifecycle_registry.py`,
  `orchestrator/tests/test_merge_queue_finalize_head_visibility.py`, `orchestrator/tests/test_merge_lane_lifecycle_boundary.py` (new; rows 7–9 and the AST guard),
  `orchestrator/tests/test_merge_queue_redispatch_entries.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-313f404f3cc6`, `fk-a2984bf98f85`.
- Deps: θ, 5040 (S6-2 = B: ζ builds on 5040's extracted modules), S3 α (intake.py), task 4833
  (intake.py; it lands first in today's files, lead 2026-11-10). Produces C2
  (consumed by φ, 5042, 5306, κ, μ).
- Signal: rows 7–9; `get_merge_queue` on a live lane lists every in-flight request once and no
  retired one across a stop/start; the AST guard passes and fails on a seeded fixture module; the
  refusal alarm still emits `escalation_created` with `request_id`, `from_state`, `to_state`.
- Deletes: the 21 raw `set_result` sites, `_retire_item`/`_resolve_or_drop_abandoned`/`_note_requeue`
  (3 defs, 32 calls), `_note_transition`'s catch-and-continue (replaced by the typed refusal and the
  one default branch), `_finalizing_head_entry`'s type-plus-absence derivation, `snapshot`'s
  container census and `_first_copy_per_request`.
- G7: INV-10; INV-11 (refusal typed and branched on); INV-6.

**φ — The lifecycle-refusal policy: `escalate` | `hold_lane`, the hold arm, the census and the decision-4 amendment** (S6-5 = B staged)
(`task_kind=normal`, priority medium, est. ~550 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/item_lifecycle.py` (the default refusal branch),
  `orchestrator/src/orchestrator/merge_lane/worker.py` (responder wiring; snapshot key
  `lifecycle_refusal_policy`), `orchestrator/src/orchestrator/config.py` (`MergeLaneConfig`, red tier),
  `orchestrator/src/orchestrator/defaults.yaml` (`merge_lane.lifecycle_refusal_policy: escalate`),
  `orchestrator/src/orchestrator/harness.py` (rehydration arm for a refusal holder, or 3876's module),
  `OPERATIONS.md` (red-tier list; what a `lifecycle_refused` holder means and how it is released),
  `plans/merge-queue-modularization-invariants-prd.md` (decision 4 amendment, below),
  `scripts/lifecycle_refusal_census.py` (new), `scripts/tests/test_lifecycle_refusal_census.py` (new; row 18),
  `orchestrator/tests/test_merge_lane_lifecycle_refusal_policy.py` (new; rows 9b–9e),
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: none of its own (Leo's ruling S6-5; amends decision 4).
- Deps: ζ (the typed refusal and its one default branch), θ through ζ (the `lifecycle_refused`
  holder kind).
- Work: (1) `MergeLaneConfig.lifecycle_refusal_policy: Literal['escalate', 'hold_lane'] = 'escalate'`,
  not in `RELOADABLE_FIELDS`; the worker reads it once at start. (2) The default branch under
  `escalate` is ζ's (L1, continue). Under `hold_lane` it mints the refusal escalation's id, holds the
  request's lane with `HaltHolder(id=<minted id>, kind=lifecycle_refused, reason=…,
  refused_edge=(current, to))` before filing, files through θ's `file_holder_escalation` (a failed
  filing releases that holder only), and returns `RefusalAction.FREEZE`: the caller takes no action
  and the request stays in its registry state. (3) Storm guard: no second holder on a lane for an
  edge already held there. (4) Rehydration: under `hold_lane`, a pending
  `merge_lifecycle_transition_rejected` escalation that owned a holder re-holds its lane on restart
  under its preserved id and files nothing, as a WIP holder does (the ruling's bound is "the same
  bound WIP halts carry"); only resolving it releases the lane. (5) `scripts/lifecycle_refusal_census.py`
  in the shape of `scripts/merge_deep_canary_predicate.py`: `--runs-db PATH` (repeatable; a store
  named twice is refused), `--window-days N`, `--now EPOCH`; opens each store read-only (`mode=ro`);
  counts `escalation_created` events whose `data` carries both `from_state` and `to_state` inside the
  window; prints one `store <path>: refusals=<n>` line per store; exits 0 iff the pooled count is 0,
  1 otherwise, 2 on an unreadable store. (6) The amendment, appended to decision 4:

  > **Amendment (S6 ruling S6-5, 2026-11-10) — a refused lifecycle transition stops its lane.**
  > Once `ItemLifecycle` is the operative state (S6 ζ), a caller that continues past a refused
  > transition acts on a state the lane's own model forbids, and escalate-and-continue left a real
  > defect (task 4151) open for 90 days. So this one invariant is the exception: under
  > `merge_lane.lifecycle_refusal_policy: hold_lane` a refusal places a per-lane holder of kind
  > `lifecycle_refused`, owned by the refusal's escalation, and leaves the request where it is;
  > resolving that escalation releases the lane, and nothing else does. The default stays `escalate`
  > (file the L1 and continue) until the operative registry has run 14 days with zero refusals on
  > dark-factory and reify; a dated deterministic gate then flips it per project. Every other
  > invariant still escalates and continues; world conditions keep the existing halt machinery.
- Signal: rows 9–9e, 18; on dark-factory `get_merge_queue` reports `lifecycle_refusal_policy:
  escalate`; `scripts/lifecycle_refusal_census.py --runs-db data/orchestrator/runs.db --window-days 90`
  exits 0 on dark-factory today, and with `--window-days 365` reports the 2 historical refusals.
- G7: INV-7 (the holder's owner is the escalation id, its bound the L1→L2 ladder); INV-11 (a failed
  filing is an event, never a silent ownerless hold); INV-10 (rows 9b–9d drive the real branch);
  INV-4 (the storm guard).

**κ — `LaneLimits` and `RetryBudget`; pipeline functions take a narrow host; test pins move onto `LaneLimits`**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC; 25 files — review trigger, one mechanism)
- Files: `orchestrator/src/orchestrator/merge_lane/types.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`,
  `orchestrator/src/orchestrator/merge_lane/verify_dispatch.py` and 5040's `post_merge_verify.py`
  (where `_run_post_merge_verify` lives after 5040), `orchestrator/src/orchestrator/merge_lane/gates.py`,
  `orchestrator/src/orchestrator/merge_lane/intake.py` (`_emit_merge_queued` params, 6162 item 3),
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/tests/_merge_lane_fakes.py`,
  the 16 test files that pin the 19 constants α leaves (§Appendix), `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-769c3bb58ce1`, `fk-201d70aaf409`, `fk-e7deb8bcfa95`.
- Deps: φ (same files after ζ), 5040 (S6-2 = B), 5446, 5777, S4 β (C1), S5 θ (worker.py/gates.py
  after S5 γ…θ), task 4833 (`intake.py`, `types.py`, `gates.py`, `worker.py`: 4833's `retention=`
  removal lands first in today's files, lead 2026-11-10). Cancels 6162 → κ.
- Signal: row 15; `orchestrator/tests/test_merge_lane_ratchet.py` reports 0 patch targets and 0
  private reads into the lane; `git grep -E '^    [A-Z_]+: (int|float) = ' orchestrator/src/orchestrator/merge_lane/worker.py`
  is empty; `_run_post_merge_verify` has ≤ 6 parameters (29 at 38a200d213).
- Deletes: the 19 remaining class constants (18 if S7 δ has deleted
  `MAX_POST_MERGE_VERIFY_NARROWED_RETRIES`), ~10 per-task dicts and their bespoke resets, 104
  test-side constant pins (re-measured after α at implementation), 35 `worker._` reaches, 3
  `isinstance(worker, SpeculativeMergeWorker)` arms, `_TrainMergeHost`.
- G7: INV-5; INV-12 (no `type: ignore` added; the authorized-raises files untouched).

**λ — Drift and shadow own their state**
(`task_kind=normal`, priority medium, est. ~350 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/drift.py`, `orchestrator/src/orchestrator/merge_lane/shadow.py`,
  `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/tests/test_merge_drift.py`,
  `orchestrator/tests/test_merge_shadow.py`, `orchestrator/tests/test_merge_queue_warm_cold_shadow.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-a0f6af37f0d6`.
- Deps: κ.
- Signal: row 16; `git grep 'worker\._' orchestrator/src/orchestrator/merge_lane/drift.py
  orchestrator/src/orchestrator/merge_lane/shadow.py` is empty; neither file imports `worker`.
- Deletes: 22 private reads/writes (line rule, measured at 38a200d213), 2 `TYPE_CHECKING` worker imports.
- G7: heuristic *files make internal sense in isolation* by construction; INV-11 unchanged.

**μ — `MergeLaneQueries` built natively over the lifecycle registry; the `LIVE_STATE_OF` table** (S3-1 = A, S3-5)
(`task_kind=normal`, priority low — inherited, see below; est. ~400 LOC)
- Files: `orchestrator/src/orchestrator/merge_lane/queries.py` (S3 α creates it; μ adds `LaneQueries`
  and `LIVE_STATE_OF`), `orchestrator/src/orchestrator/merge_lane/__init__.py`,
  `orchestrator/src/orchestrator/merge_lane/landing_evidence.py` (public verdict API named in the
  façade), `orchestrator/tests/test_merge_lane_queries.py` (S3 α creates it; μ adds the native and
  exhaustiveness cases), `orchestrator/tests/merge_lane_ratchet_baseline.json`. No harness or
  `mcp/merge_tools.py` edit: S3's first merge-tool move wires `LaneQueries` into the harness.
- Keys: `fk-d9e6c03dc259`.
- Deps: λ (the chain κ → λ → μ: S4 μ0 → S6 μ then follows S6's last `worker.py`/`shadow.py` leaf, and S3
  ε1's `worker.py` producer edit follows it too), S3 α. Never any S3 merge-tool move: S3 ε1 and ε2 depend on μ (a reverse
  edge would close a cycle). Priority: filed low; the scheduler's priority inheritance
  (`scheduler.py::Scheduler._compute_effective_priorities`) runs it, and S6's chain beneath it, at the
  priority of S3's merge-tool moves once they are filed. Not hand-raised (Leo, S3-1).
- Signal: row 17; S3's merge tools (after ε1/ε2) answer `merge_status` and `get_merge_queue` reads
  through `LaneQueries`; a lifecycle member added without a `LIVE_STATE_OF` row fails the
  exhaustiveness test; `git grep 'from orchestrator.merge_lane' escalation/src` is empty once S3's
  moves land.
- Deletes: nothing of S3's (there is no adapter); `worker.py::_REGISTRY_STATE_TO_WIRE` stays as the
  human display mapping `get_merge_queue` renders (S3-5: a display, not a contract).
- G7: INV-1 (typed Protocol, pyright on both sides); INV-5 (one total table, test-checked); INV-13
  (`terminal_state` reads its producer, the journal and `merge_finalized`).

**ι — S6 lane integration gate (B+H)**
(`task_kind=normal`, priority **high**, est. ~650 LOC)
- Files: `orchestrator/tests/test_merge_lane_s6_integration_gate.py` (new; rows 1a–9e, 12, 12b, 13b,
  13c and 15–17 through the façade with real git fixtures, row 18 over fixture stores), `orchestrator/tests/test_merge_lane_ratchet.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`, `skills/merge-queue/references/two-layer-model.md`
  (symbol map), `review/briefing.yaml` (the "Silent merge-queue halt" concern's pointer).
- Keys: `fk-1432a52e4753`, `fk-153195971a14`, `fk-e521254a5b71`, `fk-e768d50f8363`.
- Deps: θ, φ, κ, λ, μ, 5045 (ο residual worker), 5047 (ρ prose), 5048 (σ, the merge-lane-quality
  gate that owns the ceilings), S7 β (α reached through the chain).
- Signal: the gate module passes; it asserts S6's boundary rows and re-runs
  `python scripts/merge_lane_metrics.py --report`, citing 5048's ceilings rather than restating
  them (INV-5). `merge_lane/queue_store.py::recover_pending_merges` measures cognitive 44 (`scripts/merge_lane_metrics.py --json` at
  d40ba048e9; `queue_store.py` is unchanged at 38a200d213) and no S6 leaf lowers it; it is 5048's
  ceiling to own.
- Deletes: nothing of its own; it records the h13/h14 measurement in its commit message.
- G7: INV-10 (every assertion executes a measurement).

**ξ — Lifecycle-refusal quiet period: 14 days, dark-factory and reify** (S6-5; filed by the lead once φ is on main)
(`task_kind=deterministic`, priority medium, no worktree)
- `metadata.milestone`: `{mode: 'delayed', after_secs: 1209600}` (fires 14 days after its
  dependencies are done). `metadata.before_done`: `{kind: 'predicate', script:
  'scripts/lifecycle_refusal_census.py', args: ['--runs-db',
  '/home/leo/src/dark-factory/data/orchestrator/runs.db', '--runs-db',
  '/home/leo/src/reify/data/orchestrator/runs.db', '--window-days', '14'], timeout_secs: 120}`.
  `always_escalates: false`.
- Deps: φ (and ζ through it).
- Outcome: exit 0 (zero refusals on both projects in the last 14 days, all of them under the
  operative registry) → `done` (`done_provenance.kind='deterministic-milestone'`), which releases υ.
  Any refusal → born-at-L2 `milestone_check_failed` to Leo carrying the per-store counts; the
  policy stays `escalate` until he resolves it (a resume re-runs the predicate) or abandons the gate.
- Keys: none. Not a gate-6698 dependency.

**υ — Arm `hold_lane` on dark-factory** (S6-5; filed by the lead with ξ)
(`task_kind=normal`, `complexity='simple'`, priority medium, est. ~10 LOC)
- Files: `dark-factory-orchestrator.yaml` (`merge_lane.lifecycle_refusal_policy: hold_lane`).
- Deps: ξ.
- Signal: after the next fleet redeploy (restart-only value), `get_merge_queue` on dark-factory
  reports `lifecycle_refusal_policy: hold_lane`. Reify's flip is the same one-line yaml edit in
  reify's `dark-factory-orchestrator.yaml`, made by the lead once ξ is done (yaml-only commit, best
  effort, S8-7's precedent): reify's scheduler is halted, so a reify-store task would not dispatch.
- Keys: none.

DAG: α → β1 → β2 → γ1 → γ2 → δ → θ → ζ → φ → κ → λ → μ; μ also after S3 α; δ, ζ, κ also after 5040;
ι after {θ, φ, κ, λ, μ, 5045, 5047, 5048, S7 β}; ξ after φ (filed by the lead once φ is on main);
υ after ξ. Cross-stream prerequisites of S6: α after none; β1 after {S3 κ, S5 λ0, S7 β}; θ after
{4838, 3876}; ζ after {S3 α, 4833}; κ after {5446, 5777, S4 β, S5 θ, 4833}. Sibling-side edges: ruled 2026-11-02 — S4 α, S4 β, S4 θ,
S4 μ0 and S5 γ each → S6 ζ; S4 θ → S6 λ; S4 μ0 → S6 μ; ruled 2026-11-10 — S10 θ → S6 α (INV
order); S3 ε1, S3 ε2 → S6 μ (S3-1 = A); ruled 2026-11-10 (lead sequencing) — S3 β → S6 α (`harness.py`);
ruled 2026-11-10 (lead, after the fold-verifiers) — S5 α, S5 λ0, S7 δ, S7 β and S7 ε each → S6 α.
Withdrawn or declined: S4 μ0 → S6 λ, S7 δ → S6 ζ; S6 α takes no edge on any sibling task. 5040 waits on
nothing of S6's or S7's (its live dependency is 8528). Acyclic: walked with every cross-stream and
out-of-batch edge above (`critique/S6.foldverify.md`; re-walked at alignment, `critique/S6.align.md`).
Signal leaf: ι.

## Live tasks (contract §8) and lead actions

Step 1 (`find_tasks_by_metadata(key="x_finding_key", value=<key>)` on all 25 keys, re-run
2026-11-10): no task carries any S6 key. Step 2 (`search_tasks` ≥ 0.6 for each leaf's remedy,
including φ's policy and census and ξ's gate, confirmed with `get_task`, re-run 2026-11-10): no new
task carries S6 work; the tasks below carry or neighbour it. Statuses re-read 2026-11-10 with
`get_statuses`/`get_task`; none changed since 2026-11-02 (5040 is still `pending`, now with a branch).

- **5040** (pending, high; deps 5305 done and 8528 pending — 8528 → 8527 → 7720 in progress, 7721, 7824,
  8526; no claimant; branch `task/5040` with 5 commits to
  `e1d8911c7a`, plan authored 2026-11-04): finishes first (S6-2 = B). Nothing parks it and no S6 or
  S7 edge is added to it; the 2026-11-02 park-by-placeholder action and the union edge set are
  void. δ, ζ and κ depend on it. Its declared files (90, including `workflow.py` and `conftest.py`)
  overlap most S6 leaves; the lead's per-file order keeps S6's typed-facts leaves from rebasing
  it mid-flight where it can.
- **4581** (pending, medium; deps [4579, 5036]; no claimant, no branch) and **7396** (pending, low;
  depends on 4581): subsumed by α; cancel both with `x_acceptance_reason` naming α (S6-1). If 4581
  is claimed with a live heartbeat at decompose, α depends on it instead (deliberation.md gotcha (a)).
- **5046** (pending, medium; `MergeFailureKind`; deps [5045]): subsumed by β1 (S6-4); before the
  cancel, its live children are repointed (cancel-with-children ruling B, 2026-11-01): **5047**
  (pending; deps [5046] only) → 5045 and λ; **6416** (pending; deps [5046, 5293, 6374]) → β2 and
  S5 η. Cancel records `children: {5047: repoint, 6416: repoint}`.
- **6162** (pending, low; branch empty): subsumed by κ (items 3 and 5 named in κ; items 2 and 4 gone
  with 5305); no dependents.
- **5038**, **5039** (deferred; coalesced into 5305): cancel as landed (`f7f9957bb5`); 5039 is
  5038's only dependent and is cancelled with it.
- **8164** (pending, low; no dependents): subsumed by γ1.
- **4838**, **3876** (pending, high; 3876's branch empty): θ depends on both (S6-8).
- **5045**: stamped `x_finding_key += [fk-1432a52e4753, fk-153195971a14, fk-e521254a5b71]`.
  **5047**: stamped `fk-e768d50f8363`. **5048**: ι depends on it.
- **5446**, **5777** (pending, test seams): κ depends on both; both stamped `fk-e7deb8bcfa95`.
- **4833** (pending, low; deps 4830, 4831, 5036 all done, so dispatchable; deletes `TerminalOutcomeRetention`
  and the `retention=` threading, today in `merge_lane/intake.py`, `types.py`, `gates.py`, `worker.py`,
  `queue_store.py` and `escalation/server.py`): lands first in today's files (lead 2026-11-10) —
  `add_dependency(ζ, 4833)` and `add_dependency(κ, 4833)`; S6 α is not held behind 4833 and takes no
  edge on it; S3's merge_request move carries the result (S3's edge; S3 amends 4833's stale
  `merge_queue.py`/`merge_types.py` file list).
- **4151** (pending since 2026-08-12): no edge; the evidence for S6-5.
- **6450**, **7309**, **6658**: no S6 action under S6-6 = (a).
- **5042** (pending, medium; decomposes `GitOps.advance_main` in place): xref to S5, no S6 action.
- **6827** (pending, low): cancelled as covered by μ's `LIVE_STATE_OF` — S3's lead action (S3-5).
- **8527**, **8528** (pending, high; merge-lane policy): same-file neighbours on worker.py
  (8528 also config.py and this repo's yaml); ordered by the lead's per-file order, no edge.
- **Acceptance task `<A>`** (S6-6 = (a)): filed at decompose, `cancelled`, carrying
  `x_finding_key = [fk-c55b12f4b66d, fk-e8a03e4ec067]`, `x_finding_run`, and
  `x_acceptance_reason = "dormant while merge_train_coalesce_enabled is false on every project (Leo's ruling 2026-10-29, 66e0b1978f; reify's yaml also false); plans/mcp-coalesce-trains-2026-10-26.md §0 rulings 3-4 call the unwind and Stream 2 moot; fk-ba531799522b's coalesce-train arm is dormant on the same terms (its declared-train half is S6 θ, live on reify); reopen all three if the flag flips to true"`.
- **ξ and υ** are filed by the lead after φ lands (ξ's `before_done.script` must exist at
  `submit_task`), under `planning_mode=True` → `add_dependency(ξ, φ)`, `add_dependency(υ, ξ)` →
  `commit_planning`, with `prd_path` and `prd_task_label` so the sidecar blocks stamp. After ξ is
  done, the lead makes reify's one-line yaml flip (υ's Signal).

## Held leaves

None (OQ-8 (a), ruled 2026-11-02: the lead decomposes S5, S4, S3 and S7 before S6, so every
sibling leaf S6 depends on has an id at S6's decompose). ξ and υ are not held leaves: they are
left out of the decompose batch and filed by the lead once φ is on main, because ξ's
`before_done.script` must exist at `submit_task` (§Live tasks).

## Out of scope

- The class extraction itself (`verify_dispatch.py`, `speculation.py`, `landing.py`, `worktrees.py`,
  `telemetry.py`, the residual worker): tasks 5040, 5041, 5042, 5306, 5045 own it.
- `GitOps.advance_main`'s `_last_*` writes, `AdvanceOutcome.detail`, `_map_advance_failure`'s
  reads and signature and its two worker.py callers (S5 η).
- `GateMeasurement`, the placement deletions and the reify retry-narrowing deletion (S7 β, δ).
- Moving the merge/ops MCP tools into `orchestrator/mcp/` and escalation's reach-backs (S3).
- The workflow's ghost-loop/landed-detector consolidation, `_mark_blocked`'s shape and
  `unblock_types.py::classify_block_reason` (S9).
- `scheduler.StaleEvidenceRejection`'s home (S17).
- Re-tuning any timeout value; α derives, it does not change what the configs authorise.
- The coalesce-train journal and unwind, while coalescing is off (S6-6 = (a)).
- Arming `hold_lane` on the seven projects other than dark-factory and reify: they keep the
  default `escalate` until the lead or Leo chooses otherwise (they have never seen a refusal).
- Off-loop escalation filing: it would need S8 to make `harness.py::Harness._on_escalation` set its
  `asyncio.Event` through `loop.call_soon_threadsafe(event.set)`; S6 files on the loop, as today.

## Rulings (sitting of 2026-11-10)

Sources: `digests/wave2-rulings-applied.md` §S6 and its cross-stream block, `digests/discussion/S6.md`
(its "Effect on the plan if adopted" is part of S6-5's ruling), `digests/wave2-leo-answers-2026-11-10.md`,
and `digests/wave2-lead-rulings.md` including its "Lead rulings revised 2026-11-10" block.

| Id (sitting key) | Ruling | By |
|---|---|---|
| OQ-1 (S6-1) | As recommended (a): cancel 4581 and 7396 naming α; α lands knob, derivation and validator in one pass. | Leo (accepted) |
| OQ-2 (S6-2) | Option B: task 5040 finishes first; nothing parks it; ζ and κ wait for 5040 and build on its result. This fold adds δ → 5040 (5040's part 2 moves the arm δ rewrites). The order inverts R1's "then `ItemLifecycle`, then the decomposition" for 5040 alone; the lead records it in the program doc as an amendment of R1's order for task 5040 only (2026-11-10). | Leo; amendment recorded by the lead |
| OQ-3 (S6-3) | As recommended (a): γ1 constructs `LandedOutbox` at the existing `SpeculativeMergeWorker(` site; S8 keeps or moves it. | Leo (accepted) |
| OQ-4 (S6-4) | As recommended (a): one enum, `MergeOutcome.kind: OutcomeKind`; 5046 cancelled after 5047 and 6416 are repointed. | Leo (accepted) |
| OQ-5 (S6-5) | Option B, staged into stop-the-line: on landing, typed refusal + escalate and continue; after the operative registry has run 14 days with zero refusals on dark-factory and reify, a refusal places a per-lane `lifecycle_refused` holder owned by the refusal's escalation and leaves the request in place; one restart-only value `merge_lane.lifecycle_refusal_policy: escalate \| hold_lane` (default `escalate`) flipped per project by a dated deterministic gate; storm guard one holder per lane per refused edge. Amends decision 4 of `plans/merge-queue-modularization-invariants-prd.md` (not a 2026-10-10 ruling); φ writes the amendment. Applied as ζ (typed refusal, default branch) + φ (policy, hold arm, guard, census, amendment) + ξ (gate) + υ (dark-factory flip). | Leo |
| OQ-8 | (a): the lead decomposes S5, S4, S3 and S7 before S6; nothing held (μ depends on S3 α directly). | lead (2026-11-02) |
| OQ-9 | Names as folded: `merge_lane/item_lifecycle.py`, `merge_lane/main_health.py`, `orchestrator/verify_budget.py`; `LaneTimingBudgets`/`LaneLimits` in `merge_lane/types.py`; slug `lane-budgets-exceed-verify-timeouts`. | lead (2026-11-02) |
| OQ-10 (S6-6) | As recommended (a): the coalesce-train keys are accepted through one cancelled acceptance task that reopens them if coalescing is re-enabled; η not filed. | Leo (accepted) |
| OQ-11 (S6-7) | As recommended (a): the invariant quantifies over the merge-verify authorised budget; R1's fixture is read as "the timeout that governs a merge verify". Live effect accepted: dark-factory no-progress budget 5400 s → 7800 s, hung-verify hold 4.5 h → 6.5 h (reify 9.5 h). | Leo (accepted) |
| OQ-12 (S6-8) | As recommended (b): a per-lane holder set; θ after 4838 and 3876. Live on reify, which forms declared trains. | Leo (accepted) |
| OQ-13 | (a): S5 η does the `_last_*` reads and writes, the outcome-taking signature and both worker.py callers; S6 ε stays deleted (= S5/OQ-8 A). | lead (2026-11-02) |
| OQ-14 | Withdrawal confirmed: `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` bounds no reaper and stays 10800 s; the earlier liveness-window derivation ruling is void. | lead (2026-11-10) |
| Invariant order | Revised: S6 α INV-15 lands first, then S10 θ INV-16, then S7 ζ INV-17; checks anchor on slugs. Applied by folding the lockstep edit into α and deleting ν. | lead (2026-11-10) |
| Lead sequencing (2026-11-10, after the ruling folds) | `harness.py`: S6 α first, then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ (S6 β1 → S3 κ). S5 λ0 is second in S5's chain and precedes every S6 leaf other than α on `types.py`/`worker.py`/`harness.py` imports (S6 β1 → S5 λ0). Task 4833 lands first in today's files (S6 ζ → 4833, S6 κ → 4833). S7 δ heads S7's chain, β after it. S4 θ → S6 λ and S4 μ0 → S6 μ stand (μ now follows λ). A `lifecycle_refused` holder is rehydrated from its pending escalation; φ's split from ζ confirmed. α's resolver keyword is `merge_cold`, no default. | lead |
| Lead rulings after the fold-verifiers (2026-11-10 late; win over the row above) | S6 α lands first on every file it shares with another plan and takes no edge on S5 α, S5 λ0, S7 δ or S7 β, nor on S3 or task 4833: S5 α, S5 λ0, S7 δ, S7 β → S6 α on the shared files and S7 ε → S6 α on the resource-audit file. S6's remaining worker.py leaves keep their edge on S7 β (β1 → S7 β, the chain carries the rest) and every S6 leaf other than α on `types.py`/`worker.py`/`harness.py` imports follows S5 λ0 (β1 → S5 λ0). S6 κ → 4833 (4833 never waits on an S6 task). Invariant order S6 α INV-15 → S10 θ INV-16 → S7 ζ INV-17 by slug. | lead |
| Cross-stream S3-1 / S3-5 | No adapter anywhere; μ builds `LaneQueries` and the total `LIVE_STATE_OF` table, depends on S3 α only, and gates S3's two merge-tool moves; priority by inheritance, not raised. | Leo (S3's rulings) |

## Open questions

None open. Applying the rulings raised no new design question: the restart behaviour of a
`lifecycle_refused` holder follows the ruling's own bound ("the same bound WIP halts carry
today"), so φ rehydrates it like a WIP holder (row 9d).

Tactical (decide in the leaf):

1. Margin defaults (600 / 1,800 / 3,600 s) — α, from the reaper and resource-audit sweep periods
   at its tree.
2. The status↔kind table's rows for statuses with no unique kind — β1.
3. `LiveItem`'s exact union after 5040 and 5041 — ζ with 5041's author.
4. Which `AutoHealContext` fields the deferred arm leaves `None` — δ, against 5040's responder.
5. Where the refusal responder lives (`item_lifecycle.py` beside the registry, taking `HaltState` by
   constructor, keeps the registry itself free of halt state) — φ.

## Appendix — file lists for the mechanical leaves (measured at 38a200d213)

**β2** (51; `orchestrator/tests/` unless prefixed; `test_merge_lane_remerge_base.py` is new with task 8524): `escalation/tests/test_merge_status_boundary.py`,
`escalation/tests/test_server_chokepoint.py`, `escalation/tests/test_server.py`, `_merge_queue_harness.py`,
`test_harness_abandoned_merge_deferred_wiring.py`, `test_lane_lock_leak_guard.py`, `test_merge_deferred_recovery.py`,
`test_merge_guard_pipeline.py`, `test_merge_lane_absorbed_request_ids.py`, `test_merge_lane_intake.py`, `test_merge_lane_remerge_base.py`, `test_merge_outcome_recording.py`,
`test_merge_queue_auto_heal.py`, `test_merge_queue_bounce.py`, `test_merge_queue_chain_intact.py`,
`test_merge_queue_coalesce.py`, `test_merge_queue_concurrent_verify.py`, `test_merge_queue_deep_integration_gate.py`,
`test_merge_queue_deep_landing.py`, `test_merge_queue_disposition_wiring.py`, `test_merge_queue_dry_run_unblock.py`,
`test_merge_queue_finalize_head_visibility.py`, `test_merge_queue_host_observability.py`,
`test_merge_queue_invariant_integration_gate.py`, `test_merge_queue_lifecycle_registry.py`,
`test_merge_queue_multihost_wiring.py`, `test_merge_queue.py`, `test_merge_queue_redispatch_entries.py`,
`test_merge_queue_request_liveness.py`, `test_merge_queue_resolve_release.py`, `test_merge_queue_restart_hook.py`,
`test_merge_queue_store.py`, `test_merge_queue_supervisor.py`, `test_merge_queue_train_attribution.py`,
`test_merge_types_invariants.py`, `test_merge_verify_lease_guard.py`, `test_permit_ledger.py`,
`test_stranded_verified_green.py`, `test_verify_plan_integration.py`, `test_workflow_e2e.py`,
`test_workflow_first_enqueue_stamp.py`, `test_workflow_main_health_routing.py`,
`test_workflow_merge_blocked_surfacing.py`, `test_workflow_merge_thrash.py`, `test_workflow.py`,
`test_workflow_ready_to_merge.py`, `test_workflow_skew_surfacing.py`, `test_workflow_train_attribution.py`,
`test_workflow_train_completion.py`, `test_workflow_train_halt_owner.py`, `test_workflow_worktree_missing.py`.

**γ2** (95, unchanged; `orchestrator/tests/`; `git grep -l 'TaskWorkflow(' -- orchestrator/tests escalation/tests`):
`_orch_helpers.py`, `_review_loop_harness.py`, `_workflow_helpers.py`, `test_agent_capability_wiring.py`,
`test_architect_all_committed_plan.py`, `test_atomic_train_merge.py`, `test_config_reload_integration_gate.py`,
`test_eval_boundary_suite.py`, `test_eval_mcp_isolation.py`, `test_harness_infra_resume_truthful.py`,
`test_harness_plan_step_rederive.py`, `test_harness_warm_lane_wiring.py`, `test_harness_wip_step_detection.py`,
`test_invoke_role_config_resolution.py`, `test_liveness_boundary_gate.py`, `test_main_health_fix_owner.py`,
`test_merge_queue_auto_heal.py`, `test_merge_queue.py`, `test_merge_skew_end_to_end.py`,
`test_plan_revalidation_skip.py`, `test_plan_text_fp_preserving_stamps.py`, `test_rebase_branch_reset_guard.py`,
`test_rebase_verify_cost.py`, `test_reconcile_done_step_commits.py`, `test_session_resume_integration_gate.py`,
`test_simple_task_workflow.py`, `test_steward_outcome.py`, `test_suggestion_triage.py`,
`test_verify_admission_call_sites.py`, `test_verify_events_units.py`, `test_verify_green_checkpoint.py`,
`test_verify_phase_rebase.py`, `test_workflow_agent_session_preserve.py`, `test_workflow_already_done.py`,
`test_workflow_amendment.py`, `test_workflow_amendment_wip_guard.py`, `test_workflow_architect_l0_promotion.py`,
`test_workflow_architect_transient_retry.py`, `test_workflow_blocking_dep.py`, `test_workflow_branch_base_sha.py`,
`test_workflow_claimant.py`, `test_workflow_dry_run_hook.py`, `test_workflow_e2e.py`,
`test_workflow_escalated_steward_stall.py`, `test_workflow_escalation_warning.py`, `test_workflow_factory.py`,
`test_workflow_false_premise.py`, `test_workflow_first_enqueue_stamp.py`, `test_workflow_halt_owner.py`,
`test_workflow_head_commit.py`, `test_workflow_infra_thrash.py`, `test_workflow_invocation_end_truthful.py`,
`test_workflow_main_health_routing.py`, `test_workflow_mark_blocked_l0_coverage.py`,
`test_workflow_merge_blocked_surfacing.py`, `test_workflow_merge_gating_strand.py`,
`test_workflow_merge_phase_block_status.py`, `test_workflow_merge_phase_liveness.py`,
`test_workflow_merge_retry_pending.py`, `test_workflow_merge_superseded.py`, `test_workflow_merge_thrash.py`,
`test_workflow_metadata_files_reconcile.py`, `test_workflow_no_plan_cycle.py`,
`test_workflow_operator_merge_cancel.py`, `test_workflow_plan_repair_replan.py`,
`test_workflow_preexisting_main_break.py`, `test_workflow_prompt_builder.py`, `test_workflow.py`,
`test_workflow_ready_to_merge.py`, `test_workflow_resettled_suppression.py`, `test_workflow_resume_on_progress.py`,
`test_workflow_review_persistence.py`, `test_workflow_routing_decision.py`, `test_workflow_sandbox_events.py`,
`test_workflow_scope_wipe_regression.py`, `test_workflow_signature_loop_guard.py`, `test_workflow_skew_surfacing.py`,
`test_workflow_state_machine_boundary.py`, `test_workflow_state_machine.py`, `test_workflow_status_on_resume.py`,
`test_workflow_train_attribution.py`, `test_workflow_train_completion.py`, `test_workflow_train_former.py`,
`test_workflow_train_halt_owner.py`, `test_workflow_train_state_escalation.py`, `test_workflow_train_thrash_guard.py`,
`test_workflow_train_worktree.py`, `test_workflow_truthful_commit_stamp.py`, `test_workflow_truthful_exits.py`,
`test_workflow_unactionable.py`, `test_workflow_verify_infra_resume.py`, `test_workflow_verify_retry.py`,
`test_workflow_worktree_missing.py`, `test_workflow_worktree_provision.py`, `test_workflow_zero_output_hang.py`.

**κ** (16; `orchestrator/tests/`; `.<NAME> =` or `setattr(…, '<NAME>')` over the 19 names α leaves;
104 pins at 38a200d213, to be re-measured after α, which edits the same files):
`test_merge_queue_request_liveness.py` 44, `test_merge_queue_resource_audit.py` 15,
`test_merge_queue_dispatch_fill_redispatch.py` 11, `test_merge_queue_concurrent_verify.py` 9,
`test_merge_queue_multihost_wiring.py` 7, `test_merge_speculation.py` 3, `test_merge_queue.py` 3,
`test_merge_queue_train_attribution.py` 2, `test_merge_queue_lifecycle_registry.py` 2,
`test_merge_queue_invariant_integration_gate.py` 2, `test_merge_queue_resolve_release.py` 1,
`test_merge_queue_permit_conservation.py` 1, `test_merge_queue_deep_dispatch.py` 1,
`test_merge_queue_chain_intact.py` 1, `test_concurrent_verify_boundary.py` 1,
`test_merge_lane_instrumentation.py` 1.
