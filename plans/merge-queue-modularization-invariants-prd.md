# Merge-queue modularization + explicit invariant enforcement

**Status:** active — authored 2026-07-02 from an interactive architecture study of
`orchestrator/src/orchestrator/merge_queue.py` (12,928 lines as of `9559f8ab0f`).
**Type:** refactoring + robustness PRD over shipped, load-bearing code. B+H
(contract + boundary tests): the merge queue is THE load-bearing seam of the
orchestrator, and this batch touches all of it.

## Goal

Retire the implementation-level bug classes that made the merge queue a hotspot
(task-1917/1928 field-drop, resource leak/double-release hazards, silent-hang
Futures, tri-plicated guard pipelines) by (a) splitting the monolith into
cohesive modules, (b) converting hand-audited prose invariants into enforced,
operator-observable checks, and (c) reifying resource ownership into objects.
The conceptual architecture — two-layer queue, submission-order advancement,
pure-git-engine layering — is deliberately **unchanged**.

Operator-observable outcome when the batch lands: the dashboard/snapshot gains
`resource_audit` and speculation-state surfaces; a wedged verify or leaked
permit/worktree fires a loud escalation instead of hanging silently; ill-formed
pipeline items fail at construction; and every future edit to the subsystem
conflicts with a narrower file footprint (the split directly thins the
factory's own conflict graph for merge-queue work).

## Background

Study findings (2026-07-01/02 session, anchors at `9559f8ab0f`):

- One 12.9k-line module holds the engine, two workers, all gates, registries,
  metrics, breaker, shadow-compare, drift-check, liveness guards.
- `SpeculativeItem` is a product type whose ~8 nullable fields encode ≥3
  distinct shapes; legal combinations documented only in prose. Hand-rebuilt at
  two CAS-loop sites (merge_queue.py:10683, :10761) — the 1928 bug was a field
  dropped at one of them.
- Resource ownership (speculation permits, merge-ahead cap, host leases, owned
  worktrees) is encoded in control-flow flags (`held_spec_permit`,
  `_entry_released`, `_skip_release`/`_cancel_release`) across ~8 exit paths,
  guarded by comments and deliberate Semaphore over-release tolerance.
- The guard pipeline (timeout breaker → branch presence → already-merged →
  merge → conflict/failure → drop-guard) exists three-plus times
  (`MergeWorker._do_merge` :5256, `_merger_loop` :8963, `_remerge` :9914 ×2).
- Five near-identical `BaseException` handlers in `_verifier_loop` (:9532,
  :9578, :9614, :9757, :9814) each hand-replicate resolve+release+clean.
- `advance_main` smuggles results via getattr side channels
  (`_last_advanced_sha`, `_rebased_from`, `_rebased_onto`; read at :10626 with
  a runtime AssertionError backstop).
- Invariant checkers already exist as observability
  (`check_frozen_prefix_invariant` :6752, `two_layer_invariants` :6804,
  log-only ε=1890 verify-base guard :6878) — the pattern to extend.

Prior art: two-layer PRD `plans/two-layer-merge-queue-prd.md` (landed, tasks
1886–1895); as-built reference `skills/merge-queue/references/two-layer-model.md`.

## Sketch of approach

Seventeen tasks in one strictly-linear dependency chain (every task touches
`merge_queue.py`, whose file-level lock serializes them anyway — the chain
prevents park churn and rebase bounce):

**Phase 1 — module split (α–δ).** Mechanical extractions with a re-export shim
in `merge_queue.py` so no importer breaks: types (α), gates (β),
shadow/drift/liveness (γ), suffix-conflict tracker (δ). Each extraction task
also updates the symbol→location map in
`skills/merge-queue/references/two-layer-model.md` for what it moved.

**Phase 2 — shape + lifecycle hardening (ε–ι).** Item-shape validation and
replace-only rebuilds (ε); single resolve-and-release chokepoint (ζ);
request-liveness ledger with heartbeat escalation (η); SpeculationController
owning permit lifecycle + merger state machine (θ); conservation audits for
permits/caps and worktrees (ι).

**Phase 3 — dedup + API cleanup (κ–ν).** Shared guard pipeline (κ);
declarative post-advance gate chain (λ); AdvanceOutcome value object replacing
side channels (μ); retire MergeWorker to a test fixture (ν).

**Phase 4 — enforcement promotion + deep type split (ξ–ο).** Promote the
log-only verify-base guard into `two_layer_invariants` + single-writer debug
assertions (ξ); full SpeculativeItem sum-type split (ο, low priority — ε
already retired the observed bug class; ο makes illegal states
unrepresentable).

**Phase 5 — gate + deploy (π–ρ).** B+H integration-gate leaf driving the full
pipeline under fault injection and asserting every new invariant surface (π);
deterministic deploy restarting **all running** `orchestrator-*.service` units
via `scripts/restart-all-orchestrators.sh` (ρ; committed alongside this PRD —
the deterministic guard validates `before_done.script` existence at
`submit_task` time).

## Resolved design decisions

1. **Split-first sequencing.** The module split lands before the behavioral
   refactors so later tasks edit the new, narrower modules. Accepted cost:
   phase-2+ task descriptions reference post-split module names; the re-export
   shim keeps `merge_queue.py` imports valid throughout.
2. **Validation now, sum-type at tail** (user decision 2026-07-02):
   `__post_init__` + `dataclasses.replace`-only + Enums in ε (early); full
   tagged-union split deferred to ο (low priority, after the invariant/test
   scaffolding exists).
3. **Deploy capstone restarts all running orchestrators** (user decision
   2026-07-02): every `orchestrator-*.service` unit runs this repo's
   orchestrator package; restarting only dark-factory would leave five units
   on stale code. `target_unit='orchestrator-dark-factory.service'` routes the
   DeterministicRunner through its detached `systemd-run` self-restart path;
   the script restarts the self unit last (defensive ordering).
4. **Invariants escalate loudly, degrade never** (standing user directive):
   liveness/conservation violations fire dedup'd L1 escalations via the
   worker's existing `_escalation_queue`; they never halt or degrade the
   pipeline themselves (observation → escalation, enforcement stays with the
   existing halt machinery).
5. **Behavior preservation is the contract** for α–δ, κ, μ, ν: existing tests
   must pass with import-path churn only; `MergeOutcome.status` literals,
   reason-prefix constants, EventType emissions, snapshot() existing keys, and
   the journal format are frozen (additive-only). Workflow.py routes on reason
   prefixes — they are load-bearing strings.
6. **MergeWorker retires** (ν): sole production worker is
   SpeculativeMergeWorker (harness.py:4958); the serial class survives only as
   whatever test fixture the ported tests need. `_TrainMergeHost` simplifies
   accordingly. The "readable serial reference" role passes to
   `skills/merge-queue/references/two-layer-model.md`.
7. **Conservation checks are audit functions, not inline asserts**: pure
   `*_violations() -> list[str]` in the `two_layer_invariants` idiom, surfaced
   via snapshot()/heartbeat, hard-assertable by tests and the π gate. Debug
   single-writer assertions (I7) are gated on a debug flag so production cost
   is nil.

## Pre-conditions for activating

- `scripts/restart-all-orchestrators.sh` committed and executable on main
  (same commit as this PRD) — required by ρ's `before_done` validation at
  filing time.
- No other pending/in-progress task touches `orchestrator/src/orchestrator/`
  merge-queue files (verified 2026-07-02: live tasks 1939–1981 are
  recon/fused-memory/reify-config work).

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/offline-deep-test-lane-worker.md` (df 1951–1956, landed) | consumes | `on_merge_landed(task_id, base_sha, head_sha)` fail-open hook (merge_queue.py:10588) | offline-lane PRD owns the notifiee; **this PRD owns keeping the hook invocation + signature stable** through all refactors | wired |
| `plans/two-layer-merge-queue-prd.md` (landed 1886–1895) | refactors its as-built code | frozen-prefix/suffix accessors, `two_layer_invariants`, bounce, aging | this PRD (α–δ, ξ) — semantics frozen, locations move; doc map updated per extraction | wired |
| `plans/deterministic-task-kind-prd.md` (landed 1898–1904) | consumes | `task_kind='deterministic'` runner + detached self-restart path | deterministic PRD owns the runner; this PRD only files a consumer task (ρ) | wired |

No contested seams; no reciprocal-ownership ambiguity.

## Contract (B+H)

**Frozen public surface (behavior-preservation contract):**
- `MergeOutcome.status` literal set and field meanings (merge_queue.py:4005).
- Reason-prefix constants (`DROPPED_PLAN_TARGETS_…`, `PLAN_FILES_NOT_TOUCHED_…`,
  `POST_MERGE_EQUIVALENCE_FAILED_…`, `NEEDS_REBASE_…`, `MERGE_WORKER_SHUTDOWN_REASON`)
  — workflow.py and operators route on these strings.
- EventType emissions and their `data` keys (merge_dequeued, merge_attempt,
  merge_queued, merge_coalesced, speculative_*, worktree_reaped).
- `snapshot()` existing keys — changes additive-only.
- `merge_queue_store` journal schema (PersistedMergeRequest fields).
- `on_merge_landed` hook signature + fail-open semantics.
- Import surface: `from orchestrator.merge_queue import X` keeps working via
  re-export shim until all importers are migrated (migration may be folded
  into the extraction tasks; shim removal is out of scope).
- Ordering invariant: main advances strictly in submission order; Future
  resolution ordering remains unspecified across item kinds.

**New invariant surfaces (additive):**
- I1 liveness ledger: every request transitions dequeued → {resolved,
  requeued} exactly once; a request unresolved-and-unowned past a heartbeat
  threshold fires a dedup'd L1 escalation naming request_id/branch/age.
- I2 item shape: `immediate_outcome XOR merge_result`; `merge_wt ⟺
  merge_result`; `already_delivered ⇒ immediate_outcome` — enforced at
  construction.
- I3 field-carry: pipeline items are constructed only at whitelisted factory
  sites; all rebuilds go through `dataclasses.replace` (grep-guard test).
- I4 permit/cap conservation: `semaphore_value + held_by_merger +
  inflight_speculative == K` (and the merge-ahead analogue) — audit fn,
  snapshot key, heartbeat check.
- I5 verify-base = frozen-prefix tip: promoted from log-only WARNING (:6878)
  to a `two_layer_invariants` violation string (hard enforcement stays out of
  scope; π asserts zero violations under its scenarios).
- I6 worktree conservation: owned-ledger ≈ disk `_merge-*` set (modulo
  documented exemptions :6206–6219); audit fn + heartbeat check + escalation
  on persistence.
- I7 single-writer: lane buffers mutated only from the merger coroutine,
  `_inflight` only from the verifier — debug-flag asserted.

## Boundary-test sketch (π's spec)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Speculative N fails, N+1 in flight | depth≥1, N+1 merged on N's commit | cascade re-merges N+1 vs main; N+1 lands; both Futures resolved; I4 holds |
| 2 | RunnerUnavailable mid-verify | remote lease active | host quarantined; item re-dispatched; not a chain failure; Future resolved |
| 3 | Operator halt mid-verify | verify running | REQUEUED; re-verifies after unhalt; per-task counters untouched |
| 4 | Waiter abandons mid-verify | sole waiter cancels Future | DROPPED; worktree cleaned; no set_result; I6 holds |
| 5 | Verify task wedges forever | injected never-completing verify | I1 escalation within heartbeat window naming the request (**new**) |
| 6 | Forced permit leak (test hook) | steady state | I4 violation string in snapshot + heartbeat WARNING (**new**) |
| 7 | Ill-formed item construction | immediate_outcome AND merge_result both set | raises at construction (**new**) |
| 8 | CAS-retry rebased landing | main advances during verify; rebase flattens merge | equivalence gate uses carried merged_branch_tip; no phantom failure (1928 regression pin) |
| 9 | Guard matrix: merger path vs _remerge path | missing branch / already-merged / conflict / drop-guard | identical MergeOutcome + events from the shared pipeline (κ) |
| 10 | Verify dispatched against non-frozen-tip base | injected mismatch | violation string in `two_layer_invariants` output (**new**, was WARN-only) |
| 11 | Module-split preservation | α–δ landed | full suite green, test edits limited to import paths; snapshot keys unchanged |
| 12 | MCP coalesce onto in-flight branch | registry slot held | attach/alias behavior unchanged (registry contract) |

## Decomposition plan

Strictly linear chain: each task depends on its predecessor. All tasks
`planning_mode=True`; priority medium unless noted.

- **α — Extract merge_types.py (request/outcome/item/entry types + registries).**
  Move MergeRequest/GroupMergeRequest/MergeOutcome/SpeculativeItem/InflightEntry/
  WaiterRecord/InFlightMergeRegistry/TerminalOutcomeRetention/MergeBounceRegistry
  (+ friends) to `orchestrator/merge_types.py`; re-export from merge_queue.py.
  Update two-layer-model.md map. Modules: orchestrator/src, orchestrator/tests,
  skills/merge-queue. Signal: `python -c "from orchestrator.merge_types import
  MergeRequest, SpeculativeItem, InflightEntry"` succeeds; full suite green with
  import-path-only test churn; dashboard snapshot keys unchanged. Unlocks β.
- **β — Extract merge_gates.py (post-merge gates + finalize + reason prefixes).**
  Move drop-guard, plan-files-touched, `_check_post_merge_equivalence`,
  `_check_post_merge_pyright`, `_rebase_delta_touched_overlap`,
  `_reverify_rebased_tree`, `_finalize_advanced_merge`, `_map_advance_failure`,
  reason-prefix constants. (`_run_post_merge_verify` placement is tactical —
  architect decides gates-vs-engine.) Signal: gate imports work; existing gate
  tests green unmodified; reason prefixes byte-identical. Unlocks γ.
- **γ — Extract shadow-compare / drift-check / liveness-guard modules.**
  Move `_run_shadow_compare`/`_maybe_schedule_shadow_compare`/ShadowCompare*,
  `_run_drift_check`/`_maybe_run_drift_check`, `check/enforce_merge_liveness_margin`,
  persistent-worktree helpers (≈ merge_queue.py:11141–12928) to
  `merge_shadow.py`/`merge_drift.py`/`merge_liveness.py` (exact grouping
  tactical). Signal: a landed merge still schedules shadow compare (journal
  line); modules importable; suite green. Unlocks δ.
- **δ — Extract SuffixConflictTracker (conflict graph + bounce state).**
  New class owning `_suffix_conflict_graph`/`_suffix_conflict_signature`/
  `_bounce_registry` + `recompute_suffix_conflict_graph`/
  `_bounce_conflicting_suffix_items`, taking git_ops + lane-buffer/frozen-tip
  accessors; worker delegates. Signal: existing conflict-graph + bounce tests
  green against the tracker; `snapshot()['suffix_conflict_graph']` unchanged.
  Unlocks ε.
- **ε — Item-shape validation + replace-only rebuilds + status Enums (I2, I3).**
  `__post_init__` invariants on SpeculativeItem/InflightEntry; both CAS-loop
  rebuild sites → `dataclasses.replace`; sentinel strings → Enum
  (string-compat); grep-guard test whitelisting factory construction sites.
  Signal: ill-formed construction raises (CI test); grep-guard fails on
  non-factory `SpeculativeItem(`; suite green. Unlocks ζ.
- **ζ — Single resolve-and-release chokepoint in the verifier loop.**
  `_resolve_and_release(entry, outcome, *, chain_failed)` replaces the five
  duplicated BaseException handlers (dispatch, passthrough-finalize,
  finalize-head, cascade, blocking-get). Signal: fault-injection tests show
  each path resolves the Future 'blocked', releases lease+slot, cleans the
  worktree through the one chokepoint; no leaked `_merge-*` dirs. Unlocks η.
- **η — Request-liveness ledger + heartbeat stuck-Future escalation (I1).** [priority: high]
  Ledger tracks dequeue → terminal transitions (hooks ζ's chokepoint + requeue
  sites); heartbeat checks for unresolved-and-unowned requests past a
  threshold; dedup'd L1 via `_escalation_queue`. Signal: harness test wedges a
  verify forever → heartbeat logs the stuck request AND an escalation appears
  in `get_pending_escalations` naming request_id/branch/age. Consumer:
  escalation-watcher + π.
- **θ — SpeculationController: explicit permit ownership + merger state machine (R3+R5).**
  Object owning spec_base/prefetched/pending-attach/permit with explicit
  transfer semantics; `_merger_loop` delegates; late-arrival attach (1862)
  becomes controller methods. Signal: permit-conservation property test across
  prefetch/attach/fallback/cascade/shutdown; snapshot() speculation-state key.
  Unlocks ι.
- **ι — Conservation audits: permits/caps + worktree ledger (I4, I6).**
  Pure `*_violations()` audit fns; `snapshot()['resource_audit']`; heartbeat
  check; dedup'd L1 on persistent violation. Signal: steady state reports zero
  violations; test-forced permit leak / unregistered worktree produces a
  violation string in snapshot + heartbeat WARNING. Consumer: dashboard,
  escalation-watcher, π.
- **κ — Shared guard pipeline `classify_and_merge` (R4).**
  One function running abandonment/timeout-breaker/branch-presence/
  already-merged/merge/conflict/failure-diagnostic/drop-guard, used by
  `_merger_loop` + `_remerge` (+ MergeWorker until ν). Signal: parameterized
  equivalence tests prove identical MergeOutcome + events across paths for the
  guard matrix; duplicated code deleted. Unlocks λ.
- **λ — Declarative post-advance gate chain (R6).**
  `POST_ADVANCE_GATES: list[Gate]` (equivalence, pyright) iterated by
  `_finalize_advanced_merge`; each returns a GateVerdict. Signal: a test
  registers a no-op gate and it runs post-advance without editing finalize;
  gate names logged per landing. Unlocks μ.
- **μ — AdvanceOutcome value object from advance_main (R7a).**
  `advance_main` returns (result, advanced_sha, rebased_from, rebased_onto);
  getattr side channels + the :10629 AssertionError deleted. Modules include
  git_ops.py. Signal: grep shows zero `_last_advanced_sha`/`_rebased_from`/
  `_rebased_onto` getattr reads; CAS-retry landing records correct
  done_provenance SHA. Unlocks ν.
- **ν — Retire MergeWorker (R7b).**
  Remove from production module; port its tests to SpeculativeMergeWorker/shim
  or a tests-local fixture; simplify `_TrainMergeHost` + harness type union.
  Signal: grep shows no MergeWorker in orchestrator/src; suite green. Unlocks ξ.
- **ξ — Promote verify-base guard into two_layer_invariants (I5) + single-writer debug assertions (I7).**
  Precondition inside the task: journal shows the ε=1890 WARNING has not fired
  spuriously (investigate any hits first). Signal: injected verify-base
  mismatch yields a violation string in `snapshot()['two_layer_invariants']`;
  debug-mode assertion fires when a test mutates lane buffers from a
  non-merger task. Unlocks ο.
- **ο — Full SpeculativeItem sum-type split (R2 deep).** [priority: low]
  RealMergeItem/DecidedItem tagged union; exhaustive matching in
  dispatch/finalize; ε's grep-guard updated to union constructors. Signal:
  pyright proves exhaustive handling; suite green. Unlocks π.
- **π — Invariant integration gate (B+H leaf).** [priority: high]
  One integration test drives merge bursts with injected faults (verify fail,
  RunnerUnavailable, operator halt, abandoned waiter, wedged verify, forced
  permit leak) and asserts at quiescence: all Futures resolved; I4 zero
  violations; worktree ledger == disk; liveness ledger empty;
  `two_layer_invariants() == []`; boundary rows 1–12 covered. Signal: the
  integration test is green in CI. Consumer: ρ + operator confidence.
- **ρ — Deterministic deploy: restart all running orchestrators.**
  `task_kind='deterministic'`, `before_done={script:
  'scripts/restart-all-orchestrators.sh', args: [], timeout_secs: 600,
  target_unit: 'orchestrator-dark-factory.service'}`, `always_escalates=false`
  (auto-deploy preset: escalate only on failure; done='scheduled' via detached
  self-restart). Signal: runner stamps `done_provenance
  kind='deterministic-deploy-scheduled'`; script output in journal shows every
  running `orchestrator-*.service` verified fresh
  (ActiveEnterTimestampMonotonic advanced). Depends on π.

## Out of scope

- Widening verify depth / speculation semantics; bounce policy changes.
- Flipping `AUTO_CHAIN_GENERATIONS_ENABLED` (γ3 preconditions unchanged).
- Removing the merge_queue.py re-export shim (follow-up once importers migrate).
- Dashboard UI work beyond rendering new snapshot keys.
- Reify-side code; fused-memory; scheduler changes.
- Performance optimization.

## Open questions (tactical)

1. **`_run_post_merge_verify` placement** (gates vs engine module). Suggested:
   keep in merge_queue.py (it is verify *execution*, not policy). Decide in β.
2. **Escalation category names** for I1/I4/I6 (e.g. `merge_request_stuck`,
   `merge_resource_leak`). Decide in η/ι; additive, watcher picks them up
   generically.
3. **Debug-flag mechanism for I7** (env var vs config knob). Suggested env var
   (`ORCH_DEBUG_ASSERTS=1`) set in test conftest. Decide in ξ.
4. **Grep-guard implementation** (pytest collecting `git grep` vs pre-merge
   hook). Suggested pytest — rides existing verify. Decide in ε.
5. **Exact module grouping in γ** (one file vs three). Suggested three small
   modules. Decide in γ.
