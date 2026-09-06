# Merge lane: quality program — ratchets, test seams, the `merge_lane` package

**Status:** active — authored 2026-09-03 (Leo + claude-interactive) from a measured
baseline of `orchestrator/src/orchestrator/merge_queue.py` and its 18 satellites.
**Type:** behaviour-preserving refactoring PRD over shipped, load-bearing code.
**Approach:** B+H — the merge lane is the single path through which every hosted
project's code reaches `main`; this batch restructures all of it.
**Code anchors** verified against main `45ebf0fbc0` (2026-09-03). Main moves fast —
cite-by-symbol; re-locate at implementation time. The measurement tables below are
dated provenance of that commit, not live counts: do not re-anchor them.
**Companion PRD:** `plans/merge-lane-throughput-prd.md` (self-hosting on the remote
verify host, arbitration, deep-chain canary, measurement — config and measurement
work that touches none of Appendix A and runs in parallel with this batch). The
*policy* follow-up it seeds (dispatch and speculation algorithm changes) is the PRD
that depends on tasks κ and λ here — see § Cross-PRD relationship.

## Goal

Cut the cost of the next agent change to the merge lane, and stop that cost from
climbing back. Operator-observable outcome when the batch lands:

- `scripts/merge_lane_metrics.py --report` prints a table in which no file in the
  lane exceeds 1,500 lines, no function exceeds cognitive complexity 40, and the
  counts of reach-back imports, re-export shims, string-path test patches into lane
  internals, and private-attribute reads from tests are all **zero**.
- `pytest orchestrator/tests/test_merge_lane_ratchet.py` is green on `main` and goes
  red on any commit that raises one of those measures above its committed baseline —
  so a fourth round of growth cannot silently undo this one, as growth undid the
  three previous refactors (§ Background).
- The lane lives in a nested package `orchestrator/merge_lane/` behind one façade;
  `orchestrator.merge_queue` no longer exists as an import path.
- Tests drive the lane through injected collaborators and real git fixtures. The
  frozen copy of the retired serial worker is gone.
- Behaviour is unchanged: every wire string, event, snapshot key, journal row and
  hook signature in § Contract is byte-identical before and after.

## Background

Definition ratified 2026-09-03: **quality is the cost and risk of the next change,
and here the next change is made by an agent.** Leo's fourteen heuristics (memory
`user-code-quality-heuristics`) are the review vocabulary for every task in this
batch; the ones this PRD leans on hardest are *small function scopes*, *minimum data
access scopes and lifetimes*, *deep modules with narrow interfaces*, *files make
internal sense in isolation*, *no file too large — and no cheating by stitching
files together with imports*, *SPOT*, and *structured data instead of meaningful
strings*.

Measured baseline at `45ebf0fbc0` (radon 6.0.1, complexipy 7.0.1, stdlib-AST
scripts; reproducible from the session scratchpad and re-derivable by task α's
script):

| Measure | Value |
|---|---|
| `merge_queue.py` lines / source lines / comment+docstring lines | 21,550 / 8,257 / 11,876 (55% prose; ~300k tokens) |
| Cluster (19 modules) lines | ~52,000; `git_ops.py` 14,721 |
| Growth of `merge_queue.py`, 2026-06-01 → 09-01 | 3,495 → 12,915 → 15,228 → 21,550 lines; source ×3.7, prose ×12.5 |
| Refactors landed in that window | task 1593 (shared core, 06-01); tasks 1985–2001 (17-task split, 07-02→05); every proposal of the 07-06 hotspot survey (PermitLedger, SpecPermit, ItemLifecycle, QueuedBranch, landed_outbox) |
| Maintainability index (radon) | 0 for `merge_queue.py` and `git_ops.py`; A for every satellite |
| Cognitive complexity, `merge_queue.py` total / functions >15 / >50 | 2,132 / 27 / 9 (was 540 / 12 / 3 on 06-01) |
| Worst functions (cognitive) | `SpeculativeMergeWorker._verifier_loop` 245 · `_run_post_merge_verify` 175 · `GitOps.advance_main` 113 · `SpeculativeMergeWorker.stop` 109 · `coalesce_or_enqueue_merge_request` 103 · `SpeculativeMergeWorker._finalize_inflight` 102 · `SpeculativeMergeWorker._run_inflight_verify` 101 · `SpeculativeMergeWorker._merger_loop` 86 |
| Same functions by 12-week churn (distinct commits) | `_finalize_inflight` 69 · `_run_inflight_verify` 65 · `_run_post_merge_verify` 60 · `_verifier_loop` 39 · `_merger_loop` 34 |
| `SpeculativeMergeWorker` | 120 methods, 661-line `__init__`; 131 instance attributes, 45 written only in `__init__`, 8 written from ≥3 concerns |
| Healthiest lane module (achievability reference) | `verify_runner.py`: 3,531 lines, cognitive total 275, max 48 |
| Fan-out / fan-in | imports 134 names from 28 sibling modules; 17 modules import it; 7 satellites re-exported as its own surface (`# noqa: F401 re-export shim`) |
| Reach-back imports (function-local `from orchestrator.merge_queue import …` inside satellites, to break cycles) | 23 sites in 6 satellites |
| Tests | 120 files, 4,875 tests, 181,798 lines (8.4 : 1) |
| String-path patches into `orchestrator.merge_queue.*` | 1,111 call sites, 79 distinct names (`run_scoped_verification` alone 375) |
| Private-attribute reads from tests (`worker._x` etc.) | 5,735 |
| Files using a real git repo | 10 of 120; conftest's autouse `_mock_merge_queue_verification` stubs verification to *passed* for all but 2 tests |
| `orchestrator/tests/_serial_merge_worker.py` | frozen copy of the retired serial `MergeWorker`, anchoring ~89 test constructions |
| 12-week commits on `merge_queue.py` | 567 (53% of all merge-lane commits); 48 `fix:` + 106 `amend:` + 66 mentioning regression |

The finding that shapes this PRD: **growth outran three landed refactors.** The July
split produced shallow satellites that are function-bags over the worker's private
state, stitched back through 23 function-local reach-back imports because the test
suite pins 79 dotted paths into the monolith. Tests froze the seams; the seams were
then cut in the wrong place; new code kept landing in the monolith. This PRD reverses
that order: ratchets first, test seams second, package and extraction third.

Prior art: `plans/merge-queue-modularization-invariants-prd.md` (landed; its
"Removing the merge_queue.py re-export shim" out-of-scope line is the tear this PRD
closes), `plans/merge-queue-reliability-prd.md` scope ε
(`orchestrator/tests/test_merge_queue_reachback_patch_guard.py`, the allowlist-style
precedent for task α), `plans/bug-hotspot-survey-2026-07-06.md` § merge-queue.

## Sketch of approach

Five phases, strictly ordered by dependency. Every task is behaviour-preserving
(§ Contract) and must leave `test_merge_lane_ratchet.py` green — a task that
moves code lowers the baseline it moved away from in the same commit; a task can
never raise one.

**Phase 0 — instruments (α).** A metrics script and a ratchet test with a committed
baseline. Nothing else starts until this is on `main`.

**Phase 1 — test seams (ζ1, β, γ1–γ10, δ, ε).** Create the package skeleton with
the collaborator *ports* first, so the worker can accept injected collaborators;
migrate the 120 test files in ten LOC-balanced groups from dotted-path patching and
private-attribute inspection to injected fakes and public observations; discard
the serial-worker copy and the autouse verification stub; measure a mutation-score
baseline. After Phase 1 the seams are no longer frozen by tests.

**Phase 2 — the package (ζ2, η).** Move the monolith and satellites under
`orchestrator/merge_lane/`, delete every reach-back import and re-export shim, then
migrate external importers and delete `orchestrator/merge_queue.py`.

**Phase 3 — seam extraction (θ, ι, κ, λ, μ, ν, ξ, ο, π).** Carve
`SpeculativeMergeWorker` into deep modules along the measured concern map
(§ Contract lists the modules and which shared state each owns). κ (verify dispatch)
and λ (speculation) are the seams the companion throughput PRD lands its features in.
π replaces reason-string routing with a structured failure kind.

**Phase 4 — prose (ρ).** Relocate task/escalation archaeology comments to one
git-tracked home with pointers in code.

**Phase 5 — gate and deploy (σ, τ).** The B+H integration gate asserts the boundary
rows and the ceilings; a deterministic deploy restarts the fleet onto the new lane.

## Resolved design decisions

1. **Ratchets before any structural change** (Leo, 2026-09-03). The instrument is a
   pytest test so it rides every verify leg without wiring; it is *hard from the
   first commit* on increases and permits equality, which is what makes it a ratchet
   rather than a gate that fails on day one. Cluster-wide totals (lines, cognitive
   complexity) are included alongside per-path measures so a rename cannot game the
   per-path baseline.
2. **Execution is orchestrated, not interactive** (Leo). His attention is the
   binding constraint. The seam design therefore lives *here* (§ Contract) rather
   than in each task; each extraction task names the source range it moves and the
   ratchet is its objective signal. No task in this batch may be scoped so that an
   agent has to read `merge_queue.py` whole.
3. **Package with a single façade** (Leo). `orchestrator/merge_lane/__init__.py`
   exports exactly the names in § Contract → *Public surface*; submodules are
   private to the package; imports flow one way (worker → concern modules → types /
   ports). Function-local imports inside the package are forbidden (ratchet measure).
4. **Discard the serial worker copy** (Leo). A fallback never exercised is not a
   fallback (INV-10). The behaviours its ~89 constructions check are re-homed onto the
   façade + fakes in γ/δ, not deleted.
5. **Tests reach internals through ports, never attributes** (Leo: "test access to
   internals is an interface design smell"). β defines `VerifyPort`, `ClockPort`,
   `EscalationPort`; tests inject fakes. A test asserting on a private attribute or a
   patched callee's call count is rewritten to assert on a `MergeOutcome`, an emitted
   event, a `snapshot()` key, or git state — or deleted if another test already
   covers the behaviour (deletions listed in the commit body).
6. **Extraction follows the measured concern map, not topic.** The July split was
   by topic and produced function-bags. The concern map (§ Contract → *Module
   ownership*) assigns every shared mutable attribute one owner; a concern module
   exposes methods, not state.
7. **Reason strings stay byte-identical; routing moves to a kind** (π). Workflow
   routes merge failures by `reason.startswith(<PREFIX>)` on ten of seventeen
   `*_REASON_PREFIX` constants. π adds `MergeFailureKind` to `MergeOutcome`, makes
   every prefix map to exactly one kind, switches workflow to route on the kind, and
   keeps the strings for humans and logs. A pinned test proves the routing decision
   is identical for every prefix.
8. **Prose relocates to one home in git, not into memory by agents** (ρ). Dispatched
   agents hold no memory-write surface, so the home is
   `docs/merge-lane/decisions.md` (one dated entry per rationale, citing its task or
   escalation id); code keeps a one-line pointer. INV-9: one home, pointers elsewhere.
9. **`git_ops.py` stays outside the package.** It is the git engine for warm lanes,
   the scheduler and recovery, not only the merge lane. μ decomposes
   `GitOps.advance_main` *in place* into named steps; the broader `git_ops.py` split
   is a follow-up PRD (§ Out of scope).
10. **Existing cluster-touching tasks are dependency-gated behind ζ2** (Leo). The
    decomposer wires every non-terminal task whose `metadata.files` (or `files_hint`)
    intersects Appendix A behind ζ2, so each such task rebases onto the package once,
    after the move, instead of colliding with it. Selection is by declared files,
    never by keyword (a keyword search returned 507 tasks, inflated by "train" and
    "landing").
11. **Deploy cadence is not a pre-condition.** Fleet auto-redeploy is paused pending
    task 5020; τ restarts the fleet deterministically so the landed lane actually
    runs. If 5020 lands first, τ is redundant and harmless. Tasks 3730/3733/4755/5020
    are *not* dependencies of anything here (Leo, 2026-09-03).

## Pre-conditions for activating

- Ten branches carry unlanded commits touching Appendix A files: tasks 3154, 3203,
  3226, 3310, 3778, 3790, 4023, 4189, 4211, 4930 (task 4122 is cancelled; its
  stranded branch is not live work). The decomposer wires ζ1 behind the ones that
  are **in-progress** at decompose time (3226 and 4930 on 2026-09-03) — those have a
  live claimant, so the hold is bounded (INV-7). The **pending** ones are gated
  behind ζ2 with every other cluster-touching task (decision 10) and will rebase
  across the package move; a dependency on a pending task with no claimant would be
  an unbounded hold and is deliberately not wired.
- `scripts/restart-all-orchestrators.sh` exists and is executable on `main`
  (verified 2026-09-03) — τ's `before_done` validation needs it at filing.
- `radon`, `complexipy`, `mutmut` are not in the lockfile; α (and ε for mutmut) add
  them to `orchestrator/pyproject.toml` `[dependency-groups] dev` and `uv.lock`.
- Fleet-wide blast radius accepted (Leo): every project's orchestrator runs this
  package; a regression here is fleet-wide within a redeploy window, and a commit
  that breaks the lane blocks its own revert from landing through the queue. The
  escalation watcher's rollback is therefore a *direct commit to the machine-operated
  main checkout* per `CLAUDE.md` § Working in the main checkout — σ's task text
  restates that one line so the runbook is on the record.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-lane-throughput-prd.md` | parallel; consumes only landed code | none of Appendix A — it edits `dark-factory-orchestrator.yaml`, remote-host state, `scripts/`, and reports; its deep-chain canary *exercises* `build_chain`/`select_chain_depth` unchanged | each PRD owns its own files; no shared task | queued (companion batch) |
| speculation/dispatch-policy follow-up PRD (to be authored from the throughput PRD's diagnosis task F, plus Leo's own ideas) | consumes | `merge_lane/verify_dispatch.py::VerifyDispatcher` (κ) and `merge_lane/speculation.py::ChainPlanner` / `Speculator` (λ) — the modules its features land in | **this PRD** owns the extraction; the follow-up's feature tasks depend on κ / λ by task id | blocked-on-consumer PRD (G1 resolution a: the seams are built for the façade's own callers today, the policy PRD is the second consumer) |
| `plans/deep-merge-ahead-prd.md` (7 of 13 landed) | refactors its as-built code | `build_chain`, `select_chain_depth`, `_deep_chain_placement`, `_land_chain_prefix`, `MergeDeepConfig.chain_cap` | deep-merge PRD owns semantics; **this PRD** moves them into `speculation.py` unchanged (λ); its 6 pending tasks are gated behind ζ2 by decision 10 and re-pointed at the new module by the throughput PRD | wired |
| `plans/merge-queue-reliability-prd.md` scope ε | supersedes | `test_merge_queue_reachback_patch_guard.py` allowlist | **this PRD**: α's ratchet subsumes it as a count; δ deletes the old guard when the count reaches 0 | wired |
| `plans/merge-queue-modularization-invariants-prd.md` | closes its out-of-scope line | re-export shim removal; `_TrainMergeHost`; `two-layer-model.md` symbol map | **this PRD** (ζ2, η update the map per move) | wired |
| `docs/prds/offline-deep-test-lane-worker.md` | consumes | `on_merge_landed(task_id, base_sha, head_sha)` hook | offline-lane PRD owns the notifiee; **this PRD** keeps signature + fail-open semantics stable | wired |
| `plans/concurrent-merge-verify-prd.md`, `plans/merge-throughput-multihost-verify-prd.md` (landed) | refactors call sites | `verify_runner.py::VerifyRunnerPool`, `HostAllocator`, `RemoteRunner.cancel_verify` | verify_runner owns the pool; κ moves the *callers* only | wired |
| workflow (`orchestrator/workflow.py`, no PRD) | consumes | reason-prefix routing → `MergeFailureKind` | **this PRD** (π) edits workflow's routing sites | queued |
| escalation server (`escalation/src/escalation/server.py`) | consumes | six names: `InFlightMergeRegistry`, `MergeOutcome`, `MergeRequest`, `QueuedBranch`, `WaiterRecord`, `coalesce_or_enqueue_merge_request`, `patch_content_contained`, `_resolve_dispatch_time_merge_base`, `retire_cancelled_merge_request` | **this PRD** (η) re-points the imports; names stay in the façade | queued |

No reciprocal-ownership ambiguity.

## Contract (B+H)

### Frozen public surface (behaviour preservation)

Byte-identical before and after every task:

- `MergeOutcome.status` literal set and field meanings; `MergeOutcome.reason` strings.
- All seventeen `*_REASON_PREFIX` / `*_REASON` constants (defined in `merge_gates.py`
  and `merge_queue.py`; workflow routes on ten of them by `startswith`).
- `EventType` emissions and their `data` keys.
- `snapshot()` keys — additive only.
- `merge_queue_store` journal schema; `landed_outbox` row schema.
- `on_merge_landed(task_id, base_sha, head_sha)` hook signature and fail-open semantics.
- The `SpeculativeMergeWorker(...)` constructor keyword set that `harness.py` passes
  (`git_ops`, queue, `speculation_depth`, `event_store`, `on_merge_landed`,
  `escalation_queue`, `train_callback_factory`, `merge_store`, `scheduler`, `mcp`,
  `usage_gate`, `cost_store`, `provenance_conflict_sink`) — the façade keeps it as
  `MergeLane(...)` with `SpeculativeMergeWorker` as an alias until η.
- The escalation server's nine imported names (table above).
- Ordering invariant: `main` advances strictly in submission order.
- Logger: the package logs under `orchestrator.merge_lane`; η adds a `caplog`
  compatibility note to the migrated tests, and greps dashboard/scripts/docs for
  the string `orchestrator.merge_queue` (the dashboard has its own unrelated
  `dashboard/data/merge_queue.py`).

### Public surface of the façade

`orchestrator/merge_lane/__init__.py` exports **only**: `MergeLane` (the worker),
`coalesce_or_enqueue_merge_request`, `enqueue_merge_request`,
`retire_cancelled_merge_request`, `_resolve_dispatch_time_merge_base` (renamed
`resolve_dispatch_time_merge_base`; alias kept until η), `patch_content_contained`,
the public value types (`MergeRequest`, `GroupMergeRequest`, `MergeOutcome`,
`MergeFailureKind`, `QueuedBranch`, `WaiterRecord`, `InFlightMergeRegistry`), the
seventeen reason constants, and the ports. Anything else imported from outside the
package is a ratchet violation.

### Ports (β)

```python
class VerifyPort(Protocol):
    async def run_scoped(self, spec: MergeVerifySpec, *, worktree: Path, env: Mapping[str, str]) -> VerifyResult: ...
    async def run_unscoped_typechecks(self, worktree: Path) -> VerifyResult: ...
    async def check_post_merge_pyright(self, worktree: Path, *, base_sha: str) -> GateVerdict: ...
    async def check_post_merge_equivalence(self, worktree: Path, *, merged_tip: str) -> GateVerdict: ...
    async def ensure_disk_space(self, worktree: Path) -> DiskGuardOutcome: ...
    async def cold_shadow(self, spec: MergeVerifySpec) -> VerifyResult | None: ...
    async def dry_run_unblock(self, request: MergeRequest, failure: VerifyResult) -> DryRunProposal | None: ...

class ClockPort(Protocol):
    def now(self) -> float: ...
    def newest_content_mtime(self, worktree: Path) -> float: ...
    async def sleep(self, secs: float) -> None: ...

class EscalationPort(Protocol):
    async def file(self, record: EscalationRecord) -> str: ...
```

The production implementations wrap today's module-level functions
(`run_scoped_verification`, `_run_unscoped_typechecks`, `_check_post_merge_pyright`,
`_check_post_merge_equivalence`, `_ensure_verify_disk_space`,
`_run_cold_shadow_verify`, `run_dry_run_unblock`) and `VerifyRunnerPool`; exact
signatures are settled in β against those functions' current parameters (tactical).
G3 note: `MergeVerifySpec` and `VerifyResult` exist (`verify_runner.py`,
`verify.py`); `GateVerdict` exists from the modularization PRD's λ; the result
types `DiskGuardOutcome`, `DryRunProposal` and `EscalationRecord` do **not** exist
today and are **defined by β** as frozen dataclasses wrapping what the seven
functions return now — no task may assume them before β lands.
`FakeVerifier` (scripted per branch tip: pass / fail-with-category / raise /
hang-until-released) and `FakeClock` live in `orchestrator/tests/_merge_lane_fakes.py`.

### Module ownership (Phase 3 target layout)

| Module | Owns (from the concern map) | Owns these shared attributes | Task |
|---|---|---|---|
| `merge_lane/worker.py` | composition root; `_merger_loop`, `_verifier_loop` as thin schedulers; `stop` | `_live_loops`, `_shutdown_signaled`, `_pending_get`, `_pending_verifier_get` | ο |
| `merge_lane/intake.py` | enqueue, coalesce, lane buffers, suffix conflict graph, bounce, `_drain_queue_into_lanes`, `_acquire_next_request` | `_lane_buffers`, `_suffix_tracker`, `_redispatch` (sole writer; others *request* redispatch via a method) | ι |
| `merge_lane/verify_dispatch.py` | `_run_post_merge_verify`, `_run_inflight_verify`, `_dispatch_item`, host quarantine (`_reprobe_quarantined_hosts` et al.), main-health probe | `_runner_unavailable`, `_runner_quarantine`, `_remerge_occurred` | κ |
| `merge_lane/speculation.py` | chain build/depth/placement, permits/leases, `_remerge`/`_void_and_remerge`, absorbing `merge_speculation_controller.py` | `_n_failed`, `_last_known_main_sha` (via the tracker), permit ledger | λ |
| `merge_lane/landing.py` | `_finalize_inflight`, CAS advance handling, `_land_chain_prefix`, outbox/journal, `_drift_base` transitions | `_inflight` (sole writer through `_inflight_append/_popleft/_clear`), `_drift_base` | μ |
| `merge_lane/worktrees.py` | merge-worktree lifecycle, owned-worktree ledger, reap | `_owned_merge_worktrees`, `_owned_merge_wt_keys` | ν |
| `merge_lane/halt.py` | `_WipHaltMixin` → `HaltState` object (composition, not inheritance); the two bypass reads (`_lane_halt`, `_operator_halt` read directly in `__init__`, `stop`, `_dispatch_item`, `_run_inflight_verify`) go through its accessors | `_lane_halt`, `_lane_halt_owner`, `_operator_halt` | θ |
| `merge_lane/telemetry.py` | `snapshot`, heartbeat/liveness audit, escalation emit helpers, contended-lease streak | `_contended_lease_*`, metrics | ξ |
| `merge_lane/types.py`, `ports.py`, and the moved satellites | value types, ports; gates, store, shadow, drift, liveness, disposition, ledger, skew, lanes, offline, warm pool, evidence, outbox, recovery, completion | — | ζ1, ζ2 |

Rules: a concern module never imports `worker.py`; `worker.py` composes the modules
and passes ports and the one registry each needs; no module-level mutable state; the
45 constructor-only attributes become fields of frozen config/dependency objects
handed to the modules; `__init__` shrinks to composition.

### Ceilings (asserted by σ, ratcheted from α)

| Measure | Ceiling at σ | Basis (G6) |
|---|---|---|
| Lines per file in `merge_lane/` | 1,500 | one default Read call for an agent is 2,000 lines; the cluster's healthiest module is 3,531 lines with max cognitive 48, so 1,500 is below what already works |
| Cognitive complexity per function | 40 | `verify_runner.py`, same authors and domain, tops at 48 with total 275 — 40 sits inside a demonstrated band |
| New functions (any task after α) | 15 | Sonar's published threshold; every satellite already satisfies it |
| Reach-back / function-local imports inside the package | 0 | structural: the reason they exist (string-path patches) is removed in Phase 1 |
| Re-export shims | 0 | same |
| String-path patches into lane internals from tests | 0 | Phase 1 replaces them with ports |
| Private-attribute reads from tests | 0 | same |
| Cluster-total cognitive complexity | ≤ baseline (2,132 + satellites) | moving code must not add branches |

## Boundary-test sketch (σ's spec)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Single branch lands through the façade with `FakeVerifier` scripted *pass* | real git fixture repo, one queued request | `MergeOutcome.status == 'done'`; `main` advanced to the merge commit; `merge_finalized` event carries the same `data` keys as at `45ebf0fbc0`; outbox row written before the CAS advance |
| 2 | Same, with the **real** `LocalRunner` on a tiny fixture project (`exercise_merge_verify`) | fixture with a trivially green test | lands; proves the port's production implementation is wired, not only the fake |
| 3 | Verify fails with a categorised failure | scripted fail | `status == 'blocked'`, `reason` starts with the frozen prefix, `MergeFailureKind` matches, workflow's routing decision identical to the pre-π decision (parametrised over all seventeen constants) |
| 4 | Two waiting singles coalesce into a train | two disjoint queued requests | `coalesce_or_enqueue_merge_request` returns the train; `merge_coalesced` event unchanged; both futures resolve |
| 5 | Chain invalidation → remerge | depth ≥ 1, head fails after follower dispatched | follower re-merged onto real `main`; lands; permit conservation audit reports zero violations |
| 6 | Waiter abandons mid-verify | sole waiter cancels | `DROPPED`; worktree ledger == disk; no `set_result` |
| 7 | Operator halt then unhalt mid-verify | verify running | `REQUEUED`; re-verifies after unhalt; halt state only reachable through `HaltState` accessors (grep in σ) |
| 8 | `stop()` mid-verify | in-flight verify + chain build | every future resolved `blocked` with `MERGE_WORKER_SHUTDOWN_REASON`; host leases released; no `_merge-*` worktree leaked |
| 9 | Runner unavailable → quarantine → reprobe | scripted `RunnerUnavailable` | item re-dispatched locally; host quarantined then re-admitted; not a chain failure |
| 10 | Package isolation | package landed | no function-local import inside `merge_lane/`; no `orchestrator.merge_queue` import anywhere in the repo; each module's imports are downward only |
| 11 | Ceilings | σ runs `scripts/merge_lane_metrics.py` | every row of § Ceilings holds; ratchet test green |
| 12 | Escalation server contract | server imports the nine names from the façade | `merge_request` / `merge_cancel` / `get_merge_queue` MCP paths exercised against the façade; responses unchanged |

## Decomposition plan

All tasks `planning_mode=True`, `task_kind='normal'` unless stated. Modules are
`orchestrator/src/orchestrator/merge_lane/**`, `orchestrator/tests/**` and the
files named. Priorities: high for the critical chain (α, ζ1, β, δ, ζ2, η, κ, λ, σ);
medium otherwise; ρ low. Sizing per the overlay bands: each γ leaf is one group of
`plans/merge-lane-quality-prd.test-groups.txt` (12–14 files, ~17k test lines; the
edit touches a fraction of those lines).

**Decompose-time gating step (decision 10):** after filing, query every non-terminal
task whose `metadata.files` or `files_hint` intersects Appendix A and
`add_dependency(existing, depends_on=ζ2)`. Also wire ζ1 behind the still-non-terminal
members of the pre-condition branch list.

- **α — Metrics script + ratchet test + committed baseline.** [high]
  `scripts/merge_lane_metrics.py` (stdlib `ast` + `complexipy` JSON) computes, for
  Appendix A: lines per file, cognitive complexity per function, cluster totals,
  function-local imports inside the cluster, re-export shim names, distinct
  `orchestrator.merge_queue.*` / `orchestrator.merge_lane.*` patch targets in
  `orchestrator/tests`, and private-attribute reads from tests (AST, not regex —
  reuse `test_merge_queue_reachback_patch_guard.py`'s helpers where they fit).
  `orchestrator/tests/merge_lane_ratchet_baseline.json` committed;
  `orchestrator/tests/test_merge_lane_ratchet.py` fails if any measure exceeds its
  baseline, any new file exceeds 1,500 lines, or any new function exceeds 15;
  marked with `WHOLE_TREE_SCAN_TEST_TIMEOUT`. A file the script cannot parse or a
  tool it cannot run is a hard failure, never a skipped measure (INV-11). Adds
  `radon`, `complexipy` to the dev
  group + `uv.lock`. Modules: scripts, orchestrator/tests, orchestrator/pyproject.toml,
  uv.lock. **Signal:** `python scripts/merge_lane_metrics.py --report` prints the
  table; the ratchet test is green on main; a seeded fixture (a synthetic +1 on each
  measure) turns it red in a self-test — INV-10 tier 1. Unlocks ζ1.
- **ζ1 — Package skeleton + ports.** [high] `orchestrator/merge_lane/__init__.py`
  façade re-exporting today's names *from* `merge_queue.py` (temporary direction),
  `merge_lane/ports.py` with `VerifyPort`, `ClockPort`, `EscalationPort` and the
  production adapters wrapping the current module-level functions. No behaviour
  change. **Signal:** `from orchestrator.merge_lane import MergeLane` works;
  `MergeLane is SpeculativeMergeWorker`; ratchet green. Unlocks β.
- **β — Worker accepts injected ports; fakes; conftest.** [high]
  `SpeculativeMergeWorker.__init__` gains `verify: VerifyPort | None`,
  `clock: ClockPort | None`, `escalation: EscalationPort | None` defaulting to the
  production adapters; every call site of the seven wrapped functions inside the
  worker goes through `self._verify` / `self._clock`. `orchestrator/tests/
  _merge_lane_fakes.py` with `FakeVerifier`, `FakeClock`, `FakeEscalations`;
  conftest replaces the autouse `_mock_merge_queue_verification` patch with an
  autouse fixture that injects `FakeVerifier(default=pass)` — same default
  behaviour, no dotted path. `test_merge_verify_mock_autouse.py` updated to pin the
  new contract. **Signal:** full merge test set green with the autouse patch gone;
  ratchet's patch-target measure unchanged or lower. Unlocks γ1–γ10.
- **γ1 … γ10 — Migrate one test group each (see sidecar).** [medium; parallel]
  For every file in the group: dotted-path patches → fake injection; private-
  attribute assertions → `MergeOutcome` / events / `snapshot()` / git state;
  timing waits on `asyncio.sleep(x)` → `FakeClock` where the wait is the subject;
  tests that only pin implementation and are covered elsewhere are deleted (listed
  in the commit body). Lower the group's contribution in the baseline JSON in the
  same commit. Files: the group's list. **Signal:** for the group's files the
  ratchet reports 0 patch targets and 0 private reads; suite green. Unlocks δ.
- **δ — Discard the serial worker; retire the old guard.** [high] Delete
  `orchestrator/tests/_serial_merge_worker.py` and re-home its remaining
  constructions (train path, workflow e2e) onto `MergeLane` + fakes; update
  `test_merge_worker_retired.py`; delete `test_merge_queue_reachback_patch_guard.py`
  once the ratchet's patch-target measure is 0. **Signal:** `git grep
  _serial_merge_worker` empty; ratchet patch-target and private-read measures = 0;
  suite green. Unlocks ζ2, ε.
- **ε — Mutation-score baseline (measurement, no threshold).** [low] Add `mutmut` to
  the dev group; run it over `merge_gates.py`, `merge_types.py` and the four
  worst-churn methods' enclosing module against the migrated suite; commit
  `plans/merge-lane-quality-prd.mutation-baseline.md`. **Signal:** the report exists
  with per-module killed/survived counts. Consumer: a follow-up ratchet decision
  (§ Open questions).
- **ζ2 — Move the lane into the package; delete reach-backs and shims.** [high]
  `git mv merge_queue.py merge_lane/worker.py`; `git mv` each satellite to its
  package name (table above; exact grouping tactical); replace every function-local
  reach-back import with a direct import of the moved symbol; delete every `# noqa:
  F401 re-export shim` block; `orchestrator/merge_queue.py` becomes a one-line alias
  module (`from orchestrator.merge_lane import *`) with a ratchet measure on external
  importers of it; update `skills/merge-queue/references/two-layer-model.md`'s symbol
  map. Rewrite the baseline JSON keys for moved paths (totals unchanged).
  **Signal:** ratchet reach-back = 0, re-export = 0; suite green; importers
  unchanged. Unlocks η and (by decision 10) every gated external task.
- **η — Migrate external importers; delete `merge_queue.py`.** [high] harness,
  workflow, escalation server, scripts, remaining tests → `orchestrator.merge_lane`;
  delete the alias module; grep dashboard/scripts/docs/systemd for the string
  `orchestrator.merge_queue` and update; logger name `orchestrator.merge_lane`.
  **Signal:** `git grep 'orchestrator.merge_queue'` returns only the decisions doc
  and this PRD; `python -c 'import orchestrator.merge_queue'` fails; suite green.
  Unlocks θ.
- **θ — `HaltState` object replaces `_WipHaltMixin`.** [medium] Composition, not
  inheritance; the four direct-read bypasses go through accessors. **Signal:**
  boundary row 7; `grep -n '_lane_halt\b\|_operator_halt\b' merge_lane/worker.py`
  shows only the composition-root assignment. Unlocks ι.
- **ι — Extract `intake.py`.** [medium] The 20 enqueue/coalesce methods and the
  suffix tracker; `coalesce_or_enqueue_merge_request` (103) decomposed into named
  steps ≤ 40. **Signal:** boundary row 4; `worker.py` shrinks by the moved lines
  (ratchet baseline lowered); no function > 40 in `intake.py`. Unlocks κ.
- **κ — Extract `verify_dispatch.py` (`VerifyDispatcher`).** [high]
  `_run_post_merge_verify` (175), `_run_inflight_verify` (101), `_dispatch_item`
  (49), host quarantine, main-health probe, behind `VerifyPort`. **Signal:** boundary
  rows 2, 3, 9; no function > 40 in the module. Consumers: `worker.py` (ο) today;
  the dispatch-policy follow-up PRD. Unlocks λ.
- **λ — Extract `speculation.py` (`ChainPlanner`, `Speculator`).** [high] Chain
  build/depth/placement/prefix-landing decision, permits, remerge; absorbs
  `merge_speculation_controller.py`; `_merger_loop` body moves here, the loop shell
  stays in `worker.py`. **Signal:** boundary row 5; deep-merge-ahead's existing
  tests green against the new module; no function > 40. Consumers: `worker.py` (ο)
  today; the speculation-policy follow-up PRD. Unlocks μ.
- **μ — Extract `landing.py` (`Lander`); decompose `advance_main` in place.**
  [medium] `_finalize_inflight` (102) into named steps; outbox-then-CAS ordering
  preserved; `GitOps.advance_main` (113) split into named private steps in
  `git_ops.py` with an unchanged signature and `AdvanceOutcome`. **Signal:** boundary
  rows 1, 8; `advance_main` ≤ 40. Unlocks ν.
- **ν — Extract `worktrees.py`.** [medium] Lifecycle, ledger, reap. **Signal:**
  boundary row 6; worktree-conservation audit unchanged. Unlocks ξ.
- **ξ — Extract `telemetry.py`.** [medium] `snapshot` (58), heartbeat/audit,
  escalation emit. **Signal:** `snapshot()` keys byte-identical (pinned by a
  key-set test); dashboard renders unchanged. Unlocks ο.
- **ο — Residual `worker.py`: composition root + two loops + `stop`.** [high]
  `__init__` (661) becomes composition of frozen config/dependency objects;
  `_verifier_loop` (245) and `stop` (109) decomposed to ≤ 40; the 45 constructor-only
  attributes become fields on those objects. **Signal:** `worker.py` ≤ 1,500 lines;
  no function > 40; boundary row 8. Unlocks π.
- **π — `MergeFailureKind`: route on structured data.** [medium] Enum on
  `MergeOutcome`; one prefix → one kind, pinned; workflow's ten `startswith` routing
  sites switch to the kind; strings untouched. Modules add `workflow.py`. **Signal:**
  boundary row 3 (parametrised routing-equivalence test); `git grep
  "reason.startswith" orchestrator/src/orchestrator/workflow.py` empty. Unlocks ρ.
- **ρ — Relocate archaeology prose.** [low; `metadata.complexity='simple'`] For each
  comment block citing a task/esc id in `merge_lane/`: move the rationale to
  `docs/merge-lane/decisions.md` (dated entry, id) and leave a one-line pointer.
  **Signal:** the metrics script's new `archaeology_blocks` measure (comment blocks
  of more than one line that cite a task/esc id) reads 0 for `merge_lane/`; the
  decisions doc exists with one entry per relocated block; INV-9 pointers only.
  No percentage target — the count is the contract. Unlocks σ.
- **σ — Integration gate (B+H leaf).** [high] One test module drives boundary rows
  1–12 through the façade with real git fixtures, `FakeVerifier` and (row 2) the
  real `LocalRunner`; asserts § Ceilings via the metrics script. Task text carries
  the rollback runbook line (pre-conditions). **Signal:** the gate is green in CI;
  `scripts/merge_lane_metrics.py --report` shows every ceiling met. Unlocks τ.
- **τ — Deterministic deploy.** `task_kind='deterministic'`,
  `before_done={script: 'scripts/restart-all-orchestrators.sh', args: [],
  timeout_secs: 900, target_unit: 'orchestrator-dark-factory.service'}`,
  `always_escalates=false`. **Signal:** `done_provenance kind=
  'deterministic-deploy-scheduled'`; journal shows every running
  `orchestrator-*.service` restarted onto a `main` containing σ. Depends on σ.

## Out of scope

- Any behaviour change to merging, verification, speculation or routing — the
  companion `plans/merge-lane-throughput-prd.md` owns functional work.
- Splitting `git_ops.py` beyond `advance_main` (follow-up PRD: git engine).
- `workflow.py` and `harness.py` structure beyond the import and routing edits named.
- Dashboard UI; reify-side code; fused-memory.
- Mutation-score *thresholds* (ε measures; a later decision ratchets).
- Removing `_TrainMergeHost` / train semantics (moved, not changed).

## Open questions (tactical)

1. **Exact satellite grouping inside the package** (e.g. whether `drift`/`shadow`/
   `skew` fold into one `observers.py`). Suggested: keep 1:1 moves in ζ2; fold only
   if a file lands under ~300 lines. Decide in ζ2.
2. **Port signatures.** β settles the exact parameter lists against the seven wrapped
   functions; the shapes above are the design, the parameters are tactical.
3. **Baseline JSON format** (per-path map vs list). Suggested: `{"files": {path:
   {lines, prose_lines}}, "functions": {qualname: cognitive}, "totals": {...},
   "tests": {...}}`. Decide in α.
4. **Mutation ratchet.** Whether ε's report becomes a ratchet measure and at what
   granularity. Decide after ε lands, outside this batch.
5. **`_merge_queue_harness.py`** (test helper): fold into `_merge_lane_fakes.py` or
   keep. Decide in β.
6. **Order of θ–ξ** if a task's extraction turns out to need another's module first.
   The chain above is the default; a task may swap with its neighbour with an
   `amend:` note, never skip.

## Appendix A — cluster paths (for the ratchet and the gating step)

```
orchestrator/src/orchestrator/merge_queue.py
orchestrator/src/orchestrator/merge_gates.py
orchestrator/src/orchestrator/merge_types.py
orchestrator/src/orchestrator/merge_shadow.py
orchestrator/src/orchestrator/merge_liveness.py
orchestrator/src/orchestrator/merge_disposition.py
orchestrator/src/orchestrator/merge_queue_store.py
orchestrator/src/orchestrator/merge_completion.py
orchestrator/src/orchestrator/merge_drift.py
orchestrator/src/orchestrator/merge_speculation_controller.py
orchestrator/src/orchestrator/merge_request_ledger.py
orchestrator/src/orchestrator/merge_skew_tripwire.py
orchestrator/src/orchestrator/lane_lifecycle.py
orchestrator/src/orchestrator/offline_lane.py
orchestrator/src/orchestrator/warm_lane_pool.py
orchestrator/src/orchestrator/landing_evidence.py
orchestrator/src/orchestrator/landed_outbox.py
orchestrator/src/orchestrator/recover_main.py
orchestrator/src/orchestrator/merge_lane/**   (after ζ1)
orchestrator/tests/_serial_merge_worker.py
orchestrator/tests/_merge_queue_harness.py
orchestrator/tests/conftest.py
```
`git_ops.py` is measured (lines, cognitive) but not gated by the file-size ceiling
(decision 9); `advance_main` is gated per function by μ.

---

## Corrections (2026-09-04 — cross-PRD coherence check after both decomposes)

Dated corrections to the text above; the original is left in place as provenance.
Filed batch: tasks **5021–5049** (α=5021, ζ1=5022, β=5023, γ1–γ10=5024–5033,
δ=5034, ε=5035, ζ2=5036, η=5037, θ=5038, ι=5039, κ=5040, λ=5041, μ=5042, ν=5043,
ξ=5044, ο=5045, π=5046, ρ=5047, σ=5048, τ=5049); manifest `d9ce99a8bb`, corrected
`13ff3d630f`.

1. **κ must NOT delete the `is_flock_contention_failure` branch.** The Cross-PRD
   row for the throughput PRD said κ would delete it once that PRD's task C made it
   unreachable. Task C was cancelled by Leo (2026-09-03, task 5052): the laptop-side
   lock is per project root, so the branch is a *within-project* orphan detector
   (tasks 2306/2307) and stays live. κ moves it behaviour-preserving with the rest
   of `_run_post_merge_verify`. Task 5040's details carry the same note.
2. **The throughput PRD does not depend on κ/λ and runs in parallel** — as the
   header already says; the follow-up dispatch/speculation-policy PRD is the one
   that will.
3. **Counts corrected by the decompose walk** (measured at `4811d62883`): workflow
   routes on **nine** `reason.startswith` sites over **eight** distinct prefixes,
   not "ten of seventeen"; the escalation server imports **nine** names (the table
   said six and then listed nine); there are 18 reason constants (17 public plus
   the private `_MERGE_CANCEL_RETIRE_REASON`).
4. **Decision 10 gating as executed:** 120 existing non-terminal tasks gained a
   dependency on ζ2 (5036) — 110 by source files, 11 by `orchestrator/tests/
   conftest.py` alone; 4755 was excluded per instruction although it declares
   conftest.py, so β and 4755 will contend on that file (accepted). **Exemption:**
   3188 (deep-merge telemetry) was un-gated by the authoring session because the
   throughput PRD's task F depends on it, which would have put that PRD's canary
   transitively behind this whole batch; 3188 is ready now and lands before ζ1.
   **Process gap:** tasks filed *after* decompose are not gated automatically —
   5064, 5070 and 5092 were gated by hand on 2026-09-04; 5063 (high-priority lane
   bug) was deliberately left ungated so it lands before ζ1. Until ζ2 lands, a
   new cluster-touching task needs the same `add_dependency(<id>, 5036)` by hand.
5. **Main is red fleet-wide at authoring+1 day** (task 5088, commit `731a0bafa9`):
   α's landing will hit the main-health path until 5088 lands. Not a dependency —
   the merge queue's main-health probe classifies it — but the first landing of
   this batch is gated on it in practice.
6. **τ (5049) passes `--drain`** (Leo, 2026-09-04): `before_done.args = ["--drain"]`,
   `timeout_secs` 900 → 5400 (seven units × up to 600s force-fire + verify grace).
   The drain gate defers a unit whose merge-idle heartbeat says mid-merge, then
   forces; a stale/absent heartbeat gets a short grace and proceeds. It protects
   in-flight merges only; in-flight task agents are still soft-cancelled.
7. **σ gains a runtime ceiling** (Leo, 2026-09-06): the ratchet module must complete
   in ≤ 60s wall on an idle box. α measured 147–251s per run (esc-5021-1); a
   single full measurement is 72.7s, of which 13s is 22 separate complexipy
   subprocesses, and the module then re-walks the 559-file test tree twice more
   and recomputes radon MI five times. The fix is one measurement per session and
   one complexipy invocation, with the CLI tests on a fixture tree — never a
   "skip when nothing changed" gate (INV-10/INV-11) and never a narrower domain.
   Until σ lands the per-verify-leg tax is accepted as measured. Task 5048's
   details carry the constraints.
