# Capability manifest: merge pre-verify screen and main-red hold

**PRD:** `plans/merge-pre-verify-screen-prd.md` · **Sidecar:** `plans/merge-pre-verify-screen-prd.capability-manifest.yaml`
**Code anchors** verified against main `7964b7850d` (2026-10-31). Cited by symbol.

Each leaf's signal is bound to the capabilities it asserts. Every binding below is PASS; none is declared-only, test-only, a downstream producer, or an absent producer.

**Who checks a binding.**
- **Mechanical.** A `grep` or `path` check in the sidecar is copied into the producer's `metadata.delivered_checks` and gates that producer's dependents. Each pattern or path was checked to be **absent on main today**, so it flips from red to green when its producer lands.
- **Manual.** Leaves with no in-batch dependent (δ, ε, ζ) carry manual checks only.

**G5 note (B + H without a single integration-gate leaf).** Each leaf owns both sides of the seams it touches, and its signal names its own boundary rows (PRD §7). So no cross-module integration step is left for a separate task to starve on:

| Leaf | Seams it owns |
|---|---|
| α | the lane screen and the workflow review-category mapping |
| β | the task-role plan; it relies on the existing contagion guard |
| γ1 | the verdict, the owner hook in `Harness._on_escalation_resolved`, and the wire state |

`signal_leaves` = α, β, γ1, the three deliverables of the PRD's goal.

## γ0, the shared node-id runner

| Capability | Binding | Verdict |
|---|---|---|
| The discriminator's group → command → junit body exists and is wired | `verify.py::confirm_isolated_rerun_verdict` (merge-gate and main-probe call sites), `::_merge_gate_isolated_rerun`, `::_group_node_ids_by_subproject` | PASS |
| junit yields failing and never-ran vectors | `verify.py::_extract_test_ids_from_junit` | PASS |
| A per-module junit path exists, to extend with a per-group infix | `verify.py::_prepare_junit_report_path` | PASS |
| `run_node_ids` is delivered | built here; mechanical: `grep 'async def run_node_ids\('` in `verify.py`, absent today | PASS |

## α, the manifest and the per-item lane screen

| Capability | Binding | Verdict |
|---|---|---|
| A pre-verify bounce template | `merge_lane/worker.py::classify_and_merge` with `gates.py::_check_plan_targets_in_tree`; `merge_lane/types.py::OutcomeKind.dropped_plan_targets` | PASS |
| The declared-train screen site | `merge_lane/worker.py::_do_train_merge`, after `merge_to_main` | PASS |
| Persistent reset in place, keeping the venv | `git_ops.py::GitOps.reset_persistent_merge_worktree`; the clean builder's `also_retain`, as used by `_reset_warm_lane` | PASS |
| Lanes survive the reapers | `git_ops.py::PROTECTED_PREFIXES`; `orchestrator/scripts/warm-lane/lib_lane_state.sh::lane_protect_glob`; names outside the `_merge-*` band that `GitOps.prune_stale_merge_worktrees` prunes | PASS |
| Review category from structured data | `workflow.py::TaskWorkflow._submit_to_merge_queue` (keys on substring today; α edits it) | PASS |
| Dashboard outcome family | `dashboard/src/dashboard/data/outcome_colors.py::classify_outcome` | PASS |
| Reloadable config | `config.py::RELOADABLE_FIELDS` via `_submodel_leaf_paths` | PASS |
| The node-id runner | producer γ0 (upstream, wired dependency) | PASS |
| ≤ 20 s median per-item screen (G6 numeric) | basis: `scripts` group 8.3 s warm (L5), `orchestrator` group ~4 s (L3), run concurrently, plus a reset; α re-measures through the runner | PASS (achievable) |
| Delivered | mechanical: path `scripts/merge_screen.yaml`; path `orchestrator/src/orchestrator/merge_lane/screen.py`; grep `screen_failed = 'screen_failed'` in `merge_lane/types.py`; all absent today | PASS |

## β, the task-role screen

| Capability | Binding | Verdict |
|---|---|---|
| The plan seam reads the tree | `verify_plan.py::derive_verify_plan(…, worktree_reader, role)` | PASS |
| junit only for structured pytest runs | `verify.py::_with_junitxml_str`; `VerifyResult.failing_test_ids` from junit | PASS |
| The contagion guard attributes a red main | `workflow.py` → `verify.py::verify_failure_is_preexisting_on_main` (baseline fork when ids are present) → `TaskWorkflow._block_on_inherited_break` | PASS |
| The manifest | producer α (upstream) | PASS |
| Delivered | mechanical: grep `merge_screen` in `verify_plan.py`, absent today | PASS |

## γ1, the main-red verdict and the F-screen hold

| Capability | Binding | Verdict |
|---|---|---|
| One shared outcome builder on both probe modes | `merge_lane/worker.py::_build_main_health_outcome` | PASS |
| An owner path to arm after | `SpeculativeMergeWorker._auto_heal_main_health_deferred` → `::_escalate_owned_main_break` (γ1 makes it return the id) → `_file_main_health_escalation` (returns `Optional[str]`) | PASS |
| An owner-resolution hook | `harness.py::Harness._on_escalation_resolved` | PASS |
| Hold carriage | `_requeue_request` (single chokepoint); the operator-halt branch's `REQUEUED_PREDISPATCH`; `SpeculationController.on_transfer_terminal` | PASS |
| Held predicate and wake | `_pop_next_pickable`, `chain_snapshot` (twin test updated), `_signal_resume`, `_acquire_next_request` | PASS |
| Owner annotation | `escalation/queue.py::EscalationQueue.amend` | PASS |
| Operator surfaces | `harness.py::Harness.get_merge_halt_status`; `_REGISTRY_STATE_TO_WIRE`, overridden by the held mark | PASS |
| Runner and screen | producers γ0 and α (upstream) | PASS |
| G6: "every merger-started verify on the sha after arming is F-passing" | every merger start passes the screen (α); items already dispatched or merged ahead are excluded, as the PRD states | PASS |
| Delivered | mechanical: path `orchestrator/src/orchestrator/merge_lane/main_red_verdict.py`; grep `class VerdictBook`; both absent today | PASS |

## δ, the chain-tip screen

| Capability | Binding | Verdict |
|---|---|---|
| Chain build site and truncation literal | `merge_lane/worker.py::SpeculativeMergeWorker._deep_chain_placement`, `::build_chain`; `merge_lane/types.py::ChainTruncationReason` | PASS |
| A working chain walk; a truncation-reason event field | producers 7720 and 7259 (external to this batch, wired dependencies) | PASS |
| The `_screen-chain` lane | producer α (upstream) | PASS |
| Delivered | manual: no in-batch dependent | PASS |

## ε, `get_changed_line_ranges`

| Capability | Binding | Verdict |
|---|---|---|
| Method, rc and canonical parser | `git_ops.py::GitOps.get_changed_line_ranges`, `git_ops.py::_run`, `merge_lane/types.py::QueuedBranch.parse` | PASS |
| Ratchet pin of 0 | `orchestrator/tests/merge_lane_ratchet_baseline.json`; extract a helper | PASS |
| Delivered | manual | PASS |

## ζ, the readout

| Capability | Binding | Verdict |
|---|---|---|
| The events it reads | `merge_attempt` outcome `screen_failed` and the `merge_screen` event (producer α); the `main_red_verdict` and `merge_held` events (producer γ1) | PASS |
| The dated hold | `metadata.milestone` delayed (`docs/task-authoring.md`) | PASS |
| Delivered | manual | PASS |
