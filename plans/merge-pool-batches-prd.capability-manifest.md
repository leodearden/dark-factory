# Capability manifest: `plans/merge-pool-batches-prd.md`

Pointers were checked against main on 2026-11-20. A `grep:` pointer names a file, or a name the module at
that path defines, that exists on main today. "built here" means the leaf itself delivers the capability;
its delivered check, in the YAML twin, is what a dependent waits for. "producer:task-X upstream" names the
task of this PRD, or the existing task, that delivers it earlier in the dependency order.

Signal leaves: ζ, η, θ, ι2, ι3.

## ι0. Merge lane: the un-chained mark survives an orchestrator restart (interim)

| Capability | Evidence | Verdict |
|---|---|---|
| persistent-set-substrate | `grep:orchestrator/src/orchestrator/guard_state.py::PersistentSet` wired: the worker already backs a set with it (`grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.__init__`) | PASS |
| mark-survives-restart | built here: the un-chained mark is reloaded by a worker built after a restart | PASS |

## α1. Merge lane: record the tested tree on each in-flight entry (root, base, members)

| Capability | Evidence | Verdict |
|---|---|---|
| entry-slot-publishes-once | `grep:orchestrator/src/orchestrator/merge_lane/types.py::InflightEntrySlot.publish_verify_base` wired: the precedent for a fact published once on an in-flight entry | PASS |
| base-is-an-entry-commit | `grep:orchestrator/src/orchestrator/merge_lane/remerge_base.py::entry_bases` wired: a base can be an in-flight entry's tip or its owner's own commit | PASS |
| tested-tree-on-the-entry | built here: merge_lane/tested_tree.py and InflightEntry.tree | PASS |

## α2. Merge lane: the verdict's facts reach the lane; one merge_observation row per verdict

| Capability | Evidence | Verdict |
|---|---|---|
| final-result-in-one-place | `grep:orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::_PostMergeVerify.run` wired: every path forms its final result here, after the cross-check | PASS |
| verdict-known-before-links-release | `grep:orchestrator/src/orchestrator/merge_lane/inflight_verify.py::InflightVerifyRun.run` wired: it takes the verdict before it releases the chain's resources | PASS |
| failing-ids-readable | `grep:orchestrator/src/orchestrator/verify.py::_extract_failing_test_ids` wired: the flake gate reads a red result's failing ids with it | PASS |
| verdict-facts-wire-types | built here: shared/merge_evidence.py | PASS |
| merge-observation-row | built here: the merge_observation event type, written at the verdict | PASS |

## σ1. Culprit scan engine: run a red batch's failing tests on the root, the base, the tip and each member alone

| Capability | Evidence | Verdict |
|---|---|---|
| node-id-run | `grep:orchestrator/src/orchestrator/verify.py` built here as a production path: run_node_ids exists (task 7918) and has no production caller; this leaf is its first | PASS |
| worktree-at-any-commit | `grep:orchestrator/src/orchestrator/verify.py::_provisioned_ephemeral_worktree` wired: it opens GitOps.ephemeral_worktree at a commit and provisions it | PASS |
| full-module-set | `grep:orchestrator/src/orchestrator/config.py::OrchestratorConfig.module_configs_or_empty` wired: the project's module registry | PASS |
| scan-engine | built here: merge_lane/culprit_scan.py with its command | PASS |
| scan-worktree-kind | built here: WorktreeKind.CULPRIT_SCAN, protected and reclaimable | PASS |

## ρ. Merge lane: batch state that survives a restart (alone with its red set, cleared, outcomes); the un-chained mark becomes "alone"

| Capability | Evidence | Verdict |
|---|---|---|
| mark-readers | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.chain_snapshot` wired: one of the readers of the mark that this leaf re-points | PASS |
| merger-removes-a-waiting-request | `grep:orchestrator/src/orchestrator/merge_lane/intake_buffers.py::MergeIntake.remove_buffered` wired: the merger-owned removal a later conviction uses | PASS |
| rows-read-back-at-start | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.seed_chain_policy` wired: the precedent for rebuilding lane state from event rows at start | PASS |
| batch-state | built here: merge_lane/batch_state.py with its rows, checkpoint and rebuild | PASS |
| merge-evidence-config | built here: orchestrator/merge_evidence_config.py | PASS |
| mark-removed | built here: no reader of the un-chained mark remains | PASS |

## σ2. Merge lane: scan a red batch and act on the verdict (convict, clear, release, fallback); dark_factory on scan: auto

| Capability | Evidence | Verdict |
|---|---|---|
| scan-engine-upstream | producer:task-σ1 upstream: culprit_scan.run_scan and attribute | PASS |
| batch-state-upstream | producer:task-ρ upstream: BatchState.begin_scan, apply_scan and pending convictions | PASS |
| red-single-outcome-shape | `grep:orchestrator/src/orchestrator/merge_lane/verify_failure_outcome.py::red_verify_outcome` wired: it builds the blocked outcome a red single gets, which a conviction reuses | PASS |
| detached-task-off-the-verify-path | `grep:orchestrator/src/orchestrator/merge_lane/main_health_probe.py::spawn_main_health_probe` wired: the precedent for a detached task registered for shutdown, and the probe spawned when the root fails | PASS |
| merge-scan-row | built here: the merge_scan event type, written when a scan is applied or abandoned | PASS |
| dark-factory-scans | built here: dark-factory-orchestrator.yaml sets merge_evidence.scan | PASS |

## β1. Batch cost rule and its fitted rates, in shadow; scan: auto

| Capability | Evidence | Verdict |
|---|---|---|
| rate-rows-upstream | producer:task-ρ and producer:task-σ2 upstream: the outcome rows and merge_scan rows the rates are fitted from | PASS |
| batch-rule | built here: merge_lane/batch_rule.py, equal to the reference computation | PASS |
| rule-rates | built here: merge_lane/rule_rates.py | PASS |

## σ3. Culprit scan: run a red lint or type-check leg alone on each tree

| Capability | Evidence | Verdict |
|---|---|---|
| scan-engine-upstream | producer:task-σ1 upstream: the engine a leg check is added to | PASS |
| leg-run-alone | built here: one module's lint or type-check command run alone on a tree | PASS |

## β2. Merge lane: the cost rule sizes and fills batches; cleared requests ride; the depth table is removed

| Capability | Evidence | Verdict |
|---|---|---|
| placement-site | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._deep_chain_placement` wired: where a batch's size is chosen today | PASS |
| rule-upstream | producer:task-β1 upstream: plan_batch and the fitted rates | PASS |
| fast-landing-walk | producer:task-10998 upstream: landing a batch of 12 without a type check per member | PASS |
| depth-table-removed | built here: merge_lane/chain_policy.py is deleted | PASS |
| cost-rule-basis | built here: chain_depth_reason.basis is cost_rule | PASS |

## ε. Readout for the culprit scan and the batch rule (scripts/merge_evidence_readout.py)

| Capability | Evidence | Verdict |
|---|---|---|
| rows-upstream | producer:task-α2, producer:task-ρ, producer:task-σ2 and producer:task-β1 upstream: the rows each section reads; a section with none prints "no <kind> rows yet" | PASS |
| readout | built here: scripts/merge_evidence_readout.py | PASS |

## ζ. Integration gate: the merge-batches boundary rows against the real lane modules

| Capability | Evidence | Verdict |
|---|---|---|
| everything-upstream | producer:task-β2, producer:task-σ3 and producer:task-ε upstream: the lane the boundary rows drive | PASS |
| boundary-rows | built here: section 7 rows 1 to 49 against the real lane modules | PASS |

## η. Human gate: design phase 2 of the merge pool batches (what an entry is built on, the statistical fallback, larger caps)

| Capability | Evidence | Verdict |
|---|---|---|
| phase-2-design | a ruling this gate makes with Leo, from the readout of producer:task-ε | PASS |

## θ. Human gate: revise the merge-chain visibility PRD for the scan and the batch rule (tasks 10759, 10761, 10762 wait)

| Capability | Evidence | Verdict |
|---|---|---|
| visibility-prd-revised | `grep:plans/merge-chain-visibility-prd.md` a revision this gate makes with Leo | PASS |

## ι1. Design invariant INV-15 scheduling-state-survives-restart joins the checked list

| Capability | Evidence | Verdict |
|---|---|---|
| invariant-sites | `grep:tests/repo_guards/test_design_invariants_consistency.py::PINNED_SITES` wired: the sites that must agree when an invariant is added | PASS |
| inv-15 | built here: INV-15 scheduling-state-survives-restart on the checked list | PASS |

## κ. Merge lane: the bounce and retry budgets, the recent-verify window and an operator's halt survive a restart

| Capability | Evidence | Verdict |
|---|---|---|
| bounce-registry | `grep:orchestrator/src/orchestrator/merge_lane/intake_buffers.py::MergeIntake.bounce_registry` wired: the lane reads the conflict-bounce count (merge_lane/types.py MergeBounceRegistry) this leaf preserves | PASS |
| operator-halt | `grep:orchestrator/src/orchestrator/merge_lane/halt.py::HaltState` wired: holds the operator halt this leaf preserves | PASS |
| budgets-survive-restart | built here: the bounce and retry budgets, the recent-verify window and an operator's halt are reloaded after a restart | PASS |

## ι2. Merge lane: a register of its scheduling state and a guard that fails on undeclared state

| Capability | Evidence | Verdict |
|---|---|---|
| ratification-table | `grep:docs/legibility/exception-ratifications.yaml` the table a dropped entry's ratification row goes in | PASS |
| governed-exceptions | `grep:shared/src/shared/governed_exceptions.py` the declaration INV-12 checks | PASS |
| inv-15-upstream | producer:task-ι1 upstream: the invariant the register answers | PASS |
| restart-state-register | built here: merge_lane/restart_state.py and its guard | PASS |

## ι3. Human gate: design the scheduler's register of restart-surviving state (INV-15)

| Capability | Evidence | Verdict |
|---|---|---|
| scheduler-register-design | a ruling this gate makes with Leo, with the register of producer:task-ι2 as the pattern | PASS |
