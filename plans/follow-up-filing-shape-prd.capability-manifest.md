# Capability manifest — `plans/follow-up-filing-shape-prd.md`

Per-leaf capability → evidence bindings (G3 + G6 mechanized). Machine-readable twin:
`plans/follow-up-filing-shape-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).
Verified against main `55b50a6212` (2026-11-04): every `expect: present` pattern below is absent in its
`paths` today and every `expect: absent` pattern is present, so each mechanical check gates something.
Anchors are symbols; nothing here is a line pin.

## Substrate (G3 re-check, symbol by symbol on `55b50a6212`)

Present and wired: `roles.py::_FOLLOWUP_FILING_INSTRUCTIONS`, `roles.py::submit_only_instructions`,
`review_suggestions/curator_intake.py` (`review_suggestions` spawn context),
`TaskInterceptor.{submit_task, _check_escalation_idempotency, _curator_worker,
_process_add_tickets_batch_prepared, _execute_combine, _append_combine_audit, _dispatch_ticket_decision,
_apply_status_transition, add_dependency, remove_dependency, _write_lock}`, `batch_target_index` in the
interceptor dispatch, `task_curator.py::{_BATCH_CONTRACT, _parse_decision_dict, _parse_batch_decisions,
_to_pool_entry, _sibling_stream, curate_batch_prepared, _CURATOR_PROMPT_HARNESS_VERSION}`,
`curator_types.py::{CandidateTask, PoolEntry, payload_hash}`, `shared/task_metadata.py::register_metadata_submodel`,
`backends/task_backend_errors.py::DeferralWriteAuthorityError`, `server/tools.py::inject_operational_routing`,
`server/main.py` (`known_project_roots_from_env`), `scheduler.py::{_persist_files_metadata,
_eligible_for_dispatch, _deferred_watch_gated}`, `dependency_verdict.py::deps_verdict`,
`reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants`,
`briefing.py::_format_delivered_checks` (the render precedent), `dashboard/data/burndown.py::collect_snapshot`,
`agents/triage.py::is_declared_simple_task`. Absent as the PRD says: `followup_items.py`, `declared_scope.py`,
`followup_hold.py`, and `middleware/agent_followup_guard.py` (task 4719's unlanded proposal).
Path correction: the config schema is `fused-memory/src/fused_memory/config/schema.py`, not
`fused-memory/config/schema.py` as §6 row ε spells it.

## Bindings

| leaf | capability | binding | verdict |
|---|---|---|---|
| α | `DeclaredScope` model + `derive_declared_scope` | new `shared/src/shared/declared_scope.py`; grep `class DeclaredScope\(`, `def derive_declared_scope\(` there | PASS |
| α | submodel registered under the `declared_scope` key | grep the quoted key literal in `shared/src/shared/` (0 today). The literal, not `register_metadata_submodel\('declared_scope'`, because a call wrapped over two lines would never match a line grep and would wedge β | PASS |
| α | stamp wired at the submit boundary | grep `derive_declared_scope` in `fused-memory/src/fused_memory/middleware/` (the interceptor, before the planning_mode split) | PASS |
| α | write-authority refusal (row 6, rejection-style) | grep `class DeclaredScopeWriteAuthorityError` in `fused-memory/src/` + `shared/src/`; the mechanism is created by α itself, so the rejection fires on α's tree | PASS |
| α | reader allowlist test | `kind: path` `shared/tests/test_declared_scope_readers.py` | PASS |
| α | row 20 band `undeclared` via `get_task` | manual; the curator-render half of row 20 is γ's (downstream render), so α's signal is the stamp only | PASS |
| β | `FollowupItem` + `merge_items` | new `shared/src/shared/followup_items.py`; grep `class FollowupItem\(`, `def merge_items\(` | PASS |
| β | key derivation + item invariants at submit | grep `followup_items` in `fused-memory/src/fused_memory/middleware/` | PASS |
| β | filing prompt asks for one bundle per area | grep `followup_items` in `orchestrator/src/orchestrator/agents/roles.py` | PASS |
| β | briefing renders the numbered list | grep `followup_items` in `orchestrator/src/orchestrator/agents/briefing.py` (the PRD names `_format_followup_items`; the key is the safer anchor) | PASS |
| β | an architect yields ONE bundled task in production | manual; agent behaviour, proven at θ | PASS |
| γ | combine audit fields | grep `items_before` and `candidate_source` in `fused-memory/src/fused_memory/middleware/` | PASS |
| γ | pool entry carries item count | grep `item_count` in `curator_types.py` (0 there today; the 9 hits elsewhere are other modules) | PASS |
| γ | curator renders the band | grep `declared_scope` in `task_curator.py` | PASS |
| γ | the "tiny" prose rule replaced (rejection-style) | grep `Either task alone is tiny` absent in `task_curator.py` (present today) | PASS |
| γ | holder excluded from pools and combine targets (row 16) | **re-homed at decompose**: γ creates `fused-memory/src/fused_memory/middleware/followup_hold.py` holding only the marker key constant and `is_holder`, keyed on the contract-fixed `metadata.followup_holder` marker (decision 10); ε extends the same module. Without this, γ's `is_holder` conjunct named a producer (ε) that depends on γ, a `producer-downstream` FAIL. Row 16 with a really minted holder is asserted end to end by ι | PASS |
| δ | batch rule 2 lifted (rejection-style) | grep `Combine between two new batch items is NOT supported` absent in `task_curator.py` (present today) | PASS |
| δ | non-create batch target degrades audited (row 5) | grep `batch-target-not-create` in `fused-memory/src/fused_memory/middleware/` | PASS |
| δ | harness version bumped | manual; 6707 may claim `curator-v2` first, so a literal grep could never go green | PASS |
| ε | holder mint + backfill | grep `def ensure_holder\(`, `def backfill\(` in `followup_hold.py` (the existing `def backfill\(` hits are other files) | PASS |
| ε | protection error (rows 14, 14b, rejection-style) | grep `class FollowupHolderProtectedError` in `fused-memory/src/`; created by ε, fires on ε's tree | PASS |
| ε | backfill wired at server startup | grep `followup_hold` in `fused-memory/src/fused_memory/server/main.py` | PASS |
| ε | `release_holder` on `set_task_status` | grep `release_holder` in `server/tools.py`. `server/tools.py` added to ε's files: §6 row ε omits it, but the MCP parameter lives there | PASS |
| ε | storm escape: one event per project | grep `followup_hold_backfilled` in `followup_hold.py` | PASS |
| ε | row 15 events half | **moved to ζ**: `followup_promoted{by: human}` needs ζ's `promote`; ε's half is the readiness (cancelled holder satisfies `deps_verdict`, existing) | PASS |
| ε | row 18 held absent | existing `deps_verdict` unsatisfied on a pending dependency, wired at `Scheduler._eligible_for_dispatch` | PASS |
| ζ | promotion closure | grep `def promote\(`, `def promote_dependencies_of\(` in `followup_hold.py` (the existing `def promote\(` hit is another file) | PASS |
| ζ | wired at the status chokepoint and edge insert | grep `promote_dependencies_of` in `task_interceptor.py` | PASS |
| ζ | event emitted | grep `followup_promoted` in `fused-memory/src/` | PASS |
| ζ | findings-contract §8, decompose-mode, cancel-with-children §4.2 edits | manual; prose edits, gating ι on their wording would be a prose guard (INV-10) | PASS |
| η | outcome event at done | grep `declared_scope_outcome` in `orchestrator/src/orchestrator/` | PASS |
| η | burndown `held` series | grep the quoted `held` key in `dashboard/src/dashboard/data/burndown.py` (0 today) | PASS |
| η | null vs 0 rendering (INV-13) | manual | PASS |
| ι | boundary suite | `kind: path` `fused-memory/tests/integration/test_followup_filing_shape.py` | PASS |
| ι | every §4.4 row but 19 green on the integrated tree | manual; the gate's own verify | PASS |
| θ | report script | `kind: path` `scripts/followup_filing_shape_report.py` | PASS |
| θ | dated report with producer-not-written states | manual; a measurement over live stores 14 days after ι | PASS |

## Resolved at decompose

1. γ / ε `is_holder` inversion: re-homed into γ (above). ε's description says it extends the module γ created.
2. Row 15 split: readiness on ε, `followup_promoted{by: human}` events on ζ.
3. Row 20 split: stamp on α, curator render on γ.

No binding resolves to declared-only, test-only, producer-downstream or producer-absent after these three.
DAG: β→α, γ→β, δ→γ, ε→γ, ζ→ε, η→α+ε, ι→δ+ζ+η, θ→ι. `signal_leaves: [ι]`.
