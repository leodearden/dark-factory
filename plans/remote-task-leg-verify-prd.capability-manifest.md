# Capability manifest — remote task-leg verifies

PRD: `plans/remote-task-leg-verify-prd.md` (landed `f188b4b568`). Decomposed 2026-10-09 by a
sibling `/prd` decompose session. Substrate re-read on main `f188b4b568`; `orchestrator/src`
is unchanged since the PRD's verify SHA `00725ff2b6`. Machine-readable twin:
`plans/remote-task-leg-verify-prd.capability-manifest.yaml` (stamped by `commit_planning`;
η hand-stamped, see below). Task ids are in the PRD's §14 decompose record.

Binding vocabulary is `skills/prd/references/gates.md` → *Capability Manifest*. Every
mechanical `delivered_check` below was linted against `main` with
`shared.delivered_check_polarity.lint_delivered_checks` before filing: zero findings.

## Resolutions made at decompose

Three bindings failed as the PRD's §9 rows were written. Each is resolved without changing
a §4 design decision.

1. **γ read config keys that δ2 produces (producer-downstream).** §5.5's `eligible(config, …)`
   and "`config` is read per call" assume `verify_task_*` fields on `OrchestratorConfig`.
   Those fields land in δ2, which is downstream of γ, and pyright would reject the attribute
   reads in γ's source. **Resolution:** γ owns a frozen `TaskVerifyPolicy` value
   (host policy, remote module prefixes, cross-check cadence, red-confirm flag).
   `eligible(policy, …)` takes it, and the dispatcher reads it per call through an injected
   `policy_of(config) -> TaskVerifyPolicy`, so a hot reload still reaches the next dispatch.
   δ2 adds the config keys and the config-backed reader. This also moves the dispatcher's
   construction from δ1 to δ2, because the reader exists only after δ2. δ1 builds the
   quarantine set, allocator and pool; δ2 builds the dispatcher and adds `task_verify=` at the
   `harness.py` `build_workflow` call. `build_workflow`'s kwarg is δ2's, so that call-site
   edit could never have been δ1's.
2. **Rows 9, 12 and 13 named halves produced downstream of γ.** The RU-tracker half of rows 9
   and 12 needs δ1's public tracker entry. The `SOFT_CANCELLED` workflow outcome in row 13
   needs δ2's workflow wiring. **Resolution:** γ asserts the `on_unavailable(name, kind)`
   hook call and the dispatcher-side cancel-before-unwind order. δ1 asserts the tracker half
   of row 9. ι executes all three rows end to end.
3. **Row 3 spans two producers.** The fake-ssh `rc ≠ 0 → RunnerUnavailable(kind='transport')`
   half is α's. The CLI exit-1-on-`VerifyInfraError` half is β's. ι runs the joined row.

Two smaller clarifications: the `verify-(merge|task)` probe-pattern constant has one home,
α's `verify_runner.py`, and β reuses it. `runner` on `workflow_verify` is read from a
dispatcher decision keyed by `(task_id, attempt_id)`, never from a shared "last decision"
slot. Up to `max_concurrent_tasks` workflows share one dispatcher (heuristic 7).

## Per-leaf bindings

### α — TaskVerifySpec + transport

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `TaskVerifySpec` type | producer α (new `task_verify_spec.py`); absent on main | PASS | grep `^class TaskVerifySpec[(:]` |
| `RemoteRunner.run_task_verify` over `_dispatch` | producer α; substrate `verify_runner.py::RemoteRunner.run_merge_verify` (push → ssh → parse → archive → delete-ref, single `_inflight_request_id`) | PASS | grep `def run_task_verify\(` |
| public no-verdict predicate | producer α over `verify.py::_NO_VERDICT_SUMMARY_MARKERS` (present) | PASS | grep `def is_no_verdict_summary\(` |
| no-verdict → `RunnerUnavailable` for both callers | rejection mechanism built by α; row 4 observes it fire | PASS | manual |
| wire-shape and exclusive-use rejections | built by α; rows 1 and 19 | PASS | manual |

### β — `orchestrator verify-task`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `verify-task` subcommand | producer β in `cli.py` beside `verify_merge`; absent on main | PASS | grep `(def verify_task\(\|['"]verify-task['"])` in `cli.py` |
| public ephemeral worktree | producer β over private `git_ops.py::GitOps._create_merge_worktree` (present) | PASS | grep `def create_ephemeral_verify_worktree\(` |
| cancel reaps the worktree | producer β beside `verify_cancel.py::write_pgid_file` / `cancel_request` (present) | PASS | grep `def write_worktree_path\(` |
| spec codec and apply | producer α upstream (β → α) | PASS | manual |
| fresh worktree reads cold | substrate `verify.py::_is_verify_cold` keys on `.task/verify_warmed` (present); row 2 | PASS | manual |

### γ — HostPool + dispatcher

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `HostPool` | producer γ over `verify_runner.py::HostAllocator` (`acquire_remote`, `cancel_and_release`, `is_parked`, `readmit` present) | PASS | grep `^class HostPool[(:]` |
| `TaskVerifyDispatcher` | producer γ (new `task_verify_dispatch.py`) | PASS | grep `^class TaskVerifyDispatcher[(:]` |
| public `build_remote_runners` | producer γ, moved from `merge_lane/drift.py::_build_remote_runners` (present) | PASS | grep `^def build_remote_runners\(` |
| policy without config keys | as written: producer-downstream (δ2) **FAIL** → resolution 1 | PASS | manual |
| dispatcher substrate | `verify_plan.py::derive_verify_plan`, `GitOps.has_uncommitted_work`, `RemoteRunner.sync_if_stale`, `verify_categories.py` `INFRA_TRANSIENT_CATEGORIES` (all present) | PASS | manual |
| lease released on every exit | built by γ; rows 13, 14, 21 | PASS | manual |

### δ1 — harness-owned allocator (hub carry, stage 1)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `merge_lane_wants_host()` | producer δ1 over worker `_redispatch`, `_verifier_queue`, `_pending_verifier_get`, `_merge_ahead_ledger` (present) | PASS | grep `def merge_lane_wants_host\(` |
| public RU tracker entry | producer δ1 over `_quarantine_unreachable_host`, `_record_runner_unavailable`, `_record_runner_recovered` (present) | PASS | grep `def note_runner_unavailable\(` |
| harness owns the pool | producer δ1 in `Harness.__init__`, injected via `Harness._start_merge_worker` (present); `HostPool` from γ upstream | PASS | grep `HostPool\(` in `harness.py` |

### δ2 — workflow, config, dispatcher wiring (hub carry, stage 2)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `verify_task_*` config keys | producer δ2 (§5.7); absent on main | PASS | grep `verify_task_host_policy` in `config.py` |
| workflow port kwarg | producer δ2; call site `TaskWorkflow._run_scoped_verification_with_infra_retry` (present) | PASS | grep `task_verify:[[:space:]]*['"]?TaskVerifyPort` in `workflow.py` |
| harness passes the dispatcher | producer δ2 at the `harness.py` `build_workflow` call | PASS | grep `task_verify=self\.` in `harness.py` |
| one task-role verify site | producer task 6580 upstream (edge δ2 → 6580) | PASS | manual |
| `workflow_verify.runner` populated | field population from γ's keyed decision; row 17 | PASS | manual |

### ι — integration gate

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| integration module | producer ι; α β γ δ1 δ2 in its closure | PASS | path `orchestrator/tests/test_task_verify_integration.py` |
| 25 rows green end to end | all producers upstream | PASS | manual |

### ε — rollout gate (operator)

All checks are `manual`, because ε is an operator gate. Its prerequisites are 6565, 6127,
3310, 6648, 6580 and ι, all wired upstream. Its first production row is the step 2
pre-flight sidecar (INV-13). The report asserts no numeric threshold in advance (G6); the
decision is Leo's.

### η — pressure hold (deferred)

η is filed `deferred` and is not part of the pending flip. Its `task_id` is hand-stamped in
the sidecar because `commit_planning` stamps only on a `pending` flip. Its check is `manual`
because η has no dependent, so a mechanical check would gate nothing. It depends on ι, so a
premature flip cannot dispatch it before the dispatcher exists. That is not the
ε-status edge the PRD forbids (§4.11, critic F11).
