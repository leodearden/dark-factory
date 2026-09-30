# Capability manifest — verify-result-cache-prd

Bindings checked against main `6cb5f4a2b6` (2026-09-19). Cited by symbol; re-locate at
implementation time. Machine-readable twin: `verify-result-cache-prd.capability-manifest.yaml`.

## α — Verify events carry the plan hash, per-unit outcomes, and task-leg duration

| capability the signal asserts | evidence | verdict |
|---|---|---|
| A plan is available at the verify result | `orchestrator/src/orchestrator/verify.py::VerifyResult.plan` (JSON-native dict of `verify_plan.py::VerifyPlan`) | PASS — wired |
| The merge-role emit site can carry new keys | `orchestrator/src/orchestrator/verify_runner.py` builds the `merge_verify` payload from the `VerifyResult` (`EventType.merge_verify`; field note in `event_store.py`) | PASS — wired |
| The task-role emit site can carry new keys | `orchestrator/src/orchestrator/workflow.py` emits `workflow_verify` with a dict literal (`passed`, `tip_sha`, `base_sha`, `branch`) | PASS — wired |
| `EventType` accepts additions | `event_store.py::EventType` is a `StrEnum` | PASS |
| The escalation-side `workflow_verify` emitter | `escalation/src/escalation/server.py::merge_request` (`verified_green`) — owned by task 5409's shared constructor; α does not touch it, coordinates by symbol | PASS — producer 5409, parallel |

## β — Registry, keys, policy

| capability | evidence | verdict |
|---|---|---|
| Git object ids for subtrees without checkout | `git rev-parse <sha>:<path>` / `git cat-file --batch-check` (used by the 09-14 measurement scripts) | PASS |
| Module list and dependency closure derivable | `config.py::ModuleConfig` (`prefix`, `test_command`, `lint_command`, `type_check_command`); workspace members' `pyproject.toml` `dependencies` + `dependency-groups.dev` | PASS |
| A green-tier config key can be added | `config.py::RELOADABLE_FIELDS` (frozenset; `git.offline_lane_commands` is an existing member) | PASS — wired |
| A CLI subcommand surface exists | `orchestrator` CLI (`verify-merge` subcommand cited in the multihost PRD; `orchestrator/src/orchestrator/cli.py`) | PASS |

## γ — Per-module decision at the fan-out seam (leaf)

| capability | evidence | verdict |
|---|---|---|
| Modules are iterated one at a time under a semaphore | `verify.py::run_full_verification` (`_fanout_sem = asyncio.Semaphore(max(1, _fanout_cap))`), `dark-factory-orchestrator.yaml` `merge_verify_max_concurrent_modules: 1` | PASS — wired |
| The task role reaches the same layer | `verify.py::run_scoped_verification` | PASS — wired |
| The duplicate type-check leg can consult executed runs | `merge_queue.py::_run_unscoped_typechecks` runs after the scoped leg from `verify_runner.py::LocalRunner.run_merge_verify`; task 5294's analysis (absorbed) | PASS — wired |
| Registry / keys / policy | producer: β, upstream | PASS |
| Events carry units | producer: α, upstream | PASS |
| "Completes under 10 min with every unit `cache_hit`" | floor when every unit replays = git merge + plan + registry lookups + the unscoped leg's consult; no pytest/pyright executes; today's p50 is 39.8 min | PASS — floor stated |
| Tree tier for any project | `verify_runner.py::RemoteRunner`/`LocalRunner` dispatch is project-agnostic; key = merge tree OID + spec hash | PASS — wired |

## δ — Repo-wide guard module (leaf)

| capability | evidence | verdict |
|---|---|---|
| Modules are registered per directory | `shared/orchestrator.yaml`, `scripts/orchestrator.yaml`, `tests/scripts/orchestrator.yaml` exist and are discovered by `config.py` | PASS — wired |
| The whole-tree tests to move are identifiable | measured escapes 09-16: `shared/tests/test_capability_manifest.py`, `shared/tests/test_silent_fallthrough_gate.py::TestWholeTreeGate`, `shared/tests/test_locking.py::TestFileExtensionsDriftGuard`, `tests/scripts/test_reify_closure_staleness_sweep_retired.py`, `scripts/tests/test_design_invariants_consistency.py`; E-selection `repowide.json` tags the wider candidate set | PASS |
| `verify_cache: never` on a module | producer: β (`ModuleConfig` field), upstream | PASS |
| The 08-20 shape is reproducible as a fixture | a `plans/*.capability-manifest.yaml` edit with `shared/` untouched; `shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus` reads `plans/` | PASS |

## ε — Audit on the offline lane (leaf)

| capability | evidence | verdict |
|---|---|---|
| A config-driven lane command runs from head | `config.py::LaneCommand`, `git.offline_lane_commands` (hot-reloadable), `offline_lane.py::OfflineLaneWorker._run_once` (always-from-head, coalescing) | PASS — wired |
| A confirmed-red path with fix-task and blocker exists | `offline_lane.py::OfflineLaneWorker._handle_red_run`, `_file_new_fix_task`, `_maybe_promote_blocker`, `_file_blocker_escalation` | PASS — wired |
| The confirmed failing set is parseable per test | `offline_lane.py::_parse_confirmed_failures`; timeout/crash misclassification is task 4226 (pending) — ε depends on it | PASS — producer 4226, upstream |
| Consumption records exist per landing | producer: β (`Registry.consume`) + γ (writes at verdict time), upstream | PASS |
| Hot-disable without restart | `policy_state` in the registry (β), read on every decision | PASS — producer β |

## ζ1 — Per-test recorder (intermediate)

| capability | evidence | verdict |
|---|---|---|
| Per-test coverage contexts | coverage 7.13.5 + pytest-cov 7.1.0 in the workspace venv (`uv run --project orchestrator`); C tracer required — `coverage/sysmon.py` leaves `should_start_context`/`switch_context` unimplemented | PASS — C tracer |
| A root plugin imported by every package conftest | `df_pytest_isolation.py` imported by `conftest.py` and `orchestrator/tests/conftest.py` (and siblings) | PASS — wired (new sibling module, not an edit to it) |
| The audit lane runs every head uncached | producer: ε, upstream | PASS |
| Unmarked repo-state read fails at task role | built and bound by ζ1 (rejection mechanism is the deliverable; row 17 executes it) | PASS — manual |

## ζ2 — Per-test replay (leaf)

| capability | evidence | verdict |
|---|---|---|
| Collection-time deselection | pytest `hookspec.py::pytest_collection_modifyitems` | PASS |
| Per-test verdicts at the gate | merge-role junit under `.df-verify-junit/report<infix>.xml` (`verify.py::_with_junitxml_str`) | PASS — wired |
| Recordings | producer: ζ1, upstream | PASS |
| Post-rebase re-verify is observable | `event_store.py::EventType.rebase_verify_cost` (`next_verify_wall_secs`) | PASS — wired |

## η — Integration gate (leaf)

| capability | evidence | verdict |
|---|---|---|
| Rows 1–19 of the boundary-test sketch | producers γ, δ, ε, ζ2 all upstream | PASS |
| A fixture-repo harness pattern | `orchestrator/tests/test_verify_admission_integration_gate.py` (real flock in a tmp dir) as the template | PASS |

## θ — Companion docs (leaf)

| capability | evidence | verdict |
|---|---|---|
| R-cache-1 recorded once and pointed to | PRD § Pre-conditions is the home; `OPERATIONS.md` and the multihost PRD's scope guard receive dated pointers | PASS |
| 5294 closes as superseded | closed by the operator at γ landing (agents hold no status tool) | PASS — manual |

## ι — Milestone: author the Reify per-step PRD (filed in reify's task store)

| capability | evidence | verdict |
|---|---|---|
| Delayed milestone on a pure human gate | `docs/task-authoring.md` §6; `shared/src/shared/task_metadata.py::Milestone`; `task_kind='deterministic'` + `always_escalates=True`, no `before_done` | PASS |
| Cross-project dependency on γ | `docs/task-authoring.md` §3.2 (`dark_factory:<γ>` → `metadata.external_deps`) | PASS |
| Reify per-step stamps | reify task 7424 (pending), local dep | PASS — producer 7424 |

Not in the YAML sidecar — the stamper keys on this project's batch.

No numeric bound is asserted as a pass criterion anywhere. The h/day figures in the PRD's
Goal and in ζ2's row are labelled projections.
