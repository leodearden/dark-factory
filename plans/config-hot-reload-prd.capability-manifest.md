# Capability manifest — plans/config-hot-reload-prd.md

Binds each leaf task's asserted capabilities to evidence (G3+G6 mechanized).
Verified 2026-07-02 against main @ bc7d20df18. No FAIL bindings.

## task γ — escalation `reload_config` tool (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Escalation server accepts a harness and tools await harness methods | grep:escalation/src/escalation/server.py:109-116 (`create_server(harness=…)`), :1157-1180 (`halt_scheduler` awaits `harness.force_halt_scheduler`, standalone `harness is None` guard) — wired production tool-registration path | PASS wired |
| `Harness.reload_config()` | producer:task-β (upstream in batch) | PASS producer-upstream |
| allowlist/diff/apply machinery | producer:task-α (upstream in batch) | PASS producer-upstream |
| Rejection: invalid config file → error, nothing mutated (I1, negative assertion) | rejection-check: pydantic validation observed to FIRE — orchestrator/tests/test_config.py:857-950 (`pytest.raises(ValidationError)` on steward-timeout invariant, load-time AND `validate_assignment` paths) | PASS rejection-fires |
| Operator-only enforcement (tool absent from agent allow-lists, I8) | grep:orchestrator/src/orchestrator/workflow.py:6878-6879 (per-role `allowed_tools`/`disallowed_tools` passed at spawn; ROLES registry workflow.py:126-132). Mechanism = omission from every role's allow-list, same as `halt_scheduler` (its docstring documents the convention) | PASS wired |
| In-process `load_config()` re-read is safe | grep:orchestrator/src/orchestrator/b3_gate.py:452 (`_resolve_cap` re-reads via `load_config` per check — live precedent) | PASS wired |

## task δ — integration gate + operator docs (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Per-role params read fresh at each agent spawn (scenarios 2, 8) | grep:orchestrator/src/orchestrator/workflow.py:6774-6775 (`self.config.max_turns` / `self.config.effort` read inside `_invoke`) | PASS wired |
| Dispatch semaphore startup-baked (scenario 3's "unchanged" assertion) | grep:orchestrator/src/orchestrator/harness.py:1114 (`asyncio.Semaphore(self.config.max_concurrent_tasks)` created once in `run()`) | PASS wired |
| GitOps holds the `config.git` submodel by reference (scenario 9 identity assert) | grep:orchestrator/src/orchestrator/git_ops.py:608-618 (`GitOps.__init__` stores the passed `config.git`) | PASS wired |
| Offline lane reads `git.offline_lane_test_threads` per run (scenario 9) | grep:orchestrator/src/orchestrator/offline_lane.py:382 (`threads = self.config.git.offline_lane_test_threads`) | PASS wired |
| Running-harness test rig with injectable config exists | orchestrator/tests/conftest.py (orch-config isolation fixtures) + orchestrator/tests/test_config.py mutation tests (:915-950) | PASS wired |
| Tool under test | producer:task-γ (upstream) | PASS producer-upstream |
| Hybrid-rollback unit scenario (5) — NOT end-to-end (unreachable with v1 allowlist; both steward-invariant sides allowlisted) | producer:task-α (upstream; α ships the synthetic-allowlist unit test) | PASS producer-upstream, premise honest |

## task ε — deterministic deploy (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| `scripts/restart-all-orchestrators.sh` committed + executable on main | git ls-tree main → `100755 blob c01fad7166` | PASS wired |
| Deterministic runner detached self-restart path (`target_unit` = own unit) | grep:orchestrator/src/orchestrator/deterministic_runner.py:167 (`_default_schedule_detached_restart`), :718 (restart_fn fallback wiring) | PASS wired |
| `before_done` validated at submit_task (script exists + executable) | fused-memory guard (tools.py pre-planning-branch check + guard.py) — additionally proven live by the ε `submit_task` call itself succeeding | PASS wired |
| Spec template | df task 2002 metadata (timeout_secs=900, target_unit=orchestrator-dark-factory.service, always_escalates=false) — retrieved 2026-07-02 | PASS wired |

## Intermediates (for completeness)

- **task α** (config.py machinery): consumes existing `OrchestratorConfig` +
  `validate_assignment=True` (config.py model_config) + `load_config`
  (config.py:2392). Consumer: β.
- **task β** (`Harness.reload_config`): consumes α; event_store append path
  (existing runs.db events usage in harness). Consumer: γ.
