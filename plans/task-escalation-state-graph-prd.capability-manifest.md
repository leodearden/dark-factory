# Capability manifest — task-escalation-state-graph-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), verified
against working tree `298556cc25` (2026-08-02). YAML sidecar twin:
`task-escalation-state-graph-prd.capability-manifest.yaml`.

## α — EscalationRef severity + shared pin classifier
- `EscalationRef` dataclass exists to widen → `task_ground_truth.py:114-120` (id/level/category today; severity dropped at :478-481) — PASS
- `Escalation.severity` field exists to carry → `escalation/models.py:92` — PASS
- escalation package importable by orchestrator (classifier home) → existing `from escalation.models import Escalation` in workflow.py — PASS
- filing claimant/session identity available at file time → escalations carry `worktree`/`agent_role`/`workflow_state` today; the filing session id is stamped by the same producer (additive field) — PASS (additive)

## η0 — dispatch gate any-level
- veto site exists → `harness.py:10137-10142` `has_open_l1` — PASS
- any-level read available → `escalation/queue.py` `get_by_task(status='pending')` — PASS
- rejection signal producible: a pending task with only an L0/L2 open is NOT flipped (rejection-mechanism check: the new test authors the shape and observes the veto fire) — PASS (test-bound)

## β — structured recovery emission + streak
- event-store emission pattern → `harness.py:5503-5514` region (existing EventType usage) — PASS
- all five veto sites enumerated → `task_ground_truth.py:613-641`, `harness.py:5560`, `scheduler.py:5719`, `harness.py:11759/:11373` — PASS
- streak-escalation house pattern → stage1_stall_detector / storm-counter precedent — PASS

## γ1 — Path B bounded escalated-wait
- Path A machinery exists → `workflow.py:12060-12343` (`_wait_for_resolution`, idle window, `_StewardReescalated` L2 short-circuit :12176-12181) — PASS
- bail site → `workflow.py:3457-3464` — PASS
- strand test file absent today (the task creates it) → `orchestrator/tests/test_workflow_merge_gating_strand.py` not in tree — PASS (deliverable, not premise)

## γ2 — merge-phase carve-out + merge-halt rewrite
- carve-out documented in `_mark_blocked` docstring (merge_phase suppresses transitions) → `workflow.py:13708` region — PASS
- halt-owner unhalt gate → `harness.py:11928-11935` `is_halt_owner` — PASS
- rehydration filter to widen → `harness.py:9804-9851` (`level == 1` at :9825-9828) — PASS

## γ3 — truthful REQUEUED/CANCELLED/infra exits
- `WarmLaneRequeue` site → `workflow.py:2853-2901` — PASS
- soft-cancel fallback → `workflow.py:14919-14923` — PASS
- infra resume write → `harness.py:12846-12866` — PASS
- no-recompete premise: the in-progress hack's rationale is in-code → `harness.py:12829-12845` — PASS (mechanism decided in-task)

## δ — CONVERT_TO_BLOCKED (log-mode)
- `_RECOVERY` table + LEAVE default → `task_ground_truth.py:533-641` — PASS
- claimant-clear-before-flip pattern → `scheduler.py:5726-5735` — PASS
- park-stop deque to exclude from → `scheduler.py:2278-2281`, `config.py:3724-3739` — PASS
- `in-progress→blocked` is a legal Table A edge → `task_transitions.py` `_UNION` — PASS
- watcher playbook file exists to gain a row → `skills/escalation-watcher/` — PASS

## ζ — resume on claimant-liveness
- cascade site → `harness.py:12789-12917` (`:12873` status gate; `:12014` level gate) — PASS
- Table B `effect_for` authority → `escalation/action_effects.py:107-142` — PASS
- claimant-liveness oracle → `shared/task_claimant.py:106-141` — PASS

## η — veto-site collapse + reaper rule
- classifier producer → task α (upstream in DAG) — PASS (producer:α)
- reaper defer sites → `harness.py:10333-10352` — PASS

## θ — tighten table + loud SM-2
- `_OUTCOME_ALLOWED` → `task_transitions.py:279-330` — PASS
- SM-2 asserts → `workflow.py:3218-3246`; harness catch `harness.py:7887-7894` — PASS
- W9 test modules to update exist → `orchestrator/tests/test_workflow_state_machine.py`, `test_workflow_state_machine_boundary.py`; `shared/tests/test_task_transitions.py` — PASS
- producer fixes upstream → γ1-γ3 (DAG-direction) — PASS

## ι — observability projection
- claimant fields on the wire → `sqlite_task_backend.py:918-937` (`_row_to_task` includes claimant_run_id/heartbeat_at); dashboard drops at `dashboard/src/dashboard/data/tasks.py:101-111` — PASS
- burndown producer → `dashboard/src/dashboard/data/burndown.py:26-216` — PASS
- escalation analytics + server read → `dashboard/src/dashboard/data/escalation_analytics.py:369-457`; `escalation/server.py:932-973` — PASS
- classifier for pins_recovery → task α (upstream) — PASS (producer:α)

## κ — docs alignment
- divergent doc sites verified → ARCHITECTURE.md §3.6/§3.7/diagram; `docs/task-authoring.md` §status — PASS

## λ / μ — deterministic operator gates
- pure-gate shape validated at submit → `task_kind='deterministic'` + `metadata.always_escalates=true`, no `before_done` (docs/task-authoring.md §5; submit-time validation) — PASS
- restart runbook substrate → `scripts/restart-all-orchestrators.sh`, OPERATIONS.md §Fleet redeploy (never `--drain` for fm) — PASS
- soak evidence sources → β's events + θ's would-violate counter (upstream producers) — PASS

No FAIL bindings. All producer references are upstream in the DAG.
