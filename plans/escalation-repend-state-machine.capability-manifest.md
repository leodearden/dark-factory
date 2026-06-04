# Capability manifest — escalation-repend-state-machine PRD

Per-leaf capability→evidence bindings (G3+G6 mechanized). All `grep:` evidence verified against
the working tree 2026-06-04 (31/31 anchor re-verification, this authoring session). Leaves:
ζ, η, θ, ι, κ. Intermediates α1/α2/β/γ/δ/ε are consumed by ζ (integration-gate pattern).
Field-population / grammar-fixture / numeric-floor checks: N/A throughout (no result-field
sampling, no grammar surface, no numeric-accuracy bounds — threshold 3 is policy, not accuracy).

## ζ — integration gate (boundary suite B1–B15)

| Capability | Evidence | Verdict |
|---|---|---|
| `resolve_issue` action enum + `Escalation.resolution_action` | producer:α1 (upstream via β, ε) | PASS |
| chokepoint downgrade + `stranded_blocked` category | producer:α2 (upstream via ε) | PASS |
| harness action dispatch, level≥1 flip, teardown suppression | producer:β (upstream) | PASS |
| three-gate born-at-L2 disjunct | producer:γ (upstream) | PASS |
| signature-aware re-block guard | producer:δ (upstream) | PASS |
| stranded sweep rework + startup pass | producer:ε (upstream) | PASS |
| soft-cancel kill machinery (restart/park/abandon substrate) | grep:orchestrator/src/orchestrator/harness.py:2358-2375 — `_workflow_cancel_events` + `_workflow_slot_tasks` + hard-cancel registration, wired into the live `release_workflow` MCP path | PASS (wired) |
| `BORN_AT_L2_SEVERITIES` | grep:escalation/src/escalation/models.py:41 — `frozenset({'critical','urgent'})` | PASS |
| flip paths to extend | grep:orchestrator/src/orchestrator/harness.py:4168-4221 — production `_on_escalation_resolved` callback | PASS (wired) |
| gates to extend | grep:orchestrator/src/orchestrator/workflow.py:3093,3442,1186 — production execute/debug/merge loops | PASS (wired) |
| `deferred`-as-park substrate | grep:orchestrator/tests/test_cascade_unblock.py — `test_criterion_3_deferred_task_not_flipped`; scheduler dispatches `pending` only | PASS |
| task-state mutation API (`set_task_status`, `update_task` append) | fused-memory MCP live (exercised this session) | PASS |

## η — L2 escalation-watcher skill rewrite

| Capability | Evidence | Verdict |
|---|---|---|
| C1 semantics table to document | producer:α1 (upstream) | PASS |
| harness behavior to document | producer:β (upstream) | PASS |
| target prose sections exist | grep:skills/escalation-watcher/SKILL.md:174 (AFK shift-1 terminate prose), :257-368 (`review_suggestions` dead handler), :378/:386 (B3-hardcoded handlers — gaps PRD-2 entry 2), :538-557 (resolution-semantics block) | PASS |

## θ — auto-watcher routing table

| Capability | Evidence | Verdict |
|---|---|---|
| action semantics + `stranded_blocked` behavior | producer:α1, ε (upstream) | PASS |
| routing-table snippets to migrate | grep:skills/escalation-watcher-auto/SKILL.md:247,252,272,297,302 — `terminate=` snippets in Per-Category table | PASS |

## ι — roles.py policy doc + orchestrator-tree caller migration

| Capability | Evidence | Verdict |
|---|---|---|
| roles.py exists with escalation-interface prose | grep:orchestrator/src/orchestrator/agents/roles.py (contains `resolve_issue` references) | PASS |
| downgrade rule to document | producer:α2 (upstream) | PASS |
| migration targets | grep -l `resolve_issue|terminate=`: harness.py, agents/briefing.py, agents/roles.py | PASS |

## κ — companion sweep (out-of-register skills)

| Capability | Evidence | Verdict |
|---|---|---|
| migration targets exist | grep:skills/unblock/SKILL.md:219,225; skills/recon-escalation-watcher/SKILL.md:109,113,128,136,142-143 | PASS |
| hard-error message (migration aid) | producer:α1 (upstream) | PASS |

**Result: 0 FAIL bindings — batch clear to queue.**
