# Capability manifest — `plans/cancel-with-children-prd.md`

Per-leaf capability → evidence bindings (G3 + G6 mechanized). Machine-readable twin:
`plans/cancel-with-children-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).
Verified against main `f7f9957bb5` (2026-11-01): every `expect: present` pattern below is ABSENT
today and every `expect: absent` pattern is PRESENT today, so each check gates something.

| leaf | capability | binding | verdict |
|---|---|---|---|
| α | contract module exists | new file `shared/src/shared/cancel_dispositions.py` (path check) | PASS |
| α | `ChildDisposition` model | grep `class ChildDisposition` in `shared/src/shared/` | PASS |
| α | `cancel_dispositions` blessed + registered | grep in `shared/src/shared/task_metadata.py` | PASS |
| α | `Escalation.cancel_children` field | grep in `escalation/src/escalation/models.py` | PASS |
| β | gate module exists | new file `fused-memory/src/fused_memory/middleware/cancel_children_gate.py` | PASS |
| β | `get_cancel_children` MCP tool | grep `def get_cancel_children` in `fused-memory/src/fused_memory/server/` | PASS |
| β | undispositioned refusal | rejection-mechanism; grep `cancel_children_undispositioned`; §6 rows 1, 27 | PASS |
| β | registry accessor | grep `def known_project_roots` in `reconciliation/backlog_policy.py`, `recon_state/backlog_policy.py` | PASS |
| β | enforce flag hot-reloadable | grep `cancel_children\.enforce` in `fused-memory/src/fused_memory/config/reload.py` | PASS |
| β | late mode never deadlocks; observe mode allows + journals | manual — §6 rows 28, 31 (real-backend tests) | PASS |
| γ | backstop module exists | new file `fused-memory/src/fused_memory/reconciliation/cancel_backstop.py` | PASS |
| γ | string-prefix recursion guard retired | grep `_PARENT_CANCELLED_REOPEN_PREFIX` absent in `fused-memory/src/fused_memory/` | PASS |
| δ | typed `CancelChildrenRejection` | grep in `orchestrator/src/orchestrator/` | PASS |
| δ | abandon pre-flight probe | grep `def cancel_children_blocker` in `orchestrator/src/orchestrator/` | PASS |
| δ | terminal-head refusal | rejection-mechanism; grep `cancel_children_parent_terminal`; §6 row 33 | PASS |
| ε | skill exists | new file `skills/cancel-with-children/SKILL.md` | PASS |
| ε | blanket "repoint every dependent" rule retired | grep `[Rr]epoint every` absent in `skills/escalation-watcher/SKILL.md` | PASS |
| ε | Stage 2 consumes the read tool | grep `get_cancel_children` in `reconciliation/prompts/` | PASS |
| ζ | end-to-end green + enforce flip | manual — integration gate | PASS |
| η | production observation (INV-13) + enforce flag retired | manual — delayed milestone 7 days after ζ | PASS |

No binding resolves to declared-only, test-only, producer-downstream or producer-absent: every
capability is produced by the leaf that asserts it, and each consumer leaf depends on its
producer (γ, δ, ε → β → α; ζ → γ, δ, ε; η → ζ).
