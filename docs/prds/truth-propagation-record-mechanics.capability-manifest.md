# Capability manifest — truth-propagation record mechanics

PRD: `docs/prds/truth-propagation-record-mechanics.md`. Bindings verified
against main `863970f336` (2026-08-24). Machine-readable twin:
`truth-propagation-record-mechanics.capability-manifest.yaml`.

## α — `amend_escalation` MCP verb

| Capability | Binding | Verdict |
|---|---|---|
| amendment-machinery-reusable | capability→substrate: `Amendment` model (`escalation/src/escalation/models.py`) + the fold path's append/cap machinery (`queue.py::add_members_to_l2`, sole `updated_at` writer today) verified present to reuse; `git grep amend_escalation escalation/src` = **0-today** (control: `def resolve_issue`/`def promote` = present) | PASS |
| updated-at-bump-on-amend | behavioural — the leaf's signal compares `get_escalation` before/after: `updated_at` newer than `triaged_at` post-amend | PASS (manual) |
| terminal-record-refusal | rejection-mechanism — the same call against a resolved/archived record refuses without mutation (fixture in the leaf's signal) | PASS (manual) |

## β — sideways census in `resolve_issue`

| Capability | Binding | Verdict |
|---|---|---|
| sideways-census-in-response | capability→producer (this task) — reuses the existing pending-queue read path (`get_pending_escalations(task_id=...)` machinery verified present); response field name is PRD OQ1, so the check is the leaf's fixture signal, not a name grep | PASS (manual) |
| report-only-no-mutation | rejection-mechanism — no listed record's file changes (fixture asserts byte-identity) | PASS (manual) |

## γ — backstop decision gate

| Capability | Binding | Verdict |
|---|---|---|
| decision-recorded | pure human gate (`deterministic` + `always_escalates`, dated milestone — `docs/task-authoring.md` §5–6 preset verified) — the decision lives in the gate's resolution; no code deliverable | PASS (manual) |

## δ — `delivered_check` premise-recheck

| Capability | Binding | Verdict |
|---|---|---|
| premise-shape-validated | capability→producer (this task) — `delivered_checks` write-time validation substrate verified (`docs/task-authoring.md` §delivered_checks; shared submit-guard precedent); field shape is PRD OQ2, so the check is the leaf's ValidationError fixture, not a name grep | PASS (manual) |
| warn-never-block | rejection-mechanism — stale-premise fixture evaluates green AND files exactly one deduped info escalation; a blocking path is a test failure | PASS (manual) |
