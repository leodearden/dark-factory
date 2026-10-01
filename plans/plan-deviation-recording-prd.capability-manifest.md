# Capability Manifest — plan-deviation-recording-prd

Mechanizes G3 (substrate exists / wired) + G6 (premise valid) per leaf of
`plans/plan-deviation-recording-prd.md`. One block per leaf; each capability
bound to on-main evidence. Any **FAIL** binding blocks queueing.

**Domain flags:** tooling/infra — no grammar/DSL → grammar-fixture checks N/A;
no numeric accuracy bounds → numeric-floor checks N/A. Live checks:
capability→producer (wired), DAG-direction, rejection-mechanism.

Evidence verified against main `2c78539443` (2026-09-21), cited by symbol; the
claims were re-checked by a fresh adversarial pass the same day. Machine-readable
twin: `plans/plan-deviation-recording-prd.capability-manifest.yaml` (stamped by
`commit_planning`; mechanical checks are copied into each producer's
`metadata.delivered_checks`).

---

## α — Note-to-file escalation class + `resolve_escalation_shas`  *(intermediate → γ, δ; after 5221/5222)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `escalate_info` exists with `severity`/`evidence`/`terminal_state_is_the_bug` params to extend | capability→producer (wired) | `escalation/src/escalation/server.py::escalate_info` → `_chokepoint_or_submit` | PASS |
| Identity stamp sits above every gate in the chokepoint | capability→producer (wired) | `server.py::_chokepoint_or_submit` task-3550 block ("ABOVE everything else") — the flag is evaluated after it | PASS |
| Atomic write-as-resolved with a class; fires the resolve callback once | capability→producer (wired) | `escalation/src/escalation/queue.py::EscalationQueue.submit_resolved(resolution_class=)`; used by gate 4 today | PASS |
| `close_only` has no task-status effect, so the callback is inert | capability→producer (wired) | `escalation/src/escalation/action_effects.py::ACTION_EFFECTS[('close_only', ANY, ANY)] = TaskEffect(None, WORKFLOW_NONE)`; `models.py::Escalation.resolution_action` | PASS |
| `RESOLUTION_CLASSES` is an extensible, set-equality-tested vocabulary | capability→producer | `models.py::RESOLUTION_CLASSES` (`stale-strand` added by 3172); `escalation/tests/test_models.py::test_resolution_classes_contains_exactly_the_legal_values` — two-site edit; task 5221 adds four members first | PASS |
| New record field needs no migration | capability→producer | `models.py::Escalation.from_dict` `__dataclass_fields__` filter | PASS |
| `note_to_file=True` with non-info severity is refused | rejection-mechanism | built + bound by α; boundary row "severity guard" observes the structured error and asserts no file was written | PASS (built by α) |
| Patch-id shape to imitate for the resolver | capability→producer | `orchestrator/src/orchestrator/git_ops.py::GitOps.find_equivalent_commit` — imitated, not reused (needs a worktree + `base_sha`); `find_task_citation_commit` is not usable (matches task ids, not subjects) | PASS |
| Escalation server can reach git when a harness is wired | capability→producer (wired) | `server.py::claim_warm_worktree` → `harness.git_ops`; standalone `create_server(harness=None)` returns a structured error | PASS |
| Auto-watcher skill has the per-category section to extend | capability→producer | `skills/escalation-watcher-auto/SKILL.md` "`design_concern` / `risk_identified` / `missing_premise`"; 5222 adds the info-only-cluster rung there first | PASS |

## β — Per-step disposition record + plan-tools riders  *(intermediate → γ, δ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Plan write path + step-scoped mutation to sit beside | capability→producer (wired) | `orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.write_plan`, `::update_step_status`, `::read_agent_session`; `mcp/plan_tools.py::_mark_step_done` | PASS |
| Unknown step keys survive read / normalise | capability→producer (wired) | `artifacts.py::_normalize_plan`, `plan_tools.py::_read_plan_repaired` keep unknown keys | PASS |
| Tool registration pattern with markup guard | capability→producer (wired) | `plan_tools.py` `@mcp.tool()` + `@accepts_markup_override` on `add_plan_step` | PASS |
| `shared` importable by every consumer | capability→producer (wired) | uv workspace: `orchestrator`, `escalation`, `scripts/legibility` (`uv run --project shared`) all resolve `shared` | PASS |
| `step_type` outside `{test, impl}` is refused | rejection-mechanism | built + bound by β; today `_add_plan_step` stores it verbatim — refusal absent on main, which is the defect | PASS (built by β) |
| `_replan` prompt is the only "write plan.json directly" instruction | capability→producer | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._replan` inline prompt — the sentence to remove exists on main | PASS |
| C4 `_descoped_steps` is NOT extended | premise | absent on main (task 4032 pending); D4 records its own step-level list | N/A |
| Nested prose outside `_REPAIRABLE_PLAN_FIELDS` | premise (limit accepted) | `plan_tools.py::_walk_repairable` addresses one level; inbound `MarkupGuardMiddleware` still covers the tool call | N/A (recorded in D4) |

## γ — Doctrine inline + parity guard + one `escalate_info` contract + architect duties  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The sentence to replace exists | capability→producer | `orchestrator/src/orchestrator/agents/roles.py::IMPLEMENTER` "## Important": "note it and stop. Do NOT modify the plan." | PASS |
| Inline-with-parity precedent | capability→producer (wired) | `roles.py::CODE_QUALITY_GUIDANCE` + `orchestrator/tests/test_code_quality_guidance_parity.py` (task 5225) | PASS |
| Tool names the doctrine cites exist upstream | DAG-direction | `record_step_deviation` ← β (upstream); `escalate_info(note_to_file=)` ← α (upstream) | PASS |
| Implementer holds memory write for the dated pointer | capability→producer (wired) | `roles.py::_MEMORY_TOOLS` contains `mcp__fused-memory__add_memory` | PASS |
| Three `escalate_info` contract sites exist to collapse; two reach the same prompt | capability→producer | `roles.py::ESCALATION_LADDER_CORE`, `::DEEP_REVIEWER`, `::_FOLLOWUP_FILING_INSTRUCTIONS` (appended to ARCHITECT and IMPLEMENTER post-construction) — hence one constant + pointers | PASS |
| A separate allowlist is needed | capability→producer | `roles.py::_PLAN_STATUS_TOOLS` is also held by `SIMPLE_TASK` — hence `_PLAN_DEVIATION_TOOLS` | PASS |
| ARCHITECT rule 2 and "IMMUTABLE" sentence exist | capability→producer | `roles.py::ARCHITECT` "## Rules" 2 and "## Important" | PASS |

## δ — Plan-deviation census  *(leaf; the consumer / integration gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| One recording point every night passes | capability→producer (wired) | `scripts/legibility/nightly.py::run_nightly` `finally` → `_record_trickle_progress` (four early returns precede `evaluate_census_step`) | PASS |
| Per-project state dir; the timer can reach the escalation port | capability→producer (wired) | `scripts/legibility/trickle_state.py::trickle_state_path`; `legibility-trickle@dark_factory.timer` armed as a `systemd --user` unit, no network sandbox | PASS |
| Config block precedent — validation is δ's own | capability→producer (wired) | `scripts/legibility/config.py::Census` (`extra='allow'`, so unknown blocks pass through untyped); `census_trigger.CensusConfig.from_mapping` is the validation shape to imitate | PASS |
| Escalation envelope from a script | capability→producer (wired) | `scripts/legibility/census_trigger.py::post_mcp_tool_call`; `nightly.py::post_escalation` | PASS |
| `blocking` L0 on a synthetic task id reaches a human | capability→producer (wired) | reaper verbatim promotion for `severity != 'info'` (router PRD D6, unchanged); auto-watcher promotes `design_concern` with explicit severity; `__recovery_veto_streak__` sentinel precedent for synthetic ids | PASS |
| Archive bulk reader | capability→producer (wired) | `escalation/src/escalation/queue.py::iter_all_escalation_paths` | PASS |
| Plan bulk-reader precedent; mtime pre-filter possible | capability→producer | `scripts/scan_plan_decision_pairing.py`; `plan.json` rewritten on every write | PASS |
| Record shapes to read | DAG-direction | α (`resolution_class`, `note_to_file`) and β (`step.deviations[]`, `shared.plan_deviation.DEVIATION_KINDS`) upstream; `observation-consumed` arrives with 5221/5223 and is read when present | PASS |
| The count to retire exists | capability→producer | `skills/escalation-watcher/SKILL.md` standing rule → **Limits** ("more than 3 qualifying in one day") | PASS |
| Rate rise fires; flat rate does not | rejection-mechanism | built + bound by δ; positive/negative control fixtures in the boundary-test sketch | PASS (built by δ) |
