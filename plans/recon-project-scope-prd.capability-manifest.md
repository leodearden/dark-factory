# Capability manifest — recon-project-scope (M4)

PRD: `plans/recon-project-scope-prd.md`. Bindings verified against main on 2026-07-06
(session claude-prd-recon-project-scope). Evidence commands run from repo root unless
noted; pyright = repo-pinned 1.1.408 via `cd fused-memory && uv run pyright`.

## task-α — Define ProjectId/ProjectRoot/ProjectScope in models/scope.py

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `typing.NewType` + `dataclass(frozen=True, slots=True)` compose and type-check under repo pyright | rejection-check fixture (below) type-checked with 0 unexpected errors on the valid lines | PASS |
| pyright **rejects** transposed/bare-str/frozen-mutation usage (the PRD's negative assertion) | rejection-check: authored fixture with `ProjectScope(root, pid)`, `ProjectScope('dark_factory', '/x')`, `takes_root(scope.project_id)`, `takes_root('/plain/str')`, `scope.project_root = root`; ran `uv run pyright --outputjson`; **7/7 expected `error` diagnostics observed to fire** (arg-type ×6, frozen-assign ×1) | PASS |
| Type home exists and is import-cycle-safe | grep:`fused-memory/src/fused_memory/models/scope.py` (owns `resolve_project_id`, `build_known_projects_map`); module imports only stdlib+pydantic — nothing from `reconciliation/`; `utils/validation.py:188` documents it deliberately does NOT import models.scope (reverse edge safe) | PASS |
| Runtime validation raise-path producible by this task alone | `__post_init__` is delivered by α itself; no upstream producer needed | PASS |

## task-β — Thread ProjectScope through harness + BaseStage; delete injection

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `ProjectScope` type available | producer:task-α upstream (direct dep) | PASS |
| Registry factory to convert | grep:`reconciliation/harness.py:504-539` `_known_project_root_for` (task 1143), callers `:1770`, `:2799` — wired into the production cycle entry path | PASS |
| Injection sites exist to delete | grep:`harness.py:1831-1833` and `:2522-2524` (`stage.project_id = ...` / `stage.project_root = ...` / `stage.known_projects = ...`) | PASS |
| `''` defaults exist to delete | grep:`stages/base.py:60-61` (`self.project_id: str = ''`, `self.project_root: str = ''`) | PASS |
| `self.stages` pre-build is vestigial (safe to delete) | grep: only production consumers are `harness.py:435` (assign) and `:1371` (`_propagate_escalation_queue(self.stages)`); every cycle uses fresh `_make_stages()` instances (`:1826`, `:2504`) which self-propagate at `:604`; no non-test `harness.stages` reader elsewhere (repo grep 2026-07-06) | PASS |
| Task-1163 snapshot semantics pinned (must not regress) | grep:`tests/test_harness.py:5585` `test_remediation_pass_uses_threaded_project_root_over_registry`, `:5647` `test_remediation_uses_threaded_project_root_not_mutated_registry`; deliberate-omission comment `harness.py:2431-2436` | PASS |
| Grep-signal targets exist today (so "no matches" after is a real diff) | `grep -n "stage\.project_root = " harness.py` → `:1832`, `:2523` fire today | PASS |

## task-γ — Scope stage-level sweep/guard helpers; delete falsy-root degrades

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `scope` available on stages | producer:task-β upstream (direct dep; β delivers required-scope BaseStage + properties) | PASS |
| Pair-taking helper exists | grep:`stages/task_knowledge_sync.py:1424-1429` `_sweep_terminal_task_flag_markers(memory_service, taskmaster, project_root: str, project_id: str, run_id: str, ...)` — the adjacent-pair transposition hazard | PASS |
| Falsy-root degrade branches exist to delete | grep:`task_knowledge_sync.py:1496` (`if not taskmaster or not project_root: return 0`), `:2388` (`if not project_root or not tasks`), `:2992` (docstring'd silent skip), `memory_consolidator.py:634-645` (`_build_project_root_directive` falsy branch), call-site comment `:2705-2708` | PASS |
| taskmaster-None degrade retained is testable | existing sweep tests in `tests/test_stages.py` construct with None taskmaster (grep: `_sweep_terminal_task_flag_markers` in tests/test_stages.py) | PASS |

## task-δ — Thread ProjectScope through targeted.py handlers and sweeps

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `ProjectScope` type available | producer:task-α upstream (direct dep) | PASS |
| Validated entry point exists | grep:`targeted.py:206` `require_project_root(project_root)` inside `reconcile_task`; raising validator at `utils/validation.py:149-152` | PASS |
| Pair-taking handler signatures exist (the hazard) | grep:`targeted.py:378`, `:622`, `:729` (kw), `:1143` `(self, task_id: str, project_id: str, project_root: str, ...)`; sweeps `:794-1141` | PASS |
| External boundary safe to leave unchanged | grep:`middleware/task_interceptor.py:841` call site passes `project_id=resolve_project_id(project_root)` — project_id derived FROM root, transposition structurally impossible at the boundary this task keeps | PASS |
| End-to-end consumer signal producible | `tests/test_task_interceptor.py` exercises interceptor→`reconcile_task` today (grep: reconcile_task in that file); δ leaves its assertions unchanged | PASS |

## Summary

No FAIL bindings. No numeric bounds asserted (no floor checks applicable). One
rejection-mechanism claim (pyright transposition rejection) bound by live observation,
not assumption. Batch clear to queue.
