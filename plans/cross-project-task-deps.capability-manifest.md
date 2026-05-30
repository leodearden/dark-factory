# Capability Manifest — cross-project-task-deps

Mechanizes G3+G6 for the PRD `plans/cross-project-task-deps-prd.md`. One block per task; each capability the task's signal asserts is bound to evidence. Any FAIL value blocks the batch. All bindings below resolve PASS (the change is wiring of existing substrate plus one new read tool produced upstream of its consumers).

Greek labels match the PRD decomposition. Task IDs filled in after filing (see the decompose hand-back / `prd_task_label` metadata).

## α — get_external_statuses MCP tool  *(intermediate; has its own signal)*
- `resolve_project_id` / `build_known_projects_map` (project_id→root registry) → `grep:fused-memory/src/fused_memory/models/scope.py:117,154` wired on main — **PASS**
- per-project status read path → existing `get_statuses` backend (`sqlite_task_backend.py` per-project read) — **PASS**
- `unknown_task` distinguishable (project known, task absent) → per-project DB query returns empty for an absent top-level id — **PASS**

## β — add_dependency qualified-id → metadata.external_deps  *(intermediate; has its own signal)*
- metadata write/read path → `update_task(append=true)` / append-safe metadata merge exists; task wire shape carries `metadata` — **PASS**
- field-population (anti-sentinel) → producer (`add_dependency`) writes a real non-empty list into `metadata.external_deps`; verified through `get_task` read path — **PASS**
- integer-table path unchanged → existing bare-int `add_dependency` validation (`sqlite_task_backend.py:1042`) untouched — **PASS**

## γ — scheduler _deps_satisfied extension  *(intermediate → δ)*
- `get_external_statuses` capability → `producer:α`, in the transitive dependency closure, upstream of γ — **PASS** (DAG-direction: γ depends-on α)
- escalation pathway → `_mark_blocked(escalate_to_human=True)` / `_check_*_thrash` shape on main — **PASS**
- per-tick task-tree re-evaluation hook → `scheduler.acquire_next()` reads the full tree each tick (`scheduler.py:2071`) — **PASS**

## δ — integration gate: cross-project dispatch end-to-end  *(LEAF)*
- `get_external_statuses` → `producer:α` upstream (via γ) — **PASS**
- `metadata.external_deps` write/read → `producer:β` upstream — **PASS**
- scheduler gate + escalation policy → `producer:γ` upstream — **PASS**
- DAG-direction: δ depends-on β,γ (and transitively α); no producer is downstream — **PASS**
- no numeric bound / closed-form claim asserted (tooling domain) → G6 branches 1/2 N/A — **PASS**

## ε — dashboard render external deps + upstream status  *(LEAF)*
- reads `metadata.external_deps` → `producer:β` upstream — **PASS**
- reads upstream status → `producer:α` upstream — **PASS**
- no-synthetic-data rule honored (show the sentinel, never a fabricated status) → design constraint, enforced in the task — **PASS**

## ζ — docs  *(LEAF; companion correction)*
- documents shipped surfaces (`get_external_statuses`, `metadata.external_deps`, qualified `add_dependency`) → `producer:δ` upstream (document what landed) — **PASS**
- docs-only; no runtime capability asserted — N/A
