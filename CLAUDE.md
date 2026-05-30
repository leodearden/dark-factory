# Dark Factory

Software factory with unified memory + task management. Three subsystems — Graphiti (temporal knowledge graph), Mem0 (vector memory), Taskmaster AI (task management) — unified behind the **fused-memory** MCP server.

## Prerequisites

```bash
# Start backing stores (Neo4j/FalkorDB + Qdrant)
cd fused-memory/docker && docker-compose up -d

# Python environment
cd fused-memory && uv sync

# Required env vars (inherit from shell):
# OPENAI_API_KEY  (for embeddings; ANTHROPIC_API_KEY is NOT needed — agents use OAuth)
```

## Memory Usage

### When to read memory
- **Session start** — search for project context, recent decisions, active conventions
- **Encountering unfamiliar entities** — `get_entity` to understand relationships
- **Before architectural decisions** — search for prior decisions and rationale
- **Tasks with memory_hints** — execute hint queries via `search`, look up hint entities via `get_entity`

### When to write memory
- **Decisions made** — immediately, don't wait until session end
- **Conventions discovered** — coding patterns, naming rules, project norms
- **Session end** — reflect and write observations, summaries of what was accomplished

### Write operations

| Operation | Cost | When to use |
|-----------|------|-------------|
| `add_memory` | 0-3 LLM calls | Discrete, distilled facts — **prefer this** |
| `add_episode` | 5-15 LLM calls | Raw content needing extraction — use sparingly |

### Category routing

| Category | Primary Store | Use for |
|----------|--------------|---------|
| `entities_and_relations` | Graphiti | Facts about things and connections |
| `temporal_facts` | Graphiti | State that changes over time |
| `decisions_and_rationale` | Graphiti | Choices made and why |
| `preferences_and_norms` | Mem0 | Conventions, style rules |
| `procedural_knowledge` | Mem0 | Workflows, how-to steps |
| `observations_and_summaries` | Mem0 | High-level takeaways, session recaps |

## Write-Tagging Convention

Always pass these parameters on write operations:
- **`project_id`**: `"dark_factory"`
- **`agent_id`**: descriptive identifier, e.g. `"claude-interactive"`, `"claude-task-7"`, `"reconciliation-stage-1"`

## Task Routing

All task operations go through **fused-memory MCP tools** — not the Taskmaster CLI or Taskmaster MCP directly. This ensures the TaskInterceptor emits reconciliation events for state transitions.

Use `project_root: "/home/leo/src/dark-factory"` for all task operations.

Status transitions (`done`, `blocked`, `cancelled`, `deferred`) trigger targeted reconciliation automatically.

### Cross-project task dependencies

A task can declare a dependency on a task in **another** project using the qualified `"project_id:task_id"` form (e.g. `"dark_factory:42"`). When `add_dependency` receives a `depends_on` value that contains `:`, it routes the dep to `metadata.external_deps` (a list of canonical `"project_id:task_id"` strings) instead of the integer `dependencies` table — no schema migration required.

```python
# Qualified form → appended to metadata.external_deps
mcp__fused-memory__add_dependency(
    id="<dependent_task_id>",
    depends_on="dark_factory:42",   # project_id:task_id
    project_root="<project_root>",
)
# Bare integer → existing integer dependencies table (unchanged)
mcp__fused-memory__add_dependency(id="<id>", depends_on=13, project_root="<project_root>")
```

The foreign target is **not** verified at write time; existence is resolved at gate time.

**Resolution: `get_external_statuses`**

The scheduler resolves `metadata.external_deps` at each dispatch tick via the read-only fused-memory tool `get_external_statuses(deps: list[str]) -> dict[str, str]`. It takes a list of `"project_id:task_id"` strings, looks each up in the shared fused-memory registry, and returns a status per dep. Unresolvable deps return explicit sentinels:

| Sentinel | Meaning |
|---|---|
| `"unknown_project"` | `project_id` not in the registry |
| `"unknown_task"` | Project known; no top-level task with that id |
| `"malformed"` | Not parseable as `project_id:task_id` |

**Dispatch-time policy**

The gate lives in the **dependent's** scheduler only — it does not affect the upstream project's orchestrator. External deps are checked at dispatch time; they are not re-evaluated after a task has been dispatched.

| Resolved status | Scheduler action |
|---|---|
| `done` | Satisfied — counts toward dispatch |
| `cancelled` | Not satisfied → `_mark_blocked(escalate_to_human=True)` immediately |
| `unknown_project` / `unknown_task` / `malformed` | Not satisfied; grace period then escalate after repeated unresolved cycles |
| Any other live status (`pending`, `in-progress`, …) | Not satisfied; keep waiting |
| Resolver error (transient timeout / server hiccup) | Not satisfied this tick — fail-safe wait, no grace counter increment |

A task is dispatched only when **all** local deps **and** all `metadata.external_deps` are `done`.

## Session Lifecycle

### Starting a session
1. Search memory for project context: `search(query="project overview and current status", project_id="dark_factory")`
2. Check task tree: `get_tasks(project_root="/home/leo/src/dark-factory")`
3. If working on a specific task, check its `memory_hints` and execute the hint queries

### During a session
- Write decisions and discoveries immediately via `add_memory` — don't batch until the end
- Use `search` before making architectural choices to check for prior decisions

### Ending a session
Reflect and write each as a separate `add_memory` call:
- What decisions were made and why
- What conventions were discovered or established
- Brief session summary (what was accomplished, what's left)

Use `/memory` for detailed guidance on writing effective memories.

## Reference

- **Design docs**: `DESIGN.md` (architecture), `fused-memory/docs/reconciliation/` (reconciliation system)
- **Memory skill**: `/memory` — detailed reference for memory operations, categories, search patterns
- **Config**: `fused-memory/config/config.yaml`, `.mcp.json`
