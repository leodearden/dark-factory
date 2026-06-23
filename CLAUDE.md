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

**Deterministic deploy and gate tasks use this same dep mechanism** — including
cross-project `"project_id:task_id"` deps. The older convention of filing
deploy capstones in `dark_factory` with a `dark_factory`-internal dependency
— a workaround for an external-dep gate bug fixed by tasks 1854/1855/1799 — is
**retired**. Use a `task_kind='deterministic'` deploy or gate task with normal
deps instead. See "Deterministic task kind" below.

### Simple-task fast path (`metadata.complexity`)

Set `metadata.complexity = "simple"` to route a task to the single-agent
SIMPLE_TASK fast path (one Sonnet agent explores, plans, edits, and commits;
the architect+implementer pair is skipped, but verify/review/merge still run).
The only meaningful value is `"simple"` — absent or any other value routes to
the full architect path.

**When to declare `"simple"`:** the change is a single coherent edit — docs or
comments, a rename, a localized behaviour-preserving refactor, a typo/wording
fix, a one-spot bug fix — that needs **no new abstraction and no cross-module
design**, and you can name the target file(s). A `simple` task may be
high-priority and may touch several files/modules, as long as the *change* is
mechanically simple. **When unsure, omit it** — the full path is the safe
default, and a mis-declared task simply falls back to the architect.

**Hard-blocker veto:** if the task description contains a hard-blocker token
(`migration`, `architecture`, `integration test`, `design ... new`,
`implement ... new feature`), the fast path is vetoed even if
`complexity='simple'` is set.

**Hard escape:** `metadata.force_full_path = true` always forces the full
architect path regardless of `complexity`.

### Deterministic task kind (`task_kind='deterministic'`)

Set `task_kind='deterministic'` on `submit_task` to skip the LLM pipeline
entirely (no worktree, no branch, no agent, no diff) and route to the
**`DeterministicRunner`** — a small state machine that runs an optional
committed action, escalates born-at-L2 when required, and marks the task
`done` once both are satisfied. Dispatch eligibility uses the same dep-gate as
every other task.

**`task_kind`** is a first-class `submit_task` parameter (`'normal'` default
| `'deterministic'`), persisted to `metadata.task_kind`.

**`metadata.before_done`** — committed-script reference (set at `submit_task`):

```
{
  script: "<repo-relative path>",  # must exist & be executable
  args: [],                         # list[str], default []
  env: {},                          # dict[str,str], default {}
  cwd: "<project_root>",            # default: project_root
  timeout_secs: 120,                # int, required; runner kills + escalates on timeout
  target_unit: None                 # str|None; None → cross-unit (no self-kill)
}
```

**`metadata.always_escalates`** (`bool`, default `false`) — file a born-at-L2
escalation after the action completes (or immediately if no action); task goes
`blocked` until resolved via `resolve_issue`.

**Field-combo presets:**

| `before_done` | `always_escalates` | Behaviour | Use for |
|---|---|---|---|
| present | `false` | run action; escalate only on failure; else `done` | **auto-deploy** |
| present | `true` | run action; then escalate born-at-L2; `done` after `resume` | act-then-ask |
| absent | `true` | escalate born-at-L2 immediately; `done` after `resume` | **pure gate** |
| absent | `false` | **rejected** at `submit_task` (ill-formed no-op) | — |

**Validation (enforced at `submit_task`):** `task_kind='deterministic'` with
`before_done=None` and `always_escalates=false` is **rejected** ("ill-formed
no-op"). `before_done` set on a `normal` task is also **rejected** ("before_done
is only valid on deterministic tasks").

**Born-at-L2 escalations:** all filed with `severity ∈ {critical, urgent}` and
sentinel `agent_role='orchestrator-deterministic'`; the server retains `level=2`
(no L0→L1→L2 climb). The task goes `blocked` while the L2 is open (quiescence
guard — no re-dispatch, no churn).

**Blocking vs detached self-kill — *determined*, not a knob:**
- `before_done.target_unit` equals this orchestrator's own unit → detached
  `systemd-run --user` with `--on-failure` (done = `scheduled`; the dispatching
  orchestrator is **not** killed).
- `before_done.target_unit` differs from own unit (or is `None`) → blocking
  subprocess + fresh `MainPID`/`ActiveEnterTimestamp` verify against a
  pre-run baseline (done = `deployed-and-verified`).

**Runner stamps** (written by `DeterministicRunner`, never author-supplied):
`before_done_ran_at`, `before_done_verified_at`, `gate_escalated_at`,
`done_provenance` (`kind='deterministic-deploy'` cross-unit;
`kind='deterministic-deploy-scheduled'` self-restart).

**Dep convention:** deterministic deploys and gates use **normal** deps —
including cross-project `project_id:task_id` deps. See "Cross-project task
dependencies" above.

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
