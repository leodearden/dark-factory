# Dark Factory

Software factory with unified memory + task management. Three subsystems — Graphiti (temporal knowledge graph), Mem0 (vector memory), Taskmaster AI (task management) — unified behind the **fused-memory** MCP server.

This file is the agent-facing operating context for sessions working in
this repo. The user-facing documentation lives in `README.md` (entry
point), `SETUP.md` (install), `OPERATIONS.md` (operator runbook),
`ARCHITECTURE.md` (system design), `docs/task-authoring.md` (task metadata
reference), and `CONTRIBUTING.md` — prefer pointing humans there, and
consult those docs yourself rather than re-deriving what they cover.

## Repo Map

Package dirs follow a `<pkg>/src/<pkg>/` double-nesting convention:
`orchestrator/src/orchestrator/`, `fused-memory/src/fused_memory/`,
`escalation/src/escalation/`, `shared/src/shared/`. `skills/` is the
**in-repo** skill source (distinct from `~/.claude/skills`). Other
top-level dirs: `dashboard/` — web UI for task/escalation state;
`scripts/` — operator and CI helper scripts; `hooks/` — git hooks
(pre-commit, pre-merge-commit); `plans/` — design docs and PRDs for
in-flight/past work; `docs/` — reference docs (see `docs/legibility/` for
the confusion codebook).

Every project targeted by Dark Factory — this repo included — must expose
its top-level orchestrator config at
`<project_root>/dark-factory-orchestrator.yaml`. That is the canonical,
required filename: it's what the dashboard's escalation-URL discovery
(`_discover_escalation_urls`) keys on. Legacy spellings (`orchestrator.yaml`,
`orchestrator-config.yaml`, `orchestrator/config.yaml`) are honored only as
a discovery fallback for not-yet-migrated projects, not a supported choice
for new ones.

This repo uses `ruff check` only; `ruff format` is deliberately NOT adopted
and most files are not format-clean. Do not file or perform
formatting-cleanup work — see `CONTRIBUTING.md` section 3 (task 3441).

## Prerequisites

```bash
# Start backing stores (FalkorDB + Qdrant)
cd fused-memory/docker && docker-compose up -d

# Python environment
cd fused-memory && uv sync

# Required env vars (inherit from shell):
# OPENAI_API_KEY  (for embeddings; ANTHROPIC_API_KEY is NOT needed — agents use OAuth)
```

Full fresh-machine walkthrough: `SETUP.md`.

## Memory Usage

### When to read memory
- **Session start** — search for project context, recent decisions, active conventions
- **Encountering unfamiliar entities** — `get_entity` to understand relationships
- **Before architectural decisions** — search for prior decisions and rationale
- **Tasks with memory_hints** — only if you can actually see them. `metadata.memory_hints`
  is currently a **reconciliation-internal** channel: the sole consumer is recon Stage 1's
  context assembler, and no orchestrator-dispatched task role holds `get_task`, so a
  dispatched agent cannot read its own hints. An interactive session (which does hold
  `get_task`) can read and execute them. Dispatched agents get their memory context from
  the briefing's `# Context` block instead. Task **3254** decides whether to wire hints
  through to agents or retire the channel; until it lands, don't plan around them.

### When to write memory
- **Decisions made** — immediately, don't wait until session end
- **Conventions discovered** — coding patterns, naming rules, project norms
- **Session end** — reflect and write observations, summaries of what was accomplished
- **Before writing a gotcha-class `procedural_knowledge` entry** — `search()` first for existing coverage; if a near-duplicate already exists, consolidate into/update it instead of writing a new one. (`fused-memory/scripts/audit_duplicate_memories.py` is the automated backstop sweep for whatever slips through.) `add_memory` now ENFORCES this at write time: a `procedural_knowledge` write matching an existing entry at high similarity is soft-blocked; override with `metadata={'allow_near_duplicate': True}` only for genuinely distinct content.

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

All task operations go through **fused-memory MCP tools** — not the
Taskmaster CLI or Taskmaster MCP directly. This ensures the
TaskInterceptor emits reconciliation events for state transitions. Use
`project_root: "/home/leo/src/dark-factory"` for all task operations on
this repo. Status transitions (`done`, `blocked`, `cancelled`, `deferred`)
trigger targeted reconciliation automatically.

**When filing a task with dependencies, wire them under
`planning_mode=True` → `add_dependency` → `commit_planning`** — a plain
`submit_task` followed by `add_dependency` races the scheduler. Verify a
filed batch with `get_task` (not `search_tasks`, whose corpus excludes
`deferred`/uncommitted tasks).

The full task-authoring reference — field shapes, validation rules, and
dispatch policies for cross-project `"project_id:task_id"` external deps,
`metadata.delivered_checks` capability gates, the `complexity='simple'`
fast path, `task_kind='deterministic'` (before_done, always_escalates,
predicate checks), `metadata.milestone` (dated/delayed), per-task
`metadata.model_overrides` pins, `done_provenance` kinds, and the
metadata vocabulary/census (Tier-A blessed keys, Tier-B canonical
spellings, Tier-C `x_` namespace) — lives in **`docs/task-authoring.md`**.
Consult it before authoring any of those fields; the shapes are
validated at write time and a malformed spec is rejected.

## Model Routing

The orchestrator resolves `(model, effort, budget_usd, max_turns)` for
every LLM invocation through a single layered resolver
(`orchestrator.routing.resolve_route`): per-task
`metadata.model_overrides` pin → first matching `routing.rules` policy
rule → per-role config → role default, each layer validated fail-safe
against `routing.allowed_models` (a dispatch is never blocked by a
routing mis-config; rejections are recorded on the `routing_decision`
event and `task.metadata.routing`). `claude-fable-5` is deliberately not
yet in the stock allowlist. The operator-facing `routing.*` config
reference (rule vocabulary, ladder bumps, per-model ceilings,
`usage_cap.scoped_cap_models`) is in **`OPERATIONS.md` §"Model routing"**;
the task-author pin is in `docs/task-authoring.md`.

## Session Lifecycle

### Starting a session
1. Search memory for project context: `search(query="project overview and current status", project_id="dark_factory")`
2. Check task tree: `get_tasks(project_root="/home/leo/src/dark-factory")`
3. If working on a specific task, read it with `get_task` — and if it carries
   `memory_hints`, execute the hint queries. This works in an **interactive** session;
   an orchestrator-dispatched task agent holds no `get_task` and cannot do it (see
   "When to read memory" above, and task 3254).

### During a session
- Write decisions and discoveries immediately via `add_memory` — don't batch until the end
- Use `search` before making architectural choices to check for prior decisions

### Ending a session
Reflect and write each as a separate `add_memory` call:
- What decisions were made and why
- What conventions were discovered or established
- Brief session summary (what was accomplished, what's left)

Use `/memory` for detailed guidance on writing effective memories.

## Orchestrator Config Reload

`mcp__escalation__reload_config` hot-applies an
`dark-factory-orchestrator.yaml` edit to a **running** orchestrator
without a restart (it always re-reads that process's own
`ORCH_CONFIG_PATH`, never another project's). Green tier (hot-reloadable)
covers per-role models/budgets/max_turns/effort/timeouts/backends,
`routing.*`, steward grace, scheduler + watcher tuning, `review.*`,
`unblock_auto.*`, `verify_env`, and `git.offline_lane_*` leaves; red tier
(restart-only) covers `max_concurrent_tasks`, pool sizes, escalation
host/port, `sandbox.backend`, `project_root`, merge-lane `git.*`
structural fields, and `usage_cap.*`. **Reloaded ≠ everything took
effect** — always read the returned `applied` / `restart_required`
dispositions, not just the top-level `reloaded` flag. Full tier lists and
workflow: **`OPERATIONS.md` §"Config reload vs restart"** and
`plans/config-hot-reload-prd.md`.

## Orchestrator Fleet Redeploy

Three deliberately orthogonal restart mechanisms act on the fleet:
watchdog **liveness** probes (revive a wedged unit immediately, no clock),
the watchdog **staleness** backstop and the merge-landed **coordinator**
(both funnel through `scripts/restart-all-orchestrators.sh --drain` and
share one 8h fleet-deploy clock). Don't conflate them when debugging a
restart, and run `scripts/orchestrator-watchdog.py --report` (strictly
read-only) before manually restarting anything. Full model, `--report`
column reference, and the soak signal to watch:
**`OPERATIONS.md` §"Fleet redeploy & watchdog"**.

## Working in the main checkout

The `project_root` checkout (`/home/leo/src/dark-factory`) is **machine-operated**
— the merge worker, the startup reconciler, and git hooks all act on it
directly, not just interactive agents.

- For a direct-to-main commit under contention, use `git commit --only <path>`
  (not a bare `git commit`) so you don't sweep up unrelated staged/dirty state
  from a concurrent process.
- `pre-commit` runs pyright **only for packages with staged `.py` changes**
  (`hooks/project-checks`, task 2551): a docs-only commit prints
  `pyright skipped (no Python changes)` and is quick. When a commit does stage
  Python, pass `timeout: 300000` (or higher) to `Bash`, or run detached via
  `setsid` and poll, rather than letting the default timeout kill it mid-hook.
- **Never** run `git stash` in **any** dark-factory checkout — `project_root`
  or a `.worktrees/<id>` task worktree. `refs/stash` is a single ref in the
  shared `.git` dir and is *not* per-worktree, so every checkout pushes onto
  the same stack, which the merge worker's advance path also consumes
  (incident `13674d3c68`). A stash you push can be popped out from under you
  by an unrelated process, and a `stash pop` on a clean tree can apply another
  task's WIP into yours. Park WIP as commits on a branch instead.

## Reference

- **User docs**: `README.md` (entry point) · `SETUP.md` (install/onboard)
  · `OPERATIONS.md` (runbook: skills map, merging, resolve_issue, config
  reload, model routing, fleet redeploy, troubleshooting)
  · `ARCHITECTURE.md` (process topology, task lifecycle, agent roles,
  escalation ladder, merge lane) · `docs/task-authoring.md` (task metadata
  reference) · `CONTRIBUTING.md` (conventions, quality gates, git workflow)
- **Design docs**: `DESIGN.md` (fused-memory architecture),
  `fused-memory/src/fused_memory/reconciliation/prompts/` (reconciliation
  stage/judge prompt sources), `RECONCILIATION_PLAN.md`
- **Memory skill**: `/memory` — detailed reference for memory operations, categories, search patterns
- **Config**: `fused-memory/config/config.yaml`, `.mcp.json`
- **Design invariants**: `docs/legibility/design-invariants.md` — five checkable invariants gating `/prd` decompose (G7) and `/review` phase 2
