# Dark Factory

Software factory with unified memory + task management. Three subsystems — Graphiti (temporal knowledge graph), Mem0 (vector memory), Taskmaster AI (task management) — unified behind the **fused-memory** MCP server.

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

### Milestone tasks (dated / delayed)

`metadata.milestone` is an orthogonal, time-based dispatch gate — **not** a
new `task_kind`. Setting it holds a task out of dispatch until its time
trigger fires; once fired, the task dispatches through its normal path
unchanged. It is allowed on **both** `normal` and `deterministic` tasks
(orthogonal to `task_kind`), so it can gate a normal LLM-agent task, a
deterministic predicate check (below), or a pure human gate (a `dated`
milestone on a deterministic task with `always_escalates=True` and no
`before_done`).

Two time modes:
- **`dated`** — fires at an explicit wall-clock instant.
- **`delayed`** — fires `after_secs` seconds after the task's own
  dependencies (local deps **and** `metadata.external_deps` — see
  "Cross-project task dependencies" above) are satisfied.

**`metadata.milestone`** (set at `submit_task`; validated by the shared
`Milestone` model):

```
{
  mode: "dated" | "delayed",
  at: "<ISO-8601 datetime>",  # required iff mode="dated"; datetime.fromisoformat-parseable; forbidden iff delayed
  after_secs: <int>,          # required and > 0 iff mode="delayed"; forbidden iff dated
}
```

The `mode`-conditional fields are a strict *iff*: a `dated` milestone must
not also carry `after_secs`, and a `delayed` milestone must not also carry
`at`. This is enforced by the shared `Milestone` model
(`shared/src/shared/task_metadata.py`) and re-checked by the `submit_task`
guard — a malformed spec (`delayed` with no `after_secs`, `dated` with an
unparseable `at`) is rejected at submit with a structured `ValidationError`
and never persisted.

**Scheduler-stamped fields** (never author-supplied):
`milestone_deps_satisfied_at` (`delayed` mode only — the frozen-once
wall-clock UTC anchor, stamped the first tick all of the task's dependencies
go `done`) and, only on a predicate check failure, `gate_escalated_at`
(shared with "Deterministic task kind" above). A task author never sets
either field.

**Predicate exit-code contract (`before_done.kind`):** `before_done` gains a
discriminator, `'deploy' | 'predicate'` (default `'deploy'` — every existing
deterministic task is byte-identical). `kind='deploy'` is the existing
act-then-done/ask behaviour above; `kind='predicate'` is check-then-done-or-
escalate, new. In `kind='predicate'` mode the `DeterministicRunner` runs the
script and decides by **exit code only** — it parses no output:

| Exit code | Outcome |
|---|---|
| `0` | task `done`, `done_provenance.kind='deterministic-milestone'` (the script's stdout tail carried as `note`) |
| non-`0` | born-at-L2 `milestone_check_failed` escalation (detail carries the exit code + stdout tail) + task `blocked` |
| timeout | born-at-L2 `infra_issue` escalation (existing timeout path) + task `blocked`, **no** `gate_escalated_at` stamp |

A predicate is **read-only**, so resolving the escalation (`resume`) safely
**re-runs** the check rather than trusting the resolution blindly — it drives
to `done` or re-files `milestone_check_failed` from whatever the check
reports this time. `kind='predicate'` **forbids** `before_done.target_unit`
(no unit to verify — no systemd inspect / PID-verify on a read-only check)
and forbids top-level `always_escalates=True` (predicate escalation is
already conditional on a non-zero exit); the `submit_task` guard enforces
both. A predicate never stamps `before_done_ran_at` / `before_done_verified_at`
— re-running a read-only check on crash-resume is harmless.

**Frozen-once delayed-anchor semantics:** for `mode='delayed'`, the scheduler
stamps `milestone_deps_satisfied_at` in a per-tick sweep the first tick ALL
of the task's dependencies (local and external) are `done`; the timer then
runs `after_secs` from that anchor. The anchor is a **persisted wall-clock
UTC ISO string**, not an in-memory/monotonic timer, so a multi-day delay
**survives orchestrator restarts** — it's recomputed each tick from the
persisted value, with no in-memory timer to lose. It is frozen-once: a later
dependency regression never rewrites the anchor or restarts the timer.
Dispatch still re-checks *live* deps at eligibility, though — a milestone
fires only when both the timer has elapsed **and** deps are currently
satisfied, so a regressed dependency still withholds dispatch even after the
timer elapses. A `delayed` milestone with no dependencies has them trivially
satisfied at filing, so its timer starts immediately. `mode='dated'` simply
withholds while `now < at`. A malformed or unrecognized `metadata.milestone`
value fails safe — withhold dispatch indefinitely, with a one-time WARNING
log — rather than dispatch early or crash the tick.

**Exemplar** (autonomous — no human involvement unless the check fails):
"one week after task X lands, check merge flakiness is under 5%, else
escalate." A deterministic task depending on X, with:

```
{
  "milestone": {"mode": "delayed", "after_secs": 604800},  # 7 days
  "before_done": {
    "kind": "predicate",
    "script": "scripts/check_merge_flakiness.sh",
    "args": ["--window-days", "7", "--threshold", "0.05", "--value", "0.03"],
    "timeout_secs": 120
  },
  "always_escalates": false
}
```

(`--value` is normally populated by a wrapper that computes the measured
rate from CI logs — out of scope here; the script only owns the threshold
comparison, per its header.) The anchor stamps when X reaches `done`;
`after_secs` later the check runs. Exit `0` → `done`
(`deterministic-milestone`); non-zero → `milestone_check_failed` at L2 (same
`agent_role` / severity / level as "Born-at-L2 escalations" above).

**Cross-refs:** predicate mode is a new `before_done.kind` alongside
`deploy` and reuses the `DeterministicRunner` — see "Deterministic task
kind" above. The delayed anchor waits on the same local +
`metadata.external_deps` dependency gate described in "Cross-project task
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

## Orchestrator Config Reload

`mcp__escalation__reload_config` hot-applies an `orchestrator.yaml` edit to
a **running** orchestrator process without a restart. It takes no
arguments — it always re-reads that process's own `ORCH_CONFIG_PATH`,
never another project's.

**Green tier** (hot-reloadable): per-role `models` / `budgets` /
`max_turns` / `effort` / `timeouts` / `backends`, steward grace
(`steward_completion_timeout`, `steward_lifetime_budget`), scheduler +
watcher tuning, `review.*` checkpoint knobs, `unblock_auto.*`,
`verify_env`, and the `git.offline_lane_*` leaf tunables.

**Red tier** (restart-only — edit is accepted but has no effect until
restart): `max_concurrent_tasks`, pool sizes / `verify_runners`,
`escalation` bind host/port, `sandbox.backend`, `project_root`, and the
merge-lane `git.*` structural fields.

**Reloaded ≠ everything took effect** — always read the returned
`applied` / `restart_required` dispositions, not just the top-level
`reloaded` flag. See `plans/config-hot-reload-prd.md` for the full
allowlist and `skills/orchestrate/SKILL.md`'s "Reload Config (vs Restart)"
section for the operator workflow.

## Working in the main checkout

The `project_root` checkout (`/home/leo/src/dark-factory`) is **machine-operated**
— the merge worker, the startup reconciler, and git hooks all act on it
directly, not just interactive agents.

- For a direct-to-main commit under contention, use `git commit --only <path>`
  (not a bare `git commit`) so you don't sweep up unrelated staged/dirty state
  from a concurrent process.
- `pre-commit` runs pyright 3x — pass `timeout: 300000` (or higher) to `Bash`
  for commit commands, or run detached via `setsid` and poll, rather than
  letting the default timeout kill it mid-hook.
- **Never** run `git stash` in `project_root`: the stash stack is consumed by
  the merge worker's advance path (incident `13674d3c68`), so a stash you push
  can be popped out from under you by an unrelated process. Park WIP as
  commits on a branch instead.

## Reference

- **Design docs**: `DESIGN.md` (architecture), `fused-memory/src/fused_memory/reconciliation/prompts/` (reconciliation stage/judge prompt sources)
- **Memory skill**: `/memory` — detailed reference for memory operations, categories, search patterns
- **Config**: `fused-memory/config/config.yaml`, `.mcp.json`
- **Design invariants**: docs/legibility/design-invariants.md — five checkable invariants gating /prd decompose (G7) and /review phase 2
