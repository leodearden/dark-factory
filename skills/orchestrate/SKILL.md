---
name: orchestrate
description: "Implement software from a PRD or task tree using the dark-factory orchestrator. Use this skill when the user wants to: run the orchestrator against a PRD, check orchestrator/task status, resolve blocked tasks, resume stalled work, or manage concurrent TDD workflows. Trigger on mentions of PRDs, orchestrating implementation, running tasks through the pipeline, checking task status, or resolving blocked/failed tasks — even if the user doesn't say 'orchestrator' explicitly."
---

# Dark Factory Orchestrator

The orchestrator implements tasks as concurrent TDD workflows — planning, coding, verifying, reviewing, and merging — driven by Claude Code agent instances with module-level locking to prevent conflicts.

Task decomposition happens here, in the interactive session, where you have full codebase context. The orchestrator handles concurrent execution. This separation exists because taskmaster's `parse_prd` uses a weaker model without codebase access, which produces decent but incomplete decompositions (missed integration tasks, wrong dependency graphs, no awareness of what's already implemented). Decomposing interactively produces better tasks with less cleanup.

## Important: do not implement tasks directly

This skill is about driving the **orchestrator** — not about implementing tasks yourself in this session. When the user says "do task 7" or "run the tasks", they mean launch the orchestrator to execute them via its TDD workflow pipeline (worktrees, architect/implementer agents, verification, review, merge). Never implement task code directly in the interactive session unless the user explicitly asks you to bypass the orchestrator.

## Determine what the user wants

The user will be in one of these situations. Read the request and jump to the matching section:

| Situation | Go to |
|-----------|-------|
| "Implement this PRD" / hands you a .md file | [Decompose PRD](#decompose-prd), then [Execute Tasks](#execute-tasks) |
| "Do task X" / "Run task 7" / "Execute the tasks" | [Execute Tasks](#execute-tasks) |
| "What's the status?" / "How are the tasks going?" | [Check Status](#check-status) |
| "Task X is blocked" / "Something failed" | [Resolve Blocks](#resolve-blocks) |
| "An agent escalated" / escalation notification | [Handle Escalations](#handle-escalations) |
| Tasks already exist in Taskmaster, user wants to execute them | [Execute Tasks](#execute-tasks) (skip decomposition) |
| Docker/services aren't running, connection errors | Read `references/infrastructure.md` |

---

## Decompose PRD

This is the critical step. You decompose the PRD into tasks interactively, using full codebase context, then write them to Taskmaster via `add_task`. Do not use taskmaster's `parse_prd` — it lacks the context to do this well.

### 1. Read the PRD and the codebase

Read the PRD file. Then explore the codebase to understand:
- What already exists (don't create tasks for things already implemented)
- Module boundaries and directory structure (for accurate module tagging)
- Test infrastructure (what test frameworks, patterns, fixtures exist)
- Import graphs and dependencies between modules (for accurate task dependencies)

Use Read, Grep, Glob to build this understanding. Check `git log` for recent changes that might affect the plan.

### 2. Propose tasks

Present the user with a task table for review:

| # | Title | Modules | Dependencies | Description |
|---|-------|---------|-------------|-------------|
| 1 | ... | [dir1, dir2] | — | ... |
| 2 | ... | [dir1] | 1 | ... |

Each task should:
- **Be scoped to a coherent unit of work** — something one agent can plan, implement, and verify in a single TDD cycle
- **Have accurate module tags** — the actual directories it will modify (used for concurrency locking, so get this right)
- **Have correct dependencies** — based on real import relationships and build order, not guesses
- **Include e2e/integration tasks** — these are easy to forget but critical. If the PRD involves multiple components, add a task that wires them together and tests the full pipeline
- **Skip what's already done** — check the codebase. Don't create tasks for existing functionality

Keep descriptions concrete — the orchestrator's architect agent will read them to produce a TDD plan, so vague descriptions lead to vague plans that block at execution time.

### 3. Get user review

Present the table and dependency graph. Ask the user to review:
- Are tasks scoped correctly? (too big → split, too small → merge)
- Are dependencies right?
- Anything missing? (integration tests, config, docs)
- Any tasks that are already done?

Iterate until the user is satisfied.

### 4. Write tasks to Taskmaster

Determine the correct `project_root` — this is where `.taskmaster/tasks/tasks.json` lives. It may be the repo root or a subdirectory.

Write each task via fused-memory MCP tools:

```
add_task(
  title="<title>",
  description="<detailed description>",
  project_root="<project_root>"
)
```

Then set dependencies:

```
add_dependency(id="2", depends_on="1", project_root="<project_root>")
```

After all tasks are written, verify with `get_tasks` and show the user the final state.

### 5. Proceed to execution

Jump to [Execute Tasks](#execute-tasks).

---

## Execute Tasks

### Pre-flight

Before launching, verify:

1. **Services are reachable** — run a quick health check:
   ```bash
   # FalkorDB speaks Redis protocol, not HTTP — use a raw PING
   (echo PING; sleep 0.5) | nc -w2 localhost 6379 2>&1 | grep -q PONG && echo "FalkorDB: ok" || echo "FalkorDB: DOWN"
   # Qdrant has an HTTP readiness endpoint
   curl -sf http://localhost:6333/readyz > /dev/null 2>&1 && echo "Qdrant: ok" || echo "Qdrant: DOWN"
   ```
   If either is down, read `references/infrastructure.md` for startup instructions.

2. **Environment** — `OPENAI_API_KEY` must be set (used for embeddings). The orchestrator's Claude agents authenticate via OAuth (Max subscription), not an API key — `ANTHROPIC_API_KEY` is **not** required.

3. **Tasks exist** — verify with `get_tasks`. If the task tree is empty, go to [Decompose PRD](#decompose-prd) first.

### Launch

The orchestrator CLI still requires `--prd` even when tasks already exist (it calls `parse_prd` but Taskmaster will see existing tasks and skip decomposition). Pass the original PRD path:

```bash
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator run --prd <path-to-prd>
```

**Options:**
- `--dry-run` — verify task tree and module tags, but don't execute workflows.
- `--config <path>` — override default config (at `orchestrator/config.yaml`). Useful for adjusting concurrency, models, or budgets.
- `--verbose` — debug-level logging.

The orchestrator manages its own fused-memory HTTP server lifecycle — it starts the server before work begins and stops it when done. You don't need to start it manually.

The orchestrator also starts an **escalation MCP server** on port 8102 (configurable). Agents use this to escalate issues. The interactive session connects to the same server to resolve escalations — it's pre-configured in `.mcp.json` as the `escalation` server, so escalation tools (`mcp__escalation__resolve_issue`, etc.) are available in the interactive session during a run.

### What happens during a run

The orchestrator will:
1. Start fused-memory HTTP server and escalation MCP server
2. Check usage cap status (multi-account failover if cap hit at startup)
3. Recover crashed tasks from surviving worktrees — if a prior run crashed mid-task, it resumes from the last completed plan step rather than starting over
4. Call `parse_prd` (no-op if tasks already exist in Taskmaster)
5. Tag tasks with code modules via a classifier agent (if not already tagged)
6. Execute tasks concurrently (default 12, configurable via `max_concurrent_tasks`), each following:
   **PLAN** (architect) → **EXECUTE** (implementer, TDD) → **VERIFY** (pytest/ruff/pyright) → if verify fails: **DEBUG** (up to 5 cycles) → **REVIEW** (5 specialist reviewers) → **MERGE** (to main, with post-merge verification)
7. If an agent encounters a problem outside its scope, it **escalates** rather than retrying.
   Escalations are pushed to the handler session via a background watcher. See [Handle Escalations](#handle-escalations).
8. Print a summary report with per-task outcomes and costs

Each task gets its own git worktree and branch (`task/<id>`). Merges use `--no-ff` to preserve history.

The **debugger** is a distinct agent role invoked automatically on each verify failure. It receives the failure report (test output, lint errors, type errors) and makes targeted fixes. The verify→debug loop repeats up to `max_verify_attempts` times (default 5) before the task blocks.

After merge, **post-merge verification** re-runs the full verification suite on main. If it fails, the merge is automatically reverted and the task blocks — this catches integration issues that only appear after combining with other tasks' changes.

### Interpreting the output

The final report shows:
```
Orchestrator run complete: 5/7 tasks done
  Blocked: 2
  Total cost: $47.32
  Duration: 2025-03-16T10:00:00 → 2025-03-16T11:23:45
```

If any tasks are blocked, jump to [Resolve Blocks](#resolve-blocks).

---

## Check Status

```bash
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator status
```

This queries the fused-memory task tree and displays each task's status and module assignments:
```
  [pending     ] 1: Set up authentication module [backend, tests]
  [in-progress ] 2: Add REST endpoints [backend, server]
  [done        ] 3: Create database schema [backend]
  [blocked     ] 4: Implement caching layer [backend]
```

You can also query tasks directly via fused-memory MCP tools for more detail:

```
get_tasks(project_root="/home/leo/src/dark-factory")
get_task(id="4", project_root="/home/leo/src/dark-factory")
```

---

## Resolve Blocks

Tasks block at specific workflow stages. The approach depends on where it got stuck.

### Identify the block

1. Check status to find blocked tasks
2. Look at the task's worktree for artifacts — the orchestrator preserves worktrees for blocked tasks (cleaned up only on success):
   ```bash
   ls /home/leo/src/dark-factory/.worktrees/
   ```
3. Inside a blocked task's worktree, check `.task/` for diagnostics:
   - `.task/plan.json` — the TDD plan (shows which steps completed)
   - `.task/iterations.jsonl` — execution log
   - `.task/reviews/` — reviewer feedback (if it reached review stage)

### Common block patterns

| Stage | Symptom | Resolution |
|-------|---------|------------|
| **PLAN** | "architect failed" or no `plan.json` | Task description may be too vague. Update the task with more detail via `update_task`, reset to pending, and re-run. |
| **EXECUTE** | "Execution iterations exhausted" | Implementer couldn't complete all plan steps in 10 iterations. Check `iterations.jsonl` for where it got stuck. May need to simplify the task or split it. |
| **VERIFY** | "Verification attempts exhausted" | Tests/lint/typecheck fail persistently. Check the worktree for the actual failures: `cd <worktree> && pytest` / `ruff check` / `pyright`. Fix manually or update the task. **Caveat**: verification currently runs repo-wide commands by default — pre-existing errors in unrelated modules can block tasks. Check whether the failures are in the task's own modules or elsewhere. |
| **REVIEW** | "Review cycles exhausted" with blocking issues | Reviewers found design problems the architect couldn't resolve in 2 replan cycles. Read `.task/reviews/*.json` for the specific issues. May need architectural guidance written to memory. |
| **MERGE** | "Merge conflicts" / "Post-merge verification failed" | Another task modified the same code. Check `git log --oneline main` for recent merges. May need to rebase the task branch manually. |

### Manual resolution workflow

1. **Navigate to the worktree**: `cd /home/leo/src/dark-factory/.worktrees/<task-id>`
2. **Diagnose**: read `.task/plan.json`, check test output, review `git log`
3. **Fix**: make changes directly in the worktree
4. **Verify**: run `pytest`, `ruff check`, `pyright`
5. **Merge manually** (if the fix is good):
   ```bash
   cd /home/leo/src/dark-factory
   git merge --no-ff task/<task-id>
   ```
6. **Update task status**:
   ```
   set_task_status(id="<task-id>", status="done", project_root="/home/leo/src/dark-factory")
   ```
7. **Clean up worktree**:
   ```bash
   git worktree remove .worktrees/<task-id>
   git branch -d task/<task-id>
   ```

### Writing context to help future runs

If the block was caused by missing architectural context, write it to memory so the orchestrator's agents can find it next time:

```
add_memory(
  content="<the decision or convention that was missing>",
  category="decisions_and_rationale",  # or "preferences_and_norms" for conventions
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```

---

## Handle Escalations

During an orchestrator run, agents can escalate issues they can't solve at their scope. Unlike blocks (which are detected post-run), escalations are **pushed in real-time** — you'll be notified as they happen.

### Dismissing stale escalations

Escalations from prior runs persist on disk and can block tasks in new runs. Before starting a new run, check for and dismiss stale escalations:

```
get_pending_escalations()
```

If there are pending escalations from a prior session (check timestamps), dismiss them:

```
resolve_issue(escalation_id="<id>", resolution="Stale from prior run", terminate=true)
```

### Setting up the escalation watcher

After launching the orchestrator, start the watcher as a background task:

```bash
uv run --project escalation python -m escalation.watcher \
  --queue-dir /home/leo/src/dark-factory/data/escalations &
```

The watcher uses inotify to wait (zero CPU) for new escalation files. When one arrives, it prints the escalation JSON to stdout and exits — this triggers a background task completion notification in Claude Code, which is your signal to handle it. After resolving, re-arm the watcher with the same command.

**Watcher options:**
- `--task-id <id>` — filter to a specific task
- `--ntfy-url <url>` — send push notifications via ntfy.sh (for AFK monitoring)

### Reading an escalation

Each escalation includes:
- **id** — format `esc-{task_id}-{seq}` (e.g., `esc-42-1`)
- **task_id** — which task's agent escalated
- **severity** — `blocking` (task paused, waiting for you) or `info` (FYI, agent continued)
- **category** — `scope_violation`, `design_concern`, `cleanup_needed`, `dependency_discovered`, `risk_identified`, `infra_issue`, or `task_failure` (auto-escalation when workflow hits iteration/verify/review limits)
- **summary** / **detail** — what the agent found
- **suggested_action** — what the agent thinks should happen
- **worktree** — path to the task's worktree (useful for diagnosis)
- **workflow_state** — what stage the agent was in when it escalated

You can also fetch a specific escalation by ID:
```
get_escalation(escalation_id="esc-42-1")
```

### Resolving an escalation

These are MCP tools on the escalation server (prefixed `mcp__escalation__` in Claude Code). The server is pre-configured in `.mcp.json`, so these tools are available in the interactive session during a run.

```
resolve_issue(
  escalation_id="<id from the notification>",
  resolution="<your resolution — this text is injected into the agent's briefing when the task resumes>",
  terminate=false
)
```

The `resolution` text should be actionable instructions for the re-invoked agent. Examples:
- "Scope expanded: you now have write access to `crates/reify-compiler`. Update the trait impl in `src/lib.rs` to match the new type signature."
- "This is a known limitation in M2 scope. Skip the duplicate-ID check — it will be addressed in task 14."
- "The test infrastructure issue is fixed on main. Rebase your branch with `git rebase main`."

Set `terminate=true` if the task should be abandoned rather than resumed.

### Listing pending escalations

To see all unresolved escalations (e.g., if you missed a notification):

```
get_pending_escalations()
```

Or filtered to a specific task:

```
get_pending_escalations(task_id="7")
```

### Common escalation patterns

| Category | Typical cause | Resolution approach |
|----------|--------------|-------------------|
| `scope_violation` | Agent hit bwrap sandbox boundary | Evaluate if scope expansion is warranted. If yes, update the task's modules via `update_task` and resolve with expanded scope instructions. If no, guide the agent to an alternative approach. |
| `design_concern` | Agent found a design issue outside task scope | Assess severity. Create a follow-up task if needed, resolve with "proceed, this is tracked in task N". |
| `dependency_discovered` | Task depends on code from an unfinished task | Check if the dependency task is in progress. If so, resolve with "wait" or requeue. If not, may need to add a dependency. |
| `infra_issue` | Test framework, build tool, or environment problem | Fix the infrastructure issue, then resolve with instructions for the agent. |
| `task_failure` | Auto-escalation: workflow hit iteration/verify/review limits | Check the worktree artifacts to understand why. Either fix the issue and resolve, or terminate if the task needs to be redesigned. |

---

## Resume Existing Tasks

If tasks already exist from a prior session, first assess their state before re-running.

### 1. Audit the task tree

```
get_tasks(project_root="<project_root>")
```

Check each task against the actual codebase state. Prior sessions may have completed work without updating task statuses. Use `git log`, `grep`, and test runs to verify what's actually done:

- **Tasks marked pending but actually done** → `set_task_status(id="<id>", status="done", ...)`
- **Tasks with stale descriptions** → `update_task(id="<id>", prompt="<better description>", ...)`
- **Missing tasks** (e.g., integration/e2e) → `add_task(...)` and wire dependencies
- **Wrong dependencies** → `remove_dependency` / `add_dependency`

### 2. Execute

Once the task tree is accurate, jump to [Execute Tasks](#execute-tasks). The orchestrator's scheduler only picks up `pending` tasks with all dependencies satisfied, so completed tasks are automatically skipped.

---

## Configuration

The default config lives at `orchestrator/config.yaml`. Config is auto-discovered: if no `--config` flag is passed, the CLI checks `cwd/config.yaml`, then `cwd/orchestrator/config.yaml`. YAML values support `${VAR_NAME}` and `${VAR_NAME:default}` environment variable expansion.

### Core settings

| Setting | config.yaml | What it controls |
|---------|-------------|-----------------|
| `max_concurrent_tasks` | 12 | Parallel task workflows |
| `max_per_module` | 1 | Tasks per code module (serialization) |
| `max_execute_iterations` | 10 | Implementer retries before blocking |
| `max_verify_attempts` | 5 | Debug-fix cycles before blocking |
| `max_review_cycles` | 2 | Review-replan loops before blocking |
| `lock_depth` | 4 | Module path depth for lock normalization |
| `test_command` | pytest | Verification test runner |
| `lint_command` | ruff check ... | Verification linter |
| `type_check_command` | pyright (per-package) | Verification type checker |
| `escalation.port` | 8102 | Escalation MCP server port |
| `escalation.queue_dir` | data/escalations | Escalation queue directory |
| `sandbox.enabled` | false | Enable bwrap filesystem sandbox |

### Per-role settings

Each role has model, budget, max turns, reasoning effort, and backend:

| Role | Model | Budget | Turns | Effort | Backend |
|------|-------|--------|-------|--------|---------|
| architect | opus | $5 | 50 | max | claude |
| implementer | sonnet | $10 | 80 | max | claude |
| debugger | sonnet | $5 | 50 | max | claude |
| reviewer | sonnet | $2 | 30 | medium | claude |
| merger | opus | $5 | 50 | max | claude |
| module_tagger | (sonnet) | ($2) | (30) | medium | claude |

### Module overrides

Subprojects can override verification commands by placing an `orchestrator.yaml` in their directory (e.g., `fused-memory/orchestrator.yaml`). This scopes `test_command`, `lint_command`, and `type_check_command` to that subproject when tasks touch its modules.

### Multi-account failover

The `usage_cap` section configures automatic failover between Max subscription accounts when usage caps are hit. Accounts are tried in order; when one is capped, the orchestrator switches to the next and starts a background probe to detect when the capped account resets.

Override via YAML file (`--config`) or environment variables (`ORCH_MAX_CONCURRENT_TASKS=5`, `ORCH_MODELS__ARCHITECT=sonnet`).

---

## Known Issues & Workarounds

These are active issues that may affect orchestrator runs. Check the task tree for fix status.

### Verification runs repo-wide (task 45)

The verify step runs `ruff check`, `pyright`, and `pytest` with repo-wide scope by default. Pre-existing lint/type errors in modules outside a task's scope cause verification to fail even when the task's own code is clean. **Workaround**: ensure main is clean before starting a run (`ruff check && pyright` should pass). Module-scoped verification is planned but not yet implemented.

### Worktree imports resolve from main tree (task 46)

Worktrees share the main tree's `.venv`. Python imports resolve from the installed (main) package, not the worktree's modified source. New methods or changed signatures in worktree code fail at test time with `AttributeError` or assertion mismatches. **Workaround**: if a task blocks at verify with import errors for code it just wrote, manually verify the code is correct, merge to main, and confirm tests pass post-merge.
