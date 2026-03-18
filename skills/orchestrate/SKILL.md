---
name: orchestrate
description: "Implement software from a PRD or task tree using the dark-factory orchestrator. Use this skill when the user wants to: run the orchestrator against a PRD, check orchestrator/task status, resolve blocked tasks, resume stalled work, or manage concurrent TDD workflows. Trigger on mentions of PRDs, orchestrating implementation, running tasks through the pipeline, checking task status, or resolving blocked/failed tasks — even if the user doesn't say 'orchestrator' explicitly."
---

# Dark Factory Orchestrator

The orchestrator decomposes a PRD into tasks, then implements each task as a concurrent TDD workflow — planning, coding, verifying, reviewing, and merging — all driven by Claude Code agent instances with module-level locking to prevent conflicts.

## Determine what the user wants

The user will be in one of these situations. Read the request and jump to the matching section:

| Situation | Go to |
|-----------|-------|
| "Implement this PRD" / hands you a .md file | [Run from PRD](#run-from-prd) |
| "What's the status?" / "How are the tasks going?" | [Check Status](#check-status) |
| "Task X is blocked" / "Something failed" | [Resolve Blocks](#resolve-blocks) |
| Tasks already exist in Taskmaster, user wants to execute them | [Resume Existing Tasks](#resume-existing-tasks) |
| Docker/services aren't running, connection errors | Read `references/infrastructure.md` |

---

## Run from PRD

This is the main workflow. The orchestrator will parse the PRD into tasks, tag them with code modules, then execute them concurrently via TDD workflows.

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

2. **Environment** — `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` must be set. The orchestrator inherits them from the shell.

3. **PRD file exists** — confirm the path. If the user described requirements verbally instead of providing a file, write the PRD to a temp file first (use `/tmp/prd-<descriptive-name>.md` or ask where they want it).

### Launch

```bash
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator run --prd <path-to-prd>
```

**Options:**
- `--dry-run` — parse PRD into tasks and tag modules, but don't execute. Useful for reviewing the task decomposition before committing resources.
- `--config <path>` — override default config (at `orchestrator/config.yaml`). Useful for adjusting concurrency, models, or budgets.
- `--verbose` — debug-level logging.

The orchestrator manages its own fused-memory HTTP server lifecycle — it starts the server before work begins and stops it when done. You don't need to start it manually.

### What happens during a run

The orchestrator will:
1. Start fused-memory HTTP server
2. Call `parse_prd` to decompose the PRD into tasks
3. Invoke a module-tagger agent to label each task with the code directories it touches
4. Execute up to 3 tasks concurrently (configurable), each following:
   **PLAN** (architect) → **EXECUTE** (implementer, TDD) → **VERIFY** (pytest/ruff/pyright) → **REVIEW** (5 specialist reviewers) → **MERGE** (to main)
5. Print a summary report with per-task outcomes and costs

Each task gets its own git worktree and branch (`task/<id>`). Merges use `--no-ff` to preserve history.

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
   ls /home/leo/src/dark-factory/../worktrees/
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
| **VERIFY** | "Verification attempts exhausted" | Tests/lint/typecheck fail persistently. Check the worktree for the actual failures: `cd <worktree> && pytest` / `ruff check` / `pyright`. Fix manually or update the task. |
| **REVIEW** | "Review cycles exhausted" with blocking issues | Reviewers found design problems the architect couldn't resolve in 2 replan cycles. Read `.task/reviews/*.json` for the specific issues. May need architectural guidance written to memory. |
| **MERGE** | "Merge conflicts" / "Post-merge verification failed" | Another task modified the same code. Check `git log --oneline main` for recent merges. May need to rebase the task branch manually. |

### Manual resolution workflow

1. **Navigate to the worktree**: `cd /home/leo/src/dark-factory/../worktrees/<task-id>`
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
   git worktree remove ../worktrees/<task-id>
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

## Resume Existing Tasks

The orchestrator CLI currently requires a `--prd` flag — it always parses a PRD to create tasks. If tasks already exist in Taskmaster (from a previous `--dry-run`, a prior interrupted run, or manual creation via `add_task`), you have two options:

### Option A: Re-run with the original PRD

If the PRD file is still available, just re-run. The `parse_prd` call will interact with Taskmaster which handles deduplication based on the existing task state. Completed tasks won't be re-executed (the scheduler only picks up `pending` tasks with satisfied dependencies).

```bash
uv run --project orchestrator orchestrator run --prd <original-prd-path>
```

### Option B: Manual task execution

If there's no PRD (tasks were created manually), manage them through fused-memory MCP tools:

1. **View the task tree**: `get_tasks(project_root="/home/leo/src/dark-factory")`
2. **Check a specific task**: `get_task(id="<id>", project_root="/home/leo/src/dark-factory")`
3. **Reset blocked tasks**: `set_task_status(id="<id>", status="pending", project_root="/home/leo/src/dark-factory")`
4. **Update task details**: `update_task(id="<id>", prompt="<new description>", project_root="/home/leo/src/dark-factory")`
5. **Add dependencies**: `add_dependency(id="<id>", depends_on="<other-id>", project_root="/home/leo/src/dark-factory")`

Then run the orchestrator with a minimal PRD that references the existing work, or implement individual tasks manually using Claude Code's normal workflow.

---

## Configuration

The default config lives at `orchestrator/config.yaml`. Key knobs:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `max_concurrent_tasks` | 3 | Parallel task workflows |
| `max_per_module` | 1 | Tasks per code module (serialization) |
| `max_execute_iterations` | 10 | Implementer retries before blocking |
| `max_verify_attempts` | 5 | Debug-fix cycles before blocking |
| `max_review_cycles` | 2 | Review-replan loops before blocking |
| `models.architect` | opus | Model for planning |
| `models.implementer` | opus | Model for coding |
| `models.reviewer` | sonnet | Model for code review |
| `budgets.implementer` | $10 | Max spend per implementer invocation |
| `test_command` | pytest | Verification test runner |
| `lint_command` | ruff check | Verification linter |

Override via YAML file (`--config`) or environment variables (`ORCH_MAX_CONCURRENT_TASKS=5`, `ORCH_MODELS__ARCHITECT=sonnet`).
