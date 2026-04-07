---
name: orchestrate
description: "Implement software from a PRD or task tree using the dark-factory orchestrator. Use this skill when the user wants to: run the orchestrator against a PRD, check orchestrator/task status, resolve blocked tasks, resume stalled work, or manage concurrent TDD workflows. Trigger on mentions of PRDs, orchestrating implementation, running tasks through the pipeline, checking task status, or resolving blocked/failed tasks — even if the user doesn't say 'orchestrator' explicitly."
---

# Dark Factory Orchestrator

The orchestrator implements tasks as concurrent TDD workflows — planning, coding, verifying, reviewing, and merging — driven by Claude Code agent instances with module-level locking to prevent conflicts.

Task decomposition happens here, in the interactive session, where you have full codebase context. The orchestrator handles concurrent execution. This separation exists because taskmaster's `parse_prd` uses a weaker model without codebase access, which produces decent but incomplete decompositions (missed integration tasks, wrong dependency graphs, no awareness of what's already implemented). Decomposing interactively produces better tasks with less cleanup.

## Important: do not implement tasks directly

This skill is about driving the **orchestrator** — not about implementing tasks yourself in this session. When the user says "do task 7" or "run the tasks", they mean launch the orchestrator to execute them via its TDD workflow pipeline (worktrees, architect/implementer agents, verification, review, merge). Never implement task code directly in the interactive session unless the user explicitly asks you to bypass the orchestrator.

## Critical: identify the target project FIRST

The orchestrator binary lives in `/home/leo/src/dark-factory/orchestrator` and is **always** invoked from there (because `uv run --project orchestrator` resolves the package path). But the **target project** — the codebase whose tasks the orchestrator operates on — is determined entirely by the **config file**, which sets `project_root` and `fused_memory.project_id`.

When the user invokes /orchestrate, they expect it to operate on the project they are currently in — *not* on dark-factory by default. Running it on the wrong project will silently execute tasks against the wrong codebase. This is the single most common foot-gun in this skill. Fix it before doing anything else.

### Step 1 — capture the user's cwd

Before any other action, run `pwd` and treat the result as the **target project root** for the entire session. Do not assume it is dark-factory.

```bash
pwd
# e.g. /home/leo/src/reify   →   TARGET_PROJECT = /home/leo/src/reify
```

### Step 2 — locate the target project's orchestrator config

The orchestrator's config-discovery only checks `cwd/config.yaml` and `cwd/orchestrator/config.yaml` (note: it does **not** auto-discover `cwd/orchestrator.yaml`), and it always runs with cwd = dark-factory. **For any project other than dark-factory you must pass `--config` explicitly.** Find the target's config:

```bash
ls "$TARGET_PROJECT"/orchestrator.yaml \
   "$TARGET_PROJECT"/config.yaml \
   "$TARGET_PROJECT"/orchestrator/config.yaml 2>/dev/null
```

Verify the file actually points at the target:

```bash
grep -E '^project_root|project_id' <found-config>
# Expect: project_root: "/home/leo/src/<target>"
#         project_id: "<target>"
```

If no config exists for the target, **stop and ask the user**. Do not silently fall back to dark-factory's config — that is exactly the bug this section exists to prevent.

### Step 3 — use the target everywhere for the rest of the session

Once you've identified `TARGET_PROJECT` and `TARGET_CONFIG`, every command and MCP call below must use them — never substitute `/home/leo/src/dark-factory` unless that *is* the target:

| Where | Use |
|-------|-----|
| Launch / status command | `cd /home/leo/src/dark-factory` (binary lives there) **and** `--config "$TARGET_CONFIG"` |
| `project_root="..."` in fused-memory MCP calls (`get_tasks`, `add_task`, `set_task_status`, `add_dependency`, `update_task`, etc.) | `"$TARGET_PROJECT"` |
| Worktree inspection paths | `"$TARGET_PROJECT"/.worktrees/<task-id>` |
| `project_id` in `add_memory` writes | the `project_id` from `TARGET_CONFIG` (e.g. `"reify"`, `"dark_factory"`) |
| Manual git merge target | `"$TARGET_PROJECT"` (the target's `main` branch, not dark-factory's) |

The only time you can omit `--config` and use `/home/leo/src/dark-factory` for `project_root` is when the user is actually working on dark-factory itself (cwd = `/home/leo/src/dark-factory`).

## Determine what the user wants

The user will be in one of these situations. Read the request and jump to the matching section:

| Situation | Go to |
|-----------|-------|
| "Implement this PRD" / hands you a .md file | [Decompose PRD](#decompose-prd), then [Execute Tasks](#execute-tasks) |
| "Do task X" / "Run task 7" / "Execute the tasks" | [Execute Tasks](#execute-tasks) |
| No arguments given (bare `/orchestrate`) | [Execute Tasks](#execute-tasks) — run all pending tasks |
| "What's the status?" / "How are the tasks going?" | [Check Status](#check-status) |
| "Task X is blocked" / "Something failed" | [Resolve Blocks](#resolve-blocks) |
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

Use the `TARGET_PROJECT` you captured at the top of the session as `project_root`. (`.taskmaster/tasks/tasks.json` lives under that directory.) **Never** hardcode `/home/leo/src/dark-factory` here unless the user is actually in dark-factory.

Write each task via fused-memory MCP tools:

```
add_task(
  title="<title>",
  description="<detailed description>",
  project_root="$TARGET_PROJECT"
)
```

Then set dependencies:

```
add_dependency(id="2", depends_on="1", project_root="$TARGET_PROJECT")
```

After all tasks are written, verify with `get_tasks(project_root="$TARGET_PROJECT")` and show the user the final state.

### 5. Proceed to execution

Jump to [Execute Tasks](#execute-tasks).

---

## Execute Tasks

### Pre-flight

Before launching, verify:

0. **Target project is identified** — you must already have run `pwd` and located `TARGET_CONFIG` per the [Critical: identify the target project FIRST](#critical-identify-the-target-project-first) section. If you haven't, do that now. If you skip it, you will run dark-factory tasks by default.

1. **Services are reachable** — run a quick health check:
   ```bash
   # FalkorDB speaks Redis protocol, not HTTP — use a raw PING
   (echo PING; sleep 0.5) | nc -w2 localhost 6379 2>&1 | grep -q PONG && echo "FalkorDB: ok" || echo "FalkorDB: DOWN"
   # Qdrant has an HTTP readiness endpoint
   curl -sf http://localhost:6333/readyz > /dev/null 2>&1 && echo "Qdrant: ok" || echo "Qdrant: DOWN"
   ```
   If either is down, read `references/infrastructure.md` for startup instructions.

2. **Environment** — `OPENAI_API_KEY` must be set (used for embeddings). The orchestrator's Claude agents authenticate via OAuth (Max subscription), not an API key — `ANTHROPIC_API_KEY` is **not** required.

3. **Tasks exist** — verify with `get_tasks(project_root="$TARGET_PROJECT")`. If the task tree is empty, go to [Decompose PRD](#decompose-prd) first. Confirm the returned tasks actually belong to the target project — if you see dark-factory task titles when the user is in reify, your `project_root` is wrong; go fix step 3 above before launching.

### Launch

The orchestrator binary is invoked from `/home/leo/src/dark-factory` (that's where `uv run --project orchestrator` resolves), and `--config` selects which project it operates on:

```bash
cd /home/leo/src/dark-factory

# Target = a non-dark-factory project (the common case): --config is REQUIRED
uv run --project orchestrator orchestrator run --config "$TARGET_CONFIG"

# Or with a PRD to decompose first
uv run --project orchestrator orchestrator run --config "$TARGET_CONFIG" --prd <path-to-prd>

# Target = dark-factory itself: --config can be omitted (auto-discovery picks orchestrator/config.yaml)
uv run --project orchestrator orchestrator run
```

If you omit `--config` while the user is in any project other than dark-factory, the orchestrator will load dark-factory's config and execute dark-factory tasks. This is the bug to avoid — always pass `--config` explicitly unless you have just verified the target *is* dark-factory.

**Options:**
- `--config <path>` — selects the target project (sets `project_root` and `fused_memory.project_id`). **Required for any project other than dark-factory.** Auto-discovery only checks `cwd/config.yaml` and `cwd/orchestrator/config.yaml`, both inside dark-factory.
- `--prd <path>` — path to PRD markdown file. If omitted, skips PRD parsing and runs existing pending tasks.
- `--dry-run` — verify task tree and module tags, but don't execute workflows. Useful for confirming you're pointed at the right project before committing to a real run.
- `--verbose` — debug-level logging.

The fused-memory HTTP server runs as a **systemd service** and must already be running before launching the orchestrator. Do **not** start, restart, or stop fused-memory without explicit user permission. Verify it's up: `curl -sf http://localhost:8002/health`.

The orchestrator also starts an **escalation MCP server** on port 8102 (configurable). Agents use this to escalate issues they can't resolve. **Escalations are handled in a separate session** — this session should ignore them entirely. Use the `/escalation-watcher` skill in a dedicated session to monitor and resolve escalations during a run.

### What happens during a run

The orchestrator will:
1. Start escalation MCP server (fused-memory must already be running as a systemd service)
2. Check usage cap status (multi-account failover if cap hit at startup)
3. Recover crashed tasks from surviving worktrees — if a prior run crashed mid-task, it resumes from the last completed plan step rather than starting over
4. Call `parse_prd` (no-op if tasks already exist in Taskmaster)
5. Tag tasks with code modules via a classifier agent (if not already tagged)
6. Execute tasks concurrently (default 12, configurable via `max_concurrent_tasks`), each following:
   **PLAN** (architect) → **EXECUTE** (implementer, TDD) → **VERIFY** (pytest/ruff/pyright) → if verify fails: **DEBUG** (up to 5 cycles) → **REVIEW** (5 specialist reviewers) → **MERGE** (to main, with post-merge verification)
7. If an agent encounters a problem outside its scope, it **escalates** rather than retrying.
   Escalations are handled in a **separate session** — do not attempt to handle them here.
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

Identify the target project first (see [Critical: identify the target project FIRST](#critical-identify-the-target-project-first)), then query its status. The `status` subcommand also takes `--config`:

```bash
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator status --config "$TARGET_CONFIG"
# (omit --config only when the target is dark-factory itself)
```

This queries the fused-memory task tree and displays each task's status and module assignments:
```
  [pending     ] 1: Set up authentication module [backend, tests]
  [in-progress ] 2: Add REST endpoints [backend, server]
  [done        ] 3: Create database schema [backend]
  [blocked     ] 4: Implement caching layer [backend]
```

You can also query tasks directly via fused-memory MCP tools for more detail. Always pass `project_root="$TARGET_PROJECT"` — passing dark-factory's path here will return the wrong project's tasks:

```
get_tasks(project_root="$TARGET_PROJECT")
get_task(id="4", project_root="$TARGET_PROJECT")
```

If the returned tasks don't look like the project the user is in (e.g. you see dark-factory titles when the user is in reify), your `project_root` is wrong — re-run step 1 of "identify the target project" before continuing.

---

## Resolve Blocks

Tasks block at specific workflow stages. The approach depends on where it got stuck.

### Identify the block

1. Check status to find blocked tasks (using `--config "$TARGET_CONFIG"`)
2. Look at the task's worktree for artifacts — the orchestrator preserves worktrees for blocked tasks (cleaned up only on success). Worktrees live under the **target** project, not dark-factory:
   ```bash
   ls "$TARGET_PROJECT"/.worktrees/
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

All paths below operate on the **target** project (`$TARGET_PROJECT`), not dark-factory.

1. **Navigate to the worktree**: `cd "$TARGET_PROJECT"/.worktrees/<task-id>`
2. **Diagnose**: read `.task/plan.json`, check test output, review `git log`
3. **Fix**: make changes directly in the worktree
4. **Verify**: run the target project's verify commands (look these up in `$TARGET_CONFIG` — Rust projects use `cargo test`/`cargo clippy`, Python projects use `pytest`/`ruff`/`pyright`, etc. — do not assume Python tooling)
5. **Merge manually** (if the fix is good):
   ```bash
   cd "$TARGET_PROJECT"
   git merge --no-ff task/<task-id>
   ```
6. **Update task status**:
   ```
   set_task_status(id="<task-id>", status="done", project_root="$TARGET_PROJECT")
   ```
7. **Clean up worktree** (from inside `$TARGET_PROJECT`):
   ```bash
   git worktree remove .worktrees/<task-id>
   git branch -d task/<task-id>
   ```

### Writing context to help future runs

If the block was caused by missing architectural context, write it to memory so the orchestrator's agents can find it next time. Use the **target project's** `project_id` (read it from `$TARGET_CONFIG` under `fused_memory.project_id` — e.g. `"reify"`, `"dark_factory"`), not a hardcoded value:

```
add_memory(
  content="<the decision or convention that was missing>",
  category="decisions_and_rationale",  # or "preferences_and_norms" for conventions
  project_id="<target project_id from TARGET_CONFIG>",
  agent_id="claude-interactive"
)
```

---

---

## Resume Existing Tasks

If tasks already exist from a prior session, first assess their state before re-running. Make sure you've identified the target project (see top section) — querying with the wrong `project_root` will return a different project's task tree and silently lead you astray.

### 1. Audit the task tree

```
get_tasks(project_root="$TARGET_PROJECT")
```

Review the task tree to understand the current state. **Do not change the status of existing tasks** — they may have been set by the user or another session. If a task's status looks wrong, ask the user before changing it.

You may:
- **Add missing tasks** (e.g., integration/e2e) → `add_task(...)` and wire dependencies
- **Fix wrong dependencies** → `remove_dependency` / `add_dependency`

### 2. Execute

Once the task tree is accurate, jump to [Execute Tasks](#execute-tasks). The orchestrator's scheduler only picks up `pending` tasks with all dependencies satisfied, so completed tasks are automatically skipped.

---

## Configuration

The dark-factory default config lives at `orchestrator/config.yaml`. Config is auto-discovered relative to the orchestrator's cwd (which is `/home/leo/src/dark-factory`): if no `--config` flag is passed, the CLI checks `cwd/config.yaml`, then `cwd/orchestrator/config.yaml`. YAML values support `${VAR_NAME}` and `${VAR_NAME:default}` environment variable expansion.

**Important consequence for non-dark-factory projects**: auto-discovery only finds dark-factory's own configs. To run against any other project (e.g. reify, with its config at `/home/leo/src/reify/orchestrator.yaml`), you **must** pass `--config <path>` explicitly. See [Critical: identify the target project FIRST](#critical-identify-the-target-project-first) for the full discovery procedure. Note also that the auto-discovery names are `config.yaml` / `orchestrator/config.yaml` — a file named `orchestrator.yaml` (singular, in repo root) is not auto-discovered and must be passed via `--config`.

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
