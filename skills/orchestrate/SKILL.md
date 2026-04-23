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

When the user invokes /orchestrate, they expect it to operate on the project they are currently in — *not* on dark-factory by default. The orchestrator binary now refuses to start without an explicit target (`--config` flag or `ORCH_CONFIG_PATH` env var) — there is no auto-discovery and no default. Identify the target project before any other action.

### Step 1 — capture the user's cwd

Before any other action, run `pwd` and treat the result as the **target project root** for the entire session. Do not assume it is dark-factory.

```bash
pwd
# e.g. /home/leo/src/reify   →   TARGET_PROJECT = /home/leo/src/reify
```

### Step 2 — find the target project's config (`TARGET_CONFIG`)

First, check whether `ORCH_CONFIG_PATH` is already set in the environment — if direnv loaded the project's `.envrc`, this is automatic and you're done:

```bash
echo "${ORCH_CONFIG_PATH:-unset}"
# If a path is shown: that IS your TARGET_CONFIG. Verify it matches TARGET_PROJECT.
# If "unset": continue below.
```

If unset, find the config file in the target project. Filenames vary across projects (no auto-discovery — every project chose its own name); check all common locations:

```bash
ls "$TARGET_PROJECT"/orchestrator.yaml \
   "$TARGET_PROJECT"/orchestrator-config.yaml \
   "$TARGET_PROJECT"/config.yaml \
   "$TARGET_PROJECT"/orchestrator/config.yaml 2>/dev/null
```

Known locations for the three current projects:

| Project | TARGET_CONFIG |
|---------|---------------|
| dark-factory | `/home/leo/src/dark-factory/orchestrator/config.yaml` |
| reify | `/home/leo/src/reify/orchestrator.yaml` |
| autopilot-video | `/home/leo/src/autopilot-video/orchestrator-config.yaml` |

Verify the file actually points at the target:

```bash
grep -E '^project_root|project_id' "$TARGET_CONFIG"
# Expect: project_root: "/home/leo/src/<target>"
#         project_id: "<target>"
```

If no config exists for the target, **stop and ask the user** — the project may need an orchestrator config created before it can be run. See `references/project-setup.md` for the schema.

### Step 3 — use the target everywhere for the rest of the session

Once you've identified `TARGET_PROJECT` and `TARGET_CONFIG`, every command and MCP call below must use them — `--config "$TARGET_CONFIG"` (or `ORCH_CONFIG_PATH="$TARGET_CONFIG"`) is required on every orchestrator invocation:

| Where | Use |
|-------|-----|
| Launch / status command | `cd /home/leo/src/dark-factory` (binary lives there) **and** `--config "$TARGET_CONFIG"` |
| `project_root="..."` in fused-memory MCP calls (`get_tasks`, `submit_task`, `resolve_ticket`, `set_task_status`, `add_dependency`, `update_task`, etc.) | `"$TARGET_PROJECT"` |
| Worktree inspection paths | `"$TARGET_PROJECT"/.worktrees/<task-id>` |
| `project_id` in `add_memory` writes | the `project_id` from `TARGET_CONFIG` (e.g. `"reify"`, `"dark_factory"`, `"autopilot_video"`) |
| Manual git merge target | `"$TARGET_PROJECT"` (the target's `main` branch, not dark-factory's) |

There are no exceptions — even when the target *is* dark-factory, `--config` (or `ORCH_CONFIG_PATH`) must be set. See `references/project-setup.md` for `.envrc`/direnv ergonomics.

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

This is the critical step. You decompose the PRD into tasks interactively, using full codebase context, then write them to Taskmaster via `submit_task` + `resolve_ticket`. Do not use taskmaster's `parse_prd` — it lacks the context to do this well.

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

Write each task via fused-memory MCP tools using the two-phase pattern:

> **Note:** The `add_task` facade is deprecated and will be removed — always use the two-phase pattern shown below.

```
# Phase 1: submit — returns immediately with a ticket id
submit_result = submit_task(
    title="<title>",
    description="<detailed description>",
    project_root="$TARGET_PROJECT",
    priority="<medium|high>",
    metadata={
        "source": "prd-decomposition",
        "spawn_context": "parse_prd",
        "modules": ["<path/to/module>"],
    },
)
ticket = submit_result["ticket"]

# Phase 2: block until the curator decides (default 115 s)
# "combined" is a normal outcome when PRD decomposition re-covers ground already in the task tree
resolve = resolve_ticket(ticket=ticket, project_root="$TARGET_PROJECT")

if resolve["status"] == "created":
    task_id = resolve["task_id"]           # new task
elif resolve["status"] == "combined":
    task_id = resolve["task_id"]           # merged into existing task — normal, not an error
elif resolve["status"] == "failed":
    # reason determines how to proceed:
    # server_restart → retry the submit_task + resolve_ticket pair ONCE with the same
    #   metadata (original ticket is dead; a fresh submit is the only path forward).
    #   PRD-decomposition tasks do not set escalation_id/suggestion_hash, so the R4
    #   idempotency gate does not fire — the curator may de-duplicate via "combined"
    #   if the duplicate matches an existing task; otherwise a second task will be
    #   created and the user can merge them via the task tree.
    # timeout → first retry resolve_ticket(ticket=same_ticket, ...) with the same ticket —
    #   the worker may still be processing. Only re-submit_task if that retry returns
    #   unknown_ticket or expired. Same dedup caveat as server_restart if re-submit needed.
    # unknown_ticket | server_closed | expired → terminal: surface the reason to the
    #   user and skip this task (user can resubmit manually after investigating).
    handle_failure(resolve["reason"])
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

0. **Target project is identified** — you must already have run `pwd` and located `TARGET_CONFIG` per the [Critical: identify the target project FIRST](#critical-identify-the-target-project-first) section. If you haven't, do that now. If you skip it, the orchestrator will refuse to start and emit an educational error pointing here.

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

The orchestrator binary is invoked from `/home/leo/src/dark-factory` (that's where `uv run --project orchestrator` resolves). `--config` (or `ORCH_CONFIG_PATH`) selects the target project and is **required on every invocation** — no exceptions, no auto-discovery, no defaults.

```bash
cd /home/leo/src/dark-factory

# Run existing tasks against the target project
uv run --project orchestrator orchestrator run --config "$TARGET_CONFIG"

# Or decompose a PRD first, then run
uv run --project orchestrator orchestrator run --config "$TARGET_CONFIG" --prd <path-to-prd>

# Equivalent form using ORCH_CONFIG_PATH (e.g. when direnv loaded the project's .envrc):
ORCH_CONFIG_PATH="$TARGET_CONFIG" uv run --project orchestrator orchestrator run
```

If you omit both `--config` and `ORCH_CONFIG_PATH`, the orchestrator exits 1 with an educational error message pointing at `references/project-setup.md`. There is no silent fallback — this is the hard guard against the cross-project execution incident that lost work.

**Options:**
- `--config <path>` — **required** unless `ORCH_CONFIG_PATH` is set. Selects the target project (sets `project_root` and `fused_memory.project_id`). When both are set, `--config` wins.
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

## Stop Orchestrator

To stop a running orchestrator gracefully, send SIGTERM to the **python3 orchestrator process** — not the bash wrapper or the `uv` process.

### Find the right PID

```bash
# List orchestrator processes with their configs
pgrep -af 'orchestrator run --config'
```

Look for the `python3` (or `orchestrator`) process whose `--config` matches your target. The output shows a chain like `bash → uv → orchestrator`; you want the innermost `orchestrator` PID.

### Send SIGTERM

```bash
kill <orchestrator_pid>
```

The orchestrator handles SIGTERM gracefully:
1. The main loop is interrupted
2. In-flight agent tasks are cancelled
3. The `finally` block runs: metrics are finalized, MCP server is stopped, merge worker and escalation server are shut down
4. Process exits

Task results are persisted **incrementally** as each task completes (not batched at the end), so completed work is never lost even on ungraceful termination.

### Verify shutdown

```bash
# Confirm the process tree is gone
pgrep -af 'orchestrator run --config' | grep "$TARGET_CONFIG"
```

If children (agent subprocesses) are orphaned, kill them by PID. Do **not** use `pkill` with broad patterns — it may hit other projects' orchestrators.

---

## Check Status

Identify the target project first (see [Critical: identify the target project FIRST](#critical-identify-the-target-project-first)), then query its status. The `status` subcommand also requires `--config` (or `ORCH_CONFIG_PATH`):

```bash
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator status --config "$TARGET_CONFIG"
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
| **MERGE (halted)** | "Merge queue halted" / `wip_conflict` or `unmerged_state` escalation blocking all other merges / merge outcomes `wip_halted`, `done_wip_recovery`, `wip_recovery_no_advance`, `unmerged_state` | Exactly one escalation owns the halt (tracked on the merge worker as `_halt_owner_esc_id`). Resolve **that specific escalation** via `/escalation-watcher` or `mcp__escalation__resolve_issue` to un-halt the queue. For `wip_conflict`, check the recovery branch named in the detail (`wip/recovery-<task>-<ts>`); for `unmerged_state`, inspect `project_root` with `git status` and resolve UU/AA/DD markers. Do NOT manually toggle the halt flag — ownership is the single source of truth. |

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
   set_task_status(id="<task-id>", status="done", project_root="$TARGET_PROJECT", done_provenance={"commit": "<merge-commit-sha>"})
   ```
   Use `{"commit": "<sha>"}` when a merge commit contains the landed work (the normal case — read the SHA from `git log -1 --format=%H` after the merge). Use `{"note": "<one-sentence explanation>"}` for fast-forward merges or when the work was covered by a sibling task and no single commit applies.
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
- **Add missing tasks** (e.g., integration/e2e) → `submit_task(...)` + `resolve_ticket(...)` and wire dependencies
- **Fix wrong dependencies** → `remove_dependency` / `add_dependency`

### 2. Execute

Once the task tree is accurate, jump to [Execute Tasks](#execute-tasks). The orchestrator's scheduler only picks up `pending` tasks with all dependencies satisfied, so completed tasks are automatically skipped.

---

## Configuration

The config file is selected via `--config <path>` or `ORCH_CONFIG_PATH=<path>` — there is **no auto-discovery from cwd**. The orchestrator binary refuses to start without one of these set, by design. See `references/project-setup.md` for the rationale, the schema for new project configs, and `.envrc`/direnv ergonomics.

YAML values in the loaded config support `${VAR_NAME}` and `${VAR_NAME:default}` environment variable expansion.

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
