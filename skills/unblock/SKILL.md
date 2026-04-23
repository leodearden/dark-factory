---
name: unblock
description: "Unblock a stuck orchestrator task — whether blocked on review issues, merge failures, or agent escalations. Use this skill when the user mentions a blocked or stuck task, wants to resolve review issues in a worktree, handle an escalation for a specific task, or says things like 'unblock task 64', 'task 107 is stuck', 'fix the review issues', or '/unblock <number>'. Also trigger when the user references a specific task number alongside words like 'blocked', 'stuck', 'failed', 'escalated', or 'review issues' — even if they don't say 'unblock' explicitly."
---

# Unblock Task

You're unblocking an orchestrator task that's stuck. The task is either **blocked** (failed reviews, merge conflicts, or verification exhaustion — task status is `blocked`, worktree preserved) or **escalated** (agent found an issue outside its scope and paused via the escalation MCP — task status may still be `in-progress`, but the agent is waiting for a resolution).

Your job: triage the issues, discuss them with the user, fix what needs fixing now, defer what can wait, and get the task either cleanly merged or its escalation resolved.

## Two competing goals

**A: Get unblocked** without ugly hacks or undue scope creep.
**B: Keep quality high** — the codebase has stringent standards.

The critical decisions are *how much to do* and *what to do now (this session) vs later (via new orchestrator tasks)*. Blockers get fixed this session. Everything else gets queued.

---

## Step 0: Locate the task

Extract the task number from the user's message. Set these for the rest of the workflow:
- `TASK_ID` — the number
- `PROJECT_ROOT` — the current working directory (where `.taskmaster/` and `.worktrees/` live)
- `WORKTREE` — `<PROJECT_ROOT>/.worktrees/<TASK_ID>/`

**Check the worktree exists.** If `.worktrees/<TASK_ID>/` is not found, tell the user:
> I can't find a worktree for task `<TASK_ID>` at `<WORKTREE>/`

and stop. There's nothing to unblock without a worktree.

---

## Step 1: Gather context

Collect all available information in parallel:

### 1a. Worktree artifacts
Read from `<WORKTREE>/.task/`:
- `metadata.json` — task identity, base commit
- `plan.json` — TDD plan: which steps completed, which are pending, design decisions
- `iterations.jsonl` — execution log: what each agent iteration attempted and accomplished
- `reviews/*.json` — all reviewer verdicts and issues (there may be up to 5: architect, test analyst, robustness, performance, reuse auditor)

### 1b. Escalations
Query the escalation MCP:
```
get_pending_escalations(task_id="<TASK_ID>")
```

### 1c. Task status
```
get_task(id="<TASK_ID>", project_root="<PROJECT_ROOT>")
```

### 1d. Git state
In the worktree:
- `git log --oneline -10` — recent commits on the task branch
- `git diff main..HEAD --stat` — scope of changes
- Whether the branch can cleanly rebase on current main

---

## Step 2: Deep analysis

This is where you determine the nature and severity of each issue. The goal is to understand each issue well enough to classify it and recommend a course of action.

**Use an agent team.** Spawn parallel Explore agents — one per issue (or per cluster of related issues). Each agent should:

1. **Read the specific code** at the locations referenced in the issue
2. **Understand the surrounding architecture** — what module is this in, what are its responsibilities, what depends on it, what patterns does the codebase use in similar situations
3. **Consult fused-memory** for prior decisions about this area:
   ```
   search(query="<relevant architectural topic>", project_id="<project_id>")
   ```
4. **Check for related issues** in other reviewers' feedback or other tasks

### For each issue, determine:

**Architecture, code quality, or both?**
- Architecture: structural problems, wrong abstractions, missing layers, coupling
- Code quality: naming, error handling, edge cases, test coverage, style
- Both: a surface-level code quality issue that reveals a deeper architectural tension

**If it looks like pure code quality, probe deeper.** A missing error handler might mean the error path isn't designed. A naming issue might reveal confused responsibilities. A test gap might mean the interface is untestable by design. Don't take code quality issues at face value — ask whether they're symptoms of something structural.

**Clear-cut or complex?**
- Clear-cut: one right fix, no real alternatives
- Complex: genuine uncertainty about what's best — either *now*, or *in general*, or *in the long run*

**Blocker or nice-for-later?**
- Blocker: can't safely merge or continue without resolving this
- Nice-for-later: real issue, but safe to defer to the backlog

---

## Step 3: Present findings

Present the user with a structured summary. This is a decision point — they need enough context to make good calls quickly.

### Blockers (must fix before merge)

For each blocker, labeled B1, B2, ...:

> **B1: [Short title]** *[architecture | code quality | both]*
> [If clear-cut]: One or two sentences describing the fix.
> [If complex]:
> - Option A: [description] — [trade-off]
> - Option B: [description] — [trade-off]
> - Recommended: [A or B], because [reason]

### Non-blockers (recommend deferring)

For each non-blocker, labeled S1, S2, ...:

> **S1: [Short title]** — [one-sentence description]. Modules: [which modules it touches].

Recommend queuing all non-blockers. Group them by module overlap so the user can see which ones would naturally become tasks together.

### If there are pending escalations

Present each escalation with its severity, category, summary, and the agent's suggested action. Recommend whether to resolve-and-resume, terminate-and-reschedule, or fix-manually-and-merge.

### Open questions

Finish with a numbered list of every decision the user needs to make:

1. **[Question]** — Recommended: [answer]. Reason: [why].
2. **[Question]** — Recommended: [answer]. Reason: [why].

**Wait for the user.** They may approve your recommendations, override some, ask for more detail, or reclassify issues. Iterate until they say "do it" or similar.

---

## Step 4: Execute

When the user approves, proceed in this order:

### 4.1: Triage non-blockers

Before queuing, check each non-blocker against this filter: **Is it in code you're already modifying for a blocker fix, AND is it trivial to resolve reliably?** Both conditions must hold. A rename in a function you're rewriting qualifies. A "missing test coverage" issue in an adjacent module does not — even if it's trivial, you're not already in that code. And a design concern in code you're touching doesn't qualify either — it's not trivial.

Non-blockers that pass this filter: fold them into the blocker fix. Note in the plan what you're picking up and why (so the user can see you're not scope-creeping).

Everything else: group into logically coherent tasks and queue them.
- Things that hit the same modules go together
- Not too large (one agent should complete it in a TDD cycle)
- Not too small (don't create a task for a single rename)
- Include specific file locations and what the reviewers flagged

> **Note:** The `add_task` facade is deprecated and will be removed — use `submit_task` + `resolve_ticket`.

```
# Phase 1: submit — returns immediately with a ticket id
submit_result = submit_task(
    title="<descriptive title>",
    description="<specific issues, file locations, reviewer references>",
    project_root="<PROJECT_ROOT>",
    priority="<medium|high>",
    metadata={
        "source": "unblock-triage",
        "spawn_context": "unblock",
        "modules": ["<path/to/affected/module>"],
    },
)
ticket = submit_result["ticket"]

# Phase 2: block until the curator decides (default 115 s)
resolve = resolve_ticket(ticket=ticket, project_root="<PROJECT_ROOT>")

if resolve["status"] == "created":
    task_id = resolve["task_id"]           # new task queued successfully
elif resolve["status"] == "combined":
    task_id = resolve["task_id"]           # folded into existing task — still counts as queued
elif resolve["status"] == "failed":
    # reason determines how to proceed:
    # server_restart → retry the submit_task + resolve_ticket pair ONCE with the same
    #   metadata (original ticket is dead; a fresh submit is the only path forward).
    #   Unblock-triage tasks do not currently set escalation_id/suggestion_hash, so the
    #   R4 idempotency gate does not fire — the curator may de-duplicate via "combined",
    #   or a duplicate may slip through that the user can merge later.
    # timeout → first retry resolve_ticket(ticket=same_ticket, ...) with the same ticket —
    #   the worker may still be processing. Only re-submit_task if that retry returns
    #   unknown_ticket or expired. Same dedup caveat as server_restart if re-submit needed.
    # unknown_ticket | server_closed | expired → terminal: escalate the reason to the
    #   user and skip queuing this non-blocker.
    handle_failure(resolve["reason"])
```

### 4.2: Reflect on analysis

Use the reflect skill (`/reflect`) to capture:
- What issues were found and how they were classified
- Any architectural insights discovered during analysis
- The blocker/non-blocker split and rationale

### 4.3: Plan the fix

Enter plan mode. The plan covers two parts:

**Part 1: Fix all blocking issues** in the worktree, on the task branch.

**Part 2: Merge or resolve.** The procedure depends on the entry path:

*If this is a blocked task (task status is `blocked`, fixing in worktree):*

The merge procedure is iterative — don't assume one pass will be enough:

1. Rebase on main. Resolve any conflicts.
2. Run the project's full verification suite (tests, lint, type-check).
3. Fix any failures.
4. On green: rebase on main again — other tasks may have merged while you were fixing.
5. Repeat steps 2-4 until stable (rebase is clean AND verification passes with no new changes needed).
6. Use `/merge-queue` to merge. It routes through the orchestrator's merge queue when available (preventing races with concurrent tasks) and falls back to direct merge when the orchestrator isn't running.
7. On green: `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"commit": "<sha-of-merge>"})`
   - Pass `{"commit": "<sha>"}` when the merge landed a single commit on main (the normal case — merge_request returns the SHA). Fall back to `{"note": "<one-sentence explanation>"}` for fast-forward or covered-by-sibling cases where no single commit applies.
8. Clean up: `git worktree remove .worktrees/<TASK_ID>` and `git branch -d task/<TASK_ID>`

*If this is an escalated task (pending escalation, agent is paused):*

Choose one of these based on the analysis:

- **Resolve and resume** — if your blocker fixes address the escalation concern, resolve with actionable instructions for the resumed agent:
  ```
  resolve_issue(escalation_id="<id>", resolution="<specific instructions>", terminate=false, resolved_by="interactive", resolution_turns=<N>)
  ```
  The agent resumes with your resolution injected into its briefing. Task stays `in-progress`.

- **Terminate and reschedule** — if the task needs fundamental redesign:
  ```
  resolve_issue(escalation_id="<id>", resolution="<reason for termination>", terminate=true, resolved_by="interactive", resolution_turns=<N>)
  ```
  Then create or update tasks as needed. Task goes to `pending`.

**Turn counting:** `<N>` is the number of user messages since this skill was invoked (count each time the user sent a message, starting from the `/unblock` invocation). This tracks how much human attention the resolution required. If you lose count, estimate conservatively.

- **Fix manually and merge** — if you fix the issue yourself in the worktree, follow the blocked-task merge procedure above.

### 4.4: Execute the plan

Exit plan mode and execute. **Keep the task in its current status during the work** — don't change it until you've successfully merged or resolved. This prevents the orchestrator from trying to start new agents on it.

### 4.5: Final reflect

**The task is not done until this step completes.** After the merge succeeds (or escalation is resolved) and the task status is updated, invoke `/reflect` to capture:
- What was fixed and how
- Any decisions made during the fix (e.g., chose approach A over B because...)
- Architectural insights that surfaced during the work
- Brief summary: what was accomplished, what was deferred (with task numbers)

This is the last step. Do not consider the unblock workflow complete until reflect has run.

---

## End states

After this skill completes, the task should be in one of these states:

| Starting state | Outcome | Final state |
|---------------|---------|-------------|
| Blocked | Successfully merged to main, verification green | Done |
| Blocked | Needs redesign, can't fix this session | Pending (update task description first) |
| In Progress (escalated) | Escalation resolved, agent resumes | In Progress |
| In Progress (escalated) | Escalation terminated, work rescheduled | Pending |
| In Progress (escalated) | Fixed manually, merged to main | Done |

---

## Project-specific verification

The verification commands depend on the project. Check `orchestrator.yaml` in the project root for `test_command`, `lint_command`, and `type_check_command` overrides. If there's no override, use whatever the project's standard tooling is (check for Cargo.toml, package.json, pyproject.toml, etc.).
