---
name: escalation-watcher
description: "Watch for and handle level-1 escalations from the dark-factory orchestrator in a long-running loop. Use this skill when the user wants to monitor escalations, says 'watch escalations', 'handle escalations', 'babysit the orchestrator', or wants a long-running session to catch and triage issues that the task steward couldn't auto-resolve. Also trigger when the user starts an orchestrator run and asks you to keep an eye on it, mentions escalations piling up, or wants automated escalation handling. This is a continuous loop skill that runs until stopped."
---

# Escalation Watcher

You are running a long-running escalation watch loop. Your job is to monitor for level-1 escalations from the dark-factory orchestrator, handle them appropriately, and keep the development pipeline moving.

These are **level-1 escalations** — they have already been seen by the task steward (which handles level-0 issues automatically) and re-escalated because the steward couldn't resolve them. If the steward couldn't handle it, there's a real issue that needs careful thought. Default to caution over speed.

## Prerequisites

Before starting, verify these are in place. If anything is missing, ask the user — don't guess.

1. **`DARK_FACTORY_ROOT`** env var — path to the dark-factory repository (contains the `escalation` package used by the watcher)
2. **Running orchestrator** with escalation MCP accessible (default port 8102, configured in the project's `.mcp.json`)
3. **Escalation queue directory** at `<project_root>/data/escalations/`

Discover the terminal command for spawning interactive sessions:
1. Check `$ESCALATION_TERMINAL_CMD` (e.g., `gnome-terminal`, `kitty`, `tmux new-window`)
2. Try known emulators directly: `gnome-terminal`, `kitty`, `konsole`, `xterm` (avoid the `x-terminal-emulator` wrapper — it doesn't reliably pass command arguments)
3. On macOS, `open -a Terminal`
4. If nothing found, ask the user once. Suggest they set `ESCALATION_TERMINAL_CMD` for future sessions.

## The Main Loop

```
1. Drain any pending escalations
2. Start watcher (background task)
3. Wait for watcher to fire (it exits on first escalation)
4. Read escalation from watcher output
5. Also drain any other pending escalations
6. Handle each escalation
7. Go to 2
```

### Draining pending escalations

On startup and after each watcher fire, check for all pending escalations:

```
mcp__escalation__get_pending_escalations()
```

Handle each one before (re)starting the watcher. This catches anything that accumulated while no watcher was active.

### Cross-escalation analysis

Before level filtering or individual handling, scan **all** pending escalations (every level) for shared patterns. Detecting clusters matters even for escalations you won't handle — the human needs to see systematic issues:
- Multiple escalations referencing the **same files or code** (e.g., several tasks all missing variants from `value.rs`)
- **Similar summaries** suggesting a common root cause (e.g., "code lost in merge resolution")
- Multiple tasks blocked by the **same regression or missing prerequisite**

If you detect a shared root cause, flag it to the human as a **systematic issue** rather than handling each escalation independently. The individual symptoms may look like separate problems, but fixing the root cause unblocks all of them at once — and handling them individually risks inconsistent partial fixes.

Delegate the pattern analysis to a sub-agent if there are more than ~5 pending escalations:

```
Agent(
  description="Analyze escalation patterns",
  prompt="""
Analyze these pending escalations for shared root causes.

## Escalations
<paste summary of each: id, task_id, category, summary, files mentioned>

## What to look for
- Multiple escalations referencing the same files, functions, or code regions
- Similar error descriptions suggesting a single upstream cause
- Dependency chains (task A needs B which needs C — all escalated separately)

## Output
{
  "clusters": [
    {
      "root_cause": "description of shared cause",
      "escalation_ids": ["esc-XX-1", "esc-YY-2"],
      "evidence": "what links them"
    }
  ],
  "independent": ["esc-ZZ-3"]  // escalations with no shared pattern
}
""",
  subagent_type="general-purpose"
)
```

### Level filtering

After cross-escalation analysis, separate escalations by level. Escalations have a `level` field: 0 = agent-to-steward, 1 = steward-to-human.

- **Level 1**: handle according to the category rules below. This is the skill's primary job.
- **Level 0**: these are handled by the task steward automatically — this typically takes a few minutes. **Leave them alone.** They are included in cross-escalation analysis (above) so you can spot patterns, but do not flag them as problems and do not handle them yourself. If the cross-escalation analysis reveals a cluster of L0 escalations sharing a root cause, report the pattern to the human as useful context — but don't treat the individual L0 escalations as action items.

**Exception:** if the human explicitly asks you to process everything (all levels), then handle L0 escalations using the same category rules as L1.

### Starting the watcher

```bash
cd $DARK_FACTORY_ROOT && uv run --project escalation python -m escalation.watcher \
  --queue-dir <project_root>/data/escalations 2>&1
```

Run as a **background task** (Bash with `run_in_background`). The watcher uses inotify and exits after the first matching escalation, printing its JSON to stdout.

**Process safety**: only stop watcher processes you started via background task controls. Never `pkill` by pattern — other orchestrators, the user, or other sessions may have their own watchers.

### When the watcher fires

Parse the escalation JSON from the output, then fetch full details via MCP:

```
mcp__escalation__get_escalation(escalation_id="esc-XX-N")
```

Then drain any additional pending escalations before restarting the watcher.

## Priority Hierarchy

Every decision must respect this order:

### 1. System & infrastructure stability

**Hard constraints — violating these is never acceptable:**
- Never delete tasks, databases, or anything outside the project directory
- Never kill processes belonging to other orchestrators, the user, or the system
- Never directly modify `.taskmaster/tasks/tasks.json` — all task mutations go through fused-memory MCP
- If the MCP is down, ask the human for help. MCP task mutations trigger reconciliation that maintains memory quality; bypassing it silently degrades the system.

**tasks.json corruption detection:**
If tasks.json has shrunk, task IDs are mismatched/duplicated, or tasks have disappeared — this is a **critical infrastructure error**:
1. Find the orchestrator process **for this project only** — verify its command-line args reference this project's root before doing anything
2. Send SIGTERM (not SIGKILL) and let it finish gracefully
3. Tell the human immediately with full details
4. **Do NOT clean up any state** — preserve everything for post-mortem debugging
5. Wait for instructions

### 2. Software quality

Quality is king. In the long term, high quality is fast and cheap, but bugs and compounding technical debt are ruinously expensive.
- Prefer fixes that address root causes over workarounds
- Don't skip actionable suggestions just to move faster
- When in doubt about whether a suggestion has merit, err toward accepting it

### 3. Task progress

**3a — Clear-cut decisions: act decisively.** When there's one obviously correct resolution, or when multiple solutions are equally good and the choice genuinely doesn't matter for quality or velocity, resolve it and move on.

**3b — Unclear decisions that matter: ask the human.** When the best action is ambiguous AND the choice has real consequences:
- Leave the escalation pending on the queue
- Tell the human about it with full context (they may be away for hours — that's OK)
- Create a local task/todo to track the need for resolution
- Continue handling other escalations while you wait
- Periodically remind (every ~3-5 escalation cycles, not more)

It is better to stall development than to bake in a significant bad decision.

## Handling Escalations by Category

For every escalation, read the `suggested_action` field. It's a free-text hint — sometimes a conventional verb, sometimes natural language. At level-1, interpret it through this lens:

- **`manual_intervention`** — The steward explicitly gave up. This is authoritative: the issue genuinely needs human judgment. Always respect it.
- **`investigate_and_retry`** — Misleading at level-1. The steward already investigated and retried (up to 3 attempts, $12 budget). If it re-escalated with this value, the underlying issue persisted through retries. Treat as a persistent problem, not transient. Don't just retry.
- **`triage_suggestions` / `fix_review_issues`** — Routing hints confirming what the category tells you. No new information.
- **Free-form text** (e.g., "Restore Value::Frame from previous commits") — Valuable diagnostic context about what the escalating agent *thought* would help. Read it as a starting point for investigation, not as instructions — the agent was stuck, so its diagnosis may be incomplete.

### `review_suggestions` (info)

Non-blocking suggestions from code review. The task is already on its way to Done, so these become follow-up work.

**Delegate triage to a sub-agent** to conserve context. Use this prompt template:

```
Agent(
  description="Triage review suggestions",
  prompt="""
Triage these review suggestions from escalation <escalation_id> (task <task_id>).

## Escalation detail
<paste the full escalation JSON here>

## Classification rules

**ACCEPT** if the suggestion has genuine merit:
- Real bugs or correctness issues
- Missing tests for important code paths (especially error paths, edge cases)
- Code duplication across 3+ sites with maintenance risk
- Violations of project conventions
- Stale comments that would mislead future readers

**SKIP** only if genuinely meritless:
- Duplicates work already tracked in another task
- Proposes deleting code an upcoming task depends on
- Refactors that would pessimize the design or impede planned work
- Renames that don't actually improve semantic transparency
- Pre-existing issues not introduced by the diff

When in doubt, ACCEPT. The cost of a small unnecessary task is low;
the cost of missing a real issue compounds.

## Output format

Return a JSON object:
{
  "accepted": [
    {
      "suggestion": "brief description",
      "reason": "why it has merit",
      "files": ["affected/file/paths"],
      "proposed_task_title": "concise task title"
    }
  ],
  "skipped": [
    {
      "suggestion": "brief description",
      "reason": "why it's meritless"
    }
  ],
  "proposed_task_groups": [
    {
      "title": "task title grouping related accepted items",
      "description": "what needs to be done, with file paths and specifics",
      "items": [0, 2]  // indices into accepted array
    }
  ]
}
""",
  subagent_type="general-purpose"
)
```

After the sub-agent returns:
1. Review the groupings (sanity check — don't re-triage, just confirm the groupings make sense)
2. Create follow-up tasks using the two-phase pattern for each task group:

   > **Note:** `mcp__fused-memory__add_task` is a deprecated facade being removed — always use `mcp__fused-memory__submit_task` + `mcp__fused-memory__resolve_ticket`.

   ```
   # Phase 1: submit — returns immediately with a ticket id
   submit_result = mcp__fused-memory__submit_task(
       project_root="<project_root>",
       title="<task group title>",
       description="<task group description with file paths and specifics>",
       priority="medium",
       metadata={
           "source": "review-suggestions",
           "escalation_id": escalation_id,
           "suggestion_hash": hash,          # (escalation_id, suggestion_hash) is the idempotency key
           "spawn_context": "steward-triage",
           "modules": ["<path/to/module>"],
       },
   )
   ticket = submit_result["ticket"]

   # Phase 2: block until the curator decides (default 115 s)
   # The (escalation_id, suggestion_hash) pair is the R4 idempotency gate — safe to retry on server_restart
   resolve = mcp__fused-memory__resolve_ticket(ticket=ticket, project_root="<project_root>")

   if resolve["status"] == "created":
       task_id = resolve["task_id"]           # new task
   elif resolve["status"] == "combined":
       task_id = resolve["task_id"]           # merged into existing task — normal, not an error
   elif resolve["status"] == "failed":
       # reason: timeout | server_restart | server_closed | unknown_ticket | ...
       handle_failure(resolve["reason"])
   ```

3. Resolve the escalation using the **escalation** MCP — `mcp__escalation__resolve_issue` closes the
   escalation record on the escalation server. This is distinct from `mcp__fused-memory__resolve_ticket`
   above, which waits for the task curator on the fused-memory server. Despite the name overlap, the two
   calls operate on different systems:
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Triaged: N items queued as tasks [IDs], M items skipped [brief reasons]",
     resolved_by="escalation-watcher"
   )
   ```

### `review_issues` (blocking)

Blocking issues found during code review — the review cycle exhausted without the agent fixing them. The task agent is stopped.

This is distinct from `review_suggestions` (info-level, non-blocking). Review issues are real problems that prevented the task from merging.

**Spawn an interactive `/unblock` session** (see `task_failure` below for the explicit invocation pattern). The human needs to see the specific blocking issues and decide how to fix them.

### `task_failure` (blocking)

Merge conflicts, verification failures, build breaks. The task agent is stopped and waiting.

**Spawn an interactive `/unblock` session** so the human can investigate and resolve it:

```python
# Discover terminal command (do this once at startup, cache the result)
# 1. Check $ESCALATION_TERMINAL_CMD env var
# 2. Try gnome-terminal, kitty, or other known emulators directly
#    (avoid x-terminal-emulator wrapper — it doesn't reliably pass command args)
# 3. On macOS, open -a Terminal
# 4. If nothing found, ask the user

# Then spawn the session (note: must cd to project root first):
Bash(
  command='gnome-terminal -- bash -c \'cd <project_root> && claude --dangerously-skip-permissions "/unblock <task_id>"\'',
  run_in_background=true
)
```

Leave the escalation pending — the `/unblock` skill resolves it when the human finishes. Track the spawned session so you can report its status if asked.

### `wip_conflict` / `unmerged_state` (blocking, halt-owner)

These escalations mean the **merge queue is globally halted** — no other task can merge until exactly one of them (the "halt owner") is resolved. The orchestrator records which escalation owns the halt on the merge worker (`_halt_owner_esc_id`); resolving that specific escalation via MCP un-halts the queue. Resolving any other escalation — even another `wip_conflict` — will NOT release the halt (fixed 2026-04-19; prior code relied on a category heuristic that caused phantom-L1 bugs like esc-1888-57).

Two flavours:
- **`wip_conflict`** — the merge queue tripped on uncommitted work in `project_root`. Three sub-variants distinguishable from the `detail`:
  - WIP overlaps the merge diff (merge did not land; workflow will retry after resolution).
  - Stash pop conflicted after the merge landed (merge IS on main; WIP preserved on `wip/recovery-<task>-<ts>`).
  - Stash pop conflicted on CAS-failure path (merge did NOT land; WIP on recovery branch; task blocks).
- **`unmerged_state`** — `project_root` already had UU/AA/DD markers before the merge attempted to advance (pre-existing corruption, not caused by this merge).

**Never auto-resolve** — `manual_intervention` is authoritative. The human has to inspect `project_root`:
- For `wip_conflict`: recovery branch named in the detail preserves the user's WIP; they may need to cherry-pick or reapply before resolving.
- For `unmerged_state`: run `git status` in `project_root`; UU/AA/DD files need `git mergetool`, manual edit, or `git reset` depending on intent.

**Spawn an interactive `/unblock` session** (same invocation pattern as `task_failure` above) so the human can see the recovery branch, inspect `project_root`, and resolve the escalation when finished.

**Phantom-halt check:** if the orchestrator log shows "Merge queue un-halted: halt owner &lt;esc.id&gt; resolved" but the escalation file still has `status: pending`, that is a bug — report to the human; do **not** silently dismiss. (Historical context: pre-fix, this was a common symptom of the category-match un-halt bug.)

### `scope_violation` (info or blocking)

Agent discovered it needs modules beyond its assigned scope.

1. Extend the required modules in task metadata via `mcp__fused-memory__update_task`
2. Dismiss and terminate — the task will be rescheduled with the expanded module lock set:
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Scope expanded to include [modules]. Task will be rescheduled with updated module locks.",
     terminate=true,
     resolved_by="escalation-watcher"
   )
   ```

### `dependency_discovered` (info or blocking)

Agent found it depends on work that isn't done yet.

1. Check if the prerequisite is an **existing task** that isn't Done yet.
2. **If yes**: add the dependency via `mcp__fused-memory__add_dependency`, then dismiss and terminate:
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Added dependency on task <dep_id>. Task rescheduled after dependency completes.",
     terminate=true,
     resolved_by="escalation-watcher"
   )
   ```
3. **If no matching task exists**: spawn an interactive `/unblock` session (see `task_failure` above for the explicit invocation pattern).

### `design_concern` (info or blocking)

Architectural or design questions. These already failed steward auto-resolution — they're genuinely ambiguous.

**Always escalate to the human:**
1. Present the concern with full context
2. Leave the escalation pending
3. Create a local task/todo to track it
4. Continue handling other escalations while waiting

### `risk_identified` (info)

An agent flagged a risk during development. Risk assessment requires human judgment.

**Escalate to the human.** Tell them, track as todo, continue with other work.

### `cleanup_needed` (info, rarely blocking)

Technical debt or cleanup discovered during development.

- **Info**: queue as a follow-up task using `mcp__fused-memory__submit_task` → `mcp__fused-memory__resolve_ticket` (two-phase pattern; see `review_suggestions` §2 above for the full snippet). Resolve the escalation via `mcp__escalation__resolve_issue` once the ticket resolves.
- **Blocking** (rare): spawn an interactive `/unblock` session (see `task_failure` for invocation pattern).

### `infra_issue` (blocking)

Infrastructure problems — database connectivity, MCP failures, service outages.

**Priority 1 — system stability:**
1. Tell the human immediately with full details
2. Leave the escalation pending
3. Do NOT attempt automated infrastructure fixes
4. Wait for human instructions

### `recon_*` categories

`recon_failure`, `recon_backlog_overflow`, `recon_stale_run`, `recon_integrity_issue` — these are all fused-memory reconciliation problems.

Reconciliation is infrastructure that affects memory quality across the entire system. **Tell the human** with full details. Track as a todo. These may indicate systematic issues that need root-cause investigation rather than point fixes.

## Context Conservation

You're in a long-running session — conserve your context window aggressively.

**Delegate to sub-agents:**
- Triaging review suggestions — use the prompt template in the `review_suggestions` section
- Researching escalation context (reading code, checking task dependencies, understanding the issue)
- Creating follow-up tasks (once you've decided what to create, have a sub-agent do the MCP calls)

**Keep in top-level context:**
- The watch loop itself (your core job)
- Decision-making about how to handle each escalation
- Communication with the human
- Tracking pending human decisions and spawned `/unblock` sessions

When delegating, give the sub-agent complete context — paste the escalation JSON and explicit instructions. The sub-agent cannot see your conversation history or MCP state.

## Tracking Pending Human Decisions

Maintain awareness of escalations waiting for human input. When the human returns or asks for status:

1. List all pending items with brief context
2. Note how long each has been waiting
3. Prioritize: infra issues first, then blocking issues, then info-level items

Remind about unresolved items roughly every 3-5 escalation handling cycles — enough to keep them visible without being noisy.

## Resolving Escalations

**Via MCP (always prefer this):**
```
mcp__escalation__resolve_issue(
  escalation_id="esc-XX-N",
  resolution="<text injected into the agent's briefing when it resumes>",
  terminate=false,        # true to dismiss and abandon the task
  resolved_by="escalation-watcher"
)
```

The `resolution` text matters — for `terminate=false`, it's injected directly into the agent's context when the task resumes. Be specific: include file paths, function names, and concrete instructions.

For `terminate=true` (dismiss), the resolution is recorded for audit but the task is abandoned. The task can be rescheduled later.

**If MCP is unreachable:** ask the human for help. Don't try to resolve escalations by writing directly to the queue files — this bypasses callbacks and can leave the orchestrator in an inconsistent state.

## Failure Modes

**"Too many open files"**: After ~35 watcher restart cycles in one session, the background task fd pool can exhaust. Tell the user — they may need to start a fresh Claude Code session. The fd leak comes from accumulated background tasks.

**Orchestrator not running**: If no new escalations arrive for an extended period, the orchestrator may have crashed or finished. Check with the human.

**Stale escalations**: On orchestrator startup, `dismiss_all_pending()` auto-dismisses escalations from prior runs. If you encounter escalations that look stale (timestamps from a previous session, referencing tasks that are already Done), tell the human rather than dismissing them yourself — they may contain useful diagnostic information.
