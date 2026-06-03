---
name: escalation-watcher
description: "Watch for and handle level-2 escalations from the dark-factory orchestrator in a long-running loop. Under the 3-tier escalation ladder (L0→per-task steward, L1→escalation-watcher-auto, L2→human), this skill is the L2 consumer. Use this skill when the user wants to monitor escalations, says 'watch escalations', 'handle escalations', 'babysit the orchestrator', or wants a long-running session to catch and triage issues that the auto-watcher couldn't resolve. Also trigger when the user starts an orchestrator run and asks you to keep an eye on it, mentions escalations piling up, or wants automated escalation handling. This is a continuous loop skill that runs until stopped."
---

# Escalation Watcher

You are running a long-running escalation watch loop. Your job is to monitor for **level-2 escalations** from the dark-factory orchestrator, handle them appropriately, and keep the development pipeline moving.

The 3-tier escalation ladder determines which agent handles each level:
- **L0** → per-task steward (handles routine agent problems automatically)
- **L1** → escalation-watcher-auto (handles steward-escalated issues; performs root-cause clustering, triage, and automated resolution where possible)
- **L2** → this skill / human (handles issues the auto-watcher judged as needing human judgement)

L2 items reach this queue via two paths: (a) **born-at-L2** — severity `critical` or `urgent` at the escalation creation chokepoint, bypassing L0/L1 entirely; (b) **promoted from L1** — the auto-watcher attempted resolution and determined human input is required, typically packaging the escalation as a causal cluster with hypothesis, evidence, and proposed options pre-formed. Default to caution over speed.

## Prerequisites

Before starting, verify these are in place. If anything is missing, ask the user — don't guess.

1. **`DARK_FACTORY_ROOT`** env var — path to the dark-factory repository (contains the `escalation` package used by the watcher)
2. **Running orchestrator** with escalation MCP accessible (port `8102` for dark-factory — set in `orchestrator/config.yaml` and matching `.mcp.json`; the code default is `8100`, which other projects may use)
3. **Escalation queue directory** at `<project_root>/data/escalations/`

Terminal discovery for spawned `/unblock` sessions is handled lazily by the `/spawn` skill — no setup is required here.

## The Main Loop

```
1. Start the watcher (background task, filtered to L2); confirm its process is alive
2. Drain pending L2 escalations — only NOW, with the watcher confirmed up (drain-after-up)
3. Handle each drained escalation
4. Wait for a wake signal: the watcher firing (it exits on the first new L2 escalation), or — if any
   background merge-submission / auto-unblock sub-agent is in flight — that sub-agent completing.
   Handle whichever arrives.
5. Read the escalation from the watcher output — this is the wake signal; the drain in
   step 2 of the next pass is the authoritative source of what to handle
6. Go to 1 (restart watcher → confirm up → drain → handle)
```

The fired escalation (step 5) is just the wake; you do not handle it inline. Looping back
re-arms the watcher first, then the drain re-finds it (still pending) plus anything new — so
handling always happens with a live watcher in place and nothing slips through the gap.

### Draining pending escalations

Check for all pending L2 escalations — **compact** to keep context small:

```
mcp__escalation__get_pending_escalations(level=2, compact=True)
```

`compact=True` returns only the triage fields (`id`, `task_id`, `category`, `severity`, `level`,
`status`, `summary`, `suggested_action`, `timestamp`) and drops the heavy free-text/cluster fields
(`detail`, `members`, `options`, `root_cause`, `train_state`, …). Triage from that; fetch the full
record with `get_escalation(id)` **only** for the one item you're about to act on — and prefer doing
that full read inside the handling sub-agent (see Context Conservation). During an AFK window the
pending pile grows, and a full-dict drain every cycle is the dominant context sink — `compact=True`
is what keeps a long-running session alive.

**Drain-after-up — ordering matters.** Always (re)start the watcher and confirm its process is
alive *before* you drain, never the other way round. A pre-start drain races inotify
registration: an L2 file created in the gap between your drain and the watcher's `add_watch` is
seen by neither, and sits unhandled until some *unrelated* later escalation happens to fire the
watcher and trigger the next drain (real incident: esc-1573-8 sat 21h). Starting the watcher
first closes the gap — anything born during startup is caught by the drain that immediately
follows. This drain catches any L2 escalations that accumulated while no watcher was active.

### L2-only contract

This skill drains and waits only on **level-2 escalations**. Both the watcher subprocess and the `get_pending_escalations` draining call are filtered to `level == 2` (see details in the relevant sections below).

- **L0** is owned by per-task stewards — do not drain or handle L0 escalations here.
- **L1** is owned by escalation-watcher-auto — do not drain or handle L1 escalations here.

Never process L0 or L1 from this skill, even if explicitly asked — doing so would race with the per-task steward and escalation-watcher-auto, which own those queues and rely on their own resolution callbacks. If the user wants to handle lower-level escalations, they should invoke the appropriate skill for that level.

### Starting the watcher

```bash
cd $DARK_FACTORY_ROOT && uv run --project escalation python -m escalation.watcher \
  --queue-dir <project_root>/data/escalations --level 2 2>&1
```

Run as a **background task** (Bash with `run_in_background`). The `--level 2` flag restricts the inotify watcher to L2 escalation files only. The watcher uses inotify and exits after the first matching L2 escalation, printing its JSON to stdout.

**Process safety**: only stop watcher processes you started via background task controls. Never `pkill` by pattern — other orchestrators, the user, or other sessions may have their own watchers.

### When the watcher fires

The watcher's printed JSON is just your **wake signal** — note the `id`, but you don't need to keep
the whole blob in context. Loop back, re-arm the watcher, and let the next compact drain be the
authoritative list of what to handle. Fetch the full record via
`mcp__escalation__get_escalation(escalation_id="esc-XX-N")` only for the specific item you're about
to act on — ideally inside the handling sub-agent rather than at top level.

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

## Merge Submissions — NEVER in the Foreground

`mcp__escalation__merge_request` blocks until the merge worker finishes rebasing, running the full
verify suite, and CAS-advancing main. On a large/slow repo (e.g. reify) a single call can take
**30+ minutes** — made in the foreground it freezes the entire watch loop for that long: no
draining, no watcher re-arm, a born-at-L2 `critical` sits unseen (real incident: esc-2831-78 wedged
a reify watcher >30 min on a direct foreground retry-land).

**Hard rule: this session never calls `merge_request` at top level — no exceptions.** That covers
the documented path (the B3 low-risk auto-unblock merges inside its sub-agent) *and* any improvised
submission you're tempted to make yourself — e.g. retrying the land of a done-but-unmerged task once
the verify gates that blocked it have cleared. However legitimate the merge, the submission goes
through a NON-INTERACTIVE **background** sub-agent (`Agent` tool, general-purpose,
`run_in_background: true`):

- Give the sub-agent everything it needs up front: `task_id`, `branch` (bare name — the worker
  prepends `task/`), the absolute `worktree` path, a `description`, and what to do on each outcome
  (per `skills/merge-queue/SKILL.md`). It makes the blocking call in its own context and returns a
  compact JSON result.
- Track the launch exactly like a B3 launch: record `{task_id, escalation_id (if any),
  background-task-id}`, and never submit a second merge for a task that already has one in flight.
  (`merge_request` returns `status='in_flight'` for a duplicate branch as a backstop — if you see
  it, the merge is already covered: do NOT re-queue.)
- The sub-agent's completion is a wake signal (Main Loop step 4); the foreground stays free to
  re-arm the watcher and keep draining while the merge grinds.

## AFK Mode (extended unattended operation)

When the human will be away for an extended period (hours to days) and cannot adjudicate 3b
decisions, switch posture from "stall and ask" to "keep the pipeline moving, defer the judgement,
and leave a clean trail." Confirm AFK mode with the human if you can; otherwise infer it from an
explicit "I'll be away" or a long silence after one. Three behavioural shifts:

1. **Defer, don't wedge.** For a 3b item (ambiguous AND consequential), stalling the whole queue for
   days helps no one. Where the decision can be safely *postponed* without baking anything in:
   - Queue a follow-up task capturing the decision to be made (two-phase `submit_task` →
     `resolve_ticket`), and
   - `resolve_issue(..., terminate=true)` to reschedule/abandon the blocking task so **independent**
     work keeps flowing.
   This is parking a decision for later review — NOT making it. Only do it when terminating the task
   cannot itself cause harm (no half-merged state, no destructive side effect). The Priority
   Hierarchy bar still holds: better to defer than to bake in a bad decision — when in real doubt,
   fall back to "leave pending + digest."

2. **Don't spawn unattended terminals.** The interactive `/spawn` → `/unblock` path needs a human at
   a terminal; while AFK those sit idle and the task stays blocked anyway. So in AFK mode:
   - **`task_failure` / `review_issues`:** first try the **low-risk auto-unblock gate** (below). If
     it doesn't qualify or aborts, leave the escalation pending and add it to the digest — do NOT
     spawn an interactive `/unblock`.
   - **`wip_conflict` / `unmerged_state` / `dependency_discovered`-with-no-task / `design_concern` /
     `risk_identified` / `infra_issue` / `recon_*`:** leave pending + digest. These need a human;
     a terminal nobody attends just clutters.

3. **Batch into a digest, don't ping per-item.** Reminding "every 3-5 cycles" is noise when nobody is
   reading. Maintain a single rolling manifest at `<project_root>/data/escalations/afk-digest.md`
   (overwrite each cycle) listing every pending item: id, task_id, category, severity, age, and a
   one-line "why it's waiting / what decision is needed." On return the human reads one file. If
   phone push is configured (`--ntfy-url` on the watcher command), a born-at-L2 `critical`/`urgent`
   still pushes immediately — those are the only items worth interrupting an AFK human for.

### Low-risk auto-unblock gate (B3)

For `task_failure` and `review_issues` in AFK mode, before leaving the item for the human, check
whether the orchestrator's at-block-time dry-run investigation already found a **low-risk** fix:

1. `get_task(task_id)` → `latest = metadata.dry_run_proposals[-1]` (if any).
2. Gate — ALL must hold, else fall through to "leave pending + digest":
   - `latest['risk_label'] == 'low'` and `latest` has no `status` key (not a failed / budget-exhausted entry);
   - `latest` is fresh (the most-recent entry; the branch is not materially changed since `latest['timestamp']`);
   - you have launched **fewer than 3 low-risk auto-unblocks this session** (in-flight *or*
     completed) — a self-imposed cap. Because the sub-agent now runs in the background (step 3),
     count it at **launch**, not on completion — otherwise several concurrent background sub-agents
     could each pass the gate before any returns and blow past the cap. Over the cap, leave pending +
     digest so a runaway can't merge unattended.
3. If the gate passes, launch the **`unblock-low-risk`** skill as a NON-INTERACTIVE **background**
   sub-agent (the `Agent` tool, general-purpose, **`run_in_background: true`** — NOT `/spawn`),
   passing `task_id`, `escalation_id`, `project_root`, the `worktree` path, and the `latest`
   proposal, and telling it to read and follow `skills/unblock-low-risk/SKILL.md`. It applies the fix
   scoped to `files_referenced`, runs the verify suite, and merges via the queue — or aborts cleanly.

   **Background, not foreground — why.** The sub-agent's merge step blocks on `merge_request` until
   the merge worker finishes rebasing, verifying, and CAS-advancing main. On a large/slow repo (e.g.
   reify) that single call can take ~30 minutes. Run in the *foreground* (`Agent` without
   `run_in_background`), that whole window freezes the watch loop — new L2 escalations accumulate
   undrained until the merge returns, and a born-at-L2 `critical` could sit unseen for half an hour.
   Backgrounding keeps the foreground responsive: record the launch (below), then immediately loop
   back to re-arm the watcher and drain. The harness re-invokes you with the sub-agent's result when
   it completes — that completion is itself a wake signal (Main Loop step 4), handled in step 4 below.

   **Record the launch; don't double-launch.** Stash `{task_id, escalation_id, background-task-id}`
   in your context and count it toward the cap (step 2). The escalation stays `pending` until the
   background sub-agent resolves it, so the *next* drain WILL re-find it — before launching for any
   task, skip it if that `task_id` already has an in-flight (or this-session-completed)
   unblock-low-risk sub-agent, or you'll start a second one racing the first.
4. On the sub-agent's **completion** (you're notified asynchronously — you did not block waiting;
   match the result to a recorded launch by `task_id` / background-task-id):
   - `outcome == "merged"`: it has already set the task done and resolved the escalation. It was
     counted toward the cap at launch — don't double-count. Add a one-line success entry to the digest.
   - `outcome == "aborted"`: it changed nothing terminal and left the escalation pending. Keep the
     `task_id` in your attempted set (do NOT re-launch it — its cap slot is spent), record the abort
     reason in the digest, and move on — do NOT retry, and do NOT spawn an interactive `/unblock` in
     AFK mode; it waits for the human.

The sub-agent re-checks the gate defensively and refuses anything not unambiguously low-risk; treat
its abort as authoritative. This gate is AFK-only — when a human is present, prefer interactive
`/unblock` so they stay in the loop.

## Handling Escalations by Category

For every escalation, read the `suggested_action` field. It's a free-text hint — sometimes a conventional verb, sometimes natural language. First determine the escalation's **L2 origin**, then interpret the hint accordingly:

**Born-at-L2** (severity `critical` or `urgent` at creation — bypassed L0 and L1 entirely):
Neither the per-task steward nor the auto-watcher has seen this record. Read `suggested_action` as the originating agent's own annotation — a starting point, not evidence of prior triage. `investigate_and_retry` here means what it says: a retry may well succeed since no automated attempt has been made yet.

**Promoted-from-L1** (the auto-watcher attempted resolution and escalated to human):
- **`manual_intervention`** — The auto-watcher explicitly gave up. This is authoritative: the issue genuinely needs human judgment. Always respect it.
- **`investigate_and_retry`** — Misleading for promoted items. The item has already passed through *both* the per-task steward (L0) *and* the auto-watcher (L1) and persisted through their combined triage and retry budgets. Treat as a deeply persistent problem, not transient. Don't just retry.
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

   # Phase 2: block until the curator decides
   resolve = mcp__fused-memory__resolve_ticket(ticket=ticket, project_root="<project_root>", timeout_seconds=<see _shared/ticket-failure-handling.md>)

   if resolve["status"] == "created":
       task_id = resolve["task_id"]           # new task
   elif resolve["status"] == "combined":
       task_id = resolve["task_id"]           # merged into existing task — normal, not an error
   elif resolve["status"] == "failed":
       # On `failed`: record the reason in the escalation resolution note and skip this
       # suggestion group. This caller DOES set (escalation_id, suggestion_hash), so the
       # R4 gate fires natively.
       # See skills/_shared/ticket-failure-handling.md for the retryable/terminal reason matrix.
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

**Spawn an interactive `/unblock` session** via the `/spawn` skill: invoke `/spawn` with `prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — `/unblock` resolves it when the human finishes. The human needs to see the specific blocking issues and decide how to fix them.

**In AFK mode:** try the low-risk auto-unblock gate first (see [AFK Mode](#afk-mode-extended-unattended-operation)). Spawn the interactive session only when a human is present; otherwise, if the gate doesn't qualify or the sub-agent aborts, leave the escalation pending and add it to the digest.

### `task_failure` (blocking)

Merge conflicts, verification failures, build breaks. The task agent is stopped and waiting.

**Spawn an interactive `/unblock` session** so the human can investigate and resolve it: invoke `/spawn` with `prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`. Leave the escalation pending — the `/unblock` skill resolves it when the human finishes. Track the spawned session so you can report its status if asked.

**In AFK mode:** try the low-risk auto-unblock gate first (see [AFK Mode](#afk-mode-extended-unattended-operation)). Spawn the interactive session only when a human is present; otherwise, if the gate doesn't qualify or the sub-agent aborts, leave the escalation pending and add it to the digest.

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

**Spawn an interactive `/unblock` session** via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`) so the human can see the recovery branch, inspect `project_root`, and resolve the escalation when finished.

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
3. **If no matching task exists**: spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`).

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

- **Info**: queue as a follow-up task using `mcp__fused-memory__submit_task` → `mcp__fused-memory__resolve_ticket` (two-phase pattern; see `review_suggestions` §2 above for the full snippet). When adapting the snippet:
  1. Substitute `"source": "escalation-info"` (only this field changes).
  2. Keep `"spawn_context": "steward-triage"` — unchanged from §2; both sites feed the same steward pipeline.
  3. For the `suggestion_hash` / `escalation_id` synthesis recipe and R4 gate details, see
     [`skills/_shared/ticket-failure-handling.md`](../_shared/ticket-failure-handling.md).
     At this callsite (Case A — the escalation's id is already in scope), the concrete
     synthesis is:
     ```python
     suggestion_hash = hashlib.sha256((escalation['detail'] or escalation['summary'] or escalation['id']).encode()).hexdigest()[:16]
     ```

  Resolve via `mcp__escalation__resolve_issue` once the ticket resolves.
- **Blocking** (rare): spawn an interactive `/unblock` session via `/spawn` (`prompt="/unblock <task_id>"`, `cwd=<project_root>`, `skip_permissions=true`).

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

You're in a long-running session — conserve your context window aggressively. Over a multi-day AFK
window this is the difference between one durable session and repeated restarts.

**Read compact, expand lazily:**
- Drain with `get_pending_escalations(level=2, compact=True)` — never pull full dicts just to triage.
- Don't keep the watcher's wake-signal JSON in context; triage from the compact drain.
- Pull the full record (`get_escalation(id)`) for only the one item you're about to act on, and
  prefer doing that read inside the handling sub-agent so the heavy `detail`/`evidence` never lands
  at top level.

**Delegate to sub-agents:**
- Triaging review suggestions — use the prompt template in the `review_suggestions` section
- Researching escalation context for ANY category that needs code reading (e.g. `task_failure`,
  `design_concern`): have the sub-agent fetch the full escalation, read the code/reviews, and return
  only a compact verdict + recommended action — not the raw material
- The low-risk auto-unblock sub-agent (`unblock-low-risk`) — run it in the **background**
  (`run_in_background: true`) so a slow merge (~30 min on big repos like reify) can't freeze the
  watch loop; it does the whole apply→verify→merge in its own context and returns only a small JSON
  result when it completes
- ANY other merge submission (e.g. retrying the land of a done-but-unmerged task) — `merge_request`
  only ever runs inside a background sub-agent; see "Merge Submissions — NEVER in the Foreground"
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

**Where the `resolution` text actually goes.** It reaches the working agent **only** in the L0
steward-resolved path, where a workflow is still live and waiting (`_wait_for_resolution` →
`build_resume_prompt`). That is *not* the usual L2 case. For the escalations this skill resolves:

- **L2 cluster (has member L1s), `terminate=false`:** the resolution cascades to each member L1,
  flipping the member task `blocked→pending`. It re-dispatches into a **fresh** workflow that does
  **not** read your resolution text — the harness propagates status only. Don't rely on the string
  reaching the agent. If the agent needs specific guidance, either spawn an interactive `/unblock`
  (drive the worktree directly) or write durable guidance into fused-memory / task metadata, which
  the fresh workflow's briefing memory-search may surface.
- **Born-at-L2 with no members (a direct `critical`/`urgent` blocker), `terminate=false`:** this
  marks the record resolved but does **NOT** re-pend the task — the re-pend paths fire only for
  `level==1`, and a directly-filed born-at-L2 has no members to cascade to. The task stays
  `blocked`. To get it moving again use `terminate=true` (abandon → reschedulable) or drive it via
  `/unblock`. The resolution text is recorded for audit only.

Either way, still write a clear, specific `resolution` (file paths, function names, the decision and
why): it is the audit record and the human-readable trail even when no agent re-reads it.

**L2 cluster cascade (live).** When a resolved L2 represents a causal cluster (member L1
escalations packaged by the auto-watcher), resolving the L2 here cascades to close its L1 members
via the escalation server — this skill resolves only the L2 itself, never each member directly. The
cascade is implemented in `queue.resolve()`: it recurses over `esc.members`, resolving each with
`resolved_by='l2-cascade:<L2-id>'`, and the auto-watcher files clusters via `promote_to_l2`. For
design details, see `plans/escalation-l2-tiering.md`.

You may still occasionally see multiple *unclustered* L2s that share a root cause — the auto-watcher
deduplicates by exact root-cause string, so near-miss hypotheses file separately. When you do, scan
them for shared files, summaries, or task IDs and handle related ones together, noting the
relationship in your resolution text.

**If MCP is unreachable:** ask the human for help. Don't try to resolve escalations by writing directly to the queue files — this bypasses callbacks and can leave the orchestrator in an inconsistent state.

## Failure Modes

**"Too many open files" (historical — no longer expected)**: Early sessions could exhaust the background-task fd pool after ~35 watcher restart cycles. This is no longer observed in practice — 100+ cycle sessions are routine. The watcher exits promptly via `sys.exit(0)`, so its inotify fd is reclaimed by the kernel and the background task is reaped shortly after. If you ever do hit it, start a fresh Claude Code session.

**Orchestrator not running**: If no new escalations arrive for an extended period, the orchestrator may have crashed or finished. Check with the human.

**Stale escalations**: On orchestrator startup, `dismiss_all_pending()` auto-dismisses **L0** escalations from prior runs (filter: `level == 0`) — **L1 and L2 escalations are preserved across restarts**. So an L2 with a timestamp from a previous session that is still `status: pending` is legitimate carry-over, not stale; handle it normally. If an escalation genuinely looks wrong (e.g. references a task that is already Done), tell the human rather than dismissing it yourself — it may contain useful diagnostic information.
