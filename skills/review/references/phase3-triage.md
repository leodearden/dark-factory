# Phase 3: Triage and Task Creation — Detailed Guide

This phase answers: **what should we do about what we found?**

Triage requires judgment — distinguishing real bugs from noise, deciding what's a task vs an escalation, avoiding duplicate work. You (Opus) make the classification decisions. Sonnet agents can handle the mechanical task creation once you've decided what to create.

## Step 1: Load findings

If running all phases in sequence, you already have Phase 1 and Phase 2 results in memory.

If running `--phase triage` standalone, load the most recent reports:
- Find the latest `review/reports/phase1-*.json` and `review/reports/phase2-*.json`
- If multiple exist, use the most recent by timestamp
- If none exist, tell the user: "No Phase 1 or Phase 2 reports found. Run `/review --phase integration` and `/review --phase architecture` first, or run `/review` for a full review."

## Step 2: Classify each finding

Walk through every finding from both phases. For each one, assign a classification:

| Classification | Criteria | Action |
|---------------|----------|--------|
| **Auto-fix** | Trivially fixable with no design decisions — missing import, lint fix, type annotation, typo | Create task, priority: medium |
| **Clear-cut issue** | Unambiguous bug or gap with an obvious fix path — broken wiring, unintended stub, failing test | Create task, priority: high |
| **Design question** | Multiple valid approaches, architectural implications, or unclear intent — "should this be sync or async?", "is this the right abstraction boundary?" | Escalate to user |
| **Known/accepted** | Matches a briefing `known_gap`, a memory decision record, or a pre-existing issue baseline | Skip — log in report for audit trail |
| **Stale task** | An existing pending task addresses a concern that's no longer relevant | Flag for user review |

### Classification guidelines

**Lean toward "clear-cut" over "design question"** when the fix is obvious even if the root cause is complex. If the classifier returns a category but never calls the store, the fix is "add the store call" — that's clear-cut, not a design question, even though the bug was in complex code.

**Lean toward "auto-fix" over "clear-cut"** for anything a Sonnet agent could fix in one TDD cycle without architectural guidance. Lint fixes, missing imports, type annotations, docstring fixes.

**Escalate when genuinely ambiguous.** "Should smoke tests start dependent services?" is a design question — there are valid arguments both ways and the answer depends on the user's testing philosophy.

**Don't create tasks for pre-existing issues** unless they affect the current review scope. The review's job is to catch new/undetected problems, not to clean up historical tech debt (unless the user specifically asks for that).

## Step 3: Task tree health check

Before creating new tasks, audit the existing task tree to avoid conflicts:

### Check for duplicates

For each finding you're about to create a task for:

```
get_tasks(project_root="/home/leo/src/dark-factory")
```

Search the task tree for existing tasks that address the same issue. Check:
- Title similarity
- Same modules referenced
- Same area of concern

If a duplicate exists and is pending/in-progress → skip task creation, note in report.
If a duplicate exists and is done but the issue persists → the task didn't actually fix it. Create a new task referencing the prior attempt.

### Flag stale tasks

Look for pending tasks whose assumptions have been invalidated:
- Task references a module that's been renamed or deleted
- Task depends on a task that was cancelled
- Task describes fixing something that's already been fixed by another task
- Task's `description` mentions code patterns that no longer exist

Flag these for user review — don't delete them yourself.

### Flag blocked tasks

Look for tasks marked as blocked where the blocker may have been resolved:
- Blocking dependency is now done
- The issue described in the block reason has been fixed

Suggest unblocking these.

## Step 4: Create tasks

For each finding classified as "auto-fix" or "clear-cut issue":

> **Note:** `add_task` is a deprecated facade being removed; always use `submit_task` + `resolve_ticket`.

```
# Phase 1: submit — returns immediately with a ticket id
submit_result = submit_task(
    project_root="/home/leo/src/dark-factory",
    title="{concise description of the fix}",
    description="{what's wrong, where, evidence, and what the fix should look like}",
    priority="high",                      # or "medium" for auto-fix — see priority mapping below
    metadata={
        "source": "review-cycle",
        "review_id": "{timestamp}",
        "spawn_context": "review",
        "modules": ["{affected/module/path}"],
        "memory_hints": {
            "search_queries": ["{relevant search query}"],
            "entity_names": ["{relevant entity}"]
        },
    },
)
ticket = submit_result["ticket"]

# Phase 2: block until the curator decides (60 s is intentionally conservative; server default is 115 s)
resolve = resolve_ticket(ticket=ticket, project_root="/home/leo/src/dark-factory", timeout_seconds=60)

if resolve["status"] == "created":
    task_id = resolve["task_id"]           # new task — use for add_dependency calls
elif resolve["status"] == "combined":
    task_id = resolve["task_id"]           # merged into existing task — normal, not an error
elif resolve["status"] == "failed":
    # On `failed`: record the reason in the review report and skip this finding.
    # See skills/_shared/ticket-failure-handling.md for the retryable/terminal
    # reason matrix and R4 idempotency guidance. Review-cycle tasks don't natively
    # set escalation_id/suggestion_hash; to opt into R4 de-duplication on retry,
    # synthesize a stable pair per that doc's guidance.
    log_failure(resolve["reason"])
```

### Task quality checklist

Every created task should:

- **Have a concrete title** — "Fix Mem0 client wiring in classifier" not "Fix classifier issue"
- **Include evidence** — the specific file, line, and what's wrong
- **Describe the expected fix** — what the code should do after the fix
- **Reference the review** — include `metadata.source: "review-cycle"` and `metadata.review_id: "{timestamp}"`
- **Include memory_hints** — search queries and entity names that would help a future agent understand the context:
  ```json
  {
    "memory_hints": {
      "search_queries": ["classifier routing decision", "Mem0 client integration"],
      "entity_names": ["classifier.py", "mem0_client"]
    }
  }
  ```
- **Set dependencies** — if fix-up tasks depend on each other (e.g., "fix wiring" before "add integration test for the wiring"), set `add_dependency`

### Priority mapping

| Classification | Priority |
|---------------|----------|
| Clear-cut issue, severity: high | high |
| Clear-cut issue, severity: warning | medium |
| Auto-fix | medium |
| Auto-fix, lint-only | low |

## Step 5: Escalate ambiguous findings

For each finding classified as "design question", present it to the user with:

1. **The finding** — what was observed, with evidence
2. **Why it's ambiguous** — what makes this a judgment call rather than a clear fix
3. **Options** — 2-3 possible approaches with trade-offs
4. **Recommendation** — your best guess, clearly labelled as such

Format:

```markdown
**Escalation 1: Should smoke tests start dependent services?**

Finding: The "Health endpoint" smoke test fails because it assumes the fused-memory server is running. Currently smoke tests don't manage service lifecycle.

Options:
  a) Add setup/teardown to smoke tests that start/stop services → more realistic but slower, risk of port conflicts
  b) Skip smoke tests that require running services, rely on integration tests instead → simpler but loses the "does it actually start?" check
  c) Add a pre-review step that starts all services → separates concerns but adds a manual step

Recommendation: (a) — the whole point of smoke tests is "does it run?", which requires starting it.
```

Wait for the user to respond to escalations before creating tasks for them. If the user provides direction, create the appropriate task. If they defer, note it in the report.

## Step 6: Write review summary to memory

After triage is complete, write a summary to fused-memory:

```
add_memory(
  content="Review cycle completed for {scope} on {date}. Found {n} issues: {n_high} high, {n_medium} medium, {n_low} low. Created {n} tasks. Key findings: {1-2 sentence summary of most important issues}. Escalated {n} design questions to user.",
  category="observations_and_summaries",
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```

If the review found patterns (e.g., "multiple stubs in the same module suggest the task decomposition was too coarse"), write that as a separate observation:

```
add_memory(
  content="{pattern observation and its implications for future work}",
  category="observations_and_summaries",
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```

## Compile final output

After all three phases, produce the combined summary (see SKILL.md output format section) and write it to `review/reports/summary-{timestamp}.md`.

The summary should be self-contained — readable without the JSON reports. It's what the user will refer back to, share with others, or use to track progress on the created tasks.
