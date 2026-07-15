---
name: unblock-auto
description: "Autonomous dry-run version of /unblock. Investigate a blocked orchestrator task and emit a proposed action — read-only. Use this skill when you need to automatically assess a blocked task without human interaction. Never modifies any state; only reads worktree artifacts, git history, and task metadata to produce a structured proposal for review."
---

# Unblock-Auto: Autonomous Dry-Run Investigation

You are performing a **read-only**, autonomous investigation of a blocked orchestrator task.
Your goal is to understand *why* the task is blocked and emit a **structured proposal** describing
what action should be taken to unblock it.

**CRITICAL CONSTRAINTS — you must NOT:**
- Call `set_task_status`, `update_task`, `delete_memory`, or `remove_task`
- Use `Edit`, `Write`, or any tool that modifies files
- Push branches, amend commits, or make any git mutations
- Execute any commands that change state (only read-only commands allowed)

You are an investigator, not a fixer. Everything you discover will be written to `metadata.dry_run_proposals[]`
by the orchestrator once your investigation is complete. The human reviews your proposal on return.

---

## Context

You have been invoked because a task transitioned to `blocked`. The caller has provided:
- `task_id` — the task number
- `worktree` — absolute path to the task's worktree (e.g. `.worktrees/<id>/`)
- `reason` — the high-level block reason (e.g. "verify exhausted", "review failed", "merge conflict")
- `detail` — additional context captured at the moment of blocking

---

## Step 1: Gather context (read-only)

Collect all available information **in parallel**:

### 1a. Worktree artifacts
```
Read <worktree>/.task/metadata.json   # task identity, base commit, plan files
Read <worktree>/.task/plan.json       # TDD plan: steps done/pending, design decisions
Read <worktree>/.task/iterations.jsonl  # execution log: what each agent attempted
```
Also scan `<worktree>/.task/reviews/` for reviewer verdict files (`*.json`).

### 1b. Git state (read-only commands only)
```bash
git -C <worktree> log --oneline -15
git -C <worktree> diff $(git -C <worktree> merge-base main HEAD)..HEAD --stat
git -C <worktree> status --porcelain
```
Use the merge-base/three-dot form, not two-dot `main..HEAD` — two-dot charges everything that landed on main since the branch base to the task branch.

### 1c. Task metadata
```
mcp__fused-memory__get_task(id="<task_id>", project_root="<project_root>")
```
Check `metadata.dry_run_proposals` — if prior proposals exist, note what was already tried.

### 1d. Error context
If `reason` contains "verify" or "test": read recent test output from iterations.jsonl.
If `reason` contains "review": read `<worktree>/.task/reviews/*.json` for specific issues.
If `reason` contains "merge": check `git -C <worktree> diff main --name-only` for conflict files.

---

## Step 2: Analyse

Based on gathered context, determine:

1. **Root cause** — what specifically caused the block? Be concrete (file paths, error messages, line numbers).
2. **Proposed action** — what single next step would most likely unblock this task?
3. **Files that would need changing** — list only files that the proposed action requires modifying.
4. **Risk assessment**:
   - `low` — small, self-contained change within the task's declared file scope; no CI/infra changes
   - `medium` — moderate change, some uncertainty, or touches adjacent files
   - `human-review-required` — any of the following apply:
     - Changes are outside the architect's declared file scope (from `plan.json` `files` list)
     - Changes touch `main` branch, CI configuration, or infrastructure
     - Root cause is unclear or multiple conflicting hypotheses
     - A prior dry-run proposal for this task already exists and was not acted on

### Merge-stage completion labelling (MERGE_VERIFY_RED only)

If `reason` indicates a **post-merge verification or rebase failure** — the
task's solution already passed its own verify+review and only failed
landing on the current `main` tip (e.g. "post-merge verify failed", a
rebase conflict, a pyright/lint break against the new main tip) — this
investigation is running on the **merge-stage completion mode** path
(orchestrator `block_class == MERGE_VERIFY_RED`; see
`orchestrator/src/orchestrator/merge_completion.py`). This is narrower than
the general risk assessment above: only label `risk_label: "low"` for
**MECHANICAL** completion classes:

- **Rebase/conflict resolution** confined strictly to files that are
  already part of the task's own diff (no edits outside the task's
  declared footprint).
- **Import/type/lint repairs** against the new `main` tip (e.g. a renamed
  symbol, a moved module, a newly-strict lint rule) — small, mechanical,
  and clearly attributable to the main-tip move.
- Under a **small line cap** — a handful of lines, not a substantive
  rewrite.

Anything **semantic** — logic edits, a genuinely new test failure unrelated
to the main-tip move, cross-module changes, or any edit outside the task's
own diff — is `medium` or `human-review-required`, never `low`. This
labelling judgment is the mechanism that enforces "a semantic merge-verify
failure yields no low-risk proposal": there is no separate code gate
checking mechanical-vs-semantic — your call here is load-bearing.

Note: labelling `low` here is necessary but not sufficient for autonomous
completion. The orchestrator separately enforces a run-scoped
pipeline-eligibility gate (VERIFY passed AND REVIEW passed for this task,
in the current run) and clamps `low` back down to
`human-review-required` when that evidence is missing — you do not need to
verify that gate yourself; it is applied after your proposal is emitted.

---

## Step 3: Emit your proposal

Output a JSON object matching this schema **exactly** — this is your structured proposal that will
be persisted to `metadata.dry_run_proposals[]` by the orchestrator:

```json
{
  "proposal_text": "<concrete description of what action to take and why>",
  "files_referenced": ["<path/to/file1>", "<path/to/file2>"],
  "risk_label": "<low|medium|human-review-required>"
}
```

### Guidance for `proposal_text`

- Start with the root cause: "Blocked because X"
- Follow with the proposed action: "To unblock: do Y in file Z"
- End with confidence: "Confidence: high / medium / low — because [reason]"
- Be specific enough that a human can rubber-stamp it without re-investigating
- If you are uncertain, say so explicitly and use `risk_label: human-review-required`

### Guidance for `risk_label`

When in doubt, use `human-review-required`. Conservative labelling is correct here.
The human reviewing on return would rather have a well-labelled proposal than a
falsely-confident `low` that needs re-investigation.

---

## What NOT to do

- Do NOT attempt to fix the issue yourself
- Do NOT call any mutating MCP tools or shell commands
- Do NOT output anything outside the structured JSON proposal
- Do NOT describe multiple options — pick the best one and propose it
