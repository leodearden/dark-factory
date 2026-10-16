---
name: review-briefing
description: "Create and maintain the review briefing — a project-specific context file that tells the /review skill what matters about a project: purpose, important scenarios, architectural decisions, conventions, and known gaps. The briefing captures durable truths that can't be inferred from code alone — not code structure (which goes stale). ALWAYS use this skill for: /review-briefing commands, creating or updating review/briefing.yaml, setting up review context for a project, when the user says 'create a review briefing' or 'update the briefing', and when /review suggests running /review-briefing because no briefing exists. Also use when the user wants to validate their existing briefing or see what's changed. This is NOT for: running the actual review (/review), implementing code, or fixing issues found during review."
---

# Review Briefing Generator

The briefing captures what `/review` can't figure out alone: what the project is *for*, which scenarios matter, what architectural decisions shape the design, what's intentionally incomplete and why, and which conventions have teeth.

Code structure, module paths, function signatures, call chains — `/review` discovers all of that fresh from the code at review time. Encoding it in the briefing creates a second source of truth that goes stale the moment someone renames a function. The briefing should contain only **durable truths that contextualise reviews**.

**Rule of thumb:** If you can discover it alone from the code, don't put it in the briefing.

## Standing project invariants (record in every briefing, enforce in every review)

- **TODO tracking invariant** — every real TODO in the codebase (TODO/FIXME/HACK comment markers, and stub idioms like Rust `todo!()`/`unimplemented!()`) must be tracked by a specific **non-terminal** task whose brief names resolving that TODO as a completion condition. A TODO merely *citing* a task id is not tracked — the cited task's brief must actually cover it ("TODO-cites-task ≠ tracked"). A TODO whose tracking task is now `done`/`cancelled` while the marker still applies is **orphaned** and must be re-attached to a live task or re-filed. When creating or updating a briefing, record this invariant under `conventions`; report untracked/orphaned TODOs surfaced during validation. One becomes a `known_gaps` entry only once a task owns it (`accepted_by`, below).
- **Deferred-task invariant** — a task must never sit in `deferred` status without a concrete, actionable "flip to pending" condition recorded on the task. If the gate is dependency-shaped, express it as dependency edges and keep the task `pending` (the scheduler respects deps; `deferred` quarantines). Human judgment gates follow the escalate-on-dispatch SOP instead of deferral: file the task `pending` with correct deps and a brief that, on dispatch, immediately escalates the decision to the human with concrete evidence — judgment gates are used sparingly, justified by nontrivial risk, and must present concrete information to weigh between alternatives (A/B, go/no-go). A `deferred` task without a recorded trigger is quarantine rot: nobody polls it.

## Parse invocation

```
/review-briefing                          → create or update briefing for entire project
/review-briefing --scope fused-memory     → create/update for one subproject only
/review-briefing --validate               → check existing briefing against current codebase
/review-briefing --diff                   → show what's changed since last briefing update
```

---

## Mode: Validate (`--validate`)

Per-section staleness checks against the existing `review/briefing.yaml`. There is no commit-count or age rule: a briefing is stale exactly where one of these checks fails. Resolve `project_root` (the main checkout: `git worktree list | head -1 | awk '{print $1}'`) and `project_id` (`fused_memory.project_id` in `<project_root>/dark-factory-orchestrator.yaml`) first. Read tasks by id (`get_task`, or `get_statuses(ids=[…])` for a batch) — never `get_tasks` over the whole store.

1. **`last_updated`** — the top-level key exists and parses as an ISO-8601 date. A header comment does not count.
2. **Subprojects == workspace members** — every workspace member (root `pyproject.toml` `[tool.uv.workspace].members`, Cargo `[workspace].members`, `package.json` `workspaces`) is a `subprojects` key, and every `subprojects` key is a member or an existing directory the briefing says why it covers. A member with no entry is a defect: the findings contract's area vocabulary is these keys (`docs/quality-findings-contract.md` §3).
3. **Known-gap pointers** — every `known_gaps` entry carries `accepted_by: <task id>` (a legacy `tracking:` is read as `accepted_by` and reported for migration). The task must exist and must not be `done`. A `cancelled` task carrying `metadata.x_acceptance_reason` is an acceptance (contract §7); `pending`/`in-progress`/`blocked`/`deferred` is in-flight or postponed work, valid until it lands and reported as `in-flight`. `done` → defect (the gap closed, or the pointer is wrong); `cancelled` without `x_acceptance_reason` → defect (abandoned, not accepted); no pointer → defect.
4. **Conventions' citations** — every task id cited in a `conventions` rule or `why` exists, and one cited as tracking or owning something is non-terminal; every cited path exists at HEAD, and every cited `path::symbol` resolves (`git grep -n -E '(def|class) <symbol>\b' -- <path>`). Line numbers in a citation are not checked.
5. **TODO tracking invariant** — sweep for TODO/FIXME/HACK markers and `todo!()`/`unimplemented!()` stubs; each real one maps to a non-terminal task briefed to resolve it. Report untracked and orphaned markers (tracking task `done`/`cancelled` while the marker still applies).
6. **Deferred-task invariant** — every `deferred` task records a concrete flip-to-pending condition; report those whose only gate is dependency-shaped (should be `pending` + dep edges) or whose condition has already fired. List deferred tasks with the read-only forensic query (`status = 'deferred'`, see `CLAUDE.md` §"Forensic reads of tasks.db"), not `get_tasks`.

Output: one line per defect, `<check>: <section path>: <what is wrong>`, then pass/fail.

- **Interactive:** offer to fix each defect.
- **From `/review`:** report only. `/review` copies the defect lines into its report's `method.extra.briefing_defects` and continues; nothing here blocks a review.

If no `review/briefing.yaml` exists, say so and suggest running `/review-briefing`.

---

## Mode: Diff (`--diff`)

Show what's changed in the project since the briefing's `last_updated` key (`git log --since=<last_updated>`):

1. **New subprojects** not in the briefing
2. **Known gaps resolved** — `accepted_by` tasks now marked `done`
3. **New stubs** — `TODO`, `NotImplementedError`, `pass` bodies introduced since last update (candidates for `known_gaps`)
4. **Major structural changes** — new entry points, removed modules, renamed subprojects

Present a summary and suggest a full update if changes are significant.

---

## Mode: Create / Update (default)

### Step 0: Gather context

1. **Check for existing briefing** — if `review/briefing.yaml` exists, this is an **update**
2. **Resolve the project** — `project_root` and `project_id` as in Validate mode. Never assume either.
3. **Search project memory** for decisions, conventions, and known tensions:
   ```
   search(query="architectural decisions, conventions, and design rationale", project_id="<project_id>")
   search(query="known gaps, deferred work, intentional limitations", project_id="<project_id>")
   ```
4. **Read documentation** — CLAUDE.md, DESIGN.md, architecture docs, PRDs
5. **Task context** — read the tasks the briefing and docs cite with `get_task`, and find deferred or blocked work with `search_tasks` plus the read-only forensic query for `deferred` (which `search_tasks` excludes). Never `get_tasks` over the whole store.

### Step 1: Exploration (parallel Sonnet agents)

Detect subprojects: every workspace member first (Validate check 2), then other top-level directories with `pyproject.toml`, `Cargo.toml` or `package.json`. If `--scope` is set, explore only that one.

Spawn one Sonnet agent per subproject to build a **working understanding** of what each subproject does. The goal is not to catalogue the code — it's to understand enough to ask the user smart questions.

**Agent prompt** (adapt per subproject):

```
Explore the {subproject_name} subproject at {subproject_root}. Your goal is to understand
what this subproject IS FOR and how it fits into the larger project. Return:

1. PURPOSE — What does this subproject do? What problem does it solve? (1-2 sentences)

2. KEY SCENARIOS — What are the important end-to-end things this subproject enables?
   Not code paths, but user/system scenarios. e.g. "An agent writes a memory and later
   retrieves it via search" or "A PRD is decomposed into tasks and implemented concurrently."

3. EXTERNAL DEPENDENCIES — What services, databases, or other subprojects does it need
   to function? What breaks if those are unavailable?

4. WHAT "WORKING" LOOKS LIKE — How would you verify this subproject is functioning?
   Describe in plain English, not specific commands.

5. ANYTHING SURPRISING — Patterns, decisions, or structures that wouldn't be obvious
   from a quick glance. Things a reviewer might misunderstand or flag incorrectly.
```

Use `model: "sonnet"`. Read enough code to understand purpose and structure, but don't catalogue every file.

### Step 2: Synthesis (coordinator, high effort)

Collect discovery outputs and the memory/documentation context from Step 0. Now synthesize your understanding before going to the user.

For each subproject, draft:

- **Purpose** — one or two sentences on what it's for and why it exists
- **Key scenarios** — the important use cases a reviewer should focus on (plain English, not code traces)
- **What "working" means** — how to tell if the subproject is functioning correctly (intent, not commands)
- **Key decisions** — architectural choices that shape the design and inform review judgment (especially ones that might look wrong to a reviewer who doesn't know the context)

For the project as a whole, draft:

- **Conventions** — rules and norms, especially any from memory where you notice tension, ambiguity, or gaps between different sources. Include the *rationale* when known.
- **Known gaps** — things that are intentionally incomplete, each owned by a task (`accepted_by`) whose record holds the *why*. Describe gaps conceptually, not by filename — "there's a legacy in-memory queue superseded by the durable queue" is good; "queue_service.py is still in the codebase" is a code detail that `/review` will discover on its own and that goes stale if the file is renamed or removed
- **Exclusions** — areas to skip in review, with reasons

### Step 3: Interview the user

Present your synthesized understanding and have a conversation. The goal is to fill in what you couldn't discover from code alone — intent, priority, and context.

The interview should minimize the user's explanation burden. You've already explored; now you're asking about the things you *couldn't* figure out on your own.

**3a. Purpose and scenarios**

Present your understanding of each subproject's purpose and key scenarios.

Ask: "Here's what I think each subproject is for and what matters about it. What am I getting wrong? What scenarios are most important to you — the ones where a regression would be most painful?"

**3b. Conventions and decisions**

Present the conventions you found in memory and documentation. Surface any tensions or ambiguities — places where different sources seem to disagree, or where a convention exists but the rationale isn't clear.

Ask: "I found these conventions. Some seem to have tension between them — [specific examples]. Are there rules a reviewer needs to know that aren't obvious from the code?"

**3c. Known gaps**

Present what you identified as intentionally incomplete.

Ask: "What's intentionally incomplete and why? Anything I'm treating as a gap that's actually done? Anything I think is done that's actually deferred?"

**3d. Anything else**

Ask: "Have I missed anything? Any areas where a reviewer would make wrong assumptions without context?"

### Step 4: Write the briefing

Compile the final YAML incorporating user feedback. See `references/briefing-schema.md` for the schema.

```bash
mkdir -p review
```

**Every known gap needs an owning task.** For a gap the user confirms that no task owns yet, file the acceptance (contract §7) and point at it:

```python
r = submit_task(project_root=project_root, planning_mode=True,
                title="Accepted gap: <what>", description="<what is incomplete>",
                details="Accepted by <who> on <date>. Revisit when <condition>.",
                metadata={"source": "review-briefing", "x_acceptance_reason": "<why>"})
set_task_status(id=r["task_id"], status="cancelled", project_root=project_root)
```

The reason goes in at submit because a cancelled task's metadata is frozen; never record an acceptance by deferring a task.

The briefing entry is then `{what: "<one line>", accepted_by: <task id>}`; the why lives on the task.

Set the top-level `last_updated` key to today's date on every write.

**Create mode:** Write `review/briefing.yaml` directly.

**Update mode:**
1. Show diff against existing briefing
2. Preserve sections marked `# human-edited`
3. Confirm with user before writing

Then run Validate mode and fix what it reports before finishing.

### Step 5: Write observations to memory

Write anything you learned about the project's intent, priorities, or review context that isn't captured in the briefing itself:

```
add_memory(
  content="Review briefing created/updated. Key context: {notable discoveries about project intent, user priorities, or conventions}",
  category="observations_and_summaries",
  project_id="<project_id>",
  agent_id="claude-review-briefing",
  entities=[]
)
```

---

## Update mode

When a briefing already exists:

1. Load the existing briefing
2. Run exploration (Step 1) to detect structural changes
3. Diff against existing briefing:
   - Every Validate-mode defect
   - New conventions or decisions in memory since last update
4. Present **only the changes** to the user (don't re-interview unchanged sections)
5. Merge approved changes, preserving human edits

---

## Graceful degradation

| Missing | Impact | Behaviour |
|---------|--------|-----------|
| fused-memory | No memory context for conventions/decisions | Warn, derive from documentation only |
| Task store | Can't check `accepted_by` pointers or file acceptances | Report Validate checks 3, 4 and 6 as not run; write no new known gaps |
| CLAUDE.md / docs | Less context for conventions | Rely more on user interview |
| Workspace manifest | Can't auto-detect subprojects | Ask the user |

Never fail silently.

---

## Reference files

- `references/briefing-schema.md` — YAML schema for `review/briefing.yaml`
