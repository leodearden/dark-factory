---
name: review-briefing
description: "Create and maintain the review briefing — a project-specific context file that tells the /review skill what matters about a project: purpose, important scenarios, architectural decisions, conventions, and known gaps. The briefing captures durable truths that can't be inferred from code alone — not code structure (which goes stale). ALWAYS use this skill for: /review-briefing commands, creating or updating review/briefing.yaml, setting up review context for a project, when the user says 'create a review briefing' or 'update the briefing', and when /review suggests running /review-briefing because no briefing exists. Also use when the user wants to validate their existing briefing or see what's changed. This is NOT for: running the actual review (/review), implementing code, or fixing issues found during review."
---

# Review Briefing Generator

The briefing captures what `/review` can't figure out alone: what the project is *for*, which scenarios matter, what architectural decisions shape the design, what's intentionally incomplete and why, and which conventions have teeth.

Code structure, module paths, function signatures, call chains — `/review` discovers all of that fresh from the code at review time. Encoding it in the briefing creates a second source of truth that goes stale the moment someone renames a function. The briefing should contain only **durable truths that contextualise reviews**.

**Rule of thumb:** If you can discover it alone from the code, don't put it in the briefing.

## Parse invocation

```
/review-briefing                          → create or update briefing for entire project
/review-briefing --scope fused-memory     → create/update for one subproject only
/review-briefing --validate               → check existing briefing against current codebase
/review-briefing --diff                   → show what's changed since last briefing update
```

---

## Mode: Validate (`--validate`)

Quick structural check against the existing `review/briefing.yaml`:

1. **Subproject coverage** — detect subprojects in the repo not represented in the briefing
2. **Removed subprojects** — briefing references subprojects that no longer exist
3. **Task references** — each `known_gaps[].tracking` task ID exists in the task tree
4. **Known gap staleness** — tasks referenced in `known_gaps` that are now `done`

Output: pass/fail with specifics. Offer to fix.

If no `review/briefing.yaml` exists, say so and suggest running `/review-briefing`.

---

## Mode: Diff (`--diff`)

Show what's changed in the project since the briefing's `last_updated` timestamp:

1. **New subprojects** not in the briefing
2. **Known gaps resolved** — tasks in `known_gaps` now marked `done`
3. **New stubs** — `TODO`, `NotImplementedError`, `pass` bodies introduced since last update (candidates for `known_gaps`)
4. **Major structural changes** — new entry points, removed modules, renamed subprojects

Present a summary and suggest a full update if changes are significant.

---

## Mode: Create / Update (default)

### Step 0: Gather context

1. **Check for existing briefing** — if `review/briefing.yaml` exists, this is an **update**
2. **Search project memory** for decisions, conventions, and known tensions:
   ```
   search(query="architectural decisions, conventions, and design rationale", project_id="dark_factory")
   search(query="known gaps, deferred work, intentional limitations", project_id="dark_factory")
   ```
3. **Read documentation** — CLAUDE.md, DESIGN.md, architecture docs, PRDs
4. **Load task tree** — `get_tasks(project_root="/home/leo/src/dark-factory")` for active/blocked/deferred work

### Step 1: Exploration (parallel Sonnet agents)

Detect subprojects (directories with `pyproject.toml` or `package.json`). If `--scope` is set, explore only that one.

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

### Step 2: Synthesis (you, Opus)

Collect discovery outputs and the memory/documentation context from Step 0. Now synthesize your understanding before going to the user.

For each subproject, draft:

- **Purpose** — one or two sentences on what it's for and why it exists
- **Key scenarios** — the important use cases a reviewer should focus on (plain English, not code traces)
- **What "working" means** — how to tell if the subproject is functioning correctly (intent, not commands)
- **Key decisions** — architectural choices that shape the design and inform review judgment (especially ones that might look wrong to a reviewer who doesn't know the context)

For the project as a whole, draft:

- **Conventions** — rules and norms, especially any from memory where you notice tension, ambiguity, or gaps between different sources. Include the *rationale* when known.
- **Known gaps** — things that are intentionally incomplete, with *why* they were deferred and any tracking references. Describe gaps conceptually, not by filename — "there's a legacy in-memory queue superseded by the durable queue" is good; "queue_service.py is still in the codebase" is a code detail that `/review` will discover on its own and that goes stale if the file is renamed or removed
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

**Create mode:** Write `review/briefing.yaml` directly.

**Update mode:**
1. Show diff against existing briefing
2. Preserve sections marked `# human-edited`
3. Confirm with user before writing

### Step 5: Write observations to memory

Write anything you learned about the project's intent, priorities, or review context that isn't captured in the briefing itself:

```
add_memory(
  content="Review briefing created/updated. Key context: {notable discoveries about project intent, user priorities, or conventions}",
  category="observations_and_summaries",
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```

---

## Update mode

When a briefing already exists:

1. Load the existing briefing
2. Run exploration (Step 1) to detect structural changes
3. Diff against existing briefing:
   - New subprojects not covered
   - Subprojects removed or renamed
   - Known gaps where tracking tasks are now done
   - New conventions or decisions in memory since last update
4. Present **only the changes** to the user (don't re-interview unchanged sections)
5. Merge approved changes, preserving human edits

---

## Graceful degradation

| Missing | Impact | Behaviour |
|---------|--------|-----------|
| fused-memory | No memory context for conventions/decisions | Warn, derive from documentation only |
| Task tree | Can't cross-reference known gaps against tasks | Note gaps as "untracked" |
| CLAUDE.md / docs | Less context for conventions | Rely more on user interview |
| pyproject.toml | Can't auto-detect subprojects | Ask the user |

Never fail silently.

---

## Reference files

- `references/briefing-schema.md` — YAML schema for `review/briefing.yaml`
