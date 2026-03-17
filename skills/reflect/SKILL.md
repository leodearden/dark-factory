---
name: reflect
description: "End-of-session reflection that captures decisions, discoveries, surprises, and insights to fused-memory. Use this skill when the user wants to wrap up a session, reflect on what happened, save learnings, or says things like 'let's wrap up', 'end of session', 'save what we learned', 'reflect on this session', or '/reflect'. Also trigger when a session has been substantive and the user signals they're done."
---

# Session Reflection

You're wrapping up a work session. Your job is to review what happened and write the useful parts to fused-memory so future sessions have context. This is about institutional memory — capturing the things that would be lost when this conversation ends.

Write memories directly without waiting for user confirmation. The user may be an independent agent, or may have already moved on. Just do the reflection and write the results.

## What to capture

Scan the conversation for these categories. Cast a wide net — the memory system has good capacity, handles near-duplicates gracefully, and a background reconciliation process consolidates and distills related facts over time. The bar is: **anything at least moderately likely to be relevant to another session, or that could be consolidated with other facts to draw a useful conclusion.**

Err on the side of writing more rather than fewer memories. Many small facts that seem minor in isolation become valuable in aggregate once reconciliation connects them. You can always dial back later, but you can't recover unrecorded context.

### Decisions and rationale
Choices that were made and *why*. These are the highest-value memories because decisions without recorded rationale get revisited endlessly. Examples: "Chose X over Y because of Z", "Decided to defer X until Y", "Rejected approach X because it would break Y".

### Discoveries and surprises
Things that were learned that weren't obvious beforehand. A bug's root cause, an unexpected behavior, a dependency that turned out to work differently than expected, a performance characteristic discovered during testing.

### Conventions established
New patterns, naming rules, or architectural norms that emerged during the session.

### Facts and observations
Implementation details, relationships between components, API behaviors, library quirks, configuration notes — anything that took effort to figure out and might save time in a future session. Even if it seems minor, record it if someone might benefit from knowing it later.

### Session summary
A brief factual account of what was accomplished and what's left unfinished. This helps future sessions pick up where this one left off.

## What NOT to capture

- Things trivially derivable from the code itself (e.g., "function X takes arguments Y and Z")
- Anything in CLAUDE.md or other checked-in documentation

## Process

### 1. Review the conversation

Walk through the conversation and identify everything worth recording. Draft your memories — each one should be a self-contained fact that makes sense without the rest of the conversation as context.

A good memory reads clearly six months from now to someone who wasn't in this session.

### 2. Determine the project_id

Use the `project_id` from CLAUDE.md (e.g., `"dark_factory"`, `"reify"`). If there's no CLAUDE.md with a project_id, use the repo name or directory name.

### 3. Write memories

Write each memory as a separate `add_memory` call with the appropriate category:

| Memory type | Category |
|------------|----------|
| Decisions and rationale | `decisions_and_rationale` |
| Discoveries / surprises | `observations_and_summaries` |
| Conventions established | `preferences_and_norms` |
| Facts and observations | `observations_and_summaries` |
| Session summary | `observations_and_summaries` |

Use `agent_id: "claude-interactive"` (or whatever agent context you're in) and the correct `project_id`.

Write the content as a clear, standalone statement. Include enough context that the memory is useful without knowing which session produced it. For decisions, always include the reasoning — "we chose X" is nearly useless without "because Y".

### 4. Confirm

Briefly list what was saved — one line per memory with its category. No need to repeat full content.
