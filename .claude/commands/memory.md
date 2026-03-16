# Memory System Reference

## Quick Reference

| I want to... | Tool | Key args |
|---|---|---|
| Store a discrete fact | `add_memory` | content, project_id, category |
| Ingest raw content for extraction | `add_episode` | content, project_id, source |
| Find memories by topic | `search` | query, project_id |
| Look up a specific entity | `get_entity` | name, project_id |
| Review ingestion history | `get_episodes` | project_id, last_n |
| Remove a memory | `delete_memory` | memory_id, store, project_id |
| Remove an episode + its facts | `delete_episode` | episode_id, project_id |
| Check system health | `get_status` | project_id |

---

## Writing Good Memories

### `add_memory` — preferred for discrete facts (0-3 LLM calls)

Use when you have a specific, distilled piece of knowledge. Always include `project_id` and `agent_id`.

**Decisions & Rationale** (→ Graphiti):
```
add_memory(
  content="We chose PostgreSQL over MongoDB for the billing service because we need ACID transactions for payment processing",
  project_id="dark_factory",
  category="decisions_and_rationale",
  agent_id="claude-interactive"
)
```

**Preferences & Norms** (→ Mem0):
```
add_memory(
  content="Python code in this project uses ruff for formatting with line-length=100",
  project_id="dark_factory",
  category="preferences_and_norms",
  agent_id="claude-interactive"
)
```

**Observations & Summaries** (→ Mem0):
```
add_memory(
  content="Session summary: Implemented the context builder module, added unit tests. The ContextBuilder class produces structured briefings from memory + task state. Left TODO: integration test with live backends.",
  project_id="dark_factory",
  category="observations_and_summaries",
  agent_id="claude-interactive"
)
```

**Procedural Knowledge** (→ Mem0):
```
add_memory(
  content="To run the fused-memory server locally: 1) Start docker services (cd fused-memory/docker && docker-compose up -d), 2) cd fused-memory && uv run python -m fused_memory.server.main",
  project_id="dark_factory",
  category="procedural_knowledge",
  agent_id="claude-interactive"
)
```

**Entities & Relations** (→ Graphiti):
```
add_memory(
  content="The TaskInterceptor middleware wraps TaskmasterBackend and emits ReconciliationEvents for every task state transition",
  project_id="dark_factory",
  category="entities_and_relations",
  agent_id="claude-interactive"
)
```

**Temporal Facts** (→ Graphiti):
```
add_memory(
  content="As of 2026-03-15, the reconciliation system supports targeted reconciliation but full-cycle reconciliation is not yet implemented",
  project_id="dark_factory",
  category="temporal_facts",
  agent_id="claude-interactive"
)
```

### `add_episode` — for raw content needing extraction (5-15 LLM calls)

Use sparingly. Appropriate when you have a block of unstructured content (conversation transcript, meeting notes, design discussion) that contains multiple facts the system should extract.

```
add_episode(
  content="<paste of design discussion or meeting notes>",
  project_id="dark_factory",
  source="text",
  agent_id="claude-interactive",
  source_description="architecture discussion about auth system"
)
```

---

## Searching Effectively

### Query patterns

```
# Broad project context
search(query="project overview and architecture", project_id="dark_factory")

# Specific decisions
search(query="why was X chosen over Y", project_id="dark_factory", categories=["decisions_and_rationale"])

# Conventions
search(query="coding conventions and style rules", project_id="dark_factory", categories=["preferences_and_norms"])

# Entity relationships
search(query="what depends on the TaskInterceptor", project_id="dark_factory", stores=["graphiti"])

# Recent changes
search(query="what changed recently in the reconciliation system", project_id="dark_factory", categories=["temporal_facts"])

# How-to
search(query="how to run tests", project_id="dark_factory", categories=["procedural_knowledge"])
```

### Working with memory_hints on tasks

Tasks may carry `memory_hints` in their metadata — structured pointers for prefetching context:

```json
{
  "memory_hints": {
    "queries": ["reconciliation pipeline architecture", "task interceptor events"],
    "entities": ["ReconciliationEvent", "TaskInterceptor", "EventBuffer"]
  }
}
```

To execute hints:
1. Run each query via `search(query=..., project_id="dark_factory")`
2. Look up each entity via `get_entity(name=..., project_id="dark_factory")`
3. Compile results into working context before starting the task

### Filtering

- **By category**: `categories=["decisions_and_rationale", "preferences_and_norms"]`
- **By store**: `stores=["graphiti"]` or `stores=["mem0"]` (override auto-routing)
- **By agent**: `agent_id="claude-task-7"` (find what a specific agent wrote)

---

## Category Reference

| Category | Store | When to use | Examples |
|----------|-------|-------------|----------|
| `entities_and_relations` | Graphiti | Facts about things and how they connect | "X depends on Y", "X is part of Y" |
| `temporal_facts` | Graphiti | State that changes over time | "As of date, X is Y", "X was deprecated" |
| `decisions_and_rationale` | Graphiti | Choices made and why | "Chose X because Y", "Trade-off: X vs Y" |
| `preferences_and_norms` | Mem0 | Conventions, style rules | "Always use X", "Convention: Y" |
| `procedural_knowledge` | Mem0 | Workflows, how-to steps | "To do X: step 1, step 2" |
| `observations_and_summaries` | Mem0 | High-level takeaways | Session summaries, architecture overviews |

---

## Session Lifecycle

### Starting a session

1. **Search for project context:**
   ```
   search(query="project overview and current status", project_id="dark_factory")
   search(query="recent decisions and changes", project_id="dark_factory")
   search(query="active conventions and procedures", project_id="dark_factory")
   ```

2. **Check task tree:**
   ```
   get_tasks(project_root="/home/leo/src/dark-factory")
   ```

3. **If working on a specific task**, fetch it and execute its memory_hints:
   ```
   get_task(id="7", project_root="/home/leo/src/dark-factory")
   # Then execute any memory_hints queries and entity lookups
   ```

### During a session

- Write decisions and discoveries **immediately** — don't batch
- Each `add_memory` call should capture one discrete fact
- Include enough context in the content for the memory to be useful standalone

### Ending a session

Reflect on the session and write separate memories for:

1. **Decisions made** (category: `decisions_and_rationale`):
   - What was decided and why
   - What alternatives were considered

2. **Conventions discovered** (category: `preferences_and_norms`):
   - New patterns or rules established
   - Existing conventions that were clarified

3. **Session summary** (category: `observations_and_summaries`):
   - What was accomplished
   - What's left to do
   - Any blockers or open questions

---

## Context Briefing

To build a comprehensive context briefing (useful at session start or before a complex task):

1. **Project overview**: `search(query="project overview architecture goals", project_id="dark_factory")`
2. **Recent decisions**: `search(query="recent decisions and rationale", project_id="dark_factory", categories=["decisions_and_rationale"])`
3. **Active conventions**: `search(query="coding conventions and project norms", project_id="dark_factory", categories=["preferences_and_norms"])`
4. **Procedures**: `search(query="development workflows and procedures", project_id="dark_factory", categories=["procedural_knowledge"])`
5. **Task tree**: `get_tasks(project_root="/home/leo/src/dark-factory")`
6. **Recent activity**: `get_episodes(project_id="dark_factory", last_n=5)`

Compile results into a structured document with sections for each area. Skip sections with no results.
