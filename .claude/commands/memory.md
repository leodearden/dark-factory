# Memory System Reference

## Quick Reference

| I want to... | Tool | Key args |
|---|---|---|
| Store a discrete fact | `add_memory` | content, project_id, category, entities |
| Ingest raw content for extraction | `add_episode` | content, project_id, source, entities |
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

Declare what the write is about with `entities`. List the tasks the content names, as in `[{'kind': 'task', 'id': 3127}]`, or pass `[]` when you considered it and none apply. Omitting `entities` always succeeds, and the referents are then inferred. A declaration that your own content contradicts is rejected, and the rejection's hint states the accepted shape.

**Decisions & Rationale** (→ Graphiti):
```
add_memory(
  content="For Task 3127 we chose PostgreSQL over MongoDB for the billing service because we need ACID transactions for payment processing",
  project_id="dark_factory",
  category="decisions_and_rationale",
  agent_id="claude-interactive",
  entities=[{'kind': 'task', 'id': 3127}]
)
```

**Preferences & Norms** (→ Mem0):
```
add_memory(
  content="Python code in this project uses ruff for formatting with line-length=100",
  project_id="dark_factory",
  category="preferences_and_norms",
  agent_id="claude-interactive",
  entities=[]
)
```

**Observations & Summaries** (→ Mem0):
```
add_memory(
  content="Session summary for Task 928: Implemented the context builder module, added unit tests. The ContextBuilder class produces structured briefings from memory + task state. Left TODO: integration test with live backends.",
  project_id="dark_factory",
  category="observations_and_summaries",
  agent_id="claude-interactive",
  entities=[{'kind': 'task', 'id': 928}]
)
```

**Procedural Knowledge** (→ Mem0):
```
add_memory(
  content="To run the fused-memory server locally: 1) Start docker services (cd fused-memory/docker && docker-compose up -d), 2) cd fused-memory && uv run python -m fused_memory.server.main",
  project_id="dark_factory",
  category="procedural_knowledge",
  agent_id="claude-interactive",
  entities=[]
)
```

**Entities & Relations** (→ Graphiti):
```
add_memory(
  content="The TaskInterceptor middleware wraps TaskmasterBackend and emits ReconciliationEvents for every task state transition",
  project_id="dark_factory",
  category="entities_and_relations",
  agent_id="claude-interactive",
  entities=[]
)
```

**Temporal Facts** (→ Graphiti):
```
add_memory(
  content="As of 2026-03-15, the reconciliation system supports targeted reconciliation but full-cycle reconciliation is not yet implemented",
  project_id="dark_factory",
  category="temporal_facts",
  agent_id="claude-interactive",
  entities=[]
)
```

### Metadata vocabulary

`add_memory` also takes an optional `metadata` dict. Five keys are reserved and
validated on write:

- `topic` — kebab-case slug naming the subject; groups every entry about it
- `canonical` — the one authoritative entry for a topic (requires `topic`)
- `kind` — record type, from a closed registry; not `source`, which is writer provenance
- `parent_id` — UUID of a live entry this one attaches to
- `supersedes` — list of UUIDs this entry replaces

A small blessed set of conventional keys — `task_id`, `source`, `transition`,
`stage` and a few more — is already known and does not warn; use those exact
spellings rather than an `x_` variant of them, because metadata-keyed lookups
filter on them. Any key outside that set and the five above still writes but
warns to a census line; prefix a deliberate annotation `x_` and it passes
silently.

The exact shapes, validation rules and the kind registry itself live in
`fused-memory/src/fused_memory/memory_metadata.py` — the single normative
source. Consult it rather than this summary; the contract is
`docs/prds/memory-metadata-vocabulary.md` (V1).

### `add_episode` — for raw content needing extraction (5-15 LLM calls)

Use sparingly. Appropriate when you have a block of unstructured content (conversation transcript, meeting notes, design discussion) that contains multiple facts the system should extract.

```
add_episode(
  content="<paste of design discussion or meeting notes>",
  project_id="dark_factory",
  source="text",
  agent_id="claude-interactive",
  source_description="architecture discussion about auth system",
  entities=[]
)
```

---

## Searching Effectively

### Query patterns

```
# Specific decisions
search(query="why was X chosen over Y", project_id="dark_factory", categories=["decisions_and_rationale"])

# Conventions
search(query="conventions and gotchas for orchestrator merge lane", project_id="dark_factory", stores=["mem0"], categories=["preferences_and_norms", "procedural_knowledge"])

# Entity relationships
search(query="what depends on the TaskInterceptor", project_id="dark_factory", stores=["graphiti"])

# How one subject changed over time
search(query="how the reconciliation system changed", project_id="dark_factory", categories=["temporal_facts"])

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

1. **Recall memory context and the task tree:** run the queries in [Context Briefing](#context-briefing) below.

2. **If working on a specific task**, fetch it and execute its memory_hints:
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

To build a context briefing (at session start or before a complex task), run the queries below. Items 1-2 mirror the queries every dispatched agent's briefing fires; `shared/src/shared/briefing_queries.py` is the single home of their phrasing and of the measurements behind it, and where this section and that module disagree, the module wins. Items 3-4 are extra reads for an interactive session that the dispatched briefing does not make.

1. **Conventions for your working area**: Mem0-scoped, phrased from the prose words of the paths you work in. For example, `fused-memory/src/fused_memory/reconciliation/stages/` gives `fused memory reconciliation stages`:
   ```
   search(query="conventions and gotchas for fused memory reconciliation stages", project_id="dark_factory", stores=["mem0"], categories=["preferences_and_norms", "procedural_knowledge"])
   ```
   With no working area yet, use the same scoping with `query="conventions, norms and gotchas for working in this repository"`.
2. **Context for a specific task**: a semantic search phrased from the task's title and area words, never the bare task id, then the graph's record of the task. The entity name is `Task <id>`, as in `orchestrator/src/orchestrator/agents/memory_recall.py::MemoryRecall._task_entity`:
   ```
   search(query="<task title> <area words>", project_id="dark_factory")
   get_entity(name="Task 3127", project_id="dark_factory")
   ```
3. **Task tree**: `get_tasks(project_root="/home/leo/src/dark-factory")`
4. **Recent activity**: `get_episodes(project_id="dark_factory", last_n=5)`. This is the recency read, because `search` has no recency predicate.

Two queries are absent on purpose. There is no search for an overview of the project: memory holds no such entry, so read `CLAUDE.md`, `README.md` and `ARCHITECTURE.md` instead. There is no search for the latest decisions either: `search` cannot express recency, and that query measured 5/5 noise. The rationale lives in `shared/src/shared/briefing_queries.py::QUERY_SPECS`.

Compile results into a structured document with sections for each area. Skip sections with no results.
