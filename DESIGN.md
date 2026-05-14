# Fused Memory System Design

## Overview

A unified memory system combining Graphiti (temporal knowledge graph) and Mem0 (vector
memory store) behind a single MCP interface, with a scheduled consolidation agent.

## Architecture

```
                          Fused MCP Server
                    ┌──────────────────────────┐
                    │                          │
  Agent ──────────► │  Tool Interface          │
                    │    │                     │
                    │    ├─► Write Router       │
                    │    │    ├─► Graphiti      │──► Neo4j/FalkorDB
                    │    │    └─► Mem0          │──► Qdrant
                    │    │                     │
                    │    └─► Read Fan-out       │
                    │         ├─► Graphiti      │
                    │         ├─► Mem0          │
                    │         └─► Merge + Rank  │
                    │                          │
                    └──────────────────────────┘

  Consolidation Agent (scheduled, runs every N turns or on timer)
    ├─► Reads both stores
    ├─► Deduplicates, merges, resolves contradictions
    ├─► Cross-store promotion/demotion
    └─► Writes back to both stores via the same backend clients
```

## Scope Model

Per-project store pair. Each project gets:
- One FalkorDB graph (via Graphiti `group_id` = `project_id`)
- One Qdrant collection (or filtered partition) for Mem0

```
scope = {
  project_id: str    # required — maps to Graphiti group_id + Mem0 filter
  agent_id:   str    # optional — which agent stored/queries this
  session_id: str    # optional — which session (maps to Mem0 run_id)
}
```

Upgrade path: add a global user-level Mem0 collection for preferences/norms that span
all projects. Read path queries project scope first, enriches with global scope.


## Memory Taxonomy

Six categories, routed to primary store by retrieval pattern:

| # | Category              | Primary Store | Retrieval Pattern              |
|---|-----------------------|---------------|--------------------------------|
| 1 | Entities & Relations  | Graphiti      | Graph traversal                |
| 2 | Temporal Facts        | Graphiti      | Time-windowed queries          |
| 3 | Decisions & Rationale | Graphiti      | Entity + time                  |
| 4 | Preferences & Norms   | Mem0          | Similarity to current task     |
| 5 | Procedural Knowledge  | Mem0          | Similarity to current intent   |
| 6 | Observations & Summary| Mem0          | Broad semantic similarity      |

Dual-nature memories are written to both stores at write time.


## Tool Schema

### Write Operations

#### `add_episode` (primary write path)

Full ingestion pipeline. Raw content goes in; structured memories come out in both
stores. This is the default and recommended write operation.

```
add_episode(
  content:            str       # raw text, conversation, or JSON
  source:             enum      # "message" | "text" | "json"
  project_id:         str       # required
  agent_id:           str       # optional — who is writing
  session_id:         str       # optional — session context
  reference_time:     datetime  # optional — when the content happened (default: now)
  saga:               str       # optional — group into a named thread/timeline
  source_description: str       # optional — e.g. "pair programming session"
)
```

**Internal flow:**

1. Create Graphiti episode (stores raw content, establishes provenance)
2. Graphiti pipeline extracts entities + edges → stored in Graphiti
3. LLM classifier examines extracted facts + original content
4. Memories classified as categories 4-6 → written to Mem0
5. Dual-nature memories → written to both stores
6. Return: episode ID + summary of what was extracted and routed

**Cost:** 5-15+ LLM calls (Graphiti extraction + dedup + classification + Mem0 add).
Latency-tolerant — queue and process async, return episode ID immediately.


#### `add_memory` (direct write path)

Lightweight classified write. Skips the extraction pipeline. Use when the agent has
already identified a specific, discrete memory to store.

```
add_memory(
  content:    str       # the memory itself (a discrete fact, preference, etc.)
  category:   enum      # one of the 6 taxonomy categories
  project_id: str       # required
  agent_id:   str       # optional
  session_id: str       # optional
  metadata:   object    # optional — arbitrary key-value pairs
  dual_write: bool      # optional — force write to both stores (default: false)
)
```

**Internal flow by category:**

- Categories 1-3 (Graphiti-primary):
  Create a lightweight episode (content = the fact, source = "text") to maintain
  provenance, but skip context chaining. Faster than full add_episode but still
  gives traceability.

- Categories 4-6 (Mem0-primary):
  Write directly to Mem0. No Graphiti episode created (these don't need graph
  provenance).

- `dual_write=true`:
  Both paths execute. Graphiti gets a lightweight episode; Mem0 gets a direct write.

**Cost:** 0-3 LLM calls depending on path. Low latency.


### Read Operations

#### `search` (primary read path)

Unified search across both stores with automatic fan-out and result merging.

```
search(
  query:       str            # natural language query
  project_id:  str            # required
  categories:  list[enum]     # optional — filter to specific taxonomy categories
  stores:      list[enum]     # optional — force "graphiti" | "mem0" | "both" (default: auto)
  limit:       int            # optional — max results (default: 10)
  time_range:  {start, end}   # optional — temporal filter (Graphiti)
  agent_id:    str            # optional — filter by authoring agent
  session_id:  str            # optional — filter by session
)
```

**Read routing (when stores = "auto"):**

```
query
  → LLM classifier or heuristic:
    "who/what/how related"        → Graphiti primary, Mem0 secondary
    "how do I / what's the rule"  → Mem0 primary, Graphiti secondary
    "what changed / when did"     → Graphiti only
    ambiguous / broad             → both in parallel
```

"Primary + secondary" means: query primary store first; if results are sparse
(below a relevance threshold), also query secondary and merge.

**Returns** a unified result list:

```
MemoryResult {
  id:              str
  content:         str          # the fact, preference, or summary
  category:        enum         # taxonomy category
  source_store:    str          # "graphiti" | "mem0"
  relevance_score: float        # normalized 0-1 across stores
  provenance:      list[str]    # episode IDs (Graphiti results only)
  temporal:        {valid_at, invalid_at}  # (Graphiti results only)
  entities:        list[str]    # related entity names (Graphiti results only)
  metadata:        object
}
```


#### `get_entity`

Direct entity lookup in Graphiti. Returns the entity node, its edges, and connected
entities up to a specified depth.

```
get_entity(
  name:       str       # entity name (fuzzy matched)
  project_id: str       # required
  depth:      int       # optional — traversal depth (default: 1)
)
```


#### `get_episodes`

Retrieve raw episodes from Graphiti. Useful for reviewing interaction history,
tracing provenance, or feeding the consolidation agent.

```
get_episodes(
  project_id:     str       # required
  last_n:         int       # optional — most recent N episodes
  reference_time: datetime  # optional — episodes before this time
  saga:           str       # optional — filter by saga/thread
  source:         enum      # optional — "message" | "text" | "json"
)
```


### Delete Operations

#### `delete_memory`

```
delete_memory(
  memory_id:  str       # the memory ID (from search results)
  store:      str       # "graphiti" | "mem0" (from search results)
  project_id: str       # required
)
```

For Graphiti: cascading delete — removes facts exclusive to the source episode.
For Mem0: direct vector deletion.


#### `delete_episode`

```
delete_episode(
  episode_id: str       # Graphiti episode UUID
  project_id: str       # required
  cascade:    bool      # optional — also delete exclusive facts (default: true)
)
```


### Management Operations

#### `get_status`

Health check and statistics for both backends.

```
get_status(
  project_id: str       # optional — stats for specific project
)

Returns: {
  graphiti: { connected, node_count, edge_count, episode_count }
  mem0:     { connected, memory_count }
  project_ids: list[str]
}
```


## Write Router: Classification

The write router (used by `add_episode` after extraction, and optionally for
`add_memory` validation) uses an LLM classifier with few-shot examples.

### Classification prompt structure

```
Given a memory extracted from an agent interaction, classify it into exactly one
primary category. If the memory has a strong secondary nature, also identify that.

Categories:
1. entities_and_relations — facts about things and how they connect
2. temporal_facts — state that changes over time, with temporal markers
3. decisions_and_rationale — choices made and why
4. preferences_and_norms — how things should be done, conventions, style
5. procedural_knowledge — how to do things, workflows, steps
6. observations_and_summaries — high-level takeaways, session recaps

Memory: "{content}"

Respond as JSON:
{
  "primary": "<category>",
  "secondary": "<category or null>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief>"
}
```

When `secondary` is non-null, the memory is dual-written.

### Heuristic pre-filter (skip LLM when obvious)

Before calling the LLM classifier, apply fast heuristics:
- Contains entity names + relationship verbs ("depends on", "owns", "uses") → entities_and_relations
- Contains temporal markers ("changed", "was", "since", "before", "deprecated") → temporal_facts
- Contains decision language ("chose", "decided", "because", "trade-off") → decisions_and_rationale
- Contains preference language ("prefer", "always", "never", "should", "convention") → preferences_and_norms
- Contains procedural language ("to do X", "steps", "first...then", "run") → procedural_knowledge
- Fallback → LLM classifier

Confidence threshold: if heuristic match is ambiguous (multiple triggers), fall through
to LLM.


## Read Router: Query Classification

Similar structure for routing reads:

```
Given a search query from an agent, determine which memory store(s) to search.

Query types:
- entity_lookup: "what is X", "what does X do", "how are X and Y related"
- temporal: "what changed", "when did", "before/after", "history of"
- relational: "what depends on X", "who owns", "what uses"
- preference: "how should I", "what's the convention", "style for"
- procedural: "how do I", "steps to", "process for"
- broad: general topic query, recap, summary

Query: "{query}"

Respond as JSON:
{
  "query_type": "<type>",
  "stores": ["graphiti", "mem0"],   // which to query
  "primary_store": "graphiti" | "mem0"  // which results to rank higher
}
```


## Consolidation Agent

Runs periodically (every N agent turns, or on a timer). Operates on both stores
within a single project scope.

### Trigger mechanism

Option A (interaction-count): Increment counter on every `add_episode`. Trigger
consolidation when counter reaches threshold (e.g., every 10 episodes).

Option B (time-based): Cron/timer triggers consolidation at fixed intervals
(e.g., every 30 minutes of active use).

Option C (hybrid): Whichever fires first.

Recommended: Option C. The counter catches bursts of activity; the timer catches
slow-burn sessions.

### Consolidation operations

#### Within Mem0:
1. Retrieve all memories since last consolidation (by timestamp)
2. LLM pass: identify duplicates, contradictions, stale entries
3. Merge overlapping memories (update + delete)
4. Resolve contradictions (keep most recent, annotate)
5. Cluster related memories and create/update summary entries

#### Within Graphiti:
1. Run `build_communities()` to detect entity clusters
2. LLM pass: update community summaries
3. Identify superseded temporal facts not yet invalidated
4. Clean up orphaned entities (no remaining edges)

#### Cross-store:
1. Compare Graphiti facts against Mem0 memories for contradictions
2. Promote frequently-accessed Mem0 memories to explicit Graphiti relationships
   (when a preference has solidified into a project norm)
3. Ensure dual-written memories are consistent across stores
4. Generate a consolidation report (what changed, what was merged/deleted)

### Consolidation agent tools

The consolidation agent gets a restricted tool set (analogous to Letta's sleeptime
agent):

- `search(...)` — read from both stores
- `add_memory(...)` — write classified memories (for promotions, summaries)
- `delete_memory(...)` — remove duplicates and stale entries
- `get_episodes(...)` — review recent interaction history
- `get_status(...)` — check store health
- `consolidation_complete(report)` — signal done, log report

It does NOT get `add_episode` — consolidation writes are not themselves episodes.


## Infrastructure

### Backend stack

```
FalkorDB          — Graphiti graph store (one graph per project_id)
Qdrant            — Mem0 vector store (one collection per project_id)
LLM (configurable)— Extraction, classification, dedup, consolidation
Embedder          — Embedding generation for both stores
```

### Deployment (development)

```
docker-compose:
  falkordb:    — graph database
  qdrant:      — vector database
  fused-mcp:   — the MCP server (Python, async)
```

### Deployment (production upgrade path)

```
  + persistent volumes for FalkorDB and Qdrant
  + separate Qdrant collection for global user preferences
  + consolidation agent as a separate process/container
  + observability: latency metrics per store, LLM call counts, classification accuracy
```


## Curator Tickets & Routing Invariant

**Invariant:** every curator ticket lives on exactly one fused-memory instance.

This is enforced at three independent levels:

- **OS-wide singleton lock.** `fused-memory/src/fused_memory/server/main.py:_acquire_singleton_lock`
  binds an abstract Unix socket (`\0fused-memory-singleton`) on process start.  Any second
  fused-memory process exits with `SystemExit(1)`.  At most one daemon can hold tickets on
  a given host.

- **Per-process SQLite ticket store.** `fused-memory/src/fused_memory/middleware/ticket_store.py`
  persists tickets in a per-instance database file (`<data_dir>/tickets.db`).  There is no
  replication or cross-instance ticket sync; a ticket created by one process cannot appear in
  another process's store.

- **`DASHBOARD_FUSED_MEMORY_URLS` is failover-only, not sharding.**
  `dashboard/src/dashboard/config.py:15-20` documents that the default `('http://localhost:8002',)`
  is the single canonical systemd-managed instance, and that the ports retired when the
  architecture consolidated to a shared server cannot host a legitimate fused-memory process
  (the singleton lock prevents it).  The multi-URL config provides alternate *network paths*
  to the same logical store, not separate stores.
  `dashboard/src/dashboard/data/metrics.py:fan_out_list_tickets` uses the same
  first-success-per-root semantics ("first success wins; avoid double-counting across failover
  URLs"), confirming the project-wide pattern.

**Routing consequence for the cancel proxy.** Because only one instance can own a given ticket,
a `not_found` response from the first reachable instance is authoritative: no other URL in the
list can hold the ticket.  `dashboard/src/dashboard/app.py:api_curator_cancel` short-circuits
on `not_found` and returns 404 immediately.  The executable contract for this behavior is
`dashboard/tests/test_api_curator_cancel.py:test_two_url_not_found_short_circuits_loop`
(asserts `call_count == 1`, only URL[0] called).

**Trigger for revisiting.** If a future infra change introduces ticket replication or
multi-region deployment, this invariant must be re-evaluated and the cancel proxy
short-circuit relaxed to fan-out-then-quorum.


## Escalation cross-task dedupe: re-run-on-next-invocation contract

### Problem statement

Escalation deduplication folds similar `infra_issue` escalations into a single
"parent" on disk.  Folding can be *same-task* (two calls from task 42 with the
same summary key) or *cross-task* (task 99 escalates with the same key while
task 42's parent is still pending).  This section documents the cross-task case,
which has subtler resume semantics than same-task folding.

When task 99 submits a blocker that dedupes into task 42's pending parent:

- No escalation file is written under task 99.  The child's escalation object
  exists only in memory during the `_submit_or_dedupe` call; the on-disk change
  is a mutation of task 42's parent file (incrementing `dedupe_count` and
  appending the child's synthetic id to `dedupe_children`).
- Task 99's agent receives `{'status': 'dedup_skipped', 'action': 'terminate_cleanly'}`
  and stops working.

### Contract

**The child workflow does not block waiting on the parent's resolution.**

The orchestrator workflow's `_wait_for_resolution` predicate queries
`queue.get_by_task(self.task_id, status='pending', level=0)`.  For task 99,
this returns `[]` because there is no on-disk pending escalation under task
99's own `task_id`.  The wait loop never enters `await self._escalation_event.wait()`.

**The orchestrator's normal re-invocation is the resume mechanism.**  When the
orchestrator cycles through its task queue again, task 99 is re-invoked.  By
that time the parent's resolution has fixed the shared infra condition (e.g.
the fused-memory MCP server came back online), so task 99's re-run proceeds
normally.  If the condition is not yet fixed, task 99 re-escalates and a fresh
parent is created.

### Why not per-child wake signals

Iterating `parent.dedupe_children` on resolution and emitting a wake signal per
child `task_id` would be a functional no-op given the contract above:

1. The child workflow has no `asyncio.Event` keyed by its own `task_id` (because
   it never entered the wait loop — see above).
2. Even if the event existed, `_on_escalation_resolved` (harness.py) only sets
   events for the *parent's* `task_id`; wiring a cross-task wake signal would
   require widening the `resolve_callback` contract and adding harness changes
   outside the `escalation/` module boundary.

The re-invocation approach is simpler, correctly isolated, and avoids the
complexity of a distributed wake signal over what is essentially a file-based queue.

### Why this is safe

The parent's resolution is semantically a claim that the shared infra condition
has been addressed.  Any child task that re-runs after the parent resolves will
no longer encounter the original infra issue and will proceed normally.  If the
infra issue is not actually shared — i.e. the child task hits a distinct issue —
it re-escalates on its next run and a fresh parent is created; the old resolved
parent's `dedupe_children` list plays no role in that new fold.

### Executable contract

`escalation/tests/test_server_dedupe.py::TestCrossTaskChildResumeContract::test_cross_task_child_resume_contract`
pins this contract end-to-end using only the public escalation API.  It verifies
that `get_by_task('99', status='pending', level=0)` returns `[]` before and
after parent resolution, that the parent is properly archived, and that task 99
can submit new escalations after the parent resolves with no lingering dedupe block.

`orchestrator/tests/test_cross_task_dedupe_reengagement.py::TestCrossTaskDedupeReengagement`
extends verification one layer up into the orchestrator integration boundary.
It exercises the real `Harness._on_escalation_resolved` callback and asserts the
per-task event fan-out directly: parent A's `asyncio.Event` is set on resolve;
child B's event is not.  It also verifies that after resolve the
`EscalationQueue` gates (`get_by_task(task_id, status='pending', level=0) == []`
and `has_open_l1(task_id) is False`) are clear for both A and B, making both
tasks eligible for re-acquisition by a natural scheduler sweep.  Together these
two suites pin the contract end-to-end across both the escalation-API layer and
the orchestrator-harness integration layer.
