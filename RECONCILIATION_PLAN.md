# Sleep Mode Reconciliation — Implementation Plan

## Overview

Three-way sleep mode reconciliation system across Graphiti (temporal knowledge graph), Mem0 (vector memory store), and Taskmaster AI (task management). Inspired by Letta's sleep-time compute, adapted for a three-system architecture.

## Architecture

Three-stage sequential pipeline, triggered by event-driven payload accumulation:

```
Events buffer                     Pipeline
─────────────                     ────────

 episode added ──┐                ┌─────────────────────┐
 memory changed ─┤                │  Stage 1             │
 task completed ─┤──► Payload ──► │  Memory Consolidator │
 task blocked ───┤    Buffer      │  (Graphiti ↔ Mem0)   │
 task cancelled ─┤                └────────┬────────────┘
 tasks created ──┘                         │ report
                                  ┌────────▼────────────┐
                                  │  Stage 2             │
                                  │  Task-Knowledge Sync │
                                  │  (Memory ↔ Tasks)    │
                                  └────────┬────────────┘
                                           │ report
                                  ┌────────▼────────────┐
                                  │  Stage 3             │
                                  │  Cross-System        │
                                  │  Integrity Check     │
                                  └────────┬────────────┘
                                           │
                                  ┌────────▼────────────┐
                                  │  Journal + Report    │
                                  │  → LLM Judge (async) │
                                  └─────────────────────┘
```

Each stage gets a fresh context window. Stages communicate via structured reports.

## Settled Design Decisions

### Authority Model

| Conflict Type | Authority | Action |
|---|---|---|
| Knowledge contradicts task assumptions | Knowledge (more recent) | Modify/delete/re-scope task |
| Task intent contradicts current procedure | Task (represents new direction) | Update Mem0 procedure *after* task completes |
| Task marked done, no knowledge captured | **Codebase (git repo)** | Spawn explore agent to verify against repo state; write findings as new memories; update task if needed |
| Factual assertion about codebase disputed | **Codebase (git repo)** | Spawn explore agent to check ground truth |
| Duplicate knowledge across stores | Most recent / highest confidence | Deduplicate |
| AI-generated task content contradicts knowledge graph | Knowledge graph | Flag/modify task, add correction context |

### Trigger Model

- **Full sleep cycle**: Event-driven payload accumulation with size threshold + max staleness bound. Events buffer as they occur; trigger when buffer crosses size threshold OR oldest event exceeds max age. No overlapping runs; events during a run buffer for next run.
- **Targeted reconciliation**: Synchronous, triggered by task state transitions (done, blocked, cancelled, bulk creation). Blocks the agent until complete (with latency logged to dashboard). Scoped to the specific task and its immediate knowledge neighborhood. Cross-cutting implications deferred to next full cycle.

### Task Enrichment

Tasks get **memory hints** (structured metadata with entity references + semantic queries), NOT inline content duplication. This avoids multiple points of truth going stale. Format:

```json
{
  "memory_hints": {
    "entities": ["EntityA", "EntityB"],
    "queries": ["semantic query describing information need"]
  }
}
```

Executing agents auto-fetch these at plan/execute/review time. Hints on completed tasks become static — no further maintenance.

### Concurrency Model

- Optimistic (no locks)
- Full operation journal with before/after state, reasoning, evidence
- Concurrent access tracking for stale-write analysis
- Dashboard: stale-write frequency, blocked-time for targeted reconciliation, buffer metrics

### Safety Model

- All operations journaled (operation type, target system, before/after, reasoning, evidence, stage, timestamps, run ID)
- Async LLM-as-judge reviews journal per-run
- Severity escalation: minor → auto-fix | moderate → rollback + re-run | serious/repeated → halt
- Judge also tracks error rate trends across runs

### Stage Tool Access

| Tool | Stage 1 | Stage 2 | Stage 3 | Explore Agent |
|---|---|---|---|---|
| Memory read (search, get_entity, get_episodes) | yes | yes | yes | **no** |
| Memory write (add_memory, delete_memory) | yes | yes | no | no |
| Task read | no | yes | yes | no |
| Task write (create, modify, delete) | no | yes | no | no |
| `verify_against_codebase` | yes | yes | yes | n/a (it *is* the explore agent) |
| Codebase read (files, git) | no | no | no | **yes** |
| `stage_complete(report)` | yes | yes | yes | no |

Stage 3 is deliberately read-only + verify. Separation of concerns: the integrity checker detects problems but doesn't fix them — findings feed next cycle's Stage 1/Stage 2.

### Explore Agent (verify_against_codebase)

- Structured `verify_against_codebase` tool (not a generic `spawn_agent`)
- Explore agent is strictly read-only on the codebase
- **No access to memory systems or task system** (unbiased investigation)
- Returns VerificationResult with verdict, confidence, evidence (file paths, snippets), git context
- Reconciliation agent reasons about findings; explore agent just reports facts

### Targeted Reconciliation Interaction with Full Cycles

- Targeted reconciliation's mutations go into the payload buffer (events like any other)
- Does NOT reset the "last reconciled" watermark (only full cycles advance it)
- Events marked with `source: "targeted_reconciliation"` vs `source: "agent"`
- Can run concurrently with full cycles (optimistic model, logged as concurrent access)

### Scope

Project-scoped only for now.

### Middleware

Task state transitions intercepted at MCP layer. The fused-memory server proxies task operations to Taskmaster, with middleware that intercepts state transitions and runs targeted reconciliation synchronously before returning.

---

## Module Structure

New and modified files within `fused-memory/`:

```
src/fused_memory/
├── server/
│   ├── main.py                          # MODIFY — start event buffer, reconciliation loop
│   └── tools.py                         # MODIFY — add task proxy tools
├── services/
│   ├── memory_service.py                # MODIFY — emit events on writes
│   └── queue_service.py                 # (unchanged)
├── backends/
│   ├── graphiti_client.py               # (unchanged)
│   ├── mem0_client.py                   # (unchanged)
│   └── taskmaster_client.py             # NEW — MCP client proxy to Taskmaster
├── models/
│   ├── enums.py                         # MODIFY — add event types, stage IDs
│   ├── memory.py                        # MODIFY — add MemoryHints model
│   ├── scope.py                         # (unchanged)
│   └── reconciliation.py               # NEW — events, journal entries, reports, payloads
├── routing/
│   ├── classifier.py                    # (unchanged)
│   └── router.py                        # (unchanged)
├── reconciliation/
│   ├── __init__.py
│   ├── event_buffer.py                  # NEW — event accumulation + trigger logic
│   ├── harness.py                       # NEW — pipeline orchestrator
│   ├── journal.py                       # NEW — operation logging + watermark tracking
│   ├── targeted.py                      # NEW — targeted reconciliation for task transitions
│   ├── judge.py                         # NEW — async LLM-as-judge
│   ├── verify.py                        # NEW — verify_against_codebase tool
│   ├── agent_loop.py                    # NEW — generic LLM agent loop with tool dispatch
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py                      # NEW — base stage (tool defs, payload assembly)
│   │   ├── memory_consolidator.py       # NEW — Stage 1
│   │   ├── task_knowledge_sync.py       # NEW — Stage 2
│   │   └── integrity_check.py           # NEW — Stage 3
│   └── prompts/
│       ├── __init__.py
│       ├── stage1.py                    # NEW — Stage 1 system prompt
│       ├── stage2.py                    # NEW — Stage 2 system prompt
│       ├── stage3.py                    # NEW — Stage 3 system prompt
│       └── judge.py                     # NEW — Judge system prompt
├── middleware/
│   ├── __init__.py
│   └── task_interceptor.py              # NEW — intercepts task status transitions
└── config/
    └── schema.py                        # MODIFY — add reconciliation config section

config/
└── config.yaml                          # MODIFY — add reconciliation + taskmaster sections

tests/
├── test_event_buffer.py                 # NEW
├── test_journal.py                      # NEW
├── test_harness.py                      # NEW
├── test_targeted.py                     # NEW
├── test_taskmaster_client.py            # NEW
├── test_task_interceptor.py             # NEW
└── test_agent_loop.py                   # NEW
```

---

## Implementation Phases

### Phase 1: Foundation — Models, Journal, Event Buffer, Config

#### 1a. `models/reconciliation.py` — All Pydantic models

```python
# Event types emitted by the system
class EventType(str, Enum):
    episode_added = "episode_added"
    memory_added = "memory_added"
    memory_deleted = "memory_deleted"
    task_status_changed = "task_status_changed"
    task_created = "task_created"
    task_modified = "task_modified"
    task_deleted = "task_deleted"
    tasks_bulk_created = "tasks_bulk_created"  # parse_prd, expand_task

class EventSource(str, Enum):
    agent = "agent"                          # primary agent action
    targeted_reconciliation = "targeted_reconciliation"
    full_reconciliation = "full_reconciliation"

class StageId(str, Enum):
    memory_consolidator = "memory_consolidator"     # Stage 1
    task_knowledge_sync = "task_knowledge_sync"      # Stage 2
    integrity_check = "integrity_check"              # Stage 3

# Individual event in the buffer
class ReconciliationEvent(BaseModel):
    id: str                      # UUID
    type: EventType
    source: EventSource
    project_id: str
    timestamp: datetime
    payload: dict                # Event-specific data (task_id, memory_id, etc.)

# Journal entry — one per reconciliation operation
class JournalEntry(BaseModel):
    id: str                      # UUID
    run_id: str                  # Groups entries within a reconciliation run
    stage: StageId
    timestamp: datetime
    operation: str               # "create_memory", "delete_memory", "modify_task", etc.
    target_system: str           # "graphiti", "mem0", "taskmaster"
    before_state: dict | None    # Snapshot before mutation
    after_state: dict | None     # Snapshot after mutation
    reasoning: str               # Why the agent made this choice
    evidence: list[dict]         # Search results, verify findings, etc.

# Reconciliation run metadata
class ReconciliationRun(BaseModel):
    id: str                      # UUID
    project_id: str
    run_type: str                # "full" | "targeted"
    trigger_reason: str          # "buffer_size", "max_staleness", "task_done:42", etc.
    started_at: datetime
    completed_at: datetime | None
    events_processed: int
    stage_reports: dict[str, "StageReport"]  # StageId -> StageReport
    status: str                  # "running", "completed", "failed", "rolled_back"

# Stage report — output of each pipeline stage
class StageReport(BaseModel):
    stage: StageId
    started_at: datetime
    completed_at: datetime
    actions_taken: list[dict]    # Summary of each action
    items_flagged: list[dict]    # Items for next stage or next cycle
    stats: dict                  # Counts: created, modified, deleted, verified
    llm_calls: int
    tokens_used: int

# Memory hints attached to tasks
class MemoryHints(BaseModel):
    entities: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)

# Verification result from explore agent
class VerificationResult(BaseModel):
    verdict: str                 # "confirmed" | "contradicted" | "inconclusive"
    confidence: float
    evidence: list[dict]         # [{file_path, line_range, snippet, relevance}]
    summary: str
    git_context: dict | None     # {latest_relevant_commit, author, date}

# Reconciliation watermark — tracks what's been processed
class Watermark(BaseModel):
    project_id: str
    last_full_run_id: str | None
    last_full_run_completed: datetime | None
    last_episode_timestamp: datetime | None     # Graphiti
    last_memory_timestamp: datetime | None      # Mem0
    last_task_change_timestamp: datetime | None  # Taskmaster

# Judge verdict
class JudgeVerdict(BaseModel):
    run_id: str
    reviewed_at: datetime
    severity: str                # "ok" | "minor" | "moderate" | "serious"
    findings: list[dict]         # [{entry_id, issue, severity, recommendation}]
    action_taken: str            # "none" | "auto_fix" | "rollback" | "halt"
```

#### 1b. `reconciliation/journal.py` — Persistence layer

SQLite-backed journal (file per project, stored alongside the project data). Provides:

```python
class ReconciliationJournal:
    def __init__(self, db_path: Path):
        """Initialize SQLite database with schema."""

    # Watermark management
    async def get_watermark(self, project_id: str) -> Watermark
    async def update_watermark(self, watermark: Watermark) -> None

    # Journal entries
    async def add_entry(self, entry: JournalEntry) -> None
    async def get_entries(self, run_id: str) -> list[JournalEntry]

    # Run tracking
    async def start_run(self, run: ReconciliationRun) -> None
    async def complete_run(self, run_id: str, status: str) -> None
    async def get_run(self, run_id: str) -> ReconciliationRun
    async def get_recent_runs(self, project_id: str, limit: int) -> list[ReconciliationRun]
    async def is_run_active(self, project_id: str) -> bool

    # Judge verdicts
    async def add_verdict(self, verdict: JudgeVerdict) -> None
    async def get_recent_verdicts(self, project_id: str, limit: int) -> list[JudgeVerdict]

    # Dashboard queries
    async def get_stats(self, project_id: str, since: datetime) -> dict
```

SQLite tables: `watermarks`, `runs`, `journal_entries`, `judge_verdicts`. All keyed by project_id. Timestamps indexed for efficient range queries.

Storage location: `{data_dir}/{project_id}/reconciliation.db` — configurable `data_dir` in config.yaml.

#### 1c. `reconciliation/event_buffer.py` — Event accumulation

```python
class EventBuffer:
    def __init__(self, config: ReconciliationConfig, journal: ReconciliationJournal):
        self._buffer: dict[str, list[ReconciliationEvent]]  # project_id -> events
        self._lock: asyncio.Lock                              # per-project locks
        self._active_runs: set[str]                           # project_ids with active runs

    async def push(self, event: ReconciliationEvent) -> None:
        """Add event to buffer. Check trigger conditions."""

    async def should_trigger(self, project_id: str) -> tuple[bool, str]:
        """Check if buffer crosses threshold or max staleness.
        Returns (should_trigger, reason)."""

    async def drain(self, project_id: str) -> list[ReconciliationEvent]:
        """Atomically drain buffer for a project, returning events.
        Subsequent events go into fresh buffer."""

    async def mark_run_active(self, project_id: str) -> bool:
        """Mark a run as active. Returns False if already active (skip)."""

    async def mark_run_complete(self, project_id: str) -> None:
        """Mark run complete, allowing new triggers."""

    def get_buffer_stats(self, project_id: str) -> dict:
        """For dashboard: buffer size, oldest event age."""
```

Trigger logic (in `should_trigger`):
```python
events = self._buffer.get(project_id, [])
if not events:
    return False, ""
if project_id in self._active_runs:
    return False, ""  # Don't overlap
if len(events) >= self.config.buffer_size_threshold:
    return True, f"buffer_size:{len(events)}"
oldest = min(e.timestamp for e in events)
if (now - oldest).total_seconds() > self.config.max_staleness_seconds:
    return True, f"max_staleness:{oldest.isoformat()}"
return False, ""
```

Integration: `MemoryService.add_episode` and `add_memory` push events to the buffer after successful writes. Task proxy tools push events after successful Taskmaster operations. The buffer is checked by the harness's background loop.

#### 1d. Config additions

Add to `config/schema.py`:

```python
class TaskmasterConfig(BaseModel):
    """Connection to Taskmaster MCP server."""
    transport: str = "stdio"          # "stdio" | "http"
    command: str = "node"             # For stdio transport
    args: list[str] = ["mcp-server/server.js"]
    cwd: str = ""                     # Taskmaster project root
    http_url: str = ""                # For http transport
    project_root: str = ""            # --projectRoot for Taskmaster tools
    tool_mode: str = "standard"       # core | standard | all

class ReconciliationConfig(BaseModel):
    """Sleep mode reconciliation settings."""
    enabled: bool = True
    data_dir: str = "./data/reconciliation"

    # Buffer triggers
    buffer_size_threshold: int = 10
    max_staleness_seconds: int = 1800  # 30 minutes

    # Agent settings
    agent_llm_provider: str = "anthropic"
    agent_llm_model: str = "claude-sonnet-4-20250514"
    agent_max_tokens: int = 8192
    agent_max_steps: int = 50         # Max tool calls per stage

    # Judge settings
    judge_enabled: bool = True
    judge_llm_provider: str = "anthropic"
    judge_llm_model: str = "claude-sonnet-4-20250514"

    # Explore agent
    explore_codebase_root: str = ""   # Git repo root for verify_against_codebase

    # Safety
    max_mutations_per_stage: int = 50  # Circuit breaker
    halt_on_judge_serious: bool = True
```

Add to `config.yaml`:

```yaml
taskmaster:
  transport: "stdio"
  command: "node"
  args: ["${TASKMASTER_DIR:../taskmaster-ai}/mcp-server/server.js"]
  project_root: "${PROJECT_ROOT:.}"
  tool_mode: "standard"

reconciliation:
  enabled: true
  data_dir: "${RECONCILIATION_DATA_DIR:./data/reconciliation}"
  buffer_size_threshold: 10
  max_staleness_seconds: 1800
  agent_llm_provider: "anthropic"
  agent_llm_model: "claude-sonnet-4-20250514"
  agent_max_steps: 50
  judge_enabled: true
  explore_codebase_root: "${PROJECT_ROOT:.}"
```

---

### Phase 2: Taskmaster Integration — Client, Proxy Tools, Middleware

#### 2a. `backends/taskmaster_client.py` — MCP client to Taskmaster

Uses the `mcp` library's client capabilities to connect to Taskmaster's MCP server:

```python
class TaskmasterBackend:
    def __init__(self, config: TaskmasterConfig):
        self.config = config
        self._client: ClientSession | None = None

    async def initialize(self) -> None:
        """Start Taskmaster MCP server process, establish client session."""
        # For stdio: spawn node process, connect via stdio transport
        # For http: connect to running Taskmaster HTTP endpoint

    async def close(self) -> None:
        """Shut down client session and server process."""

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a Taskmaster MCP tool and return the result."""

    # Typed convenience methods for common operations:
    async def get_tasks(self, project_root: str, tag: str | None = None) -> dict
    async def get_task(self, task_id: str, project_root: str, tag: str | None = None) -> dict
    async def set_task_status(self, task_id: str, status: str, project_root: str, ...) -> dict
    async def add_task(self, prompt: str | None = None, title: str | None = None, ...) -> dict
    async def update_task(self, task_id: str, prompt: str, project_root: str, ...) -> dict
    async def add_subtask(self, parent_id: str, title: str, ...) -> dict
    async def remove_task(self, task_id: str, project_root: str, ...) -> dict
    async def add_dependency(self, task_id: str, depends_on: str, ...) -> dict
    async def remove_dependency(self, task_id: str, depends_on: str, ...) -> dict
    async def validate_dependencies(self, project_root: str, ...) -> dict
    async def expand_task(self, task_id: str, project_root: str, ...) -> dict
    async def parse_prd(self, input_path: str, project_root: str, ...) -> dict
```

Implementation note: The `mcp` Python library (`mcp>=1.9.4` already in deps) provides `ClientSession` that can connect to stdio or SSE/HTTP MCP servers. Use `StdioServerParameters` + `stdio_client` context manager for stdio transport, or `StreamableHTTPTransport` for HTTP.

#### 2b. `middleware/task_interceptor.py` — Status transition middleware

```python
class TaskInterceptor:
    """Wraps Taskmaster operations, intercepts state transitions for targeted reconciliation."""

    def __init__(
        self,
        taskmaster: TaskmasterBackend,
        targeted_reconciler: "TargetedReconciler",
        event_buffer: EventBuffer,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer

    STATUS_TRIGGERS = {"done", "blocked", "cancelled", "deferred"}
    BULK_TRIGGERS = {"parse_prd", "expand_task"}

    async def set_task_status(self, task_id: str, status: str, project_root: str, **kw) -> dict:
        """Proxy to Taskmaster, then run targeted reconciliation if status triggers it."""
        # 1. Get before-state
        before = await self.taskmaster.get_task(task_id, project_root)

        # 2. Execute status change
        result = await self.taskmaster.set_task_status(task_id, status, project_root, **kw)

        # 3. Emit event
        event = ReconciliationEvent(
            type=EventType.task_status_changed,
            source=EventSource.agent,
            project_id=project_root,
            payload={"task_id": task_id, "old_status": before["status"], "new_status": status},
            ...
        )
        await self.buffer.push(event)

        # 4. If triggering status, run targeted reconciliation synchronously
        if status in self.STATUS_TRIGGERS:
            recon_result = await self.reconciler.reconcile_task(
                task_id=task_id,
                transition=status,
                project_id=project_root,
                task_before=before,
            )
            result["reconciliation"] = recon_result

        return result

    async def expand_task(self, task_id: str, project_root: str, **kw) -> dict:
        """Proxy expand_task, then run bulk creation reconciliation."""
        result = await self.taskmaster.expand_task(task_id, project_root, **kw)
        event = ReconciliationEvent(
            type=EventType.tasks_bulk_created,
            source=EventSource.agent,
            project_id=project_root,
            payload={"parent_task_id": task_id, "operation": "expand_task"},
            ...
        )
        await self.buffer.push(event)
        recon_result = await self.reconciler.reconcile_bulk_tasks(
            parent_task_id=task_id,
            project_id=project_root,
        )
        result["reconciliation"] = recon_result
        return result

    # Pass-through operations — emit event but no targeted reconciliation
    async def add_task(self, **kw) -> dict:
        result = await self.taskmaster.add_task(**kw)
        await self.buffer.push(ReconciliationEvent(type=EventType.task_created, ...))
        return result

    # Pure reads — direct pass-through, no events
    async def get_tasks(self, **kw) -> dict:
        return await self.taskmaster.get_tasks(**kw)

    async def get_task(self, **kw) -> dict:
        return await self.taskmaster.get_task(**kw)
```

#### 2c. `server/tools.py` — Add task proxy tools

Add task tools to `create_mcp_server()`. These are registered alongside the existing memory tools:

```python
def create_mcp_server(
    memory_service: MemoryService,
    task_interceptor: TaskInterceptor | None = None,  # NEW parameter
) -> FastMCP:
    # ... existing memory tools ...

    if task_interceptor:
        @mcp.tool()
        async def get_tasks(project_root: str, tag: str | None = None) -> dict:
            """List all tasks in the project."""
            return await task_interceptor.get_tasks(project_root=project_root, tag=tag)

        @mcp.tool()
        async def get_task(id: str, project_root: str, tag: str | None = None) -> dict:
            """Get a single task by ID."""
            return await task_interceptor.get_task(task_id=id, project_root=project_root, tag=tag)

        @mcp.tool()
        async def set_task_status(id: str, status: str, project_root: str, ...) -> dict:
            """Update task status. Triggers targeted reconciliation for
            done/blocked/cancelled transitions."""
            return await task_interceptor.set_task_status(
                task_id=id, status=status, project_root=project_root, ...
            )

        # ... add_task, update_task, add_subtask, remove_task,
        #     add_dependency, remove_dependency, expand_task, parse_prd ...
```

The tool descriptions and parameter schemas should match Taskmaster's originals closely so agents don't need to adjust their behavior. Add a note about reconciliation in the instructions string.

#### 2d. `server/main.py` — Wire everything together

```python
async def run_server():
    # ... existing config + memory_service init ...

    # Initialize Taskmaster backend
    taskmaster = None
    task_interceptor = None
    if config.taskmaster:
        taskmaster = TaskmasterBackend(config.taskmaster)
        await taskmaster.initialize()

    # Initialize reconciliation system
    reconciliation_harness = None
    event_buffer = None
    if config.reconciliation.enabled:
        journal = ReconciliationJournal(Path(config.reconciliation.data_dir))
        event_buffer = EventBuffer(config.reconciliation, journal)

        # Targeted reconciler (needs memory_service + taskmaster + journal)
        targeted = TargetedReconciler(memory_service, taskmaster, journal, config)

        # Task interceptor (wraps taskmaster with middleware)
        if taskmaster:
            task_interceptor = TaskInterceptor(taskmaster, targeted, event_buffer)

        # Wire event emission into memory_service
        memory_service.set_event_buffer(event_buffer)

        # Full reconciliation harness (background loop)
        reconciliation_harness = ReconciliationHarness(
            memory_service, taskmaster, journal, event_buffer, config
        )
        asyncio.create_task(reconciliation_harness.run_loop())

    # Create MCP server with both memory and task tools
    mcp = create_mcp_server(memory_service, task_interceptor)
    # ... rest of transport setup ...
```

#### 2e. `services/memory_service.py` — Add event emission

Add a `set_event_buffer` method and emit events from write/delete operations:

```python
class MemoryService:
    def __init__(self, config):
        # ... existing ...
        self._event_buffer: EventBuffer | None = None

    def set_event_buffer(self, buffer: EventBuffer) -> None:
        self._event_buffer = buffer

    async def _emit_event(self, event: ReconciliationEvent) -> None:
        if self._event_buffer:
            await self._event_buffer.push(event)

    async def add_episode(self, ...) -> AddEpisodeResponse:
        # ... existing logic ...
        await self._emit_event(ReconciliationEvent(
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=project_id,
            payload={"episode_id": episode_id, "content_preview": content[:200]},
            ...
        ))
        return response

    # Similarly for add_memory, delete_memory, delete_episode
```

---

### Phase 3: Agent Loop and Reconciliation Pipeline

#### 3a. `reconciliation/agent_loop.py` — Generic agent executor

A reusable agent loop that runs an LLM with tools until it calls a terminal tool:

```python
class AgentLoop:
    """Runs an LLM agent with tool access until it signals completion."""

    def __init__(
        self,
        config: ReconciliationConfig,
        system_prompt: str,
        tools: dict[str, ToolDefinition],  # name -> {function, schema, description}
        terminal_tool: str = "stage_complete",  # Tool that ends the loop
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.terminal_tool = terminal_tool
        self._journal_entries: list[JournalEntry] = []
        self._mutation_count: int = 0
        self.llm_call_count: int = 0
        self.token_count: int = 0

    async def run(self, initial_payload: str) -> tuple[dict, list[JournalEntry]]:
        """Execute agent loop. Returns (terminal_tool_args, journal_entries).

        1. Send system_prompt + initial_payload to LLM
        2. Process tool calls in a loop
        3. Journal each mutation
        4. Stop when terminal_tool is called or max_steps reached
        5. Circuit-break at max_mutations_per_stage
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_payload},
        ]

        for step in range(self.config.agent_max_steps):
            response = await self._call_llm(messages)

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call.name == self.terminal_tool:
                    return tool_call.arguments, self._journal_entries

                result = await self._execute_tool(tool_call)
                tool_results.append(result)

            messages.append(response.to_message())
            messages.extend(tool_results)

        return {"warning": "max_steps_reached"}, self._journal_entries

    async def _execute_tool(self, tool_call) -> dict:
        """Execute a tool call, journal if it's a mutation."""
        tool = self.tools[tool_call.name]

        before_state = None
        if tool.is_mutation:
            before_state = await tool.get_before_state(tool_call.arguments)
            self._mutation_count += 1
            if self._mutation_count > self.config.max_mutations_per_stage:
                raise CircuitBreakerError(
                    f"Exceeded {self.config.max_mutations_per_stage} mutations"
                )

        result = await tool.function(**tool_call.arguments)

        if tool.is_mutation:
            self._journal_entries.append(JournalEntry(
                operation=tool_call.name,
                target_system=tool.target_system,
                before_state=before_state,
                after_state=result,
                reasoning=tool_call.reasoning or "",
                evidence=[],
                ...
            ))

        return {"tool_call_id": tool_call.id, "content": json.dumps(result)}

    async def _call_llm(self, messages) -> LLMResponse:
        """Call configured LLM provider with tool definitions."""
        # Dispatch to OpenAI or Anthropic based on config.agent_llm_provider
        # Convert self.tools to the provider's tool schema format
        # Track self.llm_call_count and self.token_count
```

ToolDefinition wraps each tool the agent can call:

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict          # JSON Schema
    function: Callable        # Async callable
    is_mutation: bool         # Whether this modifies state (for journaling)
    target_system: str        # "graphiti" | "mem0" | "taskmaster"
    get_before_state: Callable | None = None  # For journaling mutations
```

#### 3b. `reconciliation/stages/base.py` — Base stage

```python
class BaseStage:
    """Base class for reconciliation pipeline stages."""

    def __init__(
        self,
        stage_id: StageId,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None,
        journal: ReconciliationJournal,
        config: ReconciliationConfig,
    ):
        self.stage_id = stage_id
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config
        self.project_id: str = ""  # Set before each run

    def get_tools(self) -> dict[str, ToolDefinition]:
        """Override in subclass — return tools available to this stage."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        """Override in subclass — build structured initial context."""
        raise NotImplementedError

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
    ) -> StageReport:
        """Execute this stage via agent loop."""
        payload = await self.assemble_payload(events, watermark, prior_reports)
        tools = self.get_tools()

        # Add terminal tool
        tools["stage_complete"] = ToolDefinition(
            name="stage_complete",
            description="Signal that this stage is complete. Provide a structured report.",
            parameters={"type": "object", "properties": {"report": {"type": "object"}}},
            function=lambda **kw: kw,
            is_mutation=False,
            target_system="reconciliation",
        )

        agent = AgentLoop(
            config=self.config,
            system_prompt=self.get_system_prompt(),
            tools=tools,
            terminal_tool="stage_complete",
        )

        started = datetime.utcnow()
        result, journal_entries = await agent.run(payload)
        completed = datetime.utcnow()

        for entry in journal_entries:
            entry.run_id = run_id
            entry.stage = self.stage_id
            await self.journal.add_entry(entry)

        return StageReport(
            stage=self.stage_id,
            started_at=started,
            completed_at=completed,
            actions_taken=[e.model_dump() for e in journal_entries],
            items_flagged=result.get("report", {}).get("flagged_items", []),
            stats=result.get("report", {}).get("stats", {}),
            llm_calls=agent.llm_call_count,
            tokens_used=agent.token_count,
        )
```

#### 3c. `reconciliation/stages/memory_consolidator.py` — Stage 1

Tools available:
- `search(query, project_id, ...)` — read both stores
- `add_memory(content, category, project_id, ...)` — write classified memories
- `delete_memory(memory_id, store, project_id)` — remove duplicates/stale
- `get_episodes(project_id, last_n)` — review recent history
- `get_entity(name, project_id)` — graph lookup
- `get_status(project_id)` — health check
- `verify_against_codebase(claim, context, scope_hints)` — ground truth check
- `stage_complete(report)` — terminal

All wired to `MemoryService` methods. Mutations (`add_memory`, `delete_memory`) are journaled.

Payload assembly:
```python
async def assemble_payload(self, events, watermark, prior_reports):
    # 1. Episodes since last reconciliation
    episodes = await self.memory.get_episodes(
        project_id=self.project_id,
        last_n=100,
    )
    new_episodes = [e for e in episodes if e["created_at"] > watermark.last_episode_timestamp]

    # 2. Mem0 memories added/modified since last reconciliation
    all_memories = await self.memory.mem0.get_all(scope, limit=500)
    # Filter by timestamp > watermark.last_memory_timestamp

    # 3. Store stats
    status = await self.memory.get_status(project_id=self.project_id)

    # 4. Last reconciliation summary
    last_report = prior_reports[-1] if prior_reports else None

    # 5. Format as structured text
    return f"""## Reconciliation Run
## Project: {self.project_id}

### New Episodes Since Last Reconciliation ({len(new_episodes)})
{format_episodes(new_episodes)}

### Recent Mem0 Memories ({len(new_memories)})
{format_memories(new_memories)}

### Store Status
{format_status(status)}

### Previous Reconciliation Summary
{format_report(last_report) if last_report else "First run."}

## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. If you encounter conflicting factual claims about the codebase, use verify_against_codebase.
5. Flag any items that are relevant to task planning for Stage 2.
6. Call stage_complete with your report when done.
"""
```

System prompt (`prompts/stage1.py`): Describes the agent's role, the six memory categories, the dual-store architecture, consolidation operations it should perform, and the authority model. Emphasizes: resolve contradictions, deduplicate, promote patterns, and flag task-relevant findings.

#### 3d. `reconciliation/stages/task_knowledge_sync.py` — Stage 2

Tools available:
- `search(...)` — read memory stores
- `add_memory(...)` — write new knowledge
- `delete_memory(...)` — remove stale knowledge
- `get_episodes(...)` — review history
- `get_entity(...)` — graph lookup
- `get_tasks(project_root)` — read task tree
- `get_task(id, project_root)` — read single task
- `set_task_status(id, status, project_root)` — modify task status
- `add_task(prompt, project_root, ...)` — create task
- `update_task(id, prompt, project_root, ...)` — modify task
- `add_subtask(parent_id, title, ...)` — create subtask
- `remove_task(id, project_root)` — delete task
- `add_dependency(id, depends_on, project_root)` — add dependency
- `remove_dependency(id, depends_on, project_root)` — remove dependency
- `attach_memory_hints(task_id, hints)` — attach memory hints to task (custom tool)
- `verify_against_codebase(claim, context, scope_hints)` — ground truth
- `stage_complete(report)` — terminal

Note on `attach_memory_hints`: Taskmaster supports arbitrary metadata on tasks (via `update_task` with `metadata` param, requires `TASK_MASTER_ALLOW_METADATA_UPDATES=true`). The tool is a convenience wrapper:

```python
async def attach_memory_hints(task_id: str, hints: MemoryHints, project_root: str) -> dict:
    """Attach memory retrieval hints to a task's metadata."""
    return await self.taskmaster.update_task(
        task_id=task_id,
        metadata=json.dumps({"memory_hints": hints.model_dump()}),
        project_root=project_root,
    )
```

Payload assembly:
```python
async def assemble_payload(self, events, watermark, prior_reports):
    stage1_report = prior_reports[0]

    # 1. Tasks changed since last reconciliation
    all_tasks = await self.taskmaster.get_tasks(project_root=self.project_root)
    # Filter to recently changed tasks (by updatedAt > watermark)

    # 2. Active/in-progress task tree summary
    active_tasks = [t for t in all_tasks if t["status"] in ("pending", "in-progress", "review")]

    # 3. Stage 1 findings relevant to tasks
    task_relevant_flags = stage1_report.items_flagged

    return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

### Stage 1 Report Summary
{format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{format_flagged(task_relevant_flags)}

### Tasks Changed Since Last Reconciliation
{format_tasks(changed_tasks)}

### Active Task Tree
{format_task_tree(active_tasks)}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If not, use verify_against_codebase
   to check repo state, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or
   delete tasks. Update dependent tasks.
3. For AI-generated tasks (from expand_task/parse_prd): cross-reference against knowledge
   graph for factual consistency. Flag/fix contradictions.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time.
   Use entity references + semantic queries, NOT inline content.
5. Check if any knowledge implies new tasks should be created or existing tasks unblocked.
6. Hints on completed tasks are static — don't update them.
7. Call stage_complete with your report when done.
"""
```

#### 3e. `reconciliation/stages/integrity_check.py` — Stage 3

Tools available (read-only + verify):
- `search(...)` — read memory stores
- `get_entity(...)` — graph lookup
- `get_episodes(...)` — review history
- `get_tasks(project_root)` — read task tree
- `get_task(id, project_root)` — read single task
- `verify_against_codebase(claim, context, scope_hints)` — ground truth
- `stage_complete(report)` — terminal

No mutation tools. Stage 3 detects and flags; it does not fix.

Payload assembly:
```python
async def assemble_payload(self, events, watermark, prior_reports):
    stage1_report, stage2_report = prior_reports[0], prior_reports[1]
    flagged = stage1_report.items_flagged + stage2_report.items_flagged

    return f"""## Stage 3: Cross-System Integrity Check
## Project: {self.project_id}

### Stage 1 Report
{format_report(stage1_report)}

### Stage 2 Report
{format_report(stage2_report)}

### Items Flagged for Cross-System Verification
{format_flagged(flagged)}

## Your Task
Verify consistency across all three systems:
1. Spot-check: do recently modified tasks align with current memory state?
2. Spot-check: do recently written memories align with task state?
3. For flagged items: investigate and classify as consistent/inconsistent.
4. Use verify_against_codebase for any factual disputes.
5. Report all findings. Inconsistencies found here will be addressed in the next cycle's
   Stage 1 and Stage 2.
6. Call stage_complete with your report.
"""
```

#### 3f. `reconciliation/harness.py` — Pipeline orchestrator

```python
class ReconciliationHarness:
    """Orchestrates the three-stage reconciliation pipeline."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None,
        journal: ReconciliationJournal,
        event_buffer: EventBuffer,
        config: FusedMemoryConfig,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.buffer = event_buffer
        self.config = config.reconciliation
        self.judge = Judge(config.reconciliation, journal) if config.reconciliation.judge_enabled else None

        self.stages = [
            MemoryConsolidator(StageId.memory_consolidator, memory_service, taskmaster, journal, config.reconciliation),
            TaskKnowledgeSync(StageId.task_knowledge_sync, memory_service, taskmaster, journal, config.reconciliation),
            IntegrityCheck(StageId.integrity_check, memory_service, taskmaster, journal, config.reconciliation),
        ]

    async def run_loop(self) -> None:
        """Background loop — check trigger conditions, run pipeline when needed."""
        while True:
            for project_id in await self._get_active_projects():
                should, reason = await self.buffer.should_trigger(project_id)
                if should:
                    acquired = await self.buffer.mark_run_active(project_id)
                    if acquired:
                        try:
                            await self.run_full_cycle(project_id, reason)
                        finally:
                            await self.buffer.mark_run_complete(project_id)
            await asyncio.sleep(5)  # Check interval

    async def run_full_cycle(self, project_id: str, trigger_reason: str) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project."""
        run_id = str(uuid4())
        watermark = await self.journal.get_watermark(project_id)
        events = await self.buffer.drain(project_id)

        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type="full",
            trigger_reason=trigger_reason,
            started_at=datetime.utcnow(),
            events_processed=len(events),
            stage_reports={},
            status="running",
        )
        await self.journal.start_run(run)

        try:
            reports = []
            for stage in self.stages:
                stage.project_id = project_id
                report = await stage.run(events, watermark, reports, run_id)
                reports.append(report)
                run.stage_reports[stage.stage_id] = report

            # Update watermark
            watermark.last_full_run_id = run_id
            watermark.last_full_run_completed = datetime.utcnow()
            await self.journal.update_watermark(watermark)

            run.completed_at = datetime.utcnow()
            run.status = "completed"
            await self.journal.complete_run(run_id, "completed")

            # Async judge review
            if self.judge:
                asyncio.create_task(self.judge.review_run(run_id))

            return run

        except CircuitBreakerError as e:
            run.status = "circuit_breaker"
            await self.journal.complete_run(run_id, "circuit_breaker")
            logger.error(f"Reconciliation circuit breaker: {e}")
            raise
        except Exception as e:
            run.status = "failed"
            await self.journal.complete_run(run_id, "failed")
            logger.error(f"Reconciliation failed: {e}")
            raise

    async def _get_active_projects(self) -> list[str]:
        """Return project IDs that have buffered events."""
        return list(self.buffer._buffer.keys())
```

---

### Phase 4: `verify_against_codebase`

#### 4a. `reconciliation/verify.py`

```python
class CodebaseVerifier:
    """Spawns an isolated explore agent to verify factual claims against the codebase."""

    def __init__(self, config: ReconciliationConfig):
        self.codebase_root = config.explore_codebase_root
        self.config = config

    async def verify(
        self,
        claim: str,
        context: str,
        scope_hints: list[str] | None = None,
        project_id: str = "",
    ) -> VerificationResult:
        """Verify a factual claim against the codebase.

        The explore agent has:
        - Read-only file access (read, glob, grep)
        - Git log/blame/diff access
        - NO memory system access
        - NO task system access
        """
        prompt = self._build_explore_prompt(claim, context, scope_hints)

        tools = {
            "read_file": ToolDefinition(...),       # Read file contents
            "glob_search": ToolDefinition(...),     # File pattern search
            "grep_search": ToolDefinition(...),     # Content search
            "git_log": ToolDefinition(...),         # Git history
            "git_show": ToolDefinition(...),        # Show specific commit
            "verification_complete": ToolDefinition(  # Terminal tool
                name="verification_complete",
                parameters=VerificationResult.model_json_schema(),
                ...
            ),
        }

        agent = AgentLoop(
            config=self.config,
            system_prompt=EXPLORE_AGENT_SYSTEM_PROMPT,
            tools=tools,
            terminal_tool="verification_complete",
        )

        result, _ = await agent.run(prompt)
        return VerificationResult(**result)
```

Explore agent system prompt: Emphasizes neutrality (report what the code says, don't speculate), evidence requirements (every claim must cite a file path and snippet), and scope (read-only, no memory/task access, unbiased factual verification).

Tool implementations: Thin wrappers around `pathlib`, `subprocess` (for git), and file I/O, all scoped to `codebase_root`. Git commands limited to read-only operations (`log`, `show`, `diff`, `blame`).

---

### Phase 5: Targeted Reconciliation

#### 5a. `reconciliation/targeted.py`

```python
class TargetedReconciler:
    """Lightweight reconciliation triggered by task state transitions."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend,
        journal: ReconciliationJournal,
        config: FusedMemoryConfig,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config.reconciliation
        self.verifier = CodebaseVerifier(config.reconciliation)

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
    ) -> dict:
        """Run targeted reconciliation for a single task state transition."""
        run_id = str(uuid4())
        run = ReconciliationRun(
            id=run_id, project_id=project_id, run_type="targeted",
            trigger_reason=f"task_{transition}:{task_id}",
            ...
        )
        await self.journal.start_run(run)

        try:
            handler = {
                "done": self._on_task_done,
                "blocked": self._on_task_blocked,
                "cancelled": self._on_task_cancelled,
                "deferred": self._on_task_deferred,
            }.get(transition)

            result = await handler(task_id, project_id, task_before, run_id)
            await self.journal.complete_run(run_id, "completed")
            return result
        except Exception as e:
            await self.journal.complete_run(run_id, "failed")
            return {"error": str(e)}

    async def _on_task_done(self, task_id, project_id, task_before, run_id) -> dict:
        """Task completed. Verify knowledge capture, update dependents."""
        task = task_before
        result = {"task_id": task_id, "actions": []}

        # 1. Search for knowledge related to this task
        related = await self.memory.search(
            query=f"{task['title']} {task.get('description', '')}",
            project_id=project_id, limit=5,
        )

        # 2. If sparse knowledge, verify against codebase
        if len(related) < 2:
            verification = await self.verifier.verify(
                claim=f"Task '{task['title']}' has been completed",
                context=f"Task details: {task.get('details', '')}",
                scope_hints=self._extract_scope_hints(task),
            )
            if verification.verdict in ("confirmed", "contradicted"):
                await self.memory.add_memory(
                    content=f"Completed task '{task['title']}': {verification.summary}",
                    category="observations_and_summaries",
                    project_id=project_id,
                    metadata={
                        "source": "targeted_reconciliation",
                        "task_id": task_id,
                        "verification_verdict": verification.verdict,
                    },
                )
                result["actions"].append({
                    "type": "knowledge_captured",
                    "verification": verification.verdict,
                })

        # 3. Check dependent tasks — are they unblocked?
        all_tasks = await self.taskmaster.get_tasks(project_root=project_id)
        for t in all_tasks.get("tasks", []):
            if task_id in [str(d) for d in t.get("dependencies", [])]:
                all_deps_done = all(
                    any(dt["id"] == str(dep_id) and dt["status"] == "done"
                        for dt in all_tasks["tasks"])
                    for dep_id in t.get("dependencies", [])
                )
                if all_deps_done and t["status"] == "pending":
                    result["actions"].append({
                        "type": "dependent_unblocked",
                        "task_id": t["id"],
                    })

        return result

    async def _on_task_blocked(self, task_id, project_id, task_before, run_id) -> dict:
        """Task blocked. Search for relevant knowledge to attach as hints."""
        task = task_before
        result = {"task_id": task_id, "actions": []}

        related = await self.memory.search(
            query=f"blockers for: {task['title']} {task.get('description', '')}",
            project_id=project_id, limit=5,
        )

        if related:
            entities = []
            for r in related:
                entities.extend(r.entities)
            hints = MemoryHints(
                entities=list(set(entities)),
                queries=[f"resolution for: {task['title']}"],
            )
            await self.taskmaster.update_task(
                task_id=task_id,
                metadata=json.dumps({"memory_hints": hints.model_dump()}),
                project_root=project_id,
            )
            result["actions"].append({"type": "hints_attached", "hints": hints.model_dump()})

        return result

    async def _on_task_cancelled(self, task_id, project_id, task_before, run_id) -> dict:
        """Task cancelled. Flag subtasks and dependents for review."""
        # Check for subtasks/dependents that may need status changes
        ...

    async def _on_task_deferred(self, task_id, project_id, task_before, run_id) -> dict:
        """Task deferred. Similar to blocked — attach relevant knowledge hints."""
        # Similar to _on_task_blocked
        ...

    async def reconcile_bulk_tasks(self, parent_task_id, project_id) -> dict:
        """Reconcile after expand_task or parse_prd — cross-reference against knowledge."""
        # 1. Get newly created subtasks/tasks
        # 2. For each, search knowledge graph for contradictions
        # 3. Flag tasks whose assumptions contradict known facts
        # 4. Attach memory hints where relevant knowledge exists
        ...
```

---

### Phase 6: LLM-as-Judge

#### 6a. `reconciliation/judge.py`

```python
class Judge:
    """Async LLM reviewer that evaluates reconciliation run quality."""

    def __init__(self, config: ReconciliationConfig, journal: ReconciliationJournal):
        self.config = config
        self.journal = journal

    async def review_run(self, run_id: str) -> JudgeVerdict:
        """Review a completed reconciliation run asynchronously."""
        run = await self.journal.get_run(run_id)
        entries = await self.journal.get_entries(run_id)
        recent_verdicts = await self.journal.get_recent_verdicts(run.project_id, limit=10)

        prompt = self._build_review_prompt(run, entries, recent_verdicts)

        # Single LLM call (not an agent loop — judge doesn't need tools)
        response = await self._call_llm(JUDGE_SYSTEM_PROMPT, prompt)
        verdict = self._parse_verdict(response, run_id)

        await self.journal.add_verdict(verdict)

        # Act on verdict
        if verdict.severity == "moderate":
            await self._trigger_rollback_and_redo(run_id)
        elif verdict.severity == "serious":
            await self._halt_system(run.project_id, verdict)

        # Check for trending
        await self._check_error_trends(run.project_id, recent_verdicts + [verdict])

        return verdict

    async def _check_error_trends(self, project_id, verdicts):
        """Detect rising error rates across recent runs."""
        recent_non_ok = [v for v in verdicts[-10:] if v.severity != "ok"]
        if len(recent_non_ok) >= 5:
            await self._halt_system(project_id, reason="error_trend")
```

Judge system prompt (`prompts/judge.py`): Defines evaluation criteria (factual grounding, proportionality, consistency, harm potential), severity thresholds, and output format.

---

### Phase 7: Dashboard / Observability

Not a separate module — integrated into all components via two mechanisms:

#### 7a. Structured logging

Every reconciliation component logs structured JSON to stderr:

```python
logger.info("reconciliation.event_buffered", extra={
    "project_id": ..., "event_type": ..., "buffer_size": ..., "oldest_event_age_seconds": ...,
})
logger.info("reconciliation.run_started", extra={
    "run_id": ..., "project_id": ..., "run_type": ..., "trigger_reason": ..., "events_to_process": ...,
})
logger.info("reconciliation.stage_completed", extra={
    "run_id": ..., "stage": ..., "duration_seconds": ..., "actions_taken": ..., "llm_calls": ..., "tokens_used": ...,
})
logger.info("reconciliation.targeted_completed", extra={
    "task_id": ..., "transition": ..., "duration_seconds": ..., "agent_blocked_seconds": ...,
})
logger.info("reconciliation.judge_verdict", extra={
    "run_id": ..., "severity": ..., "findings_count": ..., "action_taken": ...,
})
```

#### 7b. `get_status` extension

Extend the existing MCP tool to include reconciliation stats:

```python
status["reconciliation"] = {
    "last_full_run": ...,
    "runs_24h": ...,
    "avg_duration_seconds": ...,
    "buffer_size": ...,
    "judge_verdicts_24h": ...,
    "stale_writes_24h": ...,
    "targeted_avg_latency_seconds": ...,
}
```

---

### Phase 8: Testing Strategy

#### Unit tests (mocked backends)

| Test file | What it covers |
|---|---|
| `test_event_buffer.py` | Push, trigger conditions (size, staleness, active run skip), drain atomicity, concurrent access |
| `test_journal.py` | SQLite CRUD, watermark management, stats queries |
| `test_agent_loop.py` | Tool dispatch, terminal tool detection, max steps, circuit breaker, journaling of mutations |
| `test_taskmaster_client.py` | MCP client calls, error handling, response parsing |
| `test_task_interceptor.py` | Status transition detection, event emission, targeted reconciliation invocation, pass-through for reads |
| `test_targeted.py` | Each transition handler (done/blocked/cancelled/deferred), bulk task reconciliation, hint attachment |
| `test_harness.py` | Pipeline sequencing, watermark advancement, error handling, judge invocation |

#### Integration tests (real backends, if available)

- End-to-end: add episodes → buffer fills → pipeline runs → memories consolidated → tasks updated
- Targeted: set task status → reconciliation fires → knowledge written → return with enrichment
- Verify: create contradicting memories → reconciliation resolves via codebase check

#### Prompt testing

- Stage prompts tested with representative payloads to verify agent behavior
- Judge prompt tested with known-good and known-bad journal entries

---

## Dependency Graph and Build Order

```
Phase 1: Foundation
  models/reconciliation.py ──────────────────────┐
  reconciliation/journal.py ─────────────────────┤
  reconciliation/event_buffer.py ────────────────┤
  config/schema.py (additions) ──────────────────┘
                                                  │
Phase 2: Taskmaster Integration                   │
  backends/taskmaster_client.py ─────────────────┤
  middleware/task_interceptor.py ─────────────────┤
  server/tools.py (additions) ───────────────────┤
  server/main.py (wiring) ──────────────────────┤
  services/memory_service.py (event emission) ───┘
                                                  │
Phase 3: Agent Loop + Stages                      │
  reconciliation/agent_loop.py ──────────────────┤
  reconciliation/stages/base.py ─────────────────┤
  reconciliation/prompts/*.py ───────────────────┤
  reconciliation/stages/memory_consolidator.py ──┤
  reconciliation/stages/task_knowledge_sync.py ──┤
  reconciliation/stages/integrity_check.py ──────┤
  reconciliation/harness.py ─────────────────────┘
                                                  │
Phase 4: Verify                                   │
  reconciliation/verify.py ──────────────────────┘
                                                  │
Phase 5: Targeted Reconciliation                  │
  reconciliation/targeted.py ────────────────────┘
                                                  │
Phase 6: Judge                                    │
  reconciliation/judge.py ───────────────────────┘
                                                  │
Phase 7: Observability                            │
  (Integrated into all above modules)            │
                                                  │
Phase 8: Tests                                    │
  (Parallel with each phase)
```

Phases 1 and 2 are the critical path — everything else depends on them.

---

## New Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "anthropic>=0.52.0",      # For Anthropic agent LLM calls
    "aiosqlite>=0.20.0",      # Async SQLite for journal
]
```

The `mcp` library (already `>=1.9.4`) includes the MCP client needed for Taskmaster connection.

---

## Key Risk Areas

1. **Taskmaster MCP client stability**: Taskmaster is Node.js, connected via stdio. Process management (startup, crash recovery, reconnection) needs to be robust. Recommend a health check ping on a timer and automatic restart on failure.

2. **Agent loop token consumption**: Three stages x agent loop with potentially many tool calls = significant LLM spend per reconciliation run. The circuit breaker (`max_mutations_per_stage`) and step limit (`agent_max_steps`) are the primary controls.

3. **Prompt engineering iteration**: The stage prompts determine whether the agent makes good consolidation/sync decisions. Plan for prompt refinement based on judge feedback and manual review of early runs.

4. **Taskmaster metadata for memory hints**: Requires `TASK_MASTER_ALLOW_METADATA_UPDATES=true` environment variable.

---

## Key Existing Code References

- MCP server entry: `fused-memory/src/fused_memory/server/main.py`
- MCP tools: `fused-memory/src/fused_memory/server/tools.py`
- Memory service: `fused-memory/src/fused_memory/services/memory_service.py`
- Queue service: `fused-memory/src/fused_memory/services/queue_service.py`
- Graphiti backend: `fused-memory/src/fused_memory/backends/graphiti_client.py`
- Mem0 backend: `fused-memory/src/fused_memory/backends/mem0_client.py`
- Models: `fused-memory/src/fused_memory/models/` (enums.py, memory.py, scope.py)
- Routing: `fused-memory/src/fused_memory/routing/` (classifier.py, router.py)
- Config: `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/config/config.yaml`
- Existing tests: `fused-memory/tests/`
- Taskmaster MCP tools: `taskmaster-ai/mcp-server/src/tools/`
- Taskmaster direct functions: `taskmaster-ai/mcp-server/src/core/direct-functions/`
- Taskmaster task status: `taskmaster-ai/src/constants/task-status.js`
- Design doc: `DESIGN.md`
