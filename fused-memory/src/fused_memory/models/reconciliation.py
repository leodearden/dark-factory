"""Models for the sleep mode reconciliation system."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Event types emitted by the system."""

    episode_added = 'episode_added'
    memory_added = 'memory_added'
    memory_deleted = 'memory_deleted'
    task_status_changed = 'task_status_changed'
    task_created = 'task_created'
    task_modified = 'task_modified'
    task_deleted = 'task_deleted'
    tasks_bulk_created = 'tasks_bulk_created'


class EventSource(StrEnum):
    """Origin of the event."""

    agent = 'agent'
    targeted_reconciliation = 'targeted_reconciliation'
    full_reconciliation = 'full_reconciliation'


class StageId(StrEnum):
    """Pipeline stage identifiers."""

    memory_consolidator = 'memory_consolidator'
    task_knowledge_sync = 'task_knowledge_sync'
    integrity_check = 'integrity_check'


class ReconciliationEvent(BaseModel):
    """Individual event in the buffer."""

    id: str
    type: EventType
    source: EventSource
    project_id: str
    timestamp: datetime
    payload: dict = Field(default_factory=dict)
    agent_id: str | None = None


class JournalEntry(BaseModel):
    """One per reconciliation operation — full audit trail."""

    id: str
    run_id: str = ''
    stage: StageId | None = None
    timestamp: datetime
    operation: str
    target_system: str  # 'graphiti', 'mem0', 'taskmaster'
    before_state: dict | None = None
    after_state: dict | None = None
    reasoning: str = ''
    evidence: list[dict] = Field(default_factory=list)


class StageReport(BaseModel):
    """Output of each pipeline stage."""

    stage: StageId
    started_at: datetime
    completed_at: datetime
    items_flagged: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    llm_calls: int = 0
    tokens_used: int = 0


class RunType(StrEnum):
    """Type of reconciliation run."""

    full = 'full'
    targeted = 'targeted'
    remediation = 'remediation'


class RunStatus(StrEnum):
    """Status of a reconciliation run."""

    running = 'running'
    completed = 'completed'
    failed = 'failed'
    rolled_back = 'rolled_back'
    circuit_breaker = 'circuit_breaker'


class ReconciliationRun(BaseModel):
    """Metadata for a reconciliation run."""

    id: str
    project_id: str
    run_type: RunType
    trigger_reason: str
    started_at: datetime
    completed_at: datetime | None = None
    events_processed: int = 0
    stage_reports: dict[str, StageReport | dict] = Field(default_factory=dict)
    status: RunStatus = RunStatus.running
    triggered_by: str | None = None  # parent run_id for remediation runs


class MemoryHints(BaseModel):
    """Memory retrieval hints attached to tasks."""

    entities: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class VerificationVerdict(StrEnum):
    """Verdict from the codebase verification agent."""

    confirmed = 'confirmed'
    contradicted = 'contradicted'
    inconclusive = 'inconclusive'


class VerificationResult(BaseModel):
    """Result from the explore agent's codebase verification."""

    verdict: VerificationVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[dict] = Field(default_factory=list)
    summary: str = ''
    git_context: dict | None = None


class Watermark(BaseModel):
    """Tracks what's been processed per project."""

    project_id: str
    last_full_run_id: str | None = None
    last_full_run_completed: datetime | None = None
    last_episode_timestamp: datetime | None = None
    last_memory_timestamp: datetime | None = None
    last_task_change_timestamp: datetime | None = None


class VerdictSeverity(StrEnum):
    """Severity level of a judge verdict."""

    ok = 'ok'
    minor = 'minor'
    moderate = 'moderate'
    serious = 'serious'


class VerdictAction(StrEnum):
    """Action taken based on a judge verdict."""

    none = 'none'
    auto_fix = 'auto_fix'
    rollback = 'rollback'
    halt = 'halt'


class JudgeVerdict(BaseModel):
    """LLM-as-judge verdict for a reconciliation run."""

    run_id: str
    reviewed_at: datetime
    severity: VerdictSeverity
    findings: list[dict] = Field(default_factory=list)
    action_taken: VerdictAction = VerdictAction.none
