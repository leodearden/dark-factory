"""Data model for escalations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class Escalation:
    id: str  # "esc-{task_id}-{seq}"
    task_id: str
    agent_role: str
    severity: str  # "blocking" | "info"
    category: str  # scope_violation, design_concern, cleanup_needed,
    # dependency_discovered, risk_identified, infra_issue
    summary: str  # one-line
    detail: str = ''  # full context
    suggested_action: str = ''  # expand_scope, create_followup_task, abort_task, etc.
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = 'pending'  # pending, resolved, dismissed
    resolution: str | None = None  # filled by handler
    worktree: str | None = None  # path to worktree
    workflow_state: str | None = None  # what state the agent was in

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Escalation:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, text: str) -> Escalation:
        return cls.from_dict(json.loads(text))
