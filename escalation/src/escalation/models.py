"""Data model for escalations.

Consumer-per-level contract (invariant):
  L0  level=0  producer: agent           consumer: steward (interactive review)
  L1  level=1  producer: steward/workflow consumer: escalation-watcher-auto (automated triage)
  L2  level=2  producer: auto-watcher     consumer: human (direct, bypasses auto-watcher)

Escalations are born at L2 when their severity is in BORN_AT_L2_SEVERITIES.
All other escalations start at L0 (level=0) and are promoted by handlers.

L2 cluster fields (default-empty; L0/L1 are unaffected):
  members:    list of member L1 escalation ids forming this cluster
  root_cause: exact-string dedup key for pending-L2 lookup
  options:    proposed resolution options (e.g. ['A: rollback', 'B: fix forward'])
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TypedDict


class TrainState(TypedDict):
    """Per-train context embedded in L1/L2 escalations for park-prefix derail triage.

    Shape mirrors the dict returned by TaskWorkflow._build_train_state (PRD § 9.8).
    TypedDict at runtime is a plain dict; existing from_dict / to_dict / asdict()
    paths are unaffected — round-trip fidelity is unchanged.
    """

    id: str               # train identifier
    order: int            # this task's position in the train (0-based)
    parked_members: list[str]  # sibling task_ids at merge-deferred, excluding self
    failing_member: str   # this task's id (the one that triggered BLOCKED)


# Severities that cause an escalation to be created directly at L2,
# bypassing the auto-watcher and routing straight to a human.
BORN_AT_L2_SEVERITIES: frozenset[str] = frozenset({'critical', 'urgent'})

# All valid severity values accepted by the MCP tools (escalate_blocker /
# escalate_info).  Used to validate caller input and return a clear error
# rather than silently misrouting escalations.
KNOWN_SEVERITIES: frozenset[str] = frozenset({'info', 'blocking'}) | BORN_AT_L2_SEVERITIES


@dataclass
class Escalation:
    id: str  # "esc-{task_id}-{seq}"
    task_id: str
    agent_role: str
    severity: str  # "blocking" | "info" | "critical" | "urgent"
    category: str  # scope_violation, design_concern, cleanup_needed,
    # dependency_discovered, risk_identified, infra_issue,
    # reconciliation_stale_human_operator
    summary: str  # one-line
    detail: str = ''  # full context
    suggested_action: str = ''  # expand_scope, create_followup_task, abort_task, etc.
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = 'pending'  # pending, resolved, dismissed
    resolution: str | None = None  # filled by handler
    worktree: str | None = None  # path to worktree
    workflow_state: str | None = None  # what state the agent was in
    level: int = 0  # 0 = L0 agent→steward, 1 = L1 steward/workflow→escalation-watcher-auto, 2 = L2 escalation-watcher-auto→human
    resolved_at: str | None = None
    resolved_by: str | None = None  # "steward" | "interactive" | "auto-dismissed"
    resolution_turns: int | None = None  # conversation turns to resolve
    dedupe_count: int = 0  # number of duplicate submissions folded into this parent
    dedupe_children: list[str] = field(default_factory=list)  # ids of folded duplicates
    dedupe_fingerprint: str | None = None  # content fingerprint for A7a/A7b recon dedup
    # L2 cluster fields — empty defaults keep L0/L1 escalations bit-identical on disk.
    # Old JSON files (pre-L2) deserialise correctly via from_dict's __dataclass_fields__
    # filter: absent keys map to the dataclass defaults without any migration required.
    members: list[str] = field(default_factory=list)  # member L1 escalation ids (cluster composition)
    root_cause: str = ''  # root-cause hypothesis; exact-string dedup key for pending-L2 lookup
    options: list[str] = field(default_factory=list)  # proposed resolution options ['A: ...', 'B: ...']
    # PRD § 9.8 — per-train context for park-prefix derail L2 escalations.
    # None for all non-train escalations; legacy JSON (pre-field) deserialises to None
    # via the from_dict __dataclass_fields__ filter — no migration required.
    train_state: TrainState | None = None

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
