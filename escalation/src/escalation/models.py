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

Structured-evidence field (default-empty; task 2558):
  evidence:   list of EvidenceEntry {observation, measured_at, ref} raw
              OBSERVATIONS backing the escalation (not causal diagnoses)

Filing-identity field (default-None; task 3533):
  filing_claimant_run_id:
              the FILING incarnation's claimant id in
              `shared.task_claimant.compose_claimant_run_id` format;
              None = unknown.  Semantics and the fail-safe rule are stated
              once on `escalation.pins.classify_pins` (normative source:
              spec docs/task-escalation-state-spec.md S6) — do not restate
              them here.
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


class EvidenceEntry(TypedDict):
    """A single structured raw-OBSERVATION backing an escalation (task 2558).

    Each entry records what was measured, when, and against which ref — NOT a
    causal diagnosis.  Filers put observations here (and in summary/detail) and
    keep any causal claim on a clearly-marked `Hypothesis:` line, so a reviewer
    never reads an unverified guess as fact (survey §1.7 precedent: a
    "last-green" rewind recommendation named a commit that ALSO failed).

    Shape mirrors the TrainState TypedDict.  TypedDict at runtime is a plain
    dict; existing from_dict / to_dict / asdict() paths are unaffected —
    round-trip fidelity is unchanged.  The server stores and returns it
    verbatim (no shape validation), so partial evidence is accepted.
    """

    observation: str   # what was measured, e.g. "main red at abc123 (exit 134)"
    measured_at: str   # when/where measured, e.g. "HEAD=abc123 @ 2026-07-14T00:00:00+00:00"
    ref: str           # ref/context this was measured against, e.g. "rerun#2" or "HEAD=abc123"


# Severities that cause an escalation to be created directly at L2,
# bypassing the auto-watcher and routing straight to a human.
BORN_AT_L2_SEVERITIES: frozenset[str] = frozenset({'critical', 'urgent'})

# All valid severity values accepted by the MCP tools (escalate_blocker /
# escalate_info).  Used to validate caller input and return a clear error
# rather than silently misrouting escalations.
KNOWN_SEVERITIES: frozenset[str] = frozenset({'info', 'blocking'}) | BORN_AT_L2_SEVERITIES

# Legal values for Escalation.resolution_class (escalation-lifecycle-dashboard-prd.md
# Seam 1).  Used to validate the resolve/dismiss chokepoint's optional
# resolution_class param and return a clear error naming the legal
# values rather than silently accepting an arbitrary string.
#
# 'moot-terminal-subject' (task 2724) is the DISTINCT, non-benign stamp the
# escalation-revalidation sweep writes when it auto-closes an allowlisted L2
# whose subject task went terminal (done/cancelled).  It is deliberately
# neither 'benign' nor 'actionable' so swept records stay auditable — the
# dashboard's effective_benign/_origin_block/_workflow_block count it as
# classified+stamped but bucket it into neither, preserving the evidence the
# old 'benign' mis-label destroyed.
RESOLUTION_CLASSES: frozenset[str] = frozenset({'benign', 'actionable', 'moot-terminal-subject'})


@dataclass
class Escalation:
    id: str  # "esc-{task_id}-{seq}"
    task_id: str
    agent_role: str
    severity: str  # "blocking" | "info" | "critical" | "urgent"
    category: str  # scope_violation, design_concern, cleanup_needed,
    # dependency_discovered, risk_identified, infra_issue,
    # reconciliation_stale_human_operator, reconciliation_stale_gate_backlog
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
    # Structured raw-OBSERVATION entries backing this escalation (task 2558).
    # Each is a {observation, measured_at, ref} dict recording a measurement
    # (HEAD SHA, rerun result, raw exit code) — never a causal diagnosis.
    # Empty default keeps existing free-form escalations bit-identical on disk;
    # legacy JSON without the key deserialises to [] via the from_dict
    # __dataclass_fields__ filter — zero migration, same pattern as members /
    # train_state above.  Stored/returned verbatim (no shape validation).
    evidence: list[EvidenceEntry] = field(default_factory=list)
    # PRD § 9.8 — per-train context for park-prefix derail L2 escalations.
    # None for all non-train escalations; legacy JSON (pre-field) deserialises to None
    # via the from_dict __dataclass_fields__ filter — no migration required.
    train_state: TrainState | None = None
    # C1 action chosen at resolve_issue time (resume/restart/park/abandon/close_only).
    # None for records resolved before α1 or for L2 cascade members (β derives theirs
    # from the parent via resolved_by='l2-cascade:<id>' attribution).
    resolution_action: str | None = None
    # Benign/actionable classification stamp (escalation-lifecycle-dashboard-prd.md
    # Seam 1).  Written ONLY at a terminal-write chokepoint — queue.resolve() or
    # queue.submit_resolved() — never author-supplied at filing time.  None means
    # unstamped — readers fall back to the effective_benign() proxy (see
    # escalation.classify).
    resolution_class: str | None = None
    # Triage-ack annotation (NOT a resolution) — lets escalation-watcher-auto
    # rotations skip re-deriving the disposition of a still-pending L1/L2 item
    # every rotation. triaged_at/triaged_by/triage_note are stamped by
    # queue.stamp_triage(); updated_at is a separate "changed since I triaged
    # it" signal bumped elsewhere (e.g. add_members_to_l2 on real append).
    # All four are optional with zero-migration defaults: legacy JSON files
    # without these keys deserialise correctly via the from_dict
    # __dataclass_fields__ filter below — the same pattern used for
    # train_state/resolution_action/resolution_class above.
    triaged_at: str | None = None
    triaged_by: str | None = None  # server-attributed from X-Escalation-Identity when present
    triage_note: str = ''  # freshness contract: verified predicate + probe (see watcher SKILLs)
    updated_at: str | None = None  # last-substantive-change marker; None means never bumped
    # Steward's structured scope-expansion grant (task 2505) — file-level,
    # project-relative paths — consumed by the orchestrator resume path to
    # widen plan.files/metadata.files/locks. Distinct from the free-text
    # `resolution` rationale string, which stays human-readable prose.
    # Empty-list default keeps non-grant escalations bit-identical on disk;
    # legacy JSON without this key deserialises to [] via the from_dict
    # __dataclass_fields__ filter below — no migration required.
    granted_files: list[str] = field(default_factory=list)
    # The FILING incarnation's claimant identity (task 3533) — semantics are
    # documented once on `escalation.pins.classify_pins` (see the module
    # docstring's field summary above). Zero migration, same pattern as
    # members / evidence / train_state / the triage quad / granted_files
    # above: legacy JSON without this key deserialises to None via the
    # from_dict __dataclass_fields__ filter below, to_dict's asdict()
    # serialises it automatically, and queue.submit / submit_resolved /
    # _atomic_write / resolve / park / stamp_triage need NO change (they are
    # field-agnostic passthroughs or RMW-on-hydrated-record).
    filing_claimant_run_id: str | None = None

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
