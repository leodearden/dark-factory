"""Decide which surviving ``agent_session.json`` sidecars the boot recovery scan reaps (plans/session-resume-eligibility-seam-prd.md D7)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from shared.proc_liveness import pid_alive
from shared.task_statuses import TaskStatus

from orchestrator.artifacts import AgentSession


class ReapVerdict(StrEnum):
    REAP = 'reap'
    KEEP_OWNER_ALIVE = 'keep_owner_alive'
    KEEP_IN_PROGRESS = 'keep_in_progress'
    KEEP_STATUS_UNKNOWN = 'keep_status_unknown'


def reap_verdict(*, owner_alive: bool, status: str | None) -> ReapVerdict:
    """D7: reap only when the owner is dead AND the task status is known and not in-progress."""
    if owner_alive:
        return ReapVerdict.KEEP_OWNER_ALIVE
    if status is None:
        return ReapVerdict.KEEP_STATUS_UNKNOWN
    if status == TaskStatus.IN_PROGRESS:
        return ReapVerdict.KEEP_IN_PROGRESS
    return ReapVerdict.REAP


@dataclass(frozen=True)
class SessionSidecarClaim:
    entry: Path
    sidecar_path: Path
    task_id: str
    owner_pid: object
    age_secs: float | None

    def event_data(self) -> dict:
        return {
            'sidecar_path': str(self.sidecar_path),
            'task_id': self.task_id,
            'owner_pid': self.owner_pid,
            'age_secs': self.age_secs,
        }


def _age_secs(started_at: object, now: datetime) -> float | None:
    if not isinstance(started_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (now - started).total_seconds()


def read_sidecar_claim(
    entry: Path,
    sidecar_path: Path,
    *,
    fallback_task_id: str | None,
    now: datetime,
) -> SessionSidecarClaim | None:
    """The sidecar's claim on *entry*, or None when it is unreadable or names no task. Never raises."""
    try:
        data = json.loads(sidecar_path.read_text())
        if not isinstance(data, dict):
            return None
        session = AgentSession.from_mapping(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    task_id = str(session.task_id) if session.task_id else fallback_task_id
    if task_id is None:
        return None
    return SessionSidecarClaim(
        entry=entry,
        sidecar_path=sidecar_path,
        task_id=task_id,
        owner_pid=session.owner_pid,
        age_secs=_age_secs(session.started_at, now),
    )


def owner_alive(claim: SessionSidecarClaim) -> bool:
    return pid_alive(claim.owner_pid)


@dataclass(frozen=True)
class ReapTally:
    """Per-boot counts; ``reaped`` counts only the REAP verdicts whose clear succeeded."""

    reaped: int
    clear_failed: int
    kept_owner_alive: int
    kept_in_progress: int
    kept_status_unknown: int
    unresolvable: int
    status_read_error: str | None

    @classmethod
    def from_verdicts(
        cls,
        verdicts: Iterable[ReapVerdict],
        *,
        clear_failed: int,
        unresolvable: int,
        status_read_error: str | None,
    ) -> ReapTally:
        counts = Counter(verdicts)
        if not 0 <= clear_failed <= counts[ReapVerdict.REAP]:
            raise ValueError(
                f'clear_failed={clear_failed} outside 0..{counts[ReapVerdict.REAP]} REAP verdicts'
            )
        return cls(
            reaped=counts[ReapVerdict.REAP] - clear_failed,
            clear_failed=clear_failed,
            kept_owner_alive=counts[ReapVerdict.KEEP_OWNER_ALIVE],
            kept_in_progress=counts[ReapVerdict.KEEP_IN_PROGRESS],
            kept_status_unknown=counts[ReapVerdict.KEEP_STATUS_UNKNOWN],
            unresolvable=unresolvable,
            status_read_error=status_read_error,
        )

    def event_data(self) -> dict:
        return {
            'reaped': self.reaped,
            'clear_failed': self.clear_failed,
            'kept_owner_alive': self.kept_owner_alive,
            'kept_in_progress': self.kept_in_progress,
            'kept_status_unknown': self.kept_status_unknown,
            'unresolvable': self.unresolvable,
            'status_read_error': self.status_read_error,
        }
