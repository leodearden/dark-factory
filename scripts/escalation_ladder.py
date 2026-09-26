#!/usr/bin/env python3
"""Read the escalation JSON store and measure how escalations leave the ladder.

STRICTLY READ-ONLY: it opens escalation records for reading and writes,
files and emits nothing, so it is safe against the live store.

Knows only the escalation store. It does not read runs.db; a caller that
wants "how did the escalations these runs worked on end?" joins the two
itself (scripts/review_model_admission.py does).
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

# data/ is gitignored, so a task worktree has no store of its own: the live
# store exists only in the main checkout (as audit_model_admission.DEFAULT_RUNS_DB).
DEFAULT_ESCALATIONS_DIR = Path('/home/leo/src/dark-factory/data/escalations')

# escalation/src/escalation/archive.py::ARCHIVE_SUBDIR. Restated, not imported:
# scripts/ imports no first-party package (the comment on
# dark-factory-orchestrator.yaml::test_command).
ARCHIVE_SUBDIR = 'archive'


def _parse_instant(raw: Any) -> datetime | None:
    """Parse an ISO-8601 string to an aware UTC datetime (naive reads as UTC)."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class EscalationRecord:
    """The fields of one escalation record that the ladder measurements read."""

    id: str
    task_id: str | None
    level: int
    agent_role: str
    status: str
    timestamp: datetime
    resolved_at: datetime | None
    resolved_by: str | None
    resolution_action: str | None

    @classmethod
    def from_object(cls, obj: Any) -> EscalationRecord | None:
        """Build a record from a parsed JSON value, or None when it is unusable.

        Unusable means not an object, or lacking one of the three fields every
        measurement keys on: a string id, an int level, a parseable timestamp.
        An unparseable resolved_at is not disqualifying; it reads as None.
        """
        if not isinstance(obj, dict):
            return None
        esc_id, level = obj.get('id'), obj.get('level')
        timestamp = _parse_instant(obj.get('timestamp'))
        if not isinstance(esc_id, str) or not esc_id or timestamp is None:
            return None
        if not isinstance(level, int) or isinstance(level, bool):
            return None
        task_id = obj.get('task_id')
        return cls(
            id=esc_id,
            task_id=None if task_id is None else str(task_id),
            level=level,
            agent_role=obj.get('agent_role') or '',
            status=obj.get('status') or '',
            timestamp=timestamp,
            resolved_at=_parse_instant(obj.get('resolved_at')),
            resolved_by=obj.get('resolved_by'),
            resolution_action=obj.get('resolution_action'),
        )


@dataclass(frozen=True)
class EscalationCorpus:
    """Every usable escalation record in one store, keyed by id.

    ``skipped`` counts the esc-*.json files that could not be used, so a
    dropped record stays visible. ``oldest_archive_date`` is the name of the
    oldest archive/<date>/ subdirectory: records resolved before it have been
    pruned, so it bounds how far back an escalation-derived measure reaches.
    """

    records: Mapping[str, EscalationRecord]
    skipped: int
    oldest_archive_date: str | None

    def get(self, esc_id: str) -> EscalationRecord | None:
        return self.records.get(esc_id)


def _escalation_paths(root: Path) -> Iterator[Path]:
    """Root esc-*.json first, then the archive subtree; the first copy of a stem wins.

    Mirrors escalation/src/escalation/queue.py::iter_all_escalation_paths
    (root wins on a collision, archive-only duplicates yield once), except that
    the archive is walked newest date first, so which of two archived copies
    is read does not depend on filesystem order.
    """
    archive = root / ARCHIVE_SUBDIR
    archived = sorted(archive.rglob('esc-*.json'), reverse=True) if archive.is_dir() else []
    seen: set[str] = set()
    for path in [*sorted(root.glob('esc-*.json')), *archived]:
        if path.stem not in seen:
            seen.add(path.stem)
            yield path


def _read_json(path: Path) -> Any:
    """The parsed content of *path*, or None when it is unreadable or not JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _oldest_archive_date(root: Path) -> str | None:
    archive = root / ARCHIVE_SUBDIR
    if not archive.is_dir():
        return None
    return min((entry.name for entry in archive.iterdir() if entry.is_dir()), default=None)


def load_escalation_corpus(escalations_dir: str | Path) -> EscalationCorpus:
    """Load every usable record under *escalations_dir* into a frozen corpus.

    A missing directory RAISES, where iter_all_escalation_paths yields nothing:
    an empty corpus would report every escalation a caller looks up as missing,
    which reads as a measurement rather than as a typo'd path.
    """
    root = Path(escalations_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'escalations directory not found: {root}')
    records: dict[str, EscalationRecord] = {}
    skipped = 0
    for path in _escalation_paths(root):
        record = EscalationRecord.from_object(_read_json(path))
        if record is None:
            skipped += 1
        else:
            records.setdefault(record.id, record)
    return EscalationCorpus(
        records=MappingProxyType(records),
        skipped=skipped,
        oldest_archive_date=_oldest_archive_date(root),
    )
