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
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
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


# The ladder's levels and the literals the disposition reads. The resolver
# spellings are restated from escalation/src/escalation/classify.py::
# classify_resolver_tier and orchestrator/src/orchestrator/steward.py, neither
# importable from scripts/.
STEWARD_LEVEL = 0
PROMOTED_LEVEL = 1
HUMAN_LEVEL = 2
STEWARD_ROLE = 'steward'
PENDING_STATUS = 'pending'
STEWARD_RESOLVER_PREFIX = 'claude-task-'
STEWARD_RESOLVER_SUFFIX = '-steward'
# Written by BOTH steward.py::_auto_escalate_to_human (a promotion) and
# steward.py::_patch_resolution_metadata (an in-place close), hence ambiguous.
UNATTRIBUTED_STEWARD_RESOLVER = 'steward'
AUTO_DISMISSED_RESOLVER = 'auto-dismissed'
L2_WATCHER_ROLE = 'escalation-watcher-auto'
CLOSE_ONLY_ACTION = 'close_only'

# steward.py::_auto_escalate_to_human files the L1 and dismisses the L0 in one
# call, so a promotion's L1 lands just before the L0's resolved_at.
PROMOTION_ADJACENCY = timedelta(seconds=5)


class StewardDisposition(Enum):
    """How one steward-worked escalation left the steward, judged as of a moment."""

    RECORD_MISSING = 'record missing'
    PENDING = 'pending'
    PROMOTED_TO_L1 = 'promoted to L1'
    RESOLVED_IN_PLACE = 'resolved in place'
    AUTO_DISMISSED = 'auto-dismissed'
    CLOSED_BY_OTHER = 'closed by other'


_UNDECIDED = frozenset({StewardDisposition.RECORD_MISSING, StewardDisposition.PENDING})


def _steward_l1s_by_task(corpus: EscalationCorpus) -> dict[str | None, list[datetime]]:
    """When each task's steward-filed level-1 records were stamped."""
    grouped: dict[str | None, list[datetime]] = {}
    for record in corpus.records.values():
        if record.level == PROMOTED_LEVEL and record.agent_role == STEWARD_ROLE:
            grouped.setdefault(record.task_id, []).append(record.timestamp)
    return grouped


def _is_steward_resolver(resolved_by: str) -> bool:
    return resolved_by == UNATTRIBUTED_STEWARD_RESOLVER or (
        resolved_by.startswith(STEWARD_RESOLVER_PREFIX)
        and resolved_by.endswith(STEWARD_RESOLVER_SUFFIX)
    )


def _classify(
    escalation_id: str,
    corpus: EscalationCorpus,
    steward_l1s: Mapping[str | None, list[datetime]],
    as_of: datetime,
) -> StewardDisposition:
    record = corpus.get(escalation_id)
    if record is None:
        return StewardDisposition.RECORD_MISSING
    if (record.status == PENDING_STATUS or record.resolved_at is None
            or record.resolved_at >= as_of):
        return StewardDisposition.PENDING
    # Promotion BEFORE attribution: a promoted L0 and an unattributed in-place
    # close both read resolved_by='steward'; only the adjacent L1 tells them apart.
    if any(timedelta(0) <= record.resolved_at - filed <= PROMOTION_ADJACENCY
           for filed in steward_l1s.get(record.task_id, ())):
        return StewardDisposition.PROMOTED_TO_L1
    resolver = record.resolved_by or ''
    if _is_steward_resolver(resolver):
        return StewardDisposition.RESOLVED_IN_PLACE
    if resolver == AUTO_DISMISSED_RESOLVER:
        return StewardDisposition.AUTO_DISMISSED
    return StewardDisposition.CLOSED_BY_OTHER


def classify_steward_disposition(
    escalation_id: str, corpus: EscalationCorpus, *, as_of: datetime
) -> StewardDisposition:
    """How *escalation_id* left the steward, judged as of *as_of*.

    Unresolved at *as_of* (or resolved at or after it) is PENDING, so a
    later re-run over the same window reproduces the same answer.
    """
    return _classify(escalation_id, corpus, _steward_l1s_by_task(corpus), as_of)


@dataclass(frozen=True)
class StewardDispositionSummary:
    """Dispositions of a set of steward-worked escalations, each counted once.

    ``counts`` holds every disposition in Enum order, zeros included. The
    shares are derived from it, over the DECIDED escalations (neither missing
    nor pending), and are None when none was decided.
    """

    counts: tuple[tuple[StewardDisposition, int], ...]
    unlinked_runs: int

    def count(self, disposition: StewardDisposition) -> int:
        return dict(self.counts)[disposition]

    @property
    def decided(self) -> int:
        return sum(n for disposition, n in self.counts if disposition not in _UNDECIDED)

    @property
    def resolved_in_place_share(self) -> float | None:
        return self._share(StewardDisposition.RESOLVED_IN_PLACE)

    @property
    def promoted_share(self) -> float | None:
        return self._share(StewardDisposition.PROMOTED_TO_L1)

    def _share(self, disposition: StewardDisposition) -> float | None:
        return None if self.decided == 0 else self.count(disposition) / self.decided


def summarize_steward_dispositions(
    escalation_ids: Iterable[str | None], corpus: EscalationCorpus, *, as_of: datetime
) -> StewardDispositionSummary:
    """Classify each distinct escalation once; a None id is a run that named none."""
    ids = list(escalation_ids)
    steward_l1s = _steward_l1s_by_task(corpus)
    tally = Counter(
        _classify(esc_id, corpus, steward_l1s, as_of)
        for esc_id in dict.fromkeys(i for i in ids if i is not None)
    )
    return StewardDispositionSummary(
        counts=tuple((disposition, tally[disposition]) for disposition in StewardDisposition),
        unlinked_runs=sum(1 for esc_id in ids if esc_id is None),
    )


@dataclass(frozen=True)
class L2TierMetrics:
    """Level-2 (human-tier) records filed in ``[since, until)`` and how they closed.

    ``resolved`` counts those resolved before *until*; the rates and the
    close_only share are derived, so they cannot disagree with the counts.
    """

    since: datetime
    until: datetime
    filed: int
    watcher_filed: int
    resolved: int
    close_only: int

    @property
    def days(self) -> float:
        return (self.until - self.since) / timedelta(days=1)

    @property
    def filed_per_day(self) -> float:
        return self.filed / self.days

    @property
    def watcher_filed_per_day(self) -> float:
        return self.watcher_filed / self.days

    @property
    def close_only_share(self) -> float | None:
        return None if self.resolved == 0 else self.close_only / self.resolved


def l2_tier_metrics(
    corpus: EscalationCorpus, *, since: datetime, until: datetime
) -> L2TierMetrics:
    """Measure the human tier over the half-open window ``[since, until)``."""
    if until <= since:
        raise ValueError(f'empty window: until {until} is not after since {since}')
    filed = [
        r for r in corpus.records.values()
        if r.level == HUMAN_LEVEL and since <= r.timestamp < until
    ]
    resolved = [r for r in filed if r.resolved_at is not None and r.resolved_at < until]
    return L2TierMetrics(
        since=since,
        until=until,
        filed=len(filed),
        watcher_filed=sum(1 for r in filed if r.agent_role == L2_WATCHER_ROLE),
        resolved=len(resolved),
        close_only=sum(1 for r in resolved if r.resolution_action == CLOSE_ONLY_ACTION),
    )
