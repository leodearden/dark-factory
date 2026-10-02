"""The escalation corpus datum: one walk of every escalation queue, root and archive.

Declared by ``plans/dashboard-one-datum-one-path-prd.md``, decision 13. Every
escalation surface reads the same walk: the Escalations tab's live-queue
table and the analytics aggregates. "Pending in the live queue" and "open in
history" are two named views over that one walk (:class:`EscalationView`),
so the two can no longer count different populations at different
freshnesses.

A queue whose directory was not found, or that holds a file this walk could
not parse, is a partial scan. Every view whose scope covers it is a
``lower_bound`` Datum whose reason names the queue and the cause (INV-11).

This module reads no clock: the walk's instant is injected.
"""

from __future__ import annotations

import enum
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumState, unknown_datum

logger = logging.getLogger(__name__)

CORPUS_FRESHNESS_BOUND_SECONDS = 120
"""The age past which a corpus walk is no longer fresh."""


class QueueKind(enum.StrEnum):
    ORCHESTRATOR = 'orchestrator'
    RECONCILIATION = 'reconciliation'


@dataclass(frozen=True, slots=True)
class QueueRef:
    """One escalation queue: its payload id and label, and the directory it lives in."""

    id: str
    label: str
    kind: QueueKind
    directory: Path


class Location(enum.StrEnum):
    """Where in its queue a file lay: the live queue root, or the archive under it."""

    ROOT = 'root'
    ARCHIVE = 'archive'


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    escalation: Escalation
    location: Location


@dataclass(frozen=True, slots=True)
class Unreadable:
    """A file in the queue that could not be parsed into an escalation."""

    path: str
    error: str
    location: Location


@dataclass(frozen=True, slots=True)
class QueueScan:
    """One queue's walk. ``reached`` is False when its directory was not found."""

    queue: QueueRef
    reached: bool
    records: tuple[CorpusRecord, ...]
    unreadable: tuple[Unreadable, ...]

    def partial_causes(self) -> tuple[str, ...]:
        """Why this scan may under-count, one phrase per cause; empty when complete."""
        causes: list[str] = []
        if not self.reached:
            causes.append(
                f'no queue directory at {self.queue.directory} '
                '(never escalated, or not mounted)'
            )
        if self.unreadable:
            causes.append(f'{len(self.unreadable)} file(s) unreadable')
        return tuple(causes)


@dataclass(frozen=True, slots=True)
class EscalationCorpus:
    """Every queue's scan, in :func:`corpus_queues` order."""

    scans: tuple[QueueScan, ...]

    def scan(self, queue_id: str) -> QueueScan:
        for scan in self.scans:
            if scan.queue.id == queue_id:
                return scan
        raise KeyError(queue_id)

    @property
    def reached_any(self) -> bool:
        return any(scan.reached for scan in self.scans)


def corpus_queues(config: DashboardConfig) -> tuple[QueueRef, ...]:
    """Every escalation queue: the primary root, the other known roots, reconciliation."""
    roots = list(dict.fromkeys([config.project_root, *config.known_project_roots]))
    orchestrators = tuple(
        QueueRef(
            id=str(root), label=root.name, kind=QueueKind.ORCHESTRATOR,
            directory=root / 'data' / 'escalations',
        )
        for root in roots
    )
    reconciliation = QueueRef(
        id='reconciliation', label='fused-memory', kind=QueueKind.RECONCILIATION,
        directory=config.reconciliation_escalations_dir,
    )
    return (*orchestrators, reconciliation)


def _scan(queue: QueueRef) -> QueueScan:
    if not queue.directory.is_dir():
        return QueueScan(queue, reached=False, records=(), unreadable=())
    records: list[CorpusRecord] = []
    unreadable: list[Unreadable] = []
    for path in iter_all_escalation_paths(queue.directory):
        location = Location.ROOT if path.parent == queue.directory else Location.ARCHIVE
        try:
            escalation = Escalation.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning('escalation corpus: could not read %s: %s', path, exc)
            unreadable.append(Unreadable(str(path), str(exc), location))
            continue
        records.append(CorpusRecord(escalation, location))
    return QueueScan(queue, reached=True, records=tuple(records), unreadable=tuple(unreadable))


def walk_corpus(queues: Iterable[QueueRef]) -> EscalationCorpus:
    """Walk every queue's root and archive once. Blocking filesystem I/O."""
    return EscalationCorpus(tuple(_scan(queue) for queue in queues))


class EscalationView(enum.StrEnum):
    """A named population over the corpus; the value is its wire key."""

    QUEUE_PENDING = 'queue_pending'
    OPEN_IN_HISTORY = 'open_in_history'

    def admits(self, record: CorpusRecord) -> bool:
        return _VIEW_PREDICATES[self](record)


def _is_pending(record: CorpusRecord) -> bool:
    return record.escalation.status == 'pending'


_VIEW_PREDICATES: Mapping[EscalationView, Callable[[CorpusRecord], bool]] = {
    EscalationView.QUEUE_PENDING: lambda r: _is_pending(r) and r.location is Location.ROOT,
    EscalationView.OPEN_IN_HISTORY: _is_pending,
}


def _scope_provenance(
    corpus_datum: Datum[EscalationCorpus], scans: Iterable[QueueScan],
) -> tuple[DatumState, str | None]:
    """The state and reason of a count over *scans*: a lower bound when any is partial."""
    partial = [
        f"{scan.queue.label}: {', '.join(causes)}"
        for scan in scans
        if (causes := scan.partial_causes())
    ]
    if partial:
        return DatumState.LOWER_BOUND, 'partial scan, so at least this many — ' + '; '.join(partial)
    if corpus_datum.state is DatumState.LOWER_BOUND:
        return DatumState.FRESH, None
    return corpus_datum.state, corpus_datum.reason


def views_over(
    corpus_datum: Datum[EscalationCorpus], queue_ids: Iterable[str],
) -> dict[EscalationView, Datum[int]]:
    """Each view's count over the queues named by *queue_ids*, as of the corpus walk."""
    corpus = corpus_datum.value
    if corpus is None:
        reason = corpus_datum.reason or 'the escalation corpus was not read'
        return {
            view: unknown_datum(reason, corpus_datum.freshness_bound_seconds)
            for view in EscalationView
        }
    scans = [corpus.scan(queue_id) for queue_id in queue_ids]
    state, reason = _scope_provenance(corpus_datum, scans)
    return {
        view: Datum(
            sum(view.admits(record) for scan in scans for record in scan.records),
            corpus_datum.as_of, state, reason, corpus_datum.freshness_bound_seconds,
        )
        for view in EscalationView
    }
