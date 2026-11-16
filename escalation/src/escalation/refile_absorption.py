"""Which resolutions are absorbing refiles, and how many.

A resolved or dismissed escalation whose ruling suppresses an identical refile
gains one ``refiles_suppressed`` on each suppression; what the counter means,
and why it never escalates on its own, is owned by
``escalation/src/escalation/queue.py::EscalationQueue.note_suppressed_refile``.
No triage surface lists terminal records, so without this module the counter
is readable only by someone who already knows the escalation id.

:func:`absorbing_resolutions` sweeps one queue's root and archive read-only and
returns every record with a non-zero counter, largest first, together with how
many files the sweep could not read, so an empty listing never hides a lossy
scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan


@dataclass(frozen=True)
class AbsorbingResolution:
    """One terminal record and the refiles its resolution has absorbed."""

    escalation_id: str
    task_id: str
    category: str
    status: str
    refiles_suppressed: int
    citation_sha: str | None
    resolved_at: str | None
    resolved_by: str | None

    @classmethod
    def of(cls, record: Escalation) -> AbsorbingResolution:
        return cls(
            escalation_id=record.id,
            task_id=record.task_id,
            category=record.category,
            status=record.status,
            refiles_suppressed=record.refiles_suppressed,
            citation_sha=record.citation_sha,
            resolved_at=record.resolved_at,
            resolved_by=record.resolved_by,
        )


@dataclass(frozen=True)
class AbsorptionReport:
    """The listing, plus the sweep's shortfall.

    ``scanned`` counts the records that were read and parsed; each ``skipped_*``
    counts files the sweep listed but lost, by the reason
    ``queue.py::read_escalation_for_scan`` gave.
    """

    rows: tuple[AbsorbingResolution, ...]
    scanned: int
    skipped_vanished: int
    skipped_unreadable: int
    skipped_unparsable: int

    @property
    def total_absorbed(self) -> int:
        return sum(row.refiles_suppressed for row in self.rows)


#: Content faults that cost one record rather than the listing. ``ValueError``
#: is a deliberate member: ``UnicodeDecodeError`` from a truncated or binary
#: file subclasses it.
_SCAN_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError, KeyError, TypeError, ValueError,
)

_SKIP_REASONS = ('vanished', 'unreadable', 'unparsable')


def absorbing_resolutions(escalations_dir: Path | str) -> AbsorptionReport:
    """List every record in *escalations_dir* whose resolution absorbed a refile.

    Status is not filtered here: ``note_suppressed_refile`` already refuses to
    bump a pending record, and the row carries the status it found.
    """
    rows: list[AbsorbingResolution] = []
    scanned = 0
    skipped = dict.fromkeys(_SKIP_REASONS, 0)

    for path in iter_all_escalation_paths(Path(escalations_dir)):
        record, reason = read_escalation_for_scan(
            path, context='refile_absorption.absorbing_resolutions',
            parse_errors=_SCAN_PARSE_ERRORS,
        )
        if record is not None:
            scanned += 1
            if record.refiles_suppressed > 0:
                rows.append(AbsorbingResolution.of(record))
        elif reason in skipped:
            skipped[reason] += 1
        else:
            raise ValueError(
                f'read_escalation_for_scan returned no record with an unknown '
                f'reason {reason!r} for {path}; expected one of {_SKIP_REASONS}'
            )

    rows.sort(key=lambda row: (-row.refiles_suppressed, row.escalation_id))
    return AbsorptionReport(
        rows=tuple(rows),
        scanned=scanned,
        skipped_vanished=skipped['vanished'],
        skipped_unreadable=skipped['unreadable'],
        skipped_unparsable=skipped['unparsable'],
    )
