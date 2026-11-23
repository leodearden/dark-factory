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

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbsorbingResolution:
    """One terminal record and the refiles its resolution has absorbed."""

    escalation_id: str
    task_id: str | None
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


#: Content faults that cost one record rather than the listing.
#: ``UnicodeDecodeError`` from a truncated or binary file subclasses
#: ``ValueError``; ``AttributeError`` is what ``Escalation.from_dict`` raises on
#: valid JSON that is not an object, such as ``[]``.
_LISTING_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError,
)

_SKIP_REASONS = ('vanished', 'unreadable', 'unparsable')

_CONTEXT = 'refile_absorption.absorbing_resolutions'


def _type_fault(record: Escalation) -> str | None:
    """Name the fields this listing needs that lack their declared type, if any.

    ``Escalation.from_dict`` checks no types, so ``"refiles_suppressed": null``
    parses. The counter is checked on every record; the row's text fields only
    on a record that becomes a row, so a record the listing would not show is
    never reported as lost.
    """
    count = record.refiles_suppressed
    if not isinstance(count, int) or isinstance(count, bool):
        return f'refiles_suppressed={count!r}'
    if count <= 0:
        return None
    text = {'id': record.id, 'category': record.category, 'status': record.status}
    optional_text = {
        'task_id': record.task_id, 'citation_sha': record.citation_sha,
        'resolved_at': record.resolved_at, 'resolved_by': record.resolved_by,
    }
    wrong = [f'{name}={value!r}' for name, value in text.items() if not isinstance(value, str)]
    wrong += [
        f'{name}={value!r}' for name, value in optional_text.items()
        if value is not None and not isinstance(value, str)
    ]
    return ', '.join(wrong) or None


def _read(path: Path) -> tuple[Escalation | None, str]:
    """``read_escalation_for_scan``, with an ill-typed record counted unparsable."""
    record, reason = read_escalation_for_scan(
        path, context=_CONTEXT, parse_errors=_LISTING_PARSE_ERRORS,
    )
    fault = None if record is None else _type_fault(record)
    if fault is None:
        return record, reason
    logger.warning('%s: Failed to parse %s: ill-typed %s', _CONTEXT, path, fault)
    return None, 'unparsable'


def absorbing_resolutions(escalations_dir: Path | str) -> AbsorptionReport:
    """List every record in *escalations_dir* whose resolution absorbed a refile.

    Status is not filtered here: ``note_suppressed_refile`` already refuses to
    bump a pending record, and the row carries the status it found.
    """
    rows: list[AbsorbingResolution] = []
    scanned = 0
    skipped = dict.fromkeys(_SKIP_REASONS, 0)

    for path in iter_all_escalation_paths(Path(escalations_dir)):
        record, reason = _read(path)
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


_ROW = (
    '{escalation_id:<32} {task_id:<16} {category:<28} {status:<10} {absorbed:>8} '
    '{resolved_at:<32} {resolved_by:<24} {citation_sha}'
)


def _shown(value: str | None) -> str:
    return '-' if value is None else value


def _summary(report: AbsorptionReport) -> str:
    return (
        f'total_absorbed={report.total_absorbed} records={len(report.rows)} '
        f'scanned={report.scanned} skipped_vanished={report.skipped_vanished} '
        f'skipped_unreadable={report.skipped_unreadable} '
        f'skipped_unparsable={report.skipped_unparsable}'
    )


def _as_table(report: AbsorptionReport) -> str:
    """Render the listing, ending ALWAYS with the summary line.

    The skip counts print even at zero, so a missing line never has to be read
    as "nothing was lost".
    """
    if report.rows:
        lines = [_ROW.format(
            escalation_id='escalation_id', task_id='task_id', category='category',
            status='status', absorbed='absorbed', resolved_at='resolved_at',
            resolved_by='resolved_by', citation_sha='citation_sha',
        )]
        lines.extend(
            _ROW.format(
                escalation_id=row.escalation_id, task_id=_shown(row.task_id),
                category=row.category, status=row.status,
                absorbed=row.refiles_suppressed, resolved_at=_shown(row.resolved_at),
                resolved_by=_shown(row.resolved_by), citation_sha=_shown(row.citation_sha),
            )
            for row in report.rows
        )
    else:
        lines = ['no resolution has absorbed a refile']
    lines.extend(['', _summary(report)])
    return '\n'.join(lines)


def _as_json(report: AbsorptionReport) -> str:
    return json.dumps(
        {
            'rows': [asdict(row) for row in report.rows],
            'total_absorbed': report.total_absorbed,
            'scanned': report.scanned,
            'skipped_vanished': report.skipped_vanished,
            'skipped_unreadable': report.skipped_unreadable,
            'skipped_unparsable': report.skipped_unparsable,
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m escalation.refile_absorption``.

    Invoked from the repo root as::

        uv run --directory escalation python -m escalation.refile_absorption \\
            --queue-dir <project_root>/data/escalations

    Read-only: it opens escalation JSON and prints. It never resolves, bumps or
    moves anything.
    """
    parser = argparse.ArgumentParser(
        description=(
            'List the resolved/dismissed escalations whose resolution has absorbed '
            'identical refiles (refiles_suppressed > 0), largest count first.'
        ),
    )
    parser.add_argument(
        '--queue-dir', required=True, type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--json', action='store_true', default=False,
        help='Emit machine-readable JSON instead of the table.',
    )
    args = parser.parse_args(argv)

    if not args.queue_dir.is_dir():
        print(f'queue-dir is not a directory: {args.queue_dir}', file=sys.stderr)
        return 2

    report = absorbing_resolutions(args.queue_dir)
    print(_as_json(report) if args.json else _as_table(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
