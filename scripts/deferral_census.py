"""Every deferred row's hold, classified with the shared deferral model; ``--check`` gates on it.

Contract: plans/deferral-flip-condition-prd.md §4 decision 10 ("Census") and §9 ε.
Run it as ``uv run --project shared python scripts/deferral_census.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

import _checkout as _checkout
import deferral_store
import orchestrator_projects
import project_report
from deferral_store import DeferralStore, RecordHit, StoreUnreachable, TaskRow
from orchestrator_projects import ProjectTarget
from project_report import UnreadableProject
from shared.process_identity import (
    PROC_ROOT,
    Liveness,
    ProcessIdentity,
    ProcessIdentityUnreadable,
    liveness,
)
from shared.task_deferral import (
    CarriedByDeferral,
    Deferral,
    DeferralKind,
    HeldBySessionDeferral,
    InvalidRecord,
    NeedsHumanReason,
    needs_human,
)
from shared.task_statuses import TaskStatus

# The spawn_context the flake ledger stamps on the de-flake owners it files:
# orchestrator/src/orchestrator/flake_ledger.py::build_deflake_task_arguments
DEFLAKE_SPAWN_CONTEXT: Final = 'flake_ledger_debt'
NO_RECORD: Final = 'no record'
INVALID_RECORD: Final = 'invalid record'

_FAILS_CHECK: Final = frozenset(
    {NeedsHumanReason.MISSING_RECORD, NeedsHumanReason.INVALID_RECORD, NeedsHumanReason.EXPIRY_OVERDUE}
)
_HELD_ORDER: Final = (*(kind.value for kind in DeferralKind), NO_RECORD, INVALID_RECORD)
_LISTED: Final = tuple(reason for reason in NeedsHumanReason if reason not in _FAILS_CHECK)
_DETAIL_CHARS: Final = 300
_TOOL: Final = 'deferral_census.py'


@dataclass(frozen=True)
class RowVerdict:
    task_id: int
    title: str
    held: str
    needs_human: NeedsHumanReason | None
    deflake_owner: bool
    invalid_detail: str | None = None


@dataclass(frozen=True)
class Referents:
    """What the recorded holds point at: carrier task ids, and each held row's holder."""

    carrier_ids: frozenset[int]
    holders: Mapping[int, ProcessIdentity]


def _held(record: Deferral | InvalidRecord | None) -> str:
    match record:
        case None:
            return NO_RECORD
        case InvalidRecord():
            return INVALID_RECORD
        case _:
            return record.kind.value


def _is_deflake_owner(row: TaskRow) -> bool:
    return row.metadata is not None and row.metadata.get('spawn_context') == DEFLAKE_SPAWN_CONTEXT


def referents(rows: Iterable[TaskRow]) -> Referents:
    carrier_ids: set[int] = set()
    holders: dict[int, ProcessIdentity] = {}
    for row in rows:
        record = row.deferral
        if isinstance(record, CarriedByDeferral):
            carrier_ids.add(record.carrier_task_id)
        elif isinstance(record, HeldBySessionDeferral):
            holders[row.task_id] = record.holder
    return Referents(carrier_ids=frozenset(carrier_ids), holders=MappingProxyType(holders))


def _verdict(
    row: TaskRow,
    now: datetime,
    carrier_statuses: Mapping[int, TaskStatus],
    holder_liveness: Mapping[int, Liveness | None],
) -> RowVerdict:
    record = row.deferral
    carrier_status = (
        carrier_statuses.get(record.carrier_task_id) if isinstance(record, CarriedByDeferral) else None
    )
    holder = holder_liveness.get(row.task_id) if isinstance(record, HeldBySessionDeferral) else None
    return RowVerdict(
        task_id=row.task_id,
        title=row.title,
        held=_held(record),
        needs_human=needs_human(record, now, carrier_status, holder),
        deflake_owner=_is_deflake_owner(row),
        invalid_detail=record.detail if isinstance(record, InvalidRecord) else None,
    )


def classify_deferred(
    rows: Iterable[TaskRow],
    *,
    now: datetime,
    carrier_statuses: Mapping[int, TaskStatus],
    holder_liveness: Mapping[int, Liveness | None],
) -> tuple[RowVerdict, ...]:
    """One verdict per row; a carrier absent from *carrier_statuses* is missing, a holder absent from
    *holder_liveness* is unread."""
    return tuple(_verdict(row, now, carrier_statuses, holder_liveness) for row in rows)


def fails_check(verdict: RowVerdict) -> bool:
    return verdict.needs_human in _FAILS_CHECK


def holder_liveness(identity: ProcessIdentity, *, proc_root: Path = PROC_ROOT) -> Liveness | None:
    """The holder's liveness, or None when /proc could not answer (never read as dead)."""
    try:
        return liveness(identity, proc_root=proc_root)
    except ProcessIdentityUnreadable:
        return None


class CensusExit(IntEnum):
    OK = 0
    VIOLATIONS = 1
    UNREADABLE = 2


@dataclass(frozen=True)
class ProjectCensus:
    target: ProjectTarget
    verdicts: tuple[RowVerdict, ...]
    strays: tuple[RecordHit, ...]
    record_count: int

    @property
    def failing(self) -> tuple[RowVerdict, ...]:
        return tuple(verdict for verdict in self.verdicts if fails_check(verdict))

    @property
    def violates(self) -> bool:
        return bool(self.failing or self.strays)

    @property
    def no_producer(self) -> bool:
        """Deferred rows exist and no row anywhere carries a record: nothing has written one yet."""
        return bool(self.verdicts) and self.record_count == 0


def census_project(store: DeferralStore, target: ProjectTarget, *, now: datetime) -> ProjectCensus | UnreadableProject:
    if (unknown := project_report.unknown_project(target)) is not None:
        return unknown
    try:
        rows = store.deferred_rows(target.root)
        hits = store.rows_with_record(target.root)
        found = referents(rows)
        carrier_statuses = store.statuses(target.root, found.carrier_ids)
    except StoreUnreachable as exc:
        return UnreadableProject(target, str(exc))
    holder_states = {task_id: holder_liveness(holder) for task_id, holder in found.holders.items()}
    return ProjectCensus(
        target=target,
        verdicts=classify_deferred(rows, now=now, carrier_statuses=carrier_statuses, holder_liveness=holder_states),
        strays=tuple(hit for hit in hits if hit.status is not TaskStatus.DEFERRED),
        record_count=len(hits),
    )


def _one_line(text: str) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= _DETAIL_CHARS else f'{flat[:_DETAIL_CHARS]}…'


def _failing_line(verdict: RowVerdict) -> str:
    reason = verdict.needs_human.value if verdict.needs_human else ''
    detail = f' — {_one_line(verdict.invalid_detail)}' if verdict.invalid_detail else ''
    return f'FAIL #{verdict.task_id} {reason}: {verdict.title}{detail}'


def _listed_lines(verdicts: Sequence[RowVerdict]) -> list[str]:
    lines: list[str] = []
    for reason in _LISTED:
        matching = [verdict for verdict in verdicts if verdict.needs_human is reason]
        if reason is NeedsHumanReason.LEGACY_UNKNOWN:
            owners = [verdict.task_id for verdict in matching if verdict.deflake_owner]
            others = [verdict.task_id for verdict in matching if not verdict.deflake_owner]
            lines += [f'{reason.value}: {project_report.ids(others)}'] if others else []
            lines += [f'{reason.value} (de-flake owner): {project_report.ids(owners)}'] if owners else []
        elif matching:
            lines.append(f'{reason.value}: {project_report.ids(verdict.task_id for verdict in matching)}')
    return lines


def _closing_line(census: ProjectCensus) -> list[str]:
    if census.no_producer:
        return [f'no deferral producer has written: {len(census.verdicts)} deferred rows, no record on any row']
    if census.violates:
        return []
    return ['clean: no missing, invalid or overdue record, and no stray record']


def _census_lines(census: ProjectCensus) -> list[str]:
    counts = Counter(verdict.held for verdict in census.verdicts)
    table = [f'{counts[held]:>7}  {held}' for held in _HELD_ORDER if counts[held]]
    return [
        f'deferred rows by hold ({len(census.verdicts)}):',
        *(table or ['      0  (no deferred rows)']),
        f'needs human: {sum(verdict.needs_human is not None for verdict in census.verdicts)}',
        *(_failing_line(verdict) for verdict in census.failing),
        *(f'FAIL #{hit.task_id} stray record on a {hit.status.value} row: {hit.title}' for hit in census.strays),
        *_listed_lines(census.verdicts),
        *_closing_line(census),
    ]


def exit_code(results: Sequence[ProjectCensus | UnreadableProject], *, check: bool) -> CensusExit:
    if any(isinstance(result, UnreadableProject) for result in results):
        return CensusExit.UNREADABLE
    if check and any(isinstance(result, ProjectCensus) and result.violates for result in results):
        return CensusExit.VIOLATIONS
    return CensusExit.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify every deferred task's hold with the shared deferral model, per project.",
        epilog=(
            'Exit 0: every project was read (and, with --check, passed). Exit 1: --check found a missing, '
            'invalid or overdue record, or a record on a row that is not deferred. Exit 2: a project could '
            'not be read, or there was no project to census.'
        ),
    )
    orchestrator_projects.add_arguments(parser)
    deferral_store.add_store_argument(parser)
    parser.add_argument(
        '--check',
        action='store_true',
        help='exit 1 on a missing, invalid or overdue record, or a stray record (default: report only)',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = project_report.targets_with_problems_reported(args, tool=_TOOL, verb='census')
    if not targets:
        return CensusExit.UNREADABLE
    store = deferral_store.store_from_args(args)
    now = datetime.now(UTC)
    results = [census_project(store, target, now=now) for target in targets]
    project_report.render(results, _census_lines, out=sys.stdout)
    return exit_code(results, check=args.check)


if __name__ == '__main__':
    raise SystemExit(main())
