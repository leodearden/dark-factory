"""Re-stamp every unrecorded deferred row with the record its legacy metadata names.

Dry-run by default. Contract: plans/deferral-flip-condition-prd.md §4 decision 9 and §9 ε.
Run it as ``uv run --project shared python scripts/migrate_deferrals.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import _checkout as _checkout
import deferral_store
import orchestrator_projects
import project_report
from deferral_store import (
    Changed,
    DeferralStore,
    Refused,
    Stamped,
    StoreUnreachable,
    TaskRow,
    Unechoed,
)
from orchestrator_projects import ProjectTarget
from project_report import UnreadableProject
from shared.task_deferral import (
    REASON_MAX_CHARS,
    CarriedByRequest,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralKind,
    DeferralRefusalCode,
    DeferralRequest,
    InvalidRecord,
    LegacyUnknownRequest,
    NeedsHumanReason,
    needs_human,
    parse_request,
    stamp,
)
from shared.task_statuses import TaskStatus

MIGRATION_AGENT_ID: Final = 'migrate-deferrals'
HUMAN_DECOMPOSED_NOTE: Final = 'human_decomposed: possibly an uncommitted planning birth'
ALREADY_RECORDED: Final = 'already recorded'
_VALUE_HEAD_CHARS: Final = 60
_TOOL: Final = 'migrate_deferrals.py'


class Source(StrEnum):
    """The legacy metadata key a mapped request was read from."""

    X_COALESCED_INTO = 'x_coalesced_into'
    TRIGGER = 'trigger'
    X_ARMED_BY = 'x_armed_by'


class _Mapped(NamedTuple):
    value: object
    raw: dict[str, Any]


def _coalesced_into(metadata: Mapping[str, Any]) -> _Mapped | None:
    value = metadata.get(Source.X_COALESCED_INTO)
    return None if value is None else _Mapped(value, {'kind': DeferralKind.CARRIED_BY, 'carrier_task_id': value})


def _watch_trigger(metadata: Mapping[str, Any]) -> _Mapped | None:
    value = metadata.get(Source.TRIGGER)
    if metadata.get('deferred_watch') is None or value is None:
        return None
    return _Mapped(value, {'kind': DeferralKind.UNTIL_CONDITION, 'condition': value})


def _armed_by(metadata: Mapping[str, Any]) -> _Mapped | None:
    value = metadata.get(Source.X_ARMED_BY)
    return None if value is None else _Mapped(value, {'kind': DeferralKind.UNTIL_CONDITION, 'condition': value})


_MAPPINGS: Final[tuple[tuple[Source, Callable[[Mapping[str, Any]], _Mapped | None]], ...]] = (
    (Source.X_COALESCED_INTO, _coalesced_into),
    (Source.TRIGGER, _watch_trigger),
    (Source.X_ARMED_BY, _armed_by),
)


@dataclass(frozen=True)
class Fallback:
    """A mapped request that was refused, so the row is stamped legacy_unknown instead."""

    source: Source
    value: object
    reason_code: DeferralInvalidReason


@dataclass(frozen=True)
class RowPlan:
    """The request one row will be re-stamped with, already validated by the shared parser."""

    task_id: int
    title: str
    source: Source | None
    request: DeferralRequest
    fallback: Fallback | None
    mapped_kind: DeferralKind = DeferralKind.LEGACY_UNKNOWN
    source_value: object = None
    human_decomposed: bool = False


@dataclass(frozen=True)
class AlreadyRecorded:
    task_id: int
    title: str
    kind: DeferralKind


def _truncated(text: str, budget: int) -> str:
    return text if len(text) <= budget else f'{text[: max(budget - 1, 0)]}…'


def _legacy_reason(fallback: Fallback | None, human_decomposed: bool) -> str:
    note = f'; {HUMAN_DECOMPOSED_NOTE}' if human_decomposed else ''
    if fallback is None:
        return f'migrated: metadata names no {", ".join(Source)}{note}'
    head = f'migration fallback: metadata.{fallback.source} refused {fallback.reason_code}: '
    value = _truncated(json.dumps(fallback.value), REASON_MAX_CHARS - len(head) - len(note))
    return f'{head}{value}{note}'


def _legacy_request(fallback: Fallback | None, human_decomposed: bool) -> LegacyUnknownRequest:
    return LegacyUnknownRequest(reason=_legacy_reason(fallback, human_decomposed))


def fallback_plan(plan: RowPlan, reason_code: DeferralInvalidReason) -> RowPlan:
    """*plan* re-planned as legacy_unknown because its mapped request was refused with *reason_code*."""
    if plan.source is None:
        raise ValueError(f'task {plan.task_id} was planned legacy_unknown already; it has no mapping to fall back from')
    fallback = Fallback(source=plan.source, value=plan.source_value, reason_code=reason_code)
    return replace(plan, request=_legacy_request(fallback, plan.human_decomposed), fallback=fallback)


def _mapped_plan(row: TaskRow, source: Source, mapped: _Mapped, human_decomposed: bool, now: datetime) -> RowPlan:
    fallback: Fallback | None = None
    try:
        request: DeferralRequest = parse_request({**mapped.raw, 'reason': f'migrated from metadata.{source}'}, now=now)
    except DeferralInvalid as exc:
        fallback = Fallback(source=source, value=mapped.value, reason_code=exc.reason)
        request = _legacy_request(fallback, human_decomposed)
    return RowPlan(
        task_id=row.task_id,
        title=row.title,
        source=source,
        request=request,
        fallback=fallback,
        mapped_kind=mapped.raw['kind'],
        source_value=mapped.value,
        human_decomposed=human_decomposed,
    )


def plan_row(row: TaskRow, *, now: datetime) -> RowPlan | AlreadyRecorded:
    """A row with a valid record is skipped; any other row is planned from its legacy metadata."""
    record = row.deferral
    if record is not None and not isinstance(record, InvalidRecord):
        return AlreadyRecorded(task_id=row.task_id, title=row.title, kind=record.kind)
    metadata = row.metadata or {}
    human_decomposed = bool(metadata.get('human_decomposed'))
    for source, extract in _MAPPINGS:
        mapped = extract(metadata)
        if mapped is not None:
            return _mapped_plan(row, source, mapped, human_decomposed, now)
    return RowPlan(
        task_id=row.task_id,
        title=row.title,
        source=None,
        request=_legacy_request(None, human_decomposed),
        fallback=None,
        human_decomposed=human_decomposed,
    )


def carrier_reason(plan: RowPlan, carrier_statuses: Mapping[int, TaskStatus], *, now: datetime) -> NeedsHumanReason | None:
    """Why the carried_by record *plan* would stamp needs a human (closable, orphaned, missing), else None."""
    request = plan.request
    if not isinstance(request, CarriedByRequest):
        return None
    record = stamp(request, stamped_at=now, stamped_by=MIGRATION_AGENT_ID, deferred_at=None)
    return needs_human(record, now, carrier_statuses.get(request.carrier_task_id), None)


class MigrationExit(IntEnum):
    OK = 0
    FAILED = 1
    PREDATES_GATE = 2


class PredatesGate(Exception):
    """A write succeeded without echoing its record, so the server predates the deferral gate."""

    def __init__(self, root: Path, task_id: int) -> None:
        super().__init__(f'{root} task {task_id}')
        self.root = root
        self.task_id = task_id


class Outcome(StrEnum):
    STAMPED = 'stamped'
    MOVED = 'moved since read'
    FAILED = 'failed'


@dataclass(frozen=True)
class RowOutcome:
    """What one row's write did; *plan* is the plan last written, a fallback when it fell back."""

    plan: RowPlan
    outcome: Outcome
    detail: str | None = None


@dataclass(frozen=True)
class ProjectReport:
    target: ProjectTarget
    applied: bool
    plans: tuple[RowPlan, ...]
    recorded: tuple[AlreadyRecorded, ...]
    carrier_reasons: Mapping[int, NeedsHumanReason]
    outcomes: tuple[RowOutcome, ...]

    @property
    def failed(self) -> bool:
        return any(outcome.outcome is Outcome.FAILED for outcome in self.outcomes)


def _carrier_ids(plans: Iterable[RowPlan]) -> set[int]:
    requests = (plan.request for plan in plans)
    return {request.carrier_task_id for request in requests if isinstance(request, CarriedByRequest)}


def _refused_text(refused: Refused) -> str:
    parts = [refused.code, refused.reason_code, None if refused.field is None else f'{refused.field}={refused.value!r}']
    return _truncated(' '.join(part for part in parts if part), REASON_MAX_CHARS)


def _falls_back(plan: RowPlan, refused: Refused) -> bool:
    """A mapped request refused deferral_invalid falls back once; a legacy_unknown request never does."""
    return plan.source is not None and plan.fallback is None and refused.code == DeferralRefusalCode.INVALID


def _written(store: DeferralStore, target: ProjectTarget, plan: RowPlan) -> RowOutcome:
    match store.restamp(target.root, plan.task_id, plan.request, agent_id=MIGRATION_AGENT_ID):
        case Stamped():
            return RowOutcome(plan, Outcome.STAMPED)
        case Changed(current_status=status):
            return RowOutcome(plan, Outcome.MOVED, f'now {status}')
        case Unechoed():
            raise PredatesGate(target.root, plan.task_id)
        case Refused(reason_code=DeferralInvalidReason() as reason_code) as refused if _falls_back(plan, refused):
            return _written(store, target, fallback_plan(plan, reason_code))
        case Refused() as refused:
            return RowOutcome(plan, Outcome.FAILED, _refused_text(refused))


def _apply(store: DeferralStore, target: ProjectTarget, plan: RowPlan) -> RowOutcome:
    try:
        return _written(store, target, plan)
    except StoreUnreachable as exc:
        return RowOutcome(plan, Outcome.FAILED, str(exc))


def migrate_project(
    store: DeferralStore, target: ProjectTarget, *, now: datetime, apply: bool
) -> ProjectReport | UnreadableProject:
    """Plan every deferred row of *target*; under *apply*, re-stamp each planned row in task-id order."""
    if (unknown := project_report.unknown_project(target)) is not None:
        return unknown
    try:
        rows = sorted(store.deferred_rows(target.root), key=lambda row: row.task_id)
        planned = [plan_row(row, now=now) for row in rows]
        plans = tuple(plan for plan in planned if isinstance(plan, RowPlan))
        carrier_statuses = store.statuses(target.root, _carrier_ids(plans))
    except StoreUnreachable as exc:
        return UnreadableProject(target, str(exc))
    reasons = {plan.task_id: carrier_reason(plan, carrier_statuses, now=now) for plan in plans}
    return ProjectReport(
        target=target,
        applied=apply,
        plans=plans,
        recorded=tuple(plan for plan in planned if isinstance(plan, AlreadyRecorded)),
        carrier_reasons=MappingProxyType({task_id: reason for task_id, reason in reasons.items() if reason}),
        outcomes=tuple(_apply(store, target, plan) for plan in plans) if apply else (),
    )


def _class_lines(report: ProjectReport) -> list[str]:
    classes: dict[str, list[int]] = {}
    for plan in report.plans:
        classes.setdefault(plan.mapped_kind.value, []).append(plan.task_id)
    if report.recorded:
        classes[ALREADY_RECORDED] = [recorded.task_id for recorded in report.recorded]
    labels = (*(kind.value for kind in DeferralKind), ALREADY_RECORDED)
    return [f'{label} ({len(classes[label])}): {project_report.ids(classes[label])}' for label in labels if label in classes]


def _carrier_lines(report: ProjectReport) -> list[str]:
    lines = []
    for reason in NeedsHumanReason:
        ids = [task_id for task_id, found in report.carrier_reasons.items() if found is reason]
        lines += [f'{reason.value} carried_by: {project_report.ids(ids)}'] if ids else []
    return lines


def _fallback_line(plan: RowPlan, fallback: Fallback) -> str:
    value = _truncated(json.dumps(fallback.value), _VALUE_HEAD_CHARS)
    return f'fallback #{plan.task_id} {fallback.source}={value} {fallback.reason_code} → {DeferralKind.LEGACY_UNKNOWN}'


def _outcome_lines(report: ProjectReport) -> list[str]:
    if not report.applied:
        fallbacks = [_fallback_line(plan, plan.fallback) for plan in report.plans if plan.fallback]
        return [f'would stamp: {len(report.plans)}', *fallbacks, 'dry-run: nothing written']
    counts = Counter(outcome.outcome for outcome in report.outcomes)
    return [
        *(f'{outcome.value}: {counts[outcome]}' for outcome in Outcome),
        *(_fallback_line(row.plan, row.plan.fallback) for row in report.outcomes if row.plan.fallback),
        *(f'moved since read #{row.plan.task_id} ({row.detail})' for row in report.outcomes if row.outcome is Outcome.MOVED),
        *(f'FAILED #{row.plan.task_id}: {row.detail}' for row in report.outcomes if row.outcome is Outcome.FAILED),
    ]


def _report_lines(report: ProjectReport) -> list[str]:
    return [*_class_lines(report), *_carrier_lines(report), *_outcome_lines(report)]


def exit_code(results: Sequence[ProjectReport | UnreadableProject]) -> MigrationExit:
    failed = any(isinstance(result, UnreadableProject) or result.failed for result in results)
    return MigrationExit.FAILED if failed else MigrationExit.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Re-stamp every unrecorded deferred task with the deferral its legacy metadata names.',
        epilog=(
            'Exit 0: every project was read and every write landed or moved since read. Exit 1: a project '
            'could not be read or a write failed. Exit 2: the server predates the deferral gate; the run '
            'stopped before its next write.'
        ),
    )
    orchestrator_projects.add_arguments(parser)
    deferral_store.add_store_argument(parser)
    parser.add_argument('--apply', action='store_true', help='write the re-stamps (default: a dry-run that writes nothing)')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = project_report.targets_with_problems_reported(args, tool=_TOOL, verb='migrate')
    if not targets:
        return MigrationExit.FAILED
    store = deferral_store.store_from_args(args)
    now = datetime.now(UTC)
    try:
        results = [migrate_project(store, target, now=now, apply=args.apply) for target in targets]
    except PredatesGate as exc:
        print(
            f'{_TOOL}: server predates the deferral gate: the write to {exc.root} task {exc.task_id} succeeded '
            'without echoing its record; stopped before any further write',
            file=sys.stderr,
        )
        return MigrationExit.PREDATES_GATE
    project_report.render(results, _report_lines, out=sys.stdout)
    return exit_code(results)


if __name__ == '__main__':
    raise SystemExit(main())
