"""Re-stamp every unrecorded deferred row with the record its legacy metadata names.

Dry-run by default. Contract: plans/deferral-flip-condition-prd.md §4 decision 9 and §9 ε.
Run it as ``uv run --project shared python scripts/migrate_deferrals.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, NamedTuple

from deferral_store import TaskRow
from shared.task_deferral import (
    REASON_MAX_CHARS,
    CarriedByDeferral,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralKind,
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
    """The request one row will be re-stamped with; *request* is the wire ``deferral`` argument."""

    task_id: int
    title: str
    source: Source | None
    request: dict[str, Any]
    fallback: Fallback | None
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


def _legacy_request(fallback: Fallback | None, human_decomposed: bool) -> dict[str, Any]:
    request = LegacyUnknownRequest(reason=_legacy_reason(fallback, human_decomposed))
    return request.model_dump(mode='json', exclude_none=True)


def fallback_plan(plan: RowPlan, reason_code: DeferralInvalidReason) -> RowPlan:
    """*plan* re-planned as legacy_unknown because its mapped request was refused with *reason_code*."""
    if plan.source is None:
        raise ValueError(f'task {plan.task_id} was planned legacy_unknown already; it has no mapping to fall back from')
    fallback = Fallback(source=plan.source, value=plan.source_value, reason_code=reason_code)
    return replace(plan, request=_legacy_request(fallback, plan.human_decomposed), fallback=fallback)


def _mapped_plan(row: TaskRow, source: Source, mapped: _Mapped, human_decomposed: bool, now: datetime) -> RowPlan:
    plan = RowPlan(
        task_id=row.task_id,
        title=row.title,
        source=source,
        request={**mapped.raw, 'reason': f'migrated from metadata.{source}'},
        fallback=None,
        source_value=mapped.value,
        human_decomposed=human_decomposed,
    )
    try:
        request = parse_request(plan.request, now=now)
    except DeferralInvalid as exc:
        return fallback_plan(plan, exc.reason)
    return replace(plan, request=request.model_dump(mode='json', exclude_none=True))


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
    record = stamp(parse_request(plan.request, now=now), stamped_at=now, stamped_by=MIGRATION_AGENT_ID, deferred_at=None)
    if not isinstance(record, CarriedByDeferral):
        return None
    return needs_human(record, now, carrier_statuses.get(record.carrier_task_id), None)
