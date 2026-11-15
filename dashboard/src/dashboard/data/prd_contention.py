"""The contention line: the lock keys a root's ready tasks want, and who holds them.

PURE (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.9, §5.5): no I/O and no clock
read but the injected *now*. It reads one root's status map, active rows, task
activity and scheduler state, handed in by the caller. Readiness is
``prd_view.is_ready``, so the line and the PRD rows count one ready set; the
counting and holder rules are ``lock_contention``'s, which the Scheduler tab
applies to every active row instead.
"""

from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from shared.locking import files_to_modules
from shared.scheduler_state import SchedulerStateSnapshot

from dashboard.data.census import TaskCensus
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    aged_at,
    unknown_datum,
)
from dashboard.data.lock_contention import holder_of, lock_wanters
from dashboard.data.prd_view import (
    HIGH_PRIORITIES,
    InputState,
    TaskActivity,
    ViewInput,
    composite,
    degradation_reason,
    is_ready,
    priority_of,
)
from dashboard.data.task_ids import parse_task_id

CONTENTION_TOP_N = 5
"""How many of the most-wanted lock keys the line serves; the client renders every one."""


class ContentionFigure(enum.StrEnum):
    """Every figure the contention line serves; each value IS the figure's wire field name."""

    LOCK_DEPTH = 'lock_depth'
    SNAPSHOT_AT = 'snapshot_at'
    TOP = 'top'
    HELD_SHARE = 'held_share'
    NEVER_STARTED = 'never_started'


CONTENTION_RULES: MappingProxyType[ContentionFigure, str] = MappingProxyType({
    ContentionFigure.LOCK_DEPTH: (
        "The depth the scheduler cuts lock keys to, as its snapshot records it: a task's lock "
        "keys are its metadata.files, each cut to this many path components."
    ),
    ContentionFigure.SNAPSHOT_AT: (
        "When the scheduler last wrote its state file. The file is written only when its "
        "content changes, so this is the time of the last change, not a heartbeat."
    ),
    ContentionFigure.TOP: (
        f"The {CONTENTION_TOP_N} lock keys wanted by the most ready tasks (ready as the PRD rows "
        "count it), most wanted first, ties by key; each ready task counts once per key its "
        "metadata.files name. Each key's holder is the task holding a lock that conflicts with "
        "it (one key a path prefix of the other), with its status in the store, which can be "
        "pending; no holder when none does. The Scheduler tab counts every active row, not the "
        "ready set, so its counts differ by design."
    ),
    ContentionFigure.HELD_SHARE: (
        "Per priority band (a missing priority is unrecorded), how many ready tasks there are "
        "and how many of them name at least one lock key someone holds now. Counted over the "
        "ready set, not every active row."
    ),
    ContentionFigure.NEVER_STARTED: (
        "Ready tasks at high or critical priority with no task_started event in this project's "
        "runs.db since its first event, the window start. Never started means no task_started, "
        "not no event of any kind: a task the scheduler considered and skipped still counts."
    ),
})

_STRUCTURAL_KEYS = frozenset({'inputs', 'rules'})
_SNAPSHOT_UNWRITTEN = 'no scheduler snapshot written'
_NO_LOCK_DEPTH = (
    'the scheduler snapshot carries no lock_depth (its producer predates the field)'
)


@dataclass(frozen=True)
class ContentionInputs:
    """What one root's contention line is built from.

    ``census`` is the status map's provenance; ``status_map`` is the only
    source of each task's status, holders' included.
    """

    project_root: str
    status_map: Mapping[int, str]
    census: Datum[TaskCensus]
    rows: Datum[list[dict]]
    activity: Datum[TaskActivity]
    scheduler: Datum[SchedulerStateSnapshot]


@dataclass(frozen=True, slots=True)
class Holder:
    """A lock's holder as the scheduler records it, and its status in the store (``None``: absent)."""

    task_id: str
    status: str | None

    def to_wire(self) -> dict[str, object]:
        return {'task_id': self.task_id, 'status': self.status}


@dataclass(frozen=True, slots=True)
class LockKey:
    """One lock key, how many ready tasks want it, and who holds it."""

    key: str
    wanted_by_ready: int
    holder: Holder | None

    def to_wire(self) -> dict[str, object]:
        return {
            'key': self.key,
            'wanted_by_ready': self.wanted_by_ready,
            'holder': None if self.holder is None else self.holder.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class BandShare:
    """One priority band's ready tasks, and how many of them name a held key."""

    ready: int
    naming_held_key: int

    def to_wire(self) -> dict[str, object]:
        return {'ready': self.ready, 'naming_held_key': self.naming_held_key}


@dataclass(frozen=True, slots=True)
class NeverStarted:
    """The ready high tasks with no ``task_started``, of all ready high tasks, since ``window_start``."""

    count: int
    of: int
    ids: tuple[int, ...]
    window_start: datetime | None

    def to_wire(self) -> dict[str, object]:
        return {
            'count': self.count,
            'of': self.of,
            'ids': list(self.ids),
            'window_start': _instant(self.window_start),
        }


@dataclass(frozen=True, slots=True)
class Contention:
    """One root's contention line (PRD §5.5's ``Contention``)."""

    lock_depth: int
    snapshot_at: datetime
    top: tuple[LockKey, ...]
    held_share: Mapping[str, BandShare]
    never_started: NeverStarted
    inputs: tuple[InputState, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            ContentionFigure.LOCK_DEPTH.value: self.lock_depth,
            ContentionFigure.SNAPSHOT_AT.value: _instant(self.snapshot_at),
            ContentionFigure.TOP.value: [entry.to_wire() for entry in self.top],
            ContentionFigure.HELD_SHARE.value: {
                band: share.to_wire() for band, share in self.held_share.items()
            },
            ContentionFigure.NEVER_STARTED.value: self.never_started.to_wire(),
            'inputs': [state.to_wire() for state in self.inputs],
            'rules': {figure.value: text for figure, text in CONTENTION_RULES.items()},
        }


def _instant(moment: datetime | None) -> str | None:
    return None if moment is None else moment.astimezone(UTC).isoformat()


def build_contention(inputs: ContentionInputs, *, now: datetime) -> Datum[Contention]:
    """One root's contention line at *now*, with the composite provenance of its inputs.

    It is ``unknown`` when any input is, when the scheduler has written no
    snapshot, or when the snapshot carries no ``lock_depth``; otherwise its
    state is the worst input state and its ``as_of`` the oldest input's. The
    value is checked by :func:`validate_contention` before return.
    """
    aged: dict[ViewInput, Datum[Any]] = {
        ViewInput.STATUS_MAP: aged_at(inputs.census, now),
        ViewInput.ACTIVE_ROWS: aged_at(inputs.rows, now),
        ViewInput.ACTIVITY: aged_at(inputs.activity, now),
        ViewInput.SCHEDULER_STATE: aged_at(inputs.scheduler, now),
    }
    snapshot, rows, activity = inputs.scheduler.value, inputs.rows.value, inputs.activity.value
    if inputs.census.value is None or snapshot is None or rows is None or activity is None:
        unmeasured = next(datum for datum in aged.values() if datum.value is None)
        reason = degradation_reason(aged) or 'an input has no value'
        return unknown_datum(reason, unmeasured.freshness_bound_seconds)
    if snapshot.snapshot_at is None or snapshot.lock_depth is None:
        cause = _SNAPSHOT_UNWRITTEN if snapshot.snapshot_at is None else _NO_LOCK_DEPTH
        reason = '; '.join(filter(None, (cause, degradation_reason(aged))))
        return unknown_datum(reason, inputs.scheduler.freshness_bound_seconds)
    ready = [row for row in rows if is_ready(row['id'], row, inputs.status_map, now=now)]
    wants = _wants(ready, snapshot.lock_depth, snapshot.current_holders)
    contention = Contention(
        lock_depth=snapshot.lock_depth,
        snapshot_at=snapshot.snapshot_at,
        top=_top(wants, inputs.status_map),
        held_share=_held_share(ready, wants),
        never_started=_never_started(ready, activity),
        inputs=tuple(
            InputState(name, datum.as_of, datum.state, datum.reason)
            for name, datum in aged.items()
        ),
    )
    validate_contention(contention)
    return composite(contention, aged)


@dataclass(frozen=True, slots=True)
class _Wants:
    """The lock keys each ready task wants, how many want each key, and each key's holder."""

    keys: Mapping[int, tuple[str, ...]]
    wanted: Mapping[str, int]
    holders: Mapping[str, str | None]

    def names_held_key(self, task_id: int) -> bool:
        return any(self.holders[key] is not None for key in self.keys[task_id])


def _wants(
    ready: Sequence[Mapping[str, Any]], lock_depth: int, current_holders: Mapping[str, str],
) -> _Wants:
    keys = {row['id']: _lock_keys(row, lock_depth) for row in ready}
    wanted = lock_wanters(keys.values())
    return _Wants(keys, wanted, {key: holder_of(key, current_holders) for key in wanted})


def _top(wants: _Wants, status_map: Mapping[int, str]) -> tuple[LockKey, ...]:
    ranked = sorted(wants.wanted.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        LockKey(key, count, _holder(wants.holders[key], status_map))
        for key, count in ranked[:CONTENTION_TOP_N]
    )


def _lock_keys(row: Mapping[str, Any], lock_depth: int) -> tuple[str, ...]:
    """The lock keys *row*'s ``metadata.files`` name at *lock_depth*; none unless a list."""
    metadata = row.get('metadata')
    files = metadata.get('files') if isinstance(metadata, Mapping) else None
    if not isinstance(files, list):
        return ()
    return tuple(files_to_modules((path for path in files if isinstance(path, str)), lock_depth))


def _holder(holder_id: str | None, status_map: Mapping[int, str]) -> Holder | None:
    if holder_id is None:
        return None
    task_id = parse_task_id(holder_id)
    return Holder(holder_id, None if task_id is None else status_map.get(task_id))


def _held_share(ready: Sequence[Mapping[str, Any]], wants: _Wants) -> Mapping[str, BandShare]:
    """Each band's share, the band with the most ready tasks first, ties by band."""
    by_band = Counter(priority_of(row) for row in ready)
    held_by_band = Counter(priority_of(row) for row in ready if wants.names_held_key(row['id']))
    return MappingProxyType({
        band: BandShare(count, held_by_band[band])
        for band, count in sorted(by_band.items(), key=lambda item: (-item[1], item[0]))
    })


def _never_started(ready: Sequence[Mapping[str, Any]], activity: TaskActivity) -> NeverStarted:
    high = sorted(row['id'] for row in ready if priority_of(row) in HIGH_PRIORITIES)
    ids = tuple(task_id for task_id in high if task_id not in activity.ever_started)
    return NeverStarted(
        count=len(ids), of=len(high), ids=ids, window_start=activity.first_event_at,
    )


# ---------------------------------------------------------------------------
# validate_contention
# ---------------------------------------------------------------------------


class ContentionInvariant(enum.StrEnum):
    """The rule a :class:`ContentionContractError` reports."""

    TOP_LENGTH = 'top_length'
    TOP_ORDER = 'top_order'
    NEVER_STARTED_COUNT = 'never_started_count'
    FIGURE_RULE = 'figure_rule'


class ContentionContractError(DatumContractError):
    """A :class:`Contention` broke one of its declared rules, named by :attr:`rule`."""

    def __init__(self, rule: ContentionInvariant, message: str) -> None:
        super().__init__(DatumInvariant.VALUE_CONTRACT, f'{rule.value}: {message}')
        self.rule = rule


def validate_contention(contention: Contention) -> None:
    """Raise :class:`ContentionContractError` if *contention* breaks a declared rule."""
    _check_top(contention.top)
    _check_never_started(contention.never_started)
    _check_figure_rules(contention.to_wire())


def _check_top(top: Sequence[LockKey]) -> None:
    if len(top) > CONTENTION_TOP_N:
        raise ContentionContractError(
            ContentionInvariant.TOP_LENGTH,
            f'top serves {len(top)} keys, more than CONTENTION_TOP_N={CONTENTION_TOP_N}',
        )
    ranks = [(-entry.wanted_by_ready, entry.key) for entry in top]
    if any(earlier >= later for earlier, later in pairwise(ranks)):
        served = [(entry.key, entry.wanted_by_ready) for entry in top]
        raise ContentionContractError(
            ContentionInvariant.TOP_ORDER,
            f'top must be strictly ordered by wanted_by_ready descending, then key: {served}',
        )


def _check_never_started(never_started: NeverStarted) -> None:
    if never_started.count != len(never_started.ids):
        raise ContentionContractError(
            ContentionInvariant.NEVER_STARTED_COUNT,
            f'never_started count={never_started.count} but it lists '
            f'{len(never_started.ids)} id(s): {list(never_started.ids)}',
        )


def _check_figure_rules(wire: Mapping[str, object]) -> None:
    figures = {figure.value for figure in ContentionFigure}
    unnamed = sorted(set(wire) - _STRUCTURAL_KEYS - figures)
    ruleless = sorted(figures - {figure.value for figure in CONTENTION_RULES})
    if unnamed or ruleless:
        raise ContentionContractError(
            ContentionInvariant.FIGURE_RULE,
            f'wire figures naming no ContentionFigure: {unnamed}; '
            f'figures with no rule text: {ruleless}',
        )
