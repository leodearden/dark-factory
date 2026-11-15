"""The C1 re-arm contract as pure functions over task metadata.

plans/completion-driven-triggers-prd.md C1: a predicate that exits 75 ("not
yet") on a gate declaring ``before_done.recheck_secs`` re-arms by advancing
its dated ``milestone.at``. This module holds the streak record the runner
stamps as ``metadata.predicate_recheck``, its transitions, the "is this 75
declared?" check, the ``predicate_verdict`` stamp rule and the text of the
``milestone_check_stalled`` escalation. ``DeterministicRunner`` does the I/O.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import ValidationError
from shared.task_metadata import BeforeDone, recheck_secs_milestone_conflict

__all__ = [
    'PREDICATE_RECHECK_KEY',
    'PREDICATE_VERDICT_KEY',
    'PredicateRecheck',
    'RearmedRecheck',
    'RecheckDeclaration',
    'rearmed',
    'recheck_declaration',
    'render_verdict',
    'reset',
    'stall_detail',
    'verdict_stamps',
]

logger = logging.getLogger(__name__)

PREDICATE_RECHECK_KEY = 'predicate_recheck'
PREDICATE_VERDICT_KEY = 'predicate_verdict'


@dataclass(frozen=True)
class PredicateRecheck:
    """A gate's streak of consecutive exit-75 re-arms.

    ``first_rearmed_at`` is when the gate FIRST re-armed and survives every
    reset (PRD open question 6); ``consecutive`` carries the current streak.
    ``next_check_at`` is the advanced ``milestone.at``, ``None`` once reset.
    """

    consecutive: int
    first_rearmed_at: datetime | None
    last_checked_at: datetime
    next_check_at: datetime | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> PredicateRecheck | None:
        """The stored record, or ``None`` when it is absent or malformed (logged)."""
        raw = metadata.get(PREDICATE_RECHECK_KEY)
        if raw is None:
            return None
        record = _parse_record(raw)
        if record is None:
            logger.warning(
                'predicate_recheck: ignoring malformed %s record %r; the streak restarts',
                PREDICATE_RECHECK_KEY,
                raw,
            )
        return record

    def to_metadata(self) -> dict[str, object]:
        return {
            'consecutive': self.consecutive,
            'first_rearmed_at': _iso(self.first_rearmed_at),
            'last_checked_at': self.last_checked_at.isoformat(),
            'next_check_at': _iso(self.next_check_at),
        }


@dataclass(frozen=True)
class RearmedRecheck(PredicateRecheck):
    """The record a re-arm writes: its next check is always scheduled."""

    next_check_at: datetime


def _parse_record(raw: object) -> PredicateRecheck | None:
    if not isinstance(raw, Mapping):
        return None
    consecutive = raw.get('consecutive')
    if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 0:
        return None
    try:
        first_rearmed_at = _optional_instant(raw.get('first_rearmed_at'))
        last_checked_at = _instant(raw.get('last_checked_at'))
        next_check_at = _optional_instant(raw.get('next_check_at'))
    except (TypeError, ValueError):
        return None
    return PredicateRecheck(consecutive, first_rearmed_at, last_checked_at, next_check_at)


def _instant(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f'expected an ISO-8601 string, got {type(value).__name__}')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f'{value!r} carries no UTC offset')
    return parsed


def _optional_instant(value: object) -> datetime | None:
    return None if value is None else _instant(value)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def rearmed(prev: PredicateRecheck | None, now: datetime, recheck_secs: int) -> RearmedRecheck:
    """The record after one more exit-75 re-arm checked at *now*."""
    return RearmedRecheck(
        consecutive=(prev.consecutive if prev else 0) + 1,
        first_rearmed_at=(prev.first_rearmed_at if prev else None) or now,
        last_checked_at=now,
        next_check_at=now + timedelta(seconds=recheck_secs),
    )


def reset(prev: PredicateRecheck | None, now: datetime) -> PredicateRecheck:
    """The record after a check at *now* that ended the streak (any non-re-arm exit)."""
    return PredicateRecheck(
        consecutive=0,
        first_rearmed_at=prev.first_rearmed_at if prev else None,
        last_checked_at=now,
        next_check_at=None,
    )


@dataclass(frozen=True)
class RecheckDeclaration:
    """Whether a gate may re-arm on exit 75: its ``recheck_secs``, or why not (``gap``)."""

    recheck_secs: int | None
    gap: str | None

    def __post_init__(self) -> None:
        if (self.recheck_secs is None) == (self.gap is None):
            raise ValueError(
                'RecheckDeclaration needs exactly one of recheck_secs and gap '
                f'(got recheck_secs={self.recheck_secs!r}, gap={self.gap!r})'
            )


def recheck_declaration(before_done: object, milestone: object) -> RecheckDeclaration:
    """Judge a stored gate against the same rules its submission had to pass."""
    try:
        action = BeforeDone.model_validate(before_done)
    except ValidationError as exc:
        return RecheckDeclaration(
            recheck_secs=None, gap=f'before_done does not validate: {_one_line(exc)}',
        )
    if action.recheck_secs is None:
        return RecheckDeclaration(
            recheck_secs=None,
            gap=(
                'before_done.recheck_secs is not declared, so exit 75 ("not yet") '
                'cannot re-arm this gate.'
            ),
        )
    conflict = recheck_secs_milestone_conflict(milestone)
    if conflict is not None:
        return RecheckDeclaration(recheck_secs=None, gap=conflict)
    return RecheckDeclaration(recheck_secs=action.recheck_secs, gap=None)


def _one_line(exc: ValidationError) -> str:
    """*exc*'s errors as one line: each failing ``before_done`` field and its message."""
    return '; '.join(
        f"{'.'.join(('before_done', *map(str, error['loc'])))}: {' '.join(error['msg'].split())}"
        for error in exc.errors()
    )


def verdict_stamps(
    metadata: Mapping[str, object],
    verdict: Mapping[str, object] | None,
    recheck: PredicateRecheck | None,
    *,
    always_stamp_verdict: bool = False,
) -> dict[str, object]:
    """The metadata keys a verdict-bearing exit writes.

    ``predicate_verdict`` is written when the check produced one, and cleared
    when it did not but an older stamp exists: a stamp never outlives the
    check that produced it. Nothing is written otherwise, unless
    *always_stamp_verdict*: a declared exit 75 (re-arm or stall) always
    records the verdict it answered, ``None`` included.
    """
    stamps: dict[str, object] = {}
    if always_stamp_verdict or verdict is not None or PREDICATE_VERDICT_KEY in metadata:
        stamps[PREDICATE_VERDICT_KEY] = verdict
    if recheck is not None:
        stamps[PREDICATE_RECHECK_KEY] = recheck.to_metadata()
    return stamps


def render_verdict(verdict: Mapping[str, object] | None) -> str:
    """A ``predicate_verdict`` as one compact line, for logs and escalation details."""
    if verdict is None:
        return 'no structured verdict'
    return json.dumps(verdict, separators=(',', ':'))


def stall_detail(
    description: str,
    *,
    streak: int,
    stall_cap: int,
    first_rearmed_at: datetime | None,
    now: datetime,
    recheck_secs: int,
    verdict: Mapping[str, object] | None,
) -> str:
    """The ``milestone_check_stalled`` detail: what a human weighs to resume or abandon."""
    if first_rearmed_at is None:
        first_line = 'first_rearmed_at: none — the gate stalled before it ever re-armed'
    else:
        elapsed = timedelta(seconds=int((now - first_rearmed_at).total_seconds()))
        first_line = f'first_rearmed_at: {first_rearmed_at.isoformat()} ({elapsed} ago)'
    return '\n'.join([
        description,
        f'Predicate still not satisfied: streak {streak}/{stall_cap} consecutive '
        'exit-75 checks (deterministic.predicate_recheck_stall_cap).',
        first_line,
        f'recheck_secs: {recheck_secs}',
        f'Last verdict: {render_verdict(verdict)}',
        'The streak is reset; first_rearmed_at is kept across this reset (it records '
        'when the gate first re-armed, not when this streak began).',
        'Resolutions:',
        '- resume: re-run the check now; a further not-yet re-arms with the streak restarted.',
        '- abandon: cancel this gate, which releases any dependent because a cancelled '
        'dependency satisfies. To end a trigger chain, abandon its human gate first, '
        'then this gate.',
    ])
