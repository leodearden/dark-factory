"""The C1 re-arm contract as pure functions over task metadata.

plans/completion-driven-triggers-prd.md C1: a predicate that exits 75 ("not
yet") on a gate declaring ``before_done.recheck_secs`` re-arms by advancing
its dated ``milestone.at``. This module holds the streak record the runner
stamps as ``metadata.predicate_recheck``, its transitions, the "is this 75
declared?" check and the ``predicate_verdict`` stamp rule.
``DeterministicRunner`` does the I/O.
"""

from __future__ import annotations

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
    'RecheckDeclaration',
    'rearmed',
    'recheck_declaration',
    'reset',
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


def rearmed(prev: PredicateRecheck | None, now: datetime, recheck_secs: int) -> PredicateRecheck:
    """The record after one more exit-75 re-arm checked at *now*."""
    return PredicateRecheck(
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
        return RecheckDeclaration(recheck_secs=None, gap=f'before_done does not validate: {exc}')
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


def verdict_stamps(
    metadata: Mapping[str, object],
    verdict: Mapping[str, object] | None,
    recheck: PredicateRecheck | None,
) -> dict[str, object]:
    """The metadata keys a verdict-bearing exit writes.

    ``predicate_verdict`` is written when the check produced one, and cleared
    when it did not but an older stamp exists: a stamp never outlives the
    check that produced it. Nothing is written otherwise.
    """
    stamps: dict[str, object] = {}
    if verdict is not None or PREDICATE_VERDICT_KEY in metadata:
        stamps[PREDICATE_VERDICT_KEY] = verdict
    if recheck is not None:
        stamps[PREDICATE_RECHECK_KEY] = recheck.to_metadata()
    return stamps
