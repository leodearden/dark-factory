"""The scheduler pause classes and the one policy table every pause site consults.

Each hold on the scheduler carries a ``PauseClass``; ``PAUSE_POLICY`` records,
per class, whether it may halt, whether its predicate stays measurable while
held, whether it persists, how it rehydrates on restart, how it recovers live,
and what bounds it. Contract: ``plans/scheduler-pause-halt-retirement-prd.md`` §6.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

logger = logging.getLogger(__name__)


class PauseClass(StrEnum):
    EWA_RATE = 'ewa_rate'
    PARK_STOP = 'park_stop'
    COST_CEILING = 'cost_ceiling'
    NO_LANDINGS = 'no_landings'
    WATCHER_GUARD = 'watcher_guard'
    OPERATOR = 'operator'


class Retestability(StrEnum):
    YES = 'yes'
    PARTIAL = 'partial'
    NO = 'no'
    NOT_APPLICABLE = 'not_applicable'


class Rehydration(StrEnum):
    REASSERT = 'reassert'
    RETEST_STORED_EWA = 'retest_stored_ewa'


class LiveRecovery(StrEnum):
    NEVER = 'never'
    OWN_EVIDENCE = 'own_evidence'


class Bound(StrEnum):
    UNBOUNDED = 'unbounded'


class HaltNotPermittedError(RuntimeError):
    """A pause was requested for a class whose policy forbids halting."""


@dataclass(frozen=True)
class PausePolicy:
    pause_class: PauseClass
    may_halt: bool
    predicate_retestable: Retestability
    persisted: bool
    rehydration: Rehydration
    live_recovery: LiveRecovery
    bound: Bound

    def require_may_halt(self) -> None:
        if not self.may_halt:
            raise HaltNotPermittedError(
                f'pause class {self.pause_class.value!r} may not halt the '
                f'scheduler (PRD §6 invariant 1): it escalates instead'
            )


_EWA_RATE = PausePolicy(
    pause_class=PauseClass.EWA_RATE,
    may_halt=True,
    predicate_retestable=Retestability.NO,
    persisted=True,
    rehydration=Rehydration.RETEST_STORED_EWA,
    live_recovery=LiveRecovery.NEVER,
    bound=Bound.UNBOUNDED,
)
_PARK_STOP = PausePolicy(
    pause_class=PauseClass.PARK_STOP,
    may_halt=True,
    predicate_retestable=Retestability.NO,
    persisted=True,
    rehydration=Rehydration.REASSERT,
    live_recovery=LiveRecovery.NEVER,
    bound=Bound.UNBOUNDED,
)
_COST_CEILING = PausePolicy(
    pause_class=PauseClass.COST_CEILING,
    may_halt=True,
    predicate_retestable=Retestability.YES,
    persisted=True,
    rehydration=Rehydration.REASSERT,
    live_recovery=LiveRecovery.NEVER,
    bound=Bound.UNBOUNDED,
)
_NO_LANDINGS = PausePolicy(
    pause_class=PauseClass.NO_LANDINGS,
    may_halt=True,
    predicate_retestable=Retestability.PARTIAL,
    persisted=True,
    rehydration=Rehydration.REASSERT,
    live_recovery=LiveRecovery.OWN_EVIDENCE,
    bound=Bound.UNBOUNDED,
)
_WATCHER_GUARD = PausePolicy(
    pause_class=PauseClass.WATCHER_GUARD,
    may_halt=True,
    predicate_retestable=Retestability.NO,
    persisted=True,
    rehydration=Rehydration.REASSERT,
    live_recovery=LiveRecovery.NEVER,
    bound=Bound.UNBOUNDED,
)
_OPERATOR = PausePolicy(
    pause_class=PauseClass.OPERATOR,
    may_halt=True,
    predicate_retestable=Retestability.NOT_APPLICABLE,
    persisted=True,
    rehydration=Rehydration.REASSERT,
    live_recovery=LiveRecovery.NEVER,
    bound=Bound.UNBOUNDED,
)

PAUSE_POLICY: Mapping[PauseClass, PausePolicy] = MappingProxyType({
    policy.pause_class: policy
    for policy in (
        _EWA_RATE,
        _PARK_STOP,
        _COST_CEILING,
        _NO_LANDINGS,
        _WATCHER_GUARD,
        _OPERATOR,
    )
})

_missing_classes = set(PauseClass) - set(PAUSE_POLICY)
if _missing_classes:
    raise RuntimeError(
        'PAUSE_POLICY has no row for pause class(es): '
        + ', '.join(sorted(c.value for c in _missing_classes))
    )


def _valid_values() -> str:
    return ', '.join(c.value for c in PauseClass)


def policy_for(pause_class: PauseClass | str) -> PausePolicy:
    try:
        member = PauseClass(pause_class)
    except ValueError:
        raise ValueError(
            f'{pause_class!r} is not a pause class; valid values: {_valid_values()}'
        ) from None
    return PAUSE_POLICY[member]


def stored_pause_class(value: str | None) -> PauseClass:
    """The class of a persisted hold; a NULL or unrecognised value reads as operator.

    Reading toward ``OPERATOR`` keeps the halt when the stored class is
    unknowable (PRD §6 invariant 3, open question 3).
    """
    if value is None:
        return PauseClass.OPERATOR
    try:
        return PauseClass(value)
    except ValueError:
        logger.warning(
            'Unrecognised stored pause_class %r; reading it as %r '
            '(valid values: %s)',
            value, PauseClass.OPERATOR.value, _valid_values(),
        )
        return PauseClass.OPERATOR
