"""The executor-close audit: its population, its seeded weekly draw, and the per-(queue, class) tally.

An executor close is a DecisionRecord closed with evidence whose escalation
record the server stamped ``resolved_by == EXECUTOR_IDENTITY`` under a
``policy_class``. That record is the class's one home: its claim fields
survive resolution, while ``closing_evidence`` is quoted prose and is never
parsed for a class. :func:`executor_close_of` is the one join from a decision
to its escalation record, and every reader here goes through it.

Read-only: the audit verb in ``scripts/sitting/apply.py`` writes the outcome.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from escalation.authority import EXECUTOR_IDENTITY, policy_row
from orchestrator.session_registry import (
    UNKNOWN_QUEUE,
    AuditOutcome,
    AuditVerdict,
    DecisionRecord,
    DecisionState,
    normalize_escalations_dir,
)
from shared.project_id import fold_project_token
from sitting import inventory
from sitting.fleet_state import Window
from sitting.inventory import Shortfall

SAMPLE_SIZE = 10
AUDIT_WINDOW_DAYS = 7
AUDIT_OPTIONS: Mapping[str, str] = MappingProxyType({
    AuditVerdict.OK.value: "the executor's close was right",
    AuditVerdict.WRONG.value: "the executor's close was wrong; it counts toward flipping its class to shadow",
})

DrawState = Literal['no_executor_close', 'all_audited', 'drawn']

_SHORTFALL_SOURCE = 'executor_close'
_NO_QUEUE = frozenset({'', UNKNOWN_QUEUE})


@dataclass(frozen=True)
class ExecutorClose:
    decision_id: str
    escalation_id: str
    queue_dir: str
    project: str
    policy_class: str
    closed_at: str
    evidence: str
    audited: AuditOutcome | None


@dataclass(frozen=True)
class ExecutorCloses:
    """The executor closes in one window, each class's lifetime close count, and every decision not joined."""

    in_window: tuple[ExecutorClose, ...]
    lifetime_by_class: Mapping[str, int]
    shortfalls: tuple[Shortfall, ...]


@dataclass(frozen=True)
class AuditDraw:
    """One render's audit items.

    ``full``: every unaudited close of a class still below its full-audit
    count. ``sampled``: at most the sample size of the other unaudited closes.
    ``population``: every executor close in the window, audited or not.
    """

    state: DrawState
    full: tuple[ExecutorClose, ...]
    sampled: tuple[ExecutorClose, ...]
    seed: str
    population: int


def executor_close_of(
    record: DecisionRecord, escalation_index: Mapping[str, Mapping[str, Path]],
) -> ExecutorClose | Shortfall | None:
    """*record*'s executor close; a Shortfall when its escalation record cannot be read; None when it is not one."""
    if record.state == DecisionState.OPEN or not record.closing_evidence or not record.escalation_id:
        return None
    queue_dir = normalize_escalations_dir(record.escalations_dir)
    if queue_dir in _NO_QUEUE:
        return None
    queue = escalation_index.get(queue_dir)
    if queue is None:
        return Shortfall(_SHORTFALL_SOURCE, queue_dir, f'{record.id}: queue {queue_dir} was not scanned')
    path = queue.get(record.escalation_id)
    if path is None:
        return Shortfall(_SHORTFALL_SOURCE, queue_dir, f'{record.id}: {record.escalation_id} is not in the queue')
    esc, problem = inventory.read_escalation(path)
    if esc is None:
        return Shortfall(_SHORTFALL_SOURCE, str(path), f'{record.id}: {problem.reason if problem else "vanished"}')
    if esc.resolved_by != EXECUTOR_IDENTITY or not esc.policy_class:
        return None
    return ExecutorClose(
        decision_id=record.id,
        escalation_id=esc.id,
        queue_dir=queue_dir,
        project=fold_project_token(record.project),
        policy_class=esc.policy_class,
        closed_at=record.closed_at,
        evidence=record.closing_evidence,
        audited=record.audit_outcome,
    )


def executor_closes(
    decisions_root: Path | str, escalation_index: Mapping[str, Mapping[str, Path]], window: Window,
) -> ExecutorCloses:
    """Every executor close in *window* (on ``closed_at``), oldest first, with the lifetime count per class."""
    records, shortfalls = inventory.read_decisions(decisions_root)
    joined = [executor_close_of(record, escalation_index) for record in records]
    closes = [close for close in joined if isinstance(close, ExecutorClose)]
    in_window = sorted(
        (close for close in closes if window.holds(inventory.parse_stamp(close.closed_at))), key=_draw_order,
    )
    return ExecutorCloses(
        in_window=tuple(in_window),
        lifetime_by_class=MappingProxyType(dict(Counter(close.policy_class for close in closes))),
        shortfalls=(*shortfalls, *(close for close in joined if isinstance(close, Shortfall))),
    )


def draw(
    closes: Sequence[ExecutorClose],
    *,
    seed: str,
    lifetime_by_class: Mapping[str, int],
    sample_size: int = SAMPLE_SIZE,
) -> AuditDraw:
    """This render's audit items: a pure function of *seed* and the closes, whatever order they arrive in."""
    unaudited = sorted((close for close in closes if close.audited is None), key=_draw_order)
    full = tuple(close for close in unaudited if _audited_in_full(close.policy_class, lifetime_by_class))
    pool = [close for close in unaudited if close not in full]
    picked = random.Random(seed).sample(pool, min(sample_size, len(pool)))
    return AuditDraw(
        state=_draw_state(closes, unaudited),
        full=full,
        sampled=tuple(sorted(picked, key=_draw_order)),
        seed=seed,
        population=len(closes),
    )


def class_tally(
    decisions_root: Path | str,
    escalation_index: Mapping[str, Mapping[str, Path]],
    queue_dir: str,
    policy_class: str,
) -> tuple[int, int]:
    """``(audited, wrong)`` over the lifetime executor closes of *policy_class* on *queue_dir*."""
    records, _ = inventory.read_decisions(decisions_root)
    queue = normalize_escalations_dir(queue_dir)
    joined = (executor_close_of(record, escalation_index) for record in records if record.audit_outcome is not None)
    outcomes = [
        close.audited
        for close in joined
        if isinstance(close, ExecutorClose) and close.audited is not None
        and close.queue_dir == queue and close.policy_class == policy_class
    ]
    return len(outcomes), sum(outcome.verdict == AuditVerdict.WRONG for outcome in outcomes)


def _audited_in_full(policy_class: str, lifetime_by_class: Mapping[str, int]) -> bool:
    row = policy_row(policy_class)
    return row is not None and row.audit_full_until > lifetime_by_class.get(policy_class, 0)


def _draw_state(closes: Sequence[ExecutorClose], unaudited: Sequence[ExecutorClose]) -> DrawState:
    if not closes:
        return 'no_executor_close'
    return 'drawn' if unaudited else 'all_audited'


def _draw_order(close: ExecutorClose) -> tuple[str, str]:
    return close.closed_at, close.decision_id
