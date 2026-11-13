"""The executor-close audit: its population, its seeded weekly draw, and the per-(queue, class) tally.

An executor close is a DecisionRecord closed with evidence whose escalation
record the server stamped ``resolved_by == EXECUTOR_IDENTITY`` under a
``policy_class``. That record is the class's one home for the close: its claim
fields survive resolution, while ``closing_evidence`` is quoted prose and is
never parsed for a class. :func:`executor_close_of` is the one join from a
decision to its escalation record.

Escalation archives are pruned
(escalation/src/escalation/archive.py::DEFAULT_RETENTION_DAYS), so the class
of a close is read from its escalation record inside the audit window only,
and a close there whose record is gone is counted as aged out, never a
shortfall. Every lifetime count is read from the audits instead: each
AuditOutcome names the class it judged, and the decision registry keeps it
after the escalation record is pruned.

Read-only: the audit verb in ``scripts/sitting/apply.py`` writes the outcome.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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


def answer_slot(number: int) -> str:
    """The answer grammar for audit item *number*: its bare number, as for every item, then a verdict."""
    return f'{number}={"|".join(AUDIT_OPTIONS)}[:note]'


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
class AgedOut:
    """A close whose escalation id is absent from its scanned queue: the archive pruned the record."""

    decision_id: str
    escalation_id: str
    queue_dir: str


@dataclass(frozen=True)
class ExecutorCloses:
    """The executor closes in one window, each class's audit count, and the window's closes not joined.

    ``audited_by_class``: fleet-wide, every audited close per the class its
    audit recorded. ``aged_out``: closes in the window whose escalation record
    was pruned. ``shortfalls``: the registry read's own, and every close in the
    window whose escalation record could not be read.
    """

    in_window: tuple[ExecutorClose, ...]
    audited_by_class: Mapping[str, int]
    aged_out: int
    shortfalls: tuple[Shortfall, ...]


@dataclass(frozen=True)
class AuditDraw:
    """One render's audit items.

    ``full``: every unaudited close of a class still below its full-audit
    count. ``sampled``: at most the sample size of the other unaudited closes.
    ``population``: every executor close in the window, audited or not.
    ``aged_out``: closes in the window whose class could not be read because
    their escalation record was pruned.
    """

    state: DrawState
    full: tuple[ExecutorClose, ...]
    sampled: tuple[ExecutorClose, ...]
    seed: str
    population: int
    aged_out: int


def executor_close_of(
    record: DecisionRecord, escalation_index: Mapping[str, Mapping[str, Path]],
) -> ExecutorClose | Shortfall | AgedOut | None:
    """*record*'s executor close, or None when it is not one.

    AgedOut when its escalation id is absent from its scanned queue; a
    Shortfall when its queue was not scanned or its record cannot be read.
    """
    if not _closed_with_evidence(record) or not record.escalation_id:
        return None
    queue_dir = normalize_escalations_dir(record.escalations_dir)
    if queue_dir in _NO_QUEUE:
        return None
    queue = escalation_index.get(queue_dir)
    if queue is None:
        return Shortfall(_SHORTFALL_SOURCE, queue_dir, f'{record.id}: queue {queue_dir} was not scanned')
    path = queue.get(record.escalation_id)
    if path is None:
        return AgedOut(decision_id=record.id, escalation_id=record.escalation_id, queue_dir=queue_dir)
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
        audited=record.current_audit,
    )


def executor_closes(
    decisions_root: Path | str, escalation_index: Mapping[str, Mapping[str, Path]], window: Window,
) -> ExecutorCloses:
    """Every executor close in *window* (on ``closed_at``), oldest first, with each class's audit count.

    Only the closes in *window* are joined to their escalation records.
    """
    records, shortfalls = inventory.read_decisions(decisions_root)
    joined = [
        executor_close_of(record, escalation_index)
        for record in records
        if _closed_with_evidence(record) and window.holds(inventory.parse_stamp(record.closed_at))
    ]
    return ExecutorCloses(
        in_window=tuple(sorted((close for close in joined if isinstance(close, ExecutorClose)), key=_draw_order)),
        audited_by_class=RecordedAudits.of(records).by_class(),
        aged_out=sum(isinstance(close, AgedOut) for close in joined),
        shortfalls=(*shortfalls, *(close for close in joined if isinstance(close, Shortfall))),
    )


def draw(
    closes: Sequence[ExecutorClose],
    *,
    seed: str,
    audited_by_class: Mapping[str, int],
    aged_out: int = 0,
    sample_size: int = SAMPLE_SIZE,
) -> AuditDraw:
    """This render's audit items: a pure function of *seed* and the closes, whatever order they arrive in."""
    unaudited = sorted((close for close in closes if close.audited is None), key=_draw_order)
    full = tuple(close for close in unaudited if _audited_in_full(close.policy_class, audited_by_class))
    pool = [close for close in unaudited if close not in full]
    picked = random.Random(seed).sample(pool, min(sample_size, len(pool)))
    return AuditDraw(
        state=_draw_state(closes, unaudited),
        full=full,
        sampled=tuple(sorted(picked, key=_draw_order)),
        seed=seed,
        population=len(closes),
        aged_out=aged_out,
    )


@dataclass(frozen=True)
class RecordedAudits:
    """Every audit the decision registry holds, one per audited close, keyed by (decision id, closed_at) with its queue.

    Keyed by the close, not the decision: a reopened record closed again
    carries one outcome per audited close, and every one counts. Read from the
    registry alone, so a count outlives the escalation records the archive
    prunes. :meth:`recording` folds in an audit written since the read, so a
    run that writes several audits reads the registry once.
    """

    by_close: Mapping[tuple[str, str], tuple[str, AuditOutcome]]

    @classmethod
    def of(cls, records: Iterable[DecisionRecord]) -> RecordedAudits:
        return cls(MappingProxyType({}))._with(records)

    @classmethod
    def read(cls, decisions_root: Path | str) -> RecordedAudits:
        records, _ = inventory.read_decisions(decisions_root)
        return cls.of(records)

    def recording(self, record: DecisionRecord) -> RecordedAudits:
        """These audits with each of *record*'s own in place of whatever was held for that close."""
        return self._with((record,))

    def tally(self, queue_dir: str, policy_class: str) -> tuple[int, int]:
        """``(audited, wrong)`` over every audit of *policy_class* on *queue_dir*."""
        queue = normalize_escalations_dir(queue_dir)
        outcomes = [outcome for held_queue, outcome in self.by_close.values()
                    if held_queue == queue and outcome.policy_class == policy_class]
        return len(outcomes), sum(outcome.verdict == AuditVerdict.WRONG for outcome in outcomes)

    def by_class(self) -> Mapping[str, int]:
        """Fleet-wide, each class's audit count."""
        return MappingProxyType(dict(Counter(outcome.policy_class for _, outcome in self.by_close.values())))

    def _with(self, records: Iterable[DecisionRecord]) -> RecordedAudits:
        held = dict(self.by_close)
        for record in records:
            queue = normalize_escalations_dir(record.escalations_dir)
            for outcome in record.audit_outcomes:
                held[(record.id, outcome.closed_at)] = (queue, outcome)
        return RecordedAudits(MappingProxyType(held))


def _closed_with_evidence(record: DecisionRecord) -> bool:
    return record.state != DecisionState.OPEN and bool(record.closing_evidence)


def _audited_in_full(policy_class: str, audited_by_class: Mapping[str, int]) -> bool:
    row = policy_row(policy_class)
    return row is not None and row.audit_full_until > audited_by_class.get(policy_class, 0)


def _draw_state(closes: Sequence[ExecutorClose], unaudited: Sequence[ExecutorClose]) -> DrawState:
    if not closes:
        return 'no_executor_close'
    return 'drawn' if unaudited else 'all_audited'


def _draw_order(close: ExecutorClose) -> tuple[str, str]:
    return close.closed_at, close.decision_id
