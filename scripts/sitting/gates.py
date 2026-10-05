"""Evaluate whether a pending record clears the carve-out for closing a ruling Leo has already made.

The six gates are normative in
``skills/escalation-watcher/SKILL.md::Ruled-elsewhere check (answered-but-unrecorded)``
and are not restated here. What this code adds to them:

- Unknown fails closed. Every fact defaults to ``UNKNOWN``, which never holds,
  and a fact cannot claim to hold without quotable evidence.
- ``pins_recovery=None`` is unknown by construction: it is a read-time
  annotation the escalation server adds, and the on-disk record never carries
  it, so a caller that has not asked the server cannot pass gate 5.
- ``recommend_only`` demotes a would-be close instead of short-circuiting, so
  the six results are still computed and a trial can measure how often the
  gates would have fired.

Pure: no filesystem, subprocess or MCP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sitting.inventory import cited_ids

Disposition = Literal['closeable', 'report_only']
MemberClass = Literal['ruled', 'dedup_marker', 'open']

GATE_NAMES: tuple[str, ...] = (
    'ruling_is_leos_own',
    'ruling_names_this_record',
    'ruling_was_executed',
    'session_terminated',
    'record_is_not_a_pin',
    'sideways_check_ran',
)

DOCUMENTED_RULING_SOURCES = frozenset({
    'task_description',
    'commit_message',
    'memory_ruling_record',
    'escalation_resolution',
    'task_metadata_ruling',
})

VETO_PIN_PREFIX = 'veto-pin-do-not-close:'

DEDUP_MARKER = 'survivor, stays open'
"""Same marker as ``scripts/member-chain-sweep.py::DEDUP_MARKER``; that script's hyphenated name makes it unimportable."""

TERMINAL_STATUSES = frozenset({'resolved', 'dismissed'})

RECOMMEND_ONLY = 'recommend-only'


@dataclass(frozen=True)
class Fact:
    """A tri-state finding: ``held`` True / False / None (unknown), with the evidence behind it."""

    held: bool | None
    evidence: str
    source_kind: str = ''

    def __post_init__(self) -> None:
        if self.held is True and not self.evidence.strip():
            raise ValueError('a fact cannot hold without quotable evidence')


UNKNOWN = Fact(held=None, evidence='')


@dataclass(frozen=True)
class MemberOutcome:
    member_id: str
    status: str
    resolution: str


@dataclass(frozen=True)
class CarveoutFacts:
    escalation_id: str
    ruling: Fact = UNKNOWN
    names_this_record: Fact = UNKNOWN
    executed: Fact = UNKNOWN
    session_terminated: Fact = UNKNOWN
    pin_declared_by: tuple[str, ...] = ()
    pins_recovery: tuple[str, ...] | None = None
    root_cause: str = ''
    do_not_close_companions: tuple[str, ...] | None = None
    sideways: Fact = UNKNOWN
    members: tuple[MemberOutcome, ...] = ()


@dataclass(frozen=True)
class GateResult:
    number: int
    name: str
    held: bool
    evidence: str
    note: str = ''


@dataclass(frozen=True)
class Verdict:
    disposition: Disposition
    gates: tuple[GateResult, ...]
    demoted_by: str = ''

    @property
    def missed_gates(self) -> tuple[str, ...]:
        return tuple(gate.name for gate in self.gates if not gate.held)

    @property
    def all_held(self) -> bool:
        return not self.missed_gates

    def gate(self, name: str) -> GateResult:
        return next(gate for gate in self.gates if gate.name == name)


def classify_member(status: str, resolution: str) -> MemberClass:
    """Classify a member by its resolution TEXT; status alone cannot tell a ruling from a dedup survivor."""
    if status not in TERMINAL_STATUSES:
        return 'open'
    return 'dedup_marker' if DEDUP_MARKER in resolution else 'ruled'


def evaluate_carveout(facts: CarveoutFacts, *, recommend_only: bool) -> Verdict:
    gates = (
        _gate_ruling_is_leos_own(facts),
        _gate_ruling_names_this_record(facts),
        _gate_ruling_was_executed(facts),
        _gate_session_terminated(facts),
        _gate_record_is_not_a_pin(facts),
        _gate_sideways_check_ran(facts),
    )
    if not all(gate.held for gate in gates):
        return Verdict('report_only', gates)
    if recommend_only:
        return Verdict('report_only', gates, demoted_by=RECOMMEND_ONLY)
    return Verdict('closeable', gates)


def closing_evidence(verdict: Verdict) -> str:
    """The six gates' evidence verbatim, one block per gate, for the closed DecisionRecord."""
    if verdict.disposition != 'closeable':
        raise ValueError(f'closing evidence exists only for a closeable verdict, not {verdict.disposition!r}')
    return '\n\n'.join(f'gate {gate.number} {gate.name}: held\n{gate.evidence}' for gate in verdict.gates)


def _result(number: int, held: bool, evidence: str, note: str = '') -> GateResult:
    return GateResult(number=number, name=GATE_NAMES[number - 1], held=held, evidence=evidence, note=note)


def _gate_ruling_is_leos_own(facts: CarveoutFacts) -> GateResult:
    ruling = facts.ruling
    if ruling.held is True and ruling.source_kind not in DOCUMENTED_RULING_SOURCES:
        return _result(1, False, ruling.evidence, f'source {ruling.source_kind!r} is inferred authority')
    return _result(1, ruling.held is True, ruling.evidence)


def _gate_ruling_names_this_record(facts: CarveoutFacts) -> GateResult:
    ruling = facts.ruling
    if ruling.held is True and facts.escalation_id in cited_ids(ruling.evidence)[0]:
        return _result(2, True, ruling.evidence, 'the ruling cites this record by id')
    return _result(2, facts.names_this_record.held is True, facts.names_this_record.evidence)


def _gate_ruling_was_executed(facts: CarveoutFacts) -> GateResult:
    return _result(3, facts.executed.held is True, facts.executed.evidence)


def _gate_session_terminated(facts: CarveoutFacts) -> GateResult:
    return _result(4, facts.session_terminated.held is True, facts.session_terminated.evidence)


def _gate_record_is_not_a_pin(facts: CarveoutFacts) -> GateResult:
    signals = []
    if facts.pin_declared_by:
        signals.append(f'pin_declared_by={list(facts.pin_declared_by)}')
    if facts.pins_recovery is None:
        signals.append('pins_recovery unknown')
    elif facts.pins_recovery:
        signals.append(f'pins_recovery={list(facts.pins_recovery)}')
    if facts.root_cause.startswith(VETO_PIN_PREFIX):
        signals.append(f'root_cause={facts.root_cause!r}')
    if facts.do_not_close_companions is None:
        signals.append('do-not-close companions unknown')
    elif facts.do_not_close_companions:
        signals.append(f'do-not-close companions={list(facts.do_not_close_companions)}')
    evidence = (
        f'pin_declared_by={list(facts.pin_declared_by)} pins_recovery={facts.pins_recovery!r} '
        f'root_cause={facts.root_cause!r} do_not_close_companions={facts.do_not_close_companions!r}'
    )
    return _result(5, not signals, evidence, '; '.join(signals))


def _gate_sideways_check_ran(facts: CarveoutFacts) -> GateResult:
    survivors = [m.member_id for m in facts.members if classify_member(m.status, m.resolution) == 'dedup_marker']
    if survivors:
        return _result(6, False, facts.sideways.evidence, f'dedup kept this record live via {survivors}')
    return _result(6, facts.sideways.held is True, facts.sideways.evidence)
