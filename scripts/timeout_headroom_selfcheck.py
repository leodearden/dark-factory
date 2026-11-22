"""Nightly self-check: do the archived verify runs still fit the per-test ceilings and the leg budgets sized from them?

Two checks, each judged from the durable verify archive alone:

* PER-TEST HEADROOM.  For every anchor in
  ``shared.testing_ceiling_anchors.registered_anchors``, the worst per-test
  total of any member row (its closest timeout mark, read at that run's own
  tree, admits it) must not exceed the ceiling divided by
  ``UNDER_LOAD_HEADROOM_FACTOR``.  Censored rows (xdist worker crashes) carry
  no duration: they are listed, never judged.
* LEG BUDGET.  Every registered verify leg whose tree holds a whole-tree guard
  family member must declare ``verify_command_timeout_secs >=
  min_leg_budget_for_family(W)``, W being the leg's longest green ``test``
  command in the archive, of any role and any command.  The recorded W lives in
  ``tests/scripts/test_module_verify_budgets.py`` and is cited, never read.

Each breach is one :class:`Finding`, filed as an ``escalate_info`` in the
opaque slot :meth:`Finding.slot` names.  The slot carries the bound and the
required value, so a re-size or a materially worse measurement raises a new
finding while the same measurement is never raised twice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

from shared.testing_ceiling_anchors import CeilingAnchor, required_ceiling_secs
from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import TimeoutSite
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT, min_leg_budget_for_family
from verify_budget_census import Corpus, sanitise_prefix
from verify_log_junits import CaseOutcome, JunitCase, JunitRun, TreeSource

__all__ = [
    'RECORDED_LEG_WORST_HOME',
    'AnchorEvidence',
    'AnchorReport',
    'Finding',
    'FindingKind',
    'GreenRun',
    'LegEvidence',
    'LegReport',
    'LegStatus',
    'Observation',
    'WorstRow',
    'judge_anchors',
    'judge_legs',
    'worst_green_legs',
]

RECORDED_LEG_WORST_HOME = 'tests/scripts/test_module_verify_budgets.py::MEASURED_MODULE_SUITE_WORST_SECS'
_SLOT_PREFIX = 'timeout-headroom'


class FindingKind(StrEnum):
    """Which check a finding breaches."""

    PER_TEST_HEADROOM = 'per_test_headroom'
    LEG_BUDGET = 'leg_budget'


class LegStatus(StrEnum):
    """A registered leg's standing under the leg-budget check."""

    OK = 'ok'
    BREACH = 'breach'
    NO_RUN = 'no_run'
    NOT_IN_FAMILY = 'not_in_family'


@dataclass(frozen=True)
class Observation:
    """One kept junit row and the closest timeout mark it ran under at its run's tree."""

    run: JunitRun
    case: JunitCase
    closest: TimeoutSite | None


@dataclass(frozen=True)
class WorstRow:
    """The evidence a junit row contributes: its duration and where to find it."""

    seconds: float
    task_id: str
    attempt: int
    leg: str | None
    test_id: str
    junit_path: str
    tree_sha: str
    tree_source: TreeSource

    @classmethod
    def of(cls, observation: Observation) -> WorstRow:
        run, case = observation.run, observation.case
        return cls(
            seconds=case.seconds,
            task_id=run.task_id,
            attempt=run.attempt,
            leg=run.infix,
            test_id=f'{case.classname}::{case.name}',
            junit_path=str(run.path),
            tree_sha=run.tree.sha,
            tree_source=run.tree.source,
        )


@dataclass(frozen=True)
class GreenRun:
    """A leg's longest green ``test`` command in the summary corpus."""

    infix: str
    seconds: float
    task_id: str
    attempt: int
    summary_path: str
    cmd: str
    role: str | None


@dataclass(frozen=True)
class AnchorEvidence:
    """Why an anchor's ceiling no longer clears its observed worst member."""

    anchor_id: str
    ceiling_secs: float
    headroom_factor: int
    recorded_worst_secs: float
    recorded_worst_home: str
    observed_worst_secs: float
    required_ceiling_secs: int
    worst_row: WorstRow

    @property
    def kind(self) -> FindingKind:
        return FindingKind.PER_TEST_HEADROOM

    @property
    def subject(self) -> str:
        return self.anchor_id

    @property
    def bound(self) -> float:
        return self.ceiling_secs

    @property
    def required(self) -> int:
        return self.required_ceiling_secs


@dataclass(frozen=True)
class LegEvidence:
    """Why a family leg's budget no longer outlasts its slowest green run plus the family ceiling."""

    prefix: str
    budget_secs: int
    family_ceiling_secs: int
    observed_worst_green_secs: float
    required_budget_secs: int
    recorded_worst_home: str
    worst_run: GreenRun

    @property
    def kind(self) -> FindingKind:
        return FindingKind.LEG_BUDGET

    @property
    def subject(self) -> str:
        return self.prefix

    @property
    def bound(self) -> float:
        return self.budget_secs

    @property
    def required(self) -> int:
        return self.required_budget_secs


def _plain(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {key: value.value if isinstance(value, Enum) else value for key, value in items}


def _slot_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


@dataclass(frozen=True)
class Finding:
    """One breach.  Kind, subject, bound and required are read from its evidence, never stored twice."""

    evidence: AnchorEvidence | LegEvidence

    @property
    def kind(self) -> FindingKind:
        return self.evidence.kind

    @property
    def subject(self) -> str:
        return self.evidence.subject

    @property
    def bound(self) -> float:
        return self.evidence.bound

    @property
    def required(self) -> int:
        return self.evidence.required

    def slot(self) -> str:
        """The opaque escalation slot ``timeout-headroom:<kind>:<subject>:<bound>-><required>``; compared, never parsed."""
        return (
            f'{_SLOT_PREFIX}:{self.kind}:{self.subject}:'
            f'{_slot_number(self.bound)}->{_slot_number(self.required)}'
        )

    def record(self) -> dict[str, Any]:
        """The finding as JSON-ready data."""
        return {
            'kind': self.kind.value,
            'subject': self.subject,
            'bound': self.bound,
            'required': self.required,
            'slot': self.slot(),
            'evidence': asdict(self.evidence, dict_factory=_plain),
        }


@dataclass(frozen=True)
class AnchorReport:
    """One anchor's night: ``members`` counts its timed rows at or above the read floor."""

    anchor_id: str
    ceiling_secs: float
    admissible_worst_secs: float
    members: int
    worst: WorstRow | None
    censored: tuple[WorstRow, ...]
    finding: Finding | None

    def record(self) -> dict[str, Any]:
        """The report as JSON-ready data."""
        return {
            'anchor_id': self.anchor_id,
            'ceiling_secs': self.ceiling_secs,
            'admissible_worst_secs': self.admissible_worst_secs,
            'members': self.members,
            'worst': None if self.worst is None else asdict(self.worst, dict_factory=_plain),
            'censored': [asdict(row, dict_factory=_plain) for row in self.censored],
            'slot': None if self.finding is None else self.finding.slot(),
        }


@dataclass(frozen=True)
class LegReport:
    """One registered leg's night."""

    prefix: str
    budget_secs: int
    in_family: bool
    worst_run: GreenRun | None
    required_budget_secs: int | None
    status: LegStatus
    finding: Finding | None

    def record(self) -> dict[str, Any]:
        """The report as JSON-ready data."""
        return {
            'prefix': self.prefix,
            'budget_secs': self.budget_secs,
            'in_family': self.in_family,
            'worst_run': None if self.worst_run is None else asdict(self.worst_run),
            'required_budget_secs': self.required_budget_secs,
            'status': self.status.value,
            'slot': None if self.finding is None else self.finding.slot(),
        }


def judge_anchors(
    anchors: Iterable[CeilingAnchor], observations: Iterable[Observation]
) -> tuple[AnchorReport, ...]:
    """Judge every anchor against the observations its population admits."""
    observed = tuple(observations)
    return tuple(_judge_anchor(anchor, observed) for anchor in anchors)


def _judge_anchor(anchor: CeilingAnchor, observations: tuple[Observation, ...]) -> AnchorReport:
    members = [o for o in observations if anchor.population.admits(o.closest)]
    censored = tuple(WorstRow.of(o) for o in members if o.case.outcome is CaseOutcome.CENSORED)
    timed = [o for o in members if o.case.outcome is not CaseOutcome.CENSORED]
    worst = WorstRow.of(max(timed, key=lambda o: o.case.seconds)) if timed else None
    finding = None
    if worst is not None and worst.seconds > anchor.admissible_worst_secs:
        finding = Finding(
            AnchorEvidence(
                anchor_id=anchor.anchor_id,
                ceiling_secs=anchor.ceiling_secs,
                headroom_factor=UNDER_LOAD_HEADROOM_FACTOR,
                recorded_worst_secs=anchor.measured_worst_secs,
                recorded_worst_home=anchor.provenance,
                observed_worst_secs=worst.seconds,
                required_ceiling_secs=required_ceiling_secs(worst.seconds),
                worst_row=worst,
            )
        )
    return AnchorReport(
        anchor_id=anchor.anchor_id,
        ceiling_secs=anchor.ceiling_secs,
        admissible_worst_secs=anchor.admissible_worst_secs,
        members=len(timed),
        worst=worst,
        censored=censored,
        finding=finding,
    )


def _real_duration(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_green_test(command: object) -> bool:
    if not isinstance(command, dict):
        return False
    rc = command.get('rc')
    return (
        command.get('label') == 'test'
        and isinstance(rc, int)
        and not isinstance(rc, bool)
        and rc == 0
        and command.get('timed_out') is False
        and _real_duration(command.get('duration_secs'))
    )


def worst_green_legs(corpus: Corpus, infixes: Iterable[str]) -> dict[str, GreenRun]:
    """Per wanted infix, the longest green ``test`` command of any record, any role and any command."""
    wanted = frozenset(infixes)
    worst: dict[str, GreenRun] = {}
    for record in corpus.records:
        infix = record.where.module_prefix
        commands = record.payload.get('commands')
        if infix not in wanted or not isinstance(commands, list):
            continue
        for command in commands:
            if not _is_green_test(command):
                continue
            seconds = float(command['duration_secs'])
            if infix in worst and worst[infix].seconds >= seconds:
                continue
            worst[infix] = GreenRun(
                infix=infix,
                seconds=seconds,
                task_id=record.where.task_id,
                attempt=record.where.attempt,
                summary_path=str(record.where.path),
                cmd=str(command.get('cmd', '')),
                role=record.role,
            )
    return worst


def judge_legs(
    budgets: Mapping[str, int], family_legs: frozenset[str], worst_green: Mapping[str, GreenRun]
) -> tuple[LegReport, ...]:
    """Judge every registered leg's budget; *worst_green* is keyed by the leg's sanitised infix."""
    return tuple(
        _judge_leg(prefix, budgets[prefix], prefix in family_legs, worst_green.get(sanitise_prefix(prefix)))
        for prefix in sorted(budgets)
    )


def _judge_leg(prefix: str, budget: int, in_family: bool, worst_run: GreenRun | None) -> LegReport:
    if not in_family:
        return LegReport(prefix, budget, False, worst_run, None, LegStatus.NOT_IN_FAMILY, None)
    if worst_run is None:
        return LegReport(prefix, budget, True, None, None, LegStatus.NO_RUN, None)
    required = min_leg_budget_for_family(worst_run.seconds)
    if budget >= required:
        return LegReport(prefix, budget, True, worst_run, required, LegStatus.OK, None)
    finding = Finding(
        LegEvidence(
            prefix=prefix,
            budget_secs=budget,
            family_ceiling_secs=WHOLE_TREE_SCAN_TEST_TIMEOUT,
            observed_worst_green_secs=worst_run.seconds,
            required_budget_secs=required,
            recorded_worst_home=RECORDED_LEG_WORST_HOME,
            worst_run=worst_run,
        )
    )
    return LegReport(prefix, budget, True, worst_run, required, LegStatus.BREACH, finding)
