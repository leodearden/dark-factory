"""Nightly self-check: do the archived verify runs still fit the per-test ceilings and the leg budgets sized from them?

Two checks, each judged from the durable verify archive alone:

* PER-TEST HEADROOM.  For every anchor in
  ``shared.testing_ceiling_anchors.registered_anchors``, the worst per-test
  total of any member row (its closest timeout mark, read at that run's own
  tree, admits it) must not exceed the ceiling divided by
  ``UNDER_LOAD_HEADROOM_FACTOR``.  Censored rows (xdist worker crashes) and
  timed-out rows (pytest-timeout kills) carry no duration of their own: they
  are listed, never judged.
* LEG BUDGET.  Every registered verify leg whose tree holds a whole-tree guard
  family member must declare ``verify_command_timeout_secs >=
  min_leg_budget_for_family(W)``, W being the leg's longest green ``test``
  command in the archive, of any role and any command.  The recorded W lives in
  ``tests/scripts/test_module_verify_budgets.py`` and is cited, never read.

Each breach is one :class:`Finding`, filed as an ``escalate_info`` in the
opaque slot :meth:`Finding.slot` names.  The slot carries the bound but not
the required value, so a re-size re-arms the check while a window worst that
drifts up or down under an unchanged bound is never raised twice.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from dashboard.config import DashboardConfig
from junit_case_marks import CaseMarks, Unresolved
from orchestrator.config import load_config
from orchestrator.verify_plan import FileKind, classify_file
from shared.mcp_post import mcp_server_base
from shared.mcp_session import SyncMcpSession
from shared.project_layout import default_project_root
from shared.safe_io import atomic_write_text
from shared.testing_ceiling_anchors import CeilingAnchor, registered_anchors, required_ceiling_secs
from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import TimeoutSite
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    is_whole_tree_guard_module,
    min_leg_budget_for_family,
)
from verify_budget_census import Corpus, load_records, sanitise_prefix
from verify_log_junits import CaseOutcome, JunitCase, JunitCorpus, JunitRun, TreeSource, read_runs

__all__ = [
    'AGENT_ROLE',
    'RECORDED_LEG_WORST_HOME',
    'AnchorEvidence',
    'AnchorReport',
    'FailureStage',
    'Finding',
    'FindingKind',
    'GreenRun',
    'LegEvidence',
    'LegFacts',
    'LegReport',
    'LegStatus',
    'Observation',
    'Ports',
    'Report',
    'RunConfig',
    'RunResult',
    'ToolCall',
    'WorstRow',
    'family_legs',
    'judge_anchors',
    'judge_legs',
    'main',
    'registered_leg_budgets',
    'run',
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
    """The evidence a junit row contributes: its duration, its outcome and where to find it."""

    seconds: float
    outcome: CaseOutcome
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
            outcome=case.outcome,
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
        """The leg's archive infix, the one file-name component its summaries carry (``tests_scripts``)."""
        return sanitise_prefix(self.prefix)

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


def _require_one_file_name_component(slot: str) -> None:
    if '/' in slot or '\0' in slot:
        raise ValueError(
            'an escalation slot is one file-name component, because the escalation server '
            f'mints esc-<slot> files from it; got slot {slot!r}'
        )


@dataclass(frozen=True)
class Finding:
    """One breach.  Kind, subject, bound and required are read from its evidence, never stored twice."""

    evidence: AnchorEvidence | LegEvidence

    def __post_init__(self) -> None:
        _require_one_file_name_component(self.slot())

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
        """The opaque escalation slot ``timeout-headroom:<kind>:<subject>:<bound>``; compared, never parsed.

        ``<subject>`` is the anchor id or the leg's archive infix.  ``required``
        is left out because it follows the sliding window's worst in both
        directions; it travels in the record instead.
        """
        return f'{_SLOT_PREFIX}:{self.kind}:{self.subject}:{_slot_number(self.bound)}'

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
    timed_out: tuple[WorstRow, ...]
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
            'timed_out': [asdict(row, dict_factory=_plain) for row in self.timed_out],
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
    timed = [o for o in members if o.case.outcome.has_duration]
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
        censored=_rows_of(members, CaseOutcome.CENSORED),
        timed_out=_rows_of(members, CaseOutcome.TIMED_OUT),
        finding=finding,
    )


def _rows_of(members: Iterable[Observation], outcome: CaseOutcome) -> tuple[WorstRow, ...]:
    return tuple(WorstRow.of(o) for o in members if o.case.outcome is outcome)


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


# ---------------------------------------------------------------------------
# Run: measure, judge, then file each finding whose slot has no history.

AGENT_ROLE = 'timeout-headroom-selfcheck'
ORCHESTRATOR_CONFIG = 'dark-factory-orchestrator.yaml'
ARCHIVE_PATH = 'data/verify-logs'
STAMP_PATH = 'data/timeout-headroom-selfcheck/state.json'
DRY_RUN_COMMAND = 'uv run --no-sync --project shared python scripts/timeout_headroom_selfcheck.py --dry-run'
ESCALATION_TIMEOUT_SECS = 30
_UNRESOLVED_EXAMPLES = 20

ToolCall = Callable[[str, dict[str, object]], Mapping[str, object]]
"""One MCP tool call: tool name and arguments in, the unwrapped result out."""


class FailureStage(StrEnum):
    """Where a run stopped before it could measure or act."""

    CONFIG = 'config'
    TREE = 'tree'
    ARCHIVE = 'archive'
    ESCALATION = 'escalation'
    UNEXPECTED = 'unexpected'


class SelfcheckFailure(Exception):
    """A stage that could not complete; everything before it is still reported."""

    def __init__(self, stage: FailureStage, detail: str) -> None:
        super().__init__(f'{stage}: {detail}')
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class Failure:
    stage: FailureStage
    detail: str

    @classmethod
    def of(cls, exc: Exception) -> Failure:
        """A :class:`SelfcheckFailure`'s own stage, else :attr:`FailureStage.UNEXPECTED`."""
        if isinstance(exc, SelfcheckFailure):
            return cls(exc.stage, exc.detail)
        return cls(FailureStage.UNEXPECTED, f'{type(exc).__name__}: {exc}')


@dataclass(frozen=True)
class RunConfig:
    """Which checkout's code, junit archive and stamp a run uses; ``dry_run`` files and stamps nothing."""

    project_root: Path
    archive_root: Path
    stamp_path: Path
    dry_run: bool = False

    @classmethod
    def for_project(cls, project_root: Path, *, dry_run: bool) -> RunConfig:
        return cls(project_root, project_root / ARCHIVE_PATH, project_root / STAMP_PATH, dry_run)


@dataclass(frozen=True)
class Ports:
    """Where a run writes: the escalation server."""

    escalation: ToolCall


@dataclass(frozen=True)
class LegFacts:
    """Every registered verify leg's budget, and the legs whose tracked tree holds a family member."""

    budgets: Mapping[str, int]
    family: frozenset[str]


@dataclass(frozen=True)
class CorpusTally:
    """How much of the archive the run read, and what it could not attribute."""

    runs: int
    rows_kept: int
    rows_below_floor: int
    read_floor_secs: float
    tree_sources: Mapping[str, int]
    skipped: tuple[tuple[str, str], ...]
    unresolved: int
    unresolved_examples: tuple[tuple[str, str], ...]

    def record(self) -> dict[str, Any]:
        record = asdict(self)
        record['tree_sources'] = dict(self.tree_sources)
        record['skipped'] = [{'path': path, 'reason': reason} for path, reason in self.skipped]
        record['unresolved_examples'] = [
            {'test_id': test_id, 'reason': reason} for test_id, reason in self.unresolved_examples
        ]
        return record


@dataclass(frozen=True)
class Report:
    """One night's measurement: every anchor and every registered leg, judged."""

    anchors: tuple[AnchorReport, ...]
    legs: tuple[LegReport, ...]
    corpus: CorpusTally

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            report.finding for report in (*self.anchors, *self.legs) if report.finding is not None
        )

    def record(self) -> dict[str, Any]:
        return {
            'anchors': [report.record() for report in self.anchors],
            'legs': [report.record() for report in self.legs],
            'corpus': self.corpus.record(),
        }


@dataclass(frozen=True)
class RunResult:
    """What a run measured, filed and skipped, or the stage it failed at."""

    measured_at: datetime
    dry_run: bool
    report: Report | None
    filed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failure: Failure | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None

    @property
    def findings(self) -> tuple[Finding, ...]:
        return () if self.report is None else self.report.findings

    def record(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'dry_run': self.dry_run,
            'measured_at': self.measured_at.isoformat(),
            'failure': None if self.failure is None else asdict(self.failure, dict_factory=_plain),
            'findings': [finding.record() for finding in self.findings],
            'filed': list(self.filed),
            'skipped': list(self.skipped),
            'report': None if self.report is None else self.report.record(),
        }

    def lines(self) -> list[str]:
        if self.report is None:
            return []
        lines = [_anchor_line(report) for report in self.report.anchors]
        lines += [_leg_line(report) for report in self.report.legs]
        corpus = self.report.corpus
        lines.append(
            f'read {corpus.runs} junit runs ({len(corpus.skipped)} unreadable), kept '
            f'{corpus.rows_kept} rows at >= {corpus.read_floor_secs}s, {corpus.unresolved} unresolved'
        )
        lines.append(f'filed: {list(self.filed)}; skipped (slot has history): {list(self.skipped)}')
        return lines


def _anchor_line(report: AnchorReport) -> str:
    worst = 'no member' if report.worst is None else f'worst {report.worst.seconds}s ({report.worst.test_id})'
    verdict = 'ok' if report.finding is None else f'BREACH {report.finding.slot()}'
    return (
        f'anchor {report.anchor_id}: ceiling {report.ceiling_secs}s admits {report.admissible_worst_secs}s; '
        f'{report.members} members, {worst}, {len(report.censored)} censored, '
        f'{len(report.timed_out)} timed out: {verdict}'
    )


def _leg_line(report: LegReport) -> str:
    worst = 'no green run' if report.worst_run is None else f'W {report.worst_run.seconds:.2f}s'
    return f'leg {report.prefix}: budget {report.budget_secs}s, {worst}, needs {report.required_budget_secs}: {report.status}'


def run(config: RunConfig, ports: Ports, now: datetime, *, legs: LegFacts) -> RunResult:
    """Measure the archive, judge it, and file every finding whose slot has no escalation history.

    Whatever stops a run, its result names the stage, and a real run stamps it.
    """
    try:
        result = _measure_and_file(config, ports, now, legs)
    except Exception as exc:
        result = RunResult(now, config.dry_run, None, failure=Failure.of(exc))
    return _stamped(config, result)


def _measure_and_file(config: RunConfig, ports: Ports, now: datetime, legs: LegFacts) -> RunResult:
    report = measure(config, legs)
    if config.dry_run:
        return RunResult(now, True, report)
    filed: list[str] = []
    skipped: list[str] = []
    failure = None
    try:
        for finding in report.findings:
            if _slot_has_history(ports.escalation, finding.slot()):
                skipped.append(finding.slot())
            else:
                _file(ports.escalation, finding, now)
                filed.append(finding.slot())
    except SelfcheckFailure as exc:
        failure = Failure.of(exc)
    return RunResult(now, False, report, tuple(filed), tuple(skipped), failure)


def measure(config: RunConfig, legs: LegFacts) -> Report:
    """Read the archive at each run's own tree and judge every anchor and leg; reads only."""
    if not config.archive_root.is_dir():
        raise SelfcheckFailure(FailureStage.ARCHIVE, f'{config.archive_root} is not a directory')
    anchors = registered_anchors()
    floor = min(min(anchor.measured_worst_secs, anchor.admissible_worst_secs) for anchor in anchors)
    head = _head_sha(config.project_root)
    prefixes = {sanitise_prefix(prefix): prefix for prefix in legs.budgets}
    try:
        with CaseMarks(config.project_root) as marks:
            corpus = read_runs(
                config.archive_root, min_seconds=floor, commit_exists=marks.commit_exists, head_sha=head
            )
            observations, unresolved = _observe(corpus, marks, prefixes)
    except OSError as exc:
        raise SelfcheckFailure(FailureStage.ARCHIVE, f'{type(exc).__name__}: {exc}') from exc
    worst_green = worst_green_legs(load_records([config.project_root]), prefixes)
    return Report(
        anchors=judge_anchors(anchors, observations),
        legs=judge_legs(legs.budgets, legs.family, worst_green),
        corpus=_tally(corpus, floor, unresolved),
    )


def _observe(
    corpus: JunitCorpus, marks: CaseMarks, prefixes: Mapping[str, str]
) -> tuple[tuple[Observation, ...], tuple[tuple[str, str], ...]]:
    observations: list[Observation] = []
    unresolved: list[tuple[str, str]] = []
    for junit_run in corpus.runs:
        leg_prefix = prefixes.get(junit_run.infix or '', junit_run.infix or '')
        for case in junit_run.cases:
            closest = marks.closest(junit_run.tree.sha, leg_prefix, case)
            if isinstance(closest, Unresolved):
                unresolved.append((f'{case.classname}::{case.name}', closest.reason))
            else:
                observations.append(Observation(junit_run, case, closest))
    return tuple(observations), tuple(unresolved)


def _tally(corpus: JunitCorpus, floor: float, unresolved: tuple[tuple[str, str], ...]) -> CorpusTally:
    return CorpusTally(
        runs=len(corpus.runs),
        rows_kept=sum(len(junit_run.cases) for junit_run in corpus.runs),
        rows_below_floor=sum(junit_run.rows_below_floor for junit_run in corpus.runs),
        read_floor_secs=floor,
        tree_sources=dict(Counter(junit_run.tree.source.value for junit_run in corpus.runs)),
        skipped=tuple((str(skip.path), skip.reason) for skip in corpus.skipped),
        unresolved=len(unresolved),
        unresolved_examples=unresolved[:_UNRESOLVED_EXAMPLES],
    )


def _head_sha(project_root: Path) -> str:
    try:
        return _git(project_root, 'rev-parse', 'HEAD').strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SelfcheckFailure(FailureStage.TREE, f'cannot read HEAD of {project_root}: {exc}') from exc


def _git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout


def _call(port: ToolCall, tool: str, args: dict[str, object]) -> Mapping[str, object]:
    try:
        answer = port(tool, args)
    except Exception as exc:
        raise SelfcheckFailure(FailureStage.ESCALATION, f'{tool} failed: {type(exc).__name__}: {exc}') from exc
    if not isinstance(answer, Mapping) or 'error' in answer:
        raise SelfcheckFailure(FailureStage.ESCALATION, f'{tool} answered {answer!r}')
    return answer


def _slot_has_history(escalation: ToolCall, slot: str) -> bool:
    answer = _call(escalation, 'get_task_escalations', {'task_id': slot, 'compact': True})
    records = answer.get('result')
    if not isinstance(records, list):
        raise SelfcheckFailure(FailureStage.ESCALATION, f'get_task_escalations answered {answer!r}')
    return bool(records)


@dataclass(frozen=True)
class _Body:
    summary: str
    remedy: str
    observation: str
    ref: str


def _body(finding: Finding) -> _Body:
    evidence = finding.evidence
    if isinstance(evidence, AnchorEvidence):
        row = evidence.worst_row
        return _Body(
            summary=(
                f'Per-test ceiling {evidence.anchor_id} ({evidence.ceiling_secs:g}s) no longer clears its '
                f'observed worst member {row.seconds}s x {evidence.headroom_factor}: needs '
                f'{evidence.required_ceiling_secs}s'
            ),
            remedy=(
                f'Re-size the ceiling to at least {evidence.required_ceiling_secs}s, or make {row.test_id} '
                f'cheaper; then re-anchor the recorded worst at {evidence.recorded_worst_home} with an addendum '
                'to plans/pytest-per-test-timeout-measurement-2026-09-17.md.'
            ),
            observation=(
                f'{row.test_id} took {row.seconds}s ({row.outcome}) in task {row.task_id} attempt {row.attempt} leg '
                f'{row.leg} at {row.tree_sha} ({row.tree_source})'
            ),
            ref=row.junit_path,
        )
    green = evidence.worst_run
    return _Body(
        summary=(
            f'Verify leg {evidence.prefix} budgets {evidence.budget_secs}s, below '
            f'min_leg_budget_for_family({green.seconds:.2f}s) = {evidence.required_budget_secs}s'
        ),
        remedy=(
            f"Raise {evidence.prefix}/orchestrator.yaml's verify_command_timeout_secs to at least "
            f'{evidence.required_budget_secs}, and record the new worst green run at '
            f'{evidence.recorded_worst_home}.'
        ),
        observation=(
            f'green test command took {green.seconds:.2f}s in task {green.task_id} attempt '
            f'{green.attempt} (role {green.role}): {green.cmd}'
        ),
        ref=green.summary_path,
    )


def _file(escalation: ToolCall, finding: Finding, now: datetime) -> None:
    body = _body(finding)
    detail = (
        f'{body.summary}.\n\nThis slot is filed once per kind, subject and bound; a re-size re-arms it. '
        f'Re-measure with `{DRY_RUN_COMMAND}`.\n\n'
        f'{json.dumps(finding.record(), indent=2, sort_keys=True)}'
    )
    args: dict[str, object] = {
        'task_id': finding.slot(),
        'agent_role': AGENT_ROLE,
        'category': 'risk_identified',
        'summary': body.summary,
        'detail': detail,
        'suggested_action': body.remedy,
        'evidence': [{'observation': body.observation, 'measured_at': now.isoformat(), 'ref': body.ref}],
    }
    _call(escalation, 'escalate_info', args)


def _stamped(config: RunConfig, result: RunResult) -> RunResult:
    if not config.dry_run:
        atomic_write_text(config.stamp_path, json.dumps(result.record(), indent=2, sort_keys=True) + '\n', mkdir=True)
    return result


# ---------------------------------------------------------------------------
# Adapters over the production config and the escalation server, and the CLI.


def registered_leg_budgets(project_root: Path) -> dict[str, int]:
    """Each registered module leg's ``verify_command_timeout_secs``, falling through to the root budget."""
    config = load_config(project_root / ORCHESTRATOR_CONFIG)
    root_budget = config.verify_command_timeout_secs
    return {
        prefix: int(module.verify_command_timeout_secs if module.verify_command_timeout_secs is not None else root_budget)
        for prefix, module in config.module_configs_or_empty.items()
    }


def family_legs(project_root: Path, prefixes: Iterable[str]) -> frozenset[str]:
    """The *prefixes* whose tracked tree holds a collectable whole-tree guard family member."""
    return frozenset(prefix for prefix in prefixes if _leg_tree_holds_a_member(project_root, prefix))


def _leg_tree_holds_a_member(project_root: Path, prefix: str) -> bool:
    listed = _git(project_root, 'ls-files', '-z', '--', f'{prefix}/*.py').split('\0')
    return any(
        classify_file(path, None) is FileKind.COLLECTABLE_TEST
        and is_whole_tree_guard_module(_source_or_none(project_root / path))
        for path in listed
        if path
    )


def _source_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def read_leg_facts(project_root: Path) -> LegFacts:
    """:func:`registered_leg_budgets` and :func:`family_legs`, or a CONFIG failure."""
    try:
        budgets = registered_leg_budgets(project_root)
        return LegFacts(budgets, family_legs(project_root, budgets))
    except Exception as exc:
        raise SelfcheckFailure(FailureStage.CONFIG, f'{type(exc).__name__}: {exc}') from exc


def escalation_url(project_root: Path) -> str:
    """The escalation server's MCP URL from *project_root*'s orchestrator config, or a CONFIG failure."""
    config = DashboardConfig(project_root=project_root)
    url = config.escalation_urls.get(config.project_root.name)
    if url is None:
        raise SelfcheckFailure(
            FailureStage.CONFIG, f'{config.project_root} configures no escalation server (escalation.port)'
        )
    return url


def _call_tool(base_url: str, tool: str, arguments: dict[str, object], *, timeout: float) -> dict:
    with SyncMcpSession(base_url, client_name=AGENT_ROLE, timeout=timeout) as session:
        return session.call_tool(tool, arguments)


def _no_escalation(tool: str, arguments: dict[str, object]) -> Mapping[str, object]:
    raise RuntimeError(f'a dry run calls no escalation tool, but {tool} was called')


def build_ports(project_root: Path, *, escalation_mcp_url: str | None) -> Ports:
    """The escalation server over the real transport; None takes the project's configured URL."""
    base = mcp_server_base(escalation_mcp_url or escalation_url(project_root))
    return Ports(escalation=functools.partial(_call_tool, base, timeout=ESCALATION_TIMEOUT_SECS))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Re-measure the per-test ceiling anchors and the family leg budgets against the verify archive.',
        epilog='Exit 0: the run completed and stamped its result. Exit 2: it could not measure or act.',
    )
    parser.add_argument('--project-root', type=Path, default=None, help="the main checkout (default: this checkout's)")
    parser.add_argument('--dry-run', action='store_true', help='measure and report; file and stamp nothing')
    parser.add_argument('--json', action='store_true', help='print the structured result instead of report lines')
    parser.add_argument(
        '--escalation-url', default=None, help="escalation MCP URL (default: from the project's orchestrator config)"
    )
    return parser


def _selfcheck(args: argparse.Namespace, now: datetime) -> RunResult:
    config = RunConfig.for_project(args.project_root or default_project_root(), dry_run=args.dry_run)
    try:
        legs = read_leg_facts(config.project_root)
        ports = (
            Ports(escalation=_no_escalation)
            if config.dry_run
            else build_ports(config.project_root, escalation_mcp_url=args.escalation_url)
        )
    except SelfcheckFailure as exc:
        return _stamped(config, RunResult(now, config.dry_run, None, failure=Failure.of(exc)))
    return run(config, ports, now, legs=legs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _selfcheck(args, datetime.now(UTC))
    except Exception as exc:
        print(f'timeout_headroom_selfcheck.py: could not run: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.record(), indent=2, sort_keys=True))
    else:
        for line in result.lines():
            print(line)
    if result.failure is not None:
        print(
            f'timeout_headroom_selfcheck.py: failed at {result.failure.stage}: {result.failure.detail}',
            file=sys.stderr,
        )
    return 0 if result.ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
