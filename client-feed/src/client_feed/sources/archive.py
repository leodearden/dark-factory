"""One verify run read back from the orchestrator's durable archive: its legs and its junit test cases."""

from __future__ import annotations

import gzip
import math
import os
import re
import stat
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, StrictInt, TypeAdapter, ValidationError

from client_feed.contract import (
    NOT_RECORDED,
    CaseCounts,
    CaseOutcome,
    CaseResult,
    NotRecorded,
    VerifyLeg,
)

ARCHIVE_RELPATH: Final = Path('data', 'verify-logs')

_STAMP_FORMAT: Final = '%Y%m%dT%H%M%S_%fZ'
_ARCHIVE_NAME: Final = re.compile(
    r'attempt-\d+(?P<qualifiers>(?:\.[^.]+)*?)\.(?P<kind>junit|summary|plan)'
    r'-(?P<stamp>\d{8}T\d{6}_\d{6}Z)\.(?P<extension>xml\.gz|json)'
)


class ArchiveFileKind(StrEnum):
    PLAN = 'plan'
    JUNIT = 'junit'
    SUMMARY = 'summary'


_EXTENSIONS: Final = {
    ArchiveFileKind.PLAN: 'json',
    ArchiveFileKind.JUNIT: 'xml.gz',
    ArchiveFileKind.SUMMARY: 'json',
}


class ArchiveProblem(StrEnum):
    UNREADABLE = 'unreadable'
    MALFORMED_JUNIT = 'malformed_junit'
    MALFORMED_SUMMARY = 'malformed_summary'
    MALFORMED_MAP = 'malformed_map'


class ArchiveReadError(Exception):
    """An archived file that exists but cannot be read; its str names the file relative to the root."""

    def __init__(self, *, relpath: Path, problem: ArchiveProblem) -> None:
        self.relpath = relpath
        self.problem = problem
        super().__init__(f'{problem} in {relpath}')


def _failure(root: Path, path: Path, problem: ArchiveProblem) -> ArchiveReadError:
    return ArchiveReadError(relpath=path.relative_to(root), problem=problem)


@dataclass(frozen=True)
class ArchiveFile:
    """An archived file named by the orchestrator's grammar; its qualifiers are opaque."""

    path: Path
    kind: ArchiveFileKind
    qualifiers: str
    stamp: datetime


def _archive_file(path: Path) -> ArchiveFile | None:
    match = _ARCHIVE_NAME.fullmatch(path.name)
    if match is None:
        return None
    kind = ArchiveFileKind(match['kind'])
    if match['extension'] != _EXTENSIONS[kind]:
        return None
    try:
        stamp = datetime.strptime(match['stamp'], _STAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return ArchiveFile(path=path, kind=kind, qualifiers=match['qualifiers'], stamp=stamp)


def _listed(root: Path, directory: Path) -> list[ArchiveFile]:
    """The regular files of `directory` that follow the grammar; a missing directory lists nothing."""
    try:
        with os.scandir(directory) as entries:
            paths = [Path(entry.path) for entry in entries if entry.is_file(follow_symlinks=False)]
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise _failure(root, directory, ArchiveProblem.UNREADABLE) from exc
    return [file for file in map(_archive_file, paths) if file is not None]


def _run_files(files: Sequence[ArchiveFile], ended_at: datetime) -> list[ArchiveFile]:
    """The junit and summary files of the run that ended at `ended_at`: since its newest plan, newest per name."""
    candidates = [file for file in files if file.stamp <= ended_at]
    start = max((file.stamp for file in candidates if file.kind is ArchiveFileKind.PLAN), default=None)
    newest: dict[tuple[ArchiveFileKind, str], ArchiveFile] = {}
    for file in sorted(candidates, key=lambda file: (file.stamp, file.path.name)):
        if file.kind is not ArchiveFileKind.PLAN and (start is None or file.stamp >= start):
            newest[(file.kind, file.qualifiers)] = file
    return sorted(newest.values(), key=lambda file: (file.stamp, file.qualifiers))


class _SummaryCommand(BaseModel):
    model_config = ConfigDict(frozen=True, extra='ignore')

    label: str
    rc: StrictInt


class _Summary(BaseModel):
    """The fields of an archived summary this reader uses; cmd and cause_hint are never read."""

    model_config = ConfigDict(frozen=True, extra='ignore')

    category: str
    commands: tuple[_SummaryCommand, ...]


def _read_summary(root: Path, path: Path) -> _Summary:
    try:
        return _Summary.model_validate_json(path.read_bytes())
    except OSError as exc:
        raise _failure(root, path, ArchiveProblem.UNREADABLE) from exc
    except ValidationError as exc:
        raise _failure(root, path, ArchiveProblem.MALFORMED_SUMMARY) from exc


def _legs(summaries: Sequence[_Summary]) -> tuple[VerifyLeg, ...] | NotRecorded:
    if not summaries:
        return NOT_RECORDED
    return tuple(
        VerifyLeg(name=command.label, category=summary.category, rc=command.rc)
        for summary in summaries
        for command in summary.commands
    )


_OUTCOME_TAGS: Final = {
    'failure': CaseOutcome.FAILED,
    'error': CaseOutcome.ERRORED,
    'skipped': CaseOutcome.SKIPPED,
}
_RED: Final = frozenset({CaseOutcome.FAILED, CaseOutcome.ERRORED})


def _case_id(classname: str, name: str) -> str:
    """The one spelling of a case id, `classname::name`, shared by the junit and map sides."""
    return f'{classname}::{name}' if classname else name


def _seconds(value: str | None) -> float | None:
    if value is None:
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f'a junit time must be a finite non-negative number, got {value!r}')
    return seconds


def _outcome(testcase: ET.Element) -> tuple[CaseOutcome, str | None]:
    """The first outcome child decides; a red case's kind is its `type` attribute, else its tag."""
    for child in testcase:
        outcome = _OUTCOME_TAGS.get(child.tag)
        if outcome is not None:
            return outcome, (child.get('type') or child.tag) if outcome in _RED else None
    return CaseOutcome.PASSED, None


def _junit_id_of_nodeid(nodeid: str) -> str:
    """A pytest nodeid in its junit spelling, by pytest's own junitxml mangle_test_address rule."""
    address, bracket, params = nodeid.partition('[')
    names = address.split('::')
    names[0] = names[0].replace('/', '.').removesuffix('.py')
    names[-1] += bracket + params
    return _case_id('.'.join(names[:-1]), names[-1])


class _MapEntry(BaseModel):
    """One nodeid of the defect plugin's collection map; its defects are not this reader's concern."""

    model_config = ConfigDict(frozen=True, extra='ignore')

    criterion: str | None


_MAP_ADAPTER: Final = TypeAdapter(dict[str, _MapEntry])
_JUNIT_SUFFIX: Final = f'.{_EXTENSIONS[ArchiveFileKind.JUNIT]}'
_MAP_SUFFIX: Final = '.tests.json'


def _criteria(root: Path, junit: Path) -> Mapping[str, str | None]:
    """The criterion per junit case id from the map written beside `junit`; no map binds nothing."""
    path = junit.with_name(junit.name.removesuffix(_JUNIT_SUFFIX) + _MAP_SUFFIX)
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return {}
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise _failure(root, path, ArchiveProblem.UNREADABLE) from exc
    try:
        entries = _MAP_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise _failure(root, path, ArchiveProblem.MALFORMED_MAP) from exc
    return {_junit_id_of_nodeid(nodeid): entry.criterion for nodeid, entry in entries.items()}


def _case(testcase: ET.Element, criteria: Mapping[str, str | None]) -> CaseResult:
    case_id = _case_id(testcase.get('classname', ''), testcase.get('name', ''))
    seconds = _seconds(testcase.get('time'))
    outcome, exception_type = _outcome(testcase)
    return CaseResult(
        id=case_id,
        outcome=outcome,
        duration_ms=None if seconds is None else round(seconds * 1000),
        criterion=criteria.get(case_id),
        exception_type=exception_type,
    )


@dataclass(frozen=True)
class _JunitReport:
    cases: tuple[CaseResult, ...]
    suite_seconds: tuple[float, ...]


def _parse_junit(report: gzip.GzipFile, criteria: Mapping[str, str | None]) -> _JunitReport:
    """Cases and suite times only: no message attribute and no element text is ever read."""
    cases: list[CaseResult] = []
    suite_seconds: list[float] = []
    for _, element in ET.iterparse(report, events=('end',)):
        if element.tag == 'testcase':
            cases.append(_case(element, criteria))
            element.clear()
        elif element.tag == 'testsuite' and (seconds := _seconds(element.get('time'))) is not None:
            suite_seconds.append(seconds)
    return _JunitReport(cases=tuple(cases), suite_seconds=tuple(suite_seconds))


def _read_junit(root: Path, path: Path) -> _JunitReport:
    criteria = _criteria(root, path)
    try:
        with gzip.open(path) as report:
            return _parse_junit(report, criteria)
    except (OSError, EOFError, zlib.error) as exc:
        raise _failure(root, path, ArchiveProblem.UNREADABLE) from exc
    except (ET.ParseError, ValueError) as exc:
        raise _failure(root, path, ArchiveProblem.MALFORMED_JUNIT) from exc


@dataclass(frozen=True)
class ArchivedTests:
    """A run's junit cases: counts over every case, the listing cut to the run's case cap."""

    counts: CaseCounts
    duration_ms: int | None
    cases: tuple[CaseResult, ...]


@dataclass(frozen=True)
class ArchivedRun:
    """What the archive recorded for one verify run; absent reports are not_recorded and None."""

    legs: tuple[VerifyLeg, ...] | NotRecorded
    tests: ArchivedTests | None


def _tests(reports: Sequence[_JunitReport], max_cases: int) -> ArchivedTests:
    """Counts over every case; the listing leads with red cases (report order kept) and is cut to the cap."""
    cases = [case for report in reports for case in report.cases]
    tally = Counter(case.outcome for case in cases)
    listed = tuple(sorted(cases, key=lambda case: case.outcome not in _RED)[:max_cases])
    seconds = [suite for report in reports for suite in report.suite_seconds]
    return ArchivedTests(
        counts=CaseCounts(
            passed=tally[CaseOutcome.PASSED],
            failed=tally[CaseOutcome.FAILED],
            skipped=tally[CaseOutcome.SKIPPED],
            errored=tally[CaseOutcome.ERRORED],
            truncated=len(cases) - len(listed),
        ),
        duration_ms=round(sum(seconds) * 1000) if seconds else None,
        cases=listed,
    )


def read_archived_run(
    root: Path, task_id: int, *, ended_at: datetime, max_cases: int
) -> ArchivedRun:
    """The legs and test cases the orchestrator archived for `task_id`'s verify run that ended at `ended_at`."""
    run = _run_files(_listed(root, root / ARCHIVE_RELPATH / str(task_id)), ended_at)
    summaries = [_read_summary(root, file.path) for file in run if file.kind is ArchiveFileKind.SUMMARY]
    reports = [_read_junit(root, file.path) for file in run if file.kind is ArchiveFileKind.JUNIT]
    return ArchivedRun(
        legs=_legs(summaries), tests=_tests(reports, max_cases) if reports else None
    )
