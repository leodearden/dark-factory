"""One verify run read back from the orchestrator's durable archive: its legs and its junit test cases."""

from __future__ import annotations

import gzip
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

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


def _case_id(classname: str, name: str) -> str:
    return f'{classname}::{name}' if classname else name


def _read_junit(root: Path, path: Path) -> list[CaseResult]:
    with gzip.open(path) as report:
        cases = ET.parse(report).getroot().iter('testcase')
        return [
            CaseResult(
                id=_case_id(case.get('classname', ''), case.get('name', '')),
                outcome=CaseOutcome.PASSED,
                duration_ms=None,
                criterion=None,
                exception_type=None,
            )
            for case in cases
        ]


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


def _tests(reports: Sequence[Sequence[CaseResult]], max_cases: int) -> ArchivedTests:
    cases = tuple(case for report in reports for case in report)
    return ArchivedTests(
        counts=CaseCounts(passed=len(cases), failed=0, skipped=0, errored=0, truncated=0),
        duration_ms=None,
        cases=cases[:max_cases],
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
