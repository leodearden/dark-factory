"""Acceptance criteria: the criteria file, its join with junit reports through their collection maps, and the criteria events (PRD §7.7)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from client_feed.contract import (
    CaseOutcome,
    CaseResult,
    CommitSha,
    CriteriaDeclared,
    CriteriaResults,
    Criterion,
    CriterionOutcome,
    CriterionResult,
    FeedState,
    RunKind,
    UniqueCriteria,
    UtcTimestamp,
    first_error,
)
from client_feed.envelope import Stamp, payload
from client_feed.feed_io import FeedIoError, read_tail, write_atomically
from client_feed.redact import redact_paths
from client_feed.sources.archive import read_junit_cases
from client_feed.sources.collection_map import (
    CollectionMap,
    case_entries,
    collection_map_path,
    read_collection_map,
)
from client_feed.sources.git import EntryKind, blob_reader, path_entry

CRITERIA_RELPATH: Final = 'acceptance/criteria.yaml'
STAGING_FILENAME: Final = 'criteria-staging.json'

_RED: Final = frozenset({CaseOutcome.FAILED, CaseOutcome.ERRORED})


class CriteriaProblem(StrEnum):
    INVALID_CRITERIA = 'invalid_criteria'
    NOT_A_FILE = 'not_a_file'
    INVALID_STAGING = 'invalid_staging'
    UNREADABLE = 'unreadable'
    UNWRITABLE = 'unwritable'


class CriteriaError(Exception):
    """A criteria file or staging file the criteria join refuses to read, or a staging file it cannot write."""

    def __init__(self, *, problem: CriteriaProblem, name: str, detail: str) -> None:
        self.problem = problem
        self.name = name
        self.detail = detail
        super().__init__(f'{name} refused ({problem}): {detail}')


class CriteriaFile(BaseModel):
    """acceptance/criteria.yaml: the project's acceptance criteria, in declaration order."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    format: Literal[1]
    criteria: UniqueCriteria


def _invalid_criteria(name: str, detail: str) -> CriteriaError:
    return CriteriaError(problem=CriteriaProblem.INVALID_CRITERIA, name=name, detail=detail)


def parse_criteria(content: bytes, *, name: str) -> tuple[Criterion, ...]:
    """The criteria a criteria file's bytes declare, in file order; `name` is the file named in a refusal."""
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise _invalid_criteria(name, f'not UTF-8: {exc}') from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _invalid_criteria(name, f'not YAML: {exc}') from exc
    if not isinstance(document, dict):
        raise _invalid_criteria(name, f'expected a mapping, got {type(document).__name__}')
    try:
        return CriteriaFile.model_validate(document).criteria
    except ValidationError as exc:
        raise _invalid_criteria(name, first_error(exc, whole='<file>')) from exc


def read_declared(root: Path, head: str) -> tuple[Criterion, ...]:
    """The criteria acceptance/criteria.yaml declares at `head`, read from git objects, never the working tree; () when it holds none."""
    entry = path_entry(root, head, CRITERIA_RELPATH)
    if entry is None:
        return ()
    if entry.kind is not EntryKind.FILE:
        raise CriteriaError(
            problem=CriteriaProblem.NOT_A_FILE,
            name=CRITERIA_RELPATH,
            detail=f'the commit holds a {entry.kind} there, not a file',
        )
    with blob_reader(root) as read:
        content = read(entry.object_sha)
    return parse_criteria(content, name=CRITERIA_RELPATH)


def _unreadable(name: str, exc: FeedIoError | OSError) -> CriteriaError:
    detail = exc.detail if isinstance(exc, FeedIoError) else exc.strerror or type(exc).__name__
    return CriteriaError(problem=CriteriaProblem.UNREADABLE, name=name, detail=detail)


def criterion_bindings(collection_map: CollectionMap) -> dict[str, str]:
    """Each marked test's junit case id and the criterion it evidences."""
    return {
        test: entry.criterion
        for test, entry in case_entries(collection_map)
        if entry.criterion is not None
    }


@dataclass(frozen=True)
class BoundReport:
    """One junit report's cases and the criterion each of its marked tests evidences, keyed by case id."""

    cases: tuple[CaseResult, ...]
    criteria: Mapping[str, str]


@dataclass(frozen=True)
class _Evidence:
    """One bound test and the outcomes of its cases in the report that binds it."""

    test: str
    outcomes: tuple[CaseOutcome, ...]


def _report_evidence(report: BoundReport) -> list[tuple[str, _Evidence]]:
    outcomes: dict[str, list[CaseOutcome]] = defaultdict(list)
    for case in report.cases:
        outcomes[case.id].append(case.outcome)
    return [
        (criterion, _Evidence(test=test, outcomes=tuple(outcomes.get(test, ()))))
        for test, criterion in report.criteria.items()
    ]


def _evidence(reports: Sequence[BoundReport]) -> dict[str, list[_Evidence]]:
    """Each criterion's evidence, every report read on its own so a case never borrows another report's binding."""
    by_criterion: dict[str, list[_Evidence]] = defaultdict(list)
    for report in reports:
        for criterion, evidence in _report_evidence(report):
            by_criterion[criterion].append(evidence)
    return by_criterion


def _passed(evidence: _Evidence) -> bool:
    return bool(evidence.outcomes) and all(
        outcome is CaseOutcome.PASSED for outcome in evidence.outcomes
    )


def _outcome(evidence: Sequence[_Evidence]) -> CriterionOutcome:
    """Failed on any red bound case; passed only when every bound test ran and passed; otherwise not run."""
    if any(outcome in _RED for item in evidence for outcome in item.outcomes):
        return CriterionOutcome.FAILED
    if evidence and all(map(_passed, evidence)):
        return CriterionOutcome.PASSED
    return CriterionOutcome.NOT_RUN


def join_criteria(
    declared: Sequence[Criterion], reports: Sequence[BoundReport]
) -> tuple[CriterionResult, ...]:
    """Each declared criterion's outcome over the reports' bound cases, in declaration order."""
    evidence = _evidence(reports)
    return tuple(
        CriterionResult(
            id=criterion.id,
            outcome=_outcome(evidence.get(criterion.id, ())),
            tests=tuple(sorted({item.test for item in evidence.get(criterion.id, ())})),
        )
        for criterion in declared
    )


class CriteriaRun(BaseModel):
    """What one run says about the criteria: criteria_results' payload less run_kind, and out/criteria-staging.json's shape."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    commit: CommitSha | None
    run_at: UtcTimestamp
    results: tuple[CriterionResult, ...]


def _bound_report(root: Path, junit: Path) -> BoundReport | None:
    map_relpath = collection_map_path(junit)
    collection_map = read_collection_map(root / map_relpath, name=str(map_relpath))
    if collection_map is None:
        return None
    return BoundReport(
        cases=read_junit_cases(root, junit), criteria=criterion_bindings(collection_map)
    )


def verify_criteria(
    root: Path,
    junit_relpaths: Sequence[Path],
    *,
    declared: Sequence[Criterion],
    commit: str | None,
    run_at: datetime,
) -> CriteriaRun | None:
    """The criteria a verify run's archived junits evidence, each joined with the map beside it; None when no junit has one."""
    reports = [
        report for junit in junit_relpaths if (report := _bound_report(root, junit)) is not None
    ]
    if not reports:
        return None
    return CriteriaRun(commit=commit, run_at=run_at, results=join_criteria(declared, reports))


def _redacted(criterion: Criterion) -> Criterion:
    return criterion.model_copy(
        update={
            'statement': redact_paths(criterion.statement),
            'capability': redact_paths(criterion.capability),
        }
    )


def declared_event(
    criteria: Sequence[Criterion], *, state: FeedState, stamp: Stamp, now: datetime
) -> CriteriaDeclared | None:
    """The declaration at `now` with its prose paths redacted; None when the state already declares exactly it."""
    redacted = tuple(map(_redacted, criteria))
    if {criterion.id: criterion for criterion in redacted} == state.criteria_declared:
        return None
    return CriteriaDeclared(**stamp(now), criteria=redacted)


def results_event(
    kind: RunKind, run: CriteriaRun, *, state: FeedState, stamp: Stamp
) -> CriteriaResults | None:
    """The run's criteria_results of `kind` at its run_at; None when the state's record of `kind` already says it."""
    said = {'run_kind': kind, **dict(run)}
    recorded = state.criteria.get(kind)
    if recorded is not None and payload(recorded) == said:
        return None
    return CriteriaResults(**stamp(run.run_at), **said)


def write_staging(out_dir: Path, run: CriteriaRun) -> None:
    """Replace out_dir's staging file by `run`, atomically and never through a symlink."""
    try:
        write_atomically(out_dir, STAGING_FILENAME, run.model_dump_json())
    except OSError as exc:
        raise CriteriaError(
            problem=CriteriaProblem.UNWRITABLE,
            name=STAGING_FILENAME,
            detail=exc.strerror or type(exc).__name__,
        ) from exc


def read_staging(out_dir: Path) -> CriteriaRun | None:
    """The staging run out_dir's staging file records, never read through a symlink; None when there is none."""
    try:
        tail = read_tail(out_dir / STAGING_FILENAME, offset=0)
    except (FeedIoError, OSError) as exc:
        raise _unreadable(STAGING_FILENAME, exc) from exc
    if tail is None:
        return None
    try:
        return CriteriaRun.model_validate_json(tail.data)
    except ValidationError as exc:
        raise CriteriaError(
            problem=CriteriaProblem.INVALID_STAGING,
            name=STAGING_FILENAME,
            detail=first_error(exc, whole='<file>'),
        ) from exc


def _read_criteria_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _unreadable(str(path), exc) from exc


def staging_run(
    *, junit: Path, collection_map: Path, criteria_file: Path, commit: str, now: datetime
) -> CriteriaRun:
    """A staging run's criteria: its junit report joined, through the collection map beside it, with the criteria file."""
    declared = parse_criteria(_read_criteria_file(criteria_file), name=str(criteria_file))
    cases = read_junit_cases(junit.parent, Path(junit.name))
    read = read_collection_map(collection_map, name=str(collection_map))
    if read is None:
        raise CriteriaError(
            problem=CriteriaProblem.UNREADABLE, name=str(collection_map), detail='no such file'
        )
    report = BoundReport(cases=cases, criteria=criterion_bindings(read))
    return CriteriaRun(commit=commit, run_at=now, results=join_criteria(declared, [report]))
