"""Acceptance criteria: the criteria file, the collection map, the junit join and the criteria events (PRD §7.7)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from client_feed.contract import (
    CaseOutcome,
    CaseResult,
    Criterion,
    CriterionOutcome,
    CriterionResult,
    UniqueCriteria,
    _first_error,
)
from client_feed.feed_io import FeedIoError, read_tail
from client_feed.sources.archive import case_id

CRITERIA_RELPATH: Final = 'acceptance/criteria.yaml'
COLLECTION_MAP_SUFFIX: Final = '.tests.json'

_RED: Final = frozenset({CaseOutcome.FAILED, CaseOutcome.ERRORED})


class CriteriaProblem(StrEnum):
    INVALID_CRITERIA = 'invalid_criteria'
    NOT_A_FILE = 'not_a_file'
    INVALID_MAP = 'invalid_map'
    INVALID_STAGING = 'invalid_staging'
    UNREADABLE = 'unreadable'


class CriteriaError(Exception):
    """A criteria file, collection map or staging file the criteria join refuses to read."""

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


def parse_criteria(text: str, *, name: str) -> tuple[Criterion, ...]:
    """The criteria a criteria file's text declares, in file order; `name` is the file named in a refusal."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _invalid_criteria(name, f'not YAML: {exc}') from exc
    if not isinstance(document, dict):
        raise _invalid_criteria(name, f'expected a mapping, got {type(document).__name__}')
    try:
        return CriteriaFile.model_validate(document).criteria
    except ValidationError as exc:
        raise _invalid_criteria(name, _first_error(exc)) from exc


def _unreadable(name: str, exc: FeedIoError | OSError) -> CriteriaError:
    detail = exc.detail if isinstance(exc, FeedIoError) else exc.strerror or type(exc).__name__
    return CriteriaError(problem=CriteriaProblem.UNREADABLE, name=name, detail=detail)


def collection_map_path(junit: Path) -> Path:
    """The collection map the defect-tests plugin writes beside `junit`; client_feed's copy of orchestrator verify.py::collection_map_path, which it may not import."""
    return junit.with_name(junit.name + COLLECTION_MAP_SUFFIX)


class MapEntry(BaseModel):
    """One collected test in the defect-tests collection map; `defects` is parsed here but read only by defect-tests ι."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    defects: tuple[str, ...]
    criterion: str | None


CollectionMap: TypeAlias = Mapping[str, MapEntry]
_COLLECTION_MAP: Final = TypeAdapter(dict[str, MapEntry])


def read_collection_map(path: Path, *, name: str) -> CollectionMap | None:
    """The collection map at `path`, never through a symlink; None when the run recorded none."""
    try:
        tail = read_tail(path, offset=0)
    except (FeedIoError, OSError) as exc:
        raise _unreadable(name, exc) from exc
    if tail is None:
        return None
    try:
        return _COLLECTION_MAP.validate_json(tail.data)
    except ValidationError as exc:
        raise CriteriaError(
            problem=CriteriaProblem.INVALID_MAP, name=name, detail=_first_error(exc)
        ) from exc


def _junit_address(nodeid: str) -> tuple[str, str]:
    """The junit (classname, name) pytest records for `nodeid`; restates _pytest/junitxml.py::mangle_test_address."""
    path, bracket, params = nodeid.partition('[')
    names = path.split('::')
    names[0] = names[0].replace('/', '.').removesuffix('.py')
    names[-1] += bracket + params
    return '.'.join(names[:-1]), names[-1]


def criterion_bindings(collection_map: CollectionMap) -> dict[str, str]:
    """Each marked test's junit case id and the criterion it evidences."""
    return {
        case_id(*_junit_address(nodeid)): entry.criterion
        for nodeid, entry in collection_map.items()
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
