"""Acceptance criteria: the criteria file, the collection map, the junit join and the criteria events (PRD §7.7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from client_feed.contract import Criterion, UniqueCriteria, _first_error

CRITERIA_RELPATH: Final = 'acceptance/criteria.yaml'


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
