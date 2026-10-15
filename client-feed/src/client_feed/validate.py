"""validate-feed: an out/ directory checked against the contract and the fold (PRD §7.8, B4, B5)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, computed_field

from client_feed.contract import (
    FEED_FILENAME,
    STATE_FILENAME,
    ContractViolation,
    FeedError,
    FeedLine,
    FeedState,
    UnsupportedFeedVersion,
    parse_feed_line,
    parse_state,
)
from client_feed.fold import CorruptFeed, ProjectMismatch, fold


class ProblemKind(StrEnum):
    MISSING_FILE = 'missing_file'
    CONTRACT_VIOLATION = 'contract_violation'
    UNSUPPORTED_VERSION = 'unsupported_version'
    CORRUPT_FEED = 'corrupt_feed'
    PROJECT_MISMATCH = 'project_mismatch'
    STATE_MISMATCH = 'state_mismatch'


class Problem(BaseModel):
    """One reason an out/ directory fails validation, located by feed line or seq when it has one."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: ProblemKind
    detail: str
    line: int | None = None
    seq: int | None = None


class ValidationReport(BaseModel):
    """The outcome of validate_out_dir, printed as JSON by the CLI."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    out_dir: str
    expected_project_id: str | None
    lines_checked: int
    watermark_seq: int | None
    problems: tuple[Problem, ...]

    @computed_field
    @property
    def ok(self) -> bool:
        return not self.problems


_PROBLEM_KINDS: Final[Mapping[type[FeedError], ProblemKind]] = {
    ContractViolation: ProblemKind.CONTRACT_VIOLATION,
    UnsupportedFeedVersion: ProblemKind.UNSUPPORTED_VERSION,
    CorruptFeed: ProblemKind.CORRUPT_FEED,
    ProjectMismatch: ProblemKind.PROJECT_MISMATCH,
}


class _Halted(Exception):
    """A phase found problems that make every later phase meaningless."""

    def __init__(self, *problems: Problem) -> None:
        super().__init__(problems)
        self.problems = problems


def _problem(error: FeedError, *, line: int | None = None, seq: int | None = None) -> Problem:
    return Problem(kind=_PROBLEM_KINDS[type(error)], detail=str(error), line=line, seq=seq)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise _Halted(
            Problem(kind=ProblemKind.MISSING_FILE, detail=f'{path.name} not found in {path.parent}')
        ) from exc
    except UnicodeDecodeError as exc:
        raise _Halted(
            Problem(kind=ProblemKind.CONTRACT_VIOLATION, detail=f'{path.name} is not UTF-8: {exc}')
        ) from exc


def _read_state(out_dir: Path) -> FeedState:
    try:
        return parse_state(_read_text(out_dir / STATE_FILENAME))
    except ContractViolation as error:
        raise _Halted(_problem(error)) from error


def _expected_project(state: FeedState, expected_project_id: str | None) -> str:
    if expected_project_id is None or expected_project_id == state.project_id:
        return state.project_id
    raise _Halted(
        Problem(
            kind=ProblemKind.PROJECT_MISMATCH,
            detail=f'{STATE_FILENAME} names project_id {state.project_id!r},'
            f' expected {expected_project_id!r}',
        )
    )


def _parse_lines(texts: list[str]) -> list[FeedLine]:
    lines: list[FeedLine] = []
    problems: list[Problem] = []
    for line_no, text in enumerate(texts, 1):
        try:
            lines.append(parse_feed_line(text, line_no=line_no))
        except (ContractViolation, UnsupportedFeedVersion) as error:
            problems.append(_problem(error, line=line_no))
    if problems:
        raise _Halted(*problems)
    return lines


def _checked_fold(lines: list[FeedLine], project_id: str, *, upto_seq: int | None) -> FeedState:
    try:
        return fold(lines, project_id=project_id, upto_seq=upto_seq)
    except (CorruptFeed, ProjectMismatch) as error:
        raise _Halted(_problem(error, seq=error.seq)) from error


def _compare_with_state(lines: list[FeedLine], project_id: str, state: FeedState) -> None:
    folded = _checked_fold(lines, project_id, upto_seq=state.watermark_seq)
    differing = [
        name for name in FeedState.model_fields if getattr(folded, name) != getattr(state, name)
    ]
    if differing:
        raise _Halted(
            Problem(
                kind=ProblemKind.STATE_MISMATCH,
                detail=f'{STATE_FILENAME} differs from the fold of {FEED_FILENAME} at'
                f' watermark_seq {state.watermark_seq} in: {", ".join(differing)}',
            )
        )


def validate_out_dir(out_dir: Path, *, expected_project_id: str | None = None) -> ValidationReport:
    """Check out_dir's feed and state against the contract and the fold; the first failing phase ends it."""
    lines_checked = 0
    watermark_seq: int | None = None
    problems: tuple[Problem, ...] = ()
    try:
        state = _read_state(out_dir)
        watermark_seq = state.watermark_seq
        project_id = _expected_project(state, expected_project_id)
        texts = _read_text(out_dir / FEED_FILENAME).splitlines()
        lines_checked = len(texts)
        lines = _parse_lines(texts)
        _checked_fold(lines, project_id, upto_seq=None)
        _compare_with_state(lines, project_id, state)
    except _Halted as halted:
        problems = halted.problems
    return ValidationReport(
        out_dir=str(out_dir),
        expected_project_id=expected_project_id,
        lines_checked=lines_checked,
        watermark_seq=watermark_seq,
        problems=problems,
    )
