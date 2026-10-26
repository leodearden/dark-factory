"""Normalise a reviewer artifact (verdict-tool envelope or flat payload) to a typed ReviewPayload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar


class Verdict(StrEnum):
    PASS = 'PASS'
    ISSUES_FOUND = 'ISSUES_FOUND'
    ERROR = 'ERROR'


class ReviewPayloadError(ValueError):
    pass


def _is_blocking(issue: Mapping[str, Any]) -> bool:
    return issue.get('severity') == 'blocking'


@dataclass(frozen=True)
class ReviewPayload:
    verdict: Verdict
    issues: tuple[dict[str, Any], ...]
    reviewer: str | None
    summary: str = ''

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer, str | None):
            raise ReviewPayloadError(
                f'review names its reviewer with a {type(self.reviewer).__name__}, '
                f'not a str or None: {self.reviewer!r}'
            )
        if not isinstance(self.summary, str):
            raise ReviewPayloadError(
                f'summary of the review from {self.reviewer!r} is a '
                f'{type(self.summary).__name__}, not a str: {self.summary!r}'
            )
        if self.verdict is Verdict.ERROR and self.issues:
            raise ReviewPayloadError(
                f'ERROR review from {self.reviewer!r} carries {len(self.issues)} '
                'issue(s); an ERROR review carries none'
            )
        for index, issue in enumerate(self.issues):
            if not isinstance(issue, dict):
                raise ReviewPayloadError(
                    f'issue {index} of the review from {self.reviewer!r} is a '
                    f'{type(issue).__name__}, not a dict: {issue!r}'
                )

    @property
    def blocking_issues(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue for issue in self.issues if _is_blocking(issue))

    @property
    def suggestions(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue for issue in self.issues if not _is_blocking(issue))

    def to_record(self) -> dict[str, Any]:
        return {
            'reviewer': self.reviewer,
            'verdict': self.verdict.value,
            'issues': list(self.issues),
            'summary': self.summary,
        }


def _inner_payload(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    if 'schema_version' not in artifact or 'role' not in artifact:
        return artifact
    payload = artifact.get('verdict')
    if not isinstance(payload, Mapping):
        raise ReviewPayloadError(
            f'verdict envelope for role {artifact["role"]!r} wraps a '
            f'{type(payload).__name__} verdict, not a payload dict: {payload!r}'
        )
    return payload


def _verdict(payload: Mapping[str, Any]) -> Verdict:
    value = payload.get('verdict')
    try:
        return Verdict(value)
    except ValueError:
        raise ReviewPayloadError(
            f'review from {payload.get("reviewer")!r} has verdict {value!r}; '
            f'expected one of {[v.value for v in Verdict]}'
        ) from None


def _issues(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = payload.get('issues')
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ReviewPayloadError(
            f'review from {payload.get("reviewer")!r} has issues {raw!r} '
            f'({type(raw).__name__}); expected a list'
        )
    return tuple(dict(issue) if isinstance(issue, dict) else issue for issue in raw)


def normalize_review(artifact: object) -> ReviewPayload:
    if not isinstance(artifact, Mapping):
        raise ReviewPayloadError(
            f'review artifact is a {type(artifact).__name__}, not a mapping: {artifact!r}'
        )
    payload = _inner_payload(artifact)
    verdict = _verdict(payload)
    issues = () if verdict is Verdict.ERROR else _issues(payload)
    summary = payload.get('summary')
    return ReviewPayload(
        verdict=verdict,
        issues=issues,
        reviewer=payload.get('reviewer'),
        summary='' if summary is None else summary,
    )
