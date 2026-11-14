"""Render the context-complete (D2) reviewer briefing from structured inputs.

A leaf module: it imports nothing from the orchestrator, so the config, the
roles, the briefing assembler, ``orchestrator/review_inputs.py`` and the
reviewer trial all import it downward without a cycle.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ReviewBriefing(StrEnum):
    """The reviewer generation a dispatch uses.

    The values are the reviewer trial's ``--briefing`` vocabulary, so the
    config leaf ``review.reviewer_briefing`` and the CLI share one spelling.
    """

    DIFF_ONLY = 'diff-only'
    CONTEXT_COMPLETE = 'd2'


CONTEXT_COMPLETE_INLINE_DIFF_CHARS = 200_000
"""The largest diff inlined whole. A larger one is omitted wholesale, never cut."""


@dataclass(frozen=True)
class ChangedFile:
    """One changed path and its line counts; both counts are ``None`` for a binary file."""

    path: str
    added: int | None
    deleted: int | None


@dataclass(frozen=True)
class ReviewChange:
    """The change under review: ``<base_sha>...HEAD``, its diff and every path it touches."""

    base_sha: str
    diff: str
    files: tuple[ChangedFile, ...]


@dataclass(frozen=True)
class ReviewConvention:
    """A project convention the reviewer judges the change against."""

    rule: str
    why: str


@dataclass(frozen=True)
class ReviewSubject:
    """Everything the context-complete reviewer is briefed with, besides memory.

    ``task`` and ``plan`` are read-only deep copies taken at construction, so
    every reviewer briefed from one subject sees the records as they were then,
    whatever the caller does to its own copies afterwards.
    """

    task: Mapping[str, object]
    plan: Mapping[str, object] | None
    plan_absent_reason: str
    change: ReviewChange
    conventions: tuple[ReviewConvention, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'task', _snapshot(self.task))
        if self.plan is not None:
            object.__setattr__(self, 'plan', _snapshot(self.plan))


def _snapshot(record: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(copy.deepcopy(dict(record)))


_PLAN_FIELDS = ('title', 'analysis', 'files', 'design_decisions', 'prerequisites', 'steps')

_SCOPE_SECTION = """\
# Scope Check

Flag changed files that do not belong to this task's stated work (the task
record and, when present, the plan's `files`). Report them as `suggestion`,
never `blocking`."""

_CONTEXT_COMPLETE_REVIEW_STEP = (
    'Review the change according to your specialization: read each changed file '
    'whole, and explore the codebase as needed for context.'
)


def review_action_section(reviewer_type: str, review_step: str) -> str:
    """The ``# Action`` section every reviewer briefing ends its body with."""
    return f"""\
# Action

1. {review_step}
2. Call `submit_review_verdict(reviewer="{reviewer_type}", verdict=..., issues=..., summary=...)` with your findings.

Your verdict is read from the `submit_review_verdict` tool call, not from your prose output — you MUST call it before finishing.
"""


def amendment_scope_section(suggestions: Sequence[Mapping[str, object]]) -> str:
    """The advisory section for a review that follows an amendment round (task 2750)."""
    listed = '\n'.join(
        f"- {s.get('location') or '(no location)'} — "
        f"{s.get('description') or ''}"
        for s in suggestions
    )
    return f"""\
# Amendment Re-Review Scope

This diff was just amended to address the prior review's in-scope suggestions,
listed below. Do NOT hunt for fresh, unrelated nits across the full diff:
first confirm the listed prior suggestions were addressed, then report ONLY
new findings introduced within the amendment delta — the lines this amendment
actually changed. Blocking regressions anywhere in the diff remain in scope.

Prior suggestions the amendment was asked to address:
{listed}
"""


def render_context_complete_review(
    *,
    reviewer_type: str,
    subject: ReviewSubject,
    context: str,
    task_block: str,
    amendment_suggestions: Sequence[Mapping[str, object]] | None,
) -> str:
    """The context-complete reviewer's user prompt."""
    sections = (
        context,
        f'# Task Under Review\n\n{task_block}',
        _plan_section(subject.plan, subject.plan_absent_reason),
        _change_section(subject.change),
        _SCOPE_SECTION,
        _conventions_section(subject.conventions),
        _diff_section(subject.change.diff),
        review_action_section(reviewer_type, _CONTEXT_COMPLETE_REVIEW_STEP).rstrip('\n'),
    )
    prompt = '\n\n'.join(section for section in sections if section) + '\n'
    if amendment_suggestions:
        prompt += '\n' + amendment_scope_section(amendment_suggestions)
    return prompt


def _plan_section(plan: Mapping[str, object] | None, absent_reason: str) -> str:
    if plan is None:
        reason = absent_reason or 'no reason was recorded'
        return (
            f'# Plan\n\nNo plan is available for this change: {reason}. '
            'Judge the change against the task record above.'
        )
    plan_json = json.dumps(
        {field: plan[field] for field in _PLAN_FIELDS if field in plan}, indent=2, ensure_ascii=False,
    )
    return f'# Plan\n\n```json\n{plan_json}\n```'


def _change_section(change: ReviewChange) -> str:
    listing = '\n'.join(_file_line(changed) for changed in change.files) or '(no files changed)'
    return f"""\
# Change Under Review

Range: `{change.base_sha}...HEAD`. These {len(change.files)} changed file(s) are the complete list:

{listing}

Read each changed file WHOLE in this checkout, not only its hunks. For one
file's own diff, run `git diff {change.base_sha}...HEAD -- <path>`."""


def _file_line(changed: ChangedFile) -> str:
    if changed.added is None or changed.deleted is None:
        return f'- `{changed.path}` (binary)'
    return f'- `{changed.path}` (+{changed.added} -{changed.deleted})'


def _conventions_section(conventions: tuple[ReviewConvention, ...]) -> str:
    if not conventions:
        return ''
    listed = '\n'.join(f'- {c.rule} — {c.why}' for c in conventions)
    return f'# Project Conventions\n\n{listed}'


def _diff_section(diff: str) -> str:
    if len(diff) > CONTEXT_COMPLETE_INLINE_DIFF_CHARS:
        return (
            f'# Code Diff\n\nThe full diff is {len(diff):,} characters, over the '
            f'{CONTEXT_COMPLETE_INLINE_DIFF_CHARS:,}-character inline limit, so it is '
            'omitted rather than cut. Use the per-file command above for each changed file.'
        )
    return f'# Code Diff\n\n```diff\n{diff}\n```'
