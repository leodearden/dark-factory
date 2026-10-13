"""The merge-commit subject: one writer, and readers derived from the same fragments.

Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, invariant I1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    'MergeSubject',
    'format_merge_subject',
    'merge_subject_pattern',
    'parse_merge_subject',
    'rendered_prefix',
]

_HEAD = 'Merge '
_INTO = ' into '
_LEGACY_SEPARATOR = ': '
_REF_TOKEN = '[^ ]+'
_ERE_SPECIALS = frozenset('.[\\()*+?{|^$')


def _is_ref_token(text: str) -> bool:
    return bool(text) and not any(ch.isspace() for ch in text)


def _require_ref_token(field: str, value: str) -> None:
    if not _is_ref_token(value):
        raise ValueError(
            f'{field} must be a non-empty git ref name without whitespace '
            f'(git forbids whitespace in ref names); got {value!r}'
        )


@dataclass(frozen=True)
class MergeSubject:
    branch: str
    main_branch: str

    def __post_init__(self) -> None:
        _require_ref_token('branch', self.branch)
        _require_ref_token('main_branch', self.main_branch)


def format_merge_subject(branch: str, main_branch: str) -> str:
    subject = MergeSubject(branch, main_branch)
    return f'{_HEAD}{subject.branch}{_INTO}{subject.main_branch}'


_SUBJECT_RE = re.compile(re.escape(_HEAD) + f'({_REF_TOKEN})' + re.escape(_INTO) + f'({_REF_TOKEN})')


def parse_merge_subject(subject: str) -> MergeSubject | None:
    match = _SUBJECT_RE.fullmatch(subject)
    if match is None or not all(_is_ref_token(group) for group in match.groups()):
        return None
    return MergeSubject(*match.groups())


def _ere_escape(text: str) -> str:
    return ''.join('\\' + ch if ch in _ERE_SPECIALS else ch for ch in text)


def _escaped_ref(field: str, value: str) -> str:
    _require_ref_token(field, value)
    return _ere_escape(value)


def rendered_prefix(branch_prefix: str) -> tuple[str, str]:
    """ERE (head, into): `'^' + head + <escaped id> + into` is the line arm bash's grep -E takes."""
    if branch_prefix:
        _require_ref_token('branch_prefix', branch_prefix)
    return _ere_escape(_HEAD + branch_prefix), _ere_escape(_INTO)


def merge_subject_pattern(
    *,
    branch_prefix: str,
    task_id: str | None = None,
    main_branch: str | None = None,
    legacy_colon_form: bool = False,
) -> str:
    """One pattern valid as both git --extended-regexp and Python re; None means any ref token.

    The landing form is anchored to the whole line; the legacy colon form only to its
    prefix. Its ref token excludes only ' ', so a tab or a newline gets past it: in Python
    apply it to one git %s subject line, and read any other string with
    parse_merge_subject, the exact reader.
    """
    head, into = rendered_prefix(branch_prefix)
    task = _REF_TOKEN if task_id is None else _escaped_ref('task_id', task_id)
    main = _REF_TOKEN if main_branch is None else _escaped_ref('main_branch', main_branch)
    landing = f'{into}{main}$'
    if legacy_colon_form:
        landing = f'({landing}|{_ere_escape(_LEGACY_SEPARATOR)})'
    return f'^{head}{task}{landing}'
