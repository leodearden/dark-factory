"""The merge-commit subject: one writer, and readers derived from the same fragments.

Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, invariant I1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    'MergeSubject',
    'format_merge_subject',
    'parse_merge_subject',
]

_HEAD = 'Merge '
_INTO = ' into '
_LEGACY_SEPARATOR = ': '
_REF_TOKEN = '[^ ]+'


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
