"""``metadata.declared_scope``: the size a task declared when it was filed.

Contract: plans/follow-up-filing-shape-prd.md §4.2 and §3 decisions 6-8. ``submit_task`` stamps the
record once, ``basis='submit'``. ``basis='combine'`` is reserved for the curator-combine re-stamp
that task 8415 adds in ``_execute_combine``; nothing else may add, alter or remove it. An absent key
(a task filed before the stamp existed) means no information, never ``'undeclared'``. The modules
allowed to import this one are pinned by ``tests/repo_guards/test_declared_scope_readers.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    'DECLARED_SCOPE_METADATA_KEY',
    'DeclaredScope',
    'DeclaredScopeBand',
    'DeclaredScopeBasis',
    'StampViolation',
    'derive_declared_scope',
    'stamp_violation',
]

DECLARED_SCOPE_METADATA_KEY = 'declared_scope'

DeclaredScopeBand = Literal['undeclared', 'narrow', 'moderate', 'broad']
DeclaredScopeBasis = Literal['submit', 'combine']

V1_NARROW_MAX_FILES = 2
V1_MODERATE_MAX_FILES = 5
V1_TEXT_CHARS_LIMIT = 6000


def _v1_band(files: int, text_chars: int) -> DeclaredScopeBand:
    if files == 0:
        return 'undeclared'
    if files > V1_MODERATE_MAX_FILES:
        return 'broad'
    if files > V1_NARROW_MAX_FILES or text_chars >= V1_TEXT_CHARS_LIMIT:
        return 'moderate'
    return 'narrow'


class DeclaredScope(BaseModel):
    """One stamp: the band, the counts it was derived from, why it was stamped and under which rule."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    band: DeclaredScopeBand
    files: int = Field(ge=0)
    text_chars: int = Field(ge=0)
    basis: DeclaredScopeBasis
    rule: Literal['v1']

    @model_validator(mode='after')
    def _band_matches_counts(self) -> Self:
        derived = _v1_band(self.files, self.text_chars)
        if self.band != derived:
            raise ValueError(
                f'band {self.band!r} disagrees with files={self.files}, '
                f'text_chars={self.text_chars}, which rule v1 bands as {derived!r}'
            )
        return self


def derive_declared_scope(
    files: Sequence[str], description: str, details: str, *, basis: DeclaredScopeBasis,
) -> DeclaredScope:
    if isinstance(files, str):
        raise TypeError('files must be a sequence of paths, not a single str')
    text_chars = len(description) + len(details)
    return DeclaredScope(
        band=_v1_band(len(files), text_chars),
        files=len(files),
        text_chars=text_chars,
        basis=basis,
        rule='v1',
    )


class StampViolation(StrEnum):
    """How a write would break the stamp: the ``reason`` its refusal reports."""

    ADDED = 'added'
    ALTERED = 'altered'
    REMOVED = 'removed'


def stamp_violation(
    stored: Mapping[str, object], payload: Mapping[str, object], *, replace: bool,
) -> StampViolation | None:
    """How writing ``payload`` over ``stored`` would break the stamp; None when it stays exactly as it is.

    A payload that carries the key must carry the stored value; a replace must also carry it
    whenever one is stored. Identity is JSON value equality, as for ``metadata.deferral``.
    """
    absent = object()
    kept = stored.get(DECLARED_SCOPE_METADATA_KEY, absent)
    carried = payload.get(DECLARED_SCOPE_METADATA_KEY, absent)
    if carried is absent:
        return StampViolation.REMOVED if replace and kept is not absent else None
    if kept is absent:
        return StampViolation.ADDED
    return None if carried == kept else StampViolation.ALTERED
