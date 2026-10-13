"""The one project-id rule, and the tolerant fold built on it for operator-typed tokens.

Contract: plans/review-all-s14-shared-leaves-prd.md (decision 4, invariant I2).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import PurePath
from types import MappingProxyType

__all__ = [
    'PROJECT_ID_ALIASES',
    'fold_project_token',
    'normalize_project_id',
    'project_id_for_root',
]

PROJECT_ID_ALIASES: Mapping[str, str] = MappingProxyType({'df': 'dark_factory'})
"""Folded token -> folded canonical id, applied last by fold_project_token.

Admit an entry only when its value is a real project's declared memory.project_id
and it heals an observed multi-bucket split (task 3813); see the PRD's decision 4.
"""

_UNDERSCORE_RUN = re.compile('_+')


def normalize_project_id(raw: str) -> str:
    return raw.lower().replace('-', '_')


def project_id_for_root(root: str | os.PathLike[str]) -> str:
    """normalize_project_id of the root's basename; ValueError when no basename names it."""
    basename = PurePath(os.fspath(root)).name
    if basename in ('', '..'):
        raise ValueError(
            f'project root {os.fspath(root)!r} has no basename naming a directory, so no '
            f'project id derives from it; pass the resolved project root'
        )
    return normalize_project_id(basename)


def fold_project_token(value: object) -> str:
    """normalize_project_id, made tolerant: '' for None/blank; strip, casefold, collapse '_'."""
    if value is None:
        return ''
    raw = str(value).strip()
    if not raw:
        return ''
    folded = _UNDERSCORE_RUN.sub('_', normalize_project_id(raw).casefold()).strip('_')
    return PROJECT_ID_ALIASES.get(folded, folded)
