"""``metadata.early_start``: the task an early start stacked this task's branch on.

Contract: plans/early-start-on-predecessor-branch-prd.md §5.3 and §7. The mark names only the
predecessor; the commit the branch was stacked on lives in the stack-base ledger alone (invariant 8).
The schema lives in shared so fused-memory's write boundary validates the same class.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = ['EARLY_START_METADATA_KEY', 'EarlyStartMark', 'early_start_predecessor']

EARLY_START_METADATA_KEY = 'early_start'


class EarlyStartMark(BaseModel):
    model_config = ConfigDict(frozen=True, extra='allow')

    predecessor: str = Field(min_length=1)


def early_start_predecessor(metadata: Mapping[str, Any] | None) -> str | None:
    """The predecessor a well-formed mark names; None when the mark is absent or malformed."""
    if metadata is None:
        return None
    raw = metadata.get(EARLY_START_METADATA_KEY)
    if isinstance(raw, EarlyStartMark):
        return raw.predecessor
    if not isinstance(raw, Mapping):
        return None
    try:
        return EarlyStartMark.model_validate(raw).predecessor
    except ValidationError:
        return None
