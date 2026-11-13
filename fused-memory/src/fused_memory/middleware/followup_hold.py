"""The follow-up holder marker and its predicate (plans/follow-up-filing-shape-prd.md §4.3, decision 10).

Task 8417 (ε) extends this module with the holder model, mint, hold and backfill. curator_types.py
imports it, so it must keep importing only the stdlib and lower layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from shared.task_metadata_wire import coerce_task_metadata

__all__ = ['FOLLOWUP_HOLDER_METADATA_KEY', 'is_holder']

FOLLOWUP_HOLDER_METADATA_KEY: Final = 'followup_holder'


def is_holder(task: Mapping[str, object]) -> bool:
    metadata = coerce_task_metadata(task.get('metadata'))
    return metadata is not None and metadata.get(FOLLOWUP_HOLDER_METADATA_KEY) is not None
