"""Provenance of a task row's ``created_at``: the closed ``created_at_source`` vocabulary.

PRD ``plans/instruments-honesty-prd.md`` decision 3: ``tasks.created_at`` is
the filing time, and its sibling column ``created_at_source`` says how that
time was learned, so every age reader sees the proxy in the same row.

Two writers fill the pair, and each emits only its own members:

- ``SqliteTaskBackend.add_task``, the only INSERT into ``tasks``, emits
  :attr:`CreatedAtSource.STAMPED`: the instant the row was created.
- ``scripts/backfill_task_created_at.py`` fills rows created before the
  column existed, and emits the other three members. Each names a proxy
  that is an upper bound on the filing time, and the script keeps the
  earliest usable one: the task's earliest runs.db event, its un-back-filled
  ``metadata.pending_since``, or its ``updated_at``, an exact tie going in
  that order. :attr:`CreatedAtSource.UPDATED_AT` also labels the v4 -> v5
  migration's copy of ``updated_at`` held in a back-filled ``pending_since``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ['CreatedAtSource']


class CreatedAtSource(StrEnum):
    STAMPED = 'stamped'
    FIRST_EVENT = 'first_event'
    PENDING_SINCE = 'pending_since'
    UPDATED_AT = 'updated_at'
