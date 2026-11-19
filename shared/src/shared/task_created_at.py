"""Provenance of a task row's ``created_at``: the closed ``created_at_source`` vocabulary.

PRD ``plans/instruments-honesty-prd.md`` decision 3: ``tasks.created_at`` is
the filing time, and its sibling column ``created_at_source`` says how that
time was learned, so every age reader sees the proxy in the same row.

Two writers fill the pair, and each emits only its own members:

- ``SqliteTaskBackend.add_task``, the only INSERT into ``tasks``, emits
  :attr:`CreatedAtSource.STAMPED`: the instant the row was created.
- ``scripts/backfill_task_created_at.py`` fills rows created before the
  column existed, and emits the other three members, each naming the proxy
  the value was read from. Which proxy a row takes is the rule of
  ``scripts/backfill_task_created_at.py::choose_created_at``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ['CreatedAtSource']


class CreatedAtSource(StrEnum):
    STAMPED = 'stamped'
    FIRST_EVENT = 'first_event'
    PENDING_SINCE = 'pending_since'
    UPDATED_AT = 'updated_at'
