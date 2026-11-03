"""Compare-and-swap and readback for task prose writes (task 4650).

Two defects on the ``update_task`` path: a ``description``/``details``
amendment could be reported as applied without the stored text ever being
read back, and two writers composing against the same read could silently
clobber each other. The fingerprint is the shared task-prose one; the append
and details-over-prompt semantics this module relies on live in
``backends/sqlite_task_backend.py::SqliteTaskBackend.update_task``.

A leaf ``middleware`` module: it imports nothing from ``reconciliation/`` or
``server/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from shared.task_text_fingerprint import task_text_fingerprint

TASK_TEXT_FP_KEY = 'task_text_fp'


def text_fp_of(task: object) -> str:
    if not isinstance(task, Mapping):
        return task_text_fingerprint(None, None)
    return task_text_fingerprint(task.get('description'), task.get('details'))


@dataclass(frozen=True)
class PlannedTextWrite:
    description: str | None
    details: str | None
    details_appends: bool

    @property
    def writes_text(self) -> bool:
        return self.description is not None or self.details is not None

    @property
    def replaces_text(self) -> bool:
        return self.description is not None or (
            self.details is not None and not self.details_appends
        )

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, payload in (('description', self.description), ('details', self.details))
            if payload is not None
        )


def plan_text_write(update_kwargs: Mapping[str, Any]) -> PlannedTextWrite:
    details = update_kwargs.get('details')
    return PlannedTextWrite(
        description=update_kwargs.get('description'),
        details=details if details is not None else update_kwargs.get('prompt'),
        details_appends=bool(update_kwargs.get('append')),
    )
