"""The one reader of a task's declared ``metadata.delivered_checks``.

Reading rule: absent, ``None`` or any falsy value declares nothing. A list or tuple is split
element-wise, order kept: dicts go to ``checks``, everything else to ``unreadable``. Any other
truthy value (a bare dict, a str, an int) is one ``unreadable`` entry and is never iterated.

The split is on shape only; descriptor validation stays in
``orchestrator/src/orchestrator/delivered_checks.py::run_delivered_check``. A gate treats every
``unreadable`` entry as an unevaluable check (ERRORED: never inert, never delivered). Evidence and
lint readers read ``checks`` alone. What an author sees is in ``docs/task-authoring.md`` §3.3.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ['DELIVERED_CHECKS_METADATA_KEY', 'DeclaredChecks', 'declared_checks']

DELIVERED_CHECKS_METADATA_KEY = 'delivered_checks'


@dataclass(frozen=True)
class DeclaredChecks:
    checks: tuple[dict[str, Any], ...] = ()
    unreadable: tuple[object, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.checks and not self.unreadable


def declared_checks(metadata: Mapping[str, Any] | None) -> DeclaredChecks:
    value = (metadata or {}).get(DELIVERED_CHECKS_METADATA_KEY)
    if not value:
        return DeclaredChecks()
    if not isinstance(value, (list, tuple)):
        return DeclaredChecks(unreadable=(value,))
    checks: list[dict[str, Any]] = []
    unreadable: list[object] = []
    for entry in value:
        if isinstance(entry, dict):
            checks.append(entry)
        else:
            unreadable.append(entry)
    return DeclaredChecks(checks=tuple(checks), unreadable=tuple(unreadable))
