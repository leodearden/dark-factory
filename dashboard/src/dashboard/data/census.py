"""The task census — what the nine task statuses mean to the dashboard.

Declared by ``plans/dashboard-one-datum-one-path-prd.md``, section "The task
census" (decision 3). This module is the dashboard's SINGLE home for the two
facts every surface needs about a status: which named view it belongs to, and
which tone it draws in. The vocabulary itself is not restated here — it is
imported from ``shared.task_statuses``, the one home the factory already has
for it, so a tenth member appears in every count, view and tone at once.

Why ``in_flight`` and ``backlog`` are ENUMERATED rather than derived
--------------------------------------------------------------------
``terminal`` is bound to ``shared.task_statuses.TERMINAL`` by reference,
because that name is already the authority for the concept. The other two are
written out member by member, deliberately: a derived ``in_flight = ACTIVE -
backlog`` would make the partition hold BY CONSTRUCTION and so silently
absorb a future tenth status into in-flight. Enumerated, that member falls
outside all three views and the partition test fails loudly — which is the
only way the PRD's "a test asserts the three views partition ``TaskStatus``"
actually bites.

``running`` is kept in a separate ``SUB_VIEWS`` rather than added to
``VIEWS`` so that ``sum(views.values()) == total`` holds on the wire and no
consumer can re-add the sub-view into the partition.

``TONES`` maps each status to a ``charts.jsx::PALETTE`` tone KEY, never to a
colour literal: the palette stays owned by ``charts.jsx`` alone, following the
injection convention ``burndown_bands.js`` states at length.

This module reads no clock and performs no I/O.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shared.task_statuses import TERMINAL, TaskStatus


class TaskView(enum.StrEnum):
    """The named views the dashboard groups task statuses into.

    ``IN_FLIGHT``/``BACKLOG``/``TERMINAL`` partition ``TaskStatus``;
    ``RUNNING`` is a sub-view narrowing ``IN_FLIGHT``.
    """

    IN_FLIGHT = 'in_flight'
    BACKLOG = 'backlog'
    TERMINAL = 'terminal'
    RUNNING = 'running'


VIEWS: MappingProxyType[TaskView, frozenset[TaskStatus]] = MappingProxyType(
    {
        TaskView.IN_FLIGHT: frozenset(
            {
                TaskStatus.IN_PROGRESS,
                TaskStatus.BLOCKED,
                TaskStatus.MERGE_DEFERRED,
                TaskStatus.REVIEW,
                TaskStatus.INFRA_HOLD,
            }
        ),
        TaskView.BACKLOG: frozenset({TaskStatus.PENDING, TaskStatus.DEFERRED}),
        TaskView.TERMINAL: TERMINAL,
    }
)

SUB_VIEWS: MappingProxyType[TaskView, frozenset[TaskStatus]] = MappingProxyType(
    {TaskView.RUNNING: frozenset({TaskStatus.IN_PROGRESS})}
)

TONES: MappingProxyType[TaskStatus, str] = MappingProxyType(
    {
        TaskStatus.IN_PROGRESS: 'accent',
        TaskStatus.BLOCKED: 'bad',
        TaskStatus.MERGE_DEFERRED: 'warn',
        TaskStatus.REVIEW: 'info',
        TaskStatus.INFRA_HOLD: 'stranded',
        TaskStatus.PENDING: 'warn',
        TaskStatus.DEFERRED: 'fg3',
        TaskStatus.DONE: 'ok',
        TaskStatus.CANCELLED: 'fg3',
    }
)


@dataclass(frozen=True, slots=True)
class TaskCensus:
    """How many tasks sit in each status, and in each named view of them.

    Attributes:
        counts: Every ``TaskStatus`` member, always all nine keys — a status
            nobody is in is present at zero rather than absent, so no consumer
            has to distinguish "none" from "not reported".
        total: ``sum(counts.values())``.
        views: The three-view partition; sums to ``total``.
        sub_views: The narrowing views, currently ``running`` alone. Kept
            apart from ``views`` so the partition's sum stays honest.
    """

    counts: Mapping[TaskStatus, int]
    total: int
    views: Mapping[TaskView, int]
    sub_views: Mapping[TaskView, int]


def build_census(status_map: Mapping[Any, str]) -> TaskCensus:
    """Tally *status_map* into a :class:`TaskCensus`.

    Pure: reads no clock, performs no I/O, and mutates nothing it was handed.

    Args:
        status_map: ``{task id: status value}``, the shape
            ``tasks.py::fetch_statuses`` returns. Ids are carried only so an
            off-vocabulary value can name the row that held it, so the key
            type is deliberately unconstrained. It is spelled ``Any`` rather
            than ``object`` because ``Mapping``'s key parameter is INVARIANT:
            ``Mapping[object, str]`` would reject the very ``dict[int, str]``
            this docstring describes.

    Returns:
        A census whose ``counts`` carry all nine members, seeded at zero.
    """
    counts = dict.fromkeys(TaskStatus, 0)
    for status in status_map.values():
        counts[TaskStatus(status)] += 1

    def tally(members: frozenset[TaskStatus]) -> int:
        return sum(counts[member] for member in members)

    return TaskCensus(
        counts=MappingProxyType(counts),
        total=sum(counts.values()),
        views=MappingProxyType({view: tally(members) for view, members in VIEWS.items()}),
        sub_views=MappingProxyType(
            {view: tally(members) for view, members in SUB_VIEWS.items()}
        ),
    )
