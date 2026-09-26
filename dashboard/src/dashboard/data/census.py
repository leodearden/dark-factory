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
from collections.abc import Mapping, Sequence
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


class CensusVocabularyError(ValueError):
    """A status map carried a value outside ``TaskStatus``.

    Raised rather than absorbed, deliberately. Dropping the row would
    under-report ``total`` silently — the class of uniform lie this module
    exists to remove — and a tenth bucket would break the nine-key contract
    every consumer is being written against, falsifying ``total ==
    sum(counts.values())`` either way.

    Being pure, :func:`build_census` cannot degrade gracefully itself, and it
    does not need to: the access layer above wraps the call, catches this, and
    serves the census as an ``unknown``/``stale``
    :class:`~dashboard.data.datum.Datum` carrying this message verbatim as its
    ``reason``. That is what makes the failure visible instead of fatal —
    and because that message then crosses the wire on every poll, its SIZE is
    bounded by construction (see :func:`_vocabulary_drift_message`).

    The store's vocabulary is already closed by
    ``fused_memory/server/tools.py::_VALID_TASK_STATUSES``, which imports from
    the same ``shared`` module, so an unknown value here means the vocabulary
    drifted and must be loud.
    """


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

    def to_wire(self) -> dict[str, object]:
        """Render the four contract keys with plain-string keys throughout.

        This is the one place the census wire shape is known, so a consumer
        needs neither the Python enums nor a custom JSON encoder to read it.
        """
        return {
            'counts': {member.value: count for member, count in self.counts.items()},
            'total': self.total,
            'views': {view.value: count for view, count in self.views.items()},
            'sub_views': {view.value: count for view, count in self.sub_views.items()},
        }


REPORTED_VALUES = 5
"""How many distinct off-vocabulary values the error message spells out."""


def _vocabulary_drift_message(offenders: Sequence[tuple[Any, str]]) -> str:
    """Summarise off-vocabulary rows in a message of BOUNDED size.

    Grouped by value and capped at :data:`REPORTED_VALUES`, deliberately. The
    realistic drifts make EVERY row an offender at once — a tenth status
    shipping upstream, or ``fetch_statuses`` changing the shape of its values
    — so a per-row enumeration would grow with the task tree. This message is
    served verbatim to the browser as a Datum's ``reason``, which would turn
    a diagnostic into a payload-degradation event on every poll of every
    project. A thousand rows sharing one unknown status carry one bit of
    information: the distinct VALUES are the diagnosis, and the two counts —
    rows and distinct values, both exact — say how far the drift reaches.

    Values are grouped by their ``repr`` rather than by themselves because a
    shape drift can hand us an UNHASHABLE value (``{'status': ...}``), which
    is one of the very drifts this reports. The repr is also exactly what the
    message prints, so the grouping and the rendering cannot disagree.
    """
    by_value: dict[str, list[Any]] = {}
    for task_id, status in offenders:
        by_value.setdefault(repr(status), []).append(task_id)

    shown = list(by_value.items())[:REPORTED_VALUES]
    listed = ', '.join(
        f'{value} ({len(ids)} row(s), first id={ids[0]!r})' for value, ids in shown
    )
    if len(by_value) > len(shown):
        listed += f', (+{len(by_value) - len(shown)} more distinct value(s))'

    legal = ', '.join(repr(member.value) for member in TaskStatus)
    return (
        f'status map carried {len(offenders)} row(s) outside TaskStatus '
        f'across {len(by_value)} distinct value(s): {listed}. '
        f'Legal members are: {legal}'
    )


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

    Raises:
        CensusVocabularyError: If any value falls outside ``TaskStatus``.
            Every distinct offending VALUE is reported in one raise, so a
            vocabulary drift is not found one rerun at a time, and the
            enumeration is capped so the message cannot grow with the tree.
    """
    counts = dict.fromkeys(TaskStatus, 0)
    offenders: list[tuple[Any, str]] = []
    for task_id, status in status_map.items():
        try:
            member = TaskStatus(status)
        except ValueError:
            offenders.append((task_id, status))
        else:
            counts[member] += 1

    if offenders:
        raise CensusVocabularyError(_vocabulary_drift_message(offenders))

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
