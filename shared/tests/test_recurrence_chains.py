"""Tests for shared.recurrence_chains — the one definition of a chain's current link.

docs/prds/recurring-deterministic-tasks.md R-D5: the re-mint sweep repairs, and
the chain-state gauge surfaces, exactly the chains whose current link is done.
Rows are shaped like ``SqliteTaskBackend._row_to_task`` output.
"""

from __future__ import annotations

from typing import Any

import pytest

from shared.recurrence_chains import current_links, recurrence_key
from shared.task_statuses import ACTIVE, TaskStatus

KEY = 'nightly-census'


def _link(task_id: str | int, status: str, key: str = KEY) -> dict[str, Any]:
    return {
        'id': task_id,
        'status': status,
        'metadata': {'recurrence': {'key': key, 'interval_secs': 3600}},
    }


def _current_ids(tasks: list[dict[str, Any]]) -> dict[str, str]:
    return {key: str(link['id']) for key, link in current_links(tasks).items()}


class TestRecurrenceKey:
    def test_carrier_row_yields_its_key(self) -> None:
        assert recurrence_key(_link('1', TaskStatus.PENDING)) == KEY

    @pytest.mark.parametrize(
        'row',
        [
            {'id': '1', 'status': 'done'},
            {'id': '1', 'status': 'done', 'metadata': None},
            {'id': '1', 'status': 'done', 'metadata': {}},
            {'id': '1', 'status': 'done', 'metadata': {'source': 'x'}},
            {'id': '1', 'status': 'done', 'metadata': {'recurrence': 'nightly'}},
            {'id': '1', 'status': 'done', 'metadata': {'recurrence': {'interval_secs': 60}}},
            {'id': '1', 'status': 'done', 'metadata': {'recurrence': {'key': ''}}},
            {'id': '1', 'status': 'done', 'metadata': {'recurrence': {'key': 7}}},
        ],
        ids=[
            'no-metadata',
            'null-metadata',
            'empty-metadata',
            'no-recurrence',
            'non-mapping-recurrence',
            'missing-key',
            'empty-key',
            'non-str-key',
        ],
    )
    def test_non_carrier_row_yields_none(self, row: dict[str, Any]) -> None:
        assert recurrence_key(row) is None


class TestCurrentLinks:
    def test_plain_tasks_are_not_chains(self) -> None:
        tasks = [
            {'id': '1', 'status': 'done', 'metadata': {}},
            {'id': '2', 'status': 'pending'},
        ]
        assert current_links(tasks) == {}

    def test_all_terminal_current_link_is_the_newest(self) -> None:
        tasks = [_link('1', TaskStatus.DONE), _link('2', TaskStatus.DONE)]
        assert current_links(tasks)[KEY] is tasks[1]

    def test_ended_chain_current_link_is_its_cancelled_tip(self) -> None:
        tasks = [_link('1', TaskStatus.DONE), _link('2', TaskStatus.CANCELLED)]
        assert _current_ids(tasks) == {KEY: '2'}

    def test_ids_order_numerically_not_lexicographically(self) -> None:
        tasks = [_link('9', TaskStatus.DONE), _link('10', TaskStatus.DONE)]
        assert _current_ids(tasks) == {KEY: '10'}

    def test_live_link_wins_over_a_newer_terminal_link(self) -> None:
        tasks = [_link('1', TaskStatus.PENDING), _link('2', TaskStatus.DONE)]
        assert current_links(tasks)[KEY] is tasks[0]

    def test_newest_live_link_wins_among_several(self) -> None:
        tasks = [
            _link('1', TaskStatus.BLOCKED),
            _link('2', TaskStatus.PENDING),
            _link('3', TaskStatus.DONE),
        ]
        assert _current_ids(tasks) == {KEY: '2'}

    @pytest.mark.parametrize('live_status', sorted(ACTIVE))
    def test_every_non_terminal_status_is_live(self, live_status: TaskStatus) -> None:
        tasks = [_link('1', live_status), _link('2', TaskStatus.DONE)]
        assert _current_ids(tasks) == {KEY: '1'}

    def test_chains_resolve_independently(self) -> None:
        tasks = [
            _link('1', TaskStatus.DONE, key='a'),
            _link('2', TaskStatus.DONE, key='b'),
            _link('3', TaskStatus.PENDING, key='a'),
            _link('4', TaskStatus.CANCELLED, key='b'),
            _link('5', TaskStatus.DONE, key='a'),
        ]
        assert _current_ids(tasks) == {'a': '3', 'b': '4'}

    @pytest.mark.parametrize(
        'statuses, expected',
        [
            ((TaskStatus.DONE, TaskStatus.DONE), 10),
            ((TaskStatus.PENDING, TaskStatus.DONE), 9),
        ],
    )
    def test_int_ids_give_the_same_answers(
        self, statuses: tuple[TaskStatus, TaskStatus], expected: int,
    ) -> None:
        tasks = [_link(9, statuses[0]), _link(10, statuses[1])]
        assert current_links(tasks)[KEY]['id'] == expected
