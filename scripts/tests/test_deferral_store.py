"""Tests for ``scripts/deferral_store.py``, the deferral scripts' typed MCP boundary.

Every call goes over the real transport (``census_trigger.post_mcp_tool_call``)
into ``FakeFusedMemory`` through ``install_fake_httpx``; a store built on a
plain ``post`` callable stands in only where the reply itself must be malformed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import deferral_store
import pytest
from deferral_fake_fused_memory import FakeFusedMemory
from deferral_store import DeferralStore, RecordHit, StoreUnreachable, TaskRow
from legibility import census_trigger
from shared.task_statuses import TaskStatus

URL = 'http://fake'


def _row(task_id: int | str, status: str = 'deferred', **metadata: Any) -> dict[str, Any]:
    return {'id': str(task_id), 'title': f'task {task_id}', 'status': status, 'metadata': metadata}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path.resolve() / 'project'


@pytest.fixture
def store() -> DeferralStore:
    return DeferralStore(url=URL, post=census_trigger.post_mcp_tool_call)


@pytest.fixture
def serve(install_fake_httpx: Callable[..., Any]) -> Callable[[FakeFusedMemory], FakeFusedMemory]:
    def _serve(fake: FakeFusedMemory) -> FakeFusedMemory:
        install_fake_httpx(fake.post)
        return fake

    return _serve


def _replying(reply: Any) -> DeferralStore:
    def post(url: str, tool: str, arguments: dict[str, Any], *, timeout: float) -> Any:
        return reply

    return DeferralStore(url=URL, post=post)


class TestDeferredRows:
    def test_one_unpaged_get_tasks_for_deferred_rows(self, root: Path, store: DeferralStore, serve: Any) -> None:
        fake = serve(FakeFusedMemory({str(root): [_row(1), _row(2, 'pending'), _row(3, x_armed_by='t')]}))

        rows = store.deferred_rows(root)

        assert fake.calls == [('get_tasks', {'project_root': str(root), 'statuses': ['deferred']})]
        assert [(row.task_id, row.status) for row in rows] == [(1, TaskStatus.DEFERRED), (3, TaskStatus.DEFERRED)]
        assert all(isinstance(row, TaskRow) for row in rows)
        assert rows[1].metadata == {'x_armed_by': 't'}

    def test_a_row_that_is_not_deferred_is_unreachable(self, root: Path) -> None:
        with pytest.raises(StoreUnreachable, match='get_tasks'):
            _replying({'tasks': [_row(1, 'pending')]}).deferred_rows(root)


class TestRowsWithRecord:
    def test_pages_until_has_more_is_false(self, root: Path, store: DeferralStore, serve: Any) -> None:
        rows = [_row(n, 'pending' if n % 2 else 'deferred', deferral={'kind': 'x'}) for n in range(1, 151)]
        fake = serve(FakeFusedMemory({str(root): [*rows, _row(500)]}))

        hits = store.rows_with_record(root)

        finds = fake.calls_to('find_tasks_by_metadata')
        assert [(call['key'], call['limit'], call['offset']) for call in finds] == [
            ('deferral', 100, 0),
            ('deferral', 100, 100),
        ]
        assert all(call['project_root'] == str(root) for call in finds)
        assert len(hits) == 150
        assert hits[0] == RecordHit(task_id=1, status=TaskStatus.PENDING, title='task 1')
        assert hits[1] == RecordHit(task_id=2, status=TaskStatus.DEFERRED, title='task 2')

    def test_a_page_with_has_more_and_nothing_returned_is_unreachable(self, root: Path) -> None:
        reply = {'results': [], 'pagination': {'has_more': True}}

        with pytest.raises(StoreUnreachable, match='find_tasks_by_metadata'):
            _replying(reply).rows_with_record(root)


class TestStatuses:
    def test_ids_go_as_strings_and_come_back_as_statuses(
        self, root: Path, store: DeferralStore, serve: Any
    ) -> None:
        fake = serve(FakeFusedMemory({str(root): [_row(12, 'done'), _row(13, 'cancelled'), _row(14, 'pending')]}))

        statuses = store.statuses(root, [13, 12, 99])

        assert fake.calls_to('get_statuses') == [{'project_root': str(root), 'ids': ['12', '13', '99']}]
        assert dict(statuses) == {12: TaskStatus.DONE, 13: TaskStatus.CANCELLED}

    def test_no_ids_make_no_call(self, root: Path, store: DeferralStore, serve: Any) -> None:
        fake = serve(FakeFusedMemory({str(root): []}))

        assert dict(store.statuses(root, [])) == {}
        assert fake.calls == []


class TestUnreachable:
    @pytest.mark.parametrize(
        ('tool', 'call'),
        [
            ('get_tasks', lambda store, root: store.deferred_rows(root)),
            ('find_tasks_by_metadata', lambda store, root: store.rows_with_record(root)),
            ('get_statuses', lambda store, root: store.statuses(root, [1])),
        ],
    )
    def test_an_error_payload_names_the_tool_and_the_root(
        self, root: Path, store: DeferralStore, serve: Any, tool: str, call: Any
    ) -> None:
        fake = serve(FakeFusedMemory({str(root): [_row(1)]}))
        fake.unreadable.add(str(root))

        with pytest.raises(StoreUnreachable) as raised:
            call(store, root)

        assert tool in str(raised.value)
        assert str(root) in str(raised.value)

    def test_a_post_that_raises_is_unreachable(self, root: Path) -> None:
        def post(url: str, tool: str, arguments: dict[str, Any], *, timeout: float) -> Any:
            raise ConnectionError('refused')

        with pytest.raises(StoreUnreachable, match='refused') as raised:
            DeferralStore(url=URL, post=post).deferred_rows(root)

        assert str(root) in str(raised.value)

    @pytest.mark.parametrize('reply', [{'tasks': 'nope'}, {'tasks': None}, {}, ['not', 'an', 'object']])
    def test_tasks_that_are_not_a_list_are_unreachable(self, root: Path, reply: Any) -> None:
        with pytest.raises(StoreUnreachable, match='get_tasks'):
            _replying(reply).deferred_rows(root)

    def test_a_row_with_a_non_decimal_id_is_unreachable(
        self, root: Path, store: DeferralStore, serve: Any
    ) -> None:
        serve(FakeFusedMemory({str(root): [_row('abc')]}))

        with pytest.raises(StoreUnreachable, match='abc') as raised:
            store.deferred_rows(root)

        assert str(root) in str(raised.value)

    @pytest.mark.parametrize('reply', [{'statuses': ['12']}, {'statuses': {'12': 'parked'}}, {'statuses': {'x': 'done'}}])
    def test_a_malformed_status_map_is_unreachable(self, root: Path, reply: Any) -> None:
        with pytest.raises(StoreUnreachable, match='get_statuses'):
            _replying(reply).statuses(root, [12])

    def test_a_relative_root_is_refused_before_any_call(self, store: DeferralStore, serve: Any) -> None:
        fake = serve(FakeFusedMemory({}))

        with pytest.raises(ValueError, match='absolute'):
            store.deferred_rows(Path('relative/root'))

        assert fake.calls == []


class TestKnownProject:
    def test_a_root_without_a_tasks_db_is_unknown(
        self, root: Path, project_root_with_tasks_db: Callable[[Path], Path]
    ) -> None:
        assert deferral_store.known_project(root) is False

        project_root_with_tasks_db(root)

        assert deferral_store.known_project(root) is True
