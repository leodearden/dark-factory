"""The task_lookup datum — ``dashboard.data.task_lookup``.

Driven through the PUBLIC seam only: the shared fused-memory substrate
(``tests/_canned_mcp.py``) is patched at ``dashboard.data.tasks.mcp_tool_call``,
so the real snapshot unit, ``fetch_task``, the caches and the fan-out all run
underneath every assertion.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP, _raw_row

import dashboard.data.task_lookup as task_lookup
from dashboard.data.datum import DatumState, validate_datum
from dashboard.data.task_lookup import (
    LOOKUP_CONCURRENCY,
    LOOKUP_MISS_CAP,
    TaskRef,
    lookup_tasks,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
"""The one injected instant every lookup is stamped against."""


@pytest.fixture(autouse=True)
def _isolate_caches():
    """No unit, tree read or per-id answer may cross a test."""
    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    def _clear():
        snapshot_mod._snapshot_cache_clear()
        tasks_mod._fetch_tasks_cache_clear()
        task_lookup._lookup_cache_clear()

    _clear()
    yield
    _clear()


@pytest.fixture()
def root(tmp_path) -> str:
    return str(tmp_path / 'dark-factory')


def _canned(*pairs, **kwargs) -> CannedMCP:
    """A substrate holding one row per ``(id, status)`` pair, map agreeing."""
    return CannedMCP(
        rows=[_raw_row(tid, status) for tid, status in pairs],
        status_map=dict(pairs),
        status_page_size=2000,
        **kwargs,
    )


async def _lookup(canned, client, config, refs, *, now=NOW):
    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        return await lookup_tasks(client, config, refs, now=now)


def _task_reads(canned) -> list[int]:
    return [int(call['args']['id']) for call in canned.calls_to('get_task')]


class TestTheSignal:
    """An active id comes from the snapshot; a terminal one costs one get_task."""

    async def test_an_active_id_is_served_from_the_snapshot_rows(
        self, root, dashboard_config, dummy_client,
    ):
        from dashboard.data.task_snapshot import acquire_snapshot

        canned = _canned((10, 'in-progress'), (3, 'done'))
        active = TaskRef(root, 10)

        result = await _lookup(canned, dummy_client, dashboard_config,
                               [active, TaskRef(root, 3)])

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            rows = (await acquire_snapshot(dummy_client, dashboard_config, root, now=NOW)).rows
        served = result[active]
        assert served.value is not None
        assert (served.value['id'], served.value['title']) == (10, 'task 10')
        assert (served.state, served.as_of, served.reason) == (rows.state, rows.as_of, rows.reason)
        assert served.freshness_bound_seconds == rows.freshness_bound_seconds
        assert 10 not in _task_reads(canned)

    async def test_a_terminal_id_costs_exactly_one_get_task(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((10, 'in-progress'), (3, 'done'))
        terminal = TaskRef(root, 3)

        result = await _lookup(canned, dummy_client, dashboard_config,
                               [TaskRef(root, 10), terminal])

        assert _task_reads(canned) == [3]
        served = result[terminal]
        assert served.state is DatumState.FRESH
        assert served.as_of == NOW
        assert served.value is not None
        assert (served.value['id'], served.value['status']) == (3, 'done')

    async def test_a_terminal_row_is_held_for_the_ttl(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((3, 'done'))
        terminal = TaskRef(root, 3)
        await _lookup(canned, dummy_client, dashboard_config, [terminal])

        again = await _lookup(canned, dummy_client, dashboard_config, [terminal],
                              now=NOW + timedelta(seconds=60))

        assert _task_reads(canned) == [3], 'the second lookup must not re-read'
        assert again[terminal].as_of == NOW, 'it keeps the first read’s instant'


class TestAbsentIds:

    async def test_an_absent_id_is_unknown_naming_the_id_and_root(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((3, 'done'))
        absent = TaskRef(root, 99)

        served = (await _lookup(canned, dummy_client, dashboard_config, [absent]))[absent]

        assert served.state is DatumState.UNKNOWN
        assert served.value is None
        assert served.reason is not None
        assert '99' in served.reason
        assert root in served.reason

    async def test_not_found_is_cached(self, root, dashboard_config, dummy_client):
        canned = _canned((3, 'done'))
        absent = TaskRef(root, 99)
        await _lookup(canned, dummy_client, dashboard_config, [absent])

        again = await _lookup(canned, dummy_client, dashboard_config, [absent])

        assert _task_reads(canned) == [99]
        assert again[absent].state is DatumState.UNKNOWN


class TestBudget:
    """The miss path is capped, deadlined and width-bounded (INV-8)."""

    async def test_the_miss_cap_reads_the_newest_ids_first(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned(*((tid, 'done') for tid in range(1, 71)))
        refs = [TaskRef(root, tid) for tid in range(1, 71)]

        result = await _lookup(canned, dummy_client, dashboard_config, refs)

        assert LOOKUP_MISS_CAP == 64
        assert sorted(_task_reads(canned)) == list(range(7, 71))
        for tid in range(1, 7):
            served = result[TaskRef(root, tid)]
            assert served.state is DatumState.UNKNOWN
            assert served.reason is not None
            assert served.reason.startswith('lookup budget')
        assert all(result[TaskRef(root, tid)].state is DatumState.FRESH
                   for tid in range(7, 71))

    async def test_the_deadline_keeps_what_already_resolved(
        self, root, dashboard_config, dummy_client, monkeypatch,
    ):
        monkeypatch.setattr(task_lookup, 'LOOKUP_BUDGET_SECONDS', 0.2)
        fast, slow = range(5, 9), range(1, 3)
        canned = _canned(
            *((tid, 'done') for tid in [*fast, *slow]),
            task_delays={tid: 5.0 for tid in slow},
        )
        refs = [TaskRef(root, tid) for tid in [*fast, *slow]]

        started = time.monotonic()
        result = await _lookup(canned, dummy_client, dashboard_config, refs)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        for tid in fast:
            served = result[TaskRef(root, tid)]
            assert served.state is DatumState.FRESH
            assert served.value is not None
            assert served.value['id'] == tid
        for tid in slow:
            served = result[TaskRef(root, tid)]
            assert served.state is DatumState.UNKNOWN
            assert served.reason is not None
            assert served.reason.startswith('lookup budget')

    async def test_a_held_answer_survives_a_deadline_spent_in_the_snapshot_read(
        self, root, dashboard_config, dummy_client, monkeypatch,
    ):
        import dashboard.data.task_snapshot as snapshot_mod
        import dashboard.data.tasks as tasks_mod

        canned = _canned((10, 'in-progress'), (3, 'done'))
        held, unread = TaskRef(root, 3), TaskRef(root, 10)
        await _lookup(canned, dummy_client, dashboard_config, [held])
        snapshot_mod._snapshot_cache_clear()
        tasks_mod._fetch_tasks_cache_clear()
        monkeypatch.setattr(task_lookup, 'LOOKUP_BUDGET_SECONDS', 0.2)

        async def _hung_snapshot(client, url, tool, args, **kwargs):
            if tool != 'get_task':
                await asyncio.sleep(5.0)
            return await canned(client, url, tool, args, **kwargs)

        started = time.monotonic()
        result = await _lookup(_hung_snapshot, dummy_client, dashboard_config, [held, unread])

        assert time.monotonic() - started < 2.0
        held_served, unread_served = result[held], result[unread]
        assert held_served.state is DatumState.FRESH
        assert held_served.value is not None
        assert held_served.value['title'] == 'task 3'
        assert unread_served.state is DatumState.UNKNOWN
        assert unread_served.reason is not None
        assert unread_served.reason.startswith('lookup budget')

    async def test_misses_are_read_at_most_lookup_concurrency_at_a_time(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned(
            *((tid, 'done') for tid in range(1, 21)),
            task_delays={tid: 0.05 for tid in range(1, 21)},
        )

        await _lookup(canned, dummy_client, dashboard_config,
                      [TaskRef(root, tid) for tid in range(1, 21)])

        assert LOOKUP_CONCURRENCY == 4
        assert canned.max_in_flight == LOOKUP_CONCURRENCY
        assert len(_task_reads(canned)) == 20


class TestDegradation:

    async def test_an_active_id_falls_through_to_get_task_when_the_rows_read_fails(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((10, 'in-progress'))
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        active = TaskRef(root, 10)

        served = (await _lookup(canned, dummy_client, dashboard_config, [active]))[active]

        assert _task_reads(canned) == [10]
        assert served.state is DatumState.FRESH
        assert served.value is not None
        assert served.value['status'] == 'in-progress'

    async def test_a_live_row_read_through_get_task_is_never_cached(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((10, 'in-progress'))
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        active = TaskRef(root, 10)
        await _lookup(canned, dummy_client, dashboard_config, [active])

        again = await _lookup(canned, dummy_client, dashboard_config, [active])

        assert _task_reads(canned) == [10, 10]
        assert again[active].state is DatumState.FRESH

    async def test_an_unreachable_read_is_unknown_and_never_cached(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((3, 'done'))
        canned.fail_when = lambda call: call['tool'] == 'get_task'
        terminal = TaskRef(root, 3)

        served = (await _lookup(canned, dummy_client, dashboard_config, [terminal]))[terminal]
        reads_after_first = len(_task_reads(canned))
        await _lookup(canned, dummy_client, dashboard_config, [terminal])

        assert served.state is DatumState.UNKNOWN
        assert served.reason is not None
        assert 'canned get_task read timeout' in served.reason
        assert len(_task_reads(canned)) > reads_after_first, 'offline is never cached'


class TestTheEnvelope:

    async def test_every_datum_validates_at_a_later_serving_instant(
        self, root, dashboard_config, dummy_client, monkeypatch,
    ):
        monkeypatch.setattr(task_lookup, 'LOOKUP_BUDGET_SECONDS', 0.2)
        canned = _canned(
            (10, 'in-progress'), (3, 'done'), (2, 'done'),
            task_delays={2: 5.0},
        )
        refs = [TaskRef(root, tid) for tid in (10, 3, 2, 99)]

        result = await _lookup(canned, dummy_client, dashboard_config, refs)

        served_at = NOW + timedelta(seconds=5)
        assert {ref.task_id: datum.state for ref, datum in result.items()} == {
            10: DatumState.FRESH, 3: DatumState.FRESH,
            2: DatumState.UNKNOWN, 99: DatumState.UNKNOWN,
        }
        for datum in result.values():
            validate_datum(datum, served_at)

    async def test_duplicate_refs_return_one_entry(
        self, root, dashboard_config, dummy_client,
    ):
        canned = _canned((3, 'done'), (4, 'done'))
        refs = [TaskRef(root, 3), TaskRef(root, 3), TaskRef(root, 4)]

        result = await _lookup(canned, dummy_client, dashboard_config, refs)

        assert set(result) == {TaskRef(root, 3), TaskRef(root, 4)}
        assert sorted(_task_reads(canned)) == [3, 4]

    async def test_no_refs_reads_nothing(self, dashboard_config, dummy_client):
        canned = _canned((3, 'done'))

        assert await _lookup(canned, dummy_client, dashboard_config, []) == {}
        assert canned.calls == []


def test_the_deadline_only_ever_tightens_the_shared_budget():
    from dashboard.data.tasks import DEFAULT_WHOLE_OPERATION_BUDGET

    assert task_lookup.LOOKUP_BUDGET_SECONDS <= DEFAULT_WHOLE_OPERATION_BUDGET
