"""Tests for CostStore.task_invocation_cost — the single reader of per-task invocation cost."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from shared.cost_store import CostStore

_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


async def _save(
    store: CostStore,
    *,
    cost_usd: float,
    task_id: str | None = 't1',
    project_id: str = 'dark_factory',
    completed_at: datetime = _NOW,
) -> None:
    await store.save_invocation(
        run_id='run-1',
        task_id=task_id,
        project_id=project_id,
        account_name='acct',
        model='opus',
        role='implementer',
        cost_usd=cost_usd,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_create_tokens=None,
        duration_ms=1,
        capped=False,
        started_at=completed_at.isoformat(),
        completed_at=completed_at.isoformat(),
    )


@pytest.mark.asyncio
class TestTaskInvocationCost:
    async def test_sums_exactly_that_project_and_task(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.5)
            await _save(store, cost_usd=2.25)
            await _save(store, cost_usd=10.0, task_id='t2')
            await _save(store, cost_usd=20.0, project_id='reify')
            await _save(store, cost_usd=40.0, task_id=None)

            assert await store.task_invocation_cost('dark_factory', 't1') == pytest.approx(3.75)

    async def test_nothing_matching_is_a_float_zero(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            total = await store.task_invocation_cost('dark_factory', 't1')
        assert total == 0.0
        assert isinstance(total, float)

    async def test_since_is_an_inclusive_lower_bound(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.0, completed_at=_NOW)
            await _save(store, cost_usd=2.0, completed_at=_NOW + timedelta(seconds=1, microseconds=5))
            await _save(store, cost_usd=100.0, completed_at=_NOW - timedelta(microseconds=1))
            await _save(store, cost_usd=200.0, completed_at=_NOW - timedelta(hours=25))

            assert await store.task_invocation_cost(
                'dark_factory', 't1', since=_NOW,
            ) == pytest.approx(3.0)

    async def test_an_offset_since_is_the_same_instant_in_utc(self, tmp_path: Path):
        plus_two = timezone(timedelta(hours=2))
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.0, completed_at=_NOW)
            await _save(store, cost_usd=100.0, completed_at=_NOW - timedelta(minutes=1))

            assert await store.task_invocation_cost(
                'dark_factory', 't1', since=_NOW.astimezone(plus_two),
            ) == pytest.approx(1.0)

    async def test_a_naive_since_is_refused(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            with pytest.raises(ValueError, match='since'):
                await store.task_invocation_cost(
                    'dark_factory', 't1', since=datetime(2026, 10, 19, 12, 0),
                )

    async def test_a_never_opened_store_raises_rather_than_reading_zero(self, tmp_path: Path):
        store = CostStore(tmp_path / 'runs.db')
        with pytest.raises(RuntimeError):
            await store.task_invocation_cost('dark_factory', 't1')

    async def test_a_closed_store_raises_rather_than_reading_zero(self, tmp_path: Path):
        store = CostStore(tmp_path / 'runs.db')
        await store.open()
        await store.close()
        with pytest.raises(RuntimeError):
            await store.task_invocation_cost('dark_factory', 't1')


@pytest.mark.asyncio
class TestTasksInvocationCost:
    async def test_sums_every_named_task_of_that_project(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.5, task_id='t1')
            await _save(store, cost_usd=2.25, task_id='t2')
            await _save(store, cost_usd=10.0, task_id='t3')
            await _save(store, cost_usd=20.0, task_id='t1', project_id='reify')
            await _save(store, cost_usd=40.0, task_id=None)

            assert await store.tasks_invocation_cost(
                'dark_factory', {'t1', 't2', 'absent'},
            ) == pytest.approx(3.75)

    async def test_since_bounds_every_task(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.0, task_id='t1', completed_at=_NOW)
            await _save(store, cost_usd=2.0, task_id='t2', completed_at=_NOW)
            await _save(store, cost_usd=100.0, task_id='t2', completed_at=_NOW - timedelta(hours=25))

            assert await store.tasks_invocation_cost(
                'dark_factory', ['t1', 't2'], since=_NOW,
            ) == pytest.approx(3.0)

    async def test_agrees_with_the_per_task_reader(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.5, task_id='t1')
            await _save(store, cost_usd=2.5, task_id='t2')

            per_task = [await store.task_invocation_cost('dark_factory', t) for t in ('t1', 't2')]
            assert await store.tasks_invocation_cost(
                'dark_factory', ('t1', 't2'),
            ) == pytest.approx(sum(per_task))

    async def test_no_tasks_is_a_float_zero(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            await _save(store, cost_usd=1.0)
            total = await store.tasks_invocation_cost('dark_factory', ())
        assert total == 0.0
        assert isinstance(total, float)

    async def test_a_bare_task_id_string_is_refused(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            with pytest.raises(TypeError, match='task_ids'):
                await store.tasks_invocation_cost('dark_factory', 't1')

    async def test_a_naive_since_is_refused(self, tmp_path: Path):
        async with CostStore(tmp_path / 'runs.db') as store:
            with pytest.raises(ValueError, match='since'):
                await store.tasks_invocation_cost(
                    'dark_factory', ['t1'], since=datetime(2026, 10, 19, 12, 0),
                )

    async def test_a_closed_store_raises_rather_than_reading_zero(self, tmp_path: Path):
        store = CostStore(tmp_path / 'runs.db')
        await store.open()
        await store.close()
        with pytest.raises(RuntimeError):
            await store.tasks_invocation_cost('dark_factory', ['t1'])
