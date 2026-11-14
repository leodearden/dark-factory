"""The scheduler state as one access datum, read by the Scheduler tab and the PRD view alike."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from shared.scheduler_state import SchedulerStateSnapshot

import dashboard.data.scheduler as sched
from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import TASKS_BUDGET
from dashboard.data.datum import DatumState

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
FILE_STAMP = NOW - timedelta(hours=3)
URL1, URL2 = 'http://localhost:9000', 'http://localhost:9001'


@pytest.fixture(autouse=True)
def _forget_state():
    sched.clear_scheduler_state()
    yield
    sched.clear_scheduler_state()


@pytest.fixture
def config(tmp_path: Path) -> DashboardConfig:
    return DashboardConfig(project_root=tmp_path, fused_memory_urls=[URL1, URL2])


def wire(holder: str = '7', **overrides: Any) -> dict[str, Any]:
    return {
        'current_holders': {'dashboard/src': holder},
        'lock_depth': 2,
        'snapshot_at': FILE_STAMP.isoformat(),
        **overrides,
    }


class Scheduler:
    """The MCP seam: per-url answers for get_scheduler_state, recording every call."""

    def __init__(self, **by_url: Callable[[], Any]) -> None:
        self.by_url = by_url
        self.calls: list[str] = []
        self.release: asyncio.Event | None = None

    async def __call__(self, _client, url, tool, _args, *_rest, **_kw):
        assert tool == 'get_scheduler_state'
        self.calls.append(url)
        if self.release is not None:
            await self.release.wait()
        return self.by_url[url]()


def refused() -> Any:
    raise httpx.ConnectError('refused')


async def acquire(config, scheduler: Scheduler, *, now=NOW, wait_seconds=None):
    async with httpx.AsyncClient() as client:
        with patch('dashboard.data.scheduler.mcp_tool_call', new=scheduler):
            return await sched.acquire_scheduler_state(
                client, config, config.project_root, now=now, wait_seconds=wait_seconds,
            )


async def test_the_state_carries_the_files_own_stamp_and_is_measured_at_the_request(config):
    datum = await acquire(config, Scheduler(**{URL1: wire}))

    assert datum.state is DatumState.FRESH
    assert datum.as_of == NOW
    assert datum.value is not None
    assert datum.value.lock_depth == 2
    assert dict(datum.value.current_holders) == {'dashboard/src': '7'}
    assert datum.value.snapshot_at == FILE_STAMP


async def test_a_second_acquire_inside_the_ttl_reads_nothing_new(config):
    scheduler = Scheduler(**{URL1: wire})

    await acquire(config, scheduler)
    again = await acquire(config, scheduler, now=NOW + timedelta(seconds=1))

    assert scheduler.calls == [URL1]
    assert again.as_of == NOW


async def test_a_refused_first_url_fails_over_to_the_second(config):
    datum = await acquire(config, Scheduler(**{URL1: refused, URL2: lambda: wire(holder='9')}))

    assert datum.state is DatumState.FRESH
    assert datum.value is not None
    assert dict(datum.value.current_holders) == {'dashboard/src': '9'}


async def test_every_url_failing_is_unknown_naming_the_urls(config):
    datum = await acquire(config, Scheduler(**{URL1: refused, URL2: refused}))

    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert URL1 in (datum.reason or '') and URL2 in (datum.reason or '')


async def test_every_url_failing_after_a_success_serves_the_last_snapshot_stale(config, monkeypatch):
    await acquire(config, Scheduler(**{URL1: wire}))
    monkeypatch.setattr(sched, 'SCHEDULER_STATE_TTL_SECONDS', 0.0)

    datum = await acquire(
        config, Scheduler(**{URL1: refused, URL2: refused}), now=NOW + timedelta(seconds=6),
    )

    assert datum.state is DatumState.STALE
    assert datum.as_of == NOW
    assert datum.value is not None
    assert dict(datum.value.current_holders) == {'dashboard/src': '7'}
    assert URL1 in (datum.reason or '')


async def test_a_reply_that_is_not_a_snapshot_is_a_lower_bound_empty_snapshot(config):
    datum = await acquire(config, Scheduler(**{URL1: lambda: {'skip_counts': 'not-a-mapping'}}))

    assert datum.state is DatumState.LOWER_BOUND
    assert datum.value == SchedulerStateSnapshot.empty()
    assert 'parse' in (datum.reason or '')


async def test_a_reply_slower_than_the_wait_is_unknown_and_lands_for_the_next_acquire(config):
    scheduler = Scheduler(**{URL1: wire})
    scheduler.release = asyncio.Event()

    async with httpx.AsyncClient() as client:
        with patch('dashboard.data.scheduler.mcp_tool_call', new=scheduler):
            waited = await sched.acquire_scheduler_state(
                client, config, config.project_root, now=NOW, wait_seconds=0.05,
            )
            scheduler.release.set()
            landed = await sched.acquire_scheduler_state(
                client, config, config.project_root, now=NOW + timedelta(seconds=1),
            )

    assert waited.state is DatumState.UNKNOWN
    assert 'in flight' in (waited.reason or '')
    assert landed.state is DatumState.FRESH
    assert landed.as_of == NOW
    assert scheduler.calls == [URL1]


@pytest.mark.parametrize('forget', [sched.clear_scheduler_state, sched._scheduler_cache_clear])
async def test_both_clear_hooks_forget_the_state(config, forget):
    scheduler = Scheduler(**{URL1: wire})
    await acquire(config, scheduler)

    forget()
    await acquire(config, scheduler, now=NOW + timedelta(seconds=1))

    assert scheduler.calls == [URL1, URL1]


def test_the_whole_read_bound_sits_above_the_scheduler_tabs_handler_deadline():
    handler_deadline = TASKS_BUDGET.total

    assert handler_deadline < sched.SCHEDULER_STATE_READ_TIMEOUT
    assert sched.SCHEDULER_STATE_TTL_SECONDS < sched.SCHEDULER_STATE_FRESHNESS_BOUND_SECONDS
