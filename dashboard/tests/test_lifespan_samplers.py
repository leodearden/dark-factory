"""The lifespan starts exactly the loops it was handed: the samplers iff its OWN config says so, the reaper either way, and nothing else."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from _dashboard_helpers import app_with_loops, apply_isolated_env
from fastapi import FastAPI
from starlette.testclient import TestClient

from dashboard.app import _build_http_limits, lifespan
from dashboard.background_loops import BackgroundLoops
from dashboard.config import DashboardConfig
from dashboard.loops import _QualityStore

Call = tuple[tuple[object, ...], dict[str, object]]


async def _idle() -> None:
    return None


def _recorder(
    label: str, starts: Counter[str], calls: dict[str, list[Call]],
) -> Callable[..., Coroutine[Any, Any, None]]:
    """A synchronous stand-in: it records at the create_task call site, not at first schedule."""

    def _start(*args: object, **kwargs: object) -> Coroutine[Any, Any, None]:
        starts[label] += 1
        calls.setdefault(label, []).append((args, kwargs))
        return _idle()

    return _start


def _recording_loops() -> tuple[BackgroundLoops, Counter[str], dict[str, list[Call]]]:
    starts: Counter[str] = Counter()
    calls: dict[str, list[Call]] = {}
    loops = BackgroundLoops(
        burndown=_recorder('burndown', starts, calls),
        metrics=_recorder('metrics', starts, calls),
        quality=_recorder('quality', starts, calls),
        reaper=_recorder('reaper', starts, calls),
    )
    return loops, starts, calls


def _run_one_lifespan() -> Counter[str]:
    loops, starts, _ = _recording_loops()
    with TestClient(app_with_loops(loops)):
        pass
    return starts


def test_a_default_lifespan_starts_each_sampler_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DASHBOARD_SAMPLERS', raising=False)

    starts = _run_one_lifespan()

    assert starts == Counter({'burndown': 1, 'metrics': 1, 'quality': 1, 'reaper': 1})


def test_samplers_off_starts_no_sampler_but_still_the_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('DASHBOARD_SAMPLERS', 'off')

    starts = _run_one_lifespan()

    assert starts['reaper'] == 1, 'the lifespan never ran, so the absences below prove nothing'
    assert starts['burndown'] == 0
    assert starts['metrics'] == 0
    assert starts['quality'] == 0


def test_the_switch_is_read_from_the_lifespans_own_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interleaving config that turns samplers ON must not reach this lifespan's gate.

    The swap rides ``_build_http_limits``, the earliest hook after
    ``app.state.config`` is assigned, as in
    dashboard/tests/test_lifespan_resource_binding.py::test_lifespan_binds_burndown_loop_to_the_config_it_built.
    """
    monkeypatch.setenv('DASHBOARD_SAMPLERS', 'off')
    loops, starts, _ = _recording_loops()
    app = app_with_loops(loops)
    swapped = DashboardConfig(project_root=tmp_path / 'other', samplers_enabled=True)
    swaps: list[DashboardConfig] = []

    def _swapping_limits(config: DashboardConfig) -> httpx.Limits:
        app.state.config = swapped
        swaps.append(swapped)
        return _build_http_limits(config)

    with patch('dashboard.app._build_http_limits', new=_swapping_limits), TestClient(app):
        pass

    assert swaps, 'the simulated interleave never ran, so this test proves nothing'
    assert starts['reaper'] == 1, 'the lifespan never ran, so the absences below prove nothing'
    assert starts['burndown'] == 0, 'the sampler gate read app.state.config, not its own'
    assert starts['metrics'] == 0, 'the sampler gate read app.state.config, not its own'
    assert starts['quality'] == 0, 'the sampler gate read app.state.config, not its own'


def test_the_quality_loop_is_handed_the_lifespans_own_store_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interleaving config on app.state must not reach the quality loop or its store."""
    monkeypatch.delenv('DASHBOARD_SAMPLERS', raising=False)
    loops, starts, calls = _recording_loops()
    app = app_with_loops(loops)
    swapped = DashboardConfig(project_root=tmp_path / 'other')
    swaps: list[DashboardConfig] = []

    def _swapping_limits(config: DashboardConfig) -> httpx.Limits:
        app.state.config = swapped
        swaps.append(swapped)
        return _build_http_limits(config)

    with patch('dashboard.app._build_http_limits', new=_swapping_limits), TestClient(app):
        pass

    assert swaps, 'the simulated interleave never ran, so this test proves nothing'
    assert starts['quality'] == 1
    [(args, kwargs)] = calls['quality']
    store, config = args
    assert kwargs == {}
    assert isinstance(store, _QualityStore)
    assert isinstance(config, DashboardConfig)
    assert store.db_path == config.quality_db
    assert config is not swapped
    assert config.project_root != swapped.project_root


async def test_every_task_the_lifespan_starts_comes_from_the_loops_it_was_handed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A task started beside ``loops`` would put a real loop back under every test.

    Driven directly rather than through ``TestClient``, so ``asyncio.all_tasks()``
    sees the lifespan's own event loop.
    """
    apply_isolated_env(monkeypatch, tmp_path)
    monkeypatch.delenv('DASHBOARD_SAMPLERS', raising=False)

    async def _parked(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    loops = BackgroundLoops(burndown=_parked, metrics=_parked, quality=_parked, reaper=_parked)

    before = asyncio.all_tasks()
    async with lifespan(FastAPI(), loops=loops):
        started = asyncio.all_tasks() - before

    assert len(started) == 4, f'expected the four handed loops, got {started}'
    assert all(getattr(task.get_coro(), 'cr_code', None) is _parked.__code__ for task in started), (
        f'the lifespan started a task that is not one of the loops it was handed: {started}'
    )
