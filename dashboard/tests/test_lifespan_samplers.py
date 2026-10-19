"""The lifespan starts its two samplers iff its OWN config says so, and the reaper either way."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Coroutine
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from starlette.testclient import TestClient

from dashboard.app import _build_http_limits
from dashboard.config import DashboardConfig

_LOOPS = ('_burndown_loop', '_metrics_loop', 'reaper_loop')


async def _idle() -> None:
    return None


def _recorder(label: str, starts: Counter[str]) -> Callable[..., Coroutine[Any, Any, None]]:
    """A synchronous stand-in: it records at the create_task call site, not at first schedule."""

    def _start(*args: object, **kwargs: object) -> Coroutine[Any, Any, None]:
        starts[label] += 1
        return _idle()

    return _start


def _patch_loops(stack: ExitStack) -> Counter[str]:
    starts: Counter[str] = Counter()
    for name in _LOOPS:
        # Stays on `dashboard.app`: this patches the name as `lifespan`
        # RESOLVES it. Retargeting to `dashboard.loops.*` would silently no-op.
        stack.enter_context(patch(f'dashboard.app.{name}', new=_recorder(name, starts)))
    return starts


def _run_one_lifespan() -> Counter[str]:
    from dashboard.app import app

    with ExitStack() as stack:
        starts = _patch_loops(stack)
        with TestClient(app):
            pass
    return starts


def test_a_default_lifespan_starts_each_sampler_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DASHBOARD_SAMPLERS', raising=False)

    starts = _run_one_lifespan()

    assert starts == Counter({'_burndown_loop': 1, '_metrics_loop': 1, 'reaper_loop': 1})


def test_samplers_off_starts_neither_sampler_but_still_the_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('DASHBOARD_SAMPLERS', 'off')

    starts = _run_one_lifespan()

    assert starts['reaper_loop'] == 1, 'the lifespan never ran, so the absences below prove nothing'
    assert starts['_burndown_loop'] == 0
    assert starts['_metrics_loop'] == 0


def test_the_switch_is_read_from_the_lifespans_own_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interleaving config that turns samplers ON must not reach this lifespan's gate.

    The swap rides ``_build_http_limits``, the earliest hook after
    ``app.state.config`` is assigned, as in
    dashboard/tests/test_lifespan_resource_binding.py::test_lifespan_binds_burndown_loop_to_the_config_it_built.
    """
    from dashboard.app import app

    monkeypatch.setenv('DASHBOARD_SAMPLERS', 'off')
    swapped = DashboardConfig(project_root=tmp_path / 'other', samplers_enabled=True)
    swaps: list[DashboardConfig] = []

    def _swapping_limits(config: DashboardConfig) -> httpx.Limits:
        app.state.config = swapped
        swaps.append(swapped)
        return _build_http_limits(config)

    config_before = getattr(app.state, 'config', None)
    try:
        with ExitStack() as stack:
            starts = _patch_loops(stack)
            stack.enter_context(patch('dashboard.app._build_http_limits', new=_swapping_limits))
            with TestClient(app):
                pass
    finally:
        if config_before is not None:
            app.state.config = config_before

    assert swaps, 'the simulated interleave never ran, so this test proves nothing'
    assert starts['reaper_loop'] == 1, 'the lifespan never ran, so the absences below prove nothing'
    assert starts['_burndown_loop'] == 0, 'the sampler gate read app.state.config, not its own'
    assert starts['_metrics_loop'] == 0, 'the sampler gate read app.state.config, not its own'
