"""Tests for fused_memory/middleware/curator_registry.py: the lazy, off-loop,
retrying load of one curator guard registry and the health it reports.

Every registry here is built directly with an injected loader and clock, so
nothing is patched.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import pytest

from fused_memory.middleware.curator_registry import (
    CuratorRegistry,
    RegistryHealth,
    RegistryState,
)
from fused_memory.utils.safe_yaml import YamlListFileError

LABEL = 'blocklist'
CONFIG_KEY = 'cancelled_premise_blocklist_path'


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _ScriptedLoader:
    """A loader that plays *outcomes* in order (the last one repeats) and
    records each call's path and thread."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = outcomes
        self.paths: list[Path] = []
        self.threads: list[int] = []

    def __call__(self, path: Path) -> list:
        self.paths.append(path)
        self.threads.append(threading.get_ident())
        outcome = self._outcomes[min(len(self.paths), len(self._outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def calls(self) -> int:
        return len(self.paths)


def _not_found(path: Path) -> YamlListFileError:
    return YamlListFileError(path, f'file not found: {path}')


def _registry(
    loader,
    *,
    path: str | None = 'registry.yaml',
    cwd: Path | None = None,
    retry_seconds: float = 300.0,
    clock: _Clock | None = None,
) -> CuratorRegistry:
    return CuratorRegistry(
        label=LABEL,
        config_key=CONFIG_KEY,
        loader=loader,
        configured_path=lambda: path,
        cwd=cwd,
        retry_seconds=retry_seconds,
        clock=clock or _Clock(),
    )


def _fail_open_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and 'failing open' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestCuratorRegistryLoad:
    async def test_unconfigured_returns_none_without_loading(self):
        loader = _ScriptedLoader(['entry'])
        registry = _registry(loader, path=None)

        assert await registry.entries() is None

        assert loader.calls == 0
        assert registry.health().state is RegistryState.UNCONFIGURED

    async def test_configured_before_any_call_is_not_loaded(self, tmp_path):
        registry = _registry(_ScriptedLoader(['entry']), cwd=tmp_path)

        health = registry.health()

        assert health.state is RegistryState.NOT_LOADED
        assert health.path is None
        assert health.entry_count is None
        assert health.error is None
        assert health.failures == 0

    async def test_successful_load_resolves_relative_path_and_is_terminal(self, tmp_path):
        loader = _ScriptedLoader(['a', 'b'])
        clock = _Clock()
        registry = _registry(
            loader, path='rel/registry.yaml', cwd=tmp_path, retry_seconds=0, clock=clock,
        )

        first = await registry.entries()
        clock.now += 10_000
        second = await registry.entries()

        resolved = tmp_path / 'rel/registry.yaml'
        assert first == ['a', 'b']
        assert second == ['a', 'b']
        assert loader.paths == [resolved]
        health = registry.health()
        assert health.state is RegistryState.LOADED
        assert health.entry_count == 2
        assert health.path == str(resolved)
        assert health.error is None

    async def test_empty_registry_is_loaded_not_failed(self, tmp_path):
        registry = _registry(_ScriptedLoader([]), cwd=tmp_path)

        assert await registry.entries() == []

        health = registry.health()
        assert health.state is RegistryState.LOADED
        assert health.entry_count == 0

    async def test_failed_load_fails_open_and_reports_failed(self, tmp_path, caplog):
        resolved = tmp_path / 'registry.yaml'
        registry = _registry(_ScriptedLoader(_not_found(resolved)), cwd=tmp_path)

        with caplog.at_level(logging.WARNING):
            assert await registry.entries() is None

        health = registry.health()
        assert health.state is RegistryState.FAILED
        assert health.failures == 1
        assert 'YamlListFileError' in health.error
        assert 'file not found' in health.error
        assert health.path == str(resolved)
        warnings = _fail_open_warnings(caplog)
        assert len(warnings) == 1
        assert LABEL in warnings[0].getMessage()

    async def test_failed_load_is_not_retried_inside_the_backoff(self, tmp_path):
        clock = _Clock()
        loader = _ScriptedLoader(_not_found(tmp_path / 'registry.yaml'), ['entry'])
        registry = _registry(loader, cwd=tmp_path, retry_seconds=300, clock=clock)

        await registry.entries()
        clock.now += 299
        assert await registry.entries() is None

        assert loader.calls == 1
        health = registry.health()
        assert health.state is RegistryState.FAILED
        assert health.failures == 1

    async def test_failed_load_is_retried_after_the_backoff(self, tmp_path):
        clock = _Clock()
        loader = _ScriptedLoader(_not_found(tmp_path / 'registry.yaml'), ['entry'])
        registry = _registry(loader, cwd=tmp_path, retry_seconds=300, clock=clock)

        await registry.entries()
        clock.now += 301
        assert await registry.entries() == ['entry']

        assert loader.calls == 2
        health = registry.health()
        assert health.state is RegistryState.LOADED
        assert health.entry_count == 1
        assert health.failures == 1

    async def test_each_repeated_failure_counts_and_warns(self, tmp_path, caplog):
        loader = _ScriptedLoader(_not_found(tmp_path / 'registry.yaml'))
        registry = _registry(loader, cwd=tmp_path, retry_seconds=0)

        with caplog.at_level(logging.WARNING):
            await registry.entries()
            await registry.entries()

        assert loader.calls == 2
        assert registry.health().failures == 2
        assert len(_fail_open_warnings(caplog)) == 2

    async def test_load_runs_off_the_event_loop_thread(self, tmp_path):
        loader = _ScriptedLoader(['entry'])
        registry = _registry(loader, cwd=tmp_path)

        await registry.entries()

        assert loader.threads
        assert all(tid != threading.get_ident() for tid in loader.threads)

    async def test_concurrent_first_calls_load_once_and_both_see_entries(self, tmp_path):
        calls = 0

        def slow_loader(path: Path) -> list:
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return ['entry']

        registry = _registry(slow_loader, cwd=tmp_path)

        first, second = await asyncio.gather(registry.entries(), registry.entries())

        assert first == ['entry']
        assert second == ['entry']
        assert calls == 1

    async def test_cancelled_load_leaves_state_untouched(self, tmp_path):
        loader = _ScriptedLoader(asyncio.CancelledError(), ['entry'])
        registry = _registry(loader, cwd=tmp_path, retry_seconds=300)

        with pytest.raises(asyncio.CancelledError):
            await registry.entries()

        health = registry.health()
        assert health.state is RegistryState.NOT_LOADED
        assert health.failures == 0

        assert await registry.entries() == ['entry']
        assert registry.health().state is RegistryState.LOADED

    async def test_relative_path_without_cwd_warns_naming_config_key(self, caplog):
        loader = _ScriptedLoader(['entry'])
        registry = _registry(loader, path='rel/registry.yaml', cwd=None)

        with caplog.at_level(logging.WARNING):
            await registry.entries()

        assert loader.paths == [Path('rel/registry.yaml')]
        assert any(
            CONFIG_KEY in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


class TestRegistryHealth:
    @pytest.mark.parametrize(
        ('fields', 'state_value', 'offending'),
        [
            (dict(state=RegistryState.LOADED), 'loaded', 'entry_count=None'),
            (dict(state=RegistryState.FAILED, failures=1), 'failed', 'error=None'),
            (dict(state=RegistryState.FAILED, error='boom'), 'failed', 'failures=0'),
            (dict(state=RegistryState.NOT_LOADED, error='boom'), 'not_loaded', "error='boom'"),
            (dict(state=RegistryState.UNCONFIGURED, entry_count=3), 'unconfigured', 'entry_count=3'),
        ],
        ids=[
            'loaded-without-count',
            'failed-without-error',
            'failed-with-zero-failures',
            'not-loaded-with-error',
            'unconfigured-with-count',
        ],
    )
    def test_state_field_invariant_is_enforced(self, fields, state_value, offending):
        with pytest.raises(ValueError) as excinfo:
            RegistryHealth(label=LABEL, config_key=CONFIG_KEY, **fields)

        message = str(excinfo.value)
        assert state_value in message
        assert offending in message

    def test_to_wire_carries_every_field(self):
        health = RegistryHealth(
            label=LABEL,
            config_key=CONFIG_KEY,
            state=RegistryState.FAILED,
            path='/x/registry.yaml',
            error='YamlListFileError: file not found: /x/registry.yaml',
            failures=2,
        )

        assert health.to_wire() == {
            'label': LABEL,
            'config_key': CONFIG_KEY,
            'state': 'failed',
            'path': '/x/registry.yaml',
            'entry_count': None,
            'error': 'YamlListFileError: file not found: /x/registry.yaml',
            'failures': 2,
        }
