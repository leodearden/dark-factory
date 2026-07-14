"""Tests for orchestrator.background_service (W10-η).

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): one reusable seam
collapsing the eleven background-loop/service lifecycles in harness.py so
the recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/
1080) becomes structurally impossible.

Steps covered:
  step-1   RED   — BackoffPolicy constant-delay contract
  step-2   GREEN — BackoffPolicy + DEFAULT_BACKOFF_SECS
  step-3   RED   — BackgroundService.start() idempotency
  step-4   GREEN — BackgroundService dataclass + idempotent start()
  step-5   RED   — BackgroundService loop contract (S2 / LR-1)
  step-6   GREEN — BackgroundService._loop() canonical sleep-first body
  step-7   RED   — BackgroundService.stop() contract
  step-8   GREEN — BackgroundService.stop()
  step-9   RED   — LifecycleRegistry.register()/start_all() (LR-3 start side)
  step-10  GREEN — LifecycleRegistry register/start_all
  step-11  RED   — LifecycleRegistry.stop_all() (S1 / LR-2)
  step-12  GREEN — LifecycleRegistry.stop_all()
  step-13  RED   — ManagedService adapter + LifecycleService Protocol
  step-14  GREEN — ManagedService + LifecycleService

This module imports ``orchestrator.background_service`` symbols LOCALLY
inside each test (not at module scope) so a not-yet-implemented symbol
never breaks collection of the rest of the file during the RED steps —
mirrors test_merge_queue_lifecycle_registry.py's / test_item_lifecycle.py's
convention.

This project's pyproject.toml does NOT set ``asyncio_mode = "auto"``, so
pytest-asyncio's default "strict" mode applies: every coroutine test needs
an explicit ``@pytest.mark.asyncio`` marker, applied PER TEST (never at
class or module level, since a class-level mark would error the sync tests
sharing that class) — mirrors test_merge_queue_resource_audit.py.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest


class TestBackoffPolicy:
    """step-1: BackoffPolicy constant-delay contract."""

    def test_constant_delay_regardless_of_attempt(self) -> None:
        from orchestrator.background_service import BackoffPolicy

        policy = BackoffPolicy(60.0)

        assert policy.delay_for(0) == 60.0
        assert policy.delay_for(1) == 60.0
        assert policy.delay_for(5) == 60.0
        assert policy.delay_for(100) == 60.0

    def test_default_backoff_secs_matches_harness_parity_constant(self) -> None:
        from orchestrator.background_service import DEFAULT_BACKOFF_SECS
        from orchestrator.harness import _BG_LOOP_FAILURE_BACKOFF_SECS

        assert DEFAULT_BACKOFF_SECS == _BG_LOOP_FAILURE_BACKOFF_SECS
        assert DEFAULT_BACKOFF_SECS == 60.0


def _make_service(pass_fn, **overrides):
    """Construct a BackgroundService with test-friendly defaults.

    Local import (not module-level) so a not-yet-implemented
    ``BackgroundService`` never breaks collection of ``TestBackoffPolicy``.
    """
    from orchestrator.background_service import BackgroundService, BackoffPolicy

    kwargs = {
        'name': 'test-service',
        'pass_fn': pass_fn,
        'interval_secs': 0.01,
        'backoff': BackoffPolicy(60.0),
        'stop_timeout_secs': 1.0,
        'max_failure_logs': 3,
    }
    kwargs.update(overrides)
    return BackgroundService(**kwargs)


class TestBackgroundServiceStart:
    """step-3: BackgroundService.start() idempotency."""

    @pytest.mark.asyncio
    async def test_start_creates_exactly_one_named_task(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        task = svc._task
        assert task is not None
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == 'test-service'
        assert not task.done()

        blocker.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_second_start_while_running_is_a_noop(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        first_task = svc._task
        svc.start()

        assert first_task is not None
        assert svc._task is first_task

        blocker.set()
        first_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task

    @pytest.mark.asyncio
    async def test_start_after_task_finished_creates_fresh_task(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        first_task = svc._task
        assert first_task is not None
        first_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task
        assert first_task.done()

        svc.start()
        second_task = svc._task

        assert second_task is not None
        assert second_task is not first_task
        assert not second_task.done()

        blocker.set()
        second_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await second_task
