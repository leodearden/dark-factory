"""Tests for shared.asyncio_tasks — holding a strong reference to a
fire-and-forget asyncio task until it ends (task 4530)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import weakref
from collections.abc import Iterator

from shared.asyncio_tasks import track_task


class _Boom(Exception):
    pass


async def _raise_boom() -> None:
    raise _Boom('raised inside the tracked task')


async def _drain(task: asyncio.Task) -> None:
    await asyncio.wait({task}, timeout=5.0)
    await asyncio.sleep(0)


@contextlib.contextmanager
def _recorded_loop_errors() -> Iterator[list[dict]]:
    """Record every context the running loop hands its exception handler."""
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    try:
        yield seen
    finally:
        loop.set_exception_handler(previous)


def _collect_and_confirm_gone(task_ref: weakref.ref) -> None:
    gc.collect()
    assert task_ref() is None, 'the task must be collectable for the "never retrieved" check to mean anything'


class TestTrackTask:
    async def test_task_is_in_every_registry_before_the_loop_runs_again(self):
        first: set[asyncio.Task] = set()
        second: set[asyncio.Task] = set()
        task = asyncio.create_task(asyncio.sleep(0))

        track_task(task, first, second)

        assert task in first
        assert task in second
        await _drain(task)

    async def test_finished_task_is_released_from_every_registry(self):
        first: set[asyncio.Task] = set()
        second: set[asyncio.Task] = set()
        task = asyncio.create_task(asyncio.sleep(0))
        track_task(task, first, second)

        await _drain(task)

        assert task not in first
        assert task not in second

    async def test_raising_task_is_released_and_keeps_its_exception(self):
        registry: set[asyncio.Task] = set()
        task = asyncio.create_task(_raise_boom())
        track_task(task, registry)

        await _drain(task)

        assert task not in registry
        assert isinstance(task.exception(), _Boom)

    async def test_unretrieved_exception_control_is_reported_when_untracked(self):
        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(_raise_boom())
            await _drain(task)
            task_ref = weakref.ref(task)
            del task
            _collect_and_confirm_gone(task_ref)

        assert any(isinstance(context.get('exception'), _Boom) for context in errors)

    async def test_raising_task_exception_is_consumed(self):
        registry: set[asyncio.Task] = set()
        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(_raise_boom())
            track_task(task, registry)
            await _drain(task)
            task_ref = weakref.ref(task)
            del task
            _collect_and_confirm_gone(task_ref)

        assert errors == []
        assert registry == set()

    async def test_cancelled_task_is_released_without_the_callback_raising(self):
        registry: set[asyncio.Task] = set()
        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(asyncio.Event().wait())
            track_task(task, registry)
            task.cancel()
            await _drain(task)

        assert task.cancelled()
        assert task not in registry
        assert errors == []

    async def test_zero_registries_still_consumes_the_exception(self):
        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(_raise_boom())
            track_task(task)
            await _drain(task)
            task_ref = weakref.ref(task)
            del task
            _collect_and_confirm_gone(task_ref)

        assert errors == []

    async def test_returns_none(self):
        task = asyncio.create_task(asyncio.sleep(0))

        result = track_task(task, set())

        assert result is None
        await _drain(task)
