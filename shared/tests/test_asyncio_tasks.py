"""Tests for shared.asyncio_tasks — holding a strong reference to a
fire-and-forget asyncio task until it ends (task 4530)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
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
def _recorded_loop_errors() -> Iterator[list[tuple[str, BaseException | None]]]:
    """Record (message, exception) for every report reaching the loop's handler.

    Only those two fields are kept: the full context carries the reporting
    Task itself, and holding it would resurrect a task mid-finalization.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    seen: list[tuple[str, BaseException | None]] = []
    loop.set_exception_handler(lambda _loop, context: seen.append((context['message'], context.get('exception'))))
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

        assert any(isinstance(exception, _Boom) for _message, exception in errors)

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


class _HookFailed(Exception):
    pass


class TestTrackTaskOnDone:
    async def test_hook_runs_exactly_once_with_the_finished_task(self):
        seen: list[asyncio.Task] = []
        task = asyncio.create_task(asyncio.sleep(0))

        track_task(task, set(), on_done=seen.append)
        await _drain(task)

        assert seen == [task]

    async def test_hook_sees_the_task_already_released_from_every_registry(self):
        first: set[asyncio.Task] = set()
        second: set[asyncio.Task] = set()
        still_registered: list[bool] = []

        def hook(finished: asyncio.Task) -> None:
            still_registered.append(finished in first or finished in second)

        task = asyncio.create_task(asyncio.sleep(0))
        track_task(task, first, second, on_done=hook)
        await _drain(task)

        assert still_registered == [False]

    async def test_hook_runs_for_a_cancelled_task(self):
        seen: list[bool] = []
        task = asyncio.create_task(asyncio.Event().wait())
        track_task(task, set(), on_done=lambda finished: seen.append(finished.cancelled()))

        task.cancel()
        await _drain(task)

        assert seen == [True]

    async def test_hook_can_still_read_the_exception_of_a_raising_task(self):
        seen: list[BaseException | None] = []
        task = asyncio.create_task(_raise_boom())
        track_task(task, set(), on_done=lambda finished: seen.append(finished.exception()))

        await _drain(task)

        assert len(seen) == 1
        assert isinstance(seen[0], _Boom)

    async def test_raising_hook_is_logged_and_does_not_strand_the_task(self, caplog):
        registry: set[asyncio.Task] = set()

        def raising_hook(_finished: asyncio.Task) -> None:
            raise _HookFailed('hook blew up')

        with caplog.at_level(logging.WARNING, logger='shared.asyncio_tasks'), _recorded_loop_errors() as errors:
            task = asyncio.create_task(asyncio.sleep(0))
            track_task(task, registry, on_done=raising_hook)
            await _drain(task)

        assert registry == set()
        assert errors == []
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None
        assert isinstance(warnings[0].exc_info[1], _HookFailed)
        assert raising_hook.__qualname__ in warnings[0].getMessage()

    async def test_explicit_none_hook_keeps_the_default_release_and_consumption(self):
        registry: set[asyncio.Task] = set()
        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(_raise_boom())
            track_task(task, registry, on_done=None)
            await _drain(task)
            task_ref = weakref.ref(task)
            del task
            _collect_and_confirm_gone(task_ref)

        assert errors == []
        assert registry == set()
