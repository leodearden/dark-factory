"""Tests for shared.asyncio_tasks — holding a strong reference to a
fire-and-forget asyncio task until it ends (task 4530)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import weakref
from collections.abc import Iterator

import pytest

from shared.asyncio_tasks import abandon_task, cancel_bounded, gather_unwound, track_task


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


class TestAbandonTask:
    async def test_cancels_the_task(self):
        task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        abandon_task(task, set())
        await _drain(task)

        assert task.cancelled()

    async def test_task_is_in_every_registry_before_the_loop_runs_again(self):
        first: set[asyncio.Task] = set()
        second: set[asyncio.Task] = set()
        task = asyncio.create_task(asyncio.Event().wait())

        abandon_task(task, first, second)

        assert task in first
        assert task in second
        await _drain(task)

    async def test_task_is_released_from_every_registry_once_cancelled(self):
        first: set[asyncio.Task] = set()
        second: set[asyncio.Task] = set()
        task = asyncio.create_task(asyncio.Event().wait())

        abandon_task(task, first, second)
        await _drain(task)

        assert task not in first
        assert task not in second

    async def test_forwards_the_hook_which_sees_the_cancelled_task(self):
        seen: list[tuple[asyncio.Task, bool]] = []
        task = asyncio.create_task(asyncio.Event().wait())

        abandon_task(task, set(), on_done=lambda finished: seen.append((finished, finished.cancelled())))
        await _drain(task)

        assert seen == [(task, True)]

    async def test_already_done_task_is_not_stranded(self):
        registry: set[asyncio.Task] = set()
        task = asyncio.create_task(asyncio.sleep(0))
        await _drain(task)

        with _recorded_loop_errors() as errors:
            abandon_task(task, registry)
            assert task in registry
            await asyncio.sleep(0)

        assert task not in registry
        assert errors == []


async def _survives_cancels(survivals: int, release: asyncio.Event) -> str:
    """Ignore the first *survivals* cancels, then end on the next one.

    Once *release* is set it returns at its next wake-up, so a test can always
    retire a task that outlived every cancel it was sent.
    """
    absorbed = 0
    while True:
        try:
            await release.wait()
            return 'released'
        except asyncio.CancelledError:
            if absorbed == survivals:
                raise
            absorbed += 1


class TestCancelBounded:
    async def test_a_task_that_honours_the_first_cancel_ends_and_reports_true(self):
        task = asyncio.create_task(_survives_cancels(0, asyncio.Event()))
        await asyncio.sleep(0)

        assert await cancel_bounded(task, set(), grace_secs=1.0) is True
        assert task.cancelled()

    async def test_a_task_that_survives_one_cancel_is_ended_by_the_second(self):
        task = asyncio.create_task(_survives_cancels(1, asyncio.Event()))
        await asyncio.sleep(0)

        assert await cancel_bounded(task, set(), grace_secs=0.05) is True
        assert task.cancelled()

    async def test_a_task_that_survives_both_cancels_is_tracked_and_reports_false_within_two_windows(self):
        release = asyncio.Event()
        registry: set[asyncio.Task] = set()
        task = asyncio.create_task(_survives_cancels(2, release))
        await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        started = loop.time()

        ended = await cancel_bounded(task, registry, grace_secs=0.05)

        assert ended is False
        assert loop.time() - started < 1.0
        assert task in registry
        release.set()
        await _drain(task)
        assert task not in registry

    async def test_the_task_exception_is_consumed_not_raised(self):
        async def _raises_on_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise _Boom('raised while unwinding') from None

        with _recorded_loop_errors() as errors:
            task = asyncio.create_task(_raises_on_cancel())
            await asyncio.sleep(0)
            assert await cancel_bounded(task, set(), grace_secs=1.0) is True
            task_ref = weakref.ref(task)
            del task
            _collect_and_confirm_gone(task_ref)

        assert errors == []


async def _all_started(*started: asyncio.Event) -> None:
    async with asyncio.timeout(5.0):
        for event in started:
            await event.wait()


def _record_at_return(outer: asyncio.Task, record: list[str]) -> list[list[str]]:
    """What *record* held the moment *outer* ended, not whenever the test looks."""
    snapshots: list[list[str]] = []
    outer.add_done_callback(lambda _outer: snapshots.append(list(record)))
    return snapshots


async def _parked(started: asyncio.Event) -> None:
    started.set()
    await asyncio.Event().wait()


async def _unwinds_slowly(label: str, started: asyncio.Event, record: list[str], *, cleanup_secs: float) -> None:
    started.set()
    try:
        await asyncio.Event().wait()
    finally:
        await asyncio.sleep(cleanup_secs)
        record.append(f'{label}-unwound')


async def _raises_boom_once(started: asyncio.Event, *siblings_started: asyncio.Event) -> None:
    started.set()
    await _all_started(*siblings_started)
    raise _Boom('raised by one gathered child')


async def _value_after(delay: float, value: str) -> str:
    await asyncio.sleep(delay)
    return value


class TestGatherUnwound:
    async def test_returns_each_childs_result_in_argument_order(self):
        async with asyncio.timeout(5.0):
            results = await gather_unwound(
                _value_after(0.05, 'slowest'), _value_after(0.0, 'fastest'), _value_after(0.02, 'middle'),
            )

        assert results == ['slowest', 'fastest', 'middle']

    async def test_an_outer_cancel_raises_only_after_every_child_has_unwound(self):
        record: list[str] = []
        a_started, b_started = asyncio.Event(), asyncio.Event()
        outer = asyncio.create_task(gather_unwound(
            _unwinds_slowly('a', a_started, record, cleanup_secs=0.2), _parked(b_started),
        ))
        await _all_started(a_started, b_started)
        at_return = _record_at_return(outer, record)

        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5.0):
                await outer

        assert at_return == [['a-unwound']]

    async def test_a_child_already_cancelling_is_not_cancelled_again(self):
        record: list[str] = []
        a_started, b_started = asyncio.Event(), asyncio.Event()

        async def a() -> None:
            a_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    record.append('a-cleanup-interrupted')
                    raise
                record.append('a-cleanup-finished')
                raise

        outer = asyncio.create_task(gather_unwound(a(), _parked(b_started)))
        await _all_started(a_started, b_started)
        at_return = _record_at_return(outer, record)

        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5.0):
                await outer

        assert at_return == [['a-cleanup-finished']]

    async def test_a_child_cancelled_on_its_own_cancels_and_awaits_its_siblings(self):
        record: list[str] = []
        victim_started, sibling_started = asyncio.Event(), asyncio.Event()
        victim = asyncio.create_task(_parked(victim_started))
        outer = asyncio.create_task(gather_unwound(
            victim, _unwinds_slowly('sibling', sibling_started, record, cleanup_secs=0.1),
        ))
        await _all_started(victim_started, sibling_started)
        at_return = _record_at_return(outer, record)

        victim.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5.0):
                await outer

        assert at_return == [['sibling-unwound']]

    async def test_a_failing_child_cancels_its_siblings_and_its_exception_propagates(self):
        record: list[str] = []
        failing_started, sibling_started = asyncio.Event(), asyncio.Event()
        outer = asyncio.create_task(gather_unwound(
            _raises_boom_once(failing_started, sibling_started),
            _unwinds_slowly('sibling', sibling_started, record, cleanup_secs=0.1),
        ))
        at_return = _record_at_return(outer, record)

        with pytest.raises(_Boom):
            async with asyncio.timeout(5.0):
                await outer

        assert failing_started.is_set()
        assert at_return == [['sibling-unwound']]

    async def test_a_plain_future_child_is_cancelled_when_a_sibling_fails(self):
        never_resolved: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        with pytest.raises(_Boom):
            async with asyncio.timeout(5.0):
                await gather_unwound(never_resolved, _raises_boom_once(asyncio.Event()))

        assert never_resolved.cancelled()
