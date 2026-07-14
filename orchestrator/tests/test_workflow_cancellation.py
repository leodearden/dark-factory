"""Tests for the W9-θ ``CancellationScope`` supervisor and ``WorkflowCancelled``.

PRD: plans/workflow-state-machine-prd.md task θ (§8.1 ``WorkflowCancelled``,
§8.2 CX-1, §9 boundary rows 14-15). Replaces three coupled cancellation
mechanisms (harness hard-cancel `task.cancel()` + the ``sys.exc_info()`` B1
sniff / harness B2 dual-guard; the soft ``_cancel_event`` opt-in awaits) with
ONE typed ``WorkflowCancelled(kind)`` caught at exactly one place in
``TaskWorkflow.run()``, and an ordered kind-aware ``on_terminal`` cleanup list.

Test coverage:
  step-01: pure-unit ``WorkflowCancelled`` construct/raise/catch contract
  step-03: ``CancellationScope.supervise`` soft-cancel + normal-return paths
  step-05: ``CancellationScope.supervise`` hard-cancel + repeated-cancel
    cleanup-survival paths
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from orchestrator.workflow_types import CancellationScope, WorkflowCancelled


def _make_recording_on_terminal(
    log: list[tuple[str, str | None]],
    names: tuple[str, ...] = ('a', 'b', 'c'),
    delay: float = 0.0,
) -> list[tuple[str, Callable[[str | None], Awaitable[None]]]]:
    """Build an ordered on_terminal list that appends ``(name, kind)`` to
    *log* for each entry it runs, in the order the scope invokes them —
    the "recording on_terminal list" the plan's soft/hard-cancel tests use
    to pin ordering + kind propagation without a real ``TaskWorkflow``.

    *delay*, when non-zero, makes each entry ``await asyncio.sleep(delay)``
    before recording — giving a robustness test a real window in which to
    fire a second cancel() while cleanup is genuinely in flight.
    """
    entries: list[tuple[str, Callable[[str | None], Awaitable[None]]]] = []
    for name in names:
        async def _fn(kind: str | None, _name: str = name) -> None:
            if delay:
                await asyncio.sleep(delay)
            log.append((_name, kind))
        entries.append((name, _fn))
    return entries

# ---------------------------------------------------------------------------
# step-01: WorkflowCancelled construct/raise/catch/read contract
# ---------------------------------------------------------------------------


class TestWorkflowCancelledType:
    """Pins the frozen-dataclass-Exception construct/raise/catch/read contract
    — the one asyncio-adjacent gotcha to smoke out before anything (the
    ``CancellationScope`` supervisor, ``run()``'s single catch site) depends
    on it.
    """

    def test_constructs_with_hard_kind(self):
        wc = WorkflowCancelled(kind='hard')
        assert isinstance(wc, Exception)
        assert wc.kind == 'hard'

    def test_constructs_with_soft_kind(self):
        wc = WorkflowCancelled(kind='soft')
        assert isinstance(wc, Exception)
        assert wc.kind == 'soft'

    def test_raise_and_catch_hard(self):
        with pytest.raises(WorkflowCancelled) as excinfo:
            raise WorkflowCancelled(kind='hard')
        assert excinfo.value.kind == 'hard'

    def test_raise_and_catch_soft(self):
        with pytest.raises(WorkflowCancelled) as excinfo:
            raise WorkflowCancelled(kind='soft')
        assert excinfo.value.kind == 'soft'

    def test_raise_and_catch_via_except_as_reads_kind(self):
        # The exact idiom run() uses: `except WorkflowCancelled as wc: ... wc.kind`.
        caught_kind = None
        try:
            raise WorkflowCancelled(kind='soft')
        except WorkflowCancelled as wc:
            caught_kind = wc.kind
        assert caught_kind == 'soft'

    @pytest.mark.asyncio
    async def test_raise_and_catch_across_an_await_boundary(self):
        # WorkflowCancelled must survive propagation out of a coroutine —
        # this is how it will actually travel (out of CancellationScope.supervise,
        # an `await`, up to run()'s single catch site).
        async def _raises() -> None:
            await asyncio.sleep(0)
            raise WorkflowCancelled(kind='hard')

        with pytest.raises(WorkflowCancelled) as excinfo:
            await _raises()
        assert excinfo.value.kind == 'hard'


# ---------------------------------------------------------------------------
# step-03: CancellationScope.supervise — soft-cancel + normal-return paths
# ---------------------------------------------------------------------------


class TestCancellationScopeSoftCancel:
    """No real ``TaskWorkflow`` involved — pure asyncio.Event + recording
    on_terminal list, per the plan's step-03 spec.
    """

    @pytest.mark.asyncio
    async def test_soft_cancel_raises_workflow_cancelled_and_runs_on_terminal_in_order(self):
        # Pre-setting the event (rather than racing a delayed setter against
        # supervise()) is deterministic: asyncio.wait({body, waiter},
        # FIRST_COMPLETED) sees waiter already resolvable and body
        # (asyncio.sleep(3600)) still pending, so the soft-cancel branch is
        # the only possible outcome — no timing flakiness.
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log)
        event = asyncio.Event()
        event.set()
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        body_cancelled = False

        async def _body() -> None:
            nonlocal body_cancelled
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                # Proves the scope actually cancelled+cleaned-up the inner
                # body task (rather than e.g. abandoning it) — the closest
                # observable proxy for "body.cancelled() or done" available
                # to a caller that only supplies a coroutine, not a Task.
                body_cancelled = True
                raise

        with pytest.raises(WorkflowCancelled) as excinfo:
            await scope.supervise(_body())

        assert excinfo.value.kind == 'soft'
        assert [name for name, _kind in log] == ['a', 'b', 'c']
        assert all(kind == 'soft' for _name, kind in log)
        assert body_cancelled is True

    @pytest.mark.asyncio
    async def test_body_returns_normally_runs_on_terminal_once_with_none_kind(self):
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log)
        event = asyncio.Event()  # never set — body wins on its own
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        sentinel = object()

        async def _body() -> object:
            return sentinel

        result = await scope.supervise(_body())

        assert result is sentinel
        assert [name for name, _kind in log] == ['a', 'b', 'c']
        assert all(kind is None for _name, kind in log)


# ---------------------------------------------------------------------------
# step-05: CancellationScope.supervise — hard-cancel + robustness
# ---------------------------------------------------------------------------


class TestCancellationScopeHardCancel:
    """No real ``TaskWorkflow`` involved — pure asyncio.Task cancellation +
    recording on_terminal list, per the plan's step-05 spec.
    """

    @pytest.mark.asyncio
    async def test_hard_cancel_of_outer_task_raises_workflow_cancelled_hard(self):
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log)
        event = asyncio.Event()  # never set — only the outer task is cancelled
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        async def _body() -> None:
            await asyncio.sleep(3600)

        async def _runner() -> None:
            await scope.supervise(_body())

        outer = asyncio.create_task(_runner())
        await asyncio.sleep(0)  # let supervise() reach its first real await
        outer.cancel()
        # asyncio.wait (unlike a bare `await outer`) never itself raises,
        # even if outer ends up CANCELLED — lets us inspect the outcome.
        await asyncio.wait({outer}, timeout=5.0)

        assert outer.done()
        assert not outer.cancelled(), (
            'CancelledError escaped supervise() instead of being translated '
            "to WorkflowCancelled(kind='hard')"
        )
        exc = outer.exception()
        assert isinstance(exc, WorkflowCancelled), f'expected WorkflowCancelled, got {exc!r}'
        assert exc.kind == 'hard'
        assert [name for name, _kind in log] == ['a', 'b', 'c']
        assert all(kind == 'hard' for _name, kind in log)

    @pytest.mark.asyncio
    async def test_body_raising_cancellederror_itself_is_treated_as_hard(self):
        # A spontaneous CancelledError raised BY the body (a shutdown-race
        # teardown) — not the outer task being cancelled — must also be
        # typed 'hard', matching the old exc_info sniff's behaviour of
        # catching ANY CancelledError propagating through the finally.
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log)
        event = asyncio.Event()
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        async def _body() -> None:
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

        with pytest.raises(WorkflowCancelled) as excinfo:
            await scope.supervise(_body())

        assert excinfo.value.kind == 'hard'
        assert [name for name, _kind in log] == ['a', 'b', 'c']
        assert all(kind == 'hard' for _name, kind in log)

    @pytest.mark.asyncio
    async def test_repeated_cancel_during_on_terminal_does_not_truncate_cleanup(self):
        # Mimics harness.hard_cancel_workflow's poll loop, which can call
        # task.cancel() more than once on the same slot task.
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log, delay=0.05)
        event = asyncio.Event()
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        async def _body() -> None:
            await asyncio.sleep(3600)

        async def _runner() -> None:
            await scope.supervise(_body())

        outer = asyncio.create_task(_runner())
        await asyncio.sleep(0)
        outer.cancel()  # 1st cancel: enters hard-cancel, starts on_terminal cleanup
        await asyncio.sleep(0.02)  # land mid-sleep inside the first on_terminal entry
        outer.cancel()  # 2nd cancel: must not truncate the still-running cleanup

        await asyncio.wait({outer}, timeout=5.0)

        assert outer.done()
        assert not outer.cancelled(), (
            'CancelledError escaped despite repeated cancel() during on_terminal cleanup'
        )
        exc = outer.exception()
        assert isinstance(exc, WorkflowCancelled), f'expected WorkflowCancelled, got {exc!r}'
        assert exc.kind == 'hard'
        assert [name for name, _kind in log] == ['a', 'b', 'c'], f'cleanup truncated: {log!r}'
        assert all(kind == 'hard' for _name, kind in log)
