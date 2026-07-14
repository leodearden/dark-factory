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
) -> list[tuple[str, Callable[[str | None], Awaitable[None]]]]:
    """Build an ordered on_terminal list that appends ``(name, kind)`` to
    *log* for each entry it runs, in the order the scope invokes them —
    the "recording on_terminal list" the plan's soft/hard-cancel tests use
    to pin ordering + kind propagation without a real ``TaskWorkflow``.
    """
    entries: list[tuple[str, Callable[[str | None], Awaitable[None]]]] = []
    for name in names:
        async def _fn(kind: str | None, _name: str = name) -> None:
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
