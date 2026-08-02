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
  step-07: ``TaskWorkflow.run()`` single-catch site (boundary row 14) — a
    real workflow, harness-style hard-cancelled mid-VERIFY
  step-09: ``_on_terminal_cleanups()`` ordering + kind-aware lane release
    (1:1 replacement of the deleted exc_info ``_hard_cancel`` skip)
  step-11: boundary row 15 — soft-cancel covers a new (unwrapped) await via
    the scope's own body-task race, and ``_await_cancellable`` raises
    ``WorkflowCancelled('soft')`` (orphan-avoidance detach preserved)
  step-13: harness B2 + ``synthetic_cancel`` retirement — ``TaskReport``
    can no longer be constructed with the field, the harness's hard-cancel
    safety-net report carries no such attribute, and the ``except
    asyncio.CancelledError`` safety net (for cancels landing OUTSIDE
    run()'s CancellationScope) keeps working without it
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Cross-module reuse (task 2610 precedent — see test_workflow_terminal_report.py
# for the same import shape): these factories live in _workflow_helpers.py, not
# in any single test module's private namespace.
from _orch_helpers import (
    CANCEL_SCOPE_BARRIER_TIMEOUT,
    CANCEL_SCOPE_PURE_UNIT_TIMEOUT,
    _init_harness_state_for_test,
    wire_scheduler_liveness_mock,
)
from _workflow_helpers import (
    AgentStub,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness, TaskReport
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow_types import (
    CancellationScope,
    TerminalReport,
    WorkflowCancelled,
    WorkflowOutcome,
    WorkflowState,
    outcome_allows_status,
)


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
    async def test_body_raising_workflow_cancelled_directly_propagates_its_own_kind(self):
        # W9-θ step-12 discovery: application code inside the body (e.g. the
        # merge-retry loop's explicit cancel re-check, or _await_cancellable)
        # can now raise WorkflowCancelled directly as ordinary control flow —
        # a DIFFERENT detection path than this scope's own event-race (tested
        # above). supervise() must still capture the raised exception's own
        # .kind and pass it to on_terminal — not leave `kind` at its default
        # None, which would make on_terminal treat a real soft-cancel as a
        # normal exit and silently skip the kind-aware lane-release policy.
        log: list[tuple[str, str | None]] = []
        on_terminal = _make_recording_on_terminal(log)
        event = asyncio.Event()  # never set — the body raises on its own
        scope = CancellationScope(cancel_event=event, on_terminal=on_terminal)

        async def _body() -> None:
            await asyncio.sleep(0)
            raise WorkflowCancelled(kind='soft')

        with pytest.raises(WorkflowCancelled) as excinfo:
            await scope.supervise(_body())

        assert excinfo.value.kind == 'soft'
        assert [name for name, _kind in log] == ['a', 'b', 'c']
        assert all(kind == 'soft' for _name, kind in log), (
            f'on_terminal must see kind=soft when the body raises WorkflowCancelled '
            f'directly, got {log!r}'
        )

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

    task 3307: barriers below use the small, pure-in-memory
    ``CANCEL_SCOPE_PURE_UNIT_TIMEOUT`` — NOT the real-I/O-sized
    ``CANCEL_SCOPE_BARRIER_TIMEOUT`` — so this class keeps pytest's 60s
    default hang detector. See _orch_helpers for the rationale.
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
        await asyncio.wait({outer}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)

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

        await asyncio.wait({outer}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)

        assert outer.done()
        assert not outer.cancelled(), (
            'CancelledError escaped despite repeated cancel() during on_terminal cleanup'
        )
        exc = outer.exception()
        assert isinstance(exc, WorkflowCancelled), f'expected WorkflowCancelled, got {exc!r}'
        assert exc.kind == 'hard'
        assert [name for name, _kind in log] == ['a', 'b', 'c'], f'cleanup truncated: {log!r}'
        assert all(kind == 'hard' for _name, kind in log)


# ---------------------------------------------------------------------------
# step-07: TaskWorkflow.run() single-catch site (boundary row 14)
# ---------------------------------------------------------------------------
#
# Fixtures mirror test_workflow_terminal_report.py's e2e-style setup (task
# 2610 Group D factories: _build_workflow / AgentStub / _init_repo) — a REAL
# TaskWorkflow driven through PLAN + EXECUTE via the AgentStub, then wedged
# inside VERIFY so a harness-style hard-cancel (task.cancel() on the run()
# task) has something to interrupt.


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 3307: must exceed 2x CANCEL_SCOPE_BARRIER_TIMEOUT (45s) below
class TestRunSingleCatchHardCancel:
    """Boundary row 14: ``run()`` must RETURN a ``TerminalReport`` on a
    harness-style hard-cancel, never let ``CancelledError`` escape — and
    must not crash on SM-2 (the outcome<->status half is skipped for the
    hard-cancel exit, since the live scheduler row is still 'in-progress',
    which is NOT an allowed pairing for outcome==CANCELLED).

    RED until step-8 wires the ``CancellationScope`` into ``run()``: today
    ``run()`` is a bare ``await self._drive()`` with no catch, so cancelling
    the run()-task leaves that task itself CANCELLED instead of returning.

    task 3307: barriers below use ``CANCEL_SCOPE_BARRIER_TIMEOUT`` — see
    _orch_helpers for the measurement basis and never-narrow rule.
    """

    async def test_hard_cancel_mid_verify_returns_cancelled_report(
        self,
        config,
        git_ops,
        task_assignment,
        monkeypatch,
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        wedged = asyncio.Event()

        async def _wedge_verify() -> WorkflowOutcome:
            # Entered only after _enter_phase(VERIFY) (workflow.py, just
            # above the real _verify_debugfix_loop() call) — so by the time
            # this fires, workflow.machine.state is already VERIFY.
            wedged.set()
            await asyncio.sleep(3600)
            raise AssertionError('unreachable — cancelled before the sleep returns')

        workflow._verify_debugfix_loop = _wedge_verify  # type: ignore[method-assign]

        run_task = asyncio.create_task(workflow.run())
        await asyncio.wait_for(wedged.wait(), timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)
        assert workflow.machine.state == WorkflowState.VERIFY

        # Harness-style hard-cancel: cancel the task running run() directly
        # (mirrors harness.hard_cancel_workflow's task.cancel() on the
        # registered slot task).
        run_task.cancel()
        done, _pending = await asyncio.wait({run_task}, timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)
        assert run_task in done, f'run() task did not finish within {CANCEL_SCOPE_BARRIER_TIMEOUT}s'

        assert not run_task.cancelled(), (
            'CancelledError escaped run() instead of being translated into '
            'a TerminalReport(outcome=CANCELLED)'
        )
        report = run_task.result()
        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.CANCELLED
        assert workflow.machine.state == WorkflowState.CANCELLED
        assert report.phase == WorkflowState.CANCELLED

        # The live scheduler row was claimed 'in-progress' by
        # _setup_worktree_and_artifacts and never advanced (no terminal
        # status write happens on a hard-cancel exit) — proving SM-2's
        # outcome<->status half really was SKIPPED for this exit, not just
        # coincidentally satisfied: 'in-progress' is NOT an allowed pairing
        # for outcome==CANCELLED (_OUTCOME_ALLOWED['cancelled'] == {CANCELLED}
        # only), yet run() above returned cleanly instead of raising.
        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'in-progress'
        assert not outcome_allows_status(report.outcome, last_status)


@pytest.mark.asyncio
class TestCancelFromMergeDeferred:
    """Regression (reviewer_comprehensive, esc-2252-26): a cancel — soft or
    hard — that lands while the machine is parked in ``MERGE_DEFERRED`` (a
    train member awaiting its GroupMergeRequest future inside
    ``_maybe_enqueue_group_merge``, after ``set_task_status('merge-deferred')``
    has persisted the row) must let ``run()`` RETURN a ``TerminalReport``.

    Before the fix, ``_finalise_cancellation`` drove the machine to CANCELLED
    via ``_enter_phase(CANCELLED)``, but the shared transition table had NO
    ``(MERGE_DEFERRED, CANCELLED)`` edge → ``IllegalTransition`` escaped run()'s
    single ``WorkflowCancelled`` catch. The soft path additionally tripped
    SM-2's outcome<->status half: ``outcome_allows_status('soft-cancelled',
    'merge-deferred')`` was False. Both are fixed in shared/task_transitions.py.
    """

    def _wire_cleanup_spies(self, workflow) -> None:
        async def _noop_async(*_a, **_k) -> None:
            return None

        async def _release_lane(_task_id: str) -> bool:
            return True

        workflow._stop_claimant_heartbeat = _noop_async  # type: ignore[method-assign]
        workflow._steward = SimpleNamespace(stop=_noop_async)
        workflow._maybe_cleanup_done_worktree = _noop_async  # type: ignore[method-assign]
        workflow._cleanup_config_dir = lambda *_a, **_k: None  # type: ignore[method-assign]
        workflow.git_ops.release_lane_for_terminal_task = _release_lane  # type: ignore[method-assign]

    async def _stage_merge_deferred(self, workflow, scheduler) -> None:
        # Persist the merge-deferred row and stage the machine there — exactly
        # the state a parked train member sits in at _await_cancellable(future)
        # (workflow.py:1320), reached after set_task_status('merge-deferred').
        await scheduler.set_task_status(workflow.task_id, 'merge-deferred')
        workflow.state = WorkflowState.MERGE_DEFERRED
        assert workflow.machine.state == WorkflowState.MERGE_DEFERRED
        self._wire_cleanup_spies(workflow)

    async def test_hard_cancel_from_merge_deferred_returns_cancelled_report(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        await self._stage_merge_deferred(workflow, scheduler)

        async def _drive_raises_hard() -> WorkflowOutcome:
            raise WorkflowCancelled('hard')

        workflow._drive = _drive_raises_hard  # type: ignore[method-assign]

        report = await workflow.run()

        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.CANCELLED
        assert workflow.machine.state == WorkflowState.CANCELLED
        assert report.phase == WorkflowState.CANCELLED

    async def test_soft_cancel_from_merge_deferred_returns_soft_cancelled_report(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        await self._stage_merge_deferred(workflow, scheduler)
        # A soft-cancel: the event is set, and _handle_soft_cancel re-reads the
        # (non-terminal) merge-deferred row → SOFT_CANCELLED.
        workflow._cancel_event.set()

        async def _drive_raises_soft() -> WorkflowOutcome:
            raise WorkflowCancelled('soft')

        workflow._drive = _drive_raises_soft  # type: ignore[method-assign]

        report = await workflow.run()

        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.SOFT_CANCELLED
        assert workflow.machine.state == WorkflowState.CANCELLED
        assert report.phase == WorkflowState.CANCELLED
        # SM-2 outcome<->status: the last-persisted row is still merge-deferred
        # (release_workflow parks it to blocked only after run() returns), and
        # that pairing must be allowed for the soft-cancelled outcome.
        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'merge-deferred'
        assert outcome_allows_status(report.outcome, last_status)


# ---------------------------------------------------------------------------
# step-09: _on_terminal_cleanups() ordering + kind-aware lane release
# ---------------------------------------------------------------------------
#
# Exercises _on_terminal_cleanups() directly (no run()/_drive() involved) —
# a real TaskWorkflow (config/git_ops/task_assignment fixtures shared with
# step-07 above), its .state staged directly (the setter bypasses transition
# validation), and the five cleanup targets replaced with spies that append
# their name to a shared `calls` log. This pins the 1:1 replacement of the
# deleted exc_info `_hard_cancel` skip: release iff
# `kind != 'hard' and not _worktree_external and (kind is not None or
# state in {DONE, CANCELLED})`.
#
# RED until step-10: _on_terminal_cleanups() is still the step-8 placeholder
# returning [] — every scenario below observes an empty `calls` log.


@pytest.mark.asyncio
class TestOnTerminalCleanups:
    """Boundary row 15 prep: ``_on_terminal_cleanups()`` ordering + the
    kind-aware lane-release guard."""

    def _wire_spies(self, workflow, calls: list[str]) -> None:
        async def _heartbeat() -> None:
            calls.append('stop_claimant_heartbeat')

        async def _steward_stop() -> None:
            calls.append('stop_steward')

        async def _cleanup_worktree() -> None:
            calls.append('cleanup_done_worktree')

        def _cleanup_config() -> None:
            calls.append('cleanup_config_dir')

        async def _release_lane(task_id: str) -> bool:
            calls.append('release_lane')
            return True

        workflow._stop_claimant_heartbeat = _heartbeat  # type: ignore[method-assign]
        workflow._steward = SimpleNamespace(stop=_steward_stop)
        workflow._maybe_cleanup_done_worktree = _cleanup_worktree  # type: ignore[method-assign]
        workflow._cleanup_config_dir = _cleanup_config  # type: ignore[method-assign]
        workflow.git_ops.release_lane_for_terminal_task = _release_lane  # type: ignore[method-assign]

    async def _run_cleanups(self, workflow, kind: str | None) -> None:
        for _name, fn in workflow._on_terminal_cleanups():
            await fn(kind)

    async def test_ordering_with_kind_none_and_done_state(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """A genuine DONE exit (kind=None): all five run, in order, release fires."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow.state = WorkflowState.DONE
        calls: list[str] = []
        self._wire_spies(workflow, calls)

        await self._run_cleanups(workflow, None)

        assert calls == [
            'stop_claimant_heartbeat',
            'stop_steward',
            'cleanup_done_worktree',
            'release_lane',
            'cleanup_config_dir',
        ]

    async def test_ordering_with_kind_none_and_cancelled_state(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """The authoritative-cancel exit (kind=None, state already CANCELLED
        via _handle_cancelled_terminal_exit) also releases — row @263's
        pre-existing property, preserved."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow.state = WorkflowState.CANCELLED
        calls: list[str] = []
        self._wire_spies(workflow, calls)

        await self._run_cleanups(workflow, None)

        assert calls == [
            'stop_claimant_heartbeat',
            'stop_steward',
            'cleanup_done_worktree',
            'release_lane',
            'cleanup_config_dir',
        ]

    async def test_kind_hard_skips_lane_release_but_other_four_still_run(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """kind='hard' skips release EVEN when state is already terminal —
        the branch must survive the teardown regardless of state."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow.state = WorkflowState.CANCELLED
        calls: list[str] = []
        self._wire_spies(workflow, calls)

        await self._run_cleanups(workflow, 'hard')

        assert 'release_lane' not in calls
        assert calls == [
            'stop_claimant_heartbeat',
            'stop_steward',
            'cleanup_done_worktree',
            'cleanup_config_dir',
        ]

    async def test_kind_soft_releases_lane_even_from_a_working_state(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """kind='soft' releases even mid-flight (state is VERIFY, not yet
        terminal) — boundary row 15's key property."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow.state = WorkflowState.VERIFY
        calls: list[str] = []
        self._wire_spies(workflow, calls)

        await self._run_cleanups(workflow, 'soft')

        assert 'release_lane' in calls

    async def test_kind_none_with_working_state_skips_release(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """kind=None (a normal _drive() return) with a non-terminal state
        is not a genuine terminal exit — skip release."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow.state = WorkflowState.VERIFY
        calls: list[str] = []
        self._wire_spies(workflow, calls)

        await self._run_cleanups(workflow, None)

        assert 'release_lane' not in calls

    async def test_worktree_external_skips_release_for_every_kind(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """Eval mode (_worktree_external=True) never releases, regardless
        of kind."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow._worktree_external = True
        workflow.state = WorkflowState.CANCELLED

        for kind in ('hard', 'soft', None):
            calls: list[str] = []
            self._wire_spies(workflow, calls)
            await self._run_cleanups(workflow, kind)
            assert 'release_lane' not in calls, f'release fired for kind={kind!r}'


# ---------------------------------------------------------------------------
# step-11: boundary row 15 — soft-cancel covers a new await + orphan-avoidance
# ---------------------------------------------------------------------------
#
# (a) The CancellationScope races the WHOLE _drive() body-task against
# _cancel_event (steps 03/06/08) — so ANY long await inside _drive() is
# cancellable by construction, not just ones wrapped by _await_cancellable.
# This pins that property end-to-end against a real TaskWorkflow, using a
# patched steward-shaped wait _await_cancellable never sees.
#
# (b) _await_cancellable itself (the merge-submit orphan-avoidance helper)
# currently returns None on cancel-win; callers do
# `if result is None: return await self._handle_soft_cancel(...)` — a normal
# _drive() return that never reaches run()'s single WorkflowCancelled catch
# with kind='soft'.  RED until step-12 retypes it to RAISE
# WorkflowCancelled('soft') (keeping the fut.cancel()/on_soft_cancel detach).


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 3307: must exceed 2x CANCEL_SCOPE_BARRIER_TIMEOUT (45s) below
class TestSoftCancelCoversNewAwait:
    """Boundary row 15(a): a soft-cancel during a long await NOT wrapped by
    ``_await_cancellable`` is still caught by the ``CancellationScope``'s own
    body-task race — ``run()`` returns ``TerminalReport(SOFT_CANCELLED)``,
    ``machine.state == CANCELLED``, and the lane is released (kind='soft').

    task 3307: barriers below use ``CANCEL_SCOPE_BARRIER_TIMEOUT`` — see
    _orch_helpers for the measurement basis and never-narrow rule.
    """

    async def test_soft_cancel_during_unwrapped_await_returns_soft_cancelled_report(
        self,
        config,
        git_ops,
        task_assignment,
        monkeypatch,
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        wedged = asyncio.Event()
        release_calls: list[str] = []

        async def _wedge_verify() -> WorkflowOutcome:
            # A stand-in for a long, non-_await_cancellable-wrapped wait (e.g.
            # a steward wait) — entered only after _enter_phase(VERIFY).
            wedged.set()
            await asyncio.sleep(3600)
            raise AssertionError('unreachable — soft-cancelled before the sleep returns')

        async def _spy_release_lane(task_id: str) -> bool:
            release_calls.append(task_id)
            return True

        workflow._verify_debugfix_loop = _wedge_verify  # type: ignore[method-assign]
        workflow.git_ops.release_lane_for_terminal_task = _spy_release_lane  # type: ignore[method-assign]

        run_task = asyncio.create_task(workflow.run())
        await asyncio.wait_for(wedged.wait(), timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)
        assert workflow.machine.state == WorkflowState.VERIFY

        # Soft-cancel: set the event directly — no task.cancel() involved.
        # Mirrors a human release_workflow / watcher-triggered soft-cancel
        # arriving while wedged on a wait _await_cancellable never sees.
        workflow._cancel_event.set()
        done, _pending = await asyncio.wait({run_task}, timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)
        assert run_task in done, f'run() task did not finish within {CANCEL_SCOPE_BARRIER_TIMEOUT}s'
        assert not run_task.cancelled(), 'CancelledError escaped run() on a soft-cancel'

        report = run_task.result()
        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.SOFT_CANCELLED
        assert workflow.machine.state == WorkflowState.CANCELLED
        assert report.phase == WorkflowState.CANCELLED
        assert release_calls == ['42'], (
            'kind=soft must release the lane even from a still-working state'
        )


# ---------------------------------------------------------------------------
# task 3307 (reviewer follow-up): never-narrow guards for the two barrier
# constants
# ---------------------------------------------------------------------------
#
# An earlier version of this amendment drove a full real TaskWorkflow
# through an injected 6s-slow prologue (TestCancelBarrierToleratesSlowPrologue)
# to prove CANCEL_SCOPE_BARRIER_TIMEOUT > 6.0. Review found that too weak
# (it passes for any value >~7, so a narrowing 45 -> 10 would still sail
# through green) and too expensive (~14s of unconditional asyncio.sleep,
# the two slowest tests in this module by ~5x) for what it actually pinned
# beyond TestRunSingleCatchHardCancel / TestSoftCancelCoversNewAwait above
# — those already fully cover the cancel -> TerminalReport contract this
# class re-asserted under a slow prologue. These two zero-cost assertions
# pin the same never-narrow invariant directly, so a future edit that
# quietly shrinks either constant back toward its retired literal fails
# instantly instead of surfacing as an intermittent multi-second timeout.


def test_cancel_scope_barrier_timeout_never_narrowed():
    """See _orch_helpers.CANCEL_SCOPE_BARRIER_TIMEOUT: must stay large
    enough to safely replace every retired literal <=15 it was introduced
    to cover."""
    assert CANCEL_SCOPE_BARRIER_TIMEOUT >= 15


def test_cancel_scope_pure_unit_timeout_never_narrowed():
    """See _orch_helpers.CANCEL_SCOPE_PURE_UNIT_TIMEOUT: must stay >= the
    retired 5.0s literal it replaces in TestCancellationScopeHardCancel."""
    assert CANCEL_SCOPE_PURE_UNIT_TIMEOUT >= 5


@pytest.mark.asyncio
class TestAwaitCancellableRaisesWorkflowCancelled:
    """Boundary row 15(b): ``_await_cancellable`` raises
    ``WorkflowCancelled(kind='soft')`` on cancel-win instead of returning
    ``None``, while still performing its orphan-avoidance detach
    (``fut.cancel()`` / ``on_soft_cancel``) so an enqueued merge request is
    never orphaned."""

    async def test_hook_called_future_not_cancelled_and_raises(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """cancel-win with an on_soft_cancel hook: hook fires exactly once,
        the future is left untouched (registry.detach owns it instead), and
        WorkflowCancelled('soft') is raised — not a None return."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow._cancel_event.set()  # cancel wins immediately

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        hook_calls: list[int] = []

        with pytest.raises(WorkflowCancelled) as excinfo:
            await workflow._await_cancellable(fut, on_soft_cancel=lambda: hook_calls.append(1))

        assert excinfo.value.kind == 'soft'
        assert hook_calls == [1], 'hook must be called exactly once'
        assert not fut.cancelled(), 'future must NOT be cancelled when hook is provided'

    async def test_no_hook_future_is_cancelled_and_raises(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """cancel-win with no hook (default None): the blanket fut.cancel()
        orphan-avoidance still fires, and WorkflowCancelled('soft') is raised."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        workflow._cancel_event.set()

        fut: asyncio.Future = asyncio.get_running_loop().create_future()

        with pytest.raises(WorkflowCancelled) as excinfo:
            await workflow._await_cancellable(fut)

        assert excinfo.value.kind == 'soft'
        assert fut.cancelled(), 'future must be cancelled when no hook is provided'

    async def test_future_resolves_first_returns_normally_no_raise(
        self,
        config,
        git_ops,
        task_assignment,
    ):
        """Same-window race: the awaitable's result wins over a set cancel
        event — no WorkflowCancelled, no hook call (unaffected by step-12)."""
        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        # cancel_event is NOT set.

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result('the_result')
        hook_calls: list[int] = []

        result = await workflow._await_cancellable(fut, on_soft_cancel=lambda: hook_calls.append(1))

        assert result == 'the_result'
        assert hook_calls == [], 'hook must NOT be called when future resolves first'


# ---------------------------------------------------------------------------
# step-13: harness B2 + synthetic_cancel retirement (RED)
# ---------------------------------------------------------------------------
#
# With the workflow's kind-aware on_terminal now solely owning terminal lane
# release for every run() exit (steps 8/10/12), the harness's B2 belt-and-
# suspenders release block — and its synthetic_cancel skip-signal — is fully
# redundant. But task.cancel() can still land OUTSIDE run()'s
# CancellationScope (slot setup, post-run() report building, or a unit test
# that mocks workflow.run() as a bare coroutine), so the harness's `except
# asyncio.CancelledError` safety net must survive the retirement — it still
# has to return a TaskReport(outcome=CANCELLED) so the wrapper task
# completes normally.
#
# RED until step-14 deletes TaskReport.synthetic_cancel and the B2 block
# from harness.py: today the field still exists (TaskReport(...
# synthetic_cancel=True) does NOT raise) and the harness's hard-cancel
# safety-net report still carries the attribute.


def _make_harness_for_run_slot() -> Harness:
    """Harness with all attributes needed to drive ``_run_slot`` directly.

    Mirrors test_harness_cancel_workflow.py's ``harness_for_run_slot``
    fixture (that file mocks run() as a bare sleep and asserts on
    outcome/cleanup/sem only — none of which depend on synthetic_cancel, so
    it is untouched by θ). This local helper reuses the same construction
    shape for θ's own retirement test below.
    """
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
    h._workflow_cancel_events = {}
    h._workflow_cancel_at = {}
    h._workflow_slot_tasks = {}
    h._terminal_cancel_counts = {}
    h._escalation_events = {}
    h._escalation_queue = None
    h._recovered_plans = {}
    h._recovered_sessions = {}
    h._recovered_session_config_dirs = {}
    h._preserved_worktrees = set()
    h.event_store = None
    h.config = None  # type: ignore[assignment]
    h.git_ops = None  # type: ignore[assignment]
    h.briefing = None  # type: ignore[assignment]
    h.mcp = None  # type: ignore[assignment]
    h.usage_gate = None
    h._merge_queue = None  # type: ignore[assignment]
    h._merge_worker = None
    h._merge_inflight_registry = None  # type: ignore[assignment]
    h.cost_store = None
    h._run_store = None
    h._run_id = None
    h.review_checkpoint = None
    h.scheduler.release = MagicMock()
    h.scheduler.carries_substrate_probe = MagicMock(return_value=False)
    h.scheduler.is_deterministic = MagicMock(return_value=False)
    return h


@pytest.mark.timeout(180)  # task 3307: must exceed 2x CANCEL_SCOPE_BARRIER_TIMEOUT (45s) below
class TestHarnessSyntheticCancelRetirement:
    """RED (step-13): ``TaskReport`` sheds ``synthetic_cancel`` — it can no
    longer be constructed with the field, and the harness's hard-cancel
    safety-net report carries no such attribute (field + B2 release block
    retired); the ``except asyncio.CancelledError`` safety net keeps
    returning a CANCELLED report with cleanup/sem intact.

    task 3307: the poll and the wait below use ``CANCEL_SCOPE_BARRIER_TIMEOUT``
    — see _orch_helpers for the measurement basis and never-narrow rule.
    """

    def test_task_report_construction_rejects_synthetic_cancel_kwarg(self):
        with pytest.raises(TypeError):
            TaskReport(
                task_id='42',
                title='t',
                outcome=WorkflowOutcome.CANCELLED,
                synthetic_cancel=True,  # type: ignore[call-arg]
            )

    @pytest.mark.asyncio
    async def test_hard_cancel_safety_net_returns_cancelled_report_without_synthetic_cancel(
        self,
    ):
        """Mirrors test_harness_cancel_workflow.py's hard-cancel integration
        test — drive ``_run_slot`` with ``workflow.run`` patched to a bare
        ``asyncio.sleep(3600)``, hard-cancel it — but additionally pins that
        the returned report carries no ``synthetic_cancel`` attribute at
        all, proving the except-CancelledError net still works (outcome,
        finally cleanup, sem release) without the retired flag."""
        h = _make_harness_for_run_slot()
        tid = '42'
        assignment = TaskAssignment(
            task_id=tid,
            task={'title': 'wedged task'},
            modules=[],
        )
        sem = asyncio.Semaphore(0)

        with patch('orchestrator.harness.build_workflow') as mock_wf_cls:

            async def _wedge() -> None:
                await asyncio.sleep(3600)

            mock_wf = MagicMock()
            mock_wf.run = _wedge
            mock_wf_cls.return_value = mock_wf

            wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))

            # Poll until _run_slot registers itself in _workflow_slot_tasks.
            # task 3307: monotonic-deadline poll (house idiom, e.g.
            # test_merge_queue.py:5512-5522) replacing the old fixed 50 x
            # 0.01s = 0.5s iteration budget — a synchronization barrier on
            # real scheduling latency, not a timing assertion.
            deadline = time.monotonic() + CANCEL_SCOPE_BARRIER_TIMEOUT
            while tid not in h._workflow_slot_tasks:
                if time.monotonic() >= deadline:
                    pytest.fail(
                        f'_run_slot did not register itself in _workflow_slot_tasks '
                        f'within {CANCEL_SCOPE_BARRIER_TIMEOUT}s'
                    )
                await asyncio.sleep(0.01)

            h.hard_cancel_workflow(tid)

            done, _pending = await asyncio.wait(
                {wrapper_task}, timeout=CANCEL_SCOPE_BARRIER_TIMEOUT
            )
            assert wrapper_task in done, (
                f'wrapper_task did not finish within {CANCEL_SCOPE_BARRIER_TIMEOUT}s'
            )

        assert not wrapper_task.cancelled(), (
            'Expected _run_slot to return a synthetic CANCELLED TaskReport, '
            'but the Task ended up in CANCELLED state — CancelledError escaped.'
        )
        report = wrapper_task.result()
        assert report is not None, 'Expected _run_slot to return a TaskReport, got None'
        assert report.outcome == WorkflowOutcome.CANCELLED, (
            f'Expected outcome=CANCELLED, got {report.outcome!r}'
        )
        assert not hasattr(report, 'synthetic_cancel'), (
            'TaskReport must no longer carry a synthetic_cancel attribute'
        )

        # Finally cleanup ran: registries cleared, semaphore released.
        assert tid not in h._workflow_slot_tasks, 'slot task not cleaned up in finally'
        assert tid not in h._workflow_cancel_events, 'cancel event not cleaned up in finally'
        assert sem._value == 1, f'semaphore not released by finally (value={sem._value})'


# ---------------------------------------------------------------------------
# task 3412: shared cancel-test helpers — _make_wedge / _assert_cancel_report
# ---------------------------------------------------------------------------
#
# TestRunSingleCatchHardCancel and TestSoftCancelCoversNewAwait independently
# built an identical wedging _verify_debugfix_loop stand-in and an identical
# 8-line post-cancel TerminalReport assertion block (modulo the expected
# outcome). These two module-level helpers de-duplicate that, contract-tested
# here per the test_conftest_helpers.py precedent — a bug in either helper
# would make BOTH refactored call sites pass vacuously, so the negative
# (must-raise) cases below are load-bearing, not decorative.


def _make_wedge(event: asyncio.Event) -> Callable[[], Awaitable[WorkflowOutcome]]:
    """Build a stand-in for ``TaskWorkflow._verify_debugfix_loop`` that signals
    *event* and then wedges forever, so a cancel has something to interrupt.

    Assign it with ``workflow._verify_debugfix_loop = _make_wedge(ev)``: the
    real method is entered only after ``_enter_phase(VERIFY)`` (workflow.py,
    just above the real ``_verify_debugfix_loop()`` call), so by the time
    *event* is set ``workflow.machine.state`` is already VERIFY.
    """

    async def _wedge_verify() -> WorkflowOutcome:
        event.set()
        await asyncio.sleep(3600)
        raise AssertionError('unreachable — the wedge must be cancelled before the sleep returns')

    return _wedge_verify


@pytest.mark.asyncio
class TestMakeWedgeHelper:
    """Contract tests for the ``_make_wedge`` factory helper.

    Pins the property both cancel-report call sites depend on: the returned
    coroutine function signals its event and then wedges until cancelled,
    and building the wedge has no side effect until the coroutine actually
    runs.
    """

    async def test_returns_a_callable_that_sets_the_event_then_blocks(self):
        ev = asyncio.Event()
        fn = _make_wedge(ev)
        assert not ev.is_set(), '_make_wedge must not touch the event before its coroutine runs'

        t = asyncio.create_task(fn())
        try:
            await asyncio.wait_for(ev.wait(), timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)
            assert not t.done(), 'the wedge must block after signalling, not return'
        finally:
            t.cancel()
            await asyncio.wait({t}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)

    async def test_wedge_is_cancellable_and_propagates_cancellederror(self):
        ev = asyncio.Event()
        fn = _make_wedge(ev)
        t = asyncio.create_task(fn())
        await asyncio.wait_for(ev.wait(), timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)

        t.cancel()
        await asyncio.wait({t}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)
        assert t.cancelled(), 'a cancelled wedge task must end up in the cancelled state'

    async def test_each_call_builds_an_independent_wedge(self):
        a = asyncio.Event()
        b = asyncio.Event()
        fn_a = _make_wedge(a)
        fn_b = _make_wedge(b)

        t_a = asyncio.create_task(fn_a())
        try:
            await asyncio.wait_for(a.wait(), timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)
            assert a.is_set()
            assert not b.is_set(), 'the wedge built for `a` must not also set `b`'
        finally:
            t_a.cancel()
            await asyncio.wait({t_a}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)

        t_b = asyncio.create_task(fn_b())
        try:
            await asyncio.wait_for(b.wait(), timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)
            assert b.is_set()
        finally:
            t_b.cancel()
            await asyncio.wait({t_b}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)


async def _assert_cancel_report(
    run_task: asyncio.Task,
    workflow,
    expected_outcome: WorkflowOutcome,
) -> TerminalReport:
    """Await a just-cancelled ``run()`` task and pin the shared cancel
    contract: run() RETURNED a ``TerminalReport`` of *expected_outcome*
    instead of letting ``CancelledError`` escape, and both the machine and
    the report landed on ``WorkflowState.CANCELLED``.

    Returns the report so the caller can add its own site-specific tail
    assertions.  The barrier is deliberately not parameterised — see the
    never-narrow guard at ``test_cancel_scope_barrier_timeout_never_narrowed``.
    """
    done, _pending = await asyncio.wait({run_task}, timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)
    assert run_task in done, f'run() task did not finish within {CANCEL_SCOPE_BARRIER_TIMEOUT}s'
    assert not run_task.cancelled(), (
        'CancelledError escaped run() instead of being translated into '
        f'a TerminalReport(outcome={expected_outcome.name})'
    )
    report = run_task.result()
    assert isinstance(report, TerminalReport), f'expected TerminalReport, got {report!r}'
    assert report.outcome == expected_outcome
    assert workflow.machine.state == WorkflowState.CANCELLED
    assert report.phase == WorkflowState.CANCELLED
    return report


def _done_task(result: object) -> asyncio.Task:
    """A Task that resolves to *result* almost immediately.

    Lets :class:`TestAssertCancelReportHelper` build a fake "just-finished
    run()" task without a real cancellation dance — helper-local to these
    contract tests, not one of the two extracted helpers itself.
    """

    async def _return_it() -> object:
        return result

    return asyncio.create_task(_return_it())


@pytest.mark.asyncio
class TestAssertCancelReportHelper:
    """Contract tests for the ``_assert_cancel_report`` helper.

    The negative cases are the load-bearing part of this class: an
    assertion helper that silently asserts nothing would make BOTH
    TestRunSingleCatchHardCancel and TestSoftCancelCoversNewAwait pass
    vacuously after the step-5/6 extraction, invisibly weakening the suite.
    Pinning that the helper actually RAISES on each of the five ways a
    cancel exit can go wrong is what makes the extraction safe.

    Deliberately NOT tested here: the "run() task did not finish within
    Ns" branch. Exercising it costs a real CANCEL_SCOPE_BARRIER_TIMEOUT
    (45s) wait, which is exactly the kind of multi-second unconditional
    sleep task 3307 retired from this module (see the comment block above
    test_cancel_scope_barrier_timeout_never_narrowed). This omission is
    deliberate, not an oversight.
    """

    @pytest.mark.parametrize('outcome', [WorkflowOutcome.CANCELLED, WorkflowOutcome.SOFT_CANCELLED])
    async def test_returns_the_report_on_a_clean_cancelled_exit(self, outcome):
        report = TerminalReport(
            outcome=outcome,
            reason='r',
            phase=WorkflowState.CANCELLED,
            detail='d',
            category=None,
        )
        run_task = _done_task(report)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.CANCELLED))

        result = await _assert_cancel_report(run_task, workflow, outcome)

        assert result is report, 'the helper must return the SAME report object, not a copy'

    async def test_raises_when_run_task_ended_cancelled(self):
        async def _never() -> TerminalReport:
            await asyncio.sleep(3600)
            raise AssertionError('unreachable — the task must be cancelled before the sleep returns')

        run_task = asyncio.create_task(_never())
        await asyncio.sleep(0)  # let it actually start before cancelling
        run_task.cancel()
        await asyncio.wait({run_task}, timeout=CANCEL_SCOPE_PURE_UNIT_TIMEOUT)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.CANCELLED))

        with pytest.raises(AssertionError, match='CANCELLED'):
            await _assert_cancel_report(run_task, workflow, WorkflowOutcome.CANCELLED)

    async def test_raises_on_outcome_mismatch(self):
        report = TerminalReport(
            outcome=WorkflowOutcome.CANCELLED,
            reason='r',
            phase=WorkflowState.CANCELLED,
            detail='d',
            category=None,
        )
        run_task = _done_task(report)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.CANCELLED))

        with pytest.raises(AssertionError):
            await _assert_cancel_report(run_task, workflow, WorkflowOutcome.SOFT_CANCELLED)

    async def test_raises_when_machine_state_is_not_cancelled(self):
        report = TerminalReport(
            outcome=WorkflowOutcome.CANCELLED,
            reason='r',
            phase=WorkflowState.CANCELLED,
            detail='d',
            category=None,
        )
        run_task = _done_task(report)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.VERIFY))

        with pytest.raises(AssertionError):
            await _assert_cancel_report(run_task, workflow, WorkflowOutcome.CANCELLED)

    async def test_raises_when_report_phase_is_not_cancelled(self):
        report = TerminalReport(
            outcome=WorkflowOutcome.CANCELLED,
            reason='r',
            phase=WorkflowState.BLOCKED,
            detail='d',
            category=None,
        )
        run_task = _done_task(report)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.CANCELLED))

        with pytest.raises(AssertionError):
            await _assert_cancel_report(run_task, workflow, WorkflowOutcome.CANCELLED)

    async def test_raises_when_result_is_not_a_terminal_report(self):
        run_task = _done_task(None)
        workflow = SimpleNamespace(machine=SimpleNamespace(state=WorkflowState.CANCELLED))

        with pytest.raises(AssertionError):
            await _assert_cancel_report(run_task, workflow, WorkflowOutcome.CANCELLED)
