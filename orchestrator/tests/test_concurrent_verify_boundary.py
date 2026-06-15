"""Boundary gate ζ (task 1737): OBSERVED-overlap integration tests.

Upgrades the analytical λ gate (test_multihost_verify_integration.py) which
modeled concurrency via a K-server virtual clock.  Here we drive the REAL
SpeculativeMergeWorker.run() with asyncio.Event-gated fake runners and assert
OBSERVED structure (not computed overlap).

PRD §8ζ: "Modules: orchestrator/tests/ (small glue in src only if a test seam
is missing)".  All assertions follow G6/PRD §8.8: assert STRUCTURE only —
overlap occurred, ordered advance, abort happened, zero stall, only-orphan-pruned,
mtimes advanced.  Never a throughput floor.

Fakes/fixtures are reused from the sibling γ harness
(test_merge_queue_concurrent_verify) via import — single source of truth, zero
churn to the landed γ test file.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import time
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Reuse the γ harness fakes/fixtures (established cross-test-module import pattern).
from test_merge_queue_concurrent_verify import (
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
    _make_config_no_runners,
    _make_fake_remote,
    _mock_verify_result,
    config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_ops,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_repo,  # noqa: F401 — pytest fixture re-exported from γ harness
)

from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.verify_runner import RunnerUnavailable

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class _Span(NamedTuple):
    """Recorded verify span: (name, start_monotonic, end_monotonic)."""

    name: str
    start: float
    end: float


def _overlaps(a: _Span, b: _Span) -> bool:
    """True when two verify spans overlap (a.start < b.end and b.start < a.end)."""
    return a.start < b.end and b.start < a.end


class _TrackingDeque(collections.deque):  # type: ignore[type-arg]
    """deque subclass that records peak length via append tracking."""

    def __init__(self) -> None:
        super().__init__()
        self.max_len: int = 0

    def append(self, item: Any) -> None:  # type: ignore[override]
        super().append(item)
        if len(self) > self.max_len:
            self.max_len = len(self)

    def appendleft(self, item: Any) -> None:  # type: ignore[override]
        super().appendleft(item)
        if len(self) > self.max_len:
            self.max_len = len(self)


def _make_spanning_local_verify(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event,
    spans: list[_Span],
    name: str = 'local',
    passed: bool = True,
) -> Any:
    """Return an async coroutine to patch run_scoped_verification with span recording."""

    async def _impl(*args: Any, **kwargs: Any) -> MagicMock:
        t0 = time.monotonic()
        gate_entered.set()
        await gate_release.wait()
        spans.append(_Span(name=name, start=t0, end=time.monotonic()))
        return MagicMock(
            passed=passed,
            summary='ok' if passed else 'fail',
            test_output='ok' if passed else 'FAILED',
            lint_output='',
            type_output='',
            category='' if passed else 'test_failure',
            timed_out=False,
            verify_skipped=False,
        )

    return _impl


def _make_spanning_remote(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event | None,
    spans: list[_Span],
    *,
    passed: bool = True,
    name: str = 'laptop',
) -> MagicMock:
    """Like _gated_runner but records a _Span on each verify call completion."""
    _first_blocked = [False]

    async def _side(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        if not _first_blocked[0]:
            _first_blocked[0] = True
            if gate_entered is not None:
                gate_entered.set()
            await gate_release.wait()
        spans.append(_Span(name=name, start=t0, end=time.monotonic()))
        return _mock_verify_result(passed)

    runner = MagicMock()
    runner.name = name
    runner.is_local = False
    runner.run_merge_verify = AsyncMock(side_effect=_side)
    runner.cancel_verify = AsyncMock(return_value=0)
    runner.probe_clean = AsyncMock(return_value=True)
    return runner


# ---------------------------------------------------------------------------
# prereq-1: Smoke test — de-risks harness wiring before boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessSmokeTest:
    """Smoke test: 2-host worker runs a single item end-to-end (status=='done').

    Verifies the basic _inject_two_host_allocator + worker.run() wiring before
    any of the boundary tests run.
    """

    async def test_two_host_harness_runs_single_item_green(
        self,
        git_ops: Any,
        git_config: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Build a 2-host worker, run one merged item, assert outcome status=='done'."""
        wt_a = await _make_branch_with_file(git_ops, 'task/smoke-a', 'smoke_a.py', 'x = 1\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        fake_remote = _make_fake_remote('laptop')
        _inject_two_host_allocator(worker, fake_remote)

        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id='smoke-a', branch='task/smoke-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=_mock_verify_result(True)),
        ):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome.status == 'done', f'expected done, got {outcome}'


# ---------------------------------------------------------------------------
# B1: overlap + ordered advance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB1OverlapOrderedAdvance:
    """B1: observed overlapping verify spans + main advances in submission order.

    Extends the analytical λ gate (TestOverlapSignal in test_merge_queue_concurrent_verify)
    with a _SpanRecorder approach and an on_merge_landed ordering assertion.
    """

    async def test_b1_overlapping_spans_and_ordered_advance(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Two verifies overlap (spans intersect); main advances in submission order."""
        spans: list[_Span] = []

        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated + span-recording
        local_verify = _make_spanning_local_verify(
            gate_a_release, gate_a_entered, spans, name='local', passed=True,
        )
        # N+1's remote verify: gated + span-recording
        spanning_remote = _make_spanning_remote(
            gate_b_release, gate_b_entered, spans, passed=True, name='laptop',
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/b1-a', 'b1_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b1-b', 'b1_b.py', 'b = 2\n')

        landed_order: list[str] = []

        async def _on_landed(task_id: str, base_sha: str, advanced_sha: str) -> None:
            landed_order.append(task_id)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, on_merge_landed=_on_landed)
        _inject_two_host_allocator(worker, spanning_remote)

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b1-a', branch='task/b1-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b1-b', branch='task/b1-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', local_verify):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            # Both verifies must enter before either is released
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # Release in REVERSE order to prove finalize waits for the head
            gate_b_release.set()   # N+1 (remote) completes first
            gate_a_release.set()   # N (local) completes second

            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)

        await worker.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) Spans overlap: local and laptop verifies ran simultaneously
        assert len(spans) == 2, f'expected 2 spans, got {spans}'
        span_local = next((s for s in spans if s.name == 'local'), None)
        span_remote = next((s for s in spans if s.name == 'laptop'), None)
        assert span_local is not None and span_remote is not None
        assert _overlaps(span_local, span_remote), (
            f'Expected overlapping spans — local={span_local}, remote={span_remote}. '
            'Spans must intersect (both verifies ran simultaneously).'
        )

        # (2) Main advanced in strict submission order (N before N+1)
        assert landed_order == ['b1-a', 'b1-b'], (
            f'Expected main to advance b1-a then b1-b, got {landed_order}'
        )

        # (3) Both outcomes resolved done
        assert outcome_a.status == 'done', f'N expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'N+1 expected done, got {outcome_b}'


# ---------------------------------------------------------------------------
# B2: chain-invalidation under overlap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB2ChainInvalidationUnderOverlap:
    """B2: head FAILS while N+1 mid-verify → N+1 aborted, re-merged, re-verified done."""

    async def test_b2_chain_invalidation_under_overlap(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """N's local verify FAILS while N+1 is mid-verify; N+1 re-merges and resolves done."""
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated, FAILS; subsequent calls (re-dispatch) PASS
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='test_failure', test_output='FAILED',
                    lint_output='', type_output='', category='test_failure',
                    timed_out=False, verify_skipped=False,
                )
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # N+1's remote verify: gated, PASSES when released
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, passed=True, name='laptop')

        wt_a = await _make_branch_with_file(git_ops, 'task/b2-a', 'b2_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b2-b', 'b2_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)

        # Spy on _remerge to confirm N+1 was re-merged
        _original_remerge = worker._remerge
        remerge_task_ids: list[str] = []

        async def _spy_remerge(req: Any, started_mono: Any) -> Any:
            remerge_task_ids.append(req.task_id)
            return await _original_remerge(req, started_mono)

        worker._remerge = _spy_remerge  # type: ignore[method-assign]

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b2-a', branch='task/b2-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b2-b', branch='task/b2-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            # Both verifies must enter (true overlap)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # N's verify fails
            gate_a_release.set()
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)

            # Release N+1's gate so the cascaded inner task unblocks harmlessly
            gate_b_release.set()

            # N+1 re-merges and re-verifies → done
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) N failed (not 'done' or 'already_merged')
        assert outcome_a.status not in ('done', 'already_merged'), (
            f'Expected N to fail, got status={outcome_a.status!r}'
        )

        # (2) N+1's in-flight verify was aborted (cancel_verify called).
        # Relaxed from assert_called_once() to assert_called(): task-1762 step-4
        # legitimately adds a second cancel_verify per aborted downstream entry
        # (_abort_remote_verify pre-cancel + cancel_and_release post-cancel).
        gated_remote.cancel_verify.assert_called()

        # (3) N+1 was re-merged onto actual main
        assert 'b2-b' in remerge_task_ids, (
            f'Expected _remerge called for b2-b, got {remerge_task_ids!r}'
        )

        # (4) N+1 resolved done
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after re-merge, got {outcome_b!r}'
        )

        # Confirm N+1's file landed on main
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'b2_b.py' in main_files, 'N+1 (b2_b.py) not on main after re-merge'
        assert 'b2_a.py' not in main_files, 'N (b2_a.py) must not be on main (verify failed)'


# ---------------------------------------------------------------------------
# B3: host-down mid-overlap, zero stall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB3HostDownMidOverlap:
    """B3: RunnerUnavailable mid-overlap → host quarantined, item re-dispatched local, zero stall."""

    async def test_b3_host_down_mid_overlap_zero_stall(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Remote RunnerUnavailable mid-overlap: quarantine, local re-dispatch, all drain."""
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated, PASSES
        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            gate_a_entered.set()
            await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # N+1's remote runner: raises RunnerUnavailable after signalling entry
        async def _unavailable_side(*args: Any, **kwargs: Any) -> Any:
            gate_b_entered.set()
            raise RunnerUnavailable('host unreachable')

        dead_remote = MagicMock()
        dead_remote.name = 'dead-laptop'
        dead_remote.is_local = False
        dead_remote.run_merge_verify = AsyncMock(side_effect=_unavailable_side)
        dead_remote.cancel_verify = AsyncMock(return_value=0)
        dead_remote.probe_clean = AsyncMock(return_value=True)

        wt_a = await _make_branch_with_file(git_ops, 'task/b3-a', 'b3_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b3-b', 'b3_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, dead_remote)

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b3-a', branch='task/b3-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b3-b', branch='task/b3-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            # Both verifies must enter (true overlap before unavailable fires)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # Release N's gate; N+1's RunnerUnavailable already fired
            gate_a_release.set()

            try:
                outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
                outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            except TimeoutError:
                outcome_a = None
                outcome_b = None
            finally:
                await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) Remote host quarantined
        assert dead_remote.name in worker._runner_quarantine, (
            f'Expected {dead_remote.name!r} in _runner_quarantine={worker._runner_quarantine!r}'
        )

        # (2) Both items resolved done (zero stall)
        assert outcome_a is not None and outcome_a.status == 'done', (
            f'N expected done, got {outcome_a!r}'
        )
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'N+1 expected done (re-dispatched local), got {outcome_b!r}'
        )


# ---------------------------------------------------------------------------
# B4: cancel-confirm frees slot / cancel-fail quarantines until probe clears
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB4CancelBehavior:
    """B4: cancel semantics on head-failure cascade.

    (a) cancel rc=0 → slot freed immediately, next item acquires it.
    (b) cancel rc=1 → slot PARKED until probe_clean returns True, then freed.
    """

    async def test_b4a_cancel_confirm_frees_slot(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Head-fail cascade: cancel_verify rc=0 → slot freed, N+1 resolves done."""
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated, FAILS; subsequent calls PASS (re-dispatch)
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='fail', test_output='FAILED',
                    lint_output='', type_output='', category='test_failure',
                    timed_out=False, verify_skipped=False,
                )
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # N+1's remote: gated + cancel rc=0 (clean cancel)
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, passed=True, name='laptop')
        gated_remote.cancel_verify = AsyncMock(return_value=0)

        wt_a = await _make_branch_with_file(git_ops, 'task/b4a-a', 'b4a_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b4a-b', 'b4a_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        allocator = _inject_two_host_allocator(worker, gated_remote)
        remote_name = gated_remote.name

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b4a-a', branch='task/b4a-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b4a-b', branch='task/b4a-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            gate_a_release.set()   # N fails
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)

            gate_b_release.set()   # release leaked inner task
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) cancel_verify called for the clean cancel (rc=0).
        # Relaxed from assert_called_once() to assert_called(): task-1762 step-4
        # legitimately adds a second cancel_verify per aborted downstream entry
        # (_abort_remote_verify pre-cancel + cancel_and_release post-cancel).
        gated_remote.cancel_verify.assert_called()

        # (2) Remote slot freed after cancel confirm
        assert not allocator.is_busy(remote_name), (
            f'Expected remote slot {remote_name!r} to be FREE after cancel confirm, '
            f'but is_busy={allocator.is_busy(remote_name)}'
        )

        # (3) N failed, N+1 resolved done
        assert outcome_a.status not in ('done', 'already_merged'), (
            f'N expected fail status, got {outcome_a.status!r}'
        )
        assert outcome_b.status == 'done', (
            f'N+1 expected done after re-dispatch, got {outcome_b!r}'
        )

    async def test_b4b_cancel_fail_quarantines_until_probe_clears(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Head-fail cascade: cancel rc=1 → PARK → probe True → slot freed, N+1 done."""
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='fail', test_output='FAILED',
                    lint_output='', type_output='', category='test_failure',
                    timed_out=False, verify_skipped=False,
                )
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # N+1's remote: gated + cancel rc=1 (fail), probe returns True (clears immediately)
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, passed=True, name='laptop')
        gated_remote.cancel_verify = AsyncMock(return_value=1)

        # Checking probe: observe PARKED state before returning True
        parked_observed = [False]
        probe_entered = asyncio.Event()

        wt_a = await _make_branch_with_file(git_ops, 'task/b4b-a', 'b4b_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b4b-b', 'b4b_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        allocator = _inject_two_host_allocator(worker, gated_remote)
        remote_name = gated_remote.name

        # Install checking probe AFTER allocator injection (so remote_name is known)
        async def _checking_probe() -> bool:
            probe_entered.set()
            parked_observed[0] = allocator.is_busy(remote_name)  # True when PARKED
            return True

        gated_remote.probe_clean = AsyncMock(side_effect=_checking_probe)

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b4b-a', branch='task/b4b-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b4b-b', branch='task/b4b-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            gate_a_release.set()   # N fails → cascade
            await asyncio.wait_for(req_a.result, timeout=15.0)

            gate_b_release.set()   # release leaked inner task
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) cancel_verify called → cancel-fail path taken.
        # Relaxed from assert_called_once() to assert_called(): task-1762 step-4
        # legitimately adds a second cancel_verify per aborted downstream entry
        # (_abort_remote_verify pre-cancel + cancel_and_release post-cancel).
        gated_remote.cancel_verify.assert_called()

        # (2) probe_clean called → slot was PARKED during probe
        assert probe_entered.is_set(), 'Expected probe_clean to be called (PARK → probe path)'
        assert parked_observed[0], (
            f'Expected slot {remote_name!r} to be PARKED when probe_clean ran, '
            f'but is_busy was False'
        )

        # (3) Slot freed after probe clears
        assert not allocator.is_busy(remote_name), (
            f'Expected slot {remote_name!r} to be FREE after probe_clean=True'
        )

        # (4) N+1 resolved done (re-dispatched after slot freed)
        assert outcome_b.status == 'done', (
            f'N+1 expected done after cancel-fail + probe clear, got {outcome_b!r}'
        )


# ---------------------------------------------------------------------------
# B5: operator halt aborts all in-flight + requeues; resumes on unhalt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB5OperatorHalt:
    """B5: operator_halt aborts both in-flight verifies, requeues both; unhalt drains."""

    async def test_b5_operator_halt_aborts_all_and_resumes_on_unhalt(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Halt aborts all in-flight; unhalt_all_lanes resumes, both items drain done."""
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated, PASSES when released
        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            gate_a_entered.set()
            await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # N+1's remote: gated, PASSES
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, passed=True, name='laptop')

        wt_a = await _make_branch_with_file(git_ops, 'task/b5-a', 'b5_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b5-b', 'b5_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)
        worker.VERIFY_ABANDON_POLL_SECS = 0.01  # fast abort-polls for determinism

        # Spy on _remerge: halt MUST NOT trigger chain-invalidation re-merge
        _original_remerge = worker._remerge
        remerge_task_ids: list[str] = []

        async def _spy_remerge(req: Any, started_mono: Any) -> Any:
            remerge_task_ids.append(req.task_id)
            return await _original_remerge(req, started_mono)

        worker._remerge = _spy_remerge  # type: ignore[method-assign]

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b5-a', branch='task/b5-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b5-b', branch='task/b5-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)

            # Both verifies enter (true overlap)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # Halt: abort-polls fire within VERIFY_ABANDON_POLL_SECS=0.01s
            worker.operator_halt('b5-test')
            await asyncio.sleep(0.15)  # let abort-polls fire and requeue

            # Futures must still be pending (REQUEUED does not resolve them)
            assert not req_a.result.done(), 'req_a must be pending while halted'
            assert not req_b.result.done(), 'req_b must be pending while halted'

            # Release gates so leaked inner verify tasks complete harmlessly
            gate_a_release.set()
            gate_b_release.set()

            # No cascade re-merge should have occurred (REQUEUED path skips _remerge)
            assert remerge_task_ids == [], (
                f'Halt MUST NOT trigger _remerge; got {remerge_task_ids!r}'
            )

            # Unhalt: items re-merge and drain
            worker.unhalt_all_lanes('b5-test')

            outcome_a = await asyncio.wait_for(req_a.result, timeout=30.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=30.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # Both must resolve done after unhalt + drain
        assert outcome_a.status == 'done', f'N expected done after unhalt, got {outcome_a}'
        assert outcome_b.status == 'done', f'N+1 expected done after unhalt, got {outcome_b}'


# ---------------------------------------------------------------------------
# B6: ENOSPC prune under overlap protects live set (δ regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB6EnospcPrune:
    """B6: prune_stale_merge_worktrees with keep= protects registered worktrees; orphan removed."""

    async def test_b6_enospc_prune_under_overlap_protects_live_set(
        self,
        git_ops: Any,
        git_repo: Path,
        config: Any,
    ) -> None:
        """Only the orphan (not in ledger) is removed; live + queued worktrees survive."""
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        # Create real _merge-* worktrees via the git_ops private factory
        live_wt_a, _ = await git_ops._create_merge_worktree()
        live_wt_b, _ = await git_ops._create_merge_worktree()
        queued_wt, _ = await git_ops._create_merge_worktree()
        orphan_wt, _ = await git_ops._create_merge_worktree()

        # Register live + queued worktrees in the ledger (simulating in-flight + queued)
        worker._register_owned_merge_worktree(live_wt_a)
        worker._register_owned_merge_worktree(live_wt_b)
        worker._register_owned_merge_worktree(queued_wt)
        # orphan_wt is NOT registered — simulates a crashed-merge orphan

        # Call prune with the ledger snapshot as the keep-set (δ call-site contract)
        removed = await git_ops.prune_stale_merge_worktrees(
            keep=set(worker._owned_merge_worktrees)
        )

        # Only the orphan is removed
        assert len(removed) == 1, f'Expected 1 removal (orphan only), got {removed}'
        assert not orphan_wt.exists(), f'Orphan {orphan_wt} must be removed'

        # Live and queued worktrees survive
        assert live_wt_a.exists(), f'Live worktree {live_wt_a} must survive'
        assert live_wt_b.exists(), f'Live worktree {live_wt_b} must survive'
        assert queued_wt.exists(), f'Queued worktree {queued_wt} must survive'


# ---------------------------------------------------------------------------
# B7: single-host (no runners) routes through new dispatch/finalize path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB7SingleHostNewPath:
    """B7: single-host config routes through _dispatch_item/_finalize_inflight, serial semantics."""

    async def test_b7_single_host_no_runners_routes_through_new_path_serial_order(
        self,
        git_ops: Any,
        git_repo: Path,
        git_config: Any,
        config: Any,
    ) -> None:
        """No-runner config: all items flow through _dispatch_item/_finalize_inflight, peak in-flight==1."""
        wt_a = await _make_branch_with_file(git_ops, 'task/b7-a', 'b7_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/b7-b', 'b7_b.py', 'b = 2\n')
        wt_c = await _make_branch_with_file(git_ops, 'task/b7-c', 'b7_c.py', 'c = 3\n')

        # Use single-host (no verify_runners) config
        single_config = _make_config_no_runners(git_repo, git_config)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        # Spy on _dispatch_item and _finalize_inflight
        original_dispatch = worker._dispatch_item
        original_finalize = worker._finalize_inflight
        dispatch_count = [0]
        finalize_count = [0]

        async def _spy_dispatch(item: Any) -> Any:
            dispatch_count[0] += 1
            return await original_dispatch(item)

        async def _spy_finalize(entry: Any) -> Any:
            finalize_count[0] += 1
            return await original_finalize(entry)

        worker._dispatch_item = _spy_dispatch  # type: ignore[method-assign]
        worker._finalize_inflight = _spy_finalize  # type: ignore[method-assign]

        # Track peak in-flight via _TrackingDeque
        tracked = _TrackingDeque()
        worker._inflight = tracked  # type: ignore[assignment]

        # Track landing order
        landed_order: list[str] = []

        async def _on_landed(task_id: str, base_sha: str, advanced_sha: str) -> None:
            landed_order.append(task_id)

        worker._on_merge_landed = _on_landed

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='b7-a', branch='task/b7-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=single_config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='b7-b', branch='task/b7-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=single_config, result=loop.create_future(), lane='normal',
        )
        req_c = MergeRequest(
            task_id='b7-c', branch='task/b7-c', worktree=wt_c,
            pre_rebased=False, task_files=None, module_configs=[],
            config=single_config, result=loop.create_future(), lane='normal',
        )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=_mock_verify_result(True)),
        ):
            worker_task = asyncio.create_task(worker.run())
            await q.put(req_a)
            await q.put(req_b)
            await q.put(req_c)

            outcome_a = await asyncio.wait_for(req_a.result, timeout=30.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=30.0)
            outcome_c = await asyncio.wait_for(req_c.result, timeout=30.0)
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # (1) All items routed through new _dispatch_item / _finalize_inflight: exactly once each
        assert dispatch_count[0] == 3, (
            f'Expected exactly 3 _dispatch_item calls (one per item), got {dispatch_count[0]}'
        )
        assert finalize_count[0] == 3, (
            f'Expected exactly 3 _finalize_inflight calls (one per item), got {finalize_count[0]}'
        )

        # (2) Peak in-flight == 1 (single-host: at most one verify simultaneously)
        assert tracked.max_len <= 1, (
            f'Expected peak _inflight ≤1 for single-host, got {tracked.max_len}'
        )

        # (3) All items resolve done
        assert outcome_a.status == 'done', f'A expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'B expected done, got {outcome_b}'
        assert outcome_c.status == 'done', f'C expected done, got {outcome_c}'

        # (4) Main advanced in strict submission order
        assert landed_order == ['b7-a', 'b7-b', 'b7-c'], (
            f'Expected serial submission-order advance, got {landed_order}'
        )


# ---------------------------------------------------------------------------
# B8: heartbeat advances all in-flight + queued worktree mtimes under overlap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB8HeartbeatAdvancesWorktrees:
    """B8: _touch_owned_merge_worktrees advances mtime for all registered worktrees (1728 regression)."""

    async def test_b8_heartbeat_advances_all_worktrees_under_overlap(
        self,
        tmp_path: Path,
        git_ops: Any,
    ) -> None:
        """All registered worktrees (in-flight + queued) get mtime advanced by one heartbeat tick."""
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        # Create real directories (temp dirs, not full git worktrees) for ledger entries.
        # _touch_owned_merge_worktrees calls os.utime(p, None), so paths must exist.
        in_flight_a = tmp_path / '_merge-inflight-a'
        in_flight_a.mkdir()
        in_flight_b = tmp_path / '_merge-inflight-b'
        in_flight_b.mkdir()
        queued_1 = tmp_path / '_merge-queued-1'
        queued_1.mkdir()
        queued_2 = tmp_path / '_merge-queued-2'
        queued_2.mkdir()

        # Register all (simulates: 2 live in-flight + 2 in the ledger as queued)
        worker._register_owned_merge_worktree(in_flight_a)
        worker._register_owned_merge_worktree(in_flight_b)
        worker._register_owned_merge_worktree(queued_1)
        worker._register_owned_merge_worktree(queued_2)

        # Capture initial mtimes
        def _mtime(p: Path) -> float:
            return os.stat(p).st_mtime

        initial_mtimes = {
            in_flight_a: _mtime(in_flight_a),
            in_flight_b: _mtime(in_flight_b),
            queued_1: _mtime(queued_1),
            queued_2: _mtime(queued_2),
        }

        # Pin each directory to a known past mtime so the assertion is deterministic
        # regardless of filesystem mtime granularity (avoids wall-clock sleep races on
        # coarse-mtime CI mounts where os.utime(p, None) within a short sleep may land
        # on the same granularity bucket as the initial stat).
        past_time = initial_mtimes[in_flight_a] - 2.0
        for p in initial_mtimes:
            os.utime(p, (past_time, past_time))

        # Drive one heartbeat tick directly
        touched = worker._touch_owned_merge_worktrees()

        # (1) All 4 registered worktrees were touched
        assert touched == 4, f'Expected 4 worktrees touched, got {touched}'

        # (2) Every mtime advanced past the pinned past_time
        for p in initial_mtimes:
            new_mtime = _mtime(p)
            assert new_mtime > past_time, (
                f'Expected mtime to advance for {p.name}: '
                f'pinned={past_time:.6f} new={new_mtime:.6f}'
            )
