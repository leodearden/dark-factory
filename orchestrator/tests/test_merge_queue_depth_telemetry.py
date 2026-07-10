"""Tests for merge-verify depth telemetry (task 2340).

Covers:
  step-9  RED   — SpeculativeMergeWorker._verify_frontier_depth() pure unit test
  step-10 GREEN — _verify_frontier_depth() implementation
  step-11 RED   — depth/speculative plumbing: _run_post_merge_verify's
                  pool.dispatch forwarding + _run_inflight_verify caller wiring
  step-12 GREEN — thread depth/speculative through _run_post_merge_verify /
                  _run_inflight_verify / _dispatch_item
  step-13 RED   — speculative_merge event carries depth
  step-14 GREEN — classify_and_merge threads worker._verify_frontier_depth()
                  into the speculative_merge _emit_speculative call

DEPTH DEFINITION (ε=1890 verify-frontier stack height): depth 0 = a head
verify against real main; depth d = d speculated items already
frozen/verifying ahead of the item joining the frontier.  See
_verify_frontier_depth()'s docstring and test_merge_queue_frozen_prefix.py
for the underlying frozen-prefix model this helper reuses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import MergeResult
from orchestrator.merge_queue import MergeRequest, RealMergeItem, SpeculativeMergeWorker
from orchestrator.verify import VerifyResult


def _make_bare_worker() -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for pure unit tests.

    No event loop or real git_ops required — mirrors test_halt_owner.py's
    ``SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())``
    construction style.
    """
    return SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: _verify_frontier_depth()
# ---------------------------------------------------------------------------


class TestVerifyFrontierDepth:
    """_verify_frontier_depth() == len(_frozen_inflight_entries()) (ε=1890).

    Pure/synchronous delegation test — lightly stubs _frozen_inflight_entries()
    so this test is isolated from the frozen-prefix computation itself
    (already covered by test_merge_queue_frozen_prefix.py) and asserts only
    the depth helper's own wiring.

    RED until step-10 GREEN adds the method.
    """

    def test_empty_frontier_returns_zero(self) -> None:
        """No frozen/verifying entries ahead -> depth 0 (head verify vs real main)."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        assert worker._verify_frontier_depth() == 0

    def test_one_frozen_entry_returns_one(self) -> None:
        """One speculated item ahead -> depth 1."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [object()]
        assert worker._verify_frontier_depth() == 1

    def test_three_frozen_entries_returns_three(self) -> None:
        """Three speculated items ahead -> depth 3."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [object(), object(), object()]
        assert worker._verify_frontier_depth() == 3


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: depth/speculative plumbing
# ---------------------------------------------------------------------------


def _make_bare_config() -> OrchestratorConfig:
    return OrchestratorConfig(git=GitConfig(main_branch='main'))


def _make_request(
    task_id: str, branch: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop.

    Callers must be inside an async test (all tests in this class are).
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


def _make_git_ops_mock() -> MagicMock:
    """Mock git_ops sufficient for _run_post_merge_verify's pre-verify guards.

    Mirrors TestRunPostMergeVerifyRunnerParam._make_git_ops_mock in
    test_merge_queue_concurrent_verify.py.
    """
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value='main-sha')
    mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    mock.cleanup_merge_worktree = AsyncMock()
    mock.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    return mock


def _fake_pass_runner(name: str = 'fake-runner') -> MagicMock:
    """Fake RemoteRunner-shaped object whose run_merge_verify always passes."""
    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='',
        summary='ok', category='',
    ))
    fake.cancel_verify = AsyncMock(return_value=0)
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


class _CapturingEventStore(EventStore):
    """Capturing EventStore -- records emit() calls without touching sqlite.

    Mirrors _LateArrivalFakeEventStore (test_merge_speculation.py) /
    _FakeEventStore (test_merge_queue_concurrent_verify.py).
    """

    def __init__(self) -> None:
        object.__init__(self)
        self.emitted: list[dict] = []

    def emit(  # type: ignore[override]
        self, event_type, *, task_id=None, phase=None, role=None,
        data=None, cost_usd=None, duration_ms=None,
    ) -> None:
        self.emitted.append({'event_type': event_type, 'data': data or {}})

    def events_of(self, event_type: EventType) -> list[dict]:
        return [e for e in self.emitted if e['event_type'] == event_type]


@pytest.mark.asyncio
class TestRunPostMergeVerifyDepthPlumbing:
    """_run_post_merge_verify threads depth/speculative into pool.dispatch
    (task 2340), which threads them into the emitted merge_verify event.

    RED until step-12 GREEN adds the depth/speculative kwargs.
    """

    async def test_depth_and_speculative_propagate_into_merge_verify_event(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_bare_config()
        req = _make_request('t-depth', 'task/t-depth', tmp_path, config)
        git_ops = _make_git_ops_mock()
        es = _CapturingEventStore()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={}, max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=_fake_pass_runner(), event_store=es,
            depth=3, speculative=True,  # RED: these kwargs don't exist yet
        )

        assert outcome is None  # verify passed
        events = es.events_of(EventType.merge_verify)
        assert len(events) == 1
        assert events[0]['data']['depth'] == 3
        assert events[0]['data']['speculative'] is True

    async def test_default_call_omits_depth_and_speculative_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """Legacy call form (no depth/speculative kwargs) -> both None in the event."""
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_bare_config()
        req = _make_request('t-nodepth', 'task/t-nodepth', tmp_path, config)
        git_ops = _make_git_ops_mock()
        es = _CapturingEventStore()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={}, max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=_fake_pass_runner(), event_store=es,
        )

        assert outcome is None
        events = es.events_of(EventType.merge_verify)
        assert len(events) == 1
        assert events[0]['data']['depth'] is None
        assert events[0]['data']['speculative'] is None


@pytest.mark.asyncio
class TestRunInflightVerifyDepthWiring:
    """_run_inflight_verify forwards depth + item.speculative into
    _run_post_merge_verify (task 2340).

    RED until step-12 GREEN adds the depth kwarg on _run_inflight_verify
    and forwards depth/speculative into its _run_post_merge_verify call.
    """

    async def test_depth_and_speculative_forwarded_to_run_post_merge_verify(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.verify_runner import HostLease

        config = _make_bare_config()
        req = _make_request('t-wire', 'task/t-wire', tmp_path, config)
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='dead' * 10,
            speculative=True,
        )
        lease = HostLease(name='laptop', runner=_fake_pass_runner(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return None  # pass

        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _fake_run_post_merge_verify,
        ):
            await worker._run_inflight_verify(item, lease, depth=2)  # RED: no depth kwarg yet

        assert captured.get('depth') == 2
        assert captured.get('speculative') is True
