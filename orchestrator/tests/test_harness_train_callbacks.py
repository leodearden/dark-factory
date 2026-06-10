"""Tests for the harness-injected train-callback factory.

Covers:
1. Worker-side surface (step 1/2): TrainCallbacks/TrainCallbackFactory types and
   SpeculativeMergeWorker accepting train_callback_factory kwarg.
2. Real-task flip (step 3/4): build_train_callback_factory + FakeScheduler prove
   mark_member_done flips a seeded task to 'done' with kind='merged' provenance.
3. Non-task tolerance + synthetic status + error-park (step 5/6):
   - mark_member_done no-ops (does not raise) for a member with no scheduler task.
   - status_check synthesizes 'merge-deferred' for non-task members (worker pre-check
     admitted) while preserving real statuses.
   - On get_statuses error, status_check returns partial dict (parks train_incomplete).
4. Harness wiring (step 7/8): _start_merge_worker injects a non-None, callable
   train_callback_factory kwarg into SpeculativeMergeWorker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from _workflow_helpers import FakeScheduler


# ---------------------------------------------------------------------------
# Step 1 / Step 2 — worker-side surface
# ---------------------------------------------------------------------------


class TestWorkerSideSurface:
    """TrainCallbacks + TrainCallbackFactory exist in merge_queue; worker accepts the kwarg."""

    def test_import_train_callbacks_and_factory(self) -> None:
        """Importing TrainCallbacks and TrainCallbackFactory from merge_queue must not raise."""
        from orchestrator.merge_queue import TrainCallbacks, TrainCallbackFactory  # noqa: F401

    def test_train_callbacks_holds_two_fields(self) -> None:
        """TrainCallbacks is a dataclass with status_check and mark_member_done attributes."""
        from orchestrator.merge_queue import TrainCallbacks

        async def _fake_status_check(ids: list[str]) -> dict[str, str]:
            return {}

        async def _fake_mark_done(mid: str, sha: str) -> None:
            return None

        cbs = TrainCallbacks(
            status_check=_fake_status_check,
            mark_member_done=_fake_mark_done,
        )
        assert cbs.status_check is _fake_status_check
        assert cbs.mark_member_done is _fake_mark_done

    def test_worker_accepts_train_callback_factory_kwarg(self) -> None:
        """SpeculativeMergeWorker stores the injected factory as _train_callback_factory."""
        from orchestrator.merge_queue import SpeculativeMergeWorker

        sentinel = object()
        worker = SpeculativeMergeWorker(
            MagicMock(),  # git_ops
            asyncio.Queue(),
            train_callback_factory=sentinel,
        )
        assert worker._train_callback_factory is sentinel

    def test_worker_defaults_train_callback_factory_to_none(self) -> None:
        """SpeculativeMergeWorker._train_callback_factory defaults to None."""
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(
            MagicMock(),  # git_ops
            asyncio.Queue(),
        )
        assert worker._train_callback_factory is None


# ---------------------------------------------------------------------------
# Step 3 / Step 4 — signal 1: real-task flip
# ---------------------------------------------------------------------------


class TestRealTaskFlip:
    """build_train_callback_factory flips a real (seeded) scheduler task to 'done'."""

    @pytest.mark.asyncio
    async def test_mark_member_done_flips_real_task(self) -> None:
        """Factory callbacks mark a seeded task done with kind='merged' provenance."""
        from orchestrator.harness import build_train_callback_factory
        from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome, TrainCallbacks

        sched = FakeScheduler()
        await sched.set_task_status('4442', 'merge-deferred')

        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')
        assert isinstance(cbs, TrainCallbacks)

        # Build a GroupMergeRequest wired with factory callbacks, as γ would.
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        config = MagicMock()
        config.merge_timeout_seconds = 30
        req = GroupMergeRequest(
            task_id='4442',
            branch='4442',
            worktree=MagicMock(),
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            train_id='train-xyz',
            member_task_ids=['4442'],
            tip_branch='4442',
            tip_task_id='4442',
            status_check=cbs.status_check,
            mark_member_done=cbs.mark_member_done,
        )

        await req.mark_member_done('4442', 'deadbeefcafe')

        assert sched.statuses['4442'][-1] == 'done', (
            f"expected 'done', got {sched.statuses['4442'][-1]!r}"
        )
        assert sched.provenance['4442'] == {
            'kind': 'merged',
            'commit': 'deadbeefcafe',
            'note': 'train train-xyz',
        }, f"unexpected provenance: {sched.provenance['4442']!r}"


# ---------------------------------------------------------------------------
# Step 5 / Step 6 — signal 2: non-task tolerance + synthetic status + error-park
# ---------------------------------------------------------------------------


class TestNonTaskToleranceAndSyntheticStatus:
    """mark_member_done is a no-op for non-task members; status_check synthesizes merge-deferred."""

    @pytest.mark.asyncio
    async def test_mark_member_done_noop_for_nonscheduler_member(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """mark_member_done for a member with no scheduler task must NOT raise.

        A raise would hit the worker's post-advance partial-flip path
        (merge_queue.py:3742-3766) and falsely trigger TRAIN_PARTIAL_FLIP with
        'manual cleanup required'. The no-op + info-log is the correct behaviour.
        """
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        with caplog.at_level(logging.INFO):
            # Must not raise.
            await cbs.mark_member_done('cargo-run-prebuilt-fix', 'sha123')

        # No scheduler state was written.
        assert 'cargo-run-prebuilt-fix' not in sched.statuses
        assert 'cargo-run-prebuilt-fix' not in sched.provenance

        # A log line mentioning the member and "no scheduler task" (or similar) was emitted.
        matching = [r for r in caplog.records if 'cargo-run-prebuilt-fix' in r.getMessage()]
        assert matching, (
            "Expected an INFO log line mentioning 'cargo-run-prebuilt-fix' "
            f"but got records: {[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_status_check_synthesizes_merge_deferred_for_nonscheduler_member(self) -> None:
        """status_check returns 'merge-deferred' for non-task members (worker pre-check pass).

        The worker pre-check (merge_queue.py:3508) parks the train if any member's
        status != 'merge-deferred'. Synthesizing 'merge-deferred' for non-task members
        (e.g. MCP-submitted branches) lets a mixed task/non-task train pass the gate.
        """
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        await sched.set_task_status('4442', 'merge-deferred')
        await sched.set_task_status('pending-task', 'pending')

        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        # Non-task member gets synthesized 'merge-deferred'; real member keeps its status.
        result = await cbs.status_check(['4442', 'cargo-run-prebuilt-fix'])
        assert result == {'4442': 'merge-deferred', 'cargo-run-prebuilt-fix': 'merge-deferred'}, (
            f"Unexpected status_check result: {result!r}"
        )

        # Real status is preserved, not overwritten.
        result2 = await cbs.status_check(['pending-task'])
        assert result2 == {'pending-task': 'pending'}, (
            f"Real status should not be synthesized over: {result2!r}"
        )

    @pytest.mark.asyncio
    async def test_status_check_parks_on_get_statuses_error(self) -> None:
        """On get_statuses error, status_check returns {} (parks train_incomplete, no synthesis).

        Prevents advancing a train on a lie when the backend is down.
        The worker pre-check treats a missing key as 'missing' → parks train_incomplete.
        """
        from orchestrator.harness import build_train_callback_factory

        class ErrorScheduler:
            """get_statuses always fails with a transient error."""

            async def get_statuses(
                self, ids: list[str] | None = None
            ) -> tuple[dict[str, str], Exception | None]:
                return {}, RuntimeError('backend down')

        sched = ErrorScheduler()
        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        result = await cbs.status_check(['4442'])
        # Must return the partial dict (empty here) without synthesizing.
        assert result == {}, f"Expected empty dict on error, got: {result!r}"


# ---------------------------------------------------------------------------
# Step 7 / Step 8 — harness wiring
# ---------------------------------------------------------------------------


class TestHarnessWiring:
    """_start_merge_worker injects a non-None callable train_callback_factory."""

    @pytest.mark.asyncio
    async def test_start_merge_worker_injects_train_callback_factory(
        self, tmp_path: Any,
    ) -> None:
        """_start_merge_worker passes train_callback_factory to SpeculativeMergeWorker.

        Mirrors the pattern from test_harness_resume_scheduler._make_harness:
        - Real OrchestratorConfig
        - Harness with scheduler = FakeScheduler
        - SpeculativeMergeWorker patched to a capturing fake
        - enforce_merge_liveness_margin / enforce_persistent_worktree_serial_lane patched
          to no-ops so the guard conditions don't interfere.
        """
        from orchestrator.config import OrchestratorConfig
        from orchestrator.event_store import EventStore
        from orchestrator.harness import Harness, build_train_callback_factory
        from orchestrator.merge_queue import TrainCallbacks

        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-wiring-0001')

        # Inject a FakeScheduler with one seeded member.
        sched = FakeScheduler()
        await sched.set_task_status('4442', 'merge-deferred')
        harness.scheduler = sched  # type: ignore[assignment]

        # Stub git_ops so the worker constructor doesn't fail on project_root.
        harness.git_ops = MagicMock()
        harness.git_ops.project_root = None  # no shadow-state path needed

        # Capturing fake that records kwargs and provides an async no-op run().
        captured: dict[str, Any] = {}

        class CapturingWorker:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

            async def run(self) -> None:
                await asyncio.sleep(0)  # no-op coroutine so create_task succeeds

            async def stop(self) -> None:
                pass

        # _start_merge_worker imports everything locally from orchestrator.merge_queue,
        # so we must patch on the merge_queue module (not harness).
        with (
            patch(
                'orchestrator.merge_queue.SpeculativeMergeWorker',
                CapturingWorker,
            ),
            patch(
                'orchestrator.merge_queue.enforce_merge_liveness_margin',
            ),
            patch(
                'orchestrator.merge_queue.enforce_persistent_worktree_serial_lane',
            ),
            patch.object(
                harness,
                '_build_service_restart_coordinator',
                return_value=MagicMock(note_merge=AsyncMock()),
            ),
        ):
            await harness._start_merge_worker()

        # The factory must have been injected.
        factory = captured.get('train_callback_factory')
        assert factory is not None, (
            f"Expected train_callback_factory kwarg, got captured={captured!r}"
        )
        assert callable(factory), "train_callback_factory must be callable"

        # The injected factory must produce a real TrainCallbacks that flips the
        # seeded FakeScheduler member to 'done' with kind='merged'.
        cbs = factory('train-T1')
        assert isinstance(cbs, TrainCallbacks)
        await cbs.mark_member_done('4442', 'aabbccddeeff')
        assert sched.statuses['4442'][-1] == 'done'
        assert sched.provenance['4442'] == {
            'kind': 'merged',
            'commit': 'aabbccddeeff',
            'note': 'train train-T1',
        }

        # Teardown.
        await harness._stop_merge_worker()
