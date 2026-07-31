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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler

from orchestrator.config import OrchestratorConfig
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.merge_types import QueuedBranch


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None

# ---------------------------------------------------------------------------
# Step 1 / Step 2 — worker-side surface
# ---------------------------------------------------------------------------


class TestWorkerSideSurface:
    """TrainCallbacks + TrainCallbackFactory exist in merge_queue; worker accepts the kwarg."""

    def test_import_train_callbacks_and_factory(self) -> None:
        """Importing TrainCallbacks and TrainCallbackFactory from merge_queue must not raise."""
        from orchestrator.merge_queue import TrainCallbackFactory, TrainCallbacks  # noqa: F401

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
            train_callback_factory=sentinel,  # type: ignore[arg-type]
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
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        req = GroupMergeRequest(
            task_id='4442',
            branch=QueuedBranch.parse('4442', 'task/'),
            worktree=MagicMock(),
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            train_id='train-xyz',
            member_task_ids=['4442'],
            tip_branch=QueuedBranch.parse('4442', 'task/'),
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

    @pytest.mark.asyncio
    async def test_mark_member_done_consumes_landed_row(self, tmp_path: Path) -> None:
        """mark_member_done consumes the tip's write-ahead LandedRow on success (task 2280, PRD B1).

        The train journals ONE LandedRow keyed by the tip task_id
        (merge_queue.py:5061). When the tip member flips 'done' the row must be
        consumed inline so it no longer survives to the next orchestrator startup
        for RC-3 to prune (the task-2155 KNOWN LIMITATION). Mirrors the 2681
        single-branch precedent (test_workflow_merge_provenance.py:841-857).
        """
        from orchestrator.harness import build_train_callback_factory

        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        outbox.record(LandedRow(
            task_id='4442', branch_tip_sha='tip',
            advanced_sha='deadbeefcafe', landed_at=1.0,
        ))
        MergeProvenance.bind(outbox)

        sched = FakeScheduler()
        await sched.set_task_status('4442', 'merge-deferred')

        cbs = build_train_callback_factory(sched)('train-xyz')

        # Row present BEFORE the done-write.
        assert outbox.lookup('4442') is not None

        await cbs.mark_member_done('4442', 'deadbeefcafe')

        # Member flipped done ...
        assert sched.statuses['4442'][-1] == 'done', (
            f"expected 'done', got {sched.statuses['4442'][-1]!r}"
        )
        # ... AND its write-ahead row consumed (PRD B1: lookup==None after done).
        assert outbox.lookup('4442') is None


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
# Step-1867-1 / Step-1867-2 — redrive_member callback
# ---------------------------------------------------------------------------


class TestRedriveMember:
    """build_train_callback_factory returns a redrive_member callback that drives
    absorbed coalesce members to pending (re-dispatch) or done (found_on_main)."""

    @pytest.mark.asyncio
    async def test_redrive_not_on_main_flips_to_pending(self) -> None:
        """redrive_member(mid, False, None) flips a merge-deferred member to pending."""
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        await sched.set_task_status('5001', 'merge-deferred')

        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        assert cbs.redrive_member is not None
        await cbs.redrive_member('5001', False, None)

        assert sched.statuses['5001'][-1] == 'pending', (
            f"expected 'pending', got {sched.statuses['5001'][-1]!r}"
        )

    @pytest.mark.asyncio
    async def test_redrive_on_main_marks_done_found_on_main(self) -> None:
        """redrive_member(mid, True, sha) marks seeded member done with found_on_main provenance."""
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        await sched.set_task_status('5002', 'merge-deferred')

        factory = build_train_callback_factory(sched)
        cbs = factory('train-abc')

        assert cbs.redrive_member is not None
        await cbs.redrive_member('5002', True, 'cafe1234beef')

        assert sched.statuses['5002'][-1] == 'done', (
            f"expected 'done', got {sched.statuses['5002'][-1]!r}"
        )
        prov = sched.provenance.get('5002', {})
        assert prov.get('kind') == 'found_on_main', f"kind mismatch: {prov!r}"
        assert prov.get('commit') == 'cafe1234beef', f"commit mismatch: {prov!r}"
        note = prov.get('note', '')
        assert 'on main' in note, f"note missing 'on main': {note!r}"
        assert 'train-abc' in note, f"note missing train_id: {note!r}"

    @pytest.mark.asyncio
    async def test_redrive_on_main_consumes_landed_row(self, tmp_path: Path) -> None:
        """redrive_member found_on_main guard consumes a bound LandedRow (task 2280, PRD B1).

        When a partner's merge already brought the branch into main,
        redrive_member flips the member 'done' (kind='found_on_main'); if that
        member is the train tip it owns the write-ahead LandedRow, which must be
        consumed inline so it does not survive to the next startup for RC-3.
        """
        from orchestrator.harness import build_train_callback_factory

        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        outbox.record(LandedRow(
            task_id='5002', branch_tip_sha='tip',
            advanced_sha='cafe1234beef', landed_at=1.0,
        ))
        MergeProvenance.bind(outbox)

        sched = FakeScheduler()
        await sched.set_task_status('5002', 'merge-deferred')

        cbs = build_train_callback_factory(sched)('train-abc')
        assert cbs.redrive_member is not None

        # Row present BEFORE the done-write.
        assert outbox.lookup('5002') is not None

        await cbs.redrive_member('5002', True, 'cafe1234beef')

        assert sched.statuses['5002'][-1] == 'done', (
            f"expected 'done', got {sched.statuses['5002'][-1]!r}"
        )
        assert sched.provenance.get('5002', {}).get('kind') == 'found_on_main', (
            f"kind mismatch: {sched.provenance.get('5002')!r}"
        )
        # Row CONSUMED on the found_on_main done-write (PRD B1: lookup==None after done).
        assert outbox.lookup('5002') is None

    @pytest.mark.asyncio
    async def test_redrive_noop_for_nontask_member(self) -> None:
        """redrive_member for a non-seeded member must NOT raise and must NOT write status."""
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        # Must not raise.
        assert cbs.redrive_member is not None
        await cbs.redrive_member('nonexistent-9999', False, None)

        # No scheduler state was written.
        assert 'nonexistent-9999' not in sched.statuses
        assert 'nonexistent-9999' not in sched.provenance

    @pytest.mark.asyncio
    async def test_factory_built_redrive_member_is_callable(self) -> None:
        """Factory-built TrainCallbacks.redrive_member is callable (not None)."""
        from orchestrator.harness import build_train_callback_factory
        from orchestrator.merge_queue import TrainCallbacks

        sched = FakeScheduler()
        factory = build_train_callback_factory(sched)
        cbs = factory('train-xyz')

        assert isinstance(cbs, TrainCallbacks)
        assert cbs.redrive_member is not None, "redrive_member must not be None"
        assert callable(cbs.redrive_member), "redrive_member must be callable"

    @pytest.mark.asyncio
    async def test_redrive_falls_through_on_get_statuses_error(self) -> None:
        """When get_statuses returns a transient error, redrive_member still attempts the flip.

        The documented fall-through: if err is not None the existence probe cannot
        determine whether the member is a real task, so the closure proceeds
        conservatively and attempts the status write (fail-open policy, mirrors
        mark_member_done accepted limitation).

        Pins the contract so a regression that changed the error path to early-return
        would be caught.
        """
        from orchestrator.harness import build_train_callback_factory

        class ErrorScheduler:
            """get_statuses always fails; other methods record calls normally."""

            def __init__(self):
                self.statuses: dict[str, list[str]] = {}
                self.provenance: dict[str, dict] = {}

            async def get_statuses(
                self, ids: list[str] | None = None
            ) -> tuple[dict[str, str], Exception | None]:
                return {}, RuntimeError('transient backend error')

            async def set_task_status(
                self, task_id: str, status: str, **_kwargs
            ) -> None:
                self.statuses.setdefault(task_id, []).append(status)

            def clear_requeue_count(self, task_id: str) -> None:
                pass

        sched = ErrorScheduler()
        factory = build_train_callback_factory(sched)
        cbs = factory('train-err-fallthrough')

        assert cbs.redrive_member is not None
        # Seed nothing — existence probe will return (empty, error).
        # The closure must still attempt the flip (fail-open fall-through).
        await cbs.redrive_member('task-9001', False, None)

        assert 'task-9001' in sched.statuses, (
            "redrive_member must attempt set_task_status even when get_statuses errors"
        )
        assert sched.statuses['task-9001'][-1] == 'pending', (
            f"expected 'pending' on error fall-through; got {sched.statuses['task-9001']!r}"
        )


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
        from orchestrator.harness import Harness
        from orchestrator.merge_queue import TrainCallbacks

        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-wiring-0001')

        # Inject a FakeScheduler with one seeded member.
        sched = FakeScheduler()
        await sched.set_task_status('4442', 'merge-deferred')
        # task 3057: the harness now arms the factory with self.config, so the
        # delivered-checks guard makes a get_task round-trip per member. Seed
        # the record a real scheduler would have — an ABSENT one is the
        # fail-safe "unknown metadata -> withhold" case, not the happy path
        # this test is about (that case has its own coverage in
        # TestTrainCallbacksDeliveredChecksGuard).
        sched.task_data['4442'] = {'id': '4442', 'metadata': {}}
        harness.scheduler = sched  # type: ignore[assignment]

        # Stub git_ops so the worker constructor doesn't fail on project_root.
        harness.git_ops = MagicMock()
        harness.git_ops.project_root = None  # no shadow-state path needed
        # B3 warm-lane release on member-done is awaited (harness.py:403);
        # stub it async so mark_member_done doesn't await a sync MagicMock.
        harness.git_ops.release_lane_for_terminal_task = AsyncMock()

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


# ---------------------------------------------------------------------------
# Task 3057 step-17 (RED) — seams 3 + 9: the two harness train callbacks.
#
# `mark_member_done` stamps kind='merged' and `redrive_member(found_on_main=True)`
# stamps kind='found_on_main' — both on the strength of a SIBLING's merge having
# advanced main. That is attribution by inference: it proves something of this
# branch reached main, never that THIS member's declared capability rode along.
#
# The overriding constraint at these two seams is that a withholding must RETURN,
# never RAISE: `mark_member_done` is called from `_do_train_merge`'s post-advance
# flip loop, which collects raises into TRAIN_PARTIAL_FLIP. A capability
# withholding is not a partial-flip failure — the merge genuinely advanced main —
# and raising would bounce an otherwise-healthy train.
# ---------------------------------------------------------------------------

_TC_GATE_TARGET = 'orchestrator.harness.gate_mark_done_on_delivered_checks'
_TC_CHECK = {
    'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present',
}
_TC_TRAIN = 'train-dc'


def _tc_block(reason: str = 'failed'):
    from orchestrator.delivered_checks import DeliveredChecksBlock

    return DeliveredChecksBlock(
        reason=reason,  # type: ignore[arg-type]
        main_sha='MAIN',
        failed_check=_TC_CHECK if reason == 'failed' else None,
    )


def _tc_config(*, enabled: bool = True, tmp_path: Path | None = None) -> MagicMock:
    from orchestrator.config import DeliveredChecksConfig

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = tmp_path or Path('/tmp/proj')
    config.delivered_checks = DeliveredChecksConfig(
        enabled=enabled, check_timeout_secs=7.5,
    )
    return config


def _tc_scheduler(
    mid: str = '7001',
    *,
    get_task: AsyncMock | None = None,
    member_metadata: dict | None = None,
) -> MagicMock:
    """Scheduler stub whose get_statuses passes the closures' existence probe."""
    sched = MagicMock()
    sched.get_statuses = AsyncMock(return_value=({mid: 'merge-deferred'}, None))
    sched.mark_done = AsyncMock()
    sched.set_task_status = AsyncMock()
    sched.get_task = get_task or AsyncMock(return_value={
        'id': mid,
        'metadata': (
            {'delivered_checks': [_TC_CHECK]}
            if member_metadata is None else member_metadata
        ),
    })
    return sched


async def _tc_drive(cbs, closure: str, mid: str = '7001') -> None:
    """Invoke the closure under test with args that reach its stamp."""
    if closure == 'mark_member_done':
        await cbs.mark_member_done(mid, 'deadbeefcafe')
    else:
        assert cbs.redrive_member is not None
        await cbs.redrive_member(mid, True, 'deadbeefcafe')


_TC_CLOSURES = ['mark_member_done', 'redrive_member']
_TC_SITES = {
    'mark_member_done': 'train-member-merged',
    'redrive_member': 'coalesce-derail-found-on-main',
}


@pytest.mark.asyncio
@pytest.mark.parametrize('closure', _TC_CLOSURES)
class TestTrainCallbacksDeliveredChecksGuard:
    """Task 2794's six-row matrix, applied to both harness train callbacks.

    On every block the recovery is this factory's OWN existing no-op shape:
    return without marking, WITHOUT consuming the write-ahead LandedRow (the
    reconciler still needs the crash-window record) and WITHOUT releasing the
    lane (the member is not terminal). The un-flipped member self-heals through
    the 2794-guarded stranded sweep.
    """

    # --- row 1: hollow-done regression / FAILED ---------------------------

    async def test_failed_block_withholds_flip_without_raising(
        self, closure: str, tmp_path: Path, caplog,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'), \
                patch(_TC_GATE_TARGET, AsyncMock(return_value=_tc_block('failed'))), \
                patch('orchestrator.harness.MergeProvenance') as prov:
            await _tc_drive(cbs, closure)  # must NOT raise

        sched.mark_done.assert_not_called()
        prov.consume.assert_not_called()
        git_ops.release_lane_for_terminal_task.assert_not_awaited()
        assert _TC_TRAIN in caplog.text
        assert '7001' in caplog.text

    # --- row 2: all_delivered -> byte-identical flip ----------------------

    async def test_all_delivered_flips_with_todays_exact_args_and_ordering(
        self, closure: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        calls: list[str] = []
        sched.mark_done = AsyncMock(side_effect=lambda *a, **k: calls.append('mark'))
        git_ops.release_lane_for_terminal_task = AsyncMock(
            side_effect=lambda *a, **k: calls.append('lane'),
        )
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, AsyncMock(return_value=None)), \
                patch('orchestrator.harness.MergeProvenance') as prov:
            prov.consume.side_effect = lambda *a, **k: calls.append('consume')
            await _tc_drive(cbs, closure)

        assert calls == ['mark', 'consume', 'lane'], (
            'mark_done -> MergeProvenance.consume -> lane release ordering '
            'must be byte-identical to today'
        )
        kwargs = sched.mark_done.await_args.kwargs
        if closure == 'mark_member_done':
            assert kwargs['kind'] == 'merged'
            assert kwargs['note'] == f'train {_TC_TRAIN}'
        else:
            assert kwargs['kind'] == 'found_on_main'
            assert 'on main' in kwargs['note']
            assert _TC_TRAIN in kwargs['note']
        assert kwargs['sha'] == 'deadbeefcafe'

    # --- row 3: no delivered_checks -> unchanged flip ---------------------

    async def test_member_without_delivered_checks_flips_unchanged(
        self, closure: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler(member_metadata={})
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        sched.mark_done.assert_awaited_once()

    # --- rows 4 & 5: fail-safe blocks withhold identically ----------------

    @pytest.mark.parametrize('reason', ['errored', 'main_sha_unresolved'])
    async def test_fail_safe_blocks_withhold_and_leave_status_untouched(
        self, closure: str, reason: str, tmp_path: Path,
    ) -> None:
        """A permanently-ERRORing descriptor must not wedge the member — it is
        left for the 2794-guarded stranded sweep to re-evaluate."""
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, AsyncMock(return_value=_tc_block(reason))), \
                patch('orchestrator.harness.MergeProvenance') as prov:
            await _tc_drive(cbs, closure)

        sched.mark_done.assert_not_called()
        sched.set_task_status.assert_not_called()
        prov.consume.assert_not_called()

    # --- row 6: config=None -> FULLY inert (every existing caller) --------

    async def test_config_none_keeps_the_guard_fully_inert(
        self, closure: str,
    ) -> None:
        """The bare-worker construction `build_train_callback_factory(sched)`
        must keep working with zero added I/O."""
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        guard = AsyncMock(return_value=_tc_block('failed'))
        cbs = build_train_callback_factory(sched)(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, guard), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        guard.assert_not_awaited()
        sched.get_task.assert_not_called()
        sched.mark_done.assert_awaited_once()

    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, closure: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        guard = AsyncMock(return_value=None)
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(enabled=False, tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, guard), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        assert guard.await_args.kwargs['enabled'] is False

    # --- plumbing: ONE metadata read, live config forwarded ---------------

    async def test_metadata_read_once_and_config_forwarded(
        self, closure: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        meta = {'delivered_checks': [_TC_CHECK]}
        sched = _tc_scheduler(member_metadata=meta)
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        guard = AsyncMock(return_value=None)
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, guard), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        sched.get_task.assert_awaited_once_with('7001')
        guard.assert_awaited_once()
        assert guard.await_args.args[0] == '7001'
        assert guard.await_args.args[1] == meta
        assert guard.await_args.kwargs['project_root'] == str(tmp_path)
        assert guard.await_args.kwargs['check_timeout_secs'] == 7.5
        assert guard.await_args.kwargs['enabled'] is True
        assert guard.await_args.kwargs['site'] == _TC_SITES[closure]

    # --- fail-safe: unknown metadata / an errored guard never stamp -------

    @pytest.mark.parametrize('mode', ['none', 'raises'])
    async def test_unreadable_member_metadata_withholds(
        self, closure: str, mode: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        get_task = (
            AsyncMock(return_value=None) if mode == 'none'
            else AsyncMock(side_effect=RuntimeError('scheduler down'))
        )
        sched = _tc_scheduler(get_task=get_task)
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)  # must NOT raise

        sched.mark_done.assert_not_called()

    async def test_guard_exception_withholds_rather_than_propagating(
        self, closure: str, tmp_path: Path,
    ) -> None:
        """A raise here would reach `_do_train_merge`'s post-advance flip loop
        and be misreported as TRAIN_PARTIAL_FLIP."""
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        git_ops = MagicMock()
        git_ops.release_lane_for_terminal_task = AsyncMock()
        cbs = build_train_callback_factory(
            sched, git_ops, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, AsyncMock(side_effect=RuntimeError('boom'))), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)  # must NOT raise

        sched.mark_done.assert_not_called()

    # --- git_ops unbound: degrade to TODAY's behavior, but not silently ---

    async def test_git_ops_none_degrades_inert_with_a_debug_line(
        self, closure: str, tmp_path: Path, caplog,
    ) -> None:
        """Mirrors this factory's existing `git_ops is None` lane-release
        degradation: withholding EVERY train flip in a configuration that has
        always worked would be worse than the status quo."""
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        guard = AsyncMock(return_value=_tc_block('failed'))
        cbs = build_train_callback_factory(
            sched, None, _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.harness'), \
                patch(_TC_GATE_TARGET, guard), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        guard.assert_not_awaited()
        sched.mark_done.assert_awaited_once()
        assert 'git_ops' in caplog.text, 'the degradation must not be silent'

    # --- ordering: the existence probe still wins -------------------------

    async def test_non_task_member_noops_before_any_check_work(
        self, closure: str, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = _tc_scheduler()
        sched.get_statuses = AsyncMock(return_value=({}, None))
        guard = AsyncMock(return_value=None)
        cbs = build_train_callback_factory(
            sched, MagicMock(), _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, guard), \
                patch('orchestrator.harness.MergeProvenance'):
            await _tc_drive(cbs, closure)

        guard.assert_not_awaited()
        sched.get_task.assert_not_called()
        sched.mark_done.assert_not_called()


@pytest.mark.asyncio
class TestRedrivePendingArmUnaffectedByTheGuard:
    """`redrive_member(mid, False, None)` stamps nothing — it must stay untouched."""

    async def test_pending_redrive_never_consults_the_guard(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.harness import build_train_callback_factory

        sched = FakeScheduler()
        await sched.set_task_status('5001', 'merge-deferred')
        guard = AsyncMock(return_value=_tc_block('failed'))
        cbs = build_train_callback_factory(
            sched, MagicMock(), _tc_config(tmp_path=tmp_path),
        )(_TC_TRAIN)

        with patch(_TC_GATE_TARGET, guard):
            assert cbs.redrive_member is not None
            await cbs.redrive_member('5001', False, None)

        guard.assert_not_awaited()
        assert sched.statuses['5001'][-1] == 'pending'
