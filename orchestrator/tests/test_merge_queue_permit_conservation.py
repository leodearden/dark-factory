"""Tests for the SpeculativeMergeWorker <-> SpeculationController wiring, the
additive ``snapshot()['speculation']`` key, and the permit-conservation
property across live concurrent-verify scenarios (MQ-refactor theta / task
1993).

Steps covered:
  step-9  RED   — worker wiring + additive snapshot key
  step-10 GREEN — wire SpeculationController into __init__ + snapshot()
  step-11 RED   — permit-conservation property test across live scenarios
  step-12 GREEN — _merger_loop delegates all speculation state to the
                  controller

This module reuses the bare-worker git_repo/git_config/git_ops fixtures from
test_merge_queue_request_liveness.py (per-file duplication convention — see
that module's docstring) and, from step-11 onward, the concurrent-verify
scenario helpers from test_merge_queue_concurrent_verify.py. It imports
orchestrator.merge_queue / orchestrator.merge_speculation_controller LOCALLY
inside each test (mirrors test_merge_request_ledger.py's convention) so a
not-yet-wired attribute/key never breaks collection during the RED steps.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from test_merge_queue_concurrent_verify import (
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
    _make_request,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures (per-file duplication convention — see
# test_merge_queue_request_liveness.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: worker wiring + additive snapshot key
# ---------------------------------------------------------------------------

# The full pre-existing snapshot() key set (must remain present — additive-only
# freeze). See merge_queue.py's snapshot() return dict.
_PRE_EXISTING_SNAPSHOT_KEYS = {
    'entries',
    'depth',
    'head_of_line',
    'verify_in_progress',
    'is_wip_halted',
    'halt_owner_esc_id',
    'occupancy',
    'suffix_conflict_graph',
    'metrics',
    'frozen_prefix',
    'two_layer_invariants',
}


@pytest.mark.asyncio
class TestWorkerWiringAndAdditiveSnapshotKey:
    """SpeculativeMergeWorker <-> SpeculationController wiring (task 1993 step-9).

    RED until step-10 GREEN wires the controller into __init__/snapshot().
    """

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_worker_constructs_speculation_controller_sharing_the_slot(
        self, git_ops: GitOps, depth: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.merge_speculation_controller import SpeculationController

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=depth)

        assert isinstance(worker._speculation_controller, SpeculationController)
        # Same object — so the untouched verifier-side releases (which call
        # self._speculation_slot.release() directly) keep working unchanged.
        assert worker._speculation_controller._slot is worker._speculation_slot
        assert worker._speculation_controller._depth == worker._speculation_depth
        assert worker._speculation_depth == depth

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_snapshot_speculation_key_is_additive_with_initial_values(
        self, git_ops: GitOps, depth: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=depth)

        snap = worker.snapshot()

        # Additive-only: every pre-existing key stays present.
        assert set(snap) >= _PRE_EXISTING_SNAPSHOT_KEYS

        spec = snap['speculation']
        assert spec['depth'] == depth
        assert spec['held_by_merger'] == 0
        assert spec['slot_available'] == depth
        assert spec['inflight_speculative'] == 0


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: permit-conservation property across live
# concurrent-verify scenarios
# ---------------------------------------------------------------------------
#
# Conservation identity (task 1993; consumed by task iota's downstream audit):
#
#     slot_available + held_by_merger + inflight_speculative == depth (K)
#
# This holds no matter WHERE a given permit currently sits — available,
# retained by the merger for a look-ahead/late-arrival attach, or handed off
# to an in-flight speculative verify — it is a pure counting invariant, not a
# claim about timing. Every snapshot() sampled below observes a consistent
# checkpoint because asyncio is single-threaded/cooperative: the test
# coroutine only regains control while the merger/verifier are themselves
# suspended at an `await`, so whatever SpeculationController last recorded is
# exactly what is true at that instant.
#
# RED (before step-12): `_merger_loop` still threads its own loop-locals
# (spec_base/prefetched/held_spec_permit/pending_spec_base/
# pending_predecessor) and never calls into `self._speculation_controller`,
# so `held_by_merger` reads 0 even while a permit is genuinely held on the
# shared `_speculation_slot` — the identity reads (K-1) != K at the
# retain-state checkpoints below.
#
# GREEN (step-12): `_merger_loop` delegates every speculation-state
# transition to the controller, so held_by_merger tracks the real permit
# ownership and the identity holds at every natural (pre-shutdown)
# checkpoint below.
#
# Post-`worker.stop()` checkpoints use `_assert_shutdown_quiescent` instead
# of the identity: `stop()` unconditionally over-releases the shared
# semaphore depth+1 times as a shutdown safety valve (pre-existing,
# documented, orthogonal to this task), so slot_available legitimately
# exceeds depth once stop() has run. What must still hold post-shutdown is
# that the controller's own bookkeeping is clear (held_by_merger==0,
# inflight_speculative==0) and no permit was ever lost outright
# (slot_available never drops below depth).


def _speculation_snapshot(worker: Any) -> dict:
    return worker.snapshot()['speculation']


def _assert_conservation(worker: Any, *, where: str) -> dict:
    """Assert slot_available + held_by_merger + inflight_speculative == depth.

    Returns the sampled 'speculation' snapshot dict for further inspection.
    """
    spec = _speculation_snapshot(worker)
    total = (
        spec['slot_available'] + spec['held_by_merger'] + spec['inflight_speculative']
    )
    assert total == spec['depth'], (
        f'[{where}] permit-conservation identity violated: '
        f'slot_available({spec["slot_available"]}) + '
        f'held_by_merger({spec["held_by_merger"]}) + '
        f'inflight_speculative({spec["inflight_speculative"]}) == {total}, '
        f'expected depth={spec["depth"]}.\n'
        f'Full speculation snapshot: {spec!r}'
    )
    return spec


def _assert_shutdown_quiescent(worker: Any, *, where: str, depth: int) -> None:
    """Assert the controller side is fully quiescent after worker.stop().

    ``SpeculativeMergeWorker.stop()`` unconditionally releases the shared
    ``_speculation_slot`` an extra ``depth + 1`` times as a shutdown safety
    valve (merge_queue.py's ``stop()``, "Over-releasing a plain Semaphore is
    safe") — regardless of how many permits the merger actually still held.
    That means ``slot_available`` legitimately EXCEEDS ``depth`` after
    ``stop()``, so the conservation identity does not hold verbatim here (this
    is pre-existing, documented behaviour, not something task 1993 changes).

    What must still hold post-shutdown: the controller's OWN bookkeeping is
    fully cleared (held_by_merger == 0, inflight_speculative == 0 — no leaked
    permit/state) and no permit was ever outright LOST (slot_available can
    only be inflated by the safety valve, never reduced below depth).
    """
    spec = _speculation_snapshot(worker)
    assert spec['held_by_merger'] == 0, (
        f'[{where}] held_by_merger must be 0 (idle); got {spec["held_by_merger"]}.'
    )
    assert spec['inflight_speculative'] == 0, (
        f'[{where}] inflight_speculative must be 0 (idle); '
        f'got {spec["inflight_speculative"]}.'
    )
    assert spec['slot_available'] >= depth, (
        f'[{where}] slot_available must never fall below depth={depth} '
        f'(a permit would have been lost); got {spec["slot_available"]}.'
    )


@pytest.mark.asyncio
class TestPrefetchLookaheadAndAttachConservation:
    """Scenarios (1) prefetch look-ahead + (2) late-arrival ATTACH (step-11).

    A's local verify is gated so the merger's one-shot look-ahead peek fires,
    finds nothing pickable, and RETAINS the permit for a possible late
    arrival. B then arrives late and ATTACHes to A's in-flight merge commit;
    B's remote verify is separately gated so there is an observable
    checkpoint while the transferred permit is owned by the verifier side.

    RED until step-12: held_by_merger stays 0 throughout even though a
    permit is genuinely held/transferred on the shared semaphore.
    """

    async def test_prefetch_and_attach_conserve_permits(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        K = 2
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        fake_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='pc-attach-laptop',
        )

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-attach-a', 'pc_attach_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/pc-attach-b', 'pc_attach_b.py', 'b = 2\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('pc-attach-a', 'task/pc-attach-a', wt_a, config)
        req_b = _make_request('pc-attach-b', 'task/pc-attach-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # (1) prefetch look-ahead: A merged, look-ahead peeked (found
            # nothing), permit RETAINED for a possible late arrival.
            _assert_conservation(worker, where='after look-ahead retain (A in-flight)')

            # (2) late-arrival ATTACH: B attaches to A's retained spec_base.
            await queue.put(req_b)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            _assert_conservation(worker, where='after ATTACH transfer (B in-flight)')

            gate_a_release.set()
            gate_b_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'
            assert outcome_b.status == 'done', f'B must land; got {outcome_b!r}'

            _assert_conservation(worker, where='post-landing quiescence')

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)


@pytest.mark.asyncio
class TestFallbackConservation:
    """Scenario (3) FALLBACK to plain main (step-11).

    B arrives only after A's result Future is already done, so the
    ATTACH/FALLBACK decision's condition (d) fails: the retained permit is
    released immediately and B merges non-speculatively.
    """

    async def test_fallback_conserves_permits(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        K = 2

        async def _passing_local(*args: Any, **kwargs: Any) -> MagicMock:
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='pc-fallback-laptop')

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-fallback-a', 'pc_fallback_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/pc-fallback-b', 'pc_fallback_b.py', 'b = 2\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('pc-fallback-a', 'task/pc-fallback-a', wt_a, config)
        req_b = _make_request('pc-fallback-b', 'task/pc-fallback-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _passing_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'

            # A is fully done before B is even enqueued — the ATTACH
            # condition (predecessor not yet done) fails, so B FALLBACKs.
            _assert_conservation(worker, where='after A lands, before B enqueued')

            await queue.put(req_b)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            assert outcome_b.status == 'done', f'B must land; got {outcome_b!r}'

            _assert_conservation(worker, where='after FALLBACK, B lands')

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)


@pytest.mark.asyncio
class TestCascadeRemergeConservation:
    """Scenario (4) cascade re-merge on speculative-predecessor verify failure
    (step-11).

    A's local verify is gated and FAILS; B late-arrival-attaches to A's
    in-flight commit and starts a gated remote verify. When A fails, the
    existing head-failure cascade cancels B's remote verify, re-merges B
    against actual main, and re-dispatches it — the permit must remain
    conserved throughout the cancel + remerge + redispatch sequence.
    """

    async def test_cascade_remerge_conserves_permits(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        K = 2
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls = [0]

        async def _gated_failing_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='tests failed', test_output='FAIL',
                    lint_output='', type_output='', category='',
                    timed_out=False, verify_skipped=False,
                )
            # B's re-verify after the cascade remerges it: passes.
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        fake_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='pc-cascade-laptop',
        )

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-cascade-a', 'pc_cascade_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/pc-cascade-b', 'pc_cascade_b.py', 'b = 2\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('pc-cascade-a', 'task/pc-cascade-a', wt_a, config)
        req_b = _make_request('pc-cascade-b', 'task/pc-cascade-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_failing_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            _assert_conservation(
                worker, where='after look-ahead retain (A in-flight, failing)',
            )

            await queue.put(req_b)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            _assert_conservation(
                worker, where='after ATTACH transfer (B in-flight, pre-cascade)',
            )

            # Release A's gate with passed=False -> A fails; the head-failure
            # cascade cancels B's remote verify, re-merges B against actual
            # main, and re-dispatches it (local re-verify, passes).
            gate_a_release.set()

            outcome_b = await asyncio.wait_for(req_b.result, timeout=25.0)
            outcome_a = await asyncio.wait_for(req_a.result, timeout=10.0)

            assert outcome_a.status != 'done', f'A must NOT land; got {outcome_a!r}'
            assert outcome_b.status == 'done', (
                f'B must land after cascade + remerge + re-verify; got {outcome_b!r}'
            )

            _assert_conservation(worker, where='after cascade + remerge + B lands')

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)


@pytest.mark.asyncio
class TestShutdownRetainedPermitConservation:
    """Scenario (5) shutdown while a permit is retained, no late arrival ever
    arrives (step-11).

    A's local verify is gated so the merger enters RETAIN state (permit held,
    blocked in _acquire_next_request()). The worker is then stopped with the
    queue otherwise empty — the outer finally's on_shutdown() must release
    the retained permit and fully restore the idle state. Parametrized over
    K=1 and K=2 per the task's scenario matrix.
    """

    @pytest.mark.parametrize('K', [1, 2])
    async def test_shutdown_during_retain_conserves_permits(
        self, git_ops: GitOps, git_config: GitConfig, K: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='pc-shutdown-laptop')

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-shutdown-a', 'pc_shutdown_a.py', 'a = 1\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('pc-shutdown-a', 'task/pc-shutdown-a', wt_a, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # Retain state: one permit held; no B ever arrives.
            _assert_conservation(worker, where='retain state, no late arrival')

            gate_a_release.set()

            # Shut down while the merger is blocked in _acquire_next_request()
            # (queue empty) — possibly still holding the retained permit.
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=10.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)


# ---------------------------------------------------------------------------
# task eta step-3: threaded token carried on the item + ledger.live empties
# once every transferred permit has been released THROUGH the ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLedgerLiveEmptiesAfterTransferRelease:
    """task eta (2160) step-3: once a transferred permit's underlying item
    lands, the token must be gone from ``worker._speculation_ledger.live`` —
    not just have its semaphore slot restored (task zeta's interim state,
    which drops the controller's own reference at transfer but never
    discards the token from ``live``; see ``PermitLedger``'s "Known interim
    cost" docstring).

    RED until step-4 GREEN: (a) threads the merger's transfer token onto the
    enqueued item's ``.permit`` field, and (b) migrates the verifier-side
    per-item raw releases to ``ledger.release(item.permit)`` (which discards
    the token from ``live``). Reuses the prefetch-lookahead + late-arrival
    ATTACH scenario from ``TestPrefetchLookaheadAndAttachConservation``: a
    speculative permit is acquired by the merger, retained across the
    look-ahead, ATTACH-transferred to B, and drained when B lands.
    """

    async def test_transferred_item_carries_token_and_live_empties_post_shutdown(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        K = 2
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        fake_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='pc-live-empties-laptop',
        )

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-live-a', 'pc_live_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/pc-live-b', 'pc_live_b.py', 'b = 2\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('pc-live-a', 'task/pc-live-a', wt_a, config)
        req_b = _make_request('pc-live-b', 'task/pc-live-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # (1) prefetch look-ahead: A merged, look-ahead peeked (found
            # nothing), permit RETAINED for a possible late arrival.
            await queue.put(req_b)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # (2) B is now dispatched (in-flight, verifying) — its
            # InflightEntry must carry the SAME token still registered in
            # ledger.live, proving the merger stamped item.permit at
            # transfer time (step 4a) and _dispatch_item copied it onto the
            # InflightEntry (step 4b).
            entry_b = next(
                e for e in worker._inflight
                if e.item.request.task_id == req_b.task_id
            )
            assert entry_b.permit is not None, (
                "B's InflightEntry.permit must be the transferred token, not "
                'None (see SpeculationController.on_transfer/on_transfer_terminal '
                'and the _dispatch_item InflightEntry constructions)'
            )
            assert entry_b.permit in worker._speculation_ledger.live

            gate_a_release.set()
            gate_b_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'
            assert outcome_b.status == 'done', f'B must land; got {outcome_b!r}'

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)

        # Core step-3 assertion: every transferred permit has been released
        # THROUGH the ledger by the time the worker is fully quiescent — not
        # merely had its semaphore slot restored by a raw verifier release
        # that never discarded the token (zeta's interim leak).
        assert worker._speculation_ledger.live == frozenset(), (
            'every transferred permit must be discarded from ledger.live by '
            f'shutdown quiescence; still live: {worker._speculation_ledger.live!r}'
        )


# ---------------------------------------------------------------------------
# Amendment (post-step-12 review): terminal (early-continue) transfer must
# clear spec_base, not just held_by_merger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEarlyContinueTerminalTransferClearsSpecBase:
    """A speculative (ATTACHed) request that resolves via one of the seven
    early-continue sites — here, the Step-0 loop-breaker/abandoned path —
    must return the controller to a genuinely idle state: ``spec_base``
    cleared, ``is_idle()`` True. This exercises
    ``SpeculationController.on_transfer_terminal()`` (added post-step-12) via
    the real ``_merger_loop``, not just the unit-level controller test in
    ``test_merge_speculation_controller.py``.

    Before this fix, the loop's seven early-continue sites called plain
    ``on_transfer()`` — which only clears ``held_by_merger`` (correct for the
    permit-conservation identity, which this suite already covered) — leaving
    ``spec_base`` stale until the next ``on_dequeue``. The permit-conservation
    identity alone cannot see this: it only sums permits, not ``spec_base``.

    A's local verify is gated so A stays in-flight (permit RETAINED via
    ``on_lookahead_pending``, mirroring ``TestPrefetchLookaheadAndAttachConservation``).
    B then ATTACHes to A's retained ``spec_base`` — but B's task_id is
    pre-seeded past ``MAX_POST_MERGE_VERIFY_TIMEOUTS``, so B's dequeue
    immediately hits the Step-0 loop-breaker (an early-continue site) instead
    of attempting a real merge. B never reaches a verify runner, so — unlike
    the ATTACH scenario above — no fake remote / host-allocator injection is
    needed for B; the worker's lazily-built local-only allocator (no
    ``verify_runners`` configured) is enough for A's single local verify.

    B's terminal transfer happens entirely on the MERGER side and is
    observed by polling ``is_idle()`` rather than awaiting ``req_b.result``:
    with a single local host slot occupied by A's still-gated verify, the
    verifier cannot dispatch (and inline-finalize) B's passthrough item
    until A's FINALIZE-HEAD step completes — i.e. until ``gate_a_release``
    is set — so ``req_b.result`` deliberately stays pending across the
    checkpoint this test cares about.
    """

    async def test_attach_then_abandon_returns_to_idle(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        K = 2
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        config = OrchestratorConfig(project_root=git_ops.project_root, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/pc-term-a', 'pc_term_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/pc-term-b', 'pc_term_b.py', 'b = 2\n',
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=K)

        req_a = _make_request('pc-term-a', 'task/pc-term-a', wt_a, config)
        req_b = _make_request('pc-term-b', 'task/pc-term-b', wt_b, config)

        # Pre-seed B past the loop-breaker threshold so its dequeue hits the
        # Step-0 early-continue site (abandoned) instead of a real merge.
        worker._post_merge_verify_timeouts[req_b.task_id] = (
            worker.MAX_POST_MERGE_VERIFY_TIMEOUTS
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # A in-flight, permit RETAINED (pending) for a possible late
            # arrival — genuinely not idle yet.
            assert worker._speculation_controller.is_idle() is False

            await queue.put(req_b)

            # B's terminal (early-continue) transfer happens entirely on the
            # MERGER side (on_dequeue's ATTACH + the Step-0 loop-breaker +
            # on_transfer_terminal()) and does NOT require the verifier to
            # drain B — A still holds the only local host slot until
            # gate_a_release is set, so the verifier cannot reach B's item
            # yet. Poll for the merger-side idle transition rather than
            # awaiting req_b.result (see class docstring).
            for _ in range(500):
                if worker._speculation_controller.is_idle():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail(
                    'merger never returned to idle after B was abandoned '
                    '(is_idle() stayed False — see is_idle()/spec_base below)'
                    f': {_speculation_snapshot(worker)!r}'
                )

            # Core amendment assertion: a terminal (early-continue) transfer
            # must fully clear spec_base, not just held_by_merger — so the
            # controller reads genuinely idle immediately, not just
            # permit-conserved.
            spec = _speculation_snapshot(worker)
            assert spec['spec_base'] is None, (
                f"spec_base must be None after B's terminal transfer; got "
                f'{spec["spec_base"]!r} (stale from on_transfer() not '
                f'clearing it — see SpeculationController.on_transfer_terminal())'
            )
            _assert_conservation(worker, where='after early-continue terminal transfer')

            gate_a_release.set()
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'
            assert outcome_b.status == 'blocked', f'B must be abandoned; got {outcome_b!r}'

            _assert_conservation(worker, where='post-landing quiescence')

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        _assert_shutdown_quiescent(worker, where='post-shutdown', depth=K)

        # Amendment (post-η review): TestLedgerLiveEmptiesAfterTransferRelease's
        # leak detector only covers the successful ATTACH+land path
        # (on_transfer). B's terminal (early-continue) transfer above routes
        # through on_transfer_terminal() instead — mirror that same direct
        # ledger.live check on this path, so a missed ledger.release on any
        # of the seven terminal early-continue sites would leak a permit
        # into ledger.live undetected (the spec_base assertions above do not
        # by themselves prove the permit was discarded from ``live``).
        assert worker._speculation_ledger.live == frozenset(), (
            "B's terminal-transfer permit must be discarded from "
            f'ledger.live by shutdown quiescence; still live: '
            f'{worker._speculation_ledger.live!r}'
        )
