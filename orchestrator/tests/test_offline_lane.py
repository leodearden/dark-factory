"""Tests for the offline deep-test lane singleton worker (task 1953, β2).

Covers ``orchestrator.offline_lane.OfflineLaneWorker`` in isolation: the
``on_post_merge`` enqueue-and-return trigger seam, the always-from-head
``_run_once`` snapshot, the coalescing ``run()`` loop (trigger-driven +
poll-backstop, fail-open), the lockfile singleton, and the default
``run-offline-deep.sh`` suite-runner seam.

See also ``test_harness_offline_lane.py`` for the harness-side
launch/stop/registration wiring, and
``test_harness_offline_lane_trigger.py`` (task 1951, β1) for the
``_offline_lane_notifiee`` fan-out this worker registers into.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from orchestrator.config import GitConfig
from orchestrator.offline_lane import OfflineLaneWorker

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_git_ops(*, head: str = 'headsha', worktree_path: Path | None = None) -> MagicMock:
    """MagicMock git_ops with the async methods OfflineLaneWorker calls."""
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=head)
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(
        return_value=worktree_path or Path('/tmp/_offline-deep')
    )
    return git_ops


def _make_config(tmp_path: Path, **git_overrides) -> MagicMock:
    """MagicMock OrchestratorConfig with a real GitConfig at .git."""
    config = MagicMock()
    config.project_root = tmp_path
    config.git = GitConfig(**git_overrides)
    return config


def _make_worker(
    tmp_path: Path,
    *,
    git_ops: MagicMock | None = None,
    config: MagicMock | None = None,
    suite_runner=None,
) -> OfflineLaneWorker:
    """Build an OfflineLaneWorker wired with mock deps for isolated testing."""
    return OfflineLaneWorker(
        git_ops if git_ops is not None else _make_git_ops(),
        config if config is not None else _make_config(tmp_path),
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner,
    )


# ---------------------------------------------------------------------------
# GitConfig knobs (step-1/2)
# ---------------------------------------------------------------------------


def test_git_config_offline_lane_knobs():
    """GitConfig exposes the offline-lane knobs with correct defaults.

    Step 1 (RED): the fields do not yet exist — must fail before impl.
    """
    cfg_default = GitConfig()
    assert cfg_default.offline_lane_enabled is False, (
        'offline_lane_enabled must default to False (feature off)'
    )
    assert cfg_default.offline_lane_test_threads == 6, (
        'offline_lane_test_threads must default to 6 (§11.2 small fixed N)'
    )
    assert cfg_default.offline_lane_poll_interval_secs == 120.0, (
        'offline_lane_poll_interval_secs must default to 120.0'
    )
    assert cfg_default.offline_lane_red_advances_before_blocker == 3, (
        'offline_lane_red_advances_before_blocker must default to 3 (§11.3, β3 C4)'
    )

    cfg_set = GitConfig(
        offline_lane_enabled=True,
        offline_lane_test_threads=4,
        offline_lane_poll_interval_secs=30.0,
        offline_lane_red_advances_before_blocker=5,
    )
    assert cfg_set.offline_lane_enabled is True
    assert cfg_set.offline_lane_test_threads == 4
    assert cfg_set.offline_lane_poll_interval_secs == 30.0
    assert cfg_set.offline_lane_red_advances_before_blocker == 5


def test_git_config_offline_lane_knobs_validation():
    """offline_lane_test_threads is ge=1; offline_lane_poll_interval_secs is gt=0;
    offline_lane_red_advances_before_blocker is ge=1."""
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_test_threads=0)
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_poll_interval_secs=0.0)
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_red_advances_before_blocker=0)


# ---------------------------------------------------------------------------
# on_post_merge — enqueue-and-return trigger seam (step-3/4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_post_merge_sets_dirty_and_wakes_without_running(tmp_path: Path):
    """on_post_merge flips dirty + wakes the loop and returns — never runs inline.

    Per the β1 hot-path contract (harness.py:5040-5046), the SHAs are
    advisory only: get_main_sha and suite_runner must NOT be awaited by
    on_post_merge itself.

    Step 3 (RED): on_post_merge is a NotImplementedError stub — must fail
    before impl.
    """
    git_ops = _make_git_ops()
    suite_runner = AsyncMock(return_value=(0, ''))
    worker = _make_worker(tmp_path, git_ops=git_ops, suite_runner=suite_runner)

    await worker.on_post_merge('task-1', 'base-sha', 'head-sha')

    assert worker._dirty is True
    assert worker._wake.is_set()
    git_ops.get_main_sha.assert_not_awaited()
    suite_runner.assert_not_awaited()


# ---------------------------------------------------------------------------
# _run_once — always-from-head snapshot (step-5/6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_snapshots_head_at_run_start_and_invokes_seam(tmp_path: Path):
    """_run_once clears dirty BEFORE snapshotting head, then resets + runs the seam.

    Step 5 (RED): _run_once is a NotImplementedError stub — must fail
    before impl.
    """
    worker_ref: dict[str, OfflineLaneWorker] = {}

    async def _fake_get_main_sha() -> str:
        # Clear-before-snapshot: by the time get_main_sha is awaited, dirty
        # must already be cleared (no lost-update window).
        assert worker_ref['worker']._dirty is False, (
            '_run_once must clear _dirty BEFORE snapshotting head'
        )
        return 'HEAD1'

    wt_path = tmp_path / '_offline-deep'
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(side_effect=_fake_get_main_sha)
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=wt_path)
    suite_runner = AsyncMock(return_value=(0, ''))
    config = _make_config(tmp_path, offline_lane_test_threads=6)
    worker = _make_worker(tmp_path, git_ops=git_ops, config=config, suite_runner=suite_runner)
    worker_ref['worker'] = worker
    worker._dirty = True

    await worker._run_once()

    git_ops.get_main_sha.assert_awaited_once()
    git_ops.reset_persistent_offline_deep_worktree.assert_awaited_once_with('HEAD1')
    suite_runner.assert_awaited_once_with(wt_path, 'HEAD1', 6)
    assert worker._last_run_head == 'HEAD1', (
        '_last_run_head must equal the snapshot head, never the advisory trigger SHA'
    )


# ---------------------------------------------------------------------------
# run() — coalescing loop core (step-7/8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_loop_coalesces_to_exactly_one_rerun(tmp_path: Path):
    """Multiple dirty-sets during one run collapse into exactly one re-run.

    Drives worker.run() as a real background task; the suite_runner
    side-effect sets _dirty multiple times during the FIRST run only. Exactly
    two _run_once passes must happen (initial + one coalesced re-run at the
    new head) — never a third — and the loop then quiesces.

    Step 7 (RED): run() is a NotImplementedError stub — must fail before impl.
    """
    heads = ['HEAD1', 'HEAD2']
    head_calls = {'n': 0}

    async def _fake_get_main_sha() -> str:
        idx = min(head_calls['n'], len(heads) - 1)
        head_calls['n'] += 1
        return heads[idx]

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(side_effect=_fake_get_main_sha)
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(
        return_value=tmp_path / '_offline-deep'
    )

    run_count = {'n': 0}
    done = asyncio.Event()

    async def _suite_runner(wt, head, threads):
        run_count['n'] += 1
        if run_count['n'] == 1:
            # Multiple advances landing mid-run must collapse into exactly
            # ONE coalesced re-run afterwards, not two.
            worker._dirty = True
            worker._wake.set()
            worker._dirty = True
            worker._wake.set()
        else:
            done.set()
        return (0, '')

    # Large poll interval so the poll backstop (not yet implemented at this
    # step) cannot fire and interfere with this test.
    config = _make_config(tmp_path, offline_lane_poll_interval_secs=120.0)
    worker = _make_worker(tmp_path, git_ops=git_ops, config=config, suite_runner=_suite_runner)
    worker._dirty = True  # initial trigger already queued

    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert run_count['n'] == 2, 'exactly one coalesced re-run — never a third'
    assert head_calls['n'] == 2
    git_ops.reset_persistent_offline_deep_worktree.assert_any_call('HEAD2')


# ---------------------------------------------------------------------------
# run() — poll backstop for a missed trigger (step-9/10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_poll_backstop_runs_on_missed_trigger(tmp_path: Path):
    """A missed trigger is caught by a periodic get_main_sha poll.

    With ``_dirty`` False and the wake event never set (on_post_merge was
    never called — simulating a missed trigger), and a tiny
    ``offline_lane_poll_interval_secs``, the loop's wait must time out and
    poll ``get_main_sha``: a mismatch against ``_last_run_head`` is
    run-worthy (backstop catches the advance); a match is not (no spurious
    run).

    Step 9 (RED): run() has no poll timeout yet, so on a missed trigger the
    wait blocks forever and the mismatch case's run never happens — the
    bounded ``wait_for`` below turns that hang into a clean failure rather
    than a slow pytest-timeout kill. Must fail before impl.
    """
    poll_interval = 0.05

    # -- Scenario A: get_main_sha diverges from _last_run_head -> backstop runs.
    git_ops_a = _make_git_ops(head='HEAD2')
    config_a = _make_config(tmp_path, offline_lane_poll_interval_secs=poll_interval)
    calls_a: list[tuple] = []
    done = asyncio.Event()

    async def _suite_runner_a(wt, head, threads):
        calls_a.append((wt, head, threads))
        done.set()
        return (0, '')

    worker_a = _make_worker(
        tmp_path, git_ops=git_ops_a, config=config_a, suite_runner=_suite_runner_a
    )
    worker_a._last_run_head = 'HEAD1'

    task_a = asyncio.create_task(worker_a.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=2.0)
    finally:
        task_a.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_a

    assert len(calls_a) == 1, 'a missed-trigger mismatch must run exactly once'
    assert calls_a[0][1] == 'HEAD2', 'the backstop run must use the polled (current) head'
    assert worker_a._last_run_head == 'HEAD2'

    # -- Scenario B: get_main_sha matches _last_run_head -> no spurious run.
    git_ops_b = _make_git_ops(head='HEAD1')
    config_b = _make_config(tmp_path, offline_lane_poll_interval_secs=poll_interval)
    calls_b: list[tuple] = []

    async def _suite_runner_b(wt, head, threads):
        calls_b.append((wt, head, threads))
        return (0, '')

    worker_b = _make_worker(
        tmp_path, git_ops=git_ops_b, config=config_b, suite_runner=_suite_runner_b
    )
    worker_b._last_run_head = 'HEAD1'

    task_b = asyncio.create_task(worker_b.run())
    try:
        await asyncio.sleep(poll_interval * 6)
    finally:
        task_b.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_b

    assert calls_b == [], 'no advance since the last run must never trigger a spurious run'
    assert worker_b._last_run_head == 'HEAD1'


# ---------------------------------------------------------------------------
# run() — fail-open + clean cancellation (step-11/12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_loop_is_fail_open_and_cancellation_clean(tmp_path: Path):
    """A broken lane run can never wedge the process (worker leg of C7).

    (a)+(c): a suite_runner that raises on its first call is logged as a
    bounded one-line summary (never ``logger.exception``, whose O(frames)
    traceback formatting can starve the event loop) and the failure path
    always sleeps a fixed backoff — so the loop cannot tight-spin — yet
    survives: since the failed run never updated ``_last_run_head``, the
    (already-implemented) poll backstop re-flags the still-outstanding
    advance and a subsequent run happens.
    (b): cancelling the ``run()`` task propagates ``asyncio.CancelledError``
    cleanly — never swallowed.

    Step 11 (RED): run() has no failure handling yet — an exception from
    suite_runner propagates straight out of run(), killing the loop task, so
    no second run and no backoff sleep ever happen. Must fail before impl.
    """
    # -- (a)+(c): fail-open with a bounded log + backoff, then survive.
    calls = {'n': 0}
    done = asyncio.Event()

    async def _suite_runner(wt, head, threads):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('suite boom')
        done.set()
        return (0, '')

    git_ops = _make_git_ops(head='HEAD1')
    config = _make_config(tmp_path, offline_lane_poll_interval_secs=0.05)
    worker = _make_worker(tmp_path, git_ops=git_ops, config=config, suite_runner=_suite_runner)
    worker._dirty = True

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _tracking_sleep(delay, *a, **k):
        sleeps.append(delay)
        return await real_sleep(0)

    with patch('orchestrator.offline_lane.asyncio.sleep', side_effect=_tracking_sleep), \
            patch('orchestrator.offline_lane.logger') as mock_logger:
        task = asyncio.create_task(worker.run())
        try:
            await asyncio.wait_for(done.wait(), timeout=3.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert calls['n'] == 2, 'the loop must survive the first failure and run again'
    mock_logger.exception.assert_not_called()
    assert mock_logger.error.call_count >= 1, 'expected a bounded logger.error summary'
    assert 60.0 in sleeps, f'expected a backoff sleep after the failure, got {sleeps!r}'

    # -- (b): cancelling run() propagates CancelledError cleanly (never swallowed).
    worker2 = _make_worker(
        tmp_path, config=_make_config(tmp_path, offline_lane_poll_interval_secs=120.0)
    )
    task2 = asyncio.create_task(worker2.run())
    await asyncio.sleep(0)
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2


# ---------------------------------------------------------------------------
# Lockfile singleton (step-13/14)
# ---------------------------------------------------------------------------


def test_lockfile_enforces_singleton(tmp_path: Path):
    """acquire_lock enforces a singleton via flock; release_lock frees it.

    A second OfflineLaneWorker on the SAME lock_path must refuse to start
    while the first holds the lock; after the first releases, acquiring
    again on that path succeeds.

    Step 13 (RED): acquire_lock/release_lock are NotImplementedError stubs
    — must fail before impl.
    """
    # Nested, not-yet-existing parent dirs — acquire_lock must mkdir(parents=True).
    lock_path = tmp_path / 'data' / 'orchestrator' / 'offline_lane.lock'
    worker_a = OfflineLaneWorker(_make_git_ops(), _make_config(tmp_path), lock_path=lock_path)
    worker_b = OfflineLaneWorker(_make_git_ops(), _make_config(tmp_path), lock_path=lock_path)

    assert worker_a.acquire_lock() is True
    assert lock_path.exists()
    diagnostic = lock_path.read_text()
    assert str(os.getpid()) in diagnostic, 'lockfile must record the holder PID'

    # A second worker instance on the same lock_path must refuse to start.
    assert worker_b.acquire_lock() is False

    worker_a.release_lock()

    # After the holder releases, a fresh acquire on the same path succeeds.
    assert worker_b.acquire_lock() is True
    worker_b.release_lock()


# ---------------------------------------------------------------------------
# Default suite-runner seam (step-15/16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_suite_runner_builds_run_offline_deep_command(tmp_path: Path):
    """The default (uninjected) suite_runner seam builds the real script invocation.

    No suite_runner is supplied, so the worker falls back to its
    ``_default_run_suite`` seam. ``asyncio.create_subprocess_exec`` is
    mocked, so this asserts ONLY that the argv/env/cwd invocation is built
    correctly — real script existence/execution is ζ's job (cross-project
    scope boundary; see module docstring).

    Step 15 (RED): _default_run_suite is a NotImplementedError stub — must
    fail before impl.
    """
    worker = _make_worker(tmp_path)  # no suite_runner injected -> default seam
    wt_path = tmp_path / '_offline-deep'

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b'suite output\n', b''))
    mock_proc.returncode = 0

    with patch(
        'orchestrator.offline_lane.asyncio.create_subprocess_exec',
        return_value=mock_proc,
    ) as mock_exec:
        rc, tail = await worker.suite_runner(wt_path, 'HEAD1', 6)

    assert rc == 0
    assert 'suite output' in tail

    mock_exec.assert_awaited_once()
    argv = mock_exec.call_args.args
    kwargs = mock_exec.call_args.kwargs
    assert list(argv) == ['scripts/run-offline-deep.sh', '--test-threads=6'], (
        'argv must invoke the offline-deep script with the thread count'
    )
    assert kwargs['cwd'] == str(wt_path), 'cwd must be the _offline-deep worktree'
    assert kwargs['env']['DF_VERIFY_ROLE'] == 'offline', (
        'DF_VERIFY_ROLE=offline must be set — the offline role footprint '
        '(nice/ionice) lives inside the script (Part A), not re-implemented here'
    )
    for key, value in os.environ.items():
        if key == 'DF_VERIFY_ROLE':
            continue
        assert kwargs['env'].get(key) == value, (
            f'env must overlay DF_VERIFY_ROLE onto a full os.environ copy, '
            f'not replace it (missing/altered {key!r})'
        )
    assert kwargs['stdout'] == asyncio.subprocess.PIPE
    assert kwargs['stderr'] == asyncio.subprocess.STDOUT
