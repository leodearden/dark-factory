"""Tests for the offline deep-test lane singleton worker (task 1953, β2).

Covers ``orchestrator.offline_lane.OfflineLaneWorker`` in isolation: the
``on_post_merge`` enqueue-and-return trigger seam, the always-from-head
``_run_once`` snapshot, every leg's pass through ``run_pass`` (the confirm on
red, the pass record's blocks), the coalescing ``run()`` loop (trigger-driven +
poll-backstop, fail-open), the idle poll's engine tick and the rechecks it
names, the lockfile singleton, and the default detect seams.

The red path itself is the engine's: ``test_offline_lane_episodes.py`` pins it
at its interface, ``test_offline_lane_pass.py`` pins one pass, and
``test_offline_lane_confirm.py`` pins the confirm.

See also ``test_harness_offline_lane.py`` for the harness-side
launch/stop/registration wiring, and
``test_harness_offline_lane_trigger.py`` (task 1951, β1) for the
``_offline_lane_notifiee`` fan-out this worker registers into.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _lane_episode_fakes import Clock, FakeLedgerClient
from _orch_helpers import pydantic_spec
from escalation.queue import EscalationQueue
from pydantic import ValidationError
from shared.testing_virtual_clock import virtual_clock_test

from orchestrator.config import GitConfig, LaneCommand, OrchestratorConfig
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    ConfirmTarget,
    Unconfirmable,
    outcome_to_json,
    serial_confirm_command,
)
from orchestrator.offline_lane_episode_state import PassEffect, PassObservation
from orchestrator.offline_lane_episodes import TickResult
from orchestrator.offline_lane_pass import ConfirmSeam
from orchestrator.offline_lane_records import (
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    Leg,
    PassRecord,
    Phase,
    RunRecord,
)
from orchestrator.offline_lane_runs import ConfirmRunner

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
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = tmp_path
    config.git = GitConfig(**git_overrides)
    return config


def _make_worker(
    tmp_path: Path,
    *,
    git_ops: MagicMock | None = None,
    config: MagicMock | None = None,
    suite_runner=None,
    infra_runner=None,
    command_runner=None,
    confirm: ConfirmSeam | None = None,
    task_client=None,
    escalation_queue=None,
    now: Callable[[], datetime] | None = None,
) -> OfflineLaneWorker:
    """Build an OfflineLaneWorker wired with mock deps for isolated testing."""
    return OfflineLaneWorker(
        git_ops if git_ops is not None else _make_git_ops(),
        config if config is not None else _make_config(tmp_path),
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner,
        infra_runner=infra_runner,
        command_runner=command_runner,
        confirm=confirm,
        task_client=task_client,
        escalation_queue=escalation_queue,
        now=now,
    )


def _child_writing(output: bytes, *, returncode: int):
    """A fake ``create_subprocess_exec`` whose child writes *output* into the file it is handed."""

    async def fake_exec(*argv, **kwargs):
        kwargs['stdout'].write(output)
        proc = AsyncMock()
        proc.returncode = returncode
        return proc

    return fake_exec


async def _inline(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@contextlib.contextmanager
def _inline_thread_hops():
    """Run every ``asyncio.to_thread`` hop on the loop, process-wide: the virtual clock rejects executor work."""
    with patch('orchestrator.offline_lane.asyncio.to_thread', new=_inline):
        yield


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


def test_git_config_offline_lane_infra_enabled_default_and_settable():
    """GitConfig exposes offline_lane_infra_enabled, default off (task 1959, IE1).

    Step 1 (RED): the field does not yet exist — must fail before impl.
    """
    assert GitConfig().offline_lane_infra_enabled is False, (
        'offline_lane_infra_enabled must default to False (numeric-only, '
        'byte-identical to prior behavior)'
    )
    assert GitConfig(offline_lane_infra_enabled=True).offline_lane_infra_enabled is True


# ---------------------------------------------------------------------------
# LaneCommand + generic-command GitConfig knobs (task 2789, step-1/2)
# ---------------------------------------------------------------------------


class TestLaneCommand:
    """LaneCommand — a per-project offline-lane generic command entry.

    Step 1 (RED): LaneCommand does not yet exist — must fail before impl.
    """

    def test_defaults(self) -> None:
        from orchestrator.config import LaneCommand

        cmd = LaneCommand(name='q', command='pytest -m integration')
        assert cmd.cwd == '.', "cwd must default to '.' (worktree root)"
        assert cmd.fix_task_priority == 'medium', (
            "fix_task_priority must default to 'medium'"
        )
        assert cmd.enabled is True, 'enabled must default to True'

    def test_name_and_command_required(self) -> None:
        from orchestrator.config import LaneCommand

        with pytest.raises(ValidationError):
            LaneCommand(command='pytest')  # type: ignore[call-arg]  # missing name
        with pytest.raises(ValidationError):
            LaneCommand(name='q')  # type: ignore[call-arg]  # missing command

    def test_bogus_priority_rejected(self) -> None:
        from orchestrator.config import LaneCommand

        with pytest.raises(ValidationError):
            LaneCommand(name='x', command='pytest', fix_task_priority='bogus')  # type: ignore[arg-type]

    def test_explicit_fields_round_trip(self) -> None:
        from orchestrator.config import LaneCommand

        cmd = LaneCommand(name='x', command='pytest', fix_task_priority='low')
        assert cmd.name == 'x'
        assert cmd.command == 'pytest'
        assert cmd.fix_task_priority == 'low'


def test_git_config_offline_lane_commands_defaults():
    """GitConfig exposes offline_lane_commands (default []) and
    offline_lane_legacy_numeric_enabled (default True).

    Step 1 (RED): the fields do not yet exist — must fail before impl.
    """
    cfg = GitConfig()
    assert cfg.offline_lane_commands == [], (
        'offline_lane_commands must default to [] (opt-in, no generic commands)'
    )
    assert cfg.offline_lane_legacy_numeric_enabled is True, (
        'offline_lane_legacy_numeric_enabled must default to True '
        '(D2 gate; reify byte-identical)'
    )


def test_git_config_offline_lane_commands_settable():
    """GitConfig accepts a one-command list and a False legacy-numeric gate."""
    from orchestrator.config import LaneCommand

    cfg = GitConfig(
        offline_lane_commands=[
            LaneCommand(name='q', command='pytest -m integration', cwd='fused-memory'),
        ],
    )
    assert len(cfg.offline_lane_commands) == 1
    assert cfg.offline_lane_commands[0].name == 'q'
    assert cfg.offline_lane_commands[0].cwd == 'fused-memory'

    assert (
        GitConfig(offline_lane_legacy_numeric_enabled=False).offline_lane_legacy_numeric_enabled
        is False
    )


class TestOfflineLaneCommandsReload:
    """git.offline_lane_commands + git.offline_lane_legacy_numeric_enabled are
    green-tier (D6): hot-reloadable, landing in applied_candidates not
    restart_required. Mirrors test_config_verify_admission_reload.py.

    Step 3 (RED): the dotted paths are not yet in RELOADABLE_FIELDS, so the
    leaf lands in restart_required — must fail before impl.
    """

    @pytest.mark.parametrize(
        'field',
        ['git.offline_lane_commands', 'git.offline_lane_legacy_numeric_enabled'],
    )
    def test_field_is_reloadable(self, field):
        from orchestrator.config import RELOADABLE_FIELDS

        assert field in RELOADABLE_FIELDS, (
            f'{field!r} is expected to be green-tier reloadable but is '
            f'missing from RELOADABLE_FIELDS'
        )

    def test_commands_edit_lands_in_applied_candidates(self, monkeypatch, tmp_path):
        from orchestrator.config import LaneCommand, diff_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(git=GitConfig(offline_lane_commands=[]))
        fresh = OrchestratorConfig(
            git=GitConfig(
                offline_lane_commands=[
                    LaneCommand(name='q', command='pytest -m integration'),
                ],
            ),
        )
        diff = diff_config(live, fresh)
        assert 'git.offline_lane_commands' in diff.applied_candidates
        assert 'git.offline_lane_commands' not in diff.restart_required

    def test_apply_reload_applies_commands_in_place(self, monkeypatch, tmp_path):
        from orchestrator.config import LaneCommand, apply_reload

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(git=GitConfig(offline_lane_commands=[]))
        fresh = OrchestratorConfig(
            git=GitConfig(
                offline_lane_commands=[
                    LaneCommand(name='q', command='pytest -m integration'),
                ],
            ),
        )
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert 'git.offline_lane_commands' not in report['restart_required']
        assert len(live.git.offline_lane_commands) == 1
        assert live.git.offline_lane_commands[0].name == 'q'


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
#
# The run() loop tests race the worker's poll/backoff timers against a wedge
# deadline, so each is a @virtual_clock_test: on that loop clock a host stall
# cannot expire the deadline.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
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


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
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


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
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


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_loop_backoff_grows_and_caps_on_consecutive_failures(tmp_path: Path):
    """Consecutive run() failures back off exponentially, capped (task 2016 amendment).

    Bug #2's fix means a persistently-raising red-handling/MCP leg now
    retries the SAME still-outstanding head every pass (never silently
    dropped) instead of advancing past it — so without a growing backoff, a
    stuck outage would re-run the entire heavy offline-deep suite at the
    fixed base cadence forever. The backoff must double per consecutive
    failure, capped, and reset once a pass succeeds.
    """
    calls = {'n': 0}
    done = asyncio.Event()
    failures = 5

    async def _suite_runner(wt, head, threads):
        calls['n'] += 1
        if calls['n'] <= failures:
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
            patch('orchestrator.offline_lane.logger'):
        task = asyncio.create_task(worker.run())
        try:
            await asyncio.wait_for(done.wait(), timeout=3.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert calls['n'] == failures + 1, 'the loop must keep retrying every consecutive failure'
    assert sleeps[:failures] == [60.0, 120.0, 240.0, 480.0, 900.0], (
        f'expected exponential backoff doubling from the 60s base and capping '
        f'at 900.0s, got {sleeps[:failures]!r}'
    )


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

    lane_pass = worker.records.allocate_pass(LEGACY_NUMERIC_COMMAND, Leg.NUMERIC, 'HEAD1')

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_child_writing(b'suite output\n', returncode=0),
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
    assert Path(kwargs['stdout'].name) == lane_pass.output_path(Phase.DETECT)
    assert kwargs['stderr'] == asyncio.subprocess.STDOUT


# ---------------------------------------------------------------------------
# _default_run_command — generic per-project command seam (task 2789, step-11/12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_run_command_builds_idle_niced_sh_c_invocation(tmp_path: Path):
    """The default command_runner seam launches `nice -n19 ionice -c3 sh -c <cmd>`.

    A bare pytest does not self-nice (unlike the reify scripts), so the
    generic runner applies the idle nice/ionice prefix itself (D4). cwd is
    <worktree>/<cmd.cwd>; env overlays DF_VERIFY_ROLE=offline onto a full
    os.environ copy; rc + the decoded <=2000-char tail are returned.

    Step 11 (RED): _default_run_command / the command_runner seam do not
    exist — must fail before impl.
    """
    from orchestrator.config import LaneCommand

    worker = _make_worker(tmp_path)  # no command_runner injected -> default seam
    wt_path = tmp_path / '_offline-deep'
    cmd = LaneCommand(name='q', command='pytest -m integration', cwd='fused-memory')

    lane_pass = worker.records.allocate_pass(cmd.name, Leg.GENERIC, 'HEAD1')

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_child_writing(b'command output\n', returncode=1),
    ) as mock_exec:
        rc, tail = await worker.command_runner(cmd, wt_path, 'HEAD1')

    assert rc == 1
    assert 'command output' in tail
    assert len(tail) <= 2000

    mock_exec.assert_awaited_once()
    argv = list(mock_exec.call_args.args)
    kwargs = mock_exec.call_args.kwargs
    assert argv[:5] == ['nice', '-n', '19', 'ionice', '-c3'], (
        'argv must start with the idle nice/ionice prefix (D4)'
    )
    junit_path = shlex.quote(str(lane_pass.junit_path(Phase.DETECT)))
    assert argv[-3:] == ['sh', '-c', f'pytest -m integration --junitxml {junit_path}'], (
        'argv must end with sh -c <command>'
    )
    assert kwargs['cwd'] == str(wt_path / 'fused-memory'), (
        'cwd must be <worktree>/<cmd.cwd>'
    )
    assert kwargs['env']['DF_VERIFY_ROLE'] == 'offline'
    for key, value in os.environ.items():
        if key == 'DF_VERIFY_ROLE':
            continue
        assert kwargs['env'].get(key) == value, (
            f'env must overlay DF_VERIFY_ROLE onto a full os.environ copy '
            f'(missing/altered {key!r})'
        )
    assert Path(kwargs['stdout'].name) == lane_pass.output_path(Phase.DETECT)
    assert kwargs['stderr'] == asyncio.subprocess.STDOUT


@pytest.mark.asyncio
async def test_default_run_command_cwd_dot_is_worktree_root(tmp_path: Path):
    """cwd='.' (the default) resolves to the worktree root itself."""
    from orchestrator.config import LaneCommand

    worker = _make_worker(tmp_path)
    wt_path = tmp_path / '_offline-deep'
    cmd = LaneCommand(name='r', command='pytest', cwd='.')

    worker.records.allocate_pass(cmd.name, Leg.GENERIC, 'HEAD1')

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_child_writing(b'', returncode=0),
    ) as mock_exec:
        await worker.command_runner(cmd, wt_path, 'HEAD1')

    assert mock_exec.call_args.kwargs['cwd'] == str(wt_path), (
        "cwd='.' must resolve to the worktree root"
    )


# ---------------------------------------------------------------------------
# _run_once — every leg goes through run_pass: the confirm on red, the pass record's blocks
# ---------------------------------------------------------------------------


def _pass_record(worker: OfflineLaneWorker, command: str) -> PassRecord:
    lane_pass = worker.records.current_pass(command)
    assert lane_pass is not None, f'no pass was allocated for {command}'
    record = worker.records.load_pass(command, lane_pass.pass_id)
    assert record is not None
    return record


class _RecordingConfirm:
    """An injected confirm seam answering *outcome* (or raising it) and remembering each target it was shown."""

    def __init__(self, *outcomes: Confirmed | Unconfirmable | Exception) -> None:
        self.outcomes = list(outcomes)
        self.targets: list[ConfirmTarget] = []

    async def __call__(
        self, target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner,
    ) -> Confirmed | Unconfirmable:
        self.targets.append(target)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_HELD = Confirmed(ConfirmMode.NARROWED, ('tests/t.py::test_x',), (), ())
_CLEARED = Confirmed(ConfirmMode.NARROWED, (), ('tests/t.py::test_x',), ())


@pytest.mark.asyncio
async def test_run_once_confirms_only_a_red_pass_and_records_each_pass_outcome(tmp_path: Path):
    """A green pass is observed with no outcome and asks no confirm; a red one asks the confirm once,
    and the pass record keeps that outcome and the engine's effect."""
    wt_path = tmp_path / '_offline-deep'
    confirm = _RecordingConfirm(_HELD)

    green = _make_worker(
        tmp_path / 'green', git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(tmp_path / 'green'), suite_runner=AsyncMock(return_value=(0, '')), confirm=confirm,
    )
    await green._run_once()

    assert confirm.targets == []
    green_record = _pass_record(green, LEGACY_NUMERIC_COMMAND)
    assert green_record.outcome is None
    assert isinstance(green_record.effect, dict)

    red = _make_worker(
        tmp_path / 'red', git_ops=_make_git_ops(head='HEAD2', worktree_path=wt_path),
        config=_make_config(tmp_path / 'red'), suite_runner=AsyncMock(return_value=(1, '')), confirm=confirm,
    )
    await red._run_once()

    assert confirm.targets == [ConfirmTarget.numeric(wt_path)]
    red_record = _pass_record(red, LEGACY_NUMERIC_COMMAND)
    assert red_record.head == 'HEAD2'
    assert red_record.outcome == outcome_to_json(_HELD)
    assert isinstance(red_record.effect, dict)


@pytest.mark.asyncio
async def test_run_once_gates_legacy_numeric_sub_run(tmp_path: Path):
    """_run_once runs the legacy numeric leg only when its gate is on (D2): off, no numeric pass exists."""
    wt_path = tmp_path / '_offline-deep'

    suite_runner_off = AsyncMock(return_value=(0, ''))
    worker_off = _make_worker(
        tmp_path / 'off',
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(
            tmp_path / 'off', offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[],
        ),
        suite_runner=suite_runner_off,
    )
    await worker_off._run_once()

    suite_runner_off.assert_not_awaited()
    assert worker_off.records.current_pass(LEGACY_NUMERIC_COMMAND) is None

    suite_runner_on = AsyncMock(return_value=(0, ''))
    worker_on = _make_worker(
        tmp_path / 'on',
        git_ops=_make_git_ops(head='HEAD2', worktree_path=wt_path),
        config=_make_config(tmp_path / 'on'),
        suite_runner=suite_runner_on,
    )
    await worker_on._run_once()

    suite_runner_on.assert_awaited_once()
    assert _pass_record(worker_on, LEGACY_NUMERIC_COMMAND).head == 'HEAD2'


# ---------------------------------------------------------------------------
# The default confirm, reached through the worker on the pass its detection opened
# ---------------------------------------------------------------------------


def _children(*children: tuple[bytes, int]):
    """A fake ``create_subprocess_exec`` serving one ``(output, returncode)`` child per launch, in order."""
    queued = list(children)

    async def fake_exec(*argv, **kwargs):
        output, returncode = queued.pop(0)
        return await _child_writing(output, returncode=returncode)(*argv, **kwargs)

    return fake_exec


def _assert_lane_env(env: dict[str, str]) -> None:
    assert env['DF_VERIFY_ROLE'] == 'offline'
    for key, value in os.environ.items():
        if key != 'DF_VERIFY_ROLE':
            assert env.get(key) == value, f'the lane env must overlay os.environ (missing/altered {key!r})'


@pytest.mark.asyncio
async def test_a_red_numeric_pass_runs_the_serial_confirm_failed_invocation(tmp_path: Path):
    """The default numeric confirm re-runs reify's script serially with ``--confirm-failed`` in the
    lane worktree, writes the confirm phase of the same pass, and reproduces the ids printed under exit 100.
    """
    wt_path = tmp_path / '_offline-deep'
    worker = _make_worker(tmp_path, git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path))

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_children((b'red\n', 1), (b'tests::test_b\ntests::test_a\n', 100)),
    ) as mock_exec:
        await worker._run_once()

    lane_pass = worker.records.current_pass(LEGACY_NUMERIC_COMMAND)
    assert lane_pass is not None
    confirm_call = mock_exec.call_args_list[1]
    assert list(confirm_call.args) == ['scripts/run-offline-deep.sh', '--test-threads=1', '--confirm-failed']
    assert confirm_call.kwargs['cwd'] == str(wt_path)
    _assert_lane_env(confirm_call.kwargs['env'])
    assert Path(confirm_call.kwargs['stdout'].name) == lane_pass.output_path(Phase.CONFIRM)
    assert confirm_call.kwargs['stderr'] == asyncio.subprocess.STDOUT
    assert _pass_record(worker, LEGACY_NUMERIC_COMMAND).outcome == outcome_to_json(
        Confirmed(ConfirmMode.LEG_NATIVE, ('tests::test_b', 'tests::test_a'), (), ()),
    )


@pytest.mark.asyncio
async def test_a_red_generic_pass_confirms_the_detected_failures_narrowed_and_serial(tmp_path: Path):
    """A structured pytest command's default confirm re-runs only the detection's failing tests, serially,
    at idle nice/ionice in the command's cwd, and reproduces those that fail again."""
    from orchestrator.config import LaneCommand

    wt_path = tmp_path / '_offline-deep'
    cmd = LaneCommand(name='q', command='pytest -m integration', cwd='fused-memory')
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[cmd]),
    )
    failing = ['tests/test_q.py::test_a', 'tests/test_q.py::test_b']
    stdout = b''.join(f'FAILED {node}\n'.encode() for node in failing)

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_children((stdout, 1), (stdout, 1)),
    ) as mock_exec:
        await worker._run_once()

    lane_pass = worker.records.current_pass(cmd.name)
    assert lane_pass is not None
    argv = list(mock_exec.call_args_list[1].args)
    kwargs = mock_exec.call_args_list[1].kwargs
    expected_command = serial_confirm_command(
        cmd, failing, invocation_dir=wt_path / 'fused-memory', junit_path=lane_pass.junit_path(Phase.CONFIRM),
    )
    assert argv[:5] == ['nice', '-n', '19', 'ionice', '-c3']
    assert argv[-3:] == ['sh', '-c', expected_command]
    assert '-p no:xdist' in argv[-1]
    assert kwargs['cwd'] == str(wt_path / 'fused-memory')
    _assert_lane_env(kwargs['env'])
    assert Path(kwargs['stdout'].name) == lane_pass.output_path(Phase.CONFIRM)
    assert _pass_record(worker, cmd.name).outcome == outcome_to_json(
        Confirmed(ConfirmMode.NARROWED, tuple(failing), (), ()),
    )


@pytest.mark.asyncio
async def test_a_red_infra_pass_confirms_the_failures_both_runs_name(tmp_path: Path):
    """The default infra confirm re-invokes H9 with the detect seam's argv/env/cwd; a test both runs name
    reproduces and one the confirm passed is cleared."""
    wt_path = tmp_path / '_offline-deep'
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_infra_enabled=True),
    )
    detect = (
        b'  RESULT: FAIL (test_cgroup_delegation.sh)\n'
        b'  RESULT: FAIL (test_reflink_fs.sh)\n'
        b'  RESULT: FAIL (test_cgroup_burn.sh)\n'
    )
    confirm = (
        b'  RESULT: FAIL (test_cgroup_burn.sh)\n'
        b'  RESULT: PASS (test_reflink_fs.sh)\n'
        b'  RESULT: FAIL (test_cgroup_delegation.sh)\n'
    )

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_children((detect, 1), (confirm, 1)),
    ) as mock_exec:
        await worker._run_once()

    confirm_call = mock_exec.call_args_list[1]
    assert list(confirm_call.args) == ['tests/infra/run_all.sh', '--scope', 'host-infra']
    assert confirm_call.kwargs['cwd'] == str(wt_path)
    _assert_lane_env(confirm_call.kwargs['env'])
    assert _pass_record(worker, LEGACY_INFRA_COMMAND).outcome == outcome_to_json(Confirmed(
        ConfirmMode.INTERSECTED,
        ('test_cgroup_delegation.sh', 'test_cgroup_burn.sh'),
        ('test_reflink_fs.sh',),
        (),
    ))


# ---------------------------------------------------------------------------
# infra seams — constructor wiring (task 1959, IE1, step-3/4)
# ---------------------------------------------------------------------------


def test_infra_seams_injected_or_default(tmp_path: Path):
    """infra_runner is stored when injected, else falls back to the default seam (task 1959, IE1)."""
    sentinel_infra = AsyncMock(return_value=(0, ''))
    worker = _make_worker(tmp_path, infra_runner=sentinel_infra)
    assert worker.infra_runner is sentinel_infra

    worker_default = _make_worker(tmp_path)
    assert callable(worker_default.infra_runner)
    assert worker_default.infra_runner is not sentinel_infra


# ---------------------------------------------------------------------------
# Default infra-runner seam (task 1959, IE1, step-5/6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_infra_runner_builds_run_all_host_infra_command(tmp_path: Path):
    """The default (uninjected) infra_runner seam builds reify's H9 invocation.

    No infra_runner is supplied, so the worker falls back to its
    ``_default_run_infra`` seam. ``asyncio.create_subprocess_exec`` is
    mocked, so this asserts ONLY that the argv/env/cwd invocation is built
    correctly — real script existence/execution on reify main is
    cross-project scope (see module docstring), mirroring
    ``test_default_suite_runner_builds_run_offline_deep_command``.

    NO ``--test-threads``: the host-exclusive set self-serializes under
    H9's own flock (IE-D2).

    Step 5 (RED): _default_run_infra is a NotImplementedError stub — must
    fail before impl.
    """
    worker = _make_worker(tmp_path)  # no infra_runner injected -> default seam
    wt_path = tmp_path / '_offline-deep'

    lane_pass = worker.records.allocate_pass(LEGACY_INFRA_COMMAND, Leg.INFRA, 'HEAD1')

    with patch(
        'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
        side_effect=_child_writing(b'infra output\n', returncode=0),
    ) as mock_exec:
        rc, tail = await worker.infra_runner(wt_path, 'HEAD1', 6)

    assert rc == 0
    assert 'infra output' in tail

    mock_exec.assert_awaited_once()
    argv = mock_exec.call_args.args
    kwargs = mock_exec.call_args.kwargs
    assert list(argv) == ['tests/infra/run_all.sh', '--scope', 'host-infra'], (
        'argv must invoke reify H9 with the host-infra scope and NO '
        '--test-threads (the set self-serializes under its own flock)'
    )
    assert kwargs['cwd'] == str(wt_path), 'cwd must be the _offline-deep worktree'
    assert kwargs['env']['DF_VERIFY_ROLE'] == 'offline', (
        'DF_VERIFY_ROLE=offline must be set, same as the numeric sub-run'
    )
    for key, value in os.environ.items():
        if key == 'DF_VERIFY_ROLE':
            continue
        assert kwargs['env'].get(key) == value, (
            f'env must overlay DF_VERIFY_ROLE onto a full os.environ copy, '
            f'not replace it (missing/altered {key!r})'
        )
    assert Path(kwargs['stdout'].name) == lane_pass.output_path(Phase.DETECT)
    assert kwargs['stderr'] == asyncio.subprocess.STDOUT


# ---------------------------------------------------------------------------
# _run_once — dual sub-run when infra enabled (task 1959, IE1, step-11/12)
# ---------------------------------------------------------------------------


def _sub_run_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if ' sub-run head=' in r.getMessage()]


@pytest.mark.asyncio
async def test_run_once_runs_both_sub_runs_when_infra_enabled(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """_run_once runs the infra sub-run at the SAME snapshot head, in the SAME worktree, when
    offline_lane_infra_enabled is True, confirming its red with the infra confirm target; with infra
    disabled (default) only the numeric sub-run runs."""
    wt_path = tmp_path / '_offline-deep'

    # -- Scenario A: infra enabled, numeric green + infra red.
    config_a = _make_config(tmp_path / 'a', offline_lane_infra_enabled=True)
    suite_runner_a = AsyncMock(return_value=(0, ''))
    infra_runner_a = AsyncMock(return_value=(1, ''))
    confirm_a = _RecordingConfirm(_CLEARED)
    worker_a = _make_worker(
        tmp_path / 'a',
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=config_a,
        suite_runner=suite_runner_a,
        infra_runner=infra_runner_a,
        confirm=confirm_a,
    )

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        await worker_a._run_once()

    threads_a = config_a.git.offline_lane_test_threads
    suite_runner_a.assert_awaited_once_with(wt_path, 'HEAD1', threads_a)
    infra_runner_a.assert_awaited_once_with(wt_path, 'HEAD1', threads_a)
    assert confirm_a.targets == [ConfirmTarget.infra(wt_path)]
    lines = _sub_run_lines(caplog)
    assert any(t.startswith('offline-lane: numeric sub-run head=HEAD1 status=PASS') for t in lines), lines
    assert any(t.startswith('offline-lane: infra sub-run head=HEAD1 status=FAIL') for t in lines), lines

    # -- Scenario B: infra enabled, both sub-runs green.
    config_b = _make_config(tmp_path / 'b', offline_lane_infra_enabled=True)
    infra_runner_b = AsyncMock(return_value=(0, ''))
    confirm_b = _RecordingConfirm(_CLEARED)
    worker_b = _make_worker(
        tmp_path / 'b',
        git_ops=_make_git_ops(head='HEAD2', worktree_path=wt_path),
        config=config_b,
        suite_runner=AsyncMock(return_value=(0, '')),
        infra_runner=infra_runner_b,
        confirm=confirm_b,
    )

    await worker_b._run_once()

    infra_runner_b.assert_awaited_once_with(wt_path, 'HEAD2', config_b.git.offline_lane_test_threads)
    assert confirm_b.targets == []

    # -- Scenario C: infra disabled (default) — numeric-only, unchanged.
    infra_runner_c = AsyncMock(return_value=(0, ''))
    worker_c = _make_worker(
        tmp_path / 'c',
        git_ops=_make_git_ops(head='HEAD3', worktree_path=wt_path),
        config=_make_config(tmp_path / 'c'),
        suite_runner=AsyncMock(return_value=(0, '')),
        infra_runner=infra_runner_c,
    )

    await worker_c._run_once()

    infra_runner_c.assert_not_awaited()
    assert worker_c.records.current_pass(LEGACY_INFRA_COMMAND) is None


# ---------------------------------------------------------------------------
# _run_once / run() — a raising confirm or leg must not advance _last_run_head,
# so the poll backstop can retry (task 2016, step-1/2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_confirm_exception_does_not_advance_last_run_head(tmp_path: Path):
    """A red pass whose confirm raises must leave _last_run_head unadvanced, so the poll backstop
    still sees the head as outstanding and retries it."""
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1'),
        suite_runner=AsyncMock(return_value=(1, '')),
        confirm=_RecordingConfirm(RuntimeError('confirm boom')),
    )

    with pytest.raises(RuntimeError, match='confirm boom'):
        await worker._run_once()

    assert worker._last_run_head is None, (
        '_last_run_head must NOT advance when a red pass raises — the run-start snapshot head '
        'must stay outstanding so the poll backstop retries it'
    )


@pytest.mark.asyncio
async def test_run_once_infra_leg_exception_does_not_advance_last_run_head(
    tmp_path: Path,
):
    """An infra sub-run leg exception must also NOT advance _last_run_head.

    A numeric-green pass followed by an ``infra_runner`` raise must leave
    ``_last_run_head`` unadvanced too: the advance comes after every leg.
    """
    git_ops = _make_git_ops(head='HEAD1')
    config = _make_config(tmp_path, offline_lane_infra_enabled=True)
    suite_runner = AsyncMock(return_value=(0, ''))
    infra_runner = AsyncMock(side_effect=RuntimeError('infra boom'))
    worker = _make_worker(
        tmp_path,
        git_ops=git_ops,
        config=config,
        suite_runner=suite_runner,
        infra_runner=infra_runner,
    )

    with pytest.raises(RuntimeError, match='infra boom'):
        await worker._run_once()

    assert worker._last_run_head is None, (
        '_last_run_head must NOT advance when the infra sub-run raises, even '
        'though the numeric leg was green'
    )


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_loop_retries_after_a_confirm_exception(tmp_path: Path):
    """run()'s poll backstop retries a red pass whose confirm raised (Bug #2, end-to-end): the failed
    pass leaves the head outstanding, so the next poll re-flags it and the confirm is asked again."""
    asked = {'n': 0}
    done = asyncio.Event()

    async def _confirm(target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner) -> Confirmed:
        asked['n'] += 1
        if asked['n'] == 1:
            raise RuntimeError('confirm boom')
        done.set()
        return _CLEARED

    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1'),
        config=_make_config(tmp_path, offline_lane_poll_interval_secs=0.05),
        suite_runner=AsyncMock(return_value=(1, '')),
        confirm=_confirm,
    )

    with patch('orchestrator.offline_lane.logger'):
        task = asyncio.create_task(worker.run())
        try:
            await worker.on_post_merge('task-1', 'base', 'head')
            await asyncio.wait_for(done.wait(), timeout=600)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert asked['n'] == 2, 'the poll backstop must re-run the head whose confirm raised'


# ---------------------------------------------------------------------------
# _run_once — generic per-project command sub-runs (task 2789)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_iterates_generic_commands_skipping_disabled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    """_run_once runs each ENABLED generic command once (disabled ones skipped) and logs a labeled
    sub-run line for it."""
    from orchestrator.config import LaneCommand

    wt_path = tmp_path / '_offline-deep'
    cmd_a = LaneCommand(name='cmd-alpha', command='pytest -m integration', cwd='fused-memory')
    cmd_b = LaneCommand(name='cmd-beta', command='pytest -m slow', enabled=False)
    command_runner = AsyncMock(return_value=(0, ''))
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(tmp_path, offline_lane_commands=[cmd_a, cmd_b]),
        suite_runner=AsyncMock(return_value=(0, '')),
        command_runner=command_runner,
    )

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        await worker._run_once()

    command_runner.assert_awaited_once_with(cmd_a, wt_path, 'HEAD1')
    lines = _sub_run_lines(caplog)
    assert any(t.startswith('offline-lane: cmd-alpha sub-run head=HEAD1 status=PASS') for t in lines), lines
    assert not any('cmd-beta' in t for t in lines), 'a disabled generic command must be neither run nor logged'


@pytest.mark.asyncio
async def test_run_once_confirms_each_generic_red_with_its_own_command(tmp_path: Path):
    """Each red generic command is confirmed against its OWN LaneCommand (no late-binding of the loop
    variable), and with the legacy numeric gate off only the generic commands run."""
    from orchestrator.config import LaneCommand

    wt_path = tmp_path / '_offline-deep'
    cmd_q = LaneCommand(name='q', command='pytest -m integration', cwd='fused-memory', fix_task_priority='low')
    cmd_r = LaneCommand(name='r', command='make check')
    suite_runner = AsyncMock(return_value=(0, ''))
    command_runner = AsyncMock(return_value=(1, 'tail'))
    confirm = _RecordingConfirm(_CLEARED)
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=_make_config(
            tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[cmd_q, cmd_r],
        ),
        suite_runner=suite_runner,
        command_runner=command_runner,
        confirm=confirm,
    )

    await worker._run_once()

    suite_runner.assert_not_awaited()
    assert command_runner.await_args_list == [call(cmd_q, wt_path, 'HEAD1'), call(cmd_r, wt_path, 'HEAD1')]
    assert confirm.targets == [ConfirmTarget.generic(cmd_q, wt_path), ConfirmTarget.generic(cmd_r, wt_path)]


@pytest.mark.parametrize(
    ('leg', 'priority'),
    [('numeric', 'high'), ('generic', 'low')],
    ids=['legacy-numeric-at-high', 'generic-at-its-own-priority'],
)
@pytest.mark.asyncio
async def test_run_once_a_red_reproduced_on_two_spaced_passes_files_one_task(
    tmp_path: Path, leg: str, priority: str,
):
    """The worker's engine files a fix task only on the second reproduced pass at least 30 minutes on,
    exactly once, at the leg's priority, as a normal queued task — and never touches the merge queue."""
    from orchestrator.config import LaneCommand

    wt_path = tmp_path / '_offline-deep'
    cmd = LaneCommand(name='q', command='pytest -m integration', fix_task_priority='low')
    config = (
        _make_config(tmp_path)
        if leg == 'numeric'
        else _make_config(tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[cmd])
    )
    heads = iter(['HEAD1', 'HEAD2'])
    git_ops = _make_git_ops(worktree_path=wt_path)
    git_ops.get_main_sha = AsyncMock(side_effect=lambda: next(heads))
    git_ops.is_ancestor_checked = AsyncMock(return_value=True)
    client = FakeLedgerClient([])
    clock = Clock()
    worker = _make_worker(
        tmp_path,
        git_ops=git_ops,
        config=config,
        suite_runner=AsyncMock(return_value=(1, '')),
        command_runner=AsyncMock(return_value=(1, '')),
        confirm=_RecordingConfirm(_HELD),
        task_client=client,
        now=clock,
    )

    await worker._run_once()
    assert client.submitted == [], 'a first reproduced pass files nothing'
    clock.advance(minutes=30)
    await worker._run_once()

    (arguments,) = client.submitted
    assert arguments['priority'] == priority
    assert arguments['metadata']['spawn_context'] == 'offline_lane_red'
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'the offline lane files a NORMAL queued fix task — never a merge gate'
    )
    touched = {str(c[0]).split('.')[0] for c in git_ops.mock_calls if c[0]}
    assert not any(
        kw in name.lower() for name in touched for kw in ('merge', 'queue', 'enqueue', 'target')
    ), f'offline lane must never touch the merge queue (never-a-gate); git_ops calls: {touched}'


@pytest.mark.asyncio
async def test_run_once_a_red_the_confirm_clears_files_nothing(tmp_path: Path):
    """A generic red whose confirm clears the failing test on two spaced passes is a flake: nothing is filed."""
    from orchestrator.config import LaneCommand

    cmd = LaneCommand(name='q', command='pytest -m integration')
    client = FakeLedgerClient([])
    clock = Clock()
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=tmp_path / '_offline-deep'),
        config=_make_config(tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[cmd]),
        command_runner=AsyncMock(return_value=(1, 'tail')),
        confirm=_RecordingConfirm(_CLEARED),
        task_client=client,
        now=clock,
    )

    await worker._run_once()
    clock.advance(minutes=30)
    await worker._run_once()

    assert client.submitted == []


async def _drive_one_pass(worker: OfflineLaneWorker, command: str) -> None:
    """A merge lands and ``run()`` is driven as a background task until *command*'s next pass has recorded its effect."""
    before = worker.records.current_pass(command)
    await worker.on_post_merge('task-1', 'BASE', 'HEAD1')
    task = asyncio.create_task(worker.run())
    try:
        while True:
            assert not task.done(), f'the loop died: {task.exception()!r}'
            current = worker.records.current_pass(command)
            if current is not None and current is not before:
                record = worker.records.load_pass(command, current.pass_id)
                if record is not None and record.effect is not None:
                    return
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_pass_record_older_than_the_retention_survives_while_a_held_episode_counts_it(tmp_path: Path):
    """The run-record prune spares every pass an episode the engine still holds counted: the red that opened an
    episode bound to an open task outlives the retention, while an unreferenced pass of that age is pruned."""
    cmd = LaneCommand(name='q', command='pytest -m integration')
    clock = Clock()
    client = FakeLedgerClient([])
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=tmp_path / '_offline-deep'),
        config=_make_config(tmp_path, offline_lane_legacy_numeric_enabled=False, offline_lane_commands=[cmd]),
        command_runner=AsyncMock(return_value=(1, 'tail')),
        confirm=_RecordingConfirm(_HELD),
        task_client=client,
        escalation_queue=EscalationQueue(tmp_path / 'queue'),
        now=clock,
    )
    await _drive_one_pass(worker, 'q')
    counted_pass_id = _pass_record(worker, 'q').pass_id
    unreferenced = worker.records.allocate_pass('stale', Leg.GENERIC, 'HEAD9')
    clock.advance(minutes=30)
    await _drive_one_pass(worker, 'q')
    (task_id,) = client.tasks
    client.set_status(task_id, 'in-progress')

    for _ in range(15):
        clock.advance(hours=23)
        await _drive_one_pass(worker, 'q')

    assert worker.records.load_pass('q', counted_pass_id) is not None
    assert not worker.records.pass_directory('stale', unreferenced.pass_id).exists()


@pytest.mark.asyncio
async def test_run_once_no_generic_commands_is_reify_byte_identical(tmp_path: Path):
    """With no generic commands and the legacy numeric gate ON, _run_once runs the numeric suite and
    never the command_runner (T5)."""
    wt_path = tmp_path / '_offline-deep'
    config = _make_config(tmp_path)  # defaults: legacy numeric ON, commands=[]
    suite_runner = AsyncMock(return_value=(0, ''))
    command_runner = AsyncMock(return_value=(0, ''))
    worker = _make_worker(
        tmp_path,
        git_ops=_make_git_ops(head='HEAD1', worktree_path=wt_path),
        config=config,
        suite_runner=suite_runner,
        command_runner=command_runner,
    )

    await worker._run_once()

    suite_runner.assert_awaited_once_with(wt_path, 'HEAD1', config.git.offline_lane_test_threads)
    command_runner.assert_not_awaited()


# ---------------------------------------------------------------------------
# run() — the idle poll ticks the engine and runs the rechecks it names
#
# The worker's engine is replaced by a scripted one so each tick's answer is
# chosen; every leg's seam logs ``<leg>@<head>`` and every worktree reset
# ``reset@<head>``, so a test reads the whole sequence of the loop's work.
# ---------------------------------------------------------------------------


_POLL_SECS = 120.0
_NO_EFFECT = PassEffect(
    tests_counted=(), episodes_started=(), episodes_extended=(), episodes_ended=(),
    threshold_reached=(), stale_heads=(), no_decision=(), tasks_filed=(), escalations_raised=(),
)
_NOTHING_DUE = TickResult((), ())

TickAnswer = TickResult | Exception | Callable[[], Awaitable[TickResult]]


class _ScriptedEngine:
    """The worker's engine, scripted: each tick logs ``tick@<head>`` and the loop time, then gives the
    next queued answer (a result, an exception to raise, or a coroutine to await), else nothing due."""

    def __init__(self, events: list[str], answers: Sequence[TickAnswer]) -> None:
        self.events = events
        self.answers = list(answers)
        self.tick_times: list[float] = []

    async def observe(self, observation: PassObservation) -> PassEffect:
        return _NO_EFFECT

    async def tick(self, *, head: str) -> TickResult:
        self.tick_times.append(asyncio.get_running_loop().time())
        self.events.append(f'tick@{head}')
        if not self.answers:
            return _NOTHING_DUE
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, TickResult):
            return answer
        return await answer()

    def pinned_pass_ids(self) -> frozenset[str]:
        return frozenset()


class _LoopRig:
    """A worker running numeric plus two generic commands ``q`` and ``r`` over a scripted engine.

    ``head`` is what main reads now; ``during`` maps an event to a coroutine run while that event's seam runs.
    """

    def __init__(self, tmp_path: Path, answers: Sequence[TickAnswer]) -> None:
        from orchestrator.config import LaneCommand

        self.events: list[str] = []
        self.head = 'HEAD1'
        self.during: dict[str, Callable[[], Awaitable[None]]] = {}
        self.engine = _ScriptedEngine(self.events, answers)
        git_ops = MagicMock()
        git_ops.get_main_sha = AsyncMock(side_effect=lambda: self.head)
        git_ops.reset_persistent_offline_deep_worktree = AsyncMock(side_effect=self._reset)
        config = _make_config(
            tmp_path,
            offline_lane_poll_interval_secs=_POLL_SECS,
            offline_lane_commands=[LaneCommand(name='q', command='pytest q'), LaneCommand(name='r', command='pytest r')],
        )
        with patch('orchestrator.offline_lane.LaneEngine', return_value=self.engine):
            self.worker = _make_worker(
                tmp_path, git_ops=git_ops, config=config,
                suite_runner=self._suite_runner, command_runner=self._command_runner,
            )

    async def _event(self, event: str) -> None:
        self.events.append(event)
        if (hook := self.during.pop(event, None)) is not None:
            await hook()

    async def _reset(self, head: str) -> Path:
        await self._event(f'reset@{head}')
        return Path('/tmp/_offline-deep')

    async def _suite_runner(self, wt: Path, head: str, threads: int) -> tuple[int, str]:
        await self._event(f'numeric@{head}')
        return (0, '')

    async def _command_runner(self, cmd, wt: Path, head: str) -> tuple[int, str]:
        await self._event(f'{cmd.name}@{head}')
        return (0, '')

    async def run_for(self, seconds: float) -> None:
        """Run the loop for *seconds* of loop time."""
        task = asyncio.create_task(self.worker.run())
        try:
            await asyncio.sleep(seconds)
            assert not task.done(), f'the loop died: {task.exception()!r}'
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def run_until(self, ticks: int) -> list[str]:
        """Run the loop until the engine has been ticked *ticks* times; the events up to that tick."""
        task = asyncio.create_task(self.worker.run())
        try:
            while len(self.engine.tick_times) < ticks:
                assert not task.done(), f'the loop died: {task.exception()!r}'
                await asyncio.sleep(_POLL_SECS / 4)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        tick_indexes = [i for i, event in enumerate(self.events) if event.startswith('tick@')]
        return self.events[:tick_indexes[ticks - 1] + 1]


def _due(*commands: str) -> TickResult:
    return TickResult(commands, ())


_FULL_PASS_AT_HEAD1 = ['reset@HEAD1', 'numeric@HEAD1', 'q@HEAD1', 'r@HEAD1']
_FULL_PASS_AT_HEAD2 = ['reset@HEAD2', 'numeric@HEAD2', 'q@HEAD2', 'r@HEAD2']
_FULL_PASS_AT_HEAD3 = ['reset@HEAD3', 'numeric@HEAD3', 'q@HEAD3', 'r@HEAD3']


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_poll_timeout_ticks_the_engine_at_the_current_head(tmp_path: Path):
    """Every poll interval that passes with no trigger ticks the engine once with main's head; a first
    tick at a head never run is the backstop's, and runs one full pass."""
    rig = _LoopRig(tmp_path, [])

    events = await rig.run_until(ticks=3)

    assert events == ['tick@HEAD1', *_FULL_PASS_AT_HEAD1, 'tick@HEAD1', 'tick@HEAD1']
    assert rig.engine.tick_times == [_POLL_SECS, 2 * _POLL_SECS, 3 * _POLL_SECS]


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_poll_timeout_with_no_main_head_neither_ticks_the_engine_nor_rechecks(tmp_path: Path):
    """When main reads no head, the poll ticks nothing and runs nothing, however long the loop idles; the
    first poll that reads a head ticks as usual."""
    rig = _LoopRig(tmp_path, [_due('q')])
    rig.head = ''

    await rig.run_for(5 * _POLL_SECS)

    assert rig.events == []
    assert rig.engine.tick_times == []

    rig.head = 'HEAD1'
    events = await rig.run_until(ticks=1)

    assert events == ['tick@HEAD1']


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_due_command_gets_one_recheck_pass_and_a_merge_landing_during_it_still_runs_a_full_pass(
    tmp_path: Path,
):
    """A tick naming a due command at an already-run head runs exactly one pass of that command at that
    head, after the same worktree reset, and nothing else. The recheck leaves the trigger flag alone: a
    merge landing during it runs as a full pass at the new head straight after, with no tick between."""
    rig = _LoopRig(tmp_path, [])

    async def _merge_lands() -> None:
        rig.head = 'HEAD2'
        await rig.worker.on_post_merge('task-9', 'HEAD1', 'HEAD2')

    async def _due_with_a_merge_during_the_recheck() -> TickResult:
        rig.during['q@HEAD1'] = _merge_lands
        return _due('q')

    rig.engine.answers = [_NOTHING_DUE, _due_with_a_merge_during_the_recheck]
    events = await rig.run_until(ticks=3)

    assert events == [
        'tick@HEAD1', *_FULL_PASS_AT_HEAD1,
        'tick@HEAD1', 'reset@HEAD1', 'q@HEAD1',
        *_FULL_PASS_AT_HEAD2,
        'tick@HEAD2',
    ]


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_recheck_leaves_a_missed_advance_to_the_backstop(tmp_path: Path):
    """A recheck runs at the head its tick read, even when main moves during it with no trigger, and does
    not take that head as run: the next tick sees the advance and runs a full pass there."""
    rig = _LoopRig(tmp_path, [])

    async def _main_moves_silently() -> None:
        rig.head = 'HEAD2'

    async def _due_with_a_silent_advance_during_the_recheck() -> TickResult:
        rig.during['q@HEAD1'] = _main_moves_silently
        return _due('q')

    rig.engine.answers = [_NOTHING_DUE, _due_with_a_silent_advance_during_the_recheck]
    events = await rig.run_until(ticks=4)

    assert events == [
        'tick@HEAD1', *_FULL_PASS_AT_HEAD1,
        'tick@HEAD1', 'reset@HEAD1', 'q@HEAD1',
        'tick@HEAD2', *_FULL_PASS_AT_HEAD2,
        'tick@HEAD2',
    ]


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_moved_head_runs_a_full_pass_instead_of_the_recheck(tmp_path: Path):
    """A tick naming a due command runs no recheck when main has moved since the last run, nor when a
    trigger arrived during the tick: the full pass at the current head covers every due command."""
    rig = _LoopRig(tmp_path, [])

    async def _main_moves_silently() -> None:
        rig.head = 'HEAD2'

    async def _due_while_a_trigger_arrives() -> TickResult:
        rig.head = 'HEAD3'
        await rig.worker.on_post_merge('task-9', 'HEAD2', 'HEAD3')
        return _due('q')

    rig.during['r@HEAD1'] = _main_moves_silently
    rig.engine.answers = [_NOTHING_DUE, _due('q'), _due_while_a_trigger_arrives]
    events = await rig.run_until(ticks=4)

    assert events == [
        'tick@HEAD1', *_FULL_PASS_AT_HEAD1,
        'tick@HEAD2', *_FULL_PASS_AT_HEAD2,
        'tick@HEAD2', *_FULL_PASS_AT_HEAD3,
        'tick@HEAD3',
    ]


@pytest.mark.timeout(10)
@_inline_thread_hops()
@virtual_clock_test
async def test_a_raising_tick_costs_a_backoff_not_the_loop(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A tick that raises is logged as a failed pass and backed off (60s); the loop then ticks again
    and goes on working: one bad tick never kills the worker."""
    rig = _LoopRig(tmp_path, [RuntimeError('tick boom')])

    with caplog.at_level(logging.ERROR, logger='orchestrator.offline_lane'):
        events = await rig.run_until(ticks=3)

    assert events == ['tick@HEAD1', 'tick@HEAD1', *_FULL_PASS_AT_HEAD1, 'tick@HEAD1']
    assert rig.engine.tick_times == [_POLL_SECS, 2 * _POLL_SECS + 60.0, 3 * _POLL_SECS + 60.0]
    assert [r.getMessage() for r in caplog.records] == [
        'offline-lane: run() pass failed (consecutive=1): RuntimeError: tick boom',
    ]
