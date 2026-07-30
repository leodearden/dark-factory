"""Tests for shared.config_dir — TaskConfigDir and the stale-PID-dir sweep.

Task 3086: ``UsageGate.__init__`` builds one ``TaskConfigDir`` per (account,
pid) under /tmp. Nothing reclaimed those after a SIGKILL, so the population
grew without bound (433,384 dirs / ~1.3M inodes measured on the reify host
2026-07-27). ``sweep_stale_pid_dirs`` bounds it by reclaiming dirs whose
embedded PID is dead.

Every case here is rooted at ``tmp_path`` via ``base_dir=`` — no test ever
touches the real /tmp.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from shared import config_dir as config_dir_module
from shared.config_dir import (
    CONFIG_DIR_PREFIX,
    TaskConfigDir,
    _pid_alive,
    sweep_stale_pid_dirs,
)

# The prefix the UsageGate probe dirs actually use. Built from the module
# constant rather than hard-coded so the test cannot drift from the
# construction template.
PROBE_PREFIX = CONFIG_DIR_PREFIX + 'usage-gate-probe-'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_dead_pid() -> int:
    """Return a PID that is definitively not alive right now.

    Scans upward from a high number until ``os.kill(pid, 0)`` raises
    ``ProcessLookupError``. Picking a fixed literal would be brittle: a
    recycled PID would silently turn a "dead PID" case into a live one and
    the assertion would flip. ``PermissionError`` (visible but unsignalable
    — i.e. alive) is skipped like any other live candidate.
    """
    for candidate in range(999_000, 999_000 + 5000):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    raise RuntimeError('could not find a dead PID to test against')


def age(path: Path, secs: float = 3600.0) -> None:
    """Backdate *path*'s mtime so the sweep's min-age floor does not apply."""
    old = time.time() - secs
    os.utime(path, (old, old))


def plant(base: Path, name: str, *, aged: bool = True) -> Path:
    """Create a directory named *name* under *base*, aged past the floor."""
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    (path / '.credentials.json').write_text('{}')
    if aged:
        age(path)
    return path


# ---------------------------------------------------------------------------
# PID liveness — the predicate every deletion is gated on
# ---------------------------------------------------------------------------


class TestPidAlive:
    """`_pid_alive`'s conservative branches (task 3086).

    ``find_dead_pid`` deliberately skips unsignalable candidates, so nothing
    else in this file reaches the ``PermissionError`` handler. It is the one
    branch whose loss is genuinely dangerous, so it is pinned directly.
    """

    def test_non_positive_pids_are_not_alive(self):
        assert _pid_alive(0) is False
        assert _pid_alive(-1) is False

    def test_this_process_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_permission_error_means_alive(self):
        """Visible but unsignalable — another user's live process."""
        with patch('shared.config_dir.os.kill', side_effect=PermissionError):
            assert _pid_alive(4242) is True

    def test_other_oserror_means_dead(self):
        with patch('shared.config_dir.os.kill', side_effect=OSError('EINVAL')):
            assert _pid_alive(4242) is False

    def test_another_users_live_process_dir_survives_the_sweep(self, tmp_path):
        """The dangerous failure mode, end to end.

        A future simplification collapsing the PermissionError handler into
        the generic ``except OSError: return False`` would flip an
        unsignalable-but-LIVE process to "dead" and delete a live peer's
        credential dir. It fails here instead of in production.
        """
        foreign = plant(tmp_path, f'{PROBE_PREFIX}work-4242')

        with patch('shared.config_dir.os.kill', side_effect=PermissionError):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert foreign.exists()
        assert (foreign / '.credentials.json').exists()
        assert removed == 0


# ---------------------------------------------------------------------------
# Selection semantics — what the sweep does and does not consider a candidate
# ---------------------------------------------------------------------------


class TestSweepStalePidDirsSelection:
    """Which entries the sweep selects for removal (task 3086, step 1)."""

    def test_removes_dir_whose_embedded_pid_is_dead(self, tmp_path):
        dead = find_dead_pid()
        stale = plant(tmp_path, f'{PROBE_PREFIX}work-{dead}')

        sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert not stale.exists()

    def test_removes_no_account_alias_dir_shape(self, tmp_path):
        """The no-accounts alias `...-probe-<pid>` leaks too — cover it."""
        dead = find_dead_pid()
        stale = plant(tmp_path, f'{PROBE_PREFIX}{dead}')

        sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert not stale.exists()

    def test_keeps_dir_whose_embedded_pid_is_alive(self, tmp_path):
        """This test process is definitively alive — its dir must survive.

        This is the guard against deleting a live peer's credential dir out
        from under it, which is the only genuinely dangerous failure mode.
        """
        live = plant(tmp_path, f'{PROBE_PREFIX}work-{os.getpid()}')

        sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert live.exists()
        assert (live / '.credentials.json').exists()

    def test_keeps_dir_with_no_parseable_trailing_pid(self, tmp_path):
        """Never delete what we cannot attribute to a process."""
        unattributable = plant(tmp_path, f'{PROBE_PREFIX}nopid')

        sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert unattributable.exists()

    def test_keeps_dirs_outside_the_prefix(self, tmp_path):
        """Blast radius is prefix-scoped, never all `claude-config-*`.

        Per-task config dirs and test fixtures share the `claude-config-`
        stem; only the probe prefix may be swept.
        """
        dead = find_dead_pid()
        per_task = plant(tmp_path, f'{CONFIG_DIR_PREFIX}3086')
        other_pid_shaped = plant(tmp_path, f'{CONFIG_DIR_PREFIX}some-task-{dead}')
        unrelated = plant(tmp_path, f'unrelated-{dead}')

        sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert per_task.exists()
        assert other_pid_shaped.exists()
        assert unrelated.exists()

    def test_returns_count_of_removed_dirs(self, tmp_path):
        dead = find_dead_pid()
        plant(tmp_path, f'{PROBE_PREFIX}work-{dead}')
        plant(tmp_path, f'{PROBE_PREFIX}personal-{dead}')
        plant(tmp_path, f'{PROBE_PREFIX}{dead}')
        # Non-candidates: must not be counted.
        plant(tmp_path, f'{PROBE_PREFIX}live-{os.getpid()}')
        plant(tmp_path, f'{PROBE_PREFIX}nopid')
        plant(tmp_path, f'{CONFIG_DIR_PREFIX}3086')

        removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert removed == 3

    def test_returns_zero_when_nothing_matches(self, tmp_path):
        plant(tmp_path, f'{CONFIG_DIR_PREFIX}3086')

        assert sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path) == 0


# ---------------------------------------------------------------------------
# Robustness and bounding — the sweep runs on the event-loop thread during
# harness startup, so it must never raise and never block unboundedly.
# ---------------------------------------------------------------------------


class TestSweepStalePidDirsBounding:
    """min-age floor, entry-type guards, containment, deadline (step 3)."""

    def test_min_age_floor_keeps_a_freshly_created_dead_pid_dir(self, tmp_path):
        """Covers a dir created microseconds before its owner died.

        Also gives partial cover against clock skew and any future
        PID-namespace mismatch, where a "dead" PID may not mean a dead owner.
        """
        dead = find_dead_pid()
        fresh = plant(tmp_path, f'{PROBE_PREFIX}work-{dead}', aged=False)

        removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path, min_age_secs=300.0)

        assert fresh.exists()
        assert removed == 0

    def test_min_age_floor_removes_the_same_dir_once_aged(self, tmp_path):
        dead = find_dead_pid()
        stale = plant(tmp_path, f'{PROBE_PREFIX}work-{dead}', aged=False)
        age(stale, secs=600.0)

        removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path, min_age_secs=300.0)

        assert not stale.exists()
        assert removed == 1

    def test_plain_file_matching_the_prefix_is_untouched(self, tmp_path):
        dead = find_dead_pid()
        stray = tmp_path / f'{PROBE_PREFIX}work-{dead}'
        stray.write_text('not a directory')
        age(stray)

        removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert stray.exists()
        assert removed == 0

    def test_symlink_matching_the_prefix_is_untouched(self, tmp_path):
        """Never follow a symlink out of the swept prefix."""
        dead = find_dead_pid()
        target = tmp_path / 'real-target-dir'
        target.mkdir()
        (target / 'payload.txt').write_text('precious')
        link = tmp_path / f'{PROBE_PREFIX}work-{dead}'
        link.symlink_to(target)

        removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert link.is_symlink()
        assert (target / 'payload.txt').exists()
        assert removed == 0

    def test_missing_base_dir_returns_zero_and_does_not_raise(self, tmp_path):
        missing = tmp_path / 'does-not-exist'

        assert sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=missing) == 0

    def test_rmtree_failure_is_contained_and_logged(self, tmp_path, caplog):
        """One unremovable entry must not abort the sweep.

        The next entry is still reclaimed, the failure is visible at WARNING
        (never silently swallowed), and the returned count reflects only
        actual removals.
        """
        dead = find_dead_pid()
        doomed = plant(tmp_path, f'{PROBE_PREFIX}doomed-{dead}')
        survivor_candidate = plant(tmp_path, f'{PROBE_PREFIX}other-{dead}')
        real_rmtree = shutil.rmtree

        def flaky_rmtree(path, *args, **kwargs):
            if Path(path).name == doomed.name:
                raise OSError('simulated EACCES')
            return real_rmtree(path, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger='shared.config_dir'), \
                patch('shared.config_dir.shutil.rmtree', side_effect=flaky_rmtree):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert doomed.exists()
        assert not survivor_candidate.exists()
        assert removed == 1
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses directory write permissions')
    def test_genuinely_unremovable_dir_is_loud_and_uncounted(self, tmp_path, caplog):
        """The same containment, without mocking rmtree.

        The stubbed test above can only prove the handler works if the real
        call can reach it — with ``ignore_errors=True`` it never could, and
        `removed` would have counted a drain that never happened. A read-only
        parent makes the child un-unlinkable, so this exercises the real
        failure path and pins the count to genuine removals only.
        """
        dead = find_dead_pid()
        stuck = plant(tmp_path, f'{PROBE_PREFIX}stuck-{dead}')
        removable = plant(tmp_path, f'{PROBE_PREFIX}ok-{dead}')
        os.chmod(stuck, 0o500)  # r-x: .credentials.json can no longer be unlinked
        try:
            with caplog.at_level(logging.WARNING, logger='shared.config_dir'):
                removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)
        finally:
            os.chmod(stuck, 0o700)  # so tmp_path teardown can clean up

        assert stuck.exists()
        assert not removable.exists()
        assert removed == 1, 'an unremovable dir must never be counted as reclaimed'
        assert any(
            'failed to reclaim' in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_dir_reclaimed_by_a_concurrent_sweeper_is_quietly_skipped(self, tmp_path, caplog):
        """Peers race this sweep — the fleet restarts its units together.

        A dir another sweeper already removed is neither ours to count nor a
        failure worth a WARNING, so it must not add noise to the fleet log.
        """
        dead = find_dead_pid()
        raced = plant(tmp_path, f'{PROBE_PREFIX}raced-{dead}')
        ours = plant(tmp_path, f'{PROBE_PREFIX}ours-{dead}')
        real_rmtree = shutil.rmtree

        def racing_rmtree(path, *args, **kwargs):
            if Path(path).name == raced.name:
                real_rmtree(path)  # the peer sweeper wins
                raise FileNotFoundError(path)
            return real_rmtree(path, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger='shared.config_dir'), \
                patch('shared.config_dir.shutil.rmtree', side_effect=racing_rmtree):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path)

        assert not raced.exists()
        assert not ours.exists()
        assert removed == 1
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_deadline_stops_early_without_removing_everything(self, tmp_path, caplog):
        """A wall-clock bound must never read as 'swept everything'.

        UsageGate.__init__ is synchronous and runs on the event-loop thread
        during harness startup; the pathological /tmp this task addresses has
        a 40 MB directory inode. Blocking time is the risk, so the bound is a
        deadline, not a removal cap — the population still drains fully across
        successive process starts.

        Degenerate case: `deadline_secs=0.0` trips on the very first
        iteration, so nothing at all is removed. The partial-progress case
        that actually matters is the test below.
        """
        dead = find_dead_pid()
        planted = [plant(tmp_path, f'{PROBE_PREFIX}n{i}-{dead}') for i in range(4)]

        with caplog.at_level(logging.WARNING, logger='shared.config_dir'):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path, deadline_secs=0.0)

        assert removed == 0
        assert all(p.exists() for p in planted)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'a bounded stop must be loud, not silent'
        message = ' '.join(r.getMessage().lower() for r in warnings)
        assert 'deadline' in message
        assert 'incomplete' in message
        assert 'examined' in message
        assert 'removed' in message

    def test_deadline_tripped_mid_sweep_makes_partial_progress_and_resumes(
        self, tmp_path, caplog,
    ):
        """The behaviour that actually matters: stop mid-sweep, resume later.

        The degenerate `deadline_secs=0.0` case above reports "examined 0,
        removed 0", which any regression that reset `removed` on the bounded
        path or mis-ordered the `examined` increment would still satisfy. Here
        a stepping clock trips the deadline part-way through, so the counts
        are non-zero and the remainder must survive for the NEXT process to
        reclaim — that resumption is what makes a deadline (rather than a
        removal cap) safe.
        """
        dead = find_dead_pid()
        planted = [plant(tmp_path, f'{PROBE_PREFIX}n{i}-{dead}') for i in range(4)]
        # Call 1 captures the deadline (0.0 + 1.0); calls 2-3 are the first two
        # per-iteration checks; everything after is past it.
        ticks = iter([0.0, 0.0, 0.0])

        def stepping_monotonic() -> float:
            return next(ticks, 99.0)

        with caplog.at_level(logging.WARNING, logger='shared.config_dir'), \
                patch('shared.config_dir.time.monotonic', side_effect=stepping_monotonic):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path, deadline_secs=1.0)

        assert 0 < removed < len(planted)
        survivors = [p for p in planted if p.exists()]
        assert len(survivors) == len(planted) - removed

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        message = ' '.join(r.getMessage().lower() for r in warnings)
        assert 'incomplete' in message
        assert f'removed {removed}' in message
        assert 'examined 0' not in message, 'a partial sweep must report real counts'

        # The next process start reclaims the remainder — no permanent residue.
        caplog.clear()
        assert sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path) == len(survivors)
        assert not any(p.exists() for p in planted)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_generous_deadline_sweeps_everything_and_stays_quiet(self, tmp_path, caplog):
        """The bounded-stop WARNING must not fire on the normal path."""
        dead = find_dead_pid()
        planted = [plant(tmp_path, f'{PROBE_PREFIX}n{i}-{dead}') for i in range(4)]

        with caplog.at_level(logging.WARNING, logger='shared.config_dir'):
            removed = sweep_stale_pid_dirs(PROBE_PREFIX, base_dir=tmp_path, deadline_secs=30.0)

        assert removed == len(planted)
        assert not any(p.exists() for p in planted)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# Opt-in atexit teardown — the CLEAN-EXIT half. The PID-liveness sweep above
# is what actually bounds the population under SIGKILL.
# ---------------------------------------------------------------------------


def invoke_registered_hook(mock_register) -> None:
    """Call whatever was handed to ``atexit.register``, as atexit would."""
    (func, *bound), kwargs = mock_register.call_args
    func(*bound, **kwargs)


class TestTaskConfigDirCleanupAtExit:
    """`cleanup_at_exit=True` registers a best-effort teardown (step 5)."""

    @pytest.fixture(autouse=True)
    def _isolate_registration_ledger(self, monkeypatch):
        """Start every case with an empty dedup ledger.

        The ledger is process-global by design, so without this a case would
        inherit whatever earlier cases (or an imported sibling suite)
        registered.
        """
        monkeypatch.setattr(config_dir_module, '_atexit_registered_dirs', set())

    def test_opt_in_registers_one_hook_that_removes_the_dir(self, tmp_path):
        with patch('shared.config_dir.atexit.register') as register:
            cfg = TaskConfigDir('x', base_dir=tmp_path, cleanup_at_exit=True)

        assert register.call_count == 1
        assert cfg.path.exists()

        invoke_registered_hook(register)

        assert not cfg.path.exists()

    def test_default_registers_nothing(self, tmp_path):
        """Teardown MUST stay opt-in.

        Per-task and per-investigation config dirs live under a worktree's
        `.task/` and are deliberately preserved — the session JSONL inside
        them backs transcript archival and `--resume`, and
        orchestrator/tests/test_dry_run_unblock.py asserts the
        per-investigation dir survives. A future change that makes teardown
        unconditional would silently eat those; it fails here instead.
        """
        with patch('shared.config_dir.atexit.register') as register:
            cfg = TaskConfigDir('x', base_dir=tmp_path)

        register.assert_not_called()
        assert cfg.path.exists()

    def test_hook_binds_only_the_path_not_the_instance(self, tmp_path):
        """The atexit table must not pin the TaskConfigDir alive.

        Binding `self` (or a bound method) would keep the owning UsageGate —
        and its account OAuth tokens — reachable for the life of the process.
        """
        with patch('shared.config_dir.atexit.register') as register:
            cfg = TaskConfigDir('x', base_dir=tmp_path, cleanup_at_exit=True)

        (func, *bound), kwargs = register.call_args
        assert func is shutil.rmtree
        assert cfg.path in bound
        assert not any(arg is cfg for arg in bound)
        assert not any(arg is cfg for arg in kwargs.values())

    def test_hook_is_idempotent_after_explicit_cleanup(self, tmp_path):
        """UsageGate.shutdown() already calls cleanup() on the clean path.

        The later atexit hook must be a harmless no-op after it, and must not
        raise out of interpreter shutdown.
        """
        with patch('shared.config_dir.atexit.register') as register:
            cfg = TaskConfigDir('x', base_dir=tmp_path, cleanup_at_exit=True)

        cfg.cleanup()
        assert not cfg.path.exists()

        invoke_registered_hook(register)
        invoke_registered_hook(register)

        assert not cfg.path.exists()

    def test_registration_is_deduped_by_path(self, tmp_path):
        """One hook per path, however many TaskConfigDirs claim that path.

        A probe dir is named `usage-gate-probe-<account>-<pid>`, so every gate
        built in one process resolves to the SAME paths — and
        orchestrator/evals/runner.py builds a fresh gate per eval run. Without
        dedup the atexit table grows one Path-pinning entry per (gate x
        account) and re-runs rmtree on the same path once per entry at
        shutdown.
        """
        with patch('shared.config_dir.atexit.register') as register:
            first = TaskConfigDir('dup', base_dir=tmp_path, cleanup_at_exit=True)
            second = TaskConfigDir('dup', base_dir=tmp_path, cleanup_at_exit=True)
            third = TaskConfigDir('dup', base_dir=tmp_path, cleanup_at_exit=True)

        assert first.path == second.path == third.path
        assert register.call_count == 1

        # The single surviving hook still tears the path down.
        invoke_registered_hook(register)
        assert not third.path.exists()

    def test_dedup_is_per_path_not_global(self, tmp_path):
        """Distinct paths must each get their own hook — the per-account
        probe dirs of one gate differ only by account name."""
        with patch('shared.config_dir.atexit.register') as register:
            TaskConfigDir('probe-work', base_dir=tmp_path, cleanup_at_exit=True)
            TaskConfigDir('probe-personal', base_dir=tmp_path, cleanup_at_exit=True)

        assert register.call_count == 2
