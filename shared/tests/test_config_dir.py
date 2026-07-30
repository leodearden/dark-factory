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

import os
import time
from pathlib import Path

from shared.config_dir import CONFIG_DIR_PREFIX, sweep_stale_pid_dirs

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
