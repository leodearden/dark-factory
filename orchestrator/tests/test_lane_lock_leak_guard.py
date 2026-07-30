"""Lane-lock leak guard — D8 / B12 + B13 of the warm-lane repatriation PRD.

PRD: ``plans/warm-lane-infra-repatriation-prd.md`` §D8.
Incident anchor: reify ``esc-5548-5`` — a merge-verify lane lock left held by
an orphaned ``asyncio.to_thread`` acquire.  Kernel forensics at the time read
``/proc/locks`` and inode-matched ``FLOCK ADVISORY WRITE 588232 07:1d:4300647613``
against ``_merge-verify.lock``; three tasks then blocked behind one identical
``merge_outcome_signature`` and nothing surfaced until an unattended restart
roughly three hours later.

Two defects live here:

* **B12** — a cancelled lane-lock acquire could ORPHAN the fd.
  :func:`asyncio.to_thread` cannot interrupt its worker thread: once the outer
  ``await`` is cancelled, ``asyncio.futures._copy_future_state`` bails on
  ``dest.cancelled()`` and the thread's return value — the freshly acquired fd
  — is discarded, so ``release_merge_verify_flock`` never runs for it and the
  lane lock stays held until process exit.
* **B13** — nothing DETECTED that state.  A lane lock held by an fd in THIS
  process, with no live in-process span and no live verify, is a self-owned
  leak, not foreign contention, and must be reported as such.

This module owns the new coverage for both.  It reuses the real-git fixtures of
:mod:`test_merge_verify_lease_guard` wholesale (the suite's cross-module-helper
convention) and adds two shared helpers used by every class below:
:func:`foreign_lane_lock_holder` (a GENUINELY foreign holder — same-process fds
are indistinguishable from the leak at kernel level) and :func:`wait_until` (a
bounded async poll, so orphan-release callbacks settle deterministically
instead of behind a fixed sleep).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    release_merge_verify_flock,
)

# Real-git fixtures + helpers reused from the sibling lease-guard module (the
# suite's established cross-module-helper convention, cf.
# test_verify_scope_kappa.py / test_coalesce_integration_gate.py).  `git_repo`
# and `real_git_ops` are pytest FIXTURES: importing them binds them as module
# attributes here, which is what makes them requestable by the tests below.
from test_merge_verify_lease_guard import (  # noqa: F401 -- fixtures used by name
    _get_merge_commit,
    _git_config,
    _git_ops,
    _setup_repo,
    git_repo,
    real_git_ops,
)

__all__ = [
    '_get_merge_commit',
    '_git_config',
    '_git_ops',
    '_setup_repo',
    'foreign_lane_lock_holder',
    'git_repo',
    'lane_lock_path',
    'real_git_ops',
    'wait_until',
]


#: How long the ``flock(1)`` child holds the lane lock before exiting on its
#: own.  Long enough that no test races it; the contextmanager terminates the
#: child on exit regardless, so this is only a leak backstop.
_FOREIGN_HOLDER_HOLD_SECS = 120

#: Bound on how long :func:`foreign_lane_lock_holder` waits for the child to
#: actually own the lock before failing the test.
_FOREIGN_HOLDER_STARTUP_SECS = 5.0


@contextlib.contextmanager
def foreign_lane_lock_holder(lock_path: Path) -> Iterator[subprocess.Popen]:
    """Hold *lock_path* from a GENUINELY foreign process, yielding the child.

    Spawns ``flock -x <lock_path> sleep N`` — util-linux ``flock(1)``, already
    exercised by this suite via :meth:`GitOps._seed_warm_lane`, and
    interoperable with ``fcntl.flock(2)`` on the same inode.

    Why a subprocess and not a second fd in this process: at KERNEL level a
    same-process fd is precisely the B13 self-owned case (``/proc/locks``
    reports our own pid as the holder), so it cannot stand in for foreign
    contention once the leak detector exists.  Any test that means "somebody
    else holds this lane" must use this helper.

    Blocks until the child provably owns the lock (a zero-timeout probe
    acquire returns ``None``), failing the test if that has not happened
    within :data:`_FOREIGN_HOLDER_STARTUP_SECS`.  Terminates and reaps the
    child on exit.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        ['flock', '-x', str(lock_path), 'sleep', str(_FOREIGN_HOLDER_HOLD_SECS)],
    )
    try:
        deadline = time.monotonic() + _FOREIGN_HOLDER_STARTUP_SECS
        while True:
            probe = acquire_merge_verify_flock(lock_path, 0.0)
            if probe is None:
                break  # the child owns it — contention is observable
            release_merge_verify_flock(probe)
            if time.monotonic() >= deadline:
                child.terminate()
                child.wait(timeout=5)
                pytest.fail(
                    f'flock(1) child never took {lock_path} within '
                    f'{_FOREIGN_HOLDER_STARTUP_SECS}s — the foreign-holder '
                    f'staging is broken, so any assertion built on it would '
                    f'be vacuous'
                )
            time.sleep(0.02)
        yield child
    finally:
        child.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=5)


async def wait_until(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Poll *predicate* until true, returning whether it became true in time.

    Returns ``True`` as soon as *predicate* holds, ``False`` once *timeout*
    elapses — deliberately NOT raising, so a caller can assert on the result
    with its own message.  Used to settle orphan-release callbacks (which run
    on the event loop, after the cancelling coroutine has already resumed)
    without a fixed sleep that would either flake or waste wall clock.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


def test_scaffold_helpers_are_importable(tmp_path: Path):
    """Smoke pin: the shared helpers exist and the foreign holder really holds.

    Guards the scaffold itself — every class in this module builds on
    ``foreign_lane_lock_holder`` observably owning the lock, so a broken helper
    would silently make downstream assertions vacuous rather than red.
    """
    lock_path = lane_lock_path(tmp_path / 'lane')
    with foreign_lane_lock_holder(lock_path):
        assert acquire_merge_verify_flock(lock_path, 0.0) is None
    assert asyncio.run(wait_until(lambda: True, timeout=0.1)) is True


@pytest.mark.asyncio
async def test_scaffold_real_git_fixtures_available(real_git_ops):
    """Smoke pin: the imported real-git fixtures resolve in THIS module.

    Cross-module fixture imports are load-bearing here (they are what makes
    ``real_git_ops`` requestable); pinning them once means a future refactor of
    the sibling module fails loudly here instead of erroring one test at a time.
    """
    commit = await _get_merge_commit(real_git_ops, 'scaffold-lane', 'scaffold.py')
    assert commit
