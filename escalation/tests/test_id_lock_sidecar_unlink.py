"""``escalation_id_lock`` stays exclusive when its sidecar is unlinked.

The sidecar is a lock FILE, so excluding two acquirers means both hold the
inode the path names.  A waiter that opened the sidecar and then blocked in
flock holds an fd on THAT inode; if the holder unlinks or replaces the path
before releasing, the waiter must not run on the detached inode, because a
newcomer's ``O_CREAT`` makes a fresh one and the two would run side by side.

Synchronisation is by ``/proc/self/fd``: two open fds naming the sidecar prove
the waiter has opened the OLD inode before the main thread touches the path.
That is a fact rather than a duration, so no verdict here rests on timing,
and nothing patches the queue module — the tests drive the one public seam.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from escalation.queue import escalation_id_lock

pytestmark = [
    pytest.mark.skipif(
        not Path('/proc/self/fd').is_dir(),
        reason='waiter synchronisation reads /proc/self/fd (Linux only)',
    ),
    pytest.mark.timeout(30),
]

ESC_ID = 'esc-1-1'
WAIT_SECS = 10.0


def _lock_path(queue_dir: Path) -> Path:
    return queue_dir / f'{ESC_ID}.json.lock'


def _open_fds_naming(path: Path) -> int:
    target = os.path.realpath(path)
    fd_dir = Path('/proc/self/fd')
    count = 0
    for entry in os.listdir(fd_dir):
        try:
            if os.readlink(fd_dir / entry) == target:
                count += 1
        except OSError:
            continue
    return count


def _newcomer_is_blocked(lock_path: Path) -> bool:
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _run_waiter_across(
    queue_dir: Path, between: Callable[[Path], None]
) -> dict[str, Any]:
    """Queue a waiter on the held sidecar, run ``between``, then release.

    Returns what the waiter observed from inside its own critical section.
    """
    lock_path = _lock_path(queue_dir)
    seen: dict[str, Any] = {}

    def waiter() -> None:
        try:
            with escalation_id_lock(queue_dir, ESC_ID):
                seen['present'] = lock_path.exists()
                seen['newcomer_blocked'] = _newcomer_is_blocked(lock_path)
        except BaseException as exc:  # noqa: BLE001 — surfaced by the caller
            seen['error'] = exc

    with escalation_id_lock(queue_dir, ESC_ID):
        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        deadline = time.monotonic() + WAIT_SECS
        while (opened := _open_fds_naming(lock_path)) < 2:
            if time.monotonic() > deadline:
                pytest.fail(
                    f'waiter never opened the sidecar: {opened} fd(s) name it'
                )
            time.sleep(0.005)
        between(lock_path)

    thread.join(WAIT_SECS)
    assert not thread.is_alive(), 'waiter did not finish after the holder released'
    return seen


def _unlink_then_recreate(lock_path: Path) -> None:
    os.unlink(lock_path)
    lock_path.touch()


def test_waiter_queued_on_an_unlinked_sidecar_still_excludes_a_newcomer(
    tmp_path: Path,
) -> None:
    seen = _run_waiter_across(tmp_path, os.unlink)

    assert 'error' not in seen, seen.get('error')
    assert seen['present'] is True, (
        'an acquirer queued on an unlinked sidecar must re-acquire the inode '
        'the path now names — the waiter ran with no sidecar at the path'
    )
    assert seen['newcomer_blocked'] is True, (
        'an acquirer queued on an unlinked sidecar must re-acquire the inode '
        'the path now names — a newcomer locked a fresh sidecar beside it'
    )


def test_waiter_released_onto_a_replaced_sidecar_locks_the_replacement(
    tmp_path: Path,
) -> None:
    seen = _run_waiter_across(tmp_path, _unlink_then_recreate)

    assert 'error' not in seen, seen.get('error')
    assert seen['newcomer_blocked'] is True, (
        'an acquirer released onto a replaced sidecar must lock the inode the '
        'path now names — a newcomer locked the replacement beside it'
    )


def test_sidecar_unlinked_by_its_holder_is_recreated_and_locked_by_the_next_acquirer(
    tmp_path: Path,
) -> None:
    lock_path = _lock_path(tmp_path)
    with escalation_id_lock(tmp_path, ESC_ID):
        os.unlink(lock_path)

    with escalation_id_lock(tmp_path, ESC_ID):
        assert lock_path.exists()
        assert _newcomer_is_blocked(lock_path)
