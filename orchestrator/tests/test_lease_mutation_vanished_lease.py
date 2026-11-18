"""A lease removed by a concurrent --force release or reap mid-mutation is ABSENT and quiet.

heartbeat_lease and release_lease read the lease to gate on ownership, and
heartbeat_lease then touches it. When a concurrent ``lease-release --force``
or reap removes the file while the gate is reading it, or (for a heartbeat)
before the touch, the lease is simply gone: the mutation is ABSENT, never
REFUSED, FORCED or FAULTED over an ``<unreadable lease body>`` or a failed
touch.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator import session_registry as sr

_START_TS = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC).isoformat()
_HOLDER_SLUG = 'watcher-df-100'

RemoveLeaseBefore = Callable[[object, str, Path], None]


@pytest.fixture
def remove_lease_before(monkeypatch: pytest.MonkeyPatch) -> RemoveLeaseBefore:
    """Arm ``owner.attr`` so its first call on the lease path finds the lease already removed."""

    def arm(owner: object, attr: str, lease_path: Path) -> None:
        real = getattr(owner, attr)

        def concurrent_release_then_delegate(target: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(target, str | os.PathLike) and Path(target) == lease_path and lease_path.exists():
                lease_path.unlink()
            return real(target, *args, **kwargs)

        monkeypatch.setattr(owner, attr, concurrent_release_then_delegate)

    return arm


def _claim_watcher_lease(root: Path) -> Path:
    holder = sr.LeaseHolder(session_slug=_HOLDER_SLUG, pid=os.getpid(), start_ts=_START_TS)
    sr.claim_lease('watcher-df', holder=holder, root=root)
    return sr.lease_path_for_name('watcher-df', root=root)


@pytest.mark.parametrize('force', [False, True], ids=['no-force', 'force'])
@pytest.mark.parametrize('mutate', [sr.release_lease, sr.heartbeat_lease], ids=['release', 'heartbeat'])
def test_a_lease_that_vanishes_mid_check_is_absent_and_quiet(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remove_lease_before: RemoveLeaseBefore,
    mutate: Callable[..., sr.LeaseMutation],
    force: bool,
) -> None:
    lease_path = _claim_watcher_lease(tmp_path)
    # The concurrent release landing between the gate's stat and its body read.
    remove_lease_before(Path, 'read_text', lease_path)

    with caplog.at_level(logging.WARNING):
        result = mutate('watcher-df', slug=_HOLDER_SLUG, force=force, root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert not lease_path.exists()


@pytest.mark.parametrize(
    ('requester', 'force'),
    [(_HOLDER_SLUG, False), ('watcher-df-200', True)],
    ids=['owner', 'forcing-stranger'],
)
def test_a_lease_that_vanishes_before_the_heartbeat_touch_is_absent_and_quiet(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remove_lease_before: RemoveLeaseBefore,
    requester: str,
    force: bool,
) -> None:
    lease_path = _claim_watcher_lease(tmp_path)
    remove_lease_before(os, 'utime', lease_path)

    with caplog.at_level(logging.WARNING):
        result = sr.heartbeat_lease('watcher-df', slug=requester, force=force, root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    assert not lease_path.exists()
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
