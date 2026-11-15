"""A lease that vanishes after the ownership gate's existence check is ABSENT and quiet.

A concurrent ``lease-release --force`` or reap can remove the lease file
between heartbeat_lease/release_lease checking it exists and reading its
body. The lease is then simply gone: the mutation is ABSENT, never REFUSED,
FORCED or FAULTED over an ``<unreadable lease body>``.
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


@pytest.fixture
def vanish_on_first_read(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], None]:
    real_read_text = Path.read_text

    def arm(lease_path: Path) -> None:
        def read_text_after_concurrent_release(self: Path, *args: Any, **kwargs: Any) -> str:
            if self == lease_path and self.exists():
                # The concurrent release landing between the existence check and the body read.
                self.unlink()
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', read_text_after_concurrent_release)

    return arm


@pytest.mark.parametrize('force', [False, True], ids=['no-force', 'force'])
@pytest.mark.parametrize('mutate', [sr.release_lease, sr.heartbeat_lease], ids=['release', 'heartbeat'])
def test_a_lease_that_vanishes_mid_check_is_absent_and_quiet(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    vanish_on_first_read: Callable[[Path], None],
    mutate: Callable[..., sr.LeaseMutation],
    force: bool,
) -> None:
    holder = sr.LeaseHolder(session_slug=_HOLDER_SLUG, pid=os.getpid(), start_ts=_START_TS)
    sr.claim_lease('watcher-df', holder=holder, root=tmp_path)
    lease_path = sr.lease_path_for_name('watcher-df', root=tmp_path)
    vanish_on_first_read(lease_path)

    with caplog.at_level(logging.WARNING):
        result = mutate('watcher-df', slug=_HOLDER_SLUG, force=force, root=tmp_path)

    assert result is sr.LeaseMutation.ABSENT
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert not lease_path.exists()
