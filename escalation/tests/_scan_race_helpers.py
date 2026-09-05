"""Shared test helpers for the unlocked-scan race that task 5111 hardens.

Lives here rather than in either suite because ``test_queue.py`` and
``test_sweep.py`` exercise the SAME helper
(``escalation.queue.read_escalation_for_scan``) on the SAME 'vanished'
channel, from opposite sides.  Two byte-identical copies of the interposition
— which is what existed before — drift the moment one side needs it to handle
a keyword-only argument or a ``Path`` subclass, and only one suite notices.

Not a ``conftest.py`` fixture: the interposition takes arguments (which file
is doomed, and where it goes), so a plain factory is the honest shape;
``escalation/tests`` is on ``sys.path`` under pytest's default prepend import
mode, so a flat import works from both suites.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def relocating_read_text(doomed: Path, dest_dir: Path) -> Callable[..., str]:
    """Interpose ``Path.read_text`` to really relocate *doomed* mid-scan.

    Not a mocked exception: the file is genuinely ``os.replace``d into
    *dest_dir* and the original ``read_text`` is then called through, so the
    ``FileNotFoundError`` comes from the filesystem exactly as it does when a
    concurrent ``resolve()`` — or a second startup sweep during a fleet
    redeploy — moves a record between the glob and the read.  Neither actor
    holds a lock over that window: ``sweep._relocate_terminal`` takes the
    per-id lock only around the move itself.

    The ``doomed.exists()`` guard makes the interposition fire exactly once,
    so a caller that re-reads the same path afterwards is not relocated twice.

    Usage::

        with patch.object(Path, 'read_text', relocating_read_text(doomed, dest)):
            ...
    """
    original = Path.read_text

    def flaky(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == doomed and doomed.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(doomed), str(dest_dir / doomed.name))
        return original(self, *args, **kwargs)

    return flaky


def unreadable_read_text(doomed: Path) -> Callable[..., str]:
    """Interpose ``Path.read_text`` to fail *doomed* with ``PermissionError``.

    The counterpart to :func:`relocating_read_text`, and the reason both live
    here: the two exercise the channels the fix exists to keep DISTINGUISHABLE
    — a file that vanished (benign, DEBUG) versus one that is still on disk and
    genuinely unreadable (operator-actionable, WARNING).  Unlike the relocating
    factory this one leaves the file exactly where it was, which is the whole
    distinction; ``EACCES`` stands in for the family (a bad umask, EIO, an
    exhausted fd table).
    """
    original = Path.read_text

    def flaky(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == doomed:
            # Left on disk, unlike the relocation case.
            raise PermissionError(13, 'Permission denied', str(doomed))
        return original(self, *args, **kwargs)

    return flaky
