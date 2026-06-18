"""Safe JSON I/O with explicit benign-absent vs corrupt-present handling.

Provides :func:`load_json_or_warn` which splits the common "read optional
JSON state file" pattern into two clearly-separated branches:

* **FileNotFoundError** — first-run / file absent; silently returns the
  caller-supplied ``default``.
* **JSONDecodeError / ValueError** — file present but corrupt; emits a
  deduped WARNING and dispatches on *on_corrupt*.

Other OSErrors (PermissionError, IsADirectoryError, …) propagate uncaught
so the caller sees genuinely unexpected environmental failures.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-process dedup set for corrupt-path warnings.
# Bounded by the number of distinct paths that go corrupt in a process lifetime.
# A restart re-enables the warning (mirrors sqlite_task_backend._warned_malformed_task_ids:51).
_warned_corrupt_paths: set[str] = set()

__all__ = ['load_json_or_warn']


def load_json_or_warn(
    path: str | os.PathLike[str],
    *,
    default: Any,
    on_corrupt: str = 'warn',
) -> tuple[Any, bool]:
    """Load *path* as JSON and return ``(parsed, True)`` on success.

    Parameters
    ----------
    path:
        File to read.  Accepts any ``os.PathLike`` or ``str``.
    default:
        Required keyword-only.  Returned as the value component whenever the
        file is absent or corrupt — forces callers to state their benign
        fallback explicitly.
    on_corrupt:
        One of ``'warn'`` (default), ``'fail_closed'``, or ``'quarantine'``.
        Controls what happens when a present-but-corrupt file is found;
        see return values below.

    Returns
    -------
    tuple[Any, bool]
        ``(parsed, True)`` — valid JSON was read.

        ``(default, True)`` — file did not exist (benign, no log emitted).
    """
    p = Path(path)

    # --- read phase ---
    try:
        text = p.read_text(encoding='utf-8')
    except FileNotFoundError:
        # Benign first-run absence — caller's default, no log.
        return (default, True)
    # Other OSErrors propagate uncaught (PermissionError, IsADirectoryError, …).

    # --- parse phase ---
    parsed = json.loads(text)
    return (parsed, True)
