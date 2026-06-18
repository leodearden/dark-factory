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

        ``(default, False)`` — file was corrupt; a WARNING was emitted and
        the corrective action was taken.

    Raises
    ------
    json.JSONDecodeError / ValueError
        When *on_corrupt='fail_closed'*.
    ValueError
        When *on_corrupt* is an unrecognised value.
    OSError
        When any OS error other than ``FileNotFoundError`` occurs.
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
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        # Corrupt branch: dispatch on on_corrupt.
        #
        # fail_closed: the exception IS the loud channel — re-raise BEFORE
        # any WARNING to avoid double-signalling.
        if on_corrupt == 'fail_closed':
            raise

        # Validate mode early so an unknown value raises rather than silently
        # behaving like 'warn'.
        if on_corrupt not in ('warn', 'quarantine'):
            raise ValueError(f'unknown on_corrupt={on_corrupt!r}')

        # warn / quarantine both emit the deduped WARNING.
        logger.warning('safe_io: corrupt JSON at %s: %s', p, exc)

        if on_corrupt == 'warn':
            return (default, False)

        # quarantine: not yet implemented; fall through raises ValueError above
        # for now (step-08 will add it).
        raise ValueError(f'unknown on_corrupt={on_corrupt!r}')  # unreachable after step-08

    return (parsed, True)
